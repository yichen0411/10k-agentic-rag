#!/usr/bin/env python3
"""Stage 1: offline VLM parse of table crop images into markdown + summary."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

CHUNKING_DIR = Path(__file__).resolve().parent
ROOT = CHUNKING_DIR.parents[1]
DEFAULT_VLM_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_FIREWORKS_VLM_MODEL = "accounts/fireworks/models/kimi-k2p5"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_VLM_DPI = 132
DEFAULT_VLM_MULTI_PAGE_DPI = 96
MAX_VLM_PIXELS = 850_000


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def load_env_file(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_json_response(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(candidate[start : end + 1])
        raise


def section_ref_label(table: dict[str, Any]) -> str:
    """Human-readable section path for prompts/embeddings (no ref ids)."""
    section = table.get("section_ref") or {}
    subsection = table.get("subsection_ref") or {}
    parts = [section.get("item"), section.get("section_title"), *(subsection.get("path") or [])]
    return " > ".join(part for part in parts if part)


def table_summary_topic(vlm: dict[str, Any]) -> str:
    """Semantic caption only — what the table shows, without section path."""
    if vlm.get("summary_topic"):
        return str(vlm["summary_topic"]).strip()
    summary = str(vlm.get("summary") or "").strip()
    if " · " in summary and " > " in summary.split(" · ", 1)[0]:
        return summary.split(" · ", 1)[-1].strip()
    return summary


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def line_records(page: fitz.Page) -> list[dict[str, Any]]:
    words = page.get_text("words")
    buckets: dict[int, list[tuple]] = {}
    for word in words:
        y_key = int(round(float(word[1]) / 3.0))
        buckets.setdefault(y_key, []).append(word)
    lines: list[dict[str, Any]] = []
    for y_key in sorted(buckets):
        row = sorted(buckets[y_key], key=lambda w: float(w[0]))
        text = " ".join(str(w[4]) for w in row).strip()
        if not text:
            continue
        y0 = min(float(w[1]) for w in row)
        y1 = max(float(w[3]) for w in row)
        lines.append({"y0": y0, "y1": y1, "text": text})
    return lines


def table_context_sentences(doc: fitz.Document, table: dict[str, Any], before: int = 2, after: int = 2) -> tuple[list[str], list[str]]:
    page_no = int(table.get("page_start") or 1)
    page = doc[page_no - 1]
    bbox = table.get("bbox") or [0, 0, page.rect.width, page.rect.height]
    y0 = float(bbox[1])
    y1 = float(bbox[3])
    lines = line_records(page)
    above = [line["text"] for line in lines if line["y1"] <= y0 + 4]
    below = [line["text"] for line in lines if line["y0"] >= y1 - 4]
    before_text = " ".join(above[-before * 2 :]) if above else page.get_text("text")[:800]
    after_text = " ".join(below[: after * 2]) if below else ""
    return split_sentences(before_text)[-before:], split_sentences(after_text)[:after]


def _combine_pixmaps_vertical(parts: list[fitz.Pixmap]) -> fitz.Pixmap:
    if len(parts) == 1:
        return parts[0]
    images = [Image.open(io.BytesIO(pix.tobytes("png"))) for pix in parts]
    width = max(img.width for img in images)
    height = sum(img.height for img in images)
    combined = Image.new("RGB", (width, height), "white")
    y = 0
    for img in images:
        combined.paste(img, (0, y))
        y += img.height
    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    return fitz.Pixmap(buf.getvalue())


def _vlm_clip_for_page_crop(page: fitz.Page, table: dict[str, Any], bbox: list[float], crop_idx: int) -> fitz.Rect:
    from section_asset_extractor import is_header_only_table_crop, table_crop_clip, table_crop_padding

    if is_header_only_table_crop(table, crop_idx, bbox, page.rect.height):
        if crop_idx == 0:
            return fitz.Rect(
                max(page.rect.x0, bbox[0] - 4),
                max(page.rect.y0, bbox[1] - 4),
                min(page.rect.x1, bbox[2] + 4),
                min(page.rect.y1, bbox[3] + 8),
            ) & page.rect
        return fitz.Rect(0, max(page.rect.y0, bbox[1] - 8), page.rect.x1, min(page.rect.y1, bbox[3] + 6)) & page.rect
    top_pad, bottom_pad = table_crop_padding(table, crop_idx, bbox, page.rect.height)
    return table_crop_clip(page, bbox, top_pad, bottom_pad)


def _effective_dpi(clip: fitz.Rect, dpi: int, *, max_pixels: int = MAX_VLM_PIXELS) -> int:
    width_pt = max(float(clip.width), 1.0)
    height_pt = max(float(clip.height), 1.0)
    for candidate in range(int(dpi), 71, -6):
        width_px = int(width_pt / 72.0 * candidate)
        height_px = int(height_pt / 72.0 * candidate)
        if width_px * height_px <= max_pixels:
            return candidate
    return 72


def render_table_crop(
    doc: fitz.Document,
    table: dict[str, Any],
    out_path: Path,
    dpi: int = DEFAULT_VLM_DPI,
    *,
    multi_page_dpi: int = DEFAULT_VLM_MULTI_PAGE_DPI,
) -> Path:
    """Render one PNG for VLM. Cross-page table_groups are stitched vertically."""
    crops = table.get("bbox_by_page") or []
    base_dpi = multi_page_dpi if len(crops) > 1 else dpi
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(crops) > 1:
        pixmaps: list[fitz.Pixmap] = []
        for idx, crop in enumerate(crops):
            page_no = int(crop.get("page") or table.get("page_start") or 1)
            bbox = crop.get("bbox")
            if not bbox:
                continue
            page = doc[page_no - 1]
            clip = _vlm_clip_for_page_crop(page, table, bbox, idx)
            render_dpi = _effective_dpi(clip, base_dpi)
            pixmaps.append(page.get_pixmap(clip=clip, dpi=render_dpi, alpha=False))
        if not pixmaps:
            crops = []
        else:
            combined = _combine_pixmaps_vertical(pixmaps)
            combined.save(out_path)
            return out_path

    page_no = int(table.get("page_start") or 1)
    page = doc[page_no - 1]
    bbox = table.get("bbox") or [0, 0, page.rect.x1, page.rect.y1]
    clip = _vlm_clip_for_page_crop(page, table, bbox, 0)
    render_dpi = _effective_dpi(clip, base_dpi)
    pix = page.get_pixmap(clip=clip, dpi=render_dpi, alpha=False)
    pix.save(out_path)
    return out_path


def denoise_context_sentence(sentence: str) -> str:
    """Remove numeric values from narrative context used for table summary prompting/embedding."""
    text = clean_text(sentence)
    if not text:
        return ""
    text = re.sub(r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand|m|b))?", "[amount]", text, flags=re.I)
    text = re.sub(r"\(?[\d,]+(?:\.\d+)?\)?%?", "[num]", text)
    text = re.sub(r"\b(?:20\d{2})\b", "[year]", text)
    return clean_text(text)


def denoise_summary_text(summary: str) -> str:
    """Strip dollar amounts and percentages from topic caption; keep fiscal years."""
    text = clean_text(summary)
    text = re.sub(r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand|m|b))?", " ", text, flags=re.I)
    text = re.sub(r"\(?[\d,]+(?:\.\d+)?\)?%", " ", text)
    return clean_text(re.sub(r"\s+", " ", text).strip(" ,;:."))


def compose_table_summary(table: dict[str, Any], topic: str, section_ref: str | None = None) -> str:
    """Stored summary + retrieval text: section path first, then semantic topic."""
    path = clean_text(section_ref or section_ref_label(table))
    topic = clean_text(denoise_summary_text(topic))
    if not path:
        return topic
    if not topic:
        return path
    return f"{path} · {topic}"


def table_caption_hint(table: dict[str, Any]) -> str:
    subsection = table.get("subsection_ref") or {}
    path = subsection.get("path") or []
    if path:
        return path[-1]
    section = table.get("section_ref") or {}
    return section.get("section_title") or section.get("item") or "financial table"


def build_vlm_prompt(table: dict[str, Any], before_sentences: list[str], after_sentences: list[str]) -> tuple[str, str]:
    section_ref = section_ref_label(table)
    caption_hint = table_caption_hint(table)
    before_clean = [denoise_context_sentence(sentence) for sentence in before_sentences if denoise_context_sentence(sentence)]
    after_clean = [denoise_context_sentence(sentence) for sentence in after_sentences if denoise_context_sentence(sentence)]
    context_block = "\n".join(
        [
            f"Section path (where this table appears in the filing): {section_ref}",
            f"Nearby subsection heading: {caption_hint}",
            "Context before table (numbers removed):",
            *[f"- {sentence}" for sentence in before_clean[:2]],
            "Context after table (numbers removed):",
            *[f"- {sentence}" for sentence in after_clean[:1]],
        ]
    )
    prompt = (
        "You are parsing a financial filing table image.\n"
        "Return compact JSON only with keys: markdown, summary, skip.\n"
        "If the image does NOT contain a real tabular grid (aligned rows/columns of structured data), "
        'set skip=true and leave markdown and summary empty. '
        "Plain paragraphs, section headings, or narrative text are NOT tables — skip those.\n"
        "Rules for markdown:\n"
        "- Output a complete GitHub-flavored markdown table.\n"
        "- Preserve hierarchical/multi-row headers, blank cells, and parentheses negatives exactly as shown.\n"
        "- Do not invent values not visible in the image.\n"
        "Rules for summary:\n"
        "- Write ONE concise caption (about 12-20 words) explaining what financial information the table shows.\n"
        "- Use the section path and nearby context to infer scope (company-wide consolidated vs segment vs expense category).\n"
        "- Name the statement/schedule type, main metrics or rows theme, and fiscal years when visible (e.g. FY2025 vs FY2024).\n"
        "- Do NOT repeat the full section path in the summary (path is provided separately).\n"
        "- Do NOT include table_id, section_ref_id, or other internal ids.\n"
        "- Do NOT start with 'The table presents/shows/displays'.\n"
        "- Do NOT list every row label; capture semantic meaning, not a row inventory.\n"
        "- Dollar amounts and percentages are optional; prefer labels, scope, and years.\n"
        "Good examples:\n"
        '- "Consolidated summary results of operations with revenue, margins, and EPS for FY2025 vs FY2024"\n'
        '- "Research and development operating expenses and percent of revenue, FY2025 vs FY2024"\n'
        '- "Quarterly dividend declaration dates and cash dividend per share amounts"\n'
        "Bad examples:\n"
        '- "Revenue FY2025 vs FY2024" (too terse; missing what kind of table)\n'
        '- "The table presents key financial performance metrics including revenue gross margin..."\n'
        f"Table metadata: page={table.get('page_start')}.\n\n"
        f"{context_block}"
    )
    return section_ref, prompt


def extract_fiscal_year_pair(table: dict[str, Any], vlm: dict[str, Any] | None = None) -> tuple[str, str] | None:
    vlm = vlm or table.get("vlm_parse") or {}
    corpus = " ".join(
        [
            *(vlm.get("context_after_raw") or []),
            *(vlm.get("context_before_raw") or []),
            vlm.get("markdown") or "",
        ]
    )
    match = re.search(r"Fiscal Year (\d{4}) Compared with Fiscal Year (\d{4})", corpus, flags=re.I)
    if match:
        return match.group(1), match.group(2)
    years = sorted(
        {year for year in re.findall(r"\b(20\d{2})\b", corpus) if 2010 <= int(year) <= 2035},
        reverse=True,
    )
    if len(years) >= 2:
        return years[0], years[1]
    return None


def enrich_topic_with_years(topic: str, table: dict[str, Any], vlm: dict[str, Any] | None = None) -> str:
    text = clean_text(topic)
    if re.search(r"\b20\d{2}\b", text):
        return text
    years = extract_fiscal_year_pair(table, vlm)
    if not years:
        return text
    newer, older = years
    text = re.sub(r"\bFY\s*vs\.?\s*FY\b", f"FY{newer} vs FY{older}", text, flags=re.I)
    if re.search(r"\bFY20\d{2}\b", text):
        return text
    if re.search(r"\b(?:vs\.?|compared|change|comparison|expenses|operations|revenue|income)\b", text, flags=re.I):
        return clean_text(f"{text} FY{newer} vs FY{older}")
    return text


def table_row_labels(table: dict[str, Any], limit: int = 5) -> list[str]:
    labels: list[str] = []
    for row in table.get("raw_rows") or []:
        if not row or not row[0]:
            continue
        label = clean_text(str(row[0]))
        if not label or label.startswith("$") or re.match(r"^[\d(,]+$", label):
            continue
        if re.match(r"^\(?In (?:millions|thousands)", label, flags=re.I):
            continue
        labels.append(label)
        if len(labels) >= limit:
            break
    if len(labels) >= 2:
        return labels
    vlm = table.get("vlm_parse") or {}
    for line in (vlm.get("markdown") or "").splitlines():
        if not line.startswith("|"):
            continue
        cells = [clean_text(cell) for cell in line.strip("|").split("|")]
        if not cells or not cells[0]:
            continue
        label = cells[0]
        if re.match(r"^:?-+$", label) or re.match(r"^\(?In (?:millions|thousands)", label, flags=re.I):
            continue
        if label.lower() in {"2025", "2024", "percentage change"}:
            continue
        if label not in labels:
            labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def _row_theme_phrase(labels: list[str]) -> str:
    lower = [label.lower() for label in labels]
    if any("revenue" in label for label in lower) and any("margin" in label for label in lower):
        return "revenue, margins, and earnings"
    if any("expense" in label for label in lower):
        return " and ".join(labels[:2]).lower()
    if len(labels) >= 2:
        return ", ".join(labels[:3]).lower()
    return labels[0].lower() if labels else "key metrics"


def infer_summary_topic_from_table(table: dict[str, Any], vlm: dict[str, Any] | None = None) -> str:
    vlm = vlm or table.get("vlm_parse") or {}
    corpus = " ".join([*(vlm.get("context_before_raw") or []), *(vlm.get("context_after_raw") or [])])
    heading = re.search(
        r"\b(SUMMARY RESULTS OF OPERATIONS|RESULTS OF OPERATIONS|OPERATING EXPENSES|REPORTABLE SEGMENTS)\b",
        corpus,
        flags=re.I,
    )
    labels = table_row_labels(table)
    years = extract_fiscal_year_pair(table, vlm)
    year_suffix = f" FY{years[0]} vs FY{years[1]}" if years else ""

    if heading and heading.group(1).upper() == "SUMMARY RESULTS OF OPERATIONS":
        theme = _row_theme_phrase(labels)
        return clean_text(f"consolidated summary results of operations with {theme}{year_suffix}")
    if heading and heading.group(1).upper() == "OPERATING EXPENSES" and labels:
        if any("percent" in label.lower() or "revenue" in label.lower() for label in labels[1:]):
            return clean_text(f"{labels[0].lower()} operating expenses and percent of revenue{year_suffix}")
        return clean_text(f"{labels[0].lower()} operating expenses{year_suffix}")
    label_blob = " ".join(label.lower() for label in labels)
    if "revenue" in label_blob and re.search(
        r"(?:growth across each of our segments|summary results of operations)",
        corpus,
        flags=re.I,
    ):
        theme = _row_theme_phrase(labels)
        return clean_text(f"consolidated summary results of operations with {theme}{year_suffix}")
    if len(labels) >= 3 and "revenue" in label_blob and "gross margin" in label_blob:
        theme = _row_theme_phrase(labels)
        return clean_text(f"consolidated summary results of operations with {theme}{year_suffix}")
    if "revenue" in label_blob and (
        "operating income" in label_blob
        or "net income" in label_blob
        or "diluted earnings" in label_blob
    ):
        theme = _row_theme_phrase(labels)
        return clean_text(f"consolidated summary results of operations with {theme}{year_suffix}")
    if labels:
        if len(labels) == 1 and years:
            return clean_text(f"{labels[0]} FY{years[0]} vs FY{years[1]}")
        return clean_text(f"{labels[0]}{year_suffix}")
    return ""


def is_weak_summary_topic(text: str) -> bool:
    if not text:
        return True
    if re.search(r"\bfinancial table\b$", text, flags=re.I):
        return True
    if re.match(r"^revenue fy20\d{2} vs fy20\d{2}$", text, flags=re.I):
        return True
    words = text.split()
    if len(words) <= 4 and re.search(r"fy20\d{2}", text, flags=re.I):
        return True
    return False


def polish_summary_topic(
    summary: str,
    table: dict[str, Any],
    section_ref: str,
    vlm: dict[str, Any] | None = None,
) -> str:
    vlm = vlm or table.get("vlm_parse") or {}
    text = denoise_summary_text(summary)
    text = re.sub(r"^(?:msft|microsoft|aapl|apple|googl|alphabet|nvda|nvidia)\s+", "", text, flags=re.I)
    text = clean_text(text)

    boilerplate = (
        r"^(?:the\s+)?table\s+(?:presents|shows|displays|contains|summarizes|lists)\b",
        r"^key financial performance metrics\b",
        r"^financial performance metrics\b",
    )
    generic = bool(re.search(r"\bfinancial table\b$", text, flags=re.I))
    inferred = infer_summary_topic_from_table(table, vlm)

    if text and not generic and not any(re.search(pattern, text, flags=re.I) for pattern in boilerplate):
        if inferred and (is_weak_summary_topic(text) or len(inferred.split()) >= len(text.split()) + 3):
            return inferred
        return enrich_topic_with_years(text, table, vlm)

    if inferred:
        return inferred

    hint = table_caption_hint(table)
    return enrich_topic_with_years(clean_text(f"{hint} financial table")[:120], table, vlm)


def finalize_vlm_result(
    text: str,
    *,
    table: dict[str, Any],
    section_ref: str,
    before_sentences: list[str],
    after_sentences: list[str],
    latency_ms: int,
    model: str,
) -> dict[str, Any]:
    try:
        parsed = parse_json_response(text)
    except json.JSONDecodeError:
        return {"status": "parse_error", "raw_response": text, "latency_ms": latency_ms, "model": model}

    if parsed.get("skip") is True:
        return {
            "status": "skipped",
            "reason": "not_a_table",
            "table_id": table.get("table_id"),
            "section_ref": section_ref,
            "latency_ms": latency_ms,
            "model": model,
            "parsed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }

    markdown = (parsed.get("markdown") or parsed.get("markdown_table") or "").strip()
    vlm_context = {
        "context_before_raw": before_sentences,
        "context_after_raw": after_sentences,
        "markdown": markdown,
    }
    topic = polish_summary_topic(
        (parsed.get("summary") or parsed.get("table_title") or "").strip(),
        table,
        section_ref,
        vlm_context,
    )
    summary = compose_table_summary(table, topic, section_ref)
    if not markdown:
        return {"status": "parse_error", "raw_response": text, "latency_ms": latency_ms, "model": model}

    return {
        "status": "success",
        "table_id": table.get("table_id"),
        "section_ref": section_ref,
        "markdown": markdown,
        "summary": summary,
        "summary_topic": topic,
        "page": table.get("page_start"),
        "context_before": [denoise_context_sentence(s) for s in before_sentences],
        "context_after": [denoise_context_sentence(s) for s in after_sentences],
        "context_before_raw": before_sentences,
        "context_after_raw": after_sentences,
        "latency_ms": latency_ms,
        "model": model,
        "parsed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


def call_vlm_table_parse_anthropic(image_path: Path, prompt: str, model: str) -> str:
    import anthropic

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model,
        max_tokens=int(os.environ.get("ANTHROPIC_VLM_MAX_TOKENS", "6000")),
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")


def call_vlm_table_parse_fireworks(image_path: Path, prompt: str, model: str) -> str:
    from openai import OpenAI

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    client = OpenAI(api_key=os.environ["FIREWORKS_API_KEY"], base_url=FIREWORKS_BASE_URL)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=int(os.environ.get("FIREWORKS_VLM_MAX_TOKENS", "6000")),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a financial table parser. Reply with valid JSON only. Do not include reasoning or markdown fences.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return resp.choices[0].message.content or ""


def call_vlm_table_parse(
    image_path: Path,
    table: dict[str, Any],
    before_sentences: list[str],
    after_sentences: list[str],
) -> dict[str, Any]:
    load_env_file()
    section_ref, prompt = build_vlm_prompt(table, before_sentences, after_sentences)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    fireworks_key = os.environ.get("FIREWORKS_API_KEY")
    if not anthropic_key and not fireworks_key:
        return {"status": "skipped", "error": "ANTHROPIC_API_KEY or FIREWORKS_API_KEY required"}

    provider = os.environ.get("VLM_PROVIDER", "anthropic" if anthropic_key else "fireworks").lower()
    if provider == "anthropic" and not anthropic_key:
        provider = "fireworks"
    if provider == "fireworks" and not fireworks_key:
        provider = "anthropic"

    if provider == "anthropic":
        model = os.environ.get("ANTHROPIC_VLM_MODEL", os.environ.get("ANTHROPIC_SQL_MODEL", DEFAULT_VLM_MODEL))
        caller = call_vlm_table_parse_anthropic
    else:
        model = os.environ.get("FIREWORKS_VLM_MODEL", DEFAULT_FIREWORKS_VLM_MODEL)
        caller = call_vlm_table_parse_fireworks

    started = time.perf_counter()
    try:
        text = caller(image_path, prompt, model)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "model": model}
    latency_ms = int((time.perf_counter() - started) * 1000)
    return finalize_vlm_result(
        text,
        table=table,
        section_ref=section_ref,
        before_sentences=before_sentences,
        after_sentences=after_sentences,
        latency_ms=latency_ms,
        model=model,
    )


def filter_tables_for_parse(
    tables: list[dict[str, Any]],
    *,
    min_page: int | None = None,
    exclude_items: list[str] | None = None,
    max_tables: int | None = None,
    table_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    exclude = set(exclude_items or [])
    wanted = set(table_ids or [])
    filtered: list[dict[str, Any]] = []
    for table in tables:
        if wanted and table.get("table_id") not in wanted:
            continue
        page = int(table.get("page_start") or 0)
        item = (table.get("section_ref") or {}).get("item") or ""
        if min_page is not None and page < min_page:
            continue
        if item in exclude:
            continue
        filtered.append(table)
    if max_tables is not None:
        return filtered[:max_tables]
    return filtered


def extract_summary_topic(vlm: dict[str, Any]) -> str:
    return table_summary_topic(vlm)


def refresh_vlm_denoise_fields(assets_path: Path) -> dict[str, Any]:
    """Recompute denoised summary/context fields from existing vlm_parse without re-calling the VLM."""
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    refreshed = 0
    for table in assets.get("tables", []):
        vlm = table.get("vlm_parse") or {}
        if vlm.get("status") != "success":
            continue
        before_raw = vlm.get("context_before_raw") or vlm.get("context_before") or []
        after_raw = vlm.get("context_after_raw") or vlm.get("context_after") or []
        vlm["section_ref"] = vlm.get("section_ref") or section_ref_label(table)
        vlm["context_before_raw"] = before_raw
        vlm["context_after_raw"] = after_raw
        vlm["context_before"] = [denoise_context_sentence(s) for s in before_raw if denoise_context_sentence(s)]
        vlm["context_after"] = [denoise_context_sentence(s) for s in after_raw if denoise_context_sentence(s)]
        topic = polish_summary_topic(
            extract_summary_topic(vlm),
            table,
            vlm.get("section_ref") or section_ref_label(table),
            vlm,
        )
        vlm["summary_topic"] = topic
        vlm["summary"] = compose_table_summary(table, topic, vlm.get("section_ref") or section_ref_label(table))
        refreshed += 1
    assets_path.write_text(json.dumps(assets, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"assets_path": str(assets_path), "refreshed": refreshed}


def enrich_assets_with_vlm(
    pdf_path: Path,
    assets_path: Path,
    image_dir: Path | None = None,
    max_tables: int | None = None,
    min_page: int | None = None,
    exclude_items: list[str] | None = None,
    skip_existing: bool = True,
    table_ids: list[str] | None = None,
    dpi: int = DEFAULT_VLM_DPI,
    multi_page_dpi: int = DEFAULT_VLM_MULTI_PAGE_DPI,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    image_dir = image_dir or assets_path.parent / "table_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    tables = filter_tables_for_parse(
        list(assets.get("tables", [])),
        min_page=min_page,
        exclude_items=exclude_items,
        max_tables=max_tables,
        table_ids=table_ids,
    )
    tables_by_id = {table["table_id"]: table for table in assets.get("tables", [])}

    doc = fitz.open(pdf_path)
    parsed_count = 0
    skipped_count = 0
    selected_ids: list[str] = []
    total = len(tables)
    try:
        for idx, table in enumerate(tables, start=1):
            target = tables_by_id.get(table["table_id"], table)
            selected_ids.append(target["table_id"])
            existing = target.get("vlm_parse") or {}
            if skip_existing and existing.get("status") == "success" and existing.get("markdown"):
                skipped_count += 1
                if on_progress:
                    on_progress(
                        {
                            "index": idx,
                            "total": total,
                            "table_id": target["table_id"],
                            "parsed_count": parsed_count,
                            "skipped_count": skipped_count,
                            "status": "skipped",
                        }
                    )
                continue
            before, after = table_context_sentences(doc, target)
            page_label = f"p{target.get('page_start')}"
            if target.get("page_end") and target.get("page_end") != target.get("page_start"):
                page_label = f"p{target.get('page_start')}-{target.get('page_end')}"
            image_path = image_dir / f"{target['table_id']}_{page_label}.png"
            render_table_crop(doc, target, image_path, dpi=dpi, multi_page_dpi=multi_page_dpi)
            result = call_vlm_table_parse(image_path, target, before, after)
            result["render_dpi"] = multi_page_dpi if (target.get("bbox_by_page") or []) and len(target["bbox_by_page"]) > 1 else dpi
            result["stitched_pages"] = len(target.get("bbox_by_page") or []) or 1
            try:
                result["image_path"] = str(image_path.relative_to(assets_path.parent))
            except ValueError:
                result["image_path"] = str(image_path)
            target["vlm_parse"] = result
            if result.get("status") == "success":
                parsed_count += 1
            counts = dict(assets.get("counts", {}))
            counts["vlm_tables_parsed"] = sum(
                1 for row in assets.get("tables", []) if (row.get("vlm_parse") or {}).get("status") == "success"
            )
            assets["counts"] = counts
            assets_path.write_text(json.dumps(assets, indent=2, ensure_ascii=False), encoding="utf-8")
            if on_progress:
                on_progress(
                    {
                        "index": idx,
                        "total": total,
                        "table_id": target["table_id"],
                        "parsed_count": parsed_count,
                        "skipped_count": skipped_count,
                        "status": result.get("status") or "unknown",
                    }
                )
    finally:
        doc.close()

    counts = dict(assets.get("counts", {}))
    counts["vlm_tables_parsed"] = sum(1 for table in assets.get("tables", []) if (table.get("vlm_parse") or {}).get("status") == "success")
    assets["counts"] = counts
    assets_path.write_text(json.dumps(assets, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "assets_path": str(assets_path),
        "parsed_count": parsed_count,
        "skipped_count": skipped_count,
        "vlm_tables_parsed": counts["vlm_tables_parsed"],
        "selected_table_ids": selected_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline VLM parse for table images -> markdown + summary in assets.json.")
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--max-tables", type=int, default=None)
    parser.add_argument("--min-page", type=int, default=3, help="Skip tables before this page (default 3 = after TOC, from Item 1 body).")
    parser.add_argument("--exclude-item", action="append", default=[], help="Skip tables linked to these Items, e.g. Item 5 stock table.")
    parser.add_argument("--table-id", action="append", default=[], help="Only parse these table_id values.")
    parser.add_argument("--force", action="store_true", help="Re-parse tables even if vlm_parse already exists.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_VLM_DPI, help="Render DPI for single-page tables.")
    parser.add_argument(
        "--multi-page-dpi",
        type=int,
        default=DEFAULT_VLM_MULTI_PAGE_DPI,
        help="Render DPI when stitching cross-page tables.",
    )
    parser.add_argument(
        "--refresh-denoise-only",
        action="store_true",
        help="Only refresh denoised summary/context fields from existing vlm_parse (no VLM calls).",
    )
    args = parser.parse_args()
    if args.refresh_denoise_only:
        print(json.dumps(refresh_vlm_denoise_fields(args.assets), indent=2))
        return
    if not args.pdf:
        parser.error("--pdf is required unless --refresh-denoise-only is set")
    result = enrich_assets_with_vlm(
        pdf_path=args.pdf,
        assets_path=args.assets,
        image_dir=args.image_dir,
        max_tables=args.max_tables,
        min_page=args.min_page,
        exclude_items=args.exclude_item or None,
        skip_existing=not args.force,
        table_ids=args.table_id or None,
        dpi=args.dpi,
        multi_page_dpi=args.multi_page_dpi,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
