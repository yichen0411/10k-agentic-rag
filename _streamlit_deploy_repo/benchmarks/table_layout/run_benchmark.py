#!/usr/bin/env python3
"""Benchmark layout/table extractors on MSFT 10-K hard cases (standalone, no repo code changes)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCH_ROOT.parents[1]
sys.path.insert(0, str(BENCH_ROOT))

from scoring import score_adapter  # noqa: E402
from adapters import baseline_assets, doclayout_yolo_stub, docling_adapter, pymupdf_find_tables, pymupdf4llm_layout  # noqa: E402

DEFAULT_CASES = BENCH_ROOT / "cases" / "msft_fy2025_hard_tables.json"
DEFAULT_OUT = BENCH_ROOT / "results"


def load_casebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pages_from_cases(casebook: dict) -> set[int]:
    pages: set[int] = set()
    for case in casebook.get("cases", []):
        pages.update(int(p) for p in case.get("pages") or [])
    return pages


def resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p)


def run_adapters(pdf: Path, pages: set[int], assets: Path | None, only: set[str] | None) -> list[dict]:
    specs = [
        ("baseline", lambda: baseline_assets.run(pdf, pages, assets)),
        ("find_tables", lambda: pymupdf_find_tables.run(pdf, pages)),
        ("pymupdf4llm", lambda: pymupdf4llm_layout.run(pdf, pages)),
        ("docling", lambda: docling_adapter.run(pdf, pages)),
        ("doclayout_yolo", lambda: doclayout_yolo_stub.run(pdf, pages)),
    ]
    results = []
    for name, fn in specs:
        if only and name not in only:
            continue
        results.append(fn())
    return results


def verdict(scores: list[dict]) -> str:
    lines = [
        "## Verdict for this 10-K use case",
        "",
        "These tools solve **document layout / table detection**, not **10-K Item/subsection-aware RAG**:",
        "",
        "- **Cross-page header-only bands** (e.g. `(In millions)` on p42, data on p43) require explicit page-break linking.",
        "- **Section/subsection refs** for chunk markers (`[[TABLE:...]]` in Item 7 > Unearned Revenue) need TOC-guided sectioning — none of these provide that out of the box.",
        "- **Drop-in replacement?** Unlikely for production without a middle layer similar to `table_pipeline`.",
        "",
        "| Tool | Role in your stack |",
        "|------|-------------------|",
        "| `table_pipeline_v2` (baseline) | End-to-end for 10-K + chunk refs |",
        "| PyMuPDF `find_tables` | Raw detector only; fragments + misses headers |",
        "| PyMuPDF4LLM Layout | Better markdown/RAG text; closed layout weights |",
        "| Docling | Open table structure; add section linker + cross-page merge |",
        "| DocLayout-YOLO / MinerU | Faster layout bbox; same linker gap |",
        "",
    ]
    ranked = sorted([s for s in scores if s.get("status") == "ok"], key=lambda s: s.get("avg_score", 0), reverse=True)
    if ranked:
        lines.append("**Scores on hard cases (higher = more golden text found):**")
        for s in ranked:
            lines.append(f"- {s['adapter']}: **{s['avg_score']:.3f}** ({s.get('latency_sec')}s, {s.get('table_count')} tables)")
    skipped = [s for s in scores if s.get("status") != "ok"]
    if skipped:
        lines.append("")
        lines.append("**Skipped/failed (install to include):**")
        for s in skipped:
            lines.append(f"- {s['adapter']}: {s.get('status')} — {s.get('install_note') or s.get('error')}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark table/layout extractors on MSFT 10-K hard cases")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated adapters: baseline,find_tables,pymupdf4llm,docling,doclayout_yolo",
    )
    args = parser.parse_args()

    casebook = load_casebook(args.cases)
    pdf = resolve_path(casebook["pdf"])
    assets = resolve_path(casebook["workspace_assets"]) if casebook.get("workspace_assets") else None
    pages = pages_from_cases(casebook)
    only = {x.strip() for x in args.only.split(",") if x.strip()} or None

    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")

    raw_results = run_adapters(pdf, pages, assets, only)
    scored = [score_adapter(casebook, r) for r in raw_results]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = args.output_dir / f"benchmark_{stamp}.json"
    out_md = args.output_dir / f"benchmark_{stamp}.md"

    payload = {
        "generated_at": stamp,
        "pdf": str(pdf),
        "pages": sorted(pages),
        "casebook": str(args.cases),
        "results": scored,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        f"# Table layout benchmark — MSFT FY2025 hard cases",
        "",
        f"- PDF: `{pdf}`",
        f"- Pages under test: {sorted(pages)}",
        f"- Generated: {stamp}",
        "",
        "## Summary",
        "",
        "| Adapter | Status | Avg score | Latency (s) | Tables |",
        "|---------|--------|-----------|-------------|--------|",
    ]
    for s in scored:
        md_lines.append(
            f"| {s['adapter']} | {s['status']} | {s.get('avg_score', '-')} | {s.get('latency_sec', '-')} | {s.get('table_count', '-')} |"
        )
    md_lines.append("")
    md_lines.append(verdict(scored))
    md_lines.append("")
    md_lines.append("## Per-case breakdown")
    for s in scored:
        md_lines.append(f"### {s['adapter']} ({s['status']})")
        if s.get("error"):
            md_lines.append(f"_Error: {s['error']}_")
        md_lines.append("")
        for case in s.get("cases") or []:
            md_lines.append(
                f"- **{case['case_id']}** score={case['score']} "
                f"cross_page={case['cross_page_ok']} misses={case.get('text_misses')}"
            )
        md_lines.append("")

    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))
    print(f"\nWrote {out_json}\nWrote {out_md}")


if __name__ == "__main__":
    main()
