#!/usr/bin/env python3
"""
GOOGL FY2024 10-K — page 67 (table_027) screenshot table extraction probe.

Compares non-VLM extractors on a table PNG vs:
  - production PyMuPDF find_tables raw_rows (PDF vector path)
  - VLM markdown ground truth in assets.json

Optional deps (install what you want to try):
  pip install img2table opencv-python-headless pillow pytesseract
  brew install tesseract   # macOS binary for pytesseract / img2table Tesseract backend

  pip install paddlepaddle paddleocr   # heavy; PP-Structure table on image
  pip install docling                # also works on images; first run downloads weights

Usage:
  python scripts/explore_googl_p67_image_table_extract.py
  python scripts/explore_googl_p67_image_table_extract.py --methods img2table,opencv_tesseract
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
CHUNKING = REPO / "0525_redo" / "chunking"
sys.path.insert(0, str(CHUNKING))

PDF_PATH = REPO / "data" / "pdfs" / "GOOGL_FY2024_10-K.pdf"
WORKSPACE = REPO / "data" / "chunk_studio" / "1780269752-googl-fy2024-10-k-39f774c1bb"
ASSETS_PATH = WORKSPACE / "assets.json"
TABLE_ID = "table_027"
PAGE = 67

# Spot-check cells from successful VLM parse (fair value hierarchy table, FY2023 column group).
GOLDEN_SNIPPETS = [
    "Fair value changes recorded in other comprehensive income",
    "Time deposits",
    "Level 2",
    "38,106",
    "80,434",
    "(1,950)",
    "98,407",
    "24,048",
    "86,868",
    "Money market funds",
    "6,480",
    "Fair Value Hierarchy",
    "Adjusted Cost",
]


def load_table() -> dict[str, Any]:
    assets = json.loads(ASSETS_PATH.read_text(encoding="utf-8"))
    for t in assets.get("tables", []):
        if t.get("table_id") == TABLE_ID:
            return t
    raise SystemExit(f"{TABLE_ID} not found in {ASSETS_PATH}")


def render_crop_png(table: dict[str, Any], out_dir: Path, dpi: int = 132) -> Path:
    import fitz
    from vlm_table_parse import render_table_crop

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{TABLE_ID}_p{PAGE}_dpi{dpi}.png"
    doc = fitz.open(PDF_PATH)
    try:
        render_table_crop(doc, table, out_path, dpi=dpi)
    finally:
        doc.close()
    return out_path


def score_text(label: str, text: str) -> dict[str, Any]:
    hits = [s for s in GOLDEN_SNIPPETS if s in text]
    missing = [s for s in GOLDEN_SNIPPETS if s not in text]
    return {
        "method": label,
        "golden_hits": len(hits),
        "golden_total": len(GOLDEN_SNIPPETS),
        "hit_rate": round(len(hits) / len(GOLDEN_SNIPPETS), 3),
        "missing": missing,
        "sample_hits": hits[:6],
        "char_len": len(text),
        "preview": text[:1200] + ("…" if len(text) > 1200 else ""),
    }


def method_pymupdf_find_tables(table: dict[str, Any]) -> str:
    """PDF vector path (not screenshot) — shows why raw_rows are fragmented."""
    import fitz

    doc = fitz.open(PDF_PATH)
    try:
        page = doc[PAGE - 1]
        finder = page.find_tables()
        tables = list(finder.tables)
        lines = [
            f"find_tables() count on page {PAGE}: {len(tables)}",
            f"assets.json {TABLE_ID}: col_count={table.get('col_count')} row_count={table.get('row_count')}",
            f"complexity: {table.get('complexity')} reasons={table.get('complexity_reasons')}",
            "",
            "--- first 3 raw_rows from assets (production merge) ---",
        ]
        for row in (table.get("raw_rows") or [])[:3]:
            cells = [c if c is not None else "" for c in row]
            lines.append(" | ".join(cells))
        lines.append("")
        lines.append("--- first row null density ---")
        first = table.get("first_row") or []
        nulls = sum(1 for c in first if c is None)
        lines.append(f"null cells in first_row: {nulls}/{len(first)}")
        if table.get("vlm_parse", {}).get("markdown"):
            lines.append("")
            lines.append("--- VLM markdown header (reference) ---")
            lines.append(table["vlm_parse"]["markdown"].split("\n")[0])
            lines.append(table["vlm_parse"]["markdown"].split("\n")[1])
        return "\n".join(lines)
    finally:
        doc.close()


def _img2table_ocr():
    """Prefer Tesseract if binary exists; else EasyOCR (pip-only, slower first run)."""
    import shutil

    from img2table.ocr import TesseractOCR

    if shutil.which("tesseract"):
        return TesseractOCR(n_threads=1, lang="eng"), "tesseract"
    from img2table.ocr import EasyOCR

    return EasyOCR(lang=["en"], kw={"gpu": False}), "easyocr"


def method_img2table(png_path: Path) -> str:
    from img2table.document import Image as Img2TableImage

    ocr, backend = _img2table_ocr()
    doc = Img2TableImage(str(png_path), detect_rotation=False)
    tables = doc.extract_tables(
        ocr=ocr,
        implicit_rows=True,
        borderless_tables=True,
        min_confidence=40,
    )
    if not tables:
        return f"(img2table/{backend} returned no tables)"
    parts: list[str] = [f"img2table backend: {backend}"]
    for i, tbl in enumerate(tables):
        df = tbl.df
        parts.append(f"=== img2table table {i} shape={df.shape} ===")
        parts.append(df.to_string(index=False, max_rows=30, max_cols=12))
    return "\n".join(parts)


def method_opencv_tesseract_grid(png_path: Path) -> str:
    """Naive: detect horizontal/vertical lines, OCR each cell bbox — common failure mode on 10-K tables."""
    import cv2
    import numpy as np
    import pytesseract
    from PIL import Image

    img = cv2.imread(str(png_path))
    if img is None:
        return "(failed to read image)"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    h_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, h_kernel, iterations=2)
    v_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, v_kernel, iterations=2)
    grid = cv2.add(h_lines, v_lines)

    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h_img, w_img = gray.shape
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 40 or bh < 12:
            continue
        if bw > w_img * 0.98 and bh > h_img * 0.98:
            continue
        boxes.append((x, y, bw, bh))
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))[:80]

    pil = Image.open(png_path)
    rows: list[str] = []
    rows.append(f"opencv grid cells sampled: {len(boxes)} (cap 80)")
    for x, y, bw, bh in boxes[:25]:
        crop = pil.crop((x, y, x + bw, y + bh))
        txt = pytesseract.image_to_string(crop, config="--psm 7").strip().replace("\n", " ")
        if txt:
            rows.append(f"[{x},{y},{bw}x{bh}] {txt}")
    if len(boxes) > 25:
        rows.append(f"... ({len(boxes) - 25} more cells omitted)")
    return "\n".join(rows) if rows else "(no OCR text from grid cells)"


def method_paddleocr_structure(png_path: Path) -> str:
    from paddleocr import PPStructure

    engine = PPStructure(show_log=False, lang="en", layout=False)
    result = engine(str(png_path))
    if not result:
        return "(paddle PPStructure empty)"
    parts: list[str] = []
    for block in result:
        if block.get("type") != "table":
            continue
        res = block.get("res", {})
        html = res.get("html", "")
        parts.append("=== paddle table html (truncated) ===")
        parts.append(html[:2000])
    return "\n".join(parts) if parts else f"(no table blocks; keys={[b.get('type') for b in result]})"


def method_docling_image(png_path: Path) -> str:
    from docling.document_converter import DocumentConverter

    conv = DocumentConverter()
    doc = conv.convert(str(png_path)).document
    md = doc.export_to_markdown()
    return md or "(docling empty markdown)"


METHODS: dict[str, Callable[..., str]] = {
    "pymupdf_find_tables": lambda _png, table: method_pymupdf_find_tables(table),
    "img2table": lambda png, _table: method_img2table(png),
    "opencv_tesseract": lambda png, _table: method_opencv_tesseract_grid(png),
    "paddle_ppstructure": lambda png, _table: method_paddleocr_structure(png),
    "docling_image": lambda png, _table: method_docling_image(png),
}


def diagnose_issues(table: dict[str, Any], results: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    issues.append(
        f"Table {TABLE_ID} on PDF page {PAGE}: "
        f"{table.get('row_count')} rows × {table.get('col_count')} cols, "
        f"complexity={table.get('complexity')} ({', '.join(table.get('complexity_reasons') or [])})"
    )
    issues.append(
        "Layout: multi-level header band + section subheaders spanning full width "
        "(OCI block vs net-income block) — not a simple grid."
    )
    issues.append(
        "PDF find_tables splits currency into '$' and amount cells → 21 logical columns become sparse null grids."
    )

    for r in results:
        if r.get("error"):
            issues.append(f"{r['method']}: ERROR — {r['error']}")
            continue
        rate = r.get("hit_rate", 0)
        if r["method"] == "pymupdf_find_tables":
            continue
        if rate < 0.5:
            issues.append(
                f"{r['method']}: low golden recall ({rate}) — likely lost headers, merged columns, or OCR garble."
            )
        if r.get("char_len", 0) < 200 and r["method"].startswith(("img2table", "opencv", "paddle")):
            issues.append(f"{r['method']}: very short output — detector probably missed the table region.")

    img_methods = [r for r in results if r["method"] != "pymupdf_find_tables" and not r.get("error")]
    if img_methods and all(r.get("hit_rate", 0) < 0.7 for r in img_methods):
        issues.append(
            "Image OCR paths struggle without layout semantics: parentheses negatives, footnote (1)(2), "
            "empty cells in Level-1 rows, and duplicated 'Fair Value' column labels."
        )
    return issues


def run_method(name: str, png_path: Path, table: dict[str, Any]) -> dict[str, Any]:
    fn = METHODS[name]
    t0 = time.perf_counter()
    try:
        if name == "pymupdf_find_tables":
            text = fn(png_path, table)
        else:
            text = fn(png_path, table)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        scored = score_text(name, text)
        scored["elapsed_ms"] = elapsed_ms
        scored["error"] = None
        return scored
    except Exception as exc:
        return {
            "method": name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        default=",".join(METHODS.keys()),
        help=f"Comma-separated subset of: {', '.join(METHODS)}",
    )
    parser.add_argument("--dpi", type=int, default=132, help="PNG render DPI (match VLM parse)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "scripts" / "output" / "googl_p67_table_extract",
    )
    args = parser.parse_args()
    selected = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in selected if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Choose from {list(METHODS)}")

    if not PDF_PATH.is_file():
        raise SystemExit(f"PDF not found: {PDF_PATH}")

    table = load_table()
    png_path = render_crop_png(table, args.out_dir, dpi=args.dpi)
    print(f"Rendered crop: {png_path} ({png_path.stat().st_size // 1024} KB)")
    print(f"Section: {table.get('subsection_ref', {}).get('title')}")
    print()

    results: list[dict[str, Any]] = []
    for name in selected:
        print(f"--- {name} ---")
        row = run_method(name, png_path, table)
        results.append(row)
        if row.get("error"):
            print(f"  FAILED: {row['error']}")
            if row.get("traceback"):
                print(row["traceback"])
        else:
            print(
                f"  golden {row['golden_hits']}/{row['golden_total']} "
                f"({row['hit_rate']*100:.0f}%) in {row['elapsed_ms']} ms"
            )
            if row.get("missing"):
                print(f"  missing: {row['missing'][:8]}{'…' if len(row['missing']) > 8 else ''}")
            print()
            print(row["preview"])
        print()

    report = {
        "pdf": str(PDF_PATH),
        "table_id": TABLE_ID,
        "page": PAGE,
        "png": str(png_path),
        "dpi": args.dpi,
        "golden_snippets": GOLDEN_SNIPPETS,
        "results": results,
        "issues": diagnose_issues(table, results),
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== Diagnosis ===")
    for line in report["issues"]:
        print(f"• {line}")
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
