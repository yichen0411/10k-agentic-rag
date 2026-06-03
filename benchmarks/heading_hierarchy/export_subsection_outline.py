#!/usr/bin/env python3
"""Export compact indented subsection outlines (names only)."""

from __future__ import annotations

import json
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
RESULTS = BENCH_ROOT / "results"


def outline_from_hierarchy(payload: dict) -> str:
    lines: list[str] = []
    for section in payload.get("items") or []:
        item = section.get("item") or "?"
        lines.append(item)
        for row in section.get("reading_order") or []:
            level = int(row.get("level") or 2)
            indent = "  " * max(level - 1, 0)
            title = (row.get("title") or "").strip()
            if title:
                lines.append(f"{indent}{title}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    pairs = [
        ("font_subsection_hierarchy.json", "font_subsection_outline.txt"),
        ("layout_hybrid_subsection_hierarchy.json", "layout_hybrid_subsection_outline.txt"),
    ]
    for src_name, dst_name in pairs:
        src = RESULTS / src_name
        dst = RESULTS / dst_name
        payload = json.loads(src.read_text(encoding="utf-8"))
        dst.write_text(outline_from_hierarchy(payload), encoding="utf-8")
        print(f"Wrote {dst} ({len(payload.get('items') or [])} items)")


if __name__ == "__main__":
    main()
