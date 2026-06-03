#!/usr/bin/env python3
"""Merge per-workspace assets.json files into one lookup payload for multi-filing RAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filing_metadata import filing_identity  # noqa: E402

DEFAULT_WORKSPACE_DIR = ROOT / "data" / "chunk_studio"
DEFAULT_OUT = ROOT / "data" / "index" / "merged_assets.json"


def merge_workspace_assets(workspace_dir: Path) -> dict[str, Any]:
    tables: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    source_files: list[str] = []

    for assets_path in sorted(workspace_dir.glob("*/assets.json")):
        payload = json.loads(assets_path.read_text(encoding="utf-8"))
        source_file, ticker, fiscal_year = filing_identity(
            payload.get("source_file") or assets_path.name,
            original_filename=None,
            json_path=assets_path,
        )
        if not payload.get("tables") and not payload.get("images"):
            continue
        source_files.append(source_file)
        for table in payload.get("tables", []):
            entry = dict(table)
            entry["source_file"] = source_file
            entry["ticker"] = ticker
            entry["fiscal_year"] = fiscal_year
            tables.append(entry)
        for image in payload.get("images", []):
            entry = dict(image)
            entry["source_file"] = source_file
            entry["ticker"] = ticker
            entry["fiscal_year"] = fiscal_year
            images.append(entry)

    return {
        "source_file": "all_filings",
        "method": "merged_chunk_studio_assets",
        "source_files": sorted(set(source_files)),
        "counts": {
            "filings": len(set(source_files)),
            "tables": len(tables),
            "images": len(images),
        },
        "tables": tables,
        "images": images,
    }


def write_merged_assets(workspace_dir: Path, out_path: Path) -> dict[str, Any]:
    payload = merge_workspace_assets(workspace_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge workspace assets.json into one multi-filing payload.")
    parser.add_argument("--workspace-dir", type=Path, default=DEFAULT_WORKSPACE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = write_merged_assets(args.workspace_dir, args.out)
    print(json.dumps(payload["counts"], indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
