#!/usr/bin/env python3
"""Patch Chunk Studio workspace JSON artifacts to use original_filename instead of source.pdf."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filing_metadata import patch_workspace_source_files  # noqa: E402

DEFAULT_WORKSPACE_DIR = ROOT / "data" / "chunk_studio"


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch workspace source_file metadata to original PDF filenames.")
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=DEFAULT_WORKSPACE_DIR,
        help="Chunk Studio workspace root containing per-file directories.",
    )
    args = parser.parse_args()

    if not args.workspace_dir.exists():
        raise FileNotFoundError(f"Workspace dir not found: {args.workspace_dir}")

    results = []
    for path in sorted(args.workspace_dir.iterdir()):
        if path.is_dir() and (path / "metadata.json").exists():
            results.append(patch_workspace_source_files(path))

    print(json.dumps(results, indent=2, ensure_ascii=False))
    patched = sum(1 for row in results if row.get("patched"))
    print(f"Patched {patched}/{len(results)} workspaces.")


if __name__ == "__main__":
    main()
