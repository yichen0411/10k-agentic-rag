"""DocLayout-YOLO / MinerU — optional, heavy deps; stub with install guidance."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def run(pdf_path: Path, pages: set[int]) -> dict[str, Any]:
    t0 = time.perf_counter()
    return {
        "adapter": "doclayout_yolo_mineru",
        "status": "skipped",
        "error": "Not run in this benchmark pass (MinerU stack is heavy: torch, model weights, CLI).",
        "tables": [],
        "latency_sec": round(time.perf_counter() - t0, 3),
        "install_note": "See https://github.com/opendatalab/MinerU — DocLayout-YOLO detector + pipeline",
        "applicability": "Fast layout detection; still needs custom 10-K section linker + cross-page table merge. "
        "Comparable to replacing find_tables(), not replacing full table_pipeline.",
    }
