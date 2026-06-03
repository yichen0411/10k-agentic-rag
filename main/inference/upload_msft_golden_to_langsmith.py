#!/usr/bin/env python3
"""Upload MSFT golden eval JSON results to LangSmith Experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from langsmith_eval_logging import DEFAULT_DATASET_NAME, upload_eval_results_to_langsmith
from text_vector_rag_inference import load_env_file

DEFAULT_DATASET = ROOT / "main" / "common" / "msft_fy2025_golden_eval_50.json"
DEFAULT_RESULTS = INFERENCE_DIR / "msft_fy2025_golden_eval_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload golden eval results to LangSmith Experiments.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--experiment-prefix", default="msft-fy2025-golden-eval-v1")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    args = parser.parse_args()

    load_env_file()
    info = upload_eval_results_to_langsmith(
        dataset_path=args.dataset,
        results_path=args.results,
        experiment_prefix=args.experiment_prefix,
        dataset_name=args.dataset_name,
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
