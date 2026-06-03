"""CLI entry point for the LangChain financial research agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
MAIN_ROOT = AGENT_DIR.parent
INFERENCE_DIR = MAIN_ROOT / "inference"

for path in [str(AGENT_DIR), str(INFERENCE_DIR), str(MAIN_ROOT.parent)]:
    if path not in sys.path:
        sys.path.insert(0, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LangChain agent over SQL, 10-K RAG, and optional email tools.",
    )
    parser.add_argument("query", help="User question")
    parser.add_argument("--json", action="store_true", help="Print full JSON payload")
    parser.add_argument("--max-steps", type=int, default=6, help="Max tool iterations")
    args = parser.parse_args()

    from langchain_agent import run_langchain_agent  # noqa: E402

    result = run_langchain_agent(args.query, max_steps=args.max_steps)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result.get("answer") or "")


if __name__ == "__main__":
    main()
