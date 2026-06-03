#!/usr/bin/env python3
"""Evaluate agent tool-call policy with mocked tools.

This checks whether the agent chooses the right next action and tool parameters.
It intentionally does not execute real SQL/RAG so retrieval quality cannot hide
policy mistakes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AGENT_DIR = Path(__file__).resolve().parent
MAIN_ROOT = AGENT_DIR.parent
PROJECT_ROOT = MAIN_ROOT.parent
INFERENCE_DIR = MAIN_ROOT / "inference"

for path in [str(AGENT_DIR), str(INFERENCE_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from system_prompt import build_langchain_system_prompt  # noqa: E402
from text_vector_rag_inference import load_env_file  # noqa: E402

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"


@dataclass
class ExpectedCall:
    tool: str
    args: dict[str, Any]


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "name": "sql",
            "description": "Query the read-only SQLite financial database for structured numbers, ratios, rankings, and comparisons.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A focused natural-language database question. Do not pass raw SQL.",
                    }
                },
                "required": ["question"],
            },
        },
        {
            "name": "rag",
            "description": "Search one scoped 10-K filing with vector retrieval and table context.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A single-intention filing question about one topic/section to retrieve.",
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Exactly one ticker: AAPL, MSFT, or GOOGL.",
                    },
                    "fiscal_year": {
                        "type": "string",
                        "description": "Exactly one fiscal year, e.g. FY2025.",
                    },
                },
                "required": ["question"],
            },
        },
        {
            "name": "send_email",
            "description": "Email the final research answer to the user.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "to_email": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to_email", "subject", "body"],
            },
        },
    ]


def build_system_prompt() -> str:
    system = (
        build_langchain_system_prompt()
        + "\n\n## Tool policy eval mode\n"
        + "Prior tool observations may be provided inside the user message. Treat them as already completed tool observations. "
        + "Decide only the next action: call the next necessary tool, or answer if the provided observations are sufficient. "
        + "Do not call a tool just to verify an observation that already contains the needed evidence."
    )
    return system


def call_tool_policy_model(model: str, user_input: str) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "system": build_system_prompt(),
            "messages": [{"role": "user", "content": user_input}],
            "tools": tool_schemas(),
            "temperature": 0,
            "max_tokens": 1000,
        }
    ).encode("utf-8")
    result = subprocess.run(
        [
            "curl",
            "-sS",
            "--fail-with-body",
            ANTHROPIC_MESSAGES_URL,
            "-H",
            f"x-api-key: {os.environ['ANTHROPIC_API_KEY']}",
            "-H",
            "anthropic-version: 2023-06-01",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
        ],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        body = result.stdout.decode(errors="replace") or result.stderr.decode(errors="replace")
        raise RuntimeError(f"Anthropic call failed: {body[:500]}")
    return json.loads(result.stdout.decode("utf-8"))


CASES: list[dict[str, Any]] = [
    {
        "id": "direct_sql_current_ratio",
        "question": "Which company has the highest current ratio in the most recent fiscal year?",
        "observations": [],
        "expected_action": "tool",
        "expected_calls": [
            {"tool": "sql", "args": {"question_contains": ["current ratio", "current assets", "current liabilities"]}}
        ],
    },
    {
        "id": "direct_sql_services_vs_iphone_growth",
        "question": "Apple's Services segment has been growing as a share of total revenue. Calculate the Services revenue growth rate for each of the last two years and compare it to iPhone revenue growth.",
        "observations": [],
        "expected_action": "tool",
        "expected_calls": [
            {"tool": "sql", "args": {"question_contains": ["Apple", "Services", "iPhone", "FY2023", "FY2024", "FY2025"]}}
        ],
    },
    {
        "id": "direct_sql_cross_company_margin",
        "question": "Compare Apple and Microsoft operating margin in FY2025.",
        "observations": [],
        "expected_action": "tool",
        "expected_calls": [
            {"tool": "sql", "args": {"question_contains": ["Apple", "Microsoft", "operating", "margin", "FY2025"]}}
        ],
    },
    {
        "id": "direct_rag_services_components",
        "question": "What does Apple's FY2025 10-K say Services includes?",
        "observations": [],
        "expected_action": "tool",
        "expected_calls": [
            {"tool": "rag", "args": {"question_contains": ["Services", "include"], "ticker": "AAPL", "fiscal_year": "FY2025"}}
        ],
    },
    {
        "id": "direct_rag_services_growth_drivers",
        "question": "What did Apple's FY2025 10-K say drove Services net sales growth?",
        "observations": [],
        "expected_action": "tool",
        "expected_calls": [
            {"tool": "rag", "args": {"question_contains": ["Services", "net sales", "growth"], "ticker": "AAPL", "fiscal_year": "FY2025"}}
        ],
    },
    {
        "id": "direct_rag_risk_factor",
        "question": "What cybersecurity risks did Microsoft discuss in its FY2025 10-K?",
        "observations": [],
        "expected_action": "tool",
        "expected_calls": [
            {"tool": "rag", "args": {"question_contains": ["cybersecurity", "risk"], "ticker": "MSFT", "fiscal_year": "FY2025"}}
        ],
    },
    {
        "id": "hybrid_q025_turn1_sql_first",
        "question": "Apple's Services segment has been growing as a share of total revenue. Calculate the Services revenue growth rate for each of the last two years, compare it to iPhone revenue growth, and find what Apple's 10-K says about the strategic importance of Services.",
        "observations": [],
        "expected_action": "tool",
        "expected_calls": [
            {"tool": "sql", "args": {"question_contains": ["Apple", "Services", "iPhone", "FY2023", "FY2024", "FY2025"]}}
        ],
    },
    {
        "id": "hybrid_q025_turn2_rag_after_sql",
        "question": "Apple's Services segment has been growing as a share of total revenue. Calculate the Services revenue growth rate for each of the last two years, compare it to iPhone revenue growth, and find what Apple's 10-K says about the strategic importance of Services.",
        "observations": [
            {
                "tool": "sql",
                "result": "Services grew 12.9% in FY2024 and 13.5% in FY2025; iPhone grew 0.3% and 4.2%; Services share rose to 26.2%.",
            }
        ],
        "expected_action": "tool",
        "expected_calls": [
            {
                "tool": "rag",
                "args": {
                    "question_contains_any": [["Services", "includes"], ["Services", "growth"], ["Services", "gross margin"]],
                    "ticker": "AAPL",
                    "fiscal_year": "FY2025",
                },
            }
        ],
    },
    {
        "id": "hybrid_q025_turn3_stop",
        "question": "Apple's Services segment has been growing as a share of total revenue. Calculate the Services revenue growth rate for each of the last two years, compare it to iPhone revenue growth, and find what Apple's 10-K says about the strategic importance of Services.",
        "observations": [
            {"tool": "sql", "result": "Services grew 12.9% and 13.5%; iPhone grew 0.3% and 4.2%."},
            {
                "tool": "rag",
                "result": "Apple says Services includes advertising, AppleCare, cloud services, App Store and subscription-based digital content, and payment services. Services growth was driven by advertising, App Store, and cloud services; Services gross margin was 75.4%.",
            },
        ],
        "expected_action": "answer",
        "expected_calls": [],
    },
    {
        "id": "hybrid_region_turn1_sql_first",
        "question": "Which Apple geographic region grew the fastest in FY2025, and what did the 10-K say about that region's sales performance?",
        "observations": [],
        "expected_action": "tool",
        "expected_calls": [
            {"tool": "sql", "args": {"question_contains": ["Apple", "geographic", "region", "FY2025", "growth"]}}
        ],
    },
    {
        "id": "hybrid_region_turn2_rag_after_sql",
        "question": "Which Apple geographic region grew the fastest in FY2025, and what did the 10-K say about that region's sales performance?",
        "observations": [
            {"tool": "sql", "result": "Rest of Asia Pacific grew fastest in FY2025."}
        ],
        "expected_action": "tool",
        "expected_calls": [
            {"tool": "rag", "args": {"question_contains": ["Rest of Asia Pacific", "sales"], "ticker": "AAPL", "fiscal_year": "FY2025"}}
        ],
    },
    {
        "id": "guidance_vs_actual_turn1_rag_first",
        "question": "Compare management guidance vs actual Intelligent Cloud growth in FY2025.",
        "observations": [],
        "expected_action": "tool",
        "expected_calls": [
            {"tool": "rag", "args": {"question_contains": ["Intelligent Cloud", "guidance", "growth"], "ticker": "MSFT", "fiscal_year": "FY2025"}}
        ],
    },
    {
        "id": "guidance_vs_actual_turn2_sql_after_rag",
        "question": "Compare management guidance vs actual Intelligent Cloud growth in FY2025.",
        "observations": [
            {"tool": "rag", "result": "Management guided Intelligent Cloud revenue growth of 18% to 19%."}
        ],
        "expected_action": "tool",
        "expected_calls": [
            {"tool": "sql", "args": {"question_contains": ["Intelligent Cloud", "revenue", "growth", "FY2024", "FY2025"]}}
        ],
    },
    {
        "id": "guidance_vs_actual_turn3_stop",
        "question": "Compare management guidance vs actual Intelligent Cloud growth in FY2025.",
        "observations": [
            {"tool": "rag", "result": "Management guided Intelligent Cloud revenue growth of 18% to 19%."},
            {"tool": "sql", "result": "Actual Intelligent Cloud revenue growth was 21% YoY."},
        ],
        "expected_action": "answer",
        "expected_calls": [],
    },
    {
        "id": "compare_rag_decompose_two_tickers",
        "question": "Compare Microsoft and Alphabet AI risk disclosures in FY2025.",
        "observations": [],
        "expected_action": "tool",
        "expected_calls": [
            {"tool": "rag", "args": {"question_contains": ["AI", "risk"], "ticker": "MSFT", "fiscal_year": "FY2025"}},
            {"tool": "rag", "args": {"question_contains": ["AI", "risk"], "ticker": "GOOGL", "fiscal_year": "FY2025"}},
        ],
        "allow_any_order": True,
    },
    {
        "id": "sql_observation_stop",
        "question": "What was Apple's FY2025 revenue?",
        "observations": [
            {"tool": "sql", "result": "Apple FY2025 revenue was $416.161 billion."}
        ],
        "expected_action": "answer",
        "expected_calls": [],
    },
    {
        "id": "rag_observation_stop",
        "question": "What does Apple Services include in the FY2025 10-K?",
        "observations": [
            {"tool": "rag", "result": "Services includes advertising, AppleCare, cloud services, digital content such as the App Store and Apple Music, and payment services."}
        ],
        "expected_action": "answer",
        "expected_calls": [],
    },
    {
        "id": "rag_failure_fallback_concrete_terms",
        "question": "What does Apple's 10-K say about the strategic importance of Services?",
        "observations": [
            {"tool": "rag", "result": "The provided context does not use the phrase strategic importance of Services."}
        ],
        "expected_action": "tool",
        "expected_calls": [
            {
                "tool": "rag",
                "args": {
                    "question_contains_any": [["Services", "includes"], ["Services", "growth"], ["Services", "gross margin"], ["App Store", "cloud", "advertising"]],
                    "ticker": "AAPL",
                    "fiscal_year": "FY2025",
                },
            }
        ],
    },
    {
        "id": "direct_answer_out_of_scope",
        "question": "What was Tesla revenue in FY2025?",
        "observations": [],
        "expected_action": "answer",
        "expected_calls": [],
    },
]


def format_eval_input(case: dict[str, Any]) -> str:
    lines = [
        "Evaluate the next best action for this financial research request.",
        f"User question: {case['question']}",
    ]
    if case.get("observations"):
        lines.append("\nExisting tool observations:")
        for idx, obs in enumerate(case["observations"], 1):
            lines.append(f"{idx}. tool={obs['tool']} result={obs['result']}")
    else:
        lines.append("\nExisting tool observations: none")
    lines.append("\nTake exactly the next necessary step. If the observations are sufficient, answer without calling tools.")
    return "\n".join(lines)


def normalize_text(value: Any) -> str:
    return str(value or "").lower()


def canonical_tool_input(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return {"question": raw}
    return {}


def value_matches(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual in (None, "", [], {})
    return normalize_text(actual).strip() == normalize_text(expected).strip()


def args_match(actual: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    question = normalize_text(actual.get("question"))
    for key, expected_value in expected.items():
        if key == "question_contains":
            missing = [term for term in expected_value if normalize_text(term) not in question]
            if missing:
                failures.append(f"question missing {missing}")
        elif key == "question_not_contains":
            present = [term for term in expected_value if normalize_text(term) in question]
            if present:
                failures.append(f"question unexpectedly contains {present}")
        elif key == "question_contains_any":
            matched_group = False
            for group in expected_value:
                if all(normalize_text(term) in question for term in group):
                    matched_group = True
                    break
            if not matched_group:
                failures.append(f"question matched none of {expected_value}")
        else:
            if not value_matches(actual.get(key), expected_value):
                failures.append(f"{key} expected={expected_value!r} actual={actual.get(key)!r}")
    return not failures, failures


def score_case(case: dict[str, Any], trace: list[dict[str, Any]], answer: str) -> dict[str, Any]:
    expected_action = case["expected_action"]
    expected_calls = [ExpectedCall(call["tool"], call.get("args", {})) for call in case.get("expected_calls", [])]
    actual_calls = [
        {
            "tool": step.get("tool"),
            "args": canonical_tool_input(step.get("tool_input")),
        }
        for step in trace
    ]

    result: dict[str, Any] = {
        "id": case["id"],
        "expected_action": expected_action,
        "actual_action": "tool" if actual_calls else "answer",
        "expected_calls": [call.__dict__ for call in expected_calls],
        "actual_calls": actual_calls,
        "answer_excerpt": answer[:240],
        "passed": False,
        "score": 0,
        "failures": [],
    }

    if expected_action == "answer":
        if actual_calls:
            result["failures"].append("expected answer/no tool, but tool was called")
        else:
            result["score"] = 100
            result["passed"] = True
        return result

    if not actual_calls:
        result["failures"].append("expected tool call, but agent answered")
        return result

    score = 40
    unmatched_actual = actual_calls.copy()
    for expected in expected_calls:
        candidates = [idx for idx, call in enumerate(unmatched_actual) if call["tool"] == expected.tool]
        if not candidates:
            result["failures"].append(f"missing tool {expected.tool}")
            continue
        match_idx = None
        match_failures: list[str] = []
        for idx in candidates:
            ok, failures = args_match(unmatched_actual[idx]["args"], expected.args)
            if ok:
                match_idx = idx
                break
            if not match_failures:
                match_failures = failures
        if match_idx is None:
            result["failures"].append(f"{expected.tool} args mismatch: {match_failures}")
            unmatched_actual.pop(candidates[0])
        else:
            score += 40 / max(len(expected_calls), 1)
            unmatched_actual.pop(match_idx)

    if len(actual_calls) == len(expected_calls):
        score += 20
    else:
        result["failures"].append(f"expected {len(expected_calls)} call(s), got {len(actual_calls)}")

    result["score"] = round(min(score, 100), 1)
    result["passed"] = not result["failures"]
    return result


def run_case(model: str, case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    response = call_tool_policy_model(model, format_eval_input(case))
    trace = [
        {"tool": block.get("name"), "tool_input": block.get("input") or {}}
        for block in response.get("content", [])
        if block.get("type") == "tool_use"
    ]
    answer = "\n".join(
        block.get("text", "")
        for block in response.get("content", [])
        if block.get("type") == "text"
    ).strip()
    scored = score_case(case, trace, answer)
    scored["latency_sec"] = round(time.perf_counter() - started, 3)
    scored["stop_reason"] = response.get("stop_reason")
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agent tool-call policy using mocked tools.")
    parser.add_argument("--output", type=Path, default=AGENT_DIR / "tool_policy_eval_results.json")
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_AGENT_MODEL", "claude-haiku-4-5-20251001"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    load_env_file(PROJECT_ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required.")

    cases = CASES[: args.limit] if args.limit else CASES
    results = []
    for idx, case in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] {case['id']}", flush=True)
        results.append(run_case(args.model, case))

    passed = sum(1 for row in results if row["passed"])
    summary = {
        "model": args.model,
        "n": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 3) if results else None,
        "mean_score": round(sum(row["score"] for row in results) / len(results), 1) if results else None,
    }
    payload = {"summary": summary, "results": results}
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for row in results:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"{status} {row['id']} score={row['score']} actual={row['actual_calls']} failures={row['failures']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
