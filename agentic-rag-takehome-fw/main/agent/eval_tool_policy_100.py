#!/usr/bin/env python3
"""Run a 100-case agent tool-policy eval with detailed metrics.

This evaluates only the next-step policy: whether to stop or call tools, which
tool(s), and whether tool parameters preserve the needed scope and intent.
Real SQL/RAG tools are not executed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

AGENT_DIR = Path(__file__).resolve().parent
MAIN_ROOT = AGENT_DIR.parent
PROJECT_ROOT = MAIN_ROOT.parent
INFERENCE_DIR = MAIN_ROOT / "inference"

for path in [str(AGENT_DIR), str(INFERENCE_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from eval_tool_policy import ANTHROPIC_MESSAGES_URL, tool_schemas  # noqa: E402
from text_vector_rag_inference import load_env_file  # noqa: E402

COMPANIES = [
    ("Apple", "AAPL"),
    ("Microsoft", "MSFT"),
    ("Alphabet", "GOOGL"),
]


COMPACT_SYSTEM_PROMPT = """You are evaluating the next tool/action for a financial research agent.

Available tools:
- sql(question): use for exact structured financial numbers, ratios, rankings, growth rates, margins, segment/geographic revenue, balance sheet metrics, and cross-company numeric comparisons. SQL has only a question parameter, so include company/year/metric in that question.
- rag(question, ticker, fiscal_year): use for one 10-K filing evidence lookup: narrative, risk factors, MD&A commentary, business descriptions, management explanations, or table values from the filing. Each rag call must have one ticker and one fiscal year when known.
- send_email: ignore unless user asks to email.

Policy:
- Decide only the next action. If existing observations are sufficient, answer without tools.
- Do not repeat a tool call just to verify an observation that already contains the needed evidence.
- A RAG observation saying a phrase was not found, context is insufficient, or the filing does not
  explicitly mention abstract wording is not sufficient evidence. If the original filing question is
  still unanswered, retry RAG with the same ticker/fiscal_year and a concrete single-intent question
  using filing terms likely to appear in the document.
- For hybrid numeric + filing questions, get missing numeric facts from sql and missing filing evidence from rag.
- For comparisons requiring filing evidence across companies or years, call rag separately for each ticker/year. Independent rag calls may be issued together.
- For growth over each of the last two years, SQL should request three fiscal years of raw values.
- For abstract filing questions like strategic importance, role, importance, or growth driver, search concrete filing evidence: components/includes, growth drivers, gross margin/margins, risks, or business descriptions. Do not only search the abstract phrase.
- If SQL identified a concrete segment/region/entity, carry that entity into the next rag question.
- If the next step is a tool, emit the tool call. Do not only describe the tool call in prose.
- Do not answer with a meta-explanation that a better search should be run. If a better search is the
  next step and a supported tool exists, call that tool now.
- Supported companies are Apple/AAPL, Microsoft/MSFT, and Alphabet/GOOGL only.
"""


def call_compact_policy_model(model: str, user_input: str) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "system": COMPACT_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_input}],
            "tools": tool_schemas(),
            "temperature": 0,
            "max_tokens": 700,
        }
    ).encode("utf-8")
    for attempt in range(5):
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
        if result.returncode == 0:
            return json.loads(result.stdout.decode("utf-8"))
        body = result.stdout.decode(errors="replace") or result.stderr.decode(errors="replace")
        if "rate_limit_error" not in body and "overloaded_error" not in body:
            raise RuntimeError(f"Anthropic call failed: {body[:500]}")
        time.sleep(5 * (attempt + 1))
    body = result.stdout.decode(errors="replace") or result.stderr.decode(errors="replace")
    raise RuntimeError(f"Anthropic call failed after retries: {body[:500]}")


def sql_case(case_id: str, question: str, *, entities: list[list[str]], years: list[str], metrics: list[list[str]]) -> dict[str, Any]:
    return {
        "id": case_id,
        "category": "direct_sql",
        "question": question,
        "observations": [],
        "expected_action": "tool",
        "expected_calls": [
            {
                "tool": "sql",
                "args": {"entities": entities, "years": years, "metrics": metrics},
            }
        ],
    }


def rag_case(
    case_id: str,
    question: str,
    *,
    ticker: str,
    fiscal_year: str,
    entities: list[list[str]],
    evidence_any: list[list[str]],
    category: str = "direct_rag",
    observations: list[dict[str, str]] | None = None,
    dependency_terms: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "category": category,
        "question": question,
        "observations": observations or [],
        "expected_action": "tool",
        "expected_calls": [
            {
                "tool": "rag",
                "args": {
                    "ticker": ticker,
                    "fiscal_year": fiscal_year,
                    "entities": entities,
                    "evidence_any": evidence_any,
                    "dependency_terms": dependency_terms or [],
                },
            }
        ],
    }


def answer_case(case_id: str, question: str, observations: list[dict[str, str]], *, category: str = "stop") -> dict[str, Any]:
    return {
        "id": case_id,
        "category": category,
        "question": question,
        "observations": observations,
        "expected_action": "answer",
        "expected_calls": [],
    }


def compare_rag_case(case_id: str, question: str, pairs: list[tuple[str, str]], *, fiscal_year: str, topic_terms: list[str]) -> dict[str, Any]:
    return {
        "id": case_id,
        "category": "compare_decompose",
        "question": question,
        "observations": [],
        "expected_action": "tool",
        "expected_calls": [
            {
                "tool": "rag",
                "args": {
                    "ticker": ticker,
                    "fiscal_year": fiscal_year,
                    "entities": [[topic_terms[0]]],
                    "evidence_any": [topic_terms],
                },
            }
            for _company, ticker in pairs
        ],
        "allow_any_order": True,
    }


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # 15 direct SQL cases.
    cases.extend(
        [
            sql_case(
                "direct_sql_001_current_ratio_all",
                "Which company has the highest current ratio in the most recent fiscal year?",
                entities=[["Apple", "AAPL"], ["Microsoft", "MSFT"], ["Alphabet", "GOOGL"]],
                years=["FY2025"],
                metrics=[["current ratio", "current assets", "current liabilities"]],
            ),
            sql_case(
                "direct_sql_002_services_iphone_growth",
                "Calculate Apple's Services revenue growth for each of the last two years and compare it to iPhone growth.",
                entities=[["Apple", "AAPL"], ["Services"], ["iPhone"]],
                years=["FY2023", "FY2024", "FY2025"],
                metrics=[["revenue"], ["growth"]],
            ),
            sql_case(
                "direct_sql_003_operating_margin_aapl_msft",
                "Compare Apple and Microsoft operating margin in FY2025.",
                entities=[["Apple", "AAPL"], ["Microsoft", "MSFT"]],
                years=["FY2025"],
                metrics=[["operating margin", "operating income", "revenue"]],
            ),
            sql_case(
                "direct_sql_004_msft_segment_revenue",
                "What were Microsoft's three segment revenues in FY2025?",
                entities=[["Microsoft", "MSFT"], ["Intelligent Cloud"], ["Productivity"], ["More Personal Computing"]],
                years=["FY2025"],
                metrics=[["revenue"]],
            ),
            sql_case(
                "direct_sql_005_alphabet_cloud_growth",
                "What was Google Cloud revenue growth from FY2024 to FY2025?",
                entities=[["Alphabet", "GOOGL"], ["Google Cloud"]],
                years=["FY2024", "FY2025"],
                metrics=[["revenue"], ["growth"]],
            ),
            sql_case(
                "direct_sql_006_apple_geo_region",
                "Which Apple geographic region had the highest revenue in FY2025?",
                entities=[["Apple", "AAPL"], ["region", "geographic"]],
                years=["FY2025"],
                metrics=[["revenue"], ["highest", "largest"]],
            ),
            sql_case(
                "direct_sql_007_roa_all",
                "Which company had the highest return on assets in FY2025?",
                entities=[["Apple", "AAPL"], ["Microsoft", "MSFT"], ["Alphabet", "GOOGL"]],
                years=["FY2025"],
                metrics=[["return on assets", "ROA", "net income", "total assets"]],
            ),
            sql_case(
                "direct_sql_008_eps_msft",
                "What was Microsoft's diluted EPS in FY2025 compared with FY2024?",
                entities=[["Microsoft", "MSFT"]],
                years=["FY2024", "FY2025"],
                metrics=[["EPS", "diluted"]],
            ),
            sql_case(
                "direct_sql_009_total_debt_compare",
                "Compare Apple, Microsoft, and Alphabet total debt in FY2025.",
                entities=[["Apple", "AAPL"], ["Microsoft", "MSFT"], ["Alphabet", "GOOGL"]],
                years=["FY2025"],
                metrics=[["total debt", "debt"]],
            ),
            sql_case(
                "direct_sql_010_revenue_increase_all",
                "Which company had the largest absolute revenue increase from FY2024 to FY2025?",
                entities=[["Apple", "AAPL"], ["Microsoft", "MSFT"], ["Alphabet", "GOOGL"]],
                years=["FY2024", "FY2025"],
                metrics=[["revenue"], ["increase", "growth"]],
            ),
            sql_case(
                "direct_sql_011_aapl_services_share",
                "What share of Apple's FY2025 total revenue came from Services?",
                entities=[["Apple", "AAPL"], ["Services"]],
                years=["FY2025"],
                metrics=[["revenue"], ["share", "percentage", "total"]],
            ),
            sql_case(
                "direct_sql_012_googl_services_revenue",
                "What was Alphabet Google Services revenue in FY2025?",
                entities=[["Alphabet", "GOOGL"], ["Google Services"]],
                years=["FY2025"],
                metrics=[["revenue"]],
            ),
            sql_case(
                "direct_sql_013_aapl_cash",
                "How much cash and equivalents did Apple have in FY2025?",
                entities=[["Apple", "AAPL"]],
                years=["FY2025"],
                metrics=[["cash", "equivalents"]],
            ),
            sql_case(
                "direct_sql_014_msft_intelligent_cloud_yoy",
                "Calculate Intelligent Cloud year-over-year revenue growth in FY2025.",
                entities=[["Microsoft", "MSFT"], ["Intelligent Cloud"]],
                years=["FY2024", "FY2025"],
                metrics=[["revenue"], ["growth"]],
            ),
            sql_case(
                "direct_sql_015_gross_profit_compare",
                "Compare gross profit for Apple and Alphabet in FY2025.",
                entities=[["Apple", "AAPL"], ["Alphabet", "GOOGL"]],
                years=["FY2025"],
                metrics=[["gross profit", "gross margin", "revenue"]],
            ),
        ]
    )

    # 15 direct RAG cases.
    rag_specs = [
        ("direct_rag_001_aapl_services_include", "What does Apple's FY2025 10-K say Services includes?", "AAPL", "FY2025", [["Services"]], [["Services", "include"], ["App Store"], ["cloud"], ["advertising"]]),
        ("direct_rag_002_aapl_services_drivers", "What drove Apple Services net sales growth in FY2025?", "AAPL", "FY2025", [["Services"]], [["Services", "net sales"], ["advertising"], ["App Store"], ["cloud"]]),
        ("direct_rag_003_aapl_services_margin", "What does Apple's FY2025 10-K disclose about Services gross margin?", "AAPL", "FY2025", [["Services"], ["gross margin"]], [["Services", "gross margin"]]),
        ("direct_rag_004_msft_cyber", "What cybersecurity risks did Microsoft discuss in its FY2025 10-K?", "MSFT", "FY2025", [["cybersecurity"]], [["cybersecurity", "risk"]]),
        ("direct_rag_005_msft_cloud_drivers", "What did Microsoft management cite as drivers for Intelligent Cloud performance?", "MSFT", "FY2025", [["Intelligent Cloud"]], [["Intelligent Cloud", "revenue"], ["server", "cloud"]]),
        ("direct_rag_006_googl_ai_risks", "What AI-related risks does Alphabet disclose in FY2025?", "GOOGL", "FY2025", [["AI", "artificial intelligence"]], [["AI", "risk"], ["artificial intelligence"]]),
        ("direct_rag_007_googl_cloud_drivers", "What does Alphabet say drove Google Cloud growth in FY2025?", "GOOGL", "FY2025", [["Google Cloud"]], [["Google Cloud", "revenue"], ["growth"]]),
        ("direct_rag_008_aapl_competition", "What competitive pressures does Apple describe in FY2025?", "AAPL", "FY2025", [["competition"]], [["competition"], ["competitive"]]),
        ("direct_rag_009_msft_dividends", "What does Microsoft's FY2025 10-K disclose about dividends?", "MSFT", "FY2025", [["dividend"]], [["dividend"]]),
        ("direct_rag_010_googl_ads_components", "What are the components of Alphabet advertising revenue discussed in the FY2025 filing?", "GOOGL", "FY2025", [["advertising"]], [["Search"], ["YouTube"], ["Network"], ["advertising"]]),
        ("direct_rag_011_aapl_dma", "What does Apple disclose about Digital Markets Act proceedings in FY2025?", "AAPL", "FY2025", [["Digital Markets Act", "DMA"]], [["DMA"], ["Digital Markets Act"]]),
        ("direct_rag_012_msft_research_dev", "What does Microsoft say about research and development expenses in FY2025?", "MSFT", "FY2025", [["research", "development"]], [["research", "development"], ["R&D"]]),
        ("direct_rag_013_googl_content_costs", "What does Alphabet say about traffic acquisition costs or content acquisition costs?", "GOOGL", "FY2025", [["traffic acquisition", "TAC"], ["content acquisition"]], [["traffic acquisition"], ["content acquisition"]]),
        ("direct_rag_014_aapl_geography_apac", "What does Apple say about Rest of Asia Pacific sales in FY2025?", "AAPL", "FY2025", [["Rest of Asia Pacific"]], [["Rest of Asia Pacific"], ["sales"]]),
        ("direct_rag_015_msft_unearned_revenue", "What does Microsoft disclose about unearned revenue expected to be recognized?", "MSFT", "FY2025", [["unearned revenue"]], [["unearned revenue"], ["recognized"]]),
    ]
    for cid, question, ticker, year, entities, evidence in rag_specs:
        cases.append(rag_case(cid, question, ticker=ticker, fiscal_year=year, entities=entities, evidence_any=evidence))

    # 15 hybrid first-step SQL cases.
    for i, (company, ticker) in enumerate(COMPANIES * 5, 1):
        topic = ["geographic region", "segment", "largest revenue increase", "operating margin", "revenue growth"][i % 5]
        cases.append(
            {
                "id": f"hybrid_sql_first_{i:03d}_{ticker.lower()}",
                "category": "hybrid_sql_first",
                "question": f"Identify the {topic} for {company} in FY2025, then find what the 10-K says about it.",
                "observations": [],
                "expected_action": "tool",
                "expected_calls": [
                    {
                        "tool": "sql",
                        "args": {
                            "entities": [[company, ticker]],
                            "years": ["FY2025"],
                            "metrics": [["revenue", "margin", "growth", "largest", "highest"]],
                        },
                    }
                ],
            }
        )

    # 15 hybrid follow-up RAG cases after SQL found concrete entity.
    followups = [
        ("AAPL", "FY2025", "Rest of Asia Pacific", "sales performance"),
        ("AAPL", "FY2025", "Services", "growth drivers and gross margin"),
        ("AAPL", "FY2025", "iPhone", "sales performance"),
        ("MSFT", "FY2025", "Intelligent Cloud", "performance drivers"),
        ("MSFT", "FY2025", "More Personal Computing", "operating income and drivers"),
        ("MSFT", "FY2025", "Productivity and Business Processes", "revenue drivers"),
        ("GOOGL", "FY2025", "Google Cloud", "growth drivers"),
        ("GOOGL", "FY2025", "Google Services", "advertising performance"),
        ("GOOGL", "FY2025", "Other Bets", "operating performance"),
        ("AAPL", "FY2024", "Services", "gross margin"),
        ("MSFT", "FY2024", "Intelligent Cloud", "performance drivers"),
        ("GOOGL", "FY2024", "Google Cloud", "growth drivers"),
        ("AAPL", "FY2025", "Greater China", "sales performance"),
        ("MSFT", "FY2025", "research and development", "expense explanation"),
        ("GOOGL", "FY2025", "AI", "risk disclosures"),
    ]
    for i, (ticker, year, entity, topic) in enumerate(followups, 1):
        cases.append(
            rag_case(
                f"hybrid_rag_followup_{i:03d}",
                f"Now find what the 10-K says about {entity} {topic}.",
                ticker=ticker,
                fiscal_year=year,
                entities=[[entity]],
                evidence_any=[[entity], topic.split()],
                category="hybrid_followup",
                observations=[{"tool": "sql", "result": f"SQL identified {entity} as the relevant entity for {ticker} {year}."}],
                dependency_terms=[entity],
            )
        )

    # 15 stop cases after sufficient observations.
    for i in range(1, 16):
        if i % 3 == 1:
            obs = [{"tool": "sql", "result": "The requested revenue and growth rate are already computed."}]
            question = "What was the growth rate?"
        elif i % 3 == 2:
            obs = [{"tool": "rag", "result": "The requested filing explanation is already provided with citations."}]
            question = "What does the filing say?"
        else:
            obs = [
                {"tool": "sql", "result": "The numeric comparison is complete."},
                {"tool": "rag", "result": "The narrative filing evidence is complete."},
            ]
            question = "Combine the numeric comparison with filing commentary."
        cases.append(answer_case(f"stop_after_observation_{i:03d}", question, obs))

    # 15 compare/decomposition cases.
    compare_specs = [
        ("MSFT", "GOOGL", "AI risk disclosures", ["AI", "risk"]),
        ("AAPL", "MSFT", "cybersecurity risk disclosures", ["cybersecurity", "risk"]),
        ("AAPL", "GOOGL", "regulatory risk disclosures", ["regulatory", "risk"]),
        ("MSFT", "GOOGL", "cloud growth drivers", ["cloud", "growth"]),
        ("AAPL", "MSFT", "services business disclosures", ["Services", "services"]),
        ("AAPL", "GOOGL", "advertising business disclosures", ["advertising"]),
        ("MSFT", "GOOGL", "R&D expense discussion", ["research", "development"]),
        ("AAPL", "MSFT", "gross margin discussion", ["gross margin"]),
        ("AAPL", "GOOGL", "competition risks", ["competition", "risk"]),
        ("MSFT", "GOOGL", "unearned revenue disclosures", ["unearned revenue"]),
        ("AAPL", "MSFT", "share repurchase disclosures", ["repurchase"]),
        ("AAPL", "GOOGL", "legal proceedings", ["legal", "proceedings"]),
        ("MSFT", "GOOGL", "segment performance drivers", ["segment", "performance"]),
        ("AAPL", "MSFT", "dividend disclosures", ["dividend"]),
        ("AAPL", "GOOGL", "supply chain risks", ["supply", "risk"]),
    ]
    ticker_to_company = {ticker: company for company, ticker in COMPANIES}
    for i, (t1, t2, topic, terms) in enumerate(compare_specs, 1):
        cases.append(
            compare_rag_case(
                f"compare_decompose_{i:03d}",
                f"Compare {ticker_to_company[t1]} and {ticker_to_company[t2]} {topic} in FY2025.",
                [(ticker_to_company[t1], t1), (ticker_to_company[t2], t2)],
                fiscal_year="FY2025",
                topic_terms=terms,
            )
        )

    # 10 fallback/out-of-scope cases.
    fallback_specs = [
        ("Apple", "AAPL", "FY2025", "Services", [["Services", "include"], ["Services", "growth"], ["gross margin"]]),
        ("Microsoft", "MSFT", "FY2025", "AI strategy", [["AI"], ["investment"], ["risk"]]),
        ("Alphabet", "GOOGL", "FY2025", "Google Cloud importance", [["Google Cloud"], ["growth"], ["operating"]]),
        ("Apple", "AAPL", "FY2025", "App Store importance", [["App Store"], ["Services"], ["Digital Content"]]),
        ("Microsoft", "MSFT", "FY2025", "cloud importance", [["cloud"], ["Intelligent Cloud"], ["growth"]]),
    ]
    for i, (company, ticker, year, abstract, evidence) in enumerate(fallback_specs, 1):
        cases.append(
            rag_case(
                f"fallback_abstract_{i:03d}",
                f"What does {company}'s {year} 10-K say about the strategic importance of {abstract}?",
                ticker=ticker,
                fiscal_year=year,
                entities=[[abstract.split()[0]]],
                evidence_any=evidence,
                category="fallback",
                observations=[
                    {
                        "tool": "rag",
                        "result": (
                            f"In {company}'s {year} 10-K, the exact phrase strategic importance "
                            f"of {abstract} was not found."
                        ),
                    }
                ],
            )
        )
    for i, company in enumerate(["Tesla", "Amazon", "NVIDIA", "Meta", "Netflix"], 1):
        cases.append(
            answer_case(
                f"out_of_scope_{i:03d}",
                f"What was {company}'s revenue in FY2025?",
                [],
                category="out_of_scope",
            )
        )

    assert len(cases) == 100, len(cases)
    return cases


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
    lines.append(
        "\nTake exactly the next necessary step. If independent parallel tool calls are needed, issue all of them now. "
        "If a tool is needed, emit the tool call rather than describing it. "
        "If the observations are sufficient, answer without calling tools."
    )
    return "\n".join(lines)


def norm(value: Any) -> str:
    return str(value or "").lower()


def any_alt_present(text: str, alts: list[str]) -> bool:
    return any(norm(alt) in text for alt in alts)


def actual_calls_from_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"tool": block.get("name"), "args": block.get("input") or {}}
        for block in response.get("content", [])
        if block.get("type") == "tool_use"
    ]


def recall_for_groups(text: str, groups: list[list[str]]) -> tuple[int, int]:
    total = len(groups)
    hit = sum(1 for group in groups if any_alt_present(text, group))
    return hit, total


def find_best_actual(expected: dict[str, Any], actual_calls: list[dict[str, Any]], used: set[int]) -> int | None:
    candidates = [idx for idx, call in enumerate(actual_calls) if idx not in used and call["tool"] == expected["tool"]]
    if not candidates:
        return None
    expected_args = expected.get("args", {})
    expected_ticker = expected_args.get("ticker")
    if expected_ticker:
        for idx in candidates:
            if norm(actual_calls[idx]["args"].get("ticker")) == norm(expected_ticker):
                return idx
    return candidates[0]


def score_single_case(case: dict[str, Any], actual_calls: list[dict[str, Any]], answer: str) -> dict[str, Any]:
    expected_calls = case.get("expected_calls") or []
    expected_action = case["expected_action"]
    actual_action = "tool" if actual_calls else "answer"
    used: set[int] = set()
    matched: list[dict[str, Any]] = []

    for expected in expected_calls:
        idx = find_best_actual(expected, actual_calls, used)
        if idx is not None:
            used.add(idx)
            matched.append({"expected": expected, "actual": actual_calls[idx]})

    return {
        "id": case["id"],
        "category": case["category"],
        "expected_action": expected_action,
        "actual_action": actual_action,
        "expected_calls": expected_calls,
        "actual_calls": actual_calls,
        "matched_calls": matched,
        "answer_excerpt": answer[:200],
        "exact_pass": expected_action == actual_action and len(matched) == len(expected_calls) and len(actual_calls) == len(expected_calls),
    }


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    action_correct = sum(1 for r in results if r["expected_action"] == r["actual_action"])
    expected_tool = [r for r in results if r["expected_action"] == "tool"]
    expected_answer = [r for r in results if r["expected_action"] == "answer"]
    actual_tool = [r for r in results if r["actual_action"] == "tool"]
    actual_answer = [r for r in results if r["actual_action"] == "answer"]

    expected_tool_count = sum(len(r["expected_calls"]) for r in results)
    actual_tool_count = sum(len(r["actual_calls"]) for r in results)
    matched_tool_count = sum(len(r["matched_calls"]) for r in results)

    confusion = Counter()
    for r in expected_tool:
        exp_tools = [call["tool"] for call in r["expected_calls"]]
        act_tools = [call["tool"] for call in r["actual_calls"]]
        if exp_tools and act_tools and exp_tools[0] != act_tools[0]:
            confusion[f"expected_{exp_tools[0]}_actual_{act_tools[0]}"] += 1

    sql_entity_hit = sql_entity_total = 0
    sql_year_hit = sql_year_total = 0
    sql_metric_hit = sql_metric_total = 0
    rag_entity_hit = rag_entity_total = 0
    rag_evidence_hit = rag_evidence_total = 0
    rag_ticker_ok = rag_ticker_total = 0
    rag_year_ok = rag_year_total = 0
    rag_single_scope_ok = rag_single_scope_total = 0
    dependency_ok = dependency_total = 0
    abstract_ok = abstract_total = 0

    for r in results:
        for pair in r["matched_calls"]:
            expected = pair["expected"]
            actual = pair["actual"]
            args = expected.get("args", {})
            actual_args = actual.get("args", {})
            question = norm(actual_args.get("question"))
            if expected["tool"] == "sql":
                h, t = recall_for_groups(question, args.get("entities", []))
                sql_entity_hit += h
                sql_entity_total += t
                h, t = recall_for_groups(question, [[year] for year in args.get("years", [])])
                sql_year_hit += h
                sql_year_total += t
                h, t = recall_for_groups(question, args.get("metrics", []))
                sql_metric_hit += h
                sql_metric_total += t
            elif expected["tool"] == "rag":
                rag_ticker_total += 1
                rag_year_total += 1
                rag_single_scope_total += 1
                if norm(actual_args.get("ticker")) == norm(args.get("ticker")):
                    rag_ticker_ok += 1
                if norm(actual_args.get("fiscal_year")) == norm(args.get("fiscal_year")):
                    rag_year_ok += 1
                ticker_value = actual_args.get("ticker")
                fiscal_value = actual_args.get("fiscal_year")
                if isinstance(ticker_value, str) and "," not in ticker_value and isinstance(fiscal_value, str) and "," not in fiscal_value:
                    rag_single_scope_ok += 1
                h, t = recall_for_groups(question, args.get("entities", []))
                rag_entity_hit += h
                rag_entity_total += t
                evidence_groups = args.get("evidence_any", [])
                if evidence_groups:
                    rag_evidence_total += 1
                    if any(all(norm(term) in question for term in group) for group in evidence_groups):
                        rag_evidence_hit += 1
                dependency_terms = args.get("dependency_terms", [])
                if dependency_terms:
                    dependency_total += 1
                    if all(norm(term) in question for term in dependency_terms):
                        dependency_ok += 1
                if r["category"] in {"fallback", "hybrid_followup"} and evidence_groups:
                    abstract_total += 1
                    if any(all(norm(term) in question for term in group) for group in evidence_groups):
                        abstract_ok += 1

    stop_cases = [r for r in results if r["category"] == "stop"]
    hybrid_followups = [r for r in results if r["category"] == "hybrid_followup"]
    decomposition_cases = [r for r in results if r["category"] == "compare_decompose"]
    decomposition_ok = sum(
        1
        for r in decomposition_cases
        if len(r["matched_calls"]) == len(r["expected_calls"]) and len(r["actual_calls"]) == len(r["expected_calls"])
    )
    missing_followup = sum(1 for r in hybrid_followups if r["actual_action"] != "tool")

    def rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    by_category: dict[str, dict[str, Any]] = {}
    for category, rows_iter in defaultdict(list).items():
        _ = category, rows_iter
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        grouped[r["category"]].append(r)
    for category, rows in sorted(grouped.items()):
        by_category[category] = {
            "n": len(rows),
            "exact_pass_rate": rate(sum(1 for r in rows if r["exact_pass"]), len(rows)),
            "action_accuracy": rate(sum(1 for r in rows if r["expected_action"] == r["actual_action"]), len(rows)),
        }

    return {
        "overall": {
            "case_count": n,
            "exact_policy_pass_rate": rate(sum(1 for r in results if r["exact_pass"]), n),
            "mean_binary_action_score": rate(action_correct, n),
        },
        "action": {
            "action_accuracy": rate(action_correct, n),
            "tool_needed_recall": rate(sum(1 for r in expected_tool if r["actual_action"] == "tool"), len(expected_tool)),
            "stop_recall": rate(sum(1 for r in expected_answer if r["actual_action"] == "answer"), len(expected_answer)),
            "stop_precision": rate(sum(1 for r in actual_answer if r["expected_action"] == "answer"), len(actual_answer)),
            "over_call_rate": rate(sum(1 for r in expected_answer if r["actual_action"] == "tool"), len(expected_answer)),
            "under_call_rate": rate(sum(1 for r in expected_tool if r["actual_action"] == "answer"), len(expected_tool)),
        },
        "tool_selection": {
            "expected_tool_calls": expected_tool_count,
            "actual_tool_calls": actual_tool_count,
            "matched_tool_calls": matched_tool_count,
            "tool_set_precision": rate(matched_tool_count, actual_tool_count),
            "tool_set_recall": rate(matched_tool_count, expected_tool_count),
            "tool_set_f1": (
                round(2 * matched_tool_count / (actual_tool_count + expected_tool_count), 4)
                if actual_tool_count + expected_tool_count
                else None
            ),
            "sql_vs_rag_confusion": dict(confusion),
        },
        "args": {
            "sql_entity_recall": rate(sql_entity_hit, sql_entity_total),
            "sql_year_recall": rate(sql_year_hit, sql_year_total),
            "sql_metric_recall": rate(sql_metric_hit, sql_metric_total),
            "rag_ticker_accuracy": rate(rag_ticker_ok, rag_ticker_total),
            "rag_fiscal_year_accuracy": rate(rag_year_ok, rag_year_total),
            "rag_single_scope_accuracy": rate(rag_single_scope_ok, rag_single_scope_total),
            "rag_entity_recall": rate(rag_entity_hit, rag_entity_total),
            "rag_evidence_terms_recall": rate(rag_evidence_hit, rag_evidence_total),
            "abstract_query_expansion_accuracy": rate(abstract_ok, abstract_total),
        },
        "trajectory": {
            "stop_after_sufficient_evidence_accuracy": rate(sum(1 for r in stop_cases if r["actual_action"] == "answer"), len(stop_cases)),
            "dependency_satisfaction_rate": rate(dependency_ok, dependency_total),
            "decomposition_accuracy": rate(decomposition_ok, len(decomposition_cases)),
            "missing_followup_rate": rate(missing_followup, len(hybrid_followups)),
        },
        "by_category": by_category,
    }


def answer_text(response: dict[str, Any]) -> str:
    return "\n".join(block.get("text", "") for block in response.get("content", []) if block.get("type") == "text").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 100-case tool policy eval.")
    parser.add_argument("--output", type=Path, default=AGENT_DIR / "tool_policy_eval_100_results.json")
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_AGENT_MODEL", "claude-haiku-4-5-20251001"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    load_env_file(PROJECT_ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required.")

    cases = build_cases()
    if args.limit:
        cases = cases[: args.limit]

    results = []
    for idx, case in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] {case['id']}", flush=True)
        started = time.perf_counter()
        response = call_compact_policy_model(args.model, format_eval_input(case))
        actual_calls = actual_calls_from_response(response)
        row = score_single_case(case, actual_calls, answer_text(response))
        row["latency_sec"] = round(time.perf_counter() - started, 3)
        row["stop_reason"] = response.get("stop_reason")
        results.append(row)

    metrics = compute_metrics(results)
    payload = {
        "model": args.model,
        "metrics": metrics,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    failures = [r for r in results if not r["exact_pass"]]
    print(f"failures={len(failures)}")
    for r in failures[:20]:
        print(f"FAIL {r['id']} cat={r['category']} expected={r['expected_calls']} actual={r['actual_calls']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
