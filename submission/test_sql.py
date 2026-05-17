"""
Interactive / batch SQL Q&A using the multi-step sql_tool pipeline.

Usage:
  python test_sql.py                    # interactive REPL
  python test_sql.py "some question"    # answer one question
  python test_sql.py --eval             # run all SQL-only dev questions
"""

import sys
import json
import re
import logging
sys.path.insert(0, ".")

from tools.sql_tool import answer_sql_question

DEV_QUESTIONS_PATH = "../questions/dev_questions_with_answers.json"


def ask(question: str) -> str:
    return answer_sql_question(question)


# ---------------------------------------------------------------------------
# Evaluation helpers (mirrors eval_sql.py logic)
# ---------------------------------------------------------------------------


def _all_numbers(text: str) -> list:
    return [
        float(n.replace(",", ""))
        for n in re.findall(r"[-+]?[\d,]+(?:\.\d+)?", text)
        if re.search(r"\d", n)
    ]


def _fuzzy_numeric_match(pred: str, gold: float, tol: float = 0.02) -> bool:
    if gold == 0:
        return False
    for n in _all_numbers(pred):
        for candidate in (n, n * 1e9, n / 1e9):
            if abs(candidate - gold) / abs(gold) <= tol:
                return True
    return False


def _entity_match(pred: str, gold_text: str) -> bool:
    entities = re.findall(
        r"\b(Apple|AAPL|Microsoft|MSFT|Alphabet|Google|GOOGL)\b", gold_text, re.I
    )
    if not entities:
        return False
    pred_lower = pred.lower()
    return any(e.lower() in pred_lower for e in entities)


def evaluate(pred: str, q: dict):
    method = q["evaluation"]
    gold_numeric = q.get("gold_answer_numeric")
    gold_text = q["gold_answer"]

    if method == "fuzzy_numeric" and gold_numeric is not None:
        return "fuzzy_numeric", _fuzzy_numeric_match(pred, gold_numeric)

    if method == "exact_match_entity":
        entity_ok = _entity_match(pred, gold_text)
        num_ok = True
        if gold_numeric is not None:
            num_ok = _fuzzy_numeric_match(pred, gold_numeric)
        return "entity+numeric", entity_ok and num_ok

    return "llm_judge (manual)", None


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_eval():
    with open(DEV_QUESTIONS_PATH) as f:
        all_questions = json.load(f)
    sql_only = [q for q in all_questions if q["required_modalities"] == ["sql"]]
    print(f"Running {len(sql_only)} SQL-only question(s)...\n" + "=" * 70)

    results = []
    for q in sql_only:
        print(f"\n[{q['id']}] {q['question']}")
        pred = ask(q["question"])
        method, passed = evaluate(pred, q)
        verdict = {True: "PASS", False: "FAIL", None: "MANUAL"}[passed]
        print(f"  PRED : {pred}")
        print(f"  GOLD : {q['gold_answer']}")
        print(f"  EVAL : [{method}] → {verdict}")
        results.append({"id": q["id"], "passed": passed, "verdict": verdict})

    print("\n" + "=" * 70)
    auto = [r for r in results if r["passed"] is not None]
    if auto:
        n_pass = sum(1 for r in auto if r["passed"])
        print(f"Auto-scored: {n_pass}/{len(auto)} passed")
    manual = [r for r in results if r["passed"] is None]
    if manual:
        print(f"Manual review needed: {[r['id'] for r in manual]}")


def interactive():
    print("SQL Q&A (multi-step pipeline) — type 'quit' to exit\n")
    while True:
        try:
            q = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        print()
        print(ask(q))
        print()


def main():
    logging.basicConfig(level=logging.WARNING)
    args = sys.argv[1:]

    if args and args[0] == "--eval":
        run_eval()
    elif args:
        question = " ".join(args)
        print(f"Q: {question}\n")
        print(ask(question))
    else:
        interactive()


if __name__ == "__main__":
    main()