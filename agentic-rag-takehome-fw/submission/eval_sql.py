"""Evaluate SQL-only questions against gold answers."""
import sys
import json
import re
sys.path.insert(0, ".")

from openai import OpenAI
from config import FIREWORKS_API_KEY, FIREWORKS_BASE_URL, CHAT_MODEL
from tools.sql_tool import run_sql, TOOL_SCHEMA

QUESTIONS_PATH = "../questions/dev_questions_with_answers.json"

client = OpenAI(api_key=FIREWORKS_API_KEY, base_url=FIREWORKS_BASE_URL)


def ask(question: str):
    """Returns (final_answer, list_of_sqls_used)."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analyst assistant. "
                "Use the query_sql tool to answer questions about AAPL, MSFT, and GOOGL financials. "
                "Always call the tool; never guess numbers. "
                "You may call the tool multiple times if the question requires data from more than one table."
            ),
        },
        {"role": "user", "content": question},
    ]

    sqls_used = []
    for _ in range(6):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=[TOOL_SCHEMA],
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content or "", sqls_used

        messages.append(msg)
        for tool_call in msg.tool_calls:
            sql = json.loads(tool_call.function.arguments)["sql"]
            sqls_used.append(sql)
            result = run_sql(sql)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "", sqls_used


def all_numbers(text: str):
    """Return all numeric values found in text."""
    return [float(n.replace(",", "")) for n in re.findall(r"[-+]?[\d,]+(?:\.\d+)?", text)
            if re.search(r"\d", n)]


def fuzzy_numeric_match(pred: str, gold_numeric: float, tol: float = 0.02) -> bool:
    """Pass if ANY number in pred is within tol (2%) of gold, with unit scaling."""
    if gold_numeric == 0:
        return False
    for n in all_numbers(pred):
        for candidate in (n, n * 1e9, n / 1e9):
            if abs(candidate - gold_numeric) / abs(gold_numeric) <= tol:
                return True
    return False


def entity_match(pred: str, gold: str) -> bool:
    """Check if the key entity (company name or ticker) from gold appears in pred."""
    entities = re.findall(r"\b(Apple|AAPL|Microsoft|MSFT|Alphabet|Google|GOOGL)\b", gold, re.I)
    if not entities:
        return False
    pred_lower = pred.lower()
    return any(e.lower() in pred_lower for e in entities)


def evaluate(pred: str, question: dict):
    """Returns (verdict_label, passed_or_None)."""
    method = question["evaluation"]
    gold_numeric = question.get("gold_answer_numeric")
    gold_text = question["gold_answer"]

    if method == "fuzzy_numeric" and gold_numeric is not None:
        passed = fuzzy_numeric_match(pred, gold_numeric)
        return "fuzzy_numeric", passed

    if method == "exact_match_entity":
        passed = entity_match(pred, gold_text)
        num_ok = True
        if gold_numeric is not None:
            num_ok = fuzzy_numeric_match(pred, gold_numeric)
        return "entity+numeric", passed and num_ok

    return "llm_judge (manual)", None  # no automated verdict


def main():
    with open(QUESTIONS_PATH) as f:
        all_questions = json.load(f)

    sql_only = [q for q in all_questions if q["required_modalities"] == ["sql"]]
    print(f"Running {len(sql_only)} SQL-only question(s)...\n")
    print("=" * 70)

    results = []
    for q in sql_only:
        print(f"[{q['id']}] {q['question']}")
        pred, sqls_used = ask(q["question"])
        method, passed = evaluate(pred, q)

        for i, sql in enumerate(sqls_used):
            print(f"  SQL{i+1} : {sql}")
        print(f"  PRED : {pred}")
        print(f"  GOLD : {q['gold_answer']}")
        verdict = {
            True: "PASS",
            False: "FAIL",
            None: "MANUAL",
        }[passed]
        print(f"  EVAL : [{method}] → {verdict}")
        print()

        results.append({
            "id": q["id"],
            "question": q["question"],
            "predicted": pred,
            "gold": q["gold_answer"],
            "sql": sql,
            "evaluation": method,
            "passed": passed,
        })

    print("=" * 70)
    auto = [r for r in results if r["passed"] is not None]
    if auto:
        n_pass = sum(1 for r in auto if r["passed"])
        print(f"Auto-scored: {n_pass}/{len(auto)} passed")
    manual = [r for r in results if r["passed"] is None]
    if manual:
        print(f"Manual review needed: {len(manual)} question(s) — {[r['id'] for r in manual]}")


if __name__ == "__main__":
    main()
