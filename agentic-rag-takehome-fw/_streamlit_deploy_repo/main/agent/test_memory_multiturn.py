#!/usr/bin/env python3
"""Simulate multi-turn session memory; print snapshot checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from agent_memory import (  # noqa: E402
    append_turn,
    build_agent_memory,
    build_memory_context,
    clear_session,
    create_session,
    memory_snapshot,
)


def main() -> int:
    sid = create_session()
    turns = [
        ("What was Microsoft revenue growth FY2024 to FY2025?", "Revenue grew about 15% year over year."),
        ("Make it shorter please", "Revenue grew ~15% YoY."),
        ("My email is reviewer@example.com", "Noted your email."),
        ("Always use bullet points from now on", "Understood — bullets going forward."),
        ("Turn 4 filler about cloud segment drivers", "Cloud was a key driver."),
        ("Turn 5 filler about gaming segment", "Gaming grew modestly."),
        ("Turn 6 filler about search ads", "Search ads improved."),
        ("Turn 7 filler about capex", "Capex increased."),
        ("Turn 8 filler about dividends", "Dividends were stable."),
        ("What email should you use?", "Use reviewer@example.com."),
        ("shorter summary of cloud", "• Cloud drove growth."),
    ]
    for user, assistant in turns:
        append_turn(
            sid,
            user,
            assistant,
            tool_steps=[{"tool": "sql", "input": user[:80], "output": {"ok": True, "rows": []}}],
        )

    snap = memory_snapshot(sid, query="shorter cloud email")
    prefs = snap.get("user_preferences") or {}
    mem = build_agent_memory(sid, query="shorter cloud email")
    ctx = build_memory_context(sid, query="shorter cloud email")

    checks = {
        "user_email": prefs.get("user_email") == "reviewer@example.com",
        "response_style_shorter": prefs.get("response_style") == "shorter",
        "user_preference_stored": bool(prefs.get("user_preference")),
        "semantic_notes": (snap["long_term"]["semantic"]["note_count"] or 0) >= len(turns),
        "has_compressed_summary": snap["long_term"]["semantic"]["has_summary"],
        "retrieval_selected": (snap["retrieval"]["selected_count"] or 0) >= 0,
        "context_has_email": "reviewer@example.com" in ctx,
        "context_has_episodic": "Episodic memory" in ctx,
        "chat_history_bounded": len(mem["chat_history"]) <= 12,
        "strategies_present": bool(snap["long_term"]["strategies"].get("compression")),
    }

    print("session_id:", sid)
    print("preferences:", json.dumps(prefs, indent=2))
    print("short_term_stats:", snap.get("short_term_stats"))
    print("retrieval selected:", snap["retrieval"]["selected_count"])
    print("checks:")
    failed = []
    for name, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            failed.append(name)

    clear_session(sid)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
