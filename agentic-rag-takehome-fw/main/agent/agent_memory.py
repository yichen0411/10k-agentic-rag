"""Agent memory: storage, compression, retrieval, cleanup."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MEMORY_DB = REPO_ROOT / "data" / "agent_memory.db"

# Short-term sliding window: keep this many turn pairs verbatim; older folds to summary
# TODO: raise for production (was 6 pairs / 12 messages)
SHORT_TERM_WINDOW_TURNS = 2
PRUNE_KEEP_MESSAGES = SHORT_TERM_WINDOW_TURNS * 2
RECENT_MESSAGE_LIMIT = PRUNE_KEEP_MESSAGES

# Long-term limits
MAX_SEMANTIC_NOTES = 50
MAX_TOOL_ARTIFACTS = 24

# Retrieval: keyword overlap threshold (filters noise)
RETRIEVAL_MIN_SCORE = 0.06
RETRIEVAL_MAX_ITEMS = 8

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_WORD_RE = re.compile(r"[a-z0-9]{2,}", re.I)

_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    MEMORY_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(MEMORY_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _LOCK:
        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    file_id TEXT,
                    summary TEXT NOT NULL DEFAULT '',
                    last_scratchpad TEXT NOT NULL DEFAULT '',
                    last_prompt_snapshot TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);
                CREATE TABLE IF NOT EXISTS memory_facts (
                    session_id TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, fact_key)
                );
                CREATE TABLE IF NOT EXISTS tool_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    input_preview TEXT NOT NULL DEFAULT '',
                    output_preview TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tool_artifacts_session
                    ON tool_artifacts(session_id, id);
                CREATE TABLE IF NOT EXISTS memory_semantic (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 1.0,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_semantic_session
                    ON memory_semantic(session_id, id);
                """
            )
            conn.commit()
            for ddl in (
                "ALTER TABLE sessions ADD COLUMN last_scratchpad TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE sessions ADD COLUMN last_prompt_snapshot TEXT NOT NULL DEFAULT ''",
            ):
                try:
                    conn.execute(ddl)
                    conn.commit()
                except sqlite3.OperationalError:
                    pass
        finally:
            conn.close()


_init_db()


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text or "")}


def _score_text(query: str, text: str, *, age_days: float = 0.0) -> float:
    q = _tokenize(query)
    if not q:
        return 0.0
    t = _tokenize(text)
    if not t:
        return 0.0
    overlap = len(q & t) / len(q)
    decay = max(0.5, 1.0 - age_days * 0.02)
    return overlap * decay


def create_session() -> str:
    session_id = str(uuid.uuid4())
    now = _utc_now()
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO sessions (session_id, file_id, summary, created_at, updated_at) VALUES (?, ?, '', ?, ?)",
                (session_id, None, now, now),
            )
            conn.commit()
        finally:
            conn.close()
    return session_id


def get_or_create(session_id: Optional[str]) -> str:
    if not session_id:
        return create_session()
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row:
                return session_id
        finally:
            conn.close()
    return create_session()


def _upsert_fact(session_id: str, key: str, value: str) -> None:
    value = value.strip()
    if not value:
        return
    now = _utc_now()
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO memory_facts (session_id, fact_key, fact_value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, fact_key) DO UPDATE SET
                    fact_value = excluded.fact_value,
                    updated_at = excluded.updated_at
                """,
                (session_id, key, value, now),
            )
            conn.commit()
        finally:
            conn.close()


_PREFERENCE_KEYS = frozenset(
    {"user_email", "response_style", "output_language", "user_preference", "preferred_units"}
)

# (fact_key, stored_value, trigger substrings)
_PREFERENCE_RULES: list[tuple[str, str, tuple[str, ...]]] = [
    ("response_style", "shorter", ("shorter", "short answer", "brief", "concise", "keep it short")),
    ("response_style", "bullets", ("bullet", "bullet points", "in bullets")),
    ("response_style", "detailed", ("more detail", "elaborate", "expand", "longer answer")),
    ("output_language", "zh", ("in chinese", "用中文", "中文回答")),
    ("output_language", "en", ("in english", "英文回答")),
    ("preferred_units", "billions", ("in billions", "$b", "billion")),
    ("preferred_units", "millions", ("in millions", "$m", "million")),
]


def _split_preferences(facts: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    prefs: dict[str, str] = {}
    other: dict[str, str] = {}
    for key, value in facts.items():
        if key in _PREFERENCE_KEYS or key.startswith("pref_"):
            prefs[key] = value
        else:
            other[key] = value
    return prefs, other


def _extract_facts_from_turn(session_id: str, user: str, _assistant: str) -> None:
    """Episodic memory: email + explicit user preferences (exact key lookup)."""
    text = user or ""
    for email in _EMAIL_RE.findall(text):
        _upsert_fact(session_id, "user_email", email)
    lower = text.lower()
    for marker in ("my email is", "email is", "send to"):
        if marker in lower:
            for email in _EMAIL_RE.findall(text):
                _upsert_fact(session_id, "user_email", email)

    for fact_key, value, triggers in _PREFERENCE_RULES:
        if any(t in lower for t in triggers):
            _upsert_fact(session_id, fact_key, value)

    if any(w in lower for w in ("i prefer", "always use", "from now on", "remember that i")):
        snippet = text.strip()[:320]
        if snippet:
            _upsert_fact(session_id, "user_preference", snippet)


def _list_messages(session_id: str) -> list[dict[str, str]]:
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            return [{"role": row["role"], "content": row["content"]} for row in rows]
        finally:
            conn.close()


def _list_tool_artifacts(session_id: str, limit: int = MAX_TOOL_ARTIFACTS) -> list[dict[str, Any]]:
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT tool, input_preview, output_preview, created_at
                FROM tool_artifacts WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            items = [
                {
                    "tool": row["tool"],
                    "input_preview": row["input_preview"],
                    "output_preview": row["output_preview"],
                    "created_at": row["created_at"],
                }
                for row in reversed(rows)
            ]
            return items
        finally:
            conn.close()


def _list_semantic_notes(session_id: str) -> list[dict[str, Any]]:
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT id, content, importance, access_count, created_at, last_accessed
                FROM memory_semantic WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "importance": row["importance"],
                    "access_count": row["access_count"],
                    "created_at": row["created_at"],
                    "last_accessed": row["last_accessed"],
                }
                for row in rows
            ]
        finally:
            conn.close()


def _append_semantic_note(session_id: str, user: str, assistant: str) -> None:
    user = (user or "").strip()[:220]
    assistant = (assistant or "").strip()[:480]
    if not assistant:
        return
    content = f"Q: {user}\nA: {assistant}" if user else assistant
    now = _utc_now()
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO memory_semantic
                    (session_id, content, importance, access_count, created_at, last_accessed)
                VALUES (?, ?, 1.0, 0, ?, ?)
                """,
                (session_id, content, now, now),
            )
            conn.commit()
        finally:
            conn.close()
    _prune_semantic_notes(session_id)


def _prune_semantic_notes(session_id: str) -> None:
    with _LOCK:
        conn = _connect()
        try:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM memory_semantic WHERE session_id = ?",
                (session_id,),
            ).fetchone()["c"]
            if count <= MAX_SEMANTIC_NOTES:
                return
            excess = count - MAX_SEMANTIC_NOTES
            conn.execute(
                """
                DELETE FROM memory_semantic
                WHERE id IN (
                    SELECT id FROM memory_semantic
                    WHERE session_id = ?
                    ORDER BY id ASC LIMIT ?
                )
                """,
                (session_id, excess),
            )
            conn.commit()
        finally:
            conn.close()


def record_tool_artifacts(session_id: str, steps: Optional[list[dict[str, Any]]]) -> None:
    if not steps:
        return
    now = _utc_now()
    with _LOCK:
        conn = _connect()
        try:
            for step in steps:
                tool = str(step.get("tool") or "unknown")
                inp = (step.get("input") or "")[:500]
                out_obj = step.get("output") or {}
                try:
                    out = json.dumps(out_obj, ensure_ascii=False)[:900]
                except (TypeError, ValueError):
                    out = str(out_obj)[:900]
                conn.execute(
                    """
                    INSERT INTO tool_artifacts
                        (session_id, tool, input_preview, output_preview, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, tool, inp, out, now),
                )
            conn.execute(
                """
                DELETE FROM tool_artifacts
                WHERE session_id = ? AND id NOT IN (
                    SELECT id FROM tool_artifacts
                    WHERE session_id = ?
                    ORDER BY id DESC LIMIT ?
                )
                """,
                (session_id, session_id, MAX_TOOL_ARTIFACTS),
            )
            conn.commit()
        finally:
            conn.close()


def _append_summary(session_id: str, chunk: str) -> None:
    chunk = chunk.strip()
    if not chunk:
        return
    now = _utc_now()
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT summary FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            prev = (row["summary"] if row else "") or ""
            merged = (prev + "\n" + chunk).strip() if prev else chunk
            if len(merged) > 6000:
                merged = merged[-6000:]
            conn.execute(
                "UPDATE sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
                (merged, now, session_id),
            )
            conn.commit()
        finally:
            conn.close()


def _prune_old_messages(session_id: str) -> None:
    messages = _list_messages(session_id)
    if len(messages) <= PRUNE_KEEP_MESSAGES:
        return
    to_fold = messages[: len(messages) - PRUNE_KEEP_MESSAGES]
    lines = []
    for msg in to_fold:
        role = msg["role"]
        text = (msg["content"] or "").replace("\n", " ")[:400]
        lines.append(f"- {role}: {text}")
    _append_summary(session_id, "\n".join(lines))
    keep = messages[-PRUNE_KEEP_MESSAGES:]
    now = _utc_now()
    with _LOCK:
        conn = _connect()
        try:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            for msg in keep:
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (session_id, msg["role"], msg["content"], now),
                )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            conn.commit()
        finally:
            conn.close()


def save_last_prompt_snapshot(session_id: str, injection: Optional[dict[str, Any]]) -> None:
    """Freeze memory_context + chat_history (pre-turn) + scratchpad for the last completed run."""
    blob = ""
    if injection:
        record = {**injection, "saved_at": _utc_now()}
        try:
            blob = json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            blob = ""
    scratchpad = (injection or {}).get("scratchpad") if injection else None
    scratch_blob = ""
    if scratchpad:
        try:
            scratch_blob = json.dumps(scratchpad, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            scratch_blob = ""
    now = _utc_now()
    with _LOCK:
        conn = _connect()
        try:
            updated = conn.execute(
                """
                UPDATE sessions
                SET last_prompt_snapshot = ?, last_scratchpad = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (blob, scratch_blob, now, session_id),
            ).rowcount
            if not updated and blob:
                conn.execute(
                    """
                    INSERT INTO sessions
                        (session_id, file_id, summary, last_scratchpad, last_prompt_snapshot,
                         created_at, updated_at)
                    VALUES (?, NULL, '', ?, ?, ?, ?)
                    """,
                    (session_id, scratch_blob, blob, now, now),
                )
            conn.commit()
        finally:
            conn.close()


def _load_last_prompt_snapshot(session_id: str) -> Optional[dict[str, Any]]:
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT last_prompt_snapshot, last_scratchpad FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        finally:
            conn.close()
    if not row:
        return None
    raw = ""
    try:
        raw = row["last_prompt_snapshot"] or ""
    except (KeyError, IndexError):
        raw = ""
    if str(raw).strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    # Legacy: only scratchpad column populated
    legacy = ""
    try:
        legacy = row["last_scratchpad"] or ""
    except (KeyError, IndexError):
        legacy = ""
    if not str(legacy).strip():
        return None
    try:
        sp = json.loads(legacy)
        if isinstance(sp, list):
            return {"scratchpad": sp, "memory_context": "", "chat_history": [], "query": ""}
    except json.JSONDecodeError:
        pass
    return None


def append_turn(
    session_id: str,
    user: str,
    assistant: str,
    *,
    file_id: Optional[str] = None,
    tool_steps: Optional[list[dict[str, Any]]] = None,
    prompt_injection: Optional[dict[str, Any]] = None,
) -> None:
    user = (user or "").strip()
    assistant = (assistant or "").strip()
    now = _utc_now()
    with _LOCK:
        conn = _connect()
        try:
            if conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone() is None:
                conn.execute(
                    "INSERT INTO sessions (session_id, file_id, summary, created_at, updated_at) VALUES (?, ?, '', ?, ?)",
                    (session_id, file_id, now, now),
                )
            elif file_id:
                conn.execute(
                    "UPDATE sessions SET file_id = ?, updated_at = ? WHERE session_id = ?",
                    (file_id, now, session_id),
                )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
                (session_id, user, now),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'assistant', ?, ?)",
                (session_id, assistant, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            conn.commit()
        finally:
            conn.close()
    _extract_facts_from_turn(session_id, user, assistant)
    record_tool_artifacts(session_id, tool_steps)
    if prompt_injection:
        save_last_prompt_snapshot(session_id, prompt_injection)
    _append_semantic_note(session_id, user, assistant)
    _prune_old_messages(session_id)


def get_history(session_id: str) -> list[dict[str, str]]:
    messages = _list_messages(session_id)
    if len(messages) > RECENT_MESSAGE_LIMIT:
        return messages[-RECENT_MESSAGE_LIMIT:]
    return messages


def get_facts(session_id: str) -> dict[str, str]:
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT fact_key, fact_value FROM memory_facts WHERE session_id = ? ORDER BY fact_key",
                (session_id,),
            ).fetchall()
            return {row["fact_key"]: row["fact_value"] for row in rows}
        finally:
            conn.close()


def get_summary(session_id: str) -> str:
    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT summary FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return (row["summary"] if row else "") or ""
        finally:
            conn.close()


def _bump_semantic_access(note_ids: list[int]) -> None:
    if not note_ids:
        return
    now = _utc_now()
    with _LOCK:
        conn = _connect()
        try:
            for nid in note_ids:
                conn.execute(
                    """
                    UPDATE memory_semantic
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE id = ?
                    """,
                    (now, nid),
                )
            conn.commit()
        finally:
            conn.close()


def retrieve_memory_items(session_id: str, query: str) -> list[dict[str, Any]]:
    """Score semantic/summary chunks; episodic facts are handled separately (always inject)."""
    items: list[dict[str, Any]] = []
    summary = get_summary(session_id).strip()
    if summary:
        for line in summary.split("\n"):
            line = line.strip()
            if not line:
                continue
            items.append(
                {
                    "type": "compressed_summary",
                    "content": line,
                    "score": round(_score_text(query, line), 4),
                }
            )
    for note in _list_semantic_notes(session_id):
        score = _score_text(query, note["content"])
        items.append(
            {
                "type": "semantic_note",
                "id": note["id"],
                "content": note["content"],
                "score": round(score, 4),
                "access_count": note["access_count"],
            }
        )
    items.sort(key=lambda x: (-x["score"], x.get("type", "")))
    return items


def build_memory_context(session_id: str, query: str = "") -> str:
    parts: list[str] = []
    facts = get_facts(session_id)
    if facts:
        lines = [f"- {k}: {v}" for k, v in facts.items()]
        parts.append(
            "Episodic memory (exact key lookup — always apply when relevant):\n"
            + "\n".join(lines)
        )

    q = (query or "").strip()
    if q:
        ranked = retrieve_memory_items(session_id, q)
        selected = [i for i in ranked if i["score"] >= RETRIEVAL_MIN_SCORE][:RETRIEVAL_MAX_ITEMS]
        note_ids = [int(i["id"]) for i in selected if i.get("type") == "semantic_note" and i.get("id")]
        _bump_semantic_access(note_ids)
        if selected:
            lines = [f"- [{i['score']}] {i['content'][:500]}" for i in selected]
            parts.append(
                "Retrieved semantic memory (keyword relevance — use only if related to the current question):\n"
                + "\n".join(lines)
            )
    else:
        summary = get_summary(session_id).strip()
        if summary:
            parts.append("Compressed earlier conversation:\n" + summary[-2500:])

    if not parts:
        return ""
    parts.append(
        "If any retrieved memory above is irrelevant to the current task, ignore it."
    )
    return "\n\n".join(parts)


def build_agent_memory(session_id: str, query: str = "") -> dict[str, Any]:
    return {
        "session_id": session_id,
        "chat_history": get_history(session_id),
        "memory_context": build_memory_context(session_id, query=query),
    }


def clear_session(session_id: str) -> None:
    with _LOCK:
        conn = _connect()
        try:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM memory_facts WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM tool_artifacts WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM memory_semantic WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()


def session_info(session_id: str) -> dict[str, Any]:
    messages = _list_messages(session_id)
    return {
        "session_id": session_id,
        "turns": len(messages) // 2,
        "messages": len(messages),
        "has_summary": bool(get_summary(session_id).strip()),
        "facts": get_facts(session_id),
        "semantic_notes": len(_list_semantic_notes(session_id)),
        "tool_artifacts": len(_list_tool_artifacts(session_id)),
    }


def memory_snapshot(session_id: Optional[str], query: str = "") -> dict[str, Any]:
    empty: dict[str, Any] = {
        "active": False,
        "session_id": session_id,
        "memory_context": "",
        "agent_scratchpad": [],
        "chat_history": [],
        "chat_history_stats": {"turn_count": 0, "message_count": 0, "window_turns": PRUNE_KEEP_MESSAGES // 2},
        "user_preferences": {},
        "short_term_stats": {"turn_count": 0, "message_count": 0},
        "long_term": {
            "strategies": {"compression": {}, "retrieval": {}},
            "episodic": {"preferences": {}, "other_facts": {}, "facts": {}},
            "semantic": {"notes": [], "summary": ""},
        },
        "compression": _compression_policy(),
        "retrieval": {"query": query, "items": [], "injected_context": ""},
    }
    if not session_id:
        empty["reason"] = "no_session"
        return empty

    with _LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                """
                SELECT session_id, file_id, summary, created_at, updated_at
                FROM sessions WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        finally:
            conn.close()

    if not row:
        empty["reason"] = "not_found"
        return empty

    messages = _list_messages(session_id)
    summary = get_summary(session_id)
    facts = get_facts(session_id)
    preferences, other_facts = _split_preferences(facts)
    notes = _list_semantic_notes(session_id)
    ranked = retrieve_memory_items(session_id, query) if query.strip() else []
    preview_context = build_memory_context(session_id, query=query)
    comp = _compression_policy()
    selected = [i for i in ranked if i["score"] >= RETRIEVAL_MIN_SCORE]

    last_injection = _load_last_prompt_snapshot(session_id)
    if last_injection:
        memory_context = (last_injection.get("memory_context") or "").strip()
        scratchpad = last_injection.get("scratchpad") or []
        history = last_injection.get("chat_history") or []
        injection_query = (last_injection.get("query") or "").strip()
    else:
        memory_context = ""
        scratchpad = []
        history = []
        injection_query = ""

    turn_count = len(history) // 2
    live_history = get_history(session_id)
    live_turn_count = len(live_history) // 2

    return {
        "active": True,
        "session_id": session_id,
        "file_id": row["file_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "last_prompt_injection": last_injection,
        "injection_query": injection_query,
        "memory_context": memory_context,
        "agent_scratchpad": scratchpad,
        "chat_history": history,
        "live_chat_history": live_history,
        "live_chat_history_stats": {
            "turn_count": live_turn_count,
            "message_count": len(live_history),
            "window_turns": PRUNE_KEEP_MESSAGES // 2,
        },
        "chat_history_stats": {
            "turn_count": turn_count,
            "message_count": len(history),
            "window_turns": PRUNE_KEEP_MESSAGES // 2,
        },
        "preview_memory_context": preview_context,
        "preview_query": (query or "").strip(),
        "short_term_stats": {
            "turn_count": len(messages) // 2,
            "message_count": len(messages),
            "window_turns": PRUNE_KEEP_MESSAGES // 2,
        },
        "user_preferences": preferences,
        "long_term": {
            "strategies": {
                "compression": comp["long_term"],
                "retrieval": {
                    "episodic": "exact key lookup — preferences & facts always injected",
                    "semantic": f"keyword overlap, min_score={RETRIEVAL_MIN_SCORE}, max_items={RETRIEVAL_MAX_ITEMS}",
                    "noise_filter": "prompt: ignore retrieved lines irrelevant to current question",
                },
                "cleanup": _architecture_blurb()["cleanup"],
            },
            "episodic": {
                "store": "sqlite memory_facts",
                "preferences": preferences,
                "other_facts": other_facts,
                "facts": facts,
            },
            "semantic": {
                "store": "sqlite memory_semantic + sessions.summary",
                "summary": summary,
                "has_summary": bool(summary.strip()),
                "notes": notes,
                "note_count": len(notes),
            },
        },
        "compression": comp,
        "retrieval": {
            "query": query,
            "min_score": RETRIEVAL_MIN_SCORE,
            "max_items": RETRIEVAL_MAX_ITEMS,
            "items": ranked,
            "selected": selected,
            "selected_count": len(selected),
            "policy": "episodic always; semantic/summary only above threshold; ignore if irrelevant",
            "injected_context": preview_context,
        },
    }


def _architecture_blurb() -> dict[str, str]:
    return {
        "storage": "short-term: messages + tool_artifacts · episodic: facts · semantic: notes + summary",
        "compression": "sliding window → summary fold; semantic cap; tool artifact cap",
        "retrieval": "keyword relevance + episodic KV; prompt says ignore irrelevant",
        "cleanup": "New chat clears session; prune oldest semantic notes / tool traces",
    }


def _compression_policy() -> dict[str, Any]:
    return {
        "short_term": {
            "sliding_window": f"keep last {PRUNE_KEEP_MESSAGES // 2} turn pairs verbatim",
            "llm_summarization": "planned — currently text fold into summary",
            "importance_scoring": "episodic facts preserved across folds",
        },
        "long_term": {
            "time_window": "semantic notes capped per session",
            "access_frequency": "access_count on retrieved semantic notes",
            "merge_similar": "summary append for folded turns",
            "importance_decay": "score × age decay in retrieval",
        },
    }
