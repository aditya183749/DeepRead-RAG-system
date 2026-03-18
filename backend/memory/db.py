"""
memory/db.py — SQLite persistence layer for conversation sessions.

TABLES:
  sessions           — one row per conversation (id, title, source_ids, timestamps)
  messages           — every user/assistant turn (session_id FK, role, content)
  session_summaries  — LLM-generated summary of turns that exceed STM_TURNS window

WHY SQLITE:
  Zero infrastructure — ships with Python standard library, no server needed.
  For a local RAG system this is ideal; swap for Postgres in production.
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

from config import SESSIONS_DB_PATH
from logger import get_logger

log = get_logger("memory")


# ─── Schema ───────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New Chat',
    source_ids  TEXT NOT NULL DEFAULT '[]',   -- JSON list of document UUIDs
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_summaries (
    session_id              TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    summary                 TEXT NOT NULL,
    summarized_up_to_msg_id INTEGER NOT NULL,
    updated_at              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


# ─── Connection ───────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    """Get a thread-safe SQLite connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(str(SESSIONS_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db() -> None:
    """Create tables if they don't exist. Called once at backend startup."""
    SESSIONS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_conn() as conn:
        conn.executescript(_SCHEMA)
    log.info("session_db_initialized", path=str(SESSIONS_DB_PATH))


# ─── Session Operations ───────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session(source_ids: List[str], title: str = "New Chat") -> str:
    """Create a new session and return its UUID."""
    session_id = str(uuid.uuid4())
    now = _now()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, source_ids, created_at, updated_at) VALUES (?,?,?,?,?)",
            (session_id, title[:60], json.dumps(source_ids), now, now)
        )
    log.info("session_created", session_id=session_id)
    return session_id


def update_session_title(session_id: str, title: str) -> None:
    """Update session title (typically set to the first user message)."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
            (title[:60], _now(), session_id)
        )


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Return session metadata or None if not found."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["source_ids"] = json.loads(d["source_ids"])
    return d


def list_sessions(limit: int = 50) -> List[Dict[str, Any]]:
    """Return recent sessions ordered by last update, with message count."""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   COUNT(m.id) as message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> bool:
    """Delete session and all its messages (CASCADE). Returns True if found."""
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    deleted = cur.rowcount > 0
    if deleted:
        log.info("session_deleted", session_id=session_id)
    return deleted


# ─── Message Operations ───────────────────────────────────────────────────────

def append_message(session_id: str, role: str, content: str) -> int:
    """Append a message and update session updated_at. Returns new message id."""
    now = _now()
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (session_id, role, content, now)
        )
        conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id))
        return cur.lastrowid


def get_recent_messages(session_id: str, n: int = 5) -> List[Dict[str, Any]]:
    """
    Return the last N user+assistant turns (2*n rows) in chronological order.
    Used by STM builder to inject recent context into the prompt.
    """
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT id, role, content, created_at
            FROM messages
            WHERE session_id=?
            ORDER BY id DESC
            LIMIT ?
        """, (session_id, n * 2)).fetchall()
    # Reverse so they're in chronological order
    return [dict(r) for r in reversed(rows)]


def get_all_messages(session_id: str) -> List[Dict[str, Any]]:
    """Return full message history for a session (used by history sidebar)."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, role, content, created_at FROM messages WHERE session_id=? ORDER BY id ASC",
            (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_message_count(session_id: str) -> int:
    """Return total message count for a session."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)
        ).fetchone()
    return row[0]


# ─── Summary Operations ───────────────────────────────────────────────────────

def upsert_summary(session_id: str, summary: str, up_to_msg_id: int) -> None:
    """Save or update the LLM summary for older turns in this session."""
    now = _now()
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO session_summaries (session_id, summary, summarized_up_to_msg_id, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                summary=excluded.summary,
                summarized_up_to_msg_id=excluded.summarized_up_to_msg_id,
                updated_at=excluded.updated_at
        """, (session_id, summary, up_to_msg_id, now))


def get_summary(session_id: str) -> Optional[Dict[str, Any]]:
    """Return stored summary for a session, or None if not yet summarized."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM session_summaries WHERE session_id=?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def get_messages_after(session_id: str, after_msg_id: int) -> List[Dict[str, Any]]:
    """Return messages after a given message id (used with summary to build STM)."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, role, content FROM messages WHERE session_id=? AND id > ? ORDER BY id ASC",
            (session_id, after_msg_id)
        ).fetchall()
    return [dict(r) for r in rows]
