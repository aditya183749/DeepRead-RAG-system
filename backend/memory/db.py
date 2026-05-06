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

-- ── Document Lifecycle Management ─────────────────────────────────────────
-- Stable identity across all versions of a document (keyed by normalized filename)
CREATE TABLE IF NOT EXISTS document_registry (
    doc_key     TEXT PRIMARY KEY,   -- normalized filename hash
    name        TEXT NOT NULL,      -- human display name (original filename)
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- One row per uploaded version of a document
CREATE TABLE IF NOT EXISTS document_versions (
    source_id   TEXT PRIMARY KEY,                              -- UUID used in FAISS/BM25
    doc_key     TEXT NOT NULL REFERENCES document_registry(doc_key),
    filename    TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    status      TEXT NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active', 'archived', 'expired')),
    chunk_count INTEGER NOT NULL DEFAULT 0,
    file_size   INTEGER NOT NULL DEFAULT 0,
    uploaded_at TEXT NOT NULL,
    archived_at TEXT,
    notes       TEXT
);

CREATE INDEX IF NOT EXISTS idx_versions_dockey ON document_versions(doc_key, version DESC);
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


# ─── Document Lifecycle Management ────────────────────────────────────────────

def _normalize_doc_key(filename: str) -> str:
    """Stable doc_key from filename — lowercase stem, special chars → underscore."""
    import re
    name = Path(filename).stem.lower()
    return re.sub(r'[^a-z0-9]+', '_', name).strip('_')


def register_document(
    source_id: str,
    filename: str,
    chunk_count: int = 0,
    file_size: int = 0,
) -> int:
    """
    Register a newly uploaded document version in document_versions.
    Auto-detects if a previous version exists (same filename stem) and archives it.
    Returns the new version number.
    """
    doc_key = _normalize_doc_key(filename)
    now = _now()

    with _get_conn() as conn:
        # Ensure registry entry exists
        conn.execute("""
            INSERT INTO document_registry (doc_key, name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(doc_key) DO UPDATE SET updated_at=excluded.updated_at
        """, (doc_key, filename, now, now))

        # Find latest version number
        row = conn.execute(
            "SELECT MAX(version) FROM document_versions WHERE doc_key=?", (doc_key,)
        ).fetchone()
        latest_version = row[0] or 0
        new_version = latest_version + 1

        # Archive any currently active version
        if latest_version > 0:
            conn.execute("""
                UPDATE document_versions
                SET status='archived', archived_at=?
                WHERE doc_key=? AND status='active'
            """, (now, doc_key))

        # Insert new active version
        conn.execute("""
            INSERT INTO document_versions
                (source_id, doc_key, filename, version, status, chunk_count, file_size, uploaded_at)
            VALUES (?, ?, ?, ?, 'active', ?, ?, ?)
        """, (source_id, doc_key, filename, new_version, chunk_count, file_size, now))

    log.info("document_registered", source_id=source_id,
             filename=filename, doc_key=doc_key, version=new_version)
    return new_version


def get_document_versions(doc_key: str) -> List[Dict[str, Any]]:
    """All versions of a document by doc_key, newest first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM document_versions WHERE doc_key=? ORDER BY version DESC",
            (doc_key,)
        ).fetchall()
    return [dict(r) for r in rows]


def set_document_status(
    source_id: str,
    status: str,
    notes: Optional[str] = None
) -> bool:
    """Update the lifecycle status of a document version. Returns True if found."""
    now = _now()
    with _get_conn() as conn:
        cur = conn.execute("""
            UPDATE document_versions
            SET status=?, archived_at=?, notes=?
            WHERE source_id=?
        """, (status, now if status != 'active' else None, notes, source_id))
    changed = cur.rowcount > 0
    if changed:
        log.info("document_status_changed", source_id=source_id, status=status)
    return changed


def get_active_documents() -> List[Dict[str, Any]]:
    """All currently active document versions with registry metadata."""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT dv.*, dr.name as display_name
            FROM document_versions dv
            JOIN document_registry dr ON dr.doc_key = dv.doc_key
            WHERE dv.status = 'active'
            ORDER BY dv.uploaded_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_all_documents() -> List[Dict[str, Any]]:
    """All document versions, all statuses, grouped by doc_key."""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT dv.*, dr.name as display_name
            FROM document_versions dv
            JOIN document_registry dr ON dr.doc_key = dv.doc_key
            ORDER BY dv.doc_key, dv.version DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_version_by_source_id(source_id: str) -> Optional[Dict[str, Any]]:
    """Return a single document_versions row by source_id."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM document_versions WHERE source_id=?", (source_id,)
        ).fetchone()
    return dict(row) if row else None
