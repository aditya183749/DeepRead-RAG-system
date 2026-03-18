"""
logger.py - Structured JSON logging for the entire backend.

WHY STRUCTURED LOGGING:
  Plain print() is invisible in production and unqueryable.
  structlog writes JSON lines — each log event is a searchable,
  filterable JSON object with timestamps, levels, and all context fields.

LOG FILES (storage/logs/):
  app.jsonl       — every log event (rotate daily, 30 days retention)
  errors.jsonl    — errors only (90 days retention)
  retrieval.jsonl — retrieval audit trail per query
  ingest.jsonl    — document ingest events

USAGE:
  from logger import get_logger
  log = get_logger(__name__)
  log.info("retrieval_complete", query_id=qid, chunks=5, latency_ms=42.3)
"""

import sys
import io
import logging
import structlog
from pathlib import Path
from config import LOG_DIR


# ─── File Handlers ────────────────────────────────────────────────────────────

def _make_file_handler(filename: str, level: int = logging.DEBUG) -> logging.FileHandler:
    """Create a file handler that writes JSON lines."""
    path = LOG_DIR / filename
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    return handler


# ─── Raw stdlib logger setup ──────────────────────────────────────────────────

_configured = False


def configure_logging() -> None:
    """
    Configure structlog once at application startup.
    Call from main.py lifespan - idempotent.
    """
    # Force UTF-8 on Windows console (default is cp1252 which crashes on
    # Unicode chars like arrows or em-dashes that appear in log messages)
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    global _configured
    if _configured:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # stdlib root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console - human-readable, UTF-8 safe
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s  %(name)s  %(message)s"))
    console_handler.errors = 'replace'   # Never crash on unencodable chars
    root.addHandler(console_handler)

    # app.jsonl — everything
    root.addHandler(_make_file_handler("app.jsonl", logging.DEBUG))

    # errors.jsonl — errors only
    root.addHandler(_make_file_handler("errors.jsonl", logging.ERROR))

    # retrieval.jsonl — retrieval events (filtered by logger name in emit)
    retrieval_handler = _make_file_handler("retrieval.jsonl", logging.DEBUG)
    retrieval_logger = logging.getLogger("retrieval")
    retrieval_logger.propagate = False
    retrieval_logger.addHandler(retrieval_handler)
    retrieval_logger.addHandler(_make_file_handler("app.jsonl", logging.DEBUG))

    # ingest.jsonl — ingest events
    ingest_handler = _make_file_handler("ingest.jsonl", logging.DEBUG)
    ingest_logger = logging.getLogger("ingest")
    ingest_logger.propagate = False
    ingest_logger.addHandler(ingest_handler)
    ingest_logger.addHandler(_make_file_handler("app.jsonl", logging.DEBUG))

    # sessions.jsonl — session lifecycle: create, resume, query start/end
    sessions_logger = logging.getLogger("sessions")
    sessions_logger.propagate = False
    sessions_logger.addHandler(_make_file_handler("sessions.jsonl", logging.DEBUG))
    sessions_logger.addHandler(_make_file_handler("app.jsonl", logging.DEBUG))

    # tokens.jsonl — token counts at every pipeline stage
    tokens_logger = logging.getLogger("tokens")
    tokens_logger.propagate = False
    tokens_logger.addHandler(_make_file_handler("tokens.jsonl", logging.DEBUG))
    tokens_logger.addHandler(_make_file_handler("app.jsonl", logging.DEBUG))

    # chunks.jsonl — per-chunk detail (score, page, snippet) for every query
    chunks_logger = logging.getLogger("chunks")
    chunks_logger.propagate = False
    chunks_logger.addHandler(_make_file_handler("chunks.jsonl", logging.DEBUG))
    # chunks are debug-only, not mirrored to app.jsonl to keep it readable

    # structlog processors
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),   # output as JSON
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    _configured = True
    structlog.get_logger("app").info("logging_configured", log_dir=str(LOG_DIR))


def get_logger(name: str = "app"):
    """Get a bound structlog logger. Use module __name__ as name."""
    return structlog.get_logger(name)
