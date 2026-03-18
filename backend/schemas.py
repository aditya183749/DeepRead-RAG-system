"""
schemas.py — Pydantic models for all API request and response bodies.

WHY THIS FILE EXISTS:
  FastAPI uses Pydantic models for two things:
    1. VALIDATION: Automatically checks that incoming request data has
       the right fields and types. If a required field is missing,
       FastAPI returns a clear 422 error automatically.
    2. SERIALIZATION: Converts Python objects into JSON for responses.

  By defining all models here, every endpoint in main.py stays clean —
  no manual JSON parsing or type checking needed.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ─── Upload Endpoint (/upload) ────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """
    Returned after a file is successfully uploaded and processed.

    Example JSON response:
    {
        "source_id": "a3f9...",
        "filename": "report.pdf",
        "chunk_count": 42,
        "page_count": 10,
        "message": "File uploaded and indexed successfully"
    }
    """
    source_id: str        # UUID generated for this document
    filename: str         # Original filename (e.g. "report.pdf")
    chunk_count: int      # How many text chunks were created and stored in FAISS
    page_count: int       # How many pages the document has
    message: str          # Human-readable success message


# ─── Chat Endpoint (/chat) ────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Body sent by the frontend for every chat message.
    """
    question: str = Field(..., min_length=1, max_length=2000)
    source_ids: List[str]                              # One or more document UUIDs to search across
    session_id: Optional[str] = None                   # Session UUID for conversation memory (Phase 3)
    rewrite_query: bool = True                         # Enable LLM query rewriting (can disable for testing)


class Citation(BaseModel):
    """
    A single source citation shown below the answer.
    Points to the exact page and file where the answer came from.

    Example:
    {
        "filename": "report.pdf",
        "page_number": 7,
        "chunk_index": 12
    }
    """
    filename: str
    page_number: int
    chunk_index: int


# NOTE: The /chat endpoint returns a StreamingResponse (not this model)
# for the actual text stream. The ChatCitationEvent below is sent as a
# final JSON event AFTER the stream ends, containing citation data.

class ChatCitationEvent(BaseModel):
    """
    Sent as a final server-sent event after streaming the answer.
    The frontend listens for event type "citations" and renders these below the answer.

    Example JSON event:
    {
        "citations": [
            {"filename": "report.pdf", "page_number": 7, "chunk_index": 12},
            {"filename": "report.pdf", "page_number": 11, "chunk_index": 23}
        ]
    }
    """
    citations: List[Citation]


# ─── Summarize Endpoint (/summarize) ─────────────────────────────────────────

class SummarizeRequest(BaseModel):
    """
    Body sent when user clicks the Summarize button.

    Example JSON received:
    {
        "source_id": "a3f9..."
    }
    """
    source_id: str        # Document to summarize — fetches ALL its chunks from ChromaDB


class SummarizeResponse(BaseModel):
    """
    Returned after the LLM generates the structured summary.

    Example JSON response:
    {
        "source_id": "a3f9...",
        "filename": "report.pdf",
        "summary": "## Overview\\n...\\n## Key Points\\n- ...\\n## Conclusions\\n..."
    }
    """
    source_id: str
    filename: str
    summary: str          # Full markdown-formatted summary text from the LLM


# ─── Sources Endpoint (/sources) ─────────────────────────────────────────────

class SourceItem(BaseModel):
    """
    Represents one uploaded document in the sources list.

    Example:
    {
        "source_id": "a3f9...",
        "filename": "report.pdf",
        "chunk_count": 42,
        "page_count": 10
    }
    """
    source_id: str
    filename: str
    chunk_count: int
    page_count: int


class SourcesResponse(BaseModel):
    """
    Returned by GET /sources — the list shown in the Streamlit sidebar.

    Example JSON response:
    {
        "sources": [
            {"source_id": "a3f9...", "filename": "report.pdf", "chunk_count": 42, "page_count": 10},
            {"source_id": "b7c2...", "filename": "notes.docx", "chunk_count": 18, "page_count": 4}
        ],
        "total": 2
    }
    """
    sources: List[SourceItem]
    total: int            # Total number of documents — useful for the sidebar count badge


# ─── Error Response ───────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """
    Standard error format returned when something goes wrong.
    FastAPI uses this for HTTPException responses.

    Example:
    {
        "detail": "File type not supported. Only PDF and DOCX allowed.",
        "error_code": "UNSUPPORTED_FILE_TYPE"
    }
    """
    detail: str
    error_code: Optional[str] = None  # Optional machine-readable code for the frontend


# ─── Phase 3 — Session Memory ─────────────────────────────────────────────────

class SessionItem(BaseModel):
    """One session entry shown in the past-conversations sidebar."""
    session_id: str
    title: str           # First user message (truncated to 60 chars)
    created_at: str      # ISO timestamp
    updated_at: str
    message_count: int


class SessionsResponse(BaseModel):
    sessions: List[SessionItem]
    total: int


class MessageItem(BaseModel):
    """A single chat message (user or assistant)."""
    id: int
    role: str            # "user" | "assistant"
    content: str
    created_at: str


class SessionHistoryResponse(BaseModel):
    """Full message history for a resumed session."""
    session_id: str
    title: str
    messages: List[MessageItem]
