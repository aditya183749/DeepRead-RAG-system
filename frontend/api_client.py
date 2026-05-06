"""
api_client.py — The ONLY file that talks to the FastAPI backend.

WHY THIS FILE EXISTS:
  All HTTP calls are isolated here. This means:
    - app.py stays clean (no requests/httpx calls scattered around)
    - If we ever switch from Streamlit to React, we ONLY rewrite this file
    - If the backend URL changes (e.g. deployed to a server), we change ONE constant

  Every function here maps 1-to-1 with a FastAPI endpoint:
    upload_file()    → POST /upload
    chat_stream()    → POST /chat  (returns a streaming response)
    summarize_doc()  → POST /summarize
    get_sources()    → GET  /sources
"""

import httpx
import json
from typing import Generator, List, Dict, Any, Optional

# ─── Backend URL ──────────────────────────────────────────────────────────────
# FastAPI runs on port 8000 by default when launched with uvicorn
API_BASE_URL = "http://localhost:8000"

# Timeout for non-streaming requests (upload, summarize can be slow)
# 300 seconds = 5 minutes (for large documents/slow LLM responses)
TIMEOUT = httpx.Timeout(300.0)


# ─── Upload File (streaming progress) ────────────────────────────────────────

def upload_file_stream(file_bytes: bytes, filename: str):
    """
    Streams progress events from POST /upload as the backend ingests the file.

    Yields dicts:
      {"type": "progress", "step": "chunking",  "detail": "Splitting 12 pages…"}
      {"type": "progress", "step": "embedding", "detail": "Generating embeddings…"}
      {"type": "complete", "source_id": "…", "chunk_count": 142, "page_count": 12}
      {"type": "error",    "detail": "…"}   ← only on failure
    """
    with httpx.Client(timeout=TIMEOUT) as client:
        with client.stream(
            "POST",
            f"{API_BASE_URL}/upload",
            files={"file": (filename, file_bytes, _get_mime_type(filename))}
        ) as response:
            if response.status_code != 200:
                try:
                    detail = response.json().get("detail", "Unknown upload error")
                except Exception:
                    detail = f"HTTP {response.status_code}"
                yield {"type": "error", "detail": detail}
                return

            # Parse SSE lines
            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        import json as _json
                        event = _json.loads(line[6:])
                        yield event
                        if event.get("type") in ("complete", "error"):
                            break
                    except Exception:
                        pass


def _get_mime_type(filename: str) -> str:
    """Returns the correct MIME type for the file extension."""
    ext = filename.lower().split(".")[-1]
    mime_types = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "doc": "application/msword"
    }
    return mime_types.get(ext, "application/octet-stream")


# ─── Analyze Images (LLaVA, streaming progress) ───────────────────────────────

def analyze_images_stream(source_id: str):
    """
    Calls POST /analyze-images and streams progress events.

    Yields dicts:
      {"type": "progress", "step": "scanning",  "detail": "Scanning 24 pages…"}
      {"type": "progress", "step": "analyzing", "detail": "Analyzing page 3…"}
      {"type": "complete", "images_found": 5, "descriptions_indexed": 5}
      {"type": "error",    "detail": "…"}
    """
    import json as _json
    with httpx.Client(timeout=TIMEOUT) as client:
        with client.stream(
            "POST",
            f"{API_BASE_URL}/analyze-images",
            json={"source_id": source_id}
        ) as response:
            if response.status_code != 200:
                try:
                    detail = response.json().get("detail", "Unknown error")
                except Exception:
                    detail = f"HTTP {response.status_code}"
                yield {"type": "error", "detail": detail}
                return
            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        event = _json.loads(line[6:])
                        yield event
                        if event.get("type") in ("complete", "error"):
                            break
                    except Exception:
                        pass



# ─── Chat (Streaming) ─────────────────────────────────────────────────────────

def chat_stream(
    question: str,
    source_ids: List[str],
    session_id: Optional[str] = None
) -> Generator[Dict, None, None]:
    """
    Sends a question to POST /chat and yields parsed SSE events.
    Includes session_id for conversation memory (Phase 3).

    Yields events of types:
      {"type": "session_id", "session_id": "..."}  ← first event, always
      {"type": "token", "content": "..."}           ← streamed tokens
      {"type": "citations", "citations": [...]}     ← after stream ends
      {"type": "done"}                              ← signals completion
      {"type": "error", "content": "..."}           ← on failure
    """
    payload: Dict[str, Any] = {
        "question":   question,
        "source_ids": source_ids,
    }
    if session_id:
        payload["session_id"] = session_id

    with httpx.Client(timeout=TIMEOUT) as client:
        with client.stream("POST", f"{API_BASE_URL}/chat", json=payload) as response:

            if response.status_code != 200:
                error_body = response.read().decode()
                yield {"type": "error", "content": f"Request failed: {error_body}"}
                return

            for line in response.iter_lines():
                line = line.strip()

                if line.startswith("data: "):
                    data = line[6:]

                    if data == "[DONE]":
                        yield {"type": "done"}
                        break

                    try:
                        event = json.loads(data)
                        yield event
                    except json.JSONDecodeError:
                        continue

                elif line == "":
                    continue


# ─── Session API ──────────────────────────────────────────────────────────────

def get_sessions() -> List[Dict[str, Any]]:
    """Fetch all past sessions for the sidebar history panel."""
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            r = client.get(f"{API_BASE_URL}/sessions")
        if r.status_code == 200:
            return r.json().get("sessions", [])
    except Exception:
        pass
    return []


def get_session_history(session_id: str) -> Optional[Dict[str, Any]]:
    """Fetch full message history for a specific session."""
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            r = client.get(f"{API_BASE_URL}/sessions/{session_id}")
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def delete_session(session_id: str) -> bool:
    """Delete a session and all its messages."""
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            r = client.delete(f"{API_BASE_URL}/sessions/{session_id}")
        return r.status_code == 200
    except Exception:
        return False


# ─── Summarize ────────────────────────────────────────────────────────────────

def summarize_doc(source_id: str) -> Dict[str, Any]:
    """
    Sends a summarize request to POST /summarize.

    This is a blocking call — we wait for the full summary to generate
    before returning. The frontend shows a spinner during this wait.

    Returns:
        Dict with source_id, filename, summary (markdown text)
    """
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(
            f"{API_BASE_URL}/summarize",
            json={"source_id": source_id}
        )

    if response.status_code == 200:
        return response.json()
    else:
        error_detail = response.json().get("detail", "Unknown error during summarization")
        raise Exception(f"Summarization failed ({response.status_code}): {error_detail}")


# ─── Get Sources ──────────────────────────────────────────────────────────────

def get_sources() -> List[Dict[str, Any]]:
    """
    Fetches the list of all uploaded documents from GET /sources.

    Returns:
        List of dicts: [{"source_id": "...", "filename": "...", "chunk_count": 42, ...}]
        Returns empty list if the backend is unreachable (graceful degradation).

    Called on every Streamlit rerun to keep the sidebar document list fresh.
    """
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            # Short 10s timeout — sidebar should load fast
            response = client.get(f"{API_BASE_URL}/sources")

        if response.status_code == 200:
            return response.json().get("sources", [])
        return []

    except httpx.ConnectError:
        # Backend is not running — return empty list, let app.py handle UI
        return []
    except Exception:
        return []


# ─── Health Check ─────────────────────────────────────────────────────────────

def check_backend_health() -> bool:
    """
    Returns True if the FastAPI backend is reachable, False otherwise.
    Used by app.py to show a warning if the backend isn't running.
    """
    try:
        with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
            response = client.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except Exception:
        return False


# ─── Delete Source ────────────────────────────────────────────────────────────────────────

def delete_source(source_id: str) -> Dict[str, Any]:
    """
    Delete a document from FAISS + BM25 via DELETE /sources/{source_id}.
    Returns {'status': 'deleted', 'source_id': ..., 'chunks_removed': N}
    """
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.delete(f"{API_BASE_URL}/sources/{source_id}")
        if response.status_code == 200:
            return response.json()
        error_detail = response.json().get("detail", "Unknown error")
        raise Exception(f"Delete failed ({response.status_code}): {error_detail}")
    except httpx.ConnectError:
        raise Exception("Backend not reachable. Is it running on port 8000?")


# ─── Document Lifecycle Management ────────────────────────────────────────────

def get_documents(active_only: bool = False) -> List[Dict[str, Any]]:
    """
    GET /documents — returns all document versions grouped by doc_key.
    Each group has 'doc_key', 'display_name', and 'versions' list.
    """
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(
                f"{API_BASE_URL}/documents",
                params={"active_only": str(active_only).lower()}
            )
        if response.status_code == 200:
            return response.json().get("documents", [])
        return []
    except Exception:
        return []


def set_document_status(source_id: str, status: str, notes: str = "") -> Dict[str, Any]:
    """
    PATCH /documents/{source_id}/status — archive, expire, or restore a version.
    status: 'active' | 'archived' | 'expired'
    """
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.patch(
                f"{API_BASE_URL}/documents/{source_id}/status",
                params={"status": status, "notes": notes}
            )
        if response.status_code == 200:
            return response.json()
        raise Exception(response.json().get("detail", "Status update failed"))
    except httpx.ConnectError:
        raise Exception("Backend not reachable.")


def get_version_diff(source_id_old: str, source_id_new: str) -> Dict[str, Any]:
    """
    GET /documents/{old}/{new}/diff — chunk-level diff between two document versions.
    """
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(
                f"{API_BASE_URL}/documents/{source_id_old}/{source_id_new}/diff"
            )
        if response.status_code == 200:
            return response.json()
        raise Exception(response.json().get("detail", "Diff failed"))
    except httpx.ConnectError:
        raise Exception("Backend not reachable.")
