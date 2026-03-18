"""
llm.py — Prompt building, Ollama calls, streaming, and summarization.

WHY THIS FILE EXISTS:
  This file is the "brain" of the RAG system. It handles:
    1. Building the RAG prompt (system + context chunks + user question)
    2. Calling llama3.2:3b via Ollama with STREAMING enabled
    3. Yielding tokens as server-sent events (SSE) for real-time display
    4. Sending citation data as a final SSE event after streaming ends
    5. Building the summarization prompt and calling the model (no streaming)
"""

import json
from typing import List, Dict, Any, Generator

import ollama

from config import CHAT_MODEL, OLLAMA_BASE_URL, OLLAMA_KEEP_ALIVE


# ─── Ollama Client ────────────────────────────────────────────────────────────

def get_ollama_client() -> ollama.Client:
    """
    Returns an Ollama client pointed at the local Ollama server.
    Ollama runs as a background process on http://localhost:11434.
    """
    return ollama.Client(host=OLLAMA_BASE_URL)


# ─── RAG Prompt Builder ───────────────────────────────────────────────────────

def build_rag_prompt(
    question: str,
    context_block: str,
    stm_block: str = ""
) -> List[Dict[str, str]]:
    """
    Assembles the full prompt with optional STM conversation history.

    Message order:
      1. SYSTEM: grounding rules
      2. ASSISTANT (if stm_block): conversation history as context
      3. USER: context chunks + question

    Args:
        question      : The (rewritten) user question
        context_block : Pre-assembled, labeled context from context_assembly
        stm_block     : Formatted last-N-turns from STM builder (empty = new session)
    """
    system_message = """\
You are a precise document assistant. Your rules are absolute:

MUST DO:
- Answer ONLY from the provided context chunks below
- Always cite [filename, Page N] inline when referencing specific information
- If multiple chunks support the same point, cite all of them
- Structure your answer clearly with headings if the answer is long
- Use conversation history to understand follow-up questions

MUST NOT DO:
- Do not use any knowledge outside the provided context
- Do not speculate, infer, or fill gaps with general knowledge

IF CONTEXT IS INSUFFICIENT:
Respond with exactly: "I could not find sufficient information in the provided documents to answer this accurately."

CONTEXT CHUNKS (use ONLY these for your answer):
{context}""".format(context=context_block)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_message}
    ]

    # Inject conversation history as prior assistant context if available
    if stm_block:
        messages.append({"role": "assistant", "content": stm_block})

    messages.append({"role": "user", "content": f"Question: {question}"})
    return messages


# ─── Streaming Chat Generator ─────────────────────────────────────────────────

def stream_rag_answer(
    question: str,
    context_block: str,
    final_chunks: List[Dict[str, Any]],
    stm_block: str = ""
) -> Generator[str, None, None]:
    """
    Streams the LLM answer as Server-Sent Events.

    Args:
        question      : Rewritten user question
        context_block : Pre-assembled context from context_assembly
        final_chunks  : Post-rerank chunks (for citation metadata)
        stm_block     : Conversation history from STM builder
    """
    client = get_ollama_client()
    messages = build_rag_prompt(question, context_block, stm_block)

    try:
        stream = client.chat(
            model=CHAT_MODEL,
            messages=messages,
            stream=True,
            keep_alive=OLLAMA_KEEP_ALIVE
        )

        last_chunk = None
        for chunk in stream:
            last_chunk = chunk
            token = chunk["message"]["content"]
            if token:
                event = json.dumps({"type": "token", "content": token})
                yield f"data: {event}\n\n"

        # Emit real token counts from Ollama's final chunk
        # eval_count = output tokens, prompt_eval_count = input tokens
        if last_chunk:
            usage_event = json.dumps({
                "type": "usage",
                "prompt_tokens":     last_chunk.get("prompt_eval_count", 0),
                "completion_tokens": last_chunk.get("eval_count", 0),
                "total_tokens":      last_chunk.get("prompt_eval_count", 0) + last_chunk.get("eval_count", 0),
            })
            yield f"data: {usage_event}\n\n"

        # After full answer, send citation metadata
        citations = []
        for chunk in final_chunks:
            meta = chunk.get("metadata", {})
            citations.append({
                "filename":    meta.get("filename", ""),
                "page_number": meta.get("page_number", 0),
                "chunk_index": meta.get("chunk_index", 0),
                "section":     meta.get("section_heading", ""),
                "score":       round(chunk.get("rerank_score", chunk.get("score", 0.0)), 3)
            })

        citation_event = json.dumps({"type": "citations", "citations": citations})
        yield f"data: {citation_event}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_event = json.dumps({"type": "error", "content": str(e)})
        yield f"data: {error_event}\n\n"
        yield "data: [DONE]\n\n"


# ─── Summarization (Non-Streaming) ───────────────────────────────────────────

def generate_summary(
    chunks: List[Dict[str, Any]],
    filename: str
) -> str:
    """
    Generates a structured document summary from all chunks in order.

    Unlike chat (which uses top-K relevant chunks), summarization
    uses ALL chunks from the document sorted by chunk_index.
    This gives the model a full picture of the document.

    The prompt requests a specific 3-part structure:
      1. Overview: 2-3 sentence high-level description
      2. Key Points: 5 bullet points of main ideas
      3. Conclusions: What the document concludes or recommends

    This is NOT streamed — we wait for the complete summary
    before returning it to the frontend. Summaries take 10-30 seconds
    depending on document length.

    ⚠️  LIMITATION: Very large documents (200+ pages) may overflow
    the 3B model's context window (128K tokens for Llama 3.2 3B).
    For MVP this is acceptable. Future fix: map-reduce summarization.
    """
    client = get_ollama_client()

    # Build the full document text from all chunks in order
    full_text_parts = []
    for chunk in chunks:
        page = chunk["metadata"].get("page_number", "?")
        full_text_parts.append(f"[Page {page}]: {chunk['text']}")

    document_text = "\n\n".join(full_text_parts)

    # Truncate if extremely long (safety limit for small models)
    MAX_CHARS = 12000  # ~3000 tokens, safe for 3B model
    if len(document_text) > MAX_CHARS:
        document_text = document_text[:MAX_CHARS] + "\n\n[... document truncated for summary ...]"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a document summarization assistant. "
                "Create a clear, structured summary of the provided document."
            )
        },
        {
            "role": "user",
            "content": (
                f"Please summarize the following document: '{filename}'\n\n"
                f"{document_text}\n\n"
                f"Provide your summary in this exact format:\n\n"
                f"## Overview\n"
                f"[2-3 sentence overview of what this document is about]\n\n"
                f"## Key Points\n"
                f"- [Key point 1]\n"
                f"- [Key point 2]\n"
                f"- [Key point 3]\n"
                f"- [Key point 4]\n"
                f"- [Key point 5]\n\n"
                f"## Conclusions\n"
                f"[Main conclusions or recommendations from the document]"
            )
        }
    ]

    response = client.chat(
        model=CHAT_MODEL,
        messages=messages,
        stream=False,                      # Wait for full response
        keep_alive=OLLAMA_KEEP_ALIVE
    )

    return response["message"]["content"]


# ─── Session Summarizer ───────────────────────────────────────────────────────

def summarize_session(turns: list) -> str:
    """
    Compress a list of old conversation turns into a 1-2 sentence summary.
    Called when session exceeds STM_TURNS to keep future prompts efficient.

    Args:
        turns: List of {"role": "user"|"assistant", "content": "..."} dicts

    Returns:
        A concise text summary stored in session_summaries table.
    """
    client = get_ollama_client()

    # Format turns into a readable transcript
    transcript_lines = []
    for t in turns:
        role = "User" if t["role"] == "user" else "Assistant"
        transcript_lines.append(f"{role}: {t['content'][:400]}")
    transcript = "\n".join(transcript_lines)

    messages = [
        {
            "role": "system",
            "content": (
                "You summarize conversation history concisely. "
                "Output one short paragraph (2-3 sentences) capturing the key topics discussed. "
                "Do not add any preamble, just the summary paragraph."
            )
        },
        {
            "role": "user",
            "content": f"Summarize this conversation:\n\n{transcript}"
        }
    ]

    response = client.chat(
        model=CHAT_MODEL,
        messages=messages,
        stream=False,
        keep_alive=OLLAMA_KEEP_ALIVE
    )

    return response["message"]["content"].strip()
