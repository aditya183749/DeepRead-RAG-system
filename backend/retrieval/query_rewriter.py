"""
retrieval/query_rewriter.py — LLM-powered query intelligence layer.

Runs BEFORE retrieval on every /chat request.

PIPELINE:
  Step 1 — Expand    : Generate 2 alternative phrasings via Ollama
  Step 2 — HyDE      : Generate a hypothetical ideal answer, embed it
  Step 3 — Classify  : Determine query type for retrieval loop control

WHY THIS MATTERS:
  Raw user queries are almost never optimal for retrieval:
    "that thing about attention" → terrible embedding
    "What is the self-attention mechanism in transformers?" → great embedding

  HyDE (Hypothetical Document Embeddings) is especially powerful:
  instead of embedding the question, we embed a hypothetical answer.
  Documents are more similar to other documents than to questions.

OUTPUTS:
  RewriteResult(
    original         : raw user query
    variants         : [original, expanded_1, expanded_2, hyde]
    hyde_embedding   : vector of hypothetical answer
    query_type       : "factual" | "comparative" | "multi-hop" | "summarization"
    num_hops         : 1 | 2 | 3 | 0
  )
"""

import time
import json
from dataclasses import dataclass, field
from typing import List, Optional

import ollama
from logger import get_logger
from config import CHAT_MODEL, OLLAMA_BASE_URL, OLLAMA_KEEP_ALIVE
from ingestion import generate_embeddings

log = get_logger("retrieval")


@dataclass
class RewriteResult:
    original: str
    variants: List[str]
    hyde_embedding: List[float]
    query_type: str          # factual | comparative | multi-hop | summarization
    num_hops: int            # retrieval loop depth


# ─── Prompts ──────────────────────────────────────────────────────────────────

_EXPAND_PROMPT = """\
You are a query expansion assistant for a document retrieval system.

Given the user's question, produce:
1. Two alternative phrasings that capture the same intent differently
2. One short keyword-style search string (no sentences)

Respond with ONLY a JSON object, no markdown:
{{"variant_1": "...", "variant_2": "...", "keywords": "..."}}

User question: {query}"""

_HYDE_PROMPT = """\
You are helping a retrieval system work better.

Write a short paragraph (3-5 sentences) that would be a perfect answer
to the following question if the relevant document existed.
Be specific, technical, and use vocabulary that would appear in the source document.

Return ONLY the paragraph, no preamble.

Question: {query}"""

_CLASSIFY_PROMPT = """\
Classify this question into exactly one of these types:
- factual      : single specific fact, number, definition, or name
- comparative  : comparing two or more things
- multi-hop    : requires chaining multiple facts together
- summarization: asking for an overview or summary of a topic or document

Respond with ONLY one word from the list above.

Question: {query}"""


# ─── Ollama helper ────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, max_tokens: int = 300) -> str:
    """Single non-streaming Ollama call. Returns response text."""
    client = ollama.Client(host=OLLAMA_BASE_URL)
    resp = client.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": max_tokens, "temperature": 0.3},
        keep_alive=OLLAMA_KEEP_ALIVE
    )
    return resp["message"]["content"].strip()


# ─── Step 1: Expand ───────────────────────────────────────────────────────────

def _expand_query(query: str) -> List[str]:
    """
    Generate 2 alternative phrasings + keyword string via LLM.
    Falls back to [query] only if Ollama is unreachable.
    """
    try:
        raw = _call_ollama(_EXPAND_PROMPT.format(query=query))
        # Strip markdown code fences if LLM wraps it anyway
        raw = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)
        variants = [
            data.get("variant_1", ""),
            data.get("variant_2", ""),
            data.get("keywords", "")
        ]
        return [v for v in variants if v.strip()]
    except Exception as e:
        log.warning("query_expand_failed", error=str(e), query=query)
        return []


# ─── Step 2: HyDE ─────────────────────────────────────────────────────────────

def _generate_hyde(query: str) -> Optional[List[float]]:
    """
    Hypothetical Document Embedding:
    1. LLM generates a hypothetical ideal answer paragraph
    2. Embed that paragraph using BGE-M3
    3. Use the embedding as an additional retrieval vector

    Falls back to None if Ollama unavailable (retrieval still works without it).
    """
    try:
        hypothetical_doc = _call_ollama(_HYDE_PROMPT.format(query=query), max_tokens=200)
        if hypothetical_doc:
            return generate_embeddings([hypothetical_doc])[0]
    except Exception as e:
        log.warning("hyde_failed", error=str(e))
    return None


# ─── Step 3: Classify ─────────────────────────────────────────────────────────

_HOP_MAP = {
    "factual":       1,
    "comparative":   2,
    "multi-hop":     3,
    "summarization": 0,
}

def _classify_query(query: str) -> tuple[str, int]:
    """Returns (query_type, num_hops)."""
    try:
        result = _call_ollama(_CLASSIFY_PROMPT.format(query=query), max_tokens=10)
        qtype = result.lower().strip().strip(".")
        if qtype not in _HOP_MAP:
            qtype = "factual"   # safe default
        return qtype, _HOP_MAP[qtype]
    except Exception as e:
        log.warning("classify_failed", error=str(e))
        return "factual", 1


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def rewrite_query(query: str) -> RewriteResult:
    """
    Full query intelligence pipeline.
    Called once per /chat request before any retrieval.

    If Ollama is down, degrades gracefully:
      - variants = [original query]
      - hyde_embedding = embed(original query)
      - query_type = "factual", num_hops = 1
    """
    t0 = time.perf_counter()

    # All three steps are cheap (sequential small LLM calls)
    expansions = _expand_query(query)
    hyde_emb   = _generate_hyde(query)
    qtype, hops = _classify_query(query)

    # Build variant list: original first, then expansions
    variants = [query] + expansions

    # If HyDE failed, fall back to embedding the original query
    if hyde_emb is None:
        hyde_emb = generate_embeddings([query])[0]

    latency_ms = (time.perf_counter() - t0) * 1000

    log.info("query_rewritten",
        original=query,
        variants=variants,
        query_type=qtype,
        num_hops=hops,
        latency_ms=round(latency_ms, 1)
    )

    return RewriteResult(
        original=query,
        variants=variants,
        hyde_embedding=hyde_emb,
        query_type=qtype,
        num_hops=hops
    )
