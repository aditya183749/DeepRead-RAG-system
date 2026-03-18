"""
retrieval/rerank.py — Cross-encoder re-ranking + rule-based noise filter.

TWO-STAGE PIPELINE (runs after hybrid BM25+FAISS retrieval):

  Stage 1 — Cross-Encoder Re-ranking:
    Bi-encoders (BGE-M3) embed query and chunks independently.
    Cross-encoders see the FULL (query, chunk) pair together.
    This is much slower but far more accurate — they model
    fine-grained relevance interactions.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
      - Lightweight (22M params), fast on CPU
      - Trained on MS MARCO passage ranking (general retrieval)
      - Score range: ~-10 to +10 (higher = more relevant)

  Stage 2 — Rule-Based Noise Filter:
    Hard cutoffs that remove junk even if the cross-encoder missed it:
      - Too short     : < 50 chars → likely a page number / header
      - Low score     : < MIN_RERANK_SCORE → irrelevant
      - Near-duplicate: cosine > 0.95 with a higher-scored chunk → redundant

RESULT: Clean top-N signal passed to LLM instead of noisy top-K.
"""

import time
import numpy as np
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder
from logger import get_logger
from config import RERANK_MODEL, MIN_RERANK_SCORE, TOP_K_RESULTS

log = get_logger("retrieval")

# ─── Cross-Encoder Singleton ─────────────────────────────────────────────────

_cross_encoder: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    """Lazy-load cross-encoder model. First call downloads ~25MB of weights."""
    global _cross_encoder
    if _cross_encoder is None:
        log.info("loading_cross_encoder", model=RERANK_MODEL)
        _cross_encoder = CrossEncoder(RERANK_MODEL, max_length=512)
        log.info("cross_encoder_ready", model=RERANK_MODEL)
    return _cross_encoder


# ─── Stage 1: Cross-Encoder Re-ranking ───────────────────────────────────────

def _cross_encode(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Score all (query, chunk_text) pairs with cross-encoder.
    Attaches 'rerank_score' to each chunk dict.
    Returns chunks sorted by rerank_score descending.
    """
    if not chunks:
        return []

    encoder = get_cross_encoder()
    pairs = [(query, c["text"]) for c in chunks]
    scores = encoder.predict(pairs)  # shape: (N,)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)


# ─── Stage 2: Rule-Based Noise Filter ────────────────────────────────────────

def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Fast cosine similarity between two pre-normalised or un-normalised vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _noise_filter(
    chunks: List[Dict[str, Any]],
    top_n: int = TOP_K_RESULTS
) -> List[Dict[str, Any]]:
    """
    Apply rule-based noise removal after cross-encoder scoring.

    Rules (in order):
      1. Drop if text too short (< 50 chars) — likely metadata noise
      2. Drop if rerank_score < MIN_RERANK_SCORE — LLM judged it irrelevant
      3. Deduplicate — if two chunks are near-identical, keep higher scored one
      4. Keep top_n from remaining

    Returns at most top_n clean chunks.
    """
    filtered = []

    for chunk in chunks:
        text  = chunk.get("text", "")
        score = chunk.get("rerank_score", 0.0)

        # Rule 1: too short
        if len(text.strip()) < 50:
            log.debug("noise_filter_short", chars=len(text.strip()), score=score)
            continue

        # Rule 2: score below threshold
        if score < MIN_RERANK_SCORE:
            log.debug("noise_filter_low_score", score=score)
            continue

        # Rule 3: near-duplicate check against already accepted chunks
        # Use text hash as fast pre-check; fall back to cosine for edge cases
        is_dup = False
        for accepted in filtered:
            # Fast check: if texts are very similar character-wise
            if abs(len(text) - len(accepted["text"])) < 20:
                # Slow check: compare score-vectors if available
                ev = chunk.get("embedding")
                av = accepted.get("embedding")
                if ev and av:
                    if _cosine_sim(ev, av) > 0.95:
                        is_dup = True
                        log.debug("noise_filter_duplicate")
                        break
                # Fallback: exact substring check
                if text[:100] == accepted["text"][:100]:
                    is_dup = True
                    break

        if not is_dup:
            filtered.append(chunk)

        if len(filtered) >= top_n:
            break

    return filtered


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_n: int = TOP_K_RESULTS
) -> List[Dict[str, Any]]:
    """
    Full re-ranking pipeline: cross-encoder → noise filter → top-N.

    Args:
        query  : The (rewritten) query string
        chunks : Raw chunks from hybrid search (may have 15–30 candidates)
        top_n  : Final number of chunks to pass to LLM

    Returns:
        Cleaned, re-ranked list of at most top_n chunks.
        ALWAYS returns at least 1 chunk if input is non-empty (safety net).
    """
    t0 = time.perf_counter()
    n_in = len(chunks)

    if not chunks:
        return []

    # Stage 1: cross-encoder scoring
    scored = _cross_encode(query, chunks)

    # Stage 2: noise filter
    clean = _noise_filter(scored, top_n=top_n)

    # Safety net: if noise filter removed everything (e.g. low-relevance query),
    # fall back to the top-scored chunks without the score threshold.
    # The LLM will still say "insufficient context" if nothing is relevant.
    if not clean and scored:
        log.warning("noise_filter_removed_all",
            query=query[:60],
            fallback_chunks=min(top_n, len(scored)),
            top_score=round(scored[0]["rerank_score"], 2)
        )
        clean = scored[:top_n]

    latency_ms = (time.perf_counter() - t0) * 1000
    n_out = len(clean)

    log.info("rerank_complete",
        query=query[:80],
        chunks_in=n_in,
        chunks_out=n_out,
        top_score=round(clean[0]["rerank_score"], 3) if clean else None,
        latency_ms=round(latency_ms, 1)
    )

    return clean
