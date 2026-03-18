"""
retrieval/context_assembly.py — Short-Term Memory (STM) context builder.

PROBLEM:
  Raw retrieval gives us N chunks ordered by relevance score.
  Passing them directly to the LLM has two issues:
    1. Same info repeated from different chunks → wastes context tokens
    2. Chunks ordered by score, not document order → incoherent narrative

SOLUTION — Three-step assembly:
  Step 1: Deduplicate  — remove near-identical chunks (cosine > 0.95)
  Step 2: Sort         — order remaining chunks by (source_id, chunk_index)
                         so the LLM reads information in document order
  Step 3: Token budget — if context > STM_TOKEN_BUDGET tokens,
                         drop lowest-scored chunks until under budget

OUTPUT FORMAT (per chunk):
  [SOURCE: filename.pdf | Page 3 | Section: Introduction | Relevance: 0.87]
  <chunk text>
  ---

This labeled block is injected into the system prompt directly.
"""

import numpy as np
import tiktoken
from typing import List, Dict, Any

from logger import get_logger
from config import STM_TOKEN_BUDGET, STM_DEDUP_THRESHOLD

log = get_logger("retrieval")

# Tokenizer for budget enforcement — same as used at ingest time
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_tokenizer.encode(text))


def _cosine_sim(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


# ─── Step 1: Deduplicate ──────────────────────────────────────────────────────

def _deduplicate(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove semantically near-identical chunks.
    Uses text overlap heuristic (fast) before cosine (slow).
    Keeps the chunk with the higher rerank_score.
    """
    kept = []
    for chunk in chunks:
        text = chunk.get("text", "")
        is_dup = False
        for accepted in kept:
            # Fast: check text prefix similarity
            if text[:80].strip() == accepted["text"][:80].strip():
                is_dup = True
                # Keep higher scoring chunk
                if chunk.get("rerank_score", 0) > accepted.get("rerank_score", 0):
                    kept.remove(accepted)
                    kept.append(chunk)
                break
            # Slow cosine check (only if embeddings are present)
            ec, ea = chunk.get("embedding"), accepted.get("embedding")
            if ec and ea and _cosine_sim(ec, ea) > STM_DEDUP_THRESHOLD:
                is_dup = True
                if chunk.get("rerank_score", 0) > accepted.get("rerank_score", 0):
                    kept.remove(accepted)
                    kept.append(chunk)
                break
        if not is_dup:
            kept.append(chunk)
    return kept


# ─── Step 2: Sort by Document Position ───────────────────────────────────────

def _sort_by_position(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort chunks by (source_id, chunk_index) so the LLM reads
    information in the order it appears in the original document,
    not in relevance order.

    This dramatically improves answer coherence and reduces hallucination
    caused by out-of-order context.
    """
    return sorted(
        chunks,
        key=lambda c: (
            c.get("metadata", {}).get("source_id", ""),
            c.get("metadata", {}).get("chunk_index", 0)
        )
    )


# ─── Step 3: Token Budget Enforcement ────────────────────────────────────────

def _enforce_budget(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drop lowest-scored chunks until total token count is within STM_TOKEN_BUDGET.
    Operates on a copy sorted by score so we always drop the least relevant.
    """
    if not chunks:
        return []

    total = _token_count(" ".join(c.get("text", "") for c in chunks))
    if total <= STM_TOKEN_BUDGET:
        return chunks

    # Make a mutable list, drop lowest scored until under budget
    mutable = list(chunks)
    while total > STM_TOKEN_BUDGET and len(mutable) > 1:
        # Find and remove the chunk with the lowest rerank_score
        min_idx = min(range(len(mutable)),
                      key=lambda i: mutable[i].get("rerank_score", 0.0))
        dropped = mutable.pop(min_idx)
        total -= _token_count(dropped.get("text", ""))
        log.debug("stm_budget_drop",
            dropped_chunk_index=dropped.get("metadata", {}).get("chunk_index"),
            score=dropped.get("rerank_score", 0.0),
            remaining_tokens=total
        )

    return mutable


# ─── Build Context Block ─────────────────────────────────────────────────────

def _build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Format chunks as a labeled context block for the LLM prompt.

    Format per chunk:
      [SOURCE: filename.pdf | Page 3 | Section: Introduction | Relevance: 0.87]
      <chunk text>
      ---
    """
    lines = []
    for chunk in chunks:
        meta  = chunk.get("metadata", {})
        score = chunk.get("rerank_score", chunk.get("score", 0.0))

        header = (
            f"[SOURCE: {meta.get('filename','?')} "
            f"| Page {meta.get('page_number','?')} "
            f"| Section: {meta.get('section_heading','—') or '—'} "
            f"| Relevance: {score:.2f}]"
        )
        lines.append(header)
        lines.append(chunk.get("text", ""))
        lines.append("---")

    return "\n".join(lines)


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def assemble_context(chunks: List[Dict[str, Any]]) -> tuple[str, List[Dict]]:
    """
    Full STM context assembly pipeline:
      deduplicate → sort by document position → enforce token budget → build block

    Args:
        chunks: Re-ranked chunks from rerank.py

    Returns:
        (context_block: str, final_chunks: list)
        context_block is injected directly into the LLM system prompt.
        final_chunks is returned for citation metadata in the response.
    """
    n_in = len(chunks)

    deduped  = _deduplicate(chunks)
    sorted_  = _sort_by_position(deduped)
    budgeted = _enforce_budget(sorted_)
    context  = _build_context_block(budgeted)

    total_tokens = _token_count(context)

    log.info("context_assembled",
        chunks_in=n_in,
        chunks_after_dedup=len(deduped),
        chunks_final=len(budgeted),
        total_tokens=total_tokens
    )

    return context, budgeted
