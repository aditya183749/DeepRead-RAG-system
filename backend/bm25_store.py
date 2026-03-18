"""
bm25_store.py — BM25 keyword index for hybrid retrieval.

WHY BM25:
  Vector search excels at semantic similarity but misses exact keyword matches.
  BM25 (Best Match 25) is the gold-standard keyword retrieval algorithm —
  it captures precise terms, abbreviations, IDs, names, and codes that
  dense embeddings often miss.

  Together (BM25 + FAISS), fused via Reciprocal Rank Fusion, we get
  the best of both: semantic understanding + precise keyword recall.

DESIGN:
  - One global BM25 index over ALL chunks.
  - Source filtering happens in Python after scoring (same pattern as FAISS).
  - Corpus (text list) and metadata are stored to disk as a pickle so
    the BM25 index can be rebuilt on startup without re-ingesting documents.
  - BM25Okapi is used (the standard modern variant).

PERSISTENCE:
  storage/bm25/corpus.pkl — { corpus, metadatas, string_ids }
"""

import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi
from config import BM25_DIR

# ─── Corpus File ──────────────────────────────────────────────────────────────

CORPUS_PATH = BM25_DIR / "corpus.pkl"

# ─── Module-Level State ───────────────────────────────────────────────────────

_corpus:     List[str]         = []   # raw texts, indexed by position
_metadatas:  List[Dict]        = []   # metadata dicts, parallel to _corpus
_string_ids: List[str]         = []   # chunk string_ids, parallel to _corpus
_bm25:       Optional[BM25Okapi] = None  # rebuilt on load + each add


# ─── Tokenizer ────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """
    Simple whitespace + lowercase tokenizer for BM25.
    Strips punctuation, lowercases. Fast and effective.
    """
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


# ─── Init & Persistence ───────────────────────────────────────────────────────

def initialize() -> None:
    """Load BM25 corpus from disk and rebuild index. Call once at startup."""
    global _corpus, _metadatas, _string_ids, _bm25

    if CORPUS_PATH.exists():
        with open(CORPUS_PATH, "rb") as f:
            saved = pickle.load(f)
        _corpus     = saved.get("corpus",     [])
        _metadatas  = saved.get("metadatas",  [])
        _string_ids = saved.get("string_ids", [])
        _bm25 = BM25Okapi([_tokenize(t) for t in _corpus]) if _corpus else None
        print(f"[BM25] Loaded corpus with {len(_corpus)} chunks from disk.")
    else:
        _corpus, _metadatas, _string_ids = [], [], []
        _bm25 = None
        print("[BM25] Fresh corpus initialized.")


def _persist() -> None:
    """Persist corpus to disk."""
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_PATH, "wb") as f:
        pickle.dump({
            "corpus":     _corpus,
            "metadatas":  _metadatas,
            "string_ids": _string_ids
        }, f)


# ─── Add Chunks ───────────────────────────────────────────────────────────────

def add_chunks(
    ids: List[str],
    documents: List[str],
    metadatas: List[Dict[str, Any]]
) -> None:
    """
    Add chunks to the BM25 corpus and rebuild the BM25 index.

    Upsert: if a string_id already exists, replace its text + metadata.
    The BM25 index is rebuilt from scratch after each update — acceptable
    cost at ingest time since retrieval is the hot path.
    """
    global _bm25

    # Build a position map for existing IDs
    existing = {sid: i for i, sid in enumerate(_string_ids)}

    for string_id, text, meta in zip(ids, documents, metadatas):
        if string_id in existing:
            pos = existing[string_id]
            _corpus[pos]     = text
            _metadatas[pos]  = meta
        else:
            _corpus.append(text)
            _metadatas.append(meta)
            _string_ids.append(string_id)

    # Rebuild BM25 with updated corpus
    _bm25 = BM25Okapi([_tokenize(t) for t in _corpus])
    _persist()


# ─── BM25 Search ──────────────────────────────────────────────────────────────

def bm25_search(
    query: str,
    source_ids: List[str],
    n_results: int = 20
) -> List[Dict[str, Any]]:
    """
    BM25 keyword search filtered to the given source_ids.

    Returns ranked list of chunks with their BM25 scores.
    Used as one leg of hybrid retrieval fused by RRF in hybrid.py.

    Args:
        query      : Raw user query string (tokenized internally)
        source_ids : Documents to search within
        n_results  : Max results to return

    Returns:
        List of {text, metadata, score} dicts, sorted descending by score
    """
    if _bm25 is None or not _corpus:
        return []

    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)   # float array, one score per corpus doc

    source_id_set = set(source_ids)

    # Pair (score, index), filter by source_id, sort descending
    scored = [
        (float(scores[i]), i)
        for i in range(len(_corpus))
        if _metadatas[i].get("source_id") in source_id_set
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, idx in scored[:n_results]:
        results.append({
            "text":     _corpus[idx],
            "metadata": _metadatas[idx],
            "score":    score
        })

    return results


# ─── Delete Source ────────────────────────────────────────────────────────────

def delete_source(source_id: str) -> int:
    """Remove all chunks for a document from the BM25 corpus."""
    global _bm25

    keep_mask = [m.get("source_id") != source_id for m in _metadatas]
    deleted   = keep_mask.count(False)

    if deleted == 0:
        return 0

    new_corpus     = [_corpus[i]     for i, keep in enumerate(keep_mask) if keep]
    new_metadatas  = [_metadatas[i]  for i, keep in enumerate(keep_mask) if keep]
    new_string_ids = [_string_ids[i] for i, keep in enumerate(keep_mask) if keep]

    _corpus[:], _metadatas[:], _string_ids[:] = new_corpus, new_metadatas, new_string_ids
    _bm25 = BM25Okapi([_tokenize(t) for t in _corpus]) if _corpus else None
    _persist()

    return deleted
