"""
vector_store.py — FAISS-based vector index replacing ChromaDB.

DESIGN:
  FAISS is an in-process C++ library — no server overhead, supports
  billions of vectors, with GPU support when needed.

  Since FAISS has no native metadata filtering (unlike ChromaDB),
  the strategy is:
    1. Over-fetch FAISS_FETCH_K candidates
    2. Filter Python-side by source_id(s)
    3. Return top TOP_K_RESULTS

  State (kept in memory, persisted to disk):
    _index   : faiss.IndexIDMap wrapping IndexFlatIP (cosine via normalized vecs)
    _store   : dict mapping int_id → {string_id, text, metadata}
    _id_map  : dict mapping string_id → int_id (for dedup/delete)
    _counter : monotonically increasing int64 ID (never reused)

  Persistence files (storage/faiss/):
    documents.index     — binary FAISS index
    documents_meta.pkl  — pickled (_store, _id_map, _counter)

  Thread safety:
    _write_lock guards all mutations. FAISS is NOT thread-safe for writes.
    Reads (search) are safe to run concurrently.
"""

import faiss
import pickle
import threading
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import (
    FAISS_INDEX_PATH, FAISS_META_PATH,
    EMBED_DIM, FAISS_FETCH_K, TOP_K_RESULTS
)

# ─── Module-Level State ───────────────────────────────────────────────────────

_index: Optional[faiss.IndexIDMap] = None
_store: Dict[int, Dict[str, Any]] = {}   # int_id → {string_id, text, metadata}
_id_map: Dict[str, int] = {}             # string_id → int_id
_counter: int = 0                        # next available int_id
_write_lock = threading.Lock()


# ─── Init & Persistence ───────────────────────────────────────────────────────

def initialize() -> None:
    """
    Load or create the FAISS index. Call ONCE at application startup.

    IndexFlatIP: exact inner product search.
    For L2-normalised vectors, inner product == cosine similarity.
    We normalise at embed time, so this gives exact cosine ranking.

    Wrapped in IndexIDMap so we can assign our own int64 IDs and use
    remove_ids() for document deletion.
    """
    global _index, _store, _id_map, _counter

    if FAISS_INDEX_PATH.exists() and FAISS_META_PATH.exists():
        _index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_META_PATH, "rb") as f:
            saved = pickle.load(f)
        _store   = saved.get("store",   {})
        _id_map  = saved.get("id_map",  {})
        _counter = saved.get("counter", 0)
        print(f"[FAISS] Loaded index with {_index.ntotal} vectors from disk.")
    else:
        base = faiss.IndexFlatIP(EMBED_DIM)      # exact cosine (with normalised vecs)
        _index = faiss.IndexIDMap(base)
        _store   = {}
        _id_map  = {}
        _counter = 0
        print(f"[FAISS] Created new index (dim={EMBED_DIM}).")


def _persist() -> None:
    """Write index and metadata to disk. Call inside _write_lock."""
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(_index, str(FAISS_INDEX_PATH))
    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump({
            "store":   _store,
            "id_map":  _id_map,
            "counter": _counter
        }, f)


# ─── Add Chunks ───────────────────────────────────────────────────────────────

def add_chunks(
    ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[Dict[str, Any]]
) -> None:
    """
    Add a batch of chunks to the FAISS index.

    Upsert behaviour: if a string_id already exists, replace it.
    This prevents duplicates if the same document is re-uploaded.

    Args:
        ids        : String chunk IDs e.g. ["uuid_0", "uuid_1", ...]
        embeddings : L2-normalised float vectors, shape (N, EMBED_DIM)
        documents  : Raw text of each chunk
        metadatas  : Metadata dicts (source_id, filename, page_number, etc.)
    """
    global _counter

    vectors = np.array(embeddings, dtype=np.float32)

    # Ensure vectors are L2-normalised (cosine = inner product for unit vecs)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    vectors = vectors / norms

    with _write_lock:
        int_ids = []
        for i, string_id in enumerate(ids):
            if string_id in _id_map:
                # Upsert: remove old entry first
                old_int = _id_map[string_id]
                _index.remove_ids(np.array([old_int], dtype=np.int64))
                del _store[old_int]

            new_int = _counter
            _counter += 1
            _id_map[string_id] = new_int
            _store[new_int] = {
                "string_id": string_id,
                "text":      documents[i],
                "metadata":  metadatas[i]
            }
            int_ids.append(new_int)

        int_ids_arr = np.array(int_ids, dtype=np.int64)
        _index.add_with_ids(vectors, int_ids_arr)
        _persist()


# ─── Similarity Search ────────────────────────────────────────────────────────

def similarity_search(
    query_embedding: List[float],
    source_ids: List[str],
    n_results: int = TOP_K_RESULTS
) -> List[Dict[str, Any]]:
    """
    Find the top-N most relevant chunks for a query across given source_ids.

    Strategy (FAISS has no metadata filter):
      1. Over-fetch FAISS_FETCH_K candidates from the full index
      2. Look up each returned int_id in _store
      3. Filter to keep only chunks whose source_id is in the requested list
      4. Return top n_results from the filtered set

    Args:
        query_embedding : L2-normalised query vector (same model as at ingest)
        source_ids      : List of document UUIDs to search within
        n_results       : Max chunks to return

    Returns:
        List of dicts: {text, metadata, score}  (score = cosine similarity)
    """
    if _index is None or _index.ntotal == 0:
        return []

    vec = np.array([query_embedding], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    # Over-fetch to ensure enough results after filtering
    fetch_k = min(FAISS_FETCH_K, _index.ntotal)
    scores, int_ids = _index.search(vec, fetch_k)

    source_id_set = set(source_ids)
    results = []

    for score, int_id in zip(scores[0], int_ids[0]):
        if int_id == -1:          # FAISS returns -1 for empty slots
            continue
        entry = _store.get(int(int_id))
        if entry is None:
            continue
        meta = entry["metadata"]
        if meta.get("source_id") not in source_id_set:
            continue

        results.append({
            "text":     entry["text"],
            "metadata": meta,
            "score":    float(score)  # cosine similarity, higher = better
        })

        if len(results) >= n_results:
            break

    return results


# ─── Get All Chunks for Summarisation ─────────────────────────────────────────

def get_all_chunks_for_source(source_id: str) -> List[Dict[str, Any]]:
    """
    Return ALL chunks for a document, sorted by chunk_index.
    Used by /summarize — needs full document in order, not just top-K.
    """
    chunks = []
    for entry in _store.values():
        if entry["metadata"].get("source_id") == source_id:
            chunks.append({
                "text":     entry["text"],
                "metadata": entry["metadata"]
            })

    chunks.sort(key=lambda c: c["metadata"].get("chunk_index", 0))
    return chunks


# ─── List All Sources ─────────────────────────────────────────────────────────

def get_all_sources() -> List[Dict[str, Any]]:
    """
    Return deduplicated list of all uploaded documents with metadata.
    Aggregates chunks by source_id to produce document-level records.
    """
    sources_map: Dict[str, Dict] = {}

    for entry in _store.values():
        meta = entry["metadata"]
        sid  = meta.get("source_id")
        if not sid:
            continue

        if sid not in sources_map:
            sources_map[sid] = {
                "source_id":   sid,
                "filename":    meta.get("filename", "Unknown"),
                "chunk_count": 0,
                "page_count":  meta.get("total_pages", 0)
            }
        sources_map[sid]["chunk_count"] += 1

    return list(sources_map.values())


# ─── Delete Document ──────────────────────────────────────────────────────────

def delete_source(source_id: str) -> int:
    """
    Delete all chunks belonging to a document from the FAISS index.

    Returns the number of chunks deleted.
    """
    to_delete_int_ids = []
    to_delete_str_ids = []

    for entry in list(_store.values()):
        if entry["metadata"].get("source_id") == source_id:
            int_id    = _id_map[entry["string_id"]]
            to_delete_int_ids.append(int_id)
            to_delete_str_ids.append(entry["string_id"])

    if not to_delete_int_ids:
        return 0

    with _write_lock:
        ids_arr = np.array(to_delete_int_ids, dtype=np.int64)
        _index.remove_ids(ids_arr)
        for int_id, str_id in zip(to_delete_int_ids, to_delete_str_ids):
            del _store[int_id]
            del _id_map[str_id]
        _persist()

    return len(to_delete_int_ids)
