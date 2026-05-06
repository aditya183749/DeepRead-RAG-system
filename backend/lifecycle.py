"""
lifecycle.py — Document Lifecycle Management helpers.

Provides version comparison (diff) between two document versions
by comparing their indexed chunks in the BM25 store.
"""
from typing import Dict, Any, List
import difflib

from logger import get_logger
import bm25_store
import memory.db as mem_db

log = get_logger("lifecycle")


def compare_versions(
    source_id_old: str,
    source_id_new: str,
) -> Dict[str, Any]:
    """
    Compare chunks between two versions of a document.

    Strategy:
      - Pull all chunk texts for each source_id from BM25 corpus
      - Compute symmetric difference: added, removed, and a unified diff summary
      - Return structured diff result

    Returns:
        {
          "old_version": { source_id, filename, version, chunk_count },
          "new_version": { source_id, filename, version, chunk_count },
          "added_chunks": int,
          "removed_chunks": int,
          "unchanged_chunks": int,
          "diff_summary": [ "+added text...", "-removed text..." ]
        }
    """
    # Get metadata from DB
    old_meta = mem_db.get_version_by_source_id(source_id_old)
    new_meta = mem_db.get_version_by_source_id(source_id_new)

    if not old_meta or not new_meta:
        return {"error": "One or both source_ids not found in document_versions."}

    # Pull chunk texts from BM25 corpus
    def _get_texts(source_id: str) -> List[str]:
        results = bm25_store.bm25_search("", [source_id], n_results=10000)
        # bm25_search with empty query returns 0 scores — filter by source_id directly
        corpus_texts = []
        for i, meta in enumerate(bm25_store._metadatas):
            if meta.get("source_id") == source_id:
                corpus_texts.append(bm25_store._corpus[i])
        return corpus_texts

    old_texts = set(_get_texts(source_id_old))
    new_texts = set(_get_texts(source_id_new))

    added   = new_texts - old_texts
    removed = old_texts - new_texts
    kept    = old_texts & new_texts

    # Build a brief unified diff summary (first 300 chars of each changed chunk)
    diff_summary = []
    for t in list(removed)[:5]:
        diff_summary.append(f"- {t[:200].replace(chr(10), ' ')}")
    for t in list(added)[:5]:
        diff_summary.append(f"+ {t[:200].replace(chr(10), ' ')}")

    log.info("version_diff",
        source_id_old=source_id_old, source_id_new=source_id_new,
        added=len(added), removed=len(removed), unchanged=len(kept)
    )

    return {
        "old_version": {
            "source_id": source_id_old,
            "filename":  old_meta["filename"],
            "version":   old_meta["version"],
            "chunk_count": old_meta["chunk_count"]
        },
        "new_version": {
            "source_id": source_id_new,
            "filename":  new_meta["filename"],
            "version":   new_meta["version"],
            "chunk_count": new_meta["chunk_count"]
        },
        "added_chunks":     len(added),
        "removed_chunks":   len(removed),
        "unchanged_chunks": len(kept),
        "diff_summary":     diff_summary,
    }
