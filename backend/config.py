"""
config.py — Central configuration for the entire backend.

Single source of truth for all constants — models, paths, chunking,
retrieval settings, and logging. Edit here; changes propagate everywhere.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()  # Load overrides from .env if present

# ─── Base Directory ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent   # project root

# ─── Storage Paths ────────────────────────────────────────────────────────────
UPLOAD_DIR   = BASE_DIR / "storage" / "uploads"   # raw uploaded files
FAISS_DIR    = BASE_DIR / "storage" / "faiss"      # FAISS index + metadata
BM25_DIR     = BASE_DIR / "storage" / "bm25"       # BM25 corpus pickles
LOG_DIR      = BASE_DIR / "storage" / "logs"       # structured log files

# FAISS index files — always kept in sync
FAISS_INDEX_PATH = FAISS_DIR / "documents.index"   # binary FAISS index
FAISS_META_PATH  = FAISS_DIR / "documents_meta.pkl" # parallel metadata list

# Create all directories on startup
for _d in [UPLOAD_DIR, FAISS_DIR, BM25_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── LLM (Ollama — for chat & summarization only) ────────────────────────────
CHAT_MODEL       = os.getenv("CHAT_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_KEEP_ALIVE = "5m"   # keep model warm in RAM between requests

# ─── Embeddings (sentence-transformers — local, no Ollama needed) ────────────
# BGE-M3: multi-lingual, 1024-dim, state-of-the-art retrieval performance
# First run will download ~2.2 GB model weights automatically.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM   = 1024   # BGE-M3 output dimension

# ─── Chunking ─────────────────────────────────────────────────────────────────
# 300 tokens: tight enough to pinpoint specific facts, enough for context
# 30 overlap: light overlap to avoid splitting mid-sentence
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 30

# ─── FAISS Retrieval ──────────────────────────────────────────────────────────
# FAISS has no native metadata filter — we over-fetch then filter in Python.
# Fetch 50 candidates from FAISS, filter by source_id, return top TOP_K_RESULTS.
FAISS_FETCH_K  = 50    # how many candidates to pull from FAISS
TOP_K_RESULTS  = 5     # final chunks passed to LLM after filtering

# ─── Hybrid Search (BM25 + FAISS) ────────────────────────────────────────────
RETRIEVAL_MODE  = "hybrid"   # "hybrid" | "dense" | "sparse"
BM25_WEIGHT     = 0.4        # weight for BM25 score in RRF fusion
VECTOR_WEIGHT   = 0.6        # weight for FAISS score in RRF fusion
RRF_K           = 60         # RRF constant (standard default)

# ─── Phase 2 — Re-ranking ─────────────────────────────────────────────────────
RERANK_MODEL       = os.getenv("RERANK_MODEL",
                               "cross-encoder/ms-marco-MiniLM-L-6-v2")
MIN_RERANK_SCORE   = -5.0    # hard cutoff: logit below this is noise (ms-marco range ~-10 to +10)
TOP_K_RERANK_INPUT = 20      # over-fetch before cross-encoder (then filter to TOP_K_RESULTS)

# ─── Phase 2 — Short-Term Memory (STM Context Assembly) ────────────────────────
STM_TOKEN_BUDGET    = 3000   # max tokens in assembled context sent to LLM
STM_DEDUP_THRESHOLD = 0.95   # cosine above this → chunks are near-duplicates

# ─── Phase 2 — Multi-hop Retrieval ───────────────────────────────────────────────
MAX_HOPS = 3    # maximum retrieval hops for multi-hop queries

# ─── Phase 3 — Conversation Memory ────────────────────────────────────────────
SESSIONS_DB_PATH = BASE_DIR / "storage" / "sessions.db"   # SQLite session store
STM_TURNS        = 5      # last N user+assistant pairs to inject into prompt
STM_MAX_TOKENS   = 800    # token budget for the STM block (avoids prompt bloat)
SUMMARIZE_AFTER  = 10     # total messages before triggering background summarization

# ─── CORS ─────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = ["http://localhost:8501"]
