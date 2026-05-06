# RAG System — Project Documentation

**Project Name:** Local Document Intelligence (RAG System)  
**Version:** 3.0 (with Document Lifecycle Management)  
**Stack:** Python · FastAPI · Streamlit · Ollama · FAISS · BM25 · SQLite  
**Architecture:** Fully local, no cloud dependencies, no data leaves the machine

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Module Descriptions](#5-module-descriptions)
6. [RAG Pipeline — Step by Step](#6-rag-pipeline--step-by-step)
7. [Document Lifecycle Management (DLM)](#7-document-lifecycle-management-dlm)
8. [Observability & Metrics](#8-observability--metrics)
9. [API Endpoints](#9-api-endpoints)
10. [Database Schema](#10-database-schema)
11. [Supported File Formats](#11-supported-file-formats)
12. [How to Run](#12-how-to-run)
13. [Project Folder Structure](#13-project-folder-structure)

---

## 1. Project Overview

This project is a **fully local, private document Q&A system** built using Retrieval-Augmented Generation (RAG). Users can upload any document (PDF, Word, Excel, PowerPoint, CSV, HTML, etc.) and ask natural-language questions about its content. The system retrieves the most relevant sections of the document and generates a grounded, cited answer using a locally running Large Language Model (LLM).

**Key design principles:**
- Everything runs on the user's own machine — no API keys, no internet required for inference
- Multi-document support with source filtering
- Full conversation memory across sessions
- Document versioning and lifecycle management
- Comprehensive observability through structured logging

---

## 2. Key Features

### Core Capabilities
| Feature | Description |
|---|---|
| Multi-format document ingestion | PDF, DOCX, DOC, TXT, MD, CSV, XLSX, PPTX, HTML |
| Hybrid retrieval | BM25 (keyword) + FAISS (semantic vector) search combined via RRF |
| Cross-encoder re-ranking | ms-marco MiniLM cross-encoder removes noise from retrieved chunks |
| Query rewriting | LLM expands the user query into variants, generates HyDE embeddings |
| Streaming chat | Real-time token-by-token response via Server-Sent Events (SSE) |
| Citation tracking | Every answer cites [filename, Page N] from source chunks |
| Conversation memory | Last N turns injected as context; older history compressed by LLM |
| Multi-session history | All sessions stored in SQLite; resumable from sidebar |
| Image analysis | LLaVA multimodal model analyzes charts, graphs, and image-heavy PDF pages |
| Document summarization | One-click structured summary of any uploaded document |

### Advanced Features
| Feature | Description |
|---|---|
| Document Lifecycle Management | Version tracking, archive/restore, chunk-level diffing between versions |
| Performance observability | Latency breakdown, retrieval hit rate, tokens/sec, grounding %, hallucination flag |
| Session summarization | Long conversations auto-compressed by LLM to prevent context bloat |
| Noise filter | Chunks below reranker score threshold are automatically excluded |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────┐
│              FRONTEND (Streamlit :8501)              │
│                                                     │
│  ┌─────────────────────┐  ┌────────────────────┐   │
│  │   Sidebar            │  │   Chat Window      │   │
│  │  - File Upload       │  │  - Streaming tokens│   │
│  │  - Document List     │  │  - Citations panel │   │
│  │  - Version badges    │  │  - Session history │   │
│  │  - Archive/Restore   │  │                    │   │
│  │  - Diff viewer       │  │                    │   │
│  └─────────────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────┘
                          │ HTTP / SSE
                          ▼
┌─────────────────────────────────────────────────────┐
│              BACKEND (FastAPI :8000)                 │
│                                                     │
│  /upload ──► ingestion.py ──► FAISS + BM25 + DLM   │
│  /chat   ──► Query Rewrite ──► Hybrid Search        │
│              ──► Reranker ──► Context Assembly       │
│              ──► Ollama LLM (streaming)             │
│  /documents ──► lifecycle.py ──► SQLite DLM         │
│  /sessions ──► memory/db.py ──► SQLite              │
└─────────────────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
    ┌─────────┐     ┌──────────┐     ┌──────────┐
    │  FAISS  │     │   BM25   │     │  SQLite  │
    │  Index  │     │  Corpus  │     │    DB    │
    │(vectors)│     │(keywords)│     │(sessions │
    │         │     │         │     │  + DLM)  │
    └─────────┘     └──────────┘     └──────────┘
                          │
                          ▼
                   ┌──────────┐
                   │  Ollama  │
                   │ llama3.2 │
                   │  llava   │
                   └──────────┘
```

---

## 4. Technology Stack

| Component | Technology | Version |
|---|---|---|
| Backend API | FastAPI | 0.110.0 |
| Frontend UI | Streamlit | 1.32.2 |
| HTTP Client | httpx | 0.27.0 |
| LLM Runtime | Ollama | 0.1.6 |
| Chat Model | llama3.2:3b | — |
| Vision Model | llava:latest | — |
| Embedding Model | BAAI/bge-m3 (1024-dim) | via sentence-transformers 2.5.1 |
| Reranker | ms-marco-MiniLM-L-6-v2 | via sentence-transformers |
| Vector Store | FAISS (CPU) | 1.8.0 |
| Sparse Retrieval | BM25Okapi | rank_bm25 0.2.2 |
| Database | SQLite | stdlib |
| Structured Logging | structlog | 24.1.0 |
| Data Validation | Pydantic | 2.6.3 |
| Token Counting | tiktoken | 0.6.0 |
| PDF Extraction | PyMuPDF (fitz) | 1.23.26 |
| Word Extraction | python-docx | 1.1.0 |
| Excel Extraction | openpyxl | 3.1.2 |
| PowerPoint | python-pptx | 0.6.23 |
| HTML Extraction | beautifulsoup4 | 4.12.3 |

---

## 5. Module Descriptions

### Backend

| File | Purpose |
|---|---|
| `main.py` | FastAPI app entry point. Orchestrates all pipeline stages. Defines all API endpoints. |
| `config.py` | Central configuration: model names, chunk sizes, retrieval settings, paths. |
| `schemas.py` | Pydantic request/response models for all API endpoints. |
| `ingestion.py` | Multi-format document extractor. Splits into chunks, generates embeddings, indexes in FAISS + BM25. |
| `vector_store.py` | FAISS vector index wrapper. Handles add, search, delete, persistence. |
| `bm25_store.py` | BM25 keyword index wrapper. Handles corpus management and persistence. |
| `llm.py` | Prompt builder + Ollama streaming client. Emits tokens, usage stats, and citations as SSE. |
| `lifecycle.py` | Document Lifecycle Management helpers. Compares chunk sets between document versions. |
| `logger.py` | Configures structlog with JSONL output to `storage/logs/`. |

### Backend — `memory/`

| File | Purpose |
|---|---|
| `db.py` | SQLite layer for sessions, messages, summaries, and DLM tables. |
| `stm.py` | Short-Term Memory builder. Injects last N turns into LLM prompt. Triggers summarization. |

### Backend — `retrieval/`

| File | Purpose |
|---|---|
| `query_rewriter.py` | LLM-based query expansion. Generates HyDE embeddings, query variants, and query type classification. |
| `rerank.py` | Cross-encoder re-ranking pipeline. Applies noise filter, logs score distributions. |
| `context_assembly.py` | Deduplicates retrieved chunks, applies token budget, sorts by relevance. |

### Frontend

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit UI. Sidebar (upload, DLM panel) + main chat window. |
| `api_client.py` | All HTTP calls to the backend. One function per endpoint. |

---

## 6. RAG Pipeline — Step by Step

When a user asks a question, the following 7-step pipeline runs:

```
User Question
      │
      ▼
Step 1: SESSION MEMORY
   Load existing session or create new one.
   Fetch last N (=5) conversation turns as STM context.
      │
      ▼
Step 2: QUERY REWRITING (via LLM)
   - Expand into 3-4 query variants
   - Generate HyDE (Hypothetical Document Embedding)
   - Classify query type: factual / comparative / multi-hop
      │
      ▼
Step 3: HYBRID RETRIEVAL (BM25 + FAISS → RRF)
   For each query variant + HyDE embedding:
   - FAISS semantic search (dense, top-50 candidates)
   - BM25 keyword search (sparse)
   - Reciprocal Rank Fusion (RRF) to merge results
   - Deduplicate across all variants
      │
      ▼
Step 4: CROSS-ENCODER RE-RANKING
   - Score all candidate chunks against the query
   - Filter chunks below score threshold (-5.0)
   - Return top-K (=5) highest-scored chunks
      │
      ▼
Step 5: CONTEXT ASSEMBLY
   - Deduplicate near-identical chunks (cosine > 0.95)
   - Apply 3000-token budget
   - Sort by rerank score
      │
      ▼
Step 6: STREAMING LLM RESPONSE
   - Build prompt: system rules + STM history + context chunks + question
   - Stream tokens via Ollama SSE
   - Emit usage stats (tokens/sec, prompt/completion counts)
   - Emit citations as final event
      │
      ▼
Step 7: PERSIST
   - Save user Q + assistant A to SQLite
   - Log all metrics to JSONL files
   - Trigger LLM session summarization if session > 10 messages
```

---

## 7. Document Lifecycle Management (DLM)

The DLM system tracks every version of every uploaded document.

### How Versioning Works

1. User uploads `report.pdf` → registered as **v1 (active)**
2. User uploads `report.pdf` again (updated version) → v1 auto-archived, **v2 (active)** created
3. Search only uses **active** documents; archived docs are excluded automatically

### Status States

| Status | Meaning |
|---|---|
| `active` | Document is indexed and available for search/chat |
| `archived` | Hidden from search, history preserved |
| `expired` | Programmatically retired (future use) |

### Diff Engine

When comparing two versions of a document, the system:
- Pulls all text chunks for each version from the BM25 corpus
- Computes set difference: **added**, **removed**, **unchanged** chunks
- Returns a sample diff of up to 5 added + 5 removed chunk excerpts

### DLM API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/documents` | List all documents grouped by name with version history |
| PATCH | `/documents/{source_id}/status` | Archive, expire, or restore a document version |
| GET | `/documents/{old_id}/{new_id}/diff` | Compare chunks between two versions |

---

## 8. Observability & Metrics

All metrics are logged as structured JSON (JSONL) in `storage/logs/`.

### Files

| File | Contents |
|---|---|
| `retrieval.jsonl` | Per-query: latency breakdown, reranker scores, retrieval hit rate |
| `tokens.jsonl` | Per-query: token counts, tokens/sec, grounding %, hallucination flag |
| `app.jsonl` | All application events (uploads, sessions, errors) |
| `sessions.jsonl` | Session lifecycle events |
| `chunks.jsonl` | Per-chunk indexing events |

### Metrics Tracked

| Metric | Description |
|---|---|
| **Latency Breakdown** | Time for: rewrite, retrieval, reranking, context assembly, total |
| **Retrieval Hit Rate** | Did at least 1 chunk pass the reranker noise threshold? |
| **Reranker Score Distribution** | min, max, mean, median, p25, p75 for all candidates + kept chunks |
| **Tokens/Second** | LLM generation speed from Ollama's `eval_duration` |
| **Source Grounding %** | % of response sentences that contain a citation bracket |
| **Hallucination Flag** | Boolean: did the LLM use a refusal phrase (context insufficient)? |

---

## 9. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/upload` | Upload and index a document (SSE streaming progress) |
| POST | `/chat` | Ask a question (SSE streaming response) |
| POST | `/summarize` | Generate structured document summary |
| POST | `/analyze-images` | Run LLaVA on image-heavy PDF pages (SSE) |
| GET | `/sources` | List all indexed document sources |
| DELETE | `/sources/{source_id}` | Remove a document from the index |
| GET | `/documents` | List documents with full version history (DLM) |
| PATCH | `/documents/{source_id}/status` | Change document lifecycle status |
| GET | `/documents/{old}/{new}/diff` | Compare two document versions |
| GET | `/sessions` | List all conversation sessions |
| GET | `/sessions/{session_id}` | Get full message history for a session |
| DELETE | `/sessions/{session_id}` | Delete a session |

---

## 10. Database Schema

### SQLite — `storage/sessions.db`

```sql
-- Conversation Sessions
sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    source_ids  TEXT,  -- JSON list of document UUIDs
    created_at  TEXT,
    updated_at  TEXT
)

-- Chat Messages
messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT REFERENCES sessions(id),
    role        TEXT,  -- 'user' | 'assistant'
    content     TEXT,
    created_at  TEXT
)

-- LLM-Compressed Session History
session_summaries (
    session_id              TEXT PRIMARY KEY,
    summary                 TEXT,
    summarized_up_to_msg_id INTEGER,
    updated_at              TEXT
)

-- Document Identity (stable across versions)
document_registry (
    doc_key     TEXT PRIMARY KEY,  -- normalized filename hash
    name        TEXT,              -- display name
    created_at  TEXT,
    updated_at  TEXT
)

-- Document Versions
document_versions (
    source_id   TEXT PRIMARY KEY,   -- UUID used in FAISS/BM25
    doc_key     TEXT REFERENCES document_registry(doc_key),
    filename    TEXT,
    version     INTEGER,
    status      TEXT,  -- 'active' | 'archived' | 'expired'
    chunk_count INTEGER,
    file_size   INTEGER,
    uploaded_at TEXT,
    archived_at TEXT,
    notes       TEXT
)
```

---

## 11. Supported File Formats

| Format | Extension | Extractor |
|---|---|---|
| PDF | `.pdf` | PyMuPDF (fitz) |
| Word | `.docx`, `.doc` | python-docx |
| Excel | `.xlsx`, `.xls` | openpyxl |
| PowerPoint | `.pptx` | python-pptx |
| Text | `.txt`, `.md` | built-in |
| CSV | `.csv` | csv stdlib |
| HTML | `.html`, `.htm` | beautifulsoup4 |

---

## 12. How to Run

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Pull required models:
  ```
  ollama pull llama3.2:3b
  ollama pull llava
  ```

### Install Dependencies

```bash
cd "d:\Rag System"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Start Backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start Frontend

```bash
streamlit run frontend/app.py
```

Open browser at: `http://localhost:8501`

---

## 13. Project Folder Structure

```
Rag System/
│
├── backend/
│   ├── main.py                  ← FastAPI app, all endpoints
│   ├── config.py                ← All configuration constants
│   ├── schemas.py               ← Pydantic request/response models
│   ├── ingestion.py             ← Document parsing + chunking + embedding
│   ├── vector_store.py          ← FAISS vector index
│   ├── bm25_store.py            ← BM25 keyword index
│   ├── llm.py                   ← Ollama LLM client + prompt builder
│   ├── lifecycle.py             ← DLM diff engine
│   ├── logger.py                ← Structured JSONL logging
│   │
│   ├── memory/
│   │   ├── db.py                ← SQLite persistence (sessions + DLM)
│   │   └── stm.py               ← Short-Term Memory builder
│   │
│   └── retrieval/
│       ├── query_rewriter.py    ← Query expansion + HyDE + classification
│       ├── rerank.py            ← Cross-encoder re-ranking
│       └── context_assembly.py  ← Dedup + token budget management
│
├── frontend/
│   ├── app.py                   ← Streamlit UI (sidebar + chat window)
│   └── api_client.py            ← All HTTP calls to backend
│
├── storage/
│   ├── sessions.db              ← SQLite database
│   ├── uploads/                 ← Raw uploaded files
│   ├── faiss/                   ← FAISS vector index (binary)
│   ├── bm25/                    ← BM25 corpus (pickle)
│   └── logs/
│       ├── retrieval.jsonl      ← Retrieval + reranking metrics
│       ├── tokens.jsonl         ← Token usage + grounding metrics
│       ├── app.jsonl            ← All application events
│       └── sessions.jsonl       ← Session lifecycle events
│
├── requirements.txt
├── README.md
└── PROJECT_DOCUMENTATION.md    ← This file
```

---

*Documentation prepared for project handover and demo.*
