"""
main.py — FastAPI application entry point (Phase 3: Conversation Memory).

Full /chat pipeline:
  1. Session Memory     — Create/load session, build STM context block
  2. Query Rewriting    — LLM expands variants, HyDE, classifies type
  3. Hybrid Retrieval  — BM25 + FAISS RRF across all query variants
  4. Re-ranking        — Cross-encoder + noise filter
  5. Context Assembly  — Dedup, sort, token budget
  6. Streaming LLM     — Grounded prompt + STM + inline citations → SSE
  7. Persist Message   — Save user Q + assistant A to SQLite

HOW TO RUN:
  uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import sys
import os

# ── Speed up startup: skip HuggingFace version-check round-trips ──────────────
# Without this, every startup makes ~20 HTTP calls to huggingface.co to verify
# model versions, adding 1-2 minutes to startup time. Models are cached locally.
os.environ.setdefault("HF_HUB_OFFLINE", "1")               # use cached weights
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")    # suppress warning
# ─────────────────────────────────────────────────────────────────────────────

import uuid
import shutil
import time
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import (
    UPLOAD_DIR, ALLOWED_ORIGINS, EMBED_MODEL,
    TOP_K_RESULTS, RRF_K, RETRIEVAL_MODE,
    BM25_WEIGHT, VECTOR_WEIGHT, TOP_K_RERANK_INPUT
)
from schemas import (
    UploadResponse, ChatRequest,
    SummarizeRequest, SummarizeResponse,
    SourcesResponse, SourceItem,
    SessionItem, SessionsResponse,
    SessionHistoryResponse, MessageItem
)
from ingestion import ingest_document, get_embed_model, generate_embeddings
import vector_store
import bm25_store
from llm import stream_rag_answer, generate_summary, summarize_session, get_ollama_client
from retrieval.rerank import rerank_chunks, get_cross_encoder
from retrieval.context_assembly import assemble_context
from retrieval.query_rewriter import rewrite_query
from logger import configure_logging, get_logger
import memory.db as mem_db
from memory.stm import build_stm_block, should_summarize
from lifecycle import compare_versions

log          = get_logger("app")
log_sessions = get_logger("sessions")
log_tokens   = get_logger("tokens")
log_chunks   = get_logger("chunks")


# ─── Hybrid Search (BM25 + FAISS → RRF) ─────────────────────────────────────

def _hybrid_search(
    question: str,
    query_embedding: list,
    source_ids: list,
    n_results: int = TOP_K_RERANK_INPUT
) -> list:
    """
    Reciprocal Rank Fusion of FAISS (dense) and BM25 (sparse) results.
    Designed to be called with ANY query embedding (original, HyDE, or variant).
    """
    dense_results  = vector_store.similarity_search(
        query_embedding=query_embedding,
        source_ids=source_ids,
        n_results=n_results
    )

    if RETRIEVAL_MODE == "dense":
        return dense_results

    sparse_results = bm25_store.bm25_search(
        query=question,
        source_ids=source_ids,
        n_results=n_results
    )

    if RETRIEVAL_MODE == "sparse":
        return sparse_results

    # RRF Fusion
    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict]  = {}

    for rank, chunk in enumerate(dense_results):
        key = chunk["text"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + VECTOR_WEIGHT / (rank + RRF_K)
        chunk_map[key] = chunk

    for rank, chunk in enumerate(sparse_results):
        key = chunk["text"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + BM25_WEIGHT / (rank + RRF_K)
        chunk_map[key] = chunk

    ranked_keys = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
    merged = []
    for key in ranked_keys[:n_results]:
        chunk = chunk_map[key].copy()
        chunk["score"] = rrf_scores[key]
        merged.append(chunk)

    return merged


def _full_retrieval(
    question: str,
    source_ids: list,
    rewrite: bool = True
) -> tuple[list, str, str]:
    """
    Full Phase 2 retrieval pipeline.
    Returns (final_chunks, context_block, query_type).

    Steps:
      1. Query Rewriting (optional, can be disabled via rewrite=False)
      2. Hybrid search across all query variants + HyDE embedding
      3. Deduplicate cross-variant results
      4. Cross-encoder re-ranking + noise filter
      5. STM context assembly
    """
    query_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    # ── Step 1: Rewrite ───────────────────────────────────────────────────────
    t_rewrite_start = time.perf_counter()
    if rewrite:
        rw = rewrite_query(question)
        query_type   = rw.query_type
        hyde_emb     = rw.hyde_embedding
        variants     = rw.variants
        display_q    = variants[0]  # rewritten primary question
    else:
        hyde_emb   = generate_embeddings([question])[0]
        query_type = "factual"
        variants   = [question]
        display_q  = question
    rewrite_ms = round((time.perf_counter() - t_rewrite_start) * 1000, 1)

    # ── Step 2: Multi-variant hybrid search ───────────────────────────────────
    t_retrieval_start = time.perf_counter()
    all_chunks: dict[str, dict] = {}

    # HyDE embedding retrieval (most powerful)
    for chunk in _hybrid_search(question, hyde_emb, source_ids):
        all_chunks[chunk["text"]] = chunk

    # Each variant text retrieval
    for variant in variants[:3]:   # limit to 3 variants to avoid latency blowup
        v_emb = generate_embeddings([variant])[0]
        for chunk in _hybrid_search(variant, v_emb, source_ids):
            if chunk["text"] not in all_chunks:
                all_chunks[chunk["text"]] = chunk

    candidate_chunks = list(all_chunks.values())
    retrieval_ms = round((time.perf_counter() - t_retrieval_start) * 1000, 1)

    # ── Step 3: Re-rank + noise filter ────────────────────────────────────────
    t_rerank_start = time.perf_counter()
    reranked = rerank_chunks(
        query=display_q,
        chunks=candidate_chunks,
        top_n=TOP_K_RESULTS
    )
    rerank_ms = round((time.perf_counter() - t_rerank_start) * 1000, 1)

    # ── Step 4: STM context assembly ──────────────────────────────────────────
    t_assembly_start = time.perf_counter()
    context_block, final_chunks = assemble_context(reranked)
    assembly_ms = round((time.perf_counter() - t_assembly_start) * 1000, 1)

    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Retrieval hit rate signal: did at least 1 chunk pass reranking threshold?
    retrieval_hit = len(reranked) > 0

    log.info("retrieval_pipeline_complete",
        query_id=query_id,
        original_query=question,
        rewritten=display_q,
        query_type=query_type,
        candidates=len(candidate_chunks),
        after_rerank=len(reranked),
        final=len(final_chunks),
        retrieval_hit=retrieval_hit,
        latency_breakdown_ms={
            "rewrite":   rewrite_ms,
            "retrieval": retrieval_ms,
            "reranking": rerank_ms,
            "assembly":  assembly_ms,
            "total":     total_ms
        }
    )

    return final_chunks, context_block, query_type


# ─── App Lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all singletons at startup."""
    configure_logging()
    mem_db.initialize_db()        # Phase 3: create SQLite tables
    vector_store.initialize()
    bm25_store.initialize()
    get_embed_model()             # pre-warm BGE-M3
    get_cross_encoder()           # pre-warm cross-encoder (~25MB, fast)
    log.info("startup_complete", embed_model=EMBED_MODEL,
             vector_db="faiss", retrieval=RETRIEVAL_MODE)
    yield
    log.info("shutdown")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise RAG API",
    description="Phase 3: Conversation Memory — Session-aware RAG with STM",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status":     "ok",
        "version":    "3.0.0",
        "embed_model": EMBED_MODEL,
        "retrieval":  RETRIEVAL_MODE,
        "reranker":   "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "pipeline":   ["session_memory", "rewrite", "hybrid_rrf", "cross_encoder", "stm_assembly", "llm", "persist"]
    }


# ─── POST /upload (streaming progress) ───────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Receive PDF/DOCX → stream SSE progress events during ingestion.
    Events: {"type":"progress","step":"chunked","detail":"Created 142 chunks"}
            {"type":"complete","source_id":"…","chunk_count":142,"page_count":12}
            {"type":"error","detail":"…"}
    """
    allowed = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".xlsx", ".xls", ".pptx", ".html", ".htm"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Allowed: PDF, DOCX, TXT, MD, CSV, XLSX, PPTX, HTML."
        )

    temp_path = UPLOAD_DIR / f"tmp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    import asyncio as _asyncio
    import json as _json
    from concurrent.futures import ThreadPoolExecutor

    _queue: _asyncio.Queue = _asyncio.Queue()
    _loop = _asyncio.get_event_loop()
    _fname = file.filename   # capture before file object closes

    def _progress_cb(step: str, detail: str):
        _loop.call_soon_threadsafe(_queue.put_nowait, {"step": step, "detail": detail})

    def _run_ingest():
        try:
            source_id, chunk_count, page_count = ingest_document(
                file_path=temp_path,
                filename=_fname,
                progress_cb=_progress_cb
            )
            final_path = UPLOAD_DIR / f"{source_id}_{_fname}"
            temp_path.rename(final_path)
            # Register in document lifecycle management
            file_size = final_path.stat().st_size
            mem_db.register_document(
                source_id=source_id,
                filename=_fname,
                chunk_count=chunk_count,
                file_size=file_size,
            )
            log.info("upload_success", filename=_fname,
                     source_id=source_id, chunks=chunk_count, pages=page_count)
            _loop.call_soon_threadsafe(_queue.put_nowait, {
                "done": True, "source_id": source_id,
                "chunk_count": chunk_count, "page_count": page_count
            })
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            log.error("upload_failed", filename=_fname, error=str(e))
            _loop.call_soon_threadsafe(_queue.put_nowait, {"error": str(e)})

    async def _event_stream():
        _loop.run_in_executor(ThreadPoolExecutor(max_workers=1), _run_ingest)
        while True:
            msg = await _queue.get()
            if "error" in msg:
                yield f"data: {_json.dumps({'type': 'error', 'detail': msg['error']})}\n\n"
                break
            elif "done" in msg:
                yield f"data: {_json.dumps({'type': 'complete', 'source_id': msg['source_id'], 'chunk_count': msg['chunk_count'], 'page_count': msg['page_count'], 'filename': _fname})}\n\n"
                break
            else:
                yield f"data: {_json.dumps({'type': 'progress', 'step': msg['step'], 'detail': msg['detail']})}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ─── POST /chat ───────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Full Phase 3 pipeline:
      session_memory → query_rewriter → hybrid_rrf → cross_encoder → stm_assembly → stream_llm → persist
    """
    query_start_time = time.perf_counter()

    # ── Step 1: Session setup ───────────────────────────────────────────────
    session_id = request.session_id
    is_new_session = session_id is None

    if is_new_session:
        session_id = mem_db.create_session(
            source_ids=request.source_ids,
            title=request.question
        )
        log_sessions.info("session_created",
            session_id=session_id, question=request.question[:120],
            source_ids=request.source_ids)
    else:
        if not mem_db.get_session(session_id):
            session_id = mem_db.create_session(
                source_ids=request.source_ids,
                title=request.question
            )
            is_new_session = True
            log_sessions.info("session_created_recovery", session_id=session_id)
        else:
            log_sessions.info("session_resumed", session_id=session_id,
                question=request.question[:120])

    log_sessions.info("query_start", session_id=session_id,
        question=request.question[:200], num_sources=len(request.source_ids))

    # ── Step 2: Build STM context block ─────────────────────────────────────
    stm_block = build_stm_block(session_id)
    stm_tokens = len(stm_block.split()) if stm_block else 0
    log.debug("stm_block_built", session_id=session_id,
        has_stm=bool(stm_block), stm_tokens=stm_tokens, is_new=is_new_session)
    log_tokens.debug("stm_tokens", session_id=session_id,
        stage="stm_build", stm_tokens=stm_tokens, has_stm=bool(stm_block))

    # ── Step 3: Full retrieval pipeline ─────────────────────────────────────
    try:
        final_chunks, context_block, query_type = _full_retrieval(
            question=request.question,
            source_ids=request.source_ids,
            rewrite=request.rewrite_query
        )
    except Exception as e:
        log.error("retrieval_error", error=str(e), query=request.question)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    if not final_chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found in the selected documents."
        )

    # Log each final chunk (score, page, filename, snippet)
    for rank, chunk in enumerate(final_chunks):
        meta = chunk.get("metadata", {})
        log_chunks.debug("chunk_selected",
            session_id=session_id,
            query=request.question[:100],
            rank=rank + 1,
            score=round(float(chunk.get("score", 0)), 4),
            filename=meta.get("filename", ""),
            page=meta.get("page_number", "?"),
            chars=len(chunk.get("text", "")),
            snippet=chunk.get("text", "")[:150].replace("\n", " ")
        )

    # Log prompt token estimate
    context_tokens = len(context_block.split())
    question_tokens = len(request.question.split())
    stm_tokens = len(stm_block.split()) if stm_block else 0
    total_input_tokens = 300 + stm_tokens + context_tokens + question_tokens  # 300 = system prompt est
    log_tokens.info("prompt_tokens",
        session_id=session_id,
        stage="prompt_build",
        system_tokens=300,
        stm_tokens=stm_tokens,
        context_tokens=context_tokens,
        question_tokens=question_tokens,
        total_input_tokens=total_input_tokens,
        budget=3000,
        chunks_used=len(final_chunks),
        query_type=query_type
    )

    # ── Step 4: Stream + persist ──────────────────────────────────────────────
    user_msg_id = mem_db.append_message(session_id, "user", request.question)
    log_sessions.info("message_saved", session_id=session_id,
        role="user", msg_len=len(request.question))

    async def _streaming_with_persistence():
        full_response_parts = []
        real_prompt_tokens     = 0
        real_completion_tokens = 0
        real_tokens_per_sec    = 0.0
        import json as _json
        yield f"data: {_json.dumps({'type': 'session_id', 'session_id': session_id})}\n\n"

        for chunk_event in stream_rag_answer(
            question=request.question,
            context_block=context_block,
            final_chunks=final_chunks,
            stm_block=stm_block
        ):
            try:
                ev = _json.loads(chunk_event.replace("data: ", "", 1).strip())
                ev_type = ev.get("type")
                if ev_type == "token":
                    full_response_parts.append(ev.get("content", ""))
                elif ev_type == "usage":
                    # Real token counts from Ollama — not estimates
                    real_prompt_tokens     = ev.get("prompt_tokens", 0)
                    real_completion_tokens = ev.get("completion_tokens", 0)
                    real_tokens_per_sec    = ev.get("tokens_per_second", 0.0)
            except Exception:
                pass
            yield chunk_event

        # Persist assistant message
        full_response = "".join(full_response_parts)
        if full_response:
            mem_db.append_message(session_id, "assistant", full_response)
            total_ms = round((time.perf_counter() - query_start_time) * 1000, 1)

            # ── Source Grounding %: fraction of sentences that cite a source ─
            import re as _re
            sentences = [s.strip() for s in _re.split(r'[.!?]', full_response) if s.strip()]
            grounded  = [s for s in sentences if "[SOURCE" in s or "[Page" in s or "[" in s]
            source_grounding_pct = round(len(grounded) / len(sentences) * 100, 1) if sentences else 0.0

            # ── Hallucination Flag: did the LLM refuse due to missing context? ─
            REFUSAL_PHRASES = [
                "could not find", "cannot find", "not found in",
                "no information", "insufficient information",
                "not provided in", "not mentioned in"
            ]
            lower_resp = full_response.lower()
            hallucination_flagged = any(p in lower_resp for p in REFUSAL_PHRASES)

            log_tokens.info("response_tokens",
                session_id=session_id,
                stage="llm_response",
                prompt_tokens=real_prompt_tokens or total_input_tokens,
                completion_tokens=real_completion_tokens or len(full_response.split()),
                tokens_per_second=real_tokens_per_sec,
                source="ollama" if real_prompt_tokens else "estimated",
                total_latency_ms=total_ms,
                source_grounding_pct=source_grounding_pct,
                hallucination_flagged=hallucination_flagged,
            )
            log_sessions.info("query_end",
                session_id=session_id,
                question=request.question[:200],
                total_latency_ms=total_ms,
                input_tokens=real_prompt_tokens or total_input_tokens,
                output_tokens=real_completion_tokens or len(full_response.split()),
                tokens_per_second=real_tokens_per_sec,
                chunks_used=len(final_chunks),
                source_grounding_pct=source_grounding_pct,
                hallucination_flagged=hallucination_flagged,
            )


        # Trigger summary if session is long
        if should_summarize(session_id):
            try:
                all_msgs = mem_db.get_recent_messages(session_id, n=50)
                summary_turns = [{"role": m["role"], "content": m["content"]} for m in all_msgs[:-5]]
                if summary_turns:
                    summary_text = summarize_session(summary_turns)
                    last_old_msg_id = all_msgs[-6]["id"] if len(all_msgs) > 5 else all_msgs[0]["id"]
                    mem_db.upsert_summary(session_id, summary_text, last_old_msg_id)
                    log.info("session_summarized", session_id=session_id, turns=len(summary_turns))
            except Exception as e:
                log.warning("summarization_failed", error=str(e), session_id=session_id)

    return StreamingResponse(
        _streaming_with_persistence(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ─── POST /summarize ──────────────────────────────────────────────────────────

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Get all chunks in document order and generate a structured summary."""
    chunks = vector_store.get_all_chunks_for_source(request.source_id)
    if not chunks:
        raise HTTPException(status_code=404,
            detail=f"No document found with source_id '{request.source_id}'.")

    filename = chunks[0]["metadata"].get("filename", "document")
    try:
        summary = generate_summary(chunks, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

    return SummarizeResponse(source_id=request.source_id, filename=filename, summary=summary)


# ─── POST /analyze-images ─────────────────────────────────────────────────────

@app.post("/analyze-images")
async def analyze_images(request: SummarizeRequest):
    """
    On-demand LLaVA image analysis for a PDF document.

    Pipeline:
      1. Find the stored PDF for the given source_id
      2. Render each page to an image
      3. Detect image-heavy pages (< 100 chars of text but has rendered pixels)
      4. Send each image to LLaVA (llava:7b via Ollama) for description
      5. Index descriptions as new chunks in FAISS + BM25
      6. Stream progress events back to the frontend

    Events:
      {"type": "progress", "step": "scanning",  "detail": "Checking 24 pages…"}
      {"type": "progress", "step": "analyzing", "detail": "Analyzing page 3…"}
      {"type": "complete", "images_found": 5, "descriptions_indexed": 5}
      {"type": "error",   "detail": "…"}
    """
    import asyncio as _asyncio
    import json as _json
    import base64
    import fitz  # PyMuPDF
    from concurrent.futures import ThreadPoolExecutor
    from ingestion import chunk_pages, generate_embeddings
    import uuid as _uuid

    # Find the stored PDF file for this source_id
    pdf_path = None
    for f in UPLOAD_DIR.iterdir():
        if f.name.startswith(request.source_id) and f.suffix.lower() == ".pdf":
            pdf_path = f
            break

    if not pdf_path:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF found for source_id '{request.source_id}'. Image analysis only works with PDF files."
        )

    _queue: _asyncio.Queue = _asyncio.Queue()
    _loop = _asyncio.get_event_loop()

    def _analyze():
        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            _loop.call_soon_threadsafe(_queue.put_nowait, {
                "type": "progress", "step": "scanning",
                "detail": f"Scanning {total_pages} pages for images…"
            })

            image_pages = []
            for page_num in range(total_pages):
                page = doc[page_num]
                images = page.get_images(full=True)
                text = page.get_text("text").strip()
                # Analyze if the page contains images or has very little parseable text (scanned pages)
                if images or len(text) < 150:
                    image_pages.append(page_num)

            if not image_pages:
                _loop.call_soon_threadsafe(_queue.put_nowait, {
                    "type": "complete", "images_found": 0, "descriptions_indexed": 0,
                    "detail": "No image-heavy pages found in this document."
                })
                doc.close()
                return

            _loop.call_soon_threadsafe(_queue.put_nowait, {
                "type": "progress", "step": "found",
                "detail": f"Found {len(image_pages)} image-heavy pages. Sending to LLaVA…"
            })

            client = get_ollama_client()
            descriptions_indexed = 0
            new_chunks = []

            for page_num in image_pages:
                page = doc[page_num]
                _loop.call_soon_threadsafe(_queue.put_nowait, {
                    "type": "progress", "step": "analyzing",
                    "detail": f"Analyzing page {page_num + 1} with LLaVA…"
                })

                # Render page to PNG bytes
                mat = fitz.Matrix(2.0, 2.0)   # 2x scale for better quality
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                img_b64   = base64.b64encode(img_bytes).decode()

                try:
                    resp = client.chat(
                        model="llava",
                        messages=[{
                            "role": "user",
                            "content": (
                                "You are a document data extraction assistant. Analyze this page carefully and extract ALL information.\n\n"
                                "IMPORTANT RULES:\n"
                                "1. If there is a CHART or GRAPH: List EVERY data point you can read. "
                                "Example: 'January: 1200 units, February: 1500 units, March: 1700 units'. "
                                "State the chart title, axis labels (X-axis, Y-axis), and ALL visible values.\n"
                                "2. If there is a TABLE: Extract every row and column value verbatim.\n"
                                "3. If there is TEXT: Transcribe it exactly.\n"
                                "4. If there are FIGURES or DIAGRAMS: Describe every element in detail.\n\n"
                                "Be THOROUGH and PRECISE. Include all numbers, dates, labels, percentages, and categories. "
                                "Your response will be used to answer specific data questions, so completeness is critical."
                            ),
                            "images": [img_b64]
                        }]
                    )
                    description = resp["message"]["content"].strip()
                    if description:
                        new_chunks.append({
                            "id": f"{request.source_id}_img_{page_num}",
                            "text": f"[Image/Figure on Page {page_num + 1}]\n{description}",
                            "metadata": {
                                "source_id":       request.source_id,
                                "filename":        pdf_path.name.split("_", 1)[-1],
                                "page_number":     page_num + 1,
                                "chunk_index":     9000 + page_num,
                                "chunk_type":      "image_description",
                                "chunk_strategy":  "llava",
                                "embed_model":     "BAAI/bge-m3",
                                "embed_dim":       1024,
                                "ingested_at":     __import__("datetime").datetime.utcnow().isoformat() + "Z",
                                "section_heading": f"Page {page_num + 1} Visual",
                                "metrics": [], "variables": [], "dates": [],
                                "token_count": len(description.split()),
                                "total_pages": total_pages,
                                "version": 1, "retrieval_count": 0,
                            }
                        })
                        descriptions_indexed += 1
                        _loop.call_soon_threadsafe(_queue.put_nowait, {
                            "type": "progress", "step": "indexed",
                            "detail": f"Page {page_num + 1} described ({len(description)} chars)"
                        })
                except Exception as e:
                    _loop.call_soon_threadsafe(_queue.put_nowait, {
                        "type": "progress", "step": "skipped",
                        "detail": f"Page {page_num + 1} skipped: {str(e)[:60]}"
                    })

            doc.close()

            # Index all new description chunks
            if new_chunks:
                _loop.call_soon_threadsafe(_queue.put_nowait, {
                    "type": "progress", "step": "indexing",
                    "detail": f"Indexing {len(new_chunks)} image descriptions…"
                })
                texts = [c["text"] for c in new_chunks]
                embeddings = generate_embeddings(texts)
                vector_store.add_chunks(
                    ids=[c["id"] for c in new_chunks],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[c["metadata"] for c in new_chunks]
                )
                import bm25_store as _bm25
                _bm25.add_chunks(
                    ids=[c["id"] for c in new_chunks],
                    documents=texts,
                    metadatas=[c["metadata"] for c in new_chunks]
                )

            _loop.call_soon_threadsafe(_queue.put_nowait, {
                "type": "complete",
                "images_found": len(image_pages),
                "descriptions_indexed": descriptions_indexed
            })

        except Exception as e:
            log.error("analyze_images_failed", error=str(e))
            _loop.call_soon_threadsafe(_queue.put_nowait, {"type": "error", "detail": str(e)})

    async def _event_stream():
        _loop.run_in_executor(ThreadPoolExecutor(max_workers=1), _analyze)
        while True:
            msg = await _queue.get()
            yield f"data: {_json.dumps(msg)}\n\n"
            if msg.get("type") in ("complete", "error"):
                break

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )



# ─── GET /sources ─────────────────────────────────────────────────────────────

@app.get("/sources", response_model=SourcesResponse)
async def get_sources():
    """List all uploaded documents, aggregated from FAISS metadata."""
    sources_data = vector_store.get_all_sources()
    sources = [
        SourceItem(
            source_id=s["source_id"],
            filename=s["filename"],
            chunk_count=s["chunk_count"],
            page_count=s.get("page_count", 0)
        )
        for s in sources_data
    ]
    return SourcesResponse(sources=sources, total=len(sources))


# ─── DELETE /sources/{source_id} ─────────────────────────────────────────────

@app.delete("/sources/{source_id}")
async def delete_source(source_id: str):
    """Remove a document from FAISS + BM25 indexes and delete its file."""
    faiss_deleted = vector_store.delete_source(source_id)
    bm25_deleted  = bm25_store.delete_source(source_id)

    if faiss_deleted == 0:
        raise HTTPException(status_code=404,
            detail=f"No document found with source_id '{source_id}'.")

    for f in UPLOAD_DIR.glob(f"{source_id}_*"):
        f.unlink(missing_ok=True)

    log.info("document_deleted", source_id=source_id, chunks_removed=faiss_deleted)
    return {"status": "deleted", "source_id": source_id, "chunks_removed": faiss_deleted}


# ─── GET /documents ────────────────────────────────────────────────────────────

@app.get("/documents")
async def get_documents(active_only: bool = False):
    """
    List all document versions with lifecycle metadata.
    Grouped by doc_key so version history is visible.
    ?active_only=true returns only currently active versions.
    """
    rows = mem_db.get_active_documents() if active_only else mem_db.get_all_documents()

    # Group by doc_key
    grouped: dict = {}
    for row in rows:
        key = row["doc_key"]
        if key not in grouped:
            grouped[key] = {
                "doc_key":     key,
                "display_name": row["display_name"],
                "versions":    []
            }
        grouped[key]["versions"].append({
            "source_id":   row["source_id"],
            "filename":    row["filename"],
            "version":     row["version"],
            "status":      row["status"],
            "chunk_count": row["chunk_count"],
            "file_size":   row["file_size"],
            "uploaded_at": row["uploaded_at"],
            "archived_at": row.get("archived_at"),
            "notes":       row.get("notes"),
        })

    return {"documents": list(grouped.values()), "total": len(grouped)}


# ─── PATCH /documents/{source_id}/status ──────────────────────────────────────

@app.patch("/documents/{source_id}/status")
async def update_document_status(source_id: str, status: str, notes: str = ""):
    """
    Update a document version's lifecycle status.
    status: 'active' | 'archived' | 'expired'
    """
    valid = {"active", "archived", "expired"}
    if status not in valid:
        raise HTTPException(status_code=400,
            detail=f"Invalid status '{status}'. Must be one of: {valid}")

    ok = mem_db.set_document_status(source_id, status, notes or None)
    if not ok:
        raise HTTPException(status_code=404,
            detail=f"No document found with source_id '{source_id}'.")

    return {"status": "updated", "source_id": source_id, "new_status": status}


# ─── GET /documents/{old_source_id}/{new_source_id}/diff ──────────────────────

@app.get("/documents/{source_id_old}/{source_id_new}/diff")
async def get_version_diff(source_id_old: str, source_id_new: str):
    """
    Compare chunks between two document versions.
    Returns added/removed/unchanged chunk counts + a brief diff summary.
    """
    result = compare_versions(source_id_old, source_id_new)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ─── GET /sessions ────────────────────────────────────────────────────────────

@app.get("/sessions", response_model=SessionsResponse)
async def list_sessions():
    """Return all sessions ordered by last update for the sidebar history panel."""
    rows = mem_db.list_sessions(limit=50)
    sessions = [
        SessionItem(
            session_id=r["id"],
            title=r["title"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            message_count=r["message_count"]
        )
        for r in rows
    ]
    return SessionsResponse(sessions=sessions, total=len(sessions))


# ─── GET /sessions/{session_id} ───────────────────────────────────────────────

@app.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
async def get_session(session_id: str):
    """Return full message history for a session (used when user clicks to resume)."""
    session = mem_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    messages = mem_db.get_all_messages(session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        title=session["title"],
        messages=[
            MessageItem(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                created_at=m["created_at"]
            )
            for m in messages
        ]
    )


# ─── DELETE /sessions/{session_id} ────────────────────────────────────────────

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its messages."""
    deleted = mem_db.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"status": "deleted", "session_id": session_id}
