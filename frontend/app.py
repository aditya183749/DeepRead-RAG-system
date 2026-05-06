"""
app.py — Complete Streamlit UI for the local document chatbot.

WHY THIS FILE EXISTS:
  This is everything the USER sees and interacts with.
  It has two main areas:
    1. SIDEBAR: File uploader, uploaded docs list, summarize button
    2. MAIN AREA: Chat window with streaming answers + citations below

  All backend calls go through api_client.py — this file never
  makes HTTP calls directly.

HOW TO RUN:
  From the project root:
    streamlit run frontend/app.py
  
  Make sure the FastAPI backend is already running:
    uvicorn backend.main:app --reload --port 8000
"""

import streamlit as st
import sys
from pathlib import Path

# Add the frontend folder to Python path so we can import api_client
sys.path.append(str(Path(__file__).parent))
import api_client


# ─── Page Config ──────────────────────────────────────────────────────────────
# Must be the FIRST Streamlit call in the script.
# Sets the tab title, icon, and layout (wide = full browser width).

st.set_page_config(
    page_title="DocChat — Local AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─── Custom CSS ───────────────────────────────────────────────────────────────
# Streamlit's default look is serviceable but we can polish it.

st.markdown("""
<style>
    /* Make citation boxes stand out */
    .citation-box {
        background-color: #f0f2f6;
        border-left: 3px solid #4CAF50;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85em;
        margin-top: 8px;
        color: #333;
    }
    /* Reduce top padding on main area */
    .block-container {
        padding-top: 1.5rem;
    }
    /* Style the source list items */
    .source-item {
        padding: 4px 0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ───────────────────────────────────────────────────────
# st.session_state persists data across Streamlit reruns (which happen on
# every user interaction). Without this, variables would reset every time.

if "messages" not in st.session_state:
    st.session_state.messages = []             # Chat history shown in UI

if "selected_source_ids" not in st.session_state:
    st.session_state.selected_source_ids = []  # Checked document UUIDs

if "sources" not in st.session_state:
    st.session_state.sources = []             # All uploaded documents

# ── Phase 3: Session memory ───────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None         # Current session UUID (set by backend)

if "session_title" not in st.session_state:
    st.session_state.session_title = "New Chat"


# ─── Backend Health Check ─────────────────────────────────────────────────────

def check_backend():
    """Show a warning banner if FastAPI backend is not running."""
    if not api_client.check_backend_health():
        st.error(
            "⚠️ **Backend not running!** Start it first:\n\n"
            "```\nuvicorn backend.main:app --reload --port 8000\n```",
            icon="🔴"
        )
        st.stop()  # Halt the rest of the app


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    """
    Renders the left sidebar with:
      1. New Chat button + currently active session title
      2. File uploader
      3. Uploaded documents list
      4. Summarize button
      5. Past Conversations browser (Phase 3)
    """
    with st.sidebar:
        st.title("📄 DocChat")
        st.caption("Local AI")

        # ── New Chat button ───────────────────────────────────────────────
        col_title, col_new = st.columns([3, 1])
        with col_title:
            if st.session_state.session_id:
                st.caption(f"💬 {st.session_state.session_title[:40]}")
        with col_new:
            if st.button("➕", help="Start a new chat session", use_container_width=True):
                st.session_state.session_id = None
                st.session_state.session_title = "New Chat"
                st.session_state.messages = []
                st.rerun()

        st.divider()

        # ── File Uploader ──────────────────────────────────────────────────
        st.subheader("📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Drop a PDF, DOCX, TXT, CSV, XLSX, PPTX or HTML file",
            type=["pdf", "docx", "doc", "txt", "md", "csv", "xlsx", "xls", "pptx", "html", "htm"],
            help="Files are processed locally. Nothing leaves your machine."
        )

        if uploaded_file is not None:
            if st.button("⚡ Process & Index", type="primary", use_container_width=True):
                progress_box = st.empty()          # single placeholder, updated in-place
                step_icons = {
                    "extracting":     "📖",
                    "extracted":      "✅",
                    "chunking":       "✂️",
                    "chunked":        "✅",
                    "embedding":      "🧠",
                    "embedded":       "✅",
                    "indexing_faiss": "🗂️",
                    "indexed_faiss":  "✅",
                    "indexing_bm25":  "🔤",
                    "done":           "🎉",
                }
                log_lines = []
                result_data = None

                for event in api_client.upload_file_stream(
                    file_bytes=uploaded_file.read(),
                    filename=uploaded_file.name
                ):
                    ev_type = event.get("type")

                    if ev_type == "progress":
                        icon = step_icons.get(event.get("step", ""), "⏳")
                        log_lines.append(f"{icon} {event.get('detail', '')}")
                        progress_box.markdown(
                            "\n\n".join(f"`{line}`" for line in log_lines)
                        )

                    elif ev_type == "complete":
                        result_data = event
                        log_lines.append(f"🎉 Done — {event.get('chunk_count')} chunks indexed!")
                        progress_box.markdown(
                            "\n\n".join(f"`{line}`" for line in log_lines)
                        )

                    elif ev_type == "error":
                        progress_box.error(f"❌ {event.get('detail', 'Upload failed')}")
                        break

                if result_data:
                    if result_data["source_id"] not in st.session_state.selected_source_ids:
                        st.session_state.selected_source_ids.append(result_data["source_id"])
                    st.session_state.messages = []
                    st.session_state.sources = api_client.get_sources()
                    st.success(
                        f"✅ **{result_data['filename']}** indexed!\n\n"
                        f"📊 {result_data['chunk_count']} chunks · {result_data['page_count']} pages"
                    )


        st.divider()

        # ── Uploaded Documents List (DLM-aware) ───────────────────────────
        st.subheader("📚 Documents")

        # Load lifecycle-aware document list
        doc_groups = api_client.get_documents()

        if not doc_groups:
            st.caption("No documents uploaded yet.")
        else:
            st.caption("Select active documents to chat across them:")
            new_selection = []

            for group in doc_groups:
                versions   = group.get("versions", [])
                active_ver = next((v for v in versions if v["status"] == "active"), None)
                other_vers = [v for v in versions if v["status"] != "active"]

                if active_ver:
                    source_id  = active_ver["source_id"]
                    is_checked = source_id in st.session_state.selected_source_ids
                    label = (
                        f"📄 {active_ver['filename']}  "
                        f"**v{active_ver['version']}** ✅"
                    )
                    checked = st.checkbox(label, value=is_checked, key=f"chk_{source_id}")
                    if checked:
                        new_selection.append(source_id)

                    # Action buttons row
                    col_img, col_arc = st.columns(2)

                    # 🖼️ Analyze Images — only for PDFs
                    with col_img:
                        if active_ver["filename"].lower().endswith(".pdf"):
                            if st.button("🖼️ Images", key=f"img_{source_id}",
                                         help="Analyze charts/figures with LLaVA",
                                         use_container_width=True):
                                img_box   = st.empty()
                                img_lines = []
                                for event in api_client.analyze_images_stream(source_id):
                                    ev_type = event.get("type")
                                    if ev_type == "progress":
                                        icon = {"scanning":"🔍","found":"📋","analyzing":"🧠",
                                                "indexed":"✅","indexing":"🗂️","skipped":"⚠️"}.get(
                                            event.get("step",""), "⏳")
                                        img_lines.append(f"{icon} {event.get('detail','')}")
                                        img_box.markdown("\n\n".join(f"`{l}`" for l in img_lines))
                                    elif ev_type == "complete":
                                        indexed = event.get("descriptions_indexed", 0)
                                        if event.get("images_found", 0) == 0:
                                            img_box.info("ℹ️ No image-heavy pages found.")
                                        else:
                                            img_lines.append(f"🎉 Done — {indexed} pages indexed!")
                                            img_box.markdown("\n\n".join(f"`{l}`" for l in img_lines))
                                            st.success(f"✅ {indexed} image pages indexed!")
                                    elif ev_type == "error":
                                        img_box.error(f"❌ {event.get('detail','Analysis failed')}")

                    # 📦 Archive button
                    with col_arc:
                        if st.button("📦 Archive", key=f"arc_{source_id}",
                                     help="Hide this document from search (keeps history)",
                                     use_container_width=True):
                            try:
                                api_client.set_document_status(source_id, "archived",
                                                               notes="Manually archived by user")
                                st.success("📦 Document archived.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ {e}")

                    # Version history expander
                    if other_vers:
                        with st.expander(f"🕓 {len(other_vers)} older version(s)"):
                            for ov in other_vers:
                                status_icon = "📦" if ov["status"] == "archived" else "⏳"
                                st.caption(
                                    f"{status_icon} **v{ov['version']}** — "
                                    f"{ov['filename']} · {ov['chunk_count']} chunks · "
                                    f"{ov['uploaded_at'][:10]}"
                                )
                                col_r, col_d = st.columns(2)
                                with col_r:
                                    if st.button("♻️ Restore", key=f"rst_{ov['source_id']}",
                                                 use_container_width=True):
                                        try:
                                            api_client.set_document_status(ov["source_id"], "active")
                                            api_client.set_document_status(source_id, "archived")
                                            st.success("♻️ Version restored!")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ {e}")
                                with col_d:
                                    if st.button("🔍 See Changes", key=f"diff_{ov['source_id']}",
                                                 use_container_width=True):
                                        try:
                                            diff = api_client.get_version_diff(ov["source_id"], source_id)
                                            st.session_state[f"diff_{ov['source_id']}"] = diff
                                        except Exception as e:
                                            st.error(f"❌ {e}")

                                # Show diff result if available
                                diff_key = f"diff_{ov['source_id']}"
                                if diff_key in st.session_state:
                                    d = st.session_state[diff_key]
                                    st.info(
                                        f"📊 **v{d['old_version']['version']} → v{d['new_version']['version']}**: "
                                        f"🟢 +{d['added_chunks']} added · "
                                        f"🔴 -{d['removed_chunks']} removed · "
                                        f"⚪ {d['unchanged_chunks']} unchanged"
                                    )
                                    if d.get("diff_summary"):
                                        with st.expander("View sample changes"):
                                            for line in d["diff_summary"][:6]:
                                                color = "green" if line.startswith("+") else "red"
                                                st.markdown(f":{color}[{line[:120]}]")

                elif versions:
                    # All versions archived — show collapsed
                    with st.expander(f"📦 {group['display_name']} (all archived)"):
                        for ov in versions:
                            st.caption(
                                f"📦 v{ov['version']} · {ov['chunk_count']} chunks · "
                                f"{ov['uploaded_at'][:10]}"
                            )
                            if st.button("♻️ Restore", key=f"rst_all_{ov['source_id']}",
                                         use_container_width=True):
                                try:
                                    api_client.set_document_status(ov["source_id"], "active")
                                    st.success("♻️ Restored!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ {e}")

            # Update selection and refresh sources list for compatibility
            if set(new_selection) != set(st.session_state.selected_source_ids):
                st.session_state.selected_source_ids = new_selection
                st.session_state.messages = []
            # Keep backward compat: also populate st.session_state.sources
            st.session_state.sources = api_client.get_sources()


        st.divider()

        # ── Summarize Button ───────────────────────────────────────────────
        st.subheader("📝 Summarize")

        # Summarize works on ONE doc — use the first selected doc
        summarize_source_id = st.session_state.selected_source_ids[0] if st.session_state.selected_source_ids else None
        _sources_for_summ   = st.session_state.get("sources", [])
        summarize_filename  = next((s["filename"] for s in _sources_for_summ if s["source_id"] == summarize_source_id), None) if summarize_source_id else None

        if summarize_source_id:
            st.caption(f"Summarizing: **{summarize_filename}**")
            if st.button("✨ Generate Summary", use_container_width=True):
                with st.spinner("Generating summary... (this takes ~30 seconds)"):
                    try:
                        result = api_client.summarize_doc(summarize_source_id)
                        # Add summary as a bot message in the chat window
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"## 📄 Document Summary\n\n{result['summary']}",
                            "citations": []
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Summary failed: {str(e)}")
        else:
            st.caption("Upload and select a document first.")

        st.divider()

        # ── Past Conversations (Phase 3) ───────────────────────────────────
        st.subheader("💬 Past Conversations")
        sessions = api_client.get_sessions()

        if not sessions:
            st.caption("No past conversations yet.")
        else:
            for sess in sessions:
                sess_id    = sess["session_id"]
                sess_title = sess["title"][:42]
                sess_date  = sess["updated_at"][:10]
                is_active  = sess_id == st.session_state.session_id

                col_btn, col_del = st.columns([5, 1])
                with col_btn:
                    prefix = "🟢 " if is_active else ""
                    label  = f"{prefix}📅 {sess_date}\n{sess_title}"
                    if st.button(label, key=f"sess_{sess_id}", use_container_width=True):
                        history = api_client.get_session_history(sess_id)
                        if history:
                            st.session_state.session_id    = sess_id
                            st.session_state.session_title = history["title"]
                            st.session_state.messages = [
                                {"role": m["role"], "content": m["content"], "citations": []}
                                for m in history["messages"]
                            ]
                            st.rerun()
                with col_del:
                    if st.button("🗑️", key=f"del_{sess_id}", help="Delete session"):
                        if api_client.delete_session(sess_id):
                            if sess_id == st.session_state.session_id:
                                st.session_state.session_id = None
                                st.session_state.messages   = []
                            st.rerun()


# ─── Chat Window ──────────────────────────────────────────────────────────────

def render_chat():
    """
    Renders the main chat area with:
      1. Welcome screen when no document is selected
      2. Full chat history with user + assistant messages
      3. Citations displayed below each assistant message
      4. Chat input at the bottom
    """
    # ── Header ─────────────────────────────────────────────────────────────
    selected = st.session_state.selected_source_ids
    if len(selected) == 1:
        # Show the single selected filename
        filename = next((s["filename"] for s in api_client.get_sources() if s["source_id"] == selected[0]), "document")
        st.title(f"Chat with: {filename}")
    elif len(selected) > 1:
        st.title(f"Chat across {len(selected)} documents")
    else:
        st.title("DocChat — Local AI")

    # ── Welcome Screen — only if no docs selected AND no loaded history ────
    if not st.session_state.selected_source_ids and not st.session_state.messages:
        st.info(
            "👈 **Upload a document** in the sidebar to start chatting.\n\n"
            "Everything runs **locally** — no internet required, no API keys needed.",
            icon="📄"
        )
        st.markdown("### What you can do:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📤 Upload**\nDrop any PDF or DOCX file")
        with col2:
            st.markdown("**💬 Ask Questions**\nGet streamed answers with citations")
        with col3:
            st.markdown("**📝 Summarize**\nGet a structured document overview")
        return

    # ── Chat History ────────────────────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("citations"):
                render_citations(message["citations"])

    # ── Chat Input ──────────────────────────────────────────────────────────
    # Show a hint if history is loaded but no document is checked for new Q&A
    if not st.session_state.selected_source_ids and st.session_state.messages:
        st.info("☝️ Check a document in the sidebar to ask new questions in this chat.", icon="💡")
        return

    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "citations": []
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream_and_display(prompt, st.session_state.selected_source_ids,
                               st.session_state.session_id)


def stream_and_display(question: str, source_ids: list, session_id: str = None):
    """
    Handles the SSE stream from /chat:
      1. Sends session_id so backend can build STM context
      2. Captures session_id from first SSE event (new sessions)
      3. Streams tokens with typing cursor
      4. Collects citations from the final SSE event
      5. Saves full answer + citations to session_state
    """
    answer_placeholder = st.empty()
    full_answer = ""
    citations   = []

    with st.spinner("Thinking..."):
        for event in api_client.chat_stream(question, source_ids, session_id=session_id):

            if event["type"] == "session_id":
                # Capture new session_id from backend
                new_sid = event.get("session_id")
                if new_sid and not st.session_state.session_id:
                    st.session_state.session_id    = new_sid
                    st.session_state.session_title = question[:60]

            elif event["type"] == "token":
                full_answer += event["content"]
                answer_placeholder.markdown(full_answer + "▌")

            elif event["type"] == "citations":
                citations = event.get("citations", [])

            elif event["type"] == "error":
                answer_placeholder.error(f"❌ Error: {event['content']}")
                return

            elif event["type"] == "done":
                break

    answer_placeholder.markdown(full_answer)

    if citations:
        render_citations(citations)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "citations": citations
    })


def render_citations(citations: list):
    """
    Renders citation pills below an assistant answer.

    Each citation shows the filename and page number where
    the answer information was found in the document.

    Example:
      📍 report.pdf · Page 7    📍 report.pdf · Page 12
    """
    if not citations:
        return

    # Deduplicate citations (same page might appear multiple times)
    seen = set()
    unique_citations = []
    for c in citations:
        key = (c.get("filename", ""), c.get("page_number", 0))
        if key not in seen:
            seen.add(key)
            unique_citations.append(c)

    # Display as a horizontal row of citation tags
    citation_cols = st.columns(min(len(unique_citations), 4))
    for i, citation in enumerate(unique_citations[:4]):  # Max 4 shown
        with citation_cols[i]:
            st.markdown(
                f'<div class="citation-box">📍 <b>{citation.get("filename", "")}</b>'
                f'<br>Page {citation.get("page_number", "?")}</div>',
                unsafe_allow_html=True
            )


# ─── Main App Entry Point ─────────────────────────────────────────────────────

def main():
    """
    Entry point. Called on every Streamlit rerun.
    Order matters:
      1. Check backend is up
      2. Render sidebar
      3. Render chat (uses state set by sidebar)
    """
    check_backend()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
