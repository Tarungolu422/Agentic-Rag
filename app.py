"""
app.py — Streamlit Chat Interface for Agentic RAG
Run with:  streamlit run app.py
"""

import os
import sys
import shutil
from dotenv import load_dotenv
load_dotenv()  # Loads .env before anything else

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📚 Research Paper Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
DATA_DIR = "./data"
DB_DIR   = "./sarvam_db"   # Renamed from db_v2

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Research Assistant")
    st.markdown(
        """
        **Powered by:**
        - 🤖 Sarvam AI (Chat & OCR)
        - 🔗 LangGraph (Agentic RAG)
        - 🗄️ ChromaDB (Vector Memory)
        ---
        **How it works:**
        1. Upload your PDFs or images in the **Upload** tab
        2. They are automatically embedded into ChromaDB
        3. Ask questions — answers come with source citations
        ---
        """
    )

    if not os.environ.get("SARVAM_API_KEY"):
        st.error("⚠️ SARVAM_API_KEY not found in .env file.")
    else:
        st.success("✅ Sarvam API key loaded")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()



    st.markdown("---")
    st.caption("⚙️ **Advanced:**")
    if st.button("🔄 Force Rebuild Database"):
        with st.spinner("⏳ Rebuilding from scratch..."):
            try:
                # Clear the cached graph/ChromaDB connection FIRST to avoid WinError 32
                st.cache_resource.clear()
                from ingest import ingest
                ingest(force_rebuild=True)
                st.success("✅ Database rebuilt!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Tabs: Chat | Upload ────────────────────────────────────────────────────────
tab_chat, tab_upload = st.tabs(["💬 Chat", "📁 Upload Documents"])


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("## 📁 Upload Documents")
    st.markdown(
        "Upload your **PDFs or images** below. They will be automatically "
        "processed and added to your knowledge base so you can ask questions "
        "about them in the **Chat** tab."
    )

    uploaded_files = st.file_uploader(
        "Drag & drop files here or click to browse",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Supported formats: PDF, PNG, JPG, JPEG",
        label_visibility="visible",
    )

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        process_btn = st.button("⚡ Process & Ingest", type="primary", use_container_width=True)

    if uploaded_files and process_btn:
        os.makedirs(DATA_DIR, exist_ok=True)

        # ── Step 1: Save uploaded files ────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Processing files…")
        progress_bar = st.progress(0, text="Saving files…")

        saved = []
        skipped = []
        for idx, f in enumerate(uploaded_files):
            dest = os.path.join(DATA_DIR, f.name)
            if os.path.exists(dest):
                skipped.append(f.name)
            else:
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())
                saved.append(f.name)
            progress_bar.progress(
                int((idx + 1) / len(uploaded_files) * 40),
                text=f"Saving {f.name}…",
            )

        if skipped:
            st.info(f"ℹ️ Already in library (skipped): {', '.join(skipped)}")

        if not saved:
            st.warning("No new files to process — all uploads already exist in the library.")
            st.stop()

        st.success(f"✅ Saved {len(saved)} new file(s): {', '.join(saved)}")

        # ── Step 2: Ingest into ChromaDB ───────────────────────────────────────
        progress_bar.progress(50, text="Loading embedding model…")
        try:
            # ✅ CRITICAL: Clear cached ChromaDB connection BEFORE ingest opens the DB.
            # Without this, two PersistentClients try to hold the same file and
            # Windows raises WinError 32 (file locked by another process).
            st.cache_resource.clear()

            from ingest import ingest
            ingest()   # incremental — only processes newly saved files
            progress_bar.progress(100, text="Done!")
            st.balloons()
            st.success(
                f"🎉 **Successfully Added!** {len(saved)} document(s) have been processed "
                "and added to your knowledge base. Switch to the **💬 Chat** tab to start asking questions."
            )
        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ Ingestion failed: {e}")

    elif uploaded_files and not process_btn:
        # Preview the selected files before processing
        st.markdown("---")
        st.markdown(f"**{len(uploaded_files)} file(s) selected** — click **⚡ Process & Ingest** to add them.")
        for f in uploaded_files:
            size_kb = len(f.getvalue()) / 1024
            st.markdown(f"- 📄 `{f.name}` ({size_kb:.1f} KB)")

    # ── Current library snapshot ───────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📚 Current Knowledge Base", expanded=False):
        data_files = []
        if os.path.exists(DATA_DIR):
            data_files = sorted([
                f for f in os.listdir(DATA_DIR)
                if f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg"))
            ])

        if data_files:
            st.caption(f"{len(data_files)} file(s) in `./data/`")
            for fname in data_files:
                fpath = os.path.join(DATA_DIR, fname)
                size_kb = os.path.getsize(fpath) / 1024
                st.markdown(f"- 📄 `{fname}` ({size_kb:.1f} KB)")
        else:
            st.caption("No files yet. Upload some above!")


# ══════════════════════════════════════════════════════════════════════════════
# CHAT TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("## 💬 Multi-Document Research Assistant")
    st.markdown(
        "Ask anything about your research papers. "
        "All answers are **grounded in your documents** — no hallucinations."
    )
    st.divider()

    # ── Chat history display ───────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📄 Sources used", expanded=False):
                    for src in msg["sources"]:
                        if isinstance(src, dict):
                            st.markdown(f"- 📄 `{src.get('file','?')}` — Page {src.get('page','?')}")
                        else:
                            st.markdown(f"- `{src}`")

    # Guard: DB must exist
    db_has_data = False
    if os.path.exists(DB_DIR):
        tracking_file = os.path.join(DB_DIR, ".ingested_files.json")
        db_has_data = os.path.exists(os.path.join(DB_DIR, "chroma.sqlite3")) or os.path.exists(tracking_file)

    if not db_has_data:
        st.warning(
            "⚠️ **Knowledge base is empty!**\n\n"
            "Because this app was just deployed or reset, there are no documents in the database.\n"
            "Go to the **📁 Upload Documents** tab to add your PDFs first."
        )

# ── Chat input ─────────────────────────────────────────────────────────────
# Placed OUTSIDE the tab block so it natively pins to the viewport bottom
if db_has_data:
    user_input = st.chat_input("Ask a question about your research papers…")

    if user_input:
        # Guard: API key
        if not os.environ.get("SARVAM_API_KEY"):
            st.error("⚠️ SARVAM_API_KEY not found. Add it to your .env file and restart.")
            st.stop()

        # Reopen tab context to draw the response inside the Chat tab
        with tab_chat:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Build chat_history from prior session (exclude current user turn)
            history_msgs = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant")
            ]

            sources = []

            # Run the agent with streaming inside the container
            with st.chat_message("assistant"):
                try:
                    from agent_logic import run_query_stream

                    def _stream_and_capture():
                        gen = run_query_stream(user_input, chat_history=history_msgs)
                        try:
                            while True:
                                token = next(gen)
                                yield token
                        except StopIteration as e:
                            if e.value:
                                sources.extend(e.value)

                    # Phase 1 — retrieval status banner
                    status = st.status("🔍 Retrieving and re-ranking documents…", expanded=False)
                    stream_gen = _stream_and_capture()

                    # Pull first token (blocks until retrieval/grading done)
                    first_token = next(stream_gen, None)
                    status.update(label="✅ Retrieved — generating answer", state="complete")

                    def _full_stream():
                        if first_token is not None:
                            yield first_token
                        yield from stream_gen

                    answer = st.write_stream(_full_stream())

                except Exception as e:
                    answer  = f"❌ An error occurred: {str(e)}"
                    st.markdown(answer)
                    sources = []

            # ── Sources & PDF Page Preview ───────────────────────────────
            if sources:
                with st.expander("📄 Sources used & Page Preview", expanded=True):
                    for src in sources:
                        if isinstance(src, str):
                            st.markdown(f"- `{src}`")
                            continue

                        fname = src.get("file", "unknown")
                        page  = src.get("page", 1)   # 1-indexed
                        st.markdown(f"**📄 {fname}** — cited from Page {page}")

                        pdf_path = os.path.join(DATA_DIR, fname)
                        if os.path.exists(pdf_path) and fname.lower().endswith(".pdf"):
                            try:
                                import fitz  # PyMuPDF — no Poppler needed

                                doc = fitz.open(pdf_path)
                                total_pages = doc.page_count

                                # Show cited page ± 1
                                pages_to_show = [
                                    p for p in [page - 2, page - 1, page]
                                    if 0 <= p < total_pages
                                ]

                                cols = st.columns(len(pages_to_show)) if pages_to_show else []
                                for col, pg_idx in zip(cols, pages_to_show):
                                    pg = doc[pg_idx]
                                    # Render at 1.5x zoom
                                    mat = fitz.Matrix(1.5, 1.5)
                                    pix = pg.get_pixmap(matrix=mat)
                                    img_bytes = pix.tobytes("png")
                                    label = f"🔵 Page {pg_idx + 1} (cited)" if pg_idx + 1 == page else f"Page {pg_idx + 1}"
                                    with col:
                                        st.image(img_bytes, caption=label, use_container_width=True)

                                doc.close()
                                st.caption("🔵 Blue label = cited page")

                            except Exception as img_err:
                                st.caption(f"⚠️ Could not render page preview: {img_err}")
                        else:
                            st.caption(f"⚠️ File not found in data/: {fname}")

                        st.divider()
            else:
                st.caption("_No sources cited — answer may be a fallback response._")

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
