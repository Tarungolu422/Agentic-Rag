"""
app.py — Streamlit Chat Interface for Agentic RAG
Run with:  streamlit run app.py
"""

import os
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

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Research Assistant")
    st.markdown(
        """
        **Powered by:**
        - 🤖 OpenAI GPT-4o
        - 🔗 LangGraph (Agentic RAG)
        - 🗄️ ChromaDB (Vector Memory)
        ---
        **How it works:**
        1. Your question is sent to ChromaDB
        2. Top 10 chunks are retrieved & re-ranked
        3. GPT-4o grades chunk relevance
        4. If irrelevant → query is rewritten & retried
        5. Final answer is generated with citations
        ---
        """
    )

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY not found in .env file.")
    else:
        st.success("✅ OpenAI API key loaded")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("⚠️ Run `python ingest.py` first to load your PDFs.")

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📚 Multi-Document Research Assistant")
st.markdown(
    "Ask anything about your research papers. "
    "All answers are **grounded in your documents** — no hallucinations."
)
st.divider()

# ── Chat history display ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources used", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")

# ── Input ──────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about your research papers...")

if user_input:
    # Guard: API key must be set
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY not found. Please add it to your .env file and restart.")
        st.stop()

    # Guard: Vector DB must exist
    if not os.path.exists("./db_v2") or not os.listdir("./db_v2"):
        st.error(
            "⚠️ Vector database not found. "
            "Please run `python ingest.py` to process your PDF files first."
        )
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the agent
    with st.chat_message("assistant"):
        with st.spinner("🔍 Thinking — retrieving, re-ranking, and generating answer..."):
            try:
                # Import here so API key is set before importing
                from agent_logic import run_query
                result = run_query(user_input)
                answer  = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer  = f"❌ An error occurred: {str(e)}"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("📄 Sources used & Visual Context", expanded=True):
                import os
                import sys
                from pdf2image import convert_from_path

                # Smart path for Poppler: None on Linux (Streamlit Cloud), hardcoded on Windows
                POPPLER_PATH = r"C:\Users\Tarun\poppler-25.12.0\Library\bin" if sys.platform == "win32" else None

                for src in sources:
                    if isinstance(src, str):
                        st.markdown(f"- `{src}`")
                        continue

                    fname = src.get("file", "unknown")
                    page  = src.get("page", 1)
                    st.markdown(f"**📄 {fname}** — cited from Page {page}")

                    pdf_path = os.path.join("./data", fname)
                    if os.path.exists(pdf_path):
                        try:
                            # Show a window of pages: cited page minus 1, cited page, cited page plus 1
                            # This ensures we capture figures that appear on an adjacent page to their caption
                            first = max(1, page - 1)
                            last  = page + 1  # pdf2image handles out-of-range gracefully
                            images = convert_from_path(
                                pdf_path,
                                first_page=first,
                                last_page=last,
                                poppler_path=POPPLER_PATH,
                            )
                            if images:
                                cols = st.columns(len(images))
                                for col_idx, (img, pg_num) in enumerate(zip(images, range(first, last + 1))):
                                    with cols[col_idx]:
                                        border = "🔵 " if pg_num == page else ""
                                        st.image(img, caption=f"{border}Page {pg_num}")
                                st.caption("🔵 Blue border = cited page. Adjacent pages shown to help locate figures.")
                        except Exception as e:
                            st.caption(f"⚠️ Could not load image preview: {e}")
                    else:
                        st.caption(f"⚠️ File not found in data/: {fname}")
                    st.divider()
        else:
            st.caption("_No sources cited — answer may be a fallback response._")

    # Save to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })