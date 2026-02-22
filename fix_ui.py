with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

out = lines[:192]

new_code = """    # ── Chat history display ───────────────────────────────────────────────────
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
            "⚠️ **Knowledge base is empty!**\\n\\n"
            "Because this app was just deployed or reset, there are no documents in the database.\\n"
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
"""
out.append(new_code)
with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(out)
