"""
ingest.py — Data Ingestion & Vector Engine
Loads all PDFs from /data, splits them into chunks, and stores them
in a persistent ChromaDB vector store.

🆕 Incremental mode (default):
   - If the DB already exists, only newly added PDFs are ingested.
   - Already-ingested files are skipped automatically.
   - No need to delete the DB when you add new PDFs!

🆕 Figure caption extraction:
   - Automatically finds "Figure X:", "Fig. X", "Table X:" captions
   - Stores them as dedicated chunks tagged source_type='figure_caption'
   - Makes image/chart content searchable without any Vision API calls!

Embeddings: HuggingFace sentence-transformers (runs 100% locally, no API key needed)
LLM:        OpenAI GPT-4o (only used in agent_logic.py, not here)
"""

import os
import re
import shutil
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = "./data"
DB_DIR        = "./db_v2"
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200
MAX_FILES     = None   # None = ingest ALL PDFs in /data

# Best free local embedding model — small, fast, high quality
# Downloads ~90MB once, then runs offline forever
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
# ──────────────────────────────────────────────────────────────────────────────


def get_already_ingested_files(vectorstore: Chroma) -> set:
    """Return the set of filenames already present in the vector store."""
    try:
        result = vectorstore.get(include=["metadatas"])
        ingested = {
            meta["filename"]
            for meta in result["metadatas"]
            if meta and "filename" in meta
        }
        return ingested
    except Exception as e:
        print(f"[ingest] Warning: could not read existing metadata ({e}). Treating DB as empty.")
        return set()


# Caption pattern: matches "Figure 1:", "Fig. 2a", "Table 3.", "Eq. (4)" etc.
_CAPTION_RE = re.compile(
    r'((?:Fig(?:ure|\.)?|Table|Eq(?:uation|\.)?|Plate)\s*[\dA-Za-z]+[.:]?[^\n]{10,200})',
    re.IGNORECASE,
)


def extract_figure_captions(docs, fname: str):
    """
    Pull figure/table/equation captions out of loaded pages and return them
    as dedicated Document chunks tagged with source_type='figure_caption'.

    These chunks are free to produce (pure regex — no API calls) and make
    visual content searchable: when a user asks about a figure, the retriever
    can now find "Figure 5: SEM micrograph of γ′ precipitates..." directly.
    """
    from langchain_core.documents import Document
    caption_docs = []
    for doc in docs:
        page_num = doc.metadata.get("page", 0)
        matches  = _CAPTION_RE.findall(doc.page_content)
        for caption in matches:
            caption = caption.strip()
            if len(caption) < 15:   # skip noise
                continue
            caption_docs.append(Document(
                page_content=caption,
                metadata={
                    "filename":    fname,
                    "page":        page_num,
                    "source_type": "figure_caption",
                },
            ))
    return caption_docs


def ingest(force_rebuild: bool = False):
    """
    Main ingestion function.

    Parameters
    ----------
    force_rebuild : bool
        If True, delete the existing DB and re-ingest ALL PDFs from scratch.
        Default is False (incremental — only new files are added).
    """

    # ── Optionally wipe the DB ────────────────────────────────────────────────
    if force_rebuild and os.path.exists(DB_DIR):
        print(f"[ingest] force_rebuild=True — deleting existing DB at '{DB_DIR}' ...")
        shutil.rmtree(DB_DIR)
        print("[ingest] DB deleted. Re-ingesting all PDFs from scratch.")

    # ── Load embedding model (needed whether DB exists or not) ─────────────────
    print(f"[ingest] Loading embedding model '{EMBED_MODEL}' (downloads once ~90MB) ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},        # Change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Connect to (or create) the vector store ───────────────────────────────
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )

    # ── Discover which PDFs are new ───────────────────────────────────────────
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data folder '{DATA_DIR}' not found.")

    all_pdfs = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf"))
    if MAX_FILES is not None:
        all_pdfs = all_pdfs[:MAX_FILES]

    already_ingested = get_already_ingested_files(vectorstore)
    new_pdfs = [f for f in all_pdfs if f not in already_ingested]

    print(f"[ingest] PDFs found:      {len(all_pdfs)}")
    print(f"[ingest] Already in DB:   {len(already_ingested)}")
    print(f"[ingest] New to ingest:   {len(new_pdfs)}")

    if not new_pdfs:
        print("[ingest] ✅ Nothing to do — all PDFs are already in the vector store.")
        return

    # ── OCR engine (optional) ─────────────────────────────────────────────────
    try:
        import ocr_engine
        from langchain_core.documents import Document
    except ImportError:
        ocr_engine = None
        print("⚠️ ocr_engine not found or dependencies missing. Skipping OCR fallback.")

    # ── Load new PDFs ─────────────────────────────────────────────────────────
    print(f"\n[ingest] Loading {len(new_pdfs)} new PDF(s) from '{DATA_DIR}' ...")
    documents = []
    for fname in new_pdfs:
        fpath = os.path.join(DATA_DIR, fname)
        try:
            # 1. Try standard text extraction
            docs = PyPDFLoader(fpath).load()

            # 2. Check for scanned pages (avg text length < 50 chars/page → OCR)
            total_chars = sum(len(d.page_content.strip()) for d in docs)
            avg_chars   = total_chars / len(docs) if docs else 0

            if avg_chars < 50 and ocr_engine:
                print(f"  🔍 {fname} looks scanned (avg {avg_chars:.1f} chars/page). Trying OCR ...")
                ocr_texts = ocr_engine.ocr_pdf(fpath)
                if ocr_texts:
                    docs = [
                        Document(page_content=text, metadata={"filename": fname, "page": i})
                        for i, text in enumerate(ocr_texts)
                    ]
                    print(f"  ✅ OCR recovered {len(docs)} pages for {fname}")
                else:
                    print(f"  ⚠️ OCR failed for {fname}, keeping original (empty) content.")

            # 3. Ensure metadata is set
            for i, doc in enumerate(docs):
                doc.metadata["filename"]    = fname
                doc.metadata["source_type"] = "text"   # tag text chunks explicitly
                if "page" not in doc.metadata:
                    doc.metadata["page"] = i

            documents.extend(docs)

            # 4. Extract figure/table captions as dedicated searchable chunks
            captions = extract_figure_captions(docs, fname)
            documents.extend(captions)

            print(f"  ✓ {fname} ({len(docs)} pages, {len(captions)} figure captions extracted)")

        except Exception as e:
            print(f"  ✗ {fname} — skipped ({e})")

    if not documents:
        raise ValueError("No pages could be loaded from the new PDFs.")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    print("\n[ingest] Splitting text documents into chunks ...")
    text_docs    = [d for d in documents if d.metadata.get("source_type") != "figure_caption"]
    caption_docs = [d for d in documents if d.metadata.get("source_type") == "figure_caption"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    text_chunks = splitter.split_documents(text_docs)
    all_chunks  = text_chunks + caption_docs   # captions are short — no splitting needed
    print(f"[ingest] Text chunks: {len(text_chunks)} | Figure caption chunks: {len(caption_docs)}")

    # ── Embed & persist ───────────────────────────────────────────────────────
    print("[ingest] Embedding and adding chunks to ChromaDB ...")
    vectorstore.add_documents(all_chunks)

    total_stored = vectorstore._collection.count()
    print(f"\n[ingest] ✅ Done! Added {len(all_chunks)} chunks ({len(text_chunks)} text + {len(caption_docs)} captions). Total vectors in DB: {total_stored}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDFs into the ChromaDB vector store.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing DB and re-ingest ALL PDFs from scratch.",
    )
    args = parser.parse_args()

    ingest(force_rebuild=args.rebuild)