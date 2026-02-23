"""
ingest.py — Data Ingestion & Vector Engine
─────────────────────────────────────────────────────────────────────
Loads PDFs from /data, splits into chunks, stores in ChromaDB.

✅ Incremental mode (default):
   - Tracks ingested filenames in sarvam_db/.ingested_files.json
   - Only NEW files are processed — already-ingested ones are skipped.
   - The JSON file is the single source of truth (no ChromaDB metadata queries).

✅ Force rebuild:
   - Deletes DB + JSON tracking file, then re-ingests everything.

Embeddings : HuggingFace sentence-transformers (local, no API key needed)
"""

import os
import re
import json
import shutil
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = "./data"
DB_DIR          = "./sarvam_db"
COLLECTION_NAME = "rag_docs"          # Must match agent_logic.py
TRACKING_FILE   = os.path.join(DB_DIR, ".ingested_files.json")  # ← source of truth
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 150
MAX_FILES       = None                # None = ingest ALL PDFs
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
# ──────────────────────────────────────────────────────────────────────────────


# ── Tracking helpers (JSON file — reliable, no ChromaDB queries) ──────────────
def _load_tracked_files() -> set:
    """Return the set of filenames already ingested (read from JSON file)."""
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, "r") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def _save_tracked_files(filenames: set):
    """Persist the updated set of ingested filenames to the JSON file."""
    os.makedirs(DB_DIR, exist_ok=True)
    with open(TRACKING_FILE, "w") as f:
        json.dump(sorted(filenames), f, indent=2)


# ── ChromaDB helpers ──────────────────────────────────────────────────────────
def _make_vectorstore(embeddings) -> Chroma:
    """Open (or create) the ChromaDB collection using an explicit PersistentClient.
    This avoids the 'default_tenant not found' error in ChromaDB ≥1.5."""
    
    # Configure Chroma to use WAL (Write-Ahead Logging) and a higher busy timeout
    # This prevents the 'code: 1032 attempt to write a readonly database' error
    # caused by Streamlit keeping the DB open in the background.
    settings = Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True,
    )
    
    client = chromadb.PersistentClient(path=DB_DIR, settings=settings)
    
    # Try forcing PRAGMA statements on the raw sqlite3 connection if possible
    try:
        import sqlite3
        conn = sqlite3.connect(os.path.join(DB_DIR, "chroma.sqlite3"), timeout=60)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=60000;")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ingest] Warning: Could not set sqlite pragmas: {e}")

    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


# ── Caption extractor ─────────────────────────────────────────────────────────
_CAPTION_RE = re.compile(
    r'((?:Fig(?:ure|\.)?|Table|Eq(?:uation|\.)?|Plate)\s*[\dA-Za-z]+[.:]?[^\n]{10,200})',
    re.IGNORECASE,
)

def extract_figure_captions(docs, fname: str):
    """Pull figure/table captions as dedicated searchable chunks."""
    from langchain_core.documents import Document
    caption_docs = []
    for doc in docs:
        page_num = doc.metadata.get("page", 0)
        for caption in _CAPTION_RE.findall(doc.page_content):
            caption = caption.strip()
            if len(caption) < 15:
                continue
            caption_docs.append(Document(
                page_content=caption,
                metadata={"filename": fname, "page": page_num, "source_type": "figure_caption"},
            ))
    return caption_docs


# ── Main ingest function ──────────────────────────────────────────────────────
def ingest(force_rebuild: bool = False):
    """
    Parameters
    ----------
    force_rebuild : bool
        If True, wipe the DB + tracking file and re-ingest everything.
        Default is False — only new files are processed.
    """

    # ── Wipe if force_rebuild ─────────────────────────────────────────────────
    if force_rebuild:
        print(f"[ingest] force_rebuild — safely resetting ChromaDB ...")
        
        settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True,
        )
        try:
            client = chromadb.PersistentClient(path=DB_DIR, settings=settings)
            client.reset()
            print("[ingest] ChromaDB wiped via client.reset().")
        except Exception as e:
            print(f"[ingest] Warning during reset: {e}")
            
        # Manually clear our JSON tracking file
        if os.path.exists(TRACKING_FILE):
            os.remove(TRACKING_FILE)
            
        already_ingested = set()
    else:
        already_ingested = _load_tracked_files()

    os.makedirs(DB_DIR, exist_ok=True)

    # ── Load embedding model ──────────────────────────────────────────────────
    print(f"[ingest] Loading embedding model '{EMBED_MODEL}' ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data folder '{DATA_DIR}' not found.")

    all_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
                # Use a relative path so we can uniquely identify files in subfolders
                rel_path = os.path.relpath(os.path.join(root, f), DATA_DIR)
                all_files.append(rel_path.replace("\\", "/"))
    
    all_files.sort()
    if MAX_FILES is not None:
        all_files = all_files[:MAX_FILES]

    new_files = [f for f in all_files if f not in already_ingested]

    print(f"[ingest] Files found:   {len(all_files)}")
    print(f"[ingest] Already done:  {len(already_ingested)}")
    print(f"[ingest] New to ingest: {len(new_files)}")

    if not new_files:
        print("[ingest] ✅ Nothing to do — all files are already ingested.")
        return

    # ── Optional OCR fallback ─────────────────────────────────────────────────
    try:
        import ocr_engine
        from langchain_core.documents import Document as LCDoc
    except ImportError:
        ocr_engine = None

    # ── Load new files ────────────────────────────────────────────────────────
    print(f"\n[ingest] Loading {len(new_files)} new file(s) ...")
    documents       = []
    successfully_loaded = []

    for fname in new_files:
        fpath = os.path.join(DATA_DIR, fname)
        try:
            docs = []
            captions = []
            if fname.lower().endswith(".pdf"):
                try:
                    docs = PyPDFLoader(fpath).load()
                except Exception as enc_err:
                    print(f"  ⚠️  {fname} encoding issue ({enc_err}), retrying ...")
                    docs = PyPDFLoader(fpath, extraction_mode="plain").load()

                # Low text density → try Sarvam OCR
                total_chars = sum(len(d.page_content.strip()) for d in docs)
                avg_chars   = total_chars / len(docs) if docs else 0
                if avg_chars < 50 and ocr_engine:
                    print(f"  🔍 {fname} looks scanned (avg {avg_chars:.1f} chars/page), running OCR ...")
                    ocr_texts = ocr_engine.ocr_pdf(fpath)
                    if ocr_texts:
                        docs = [
                            LCDoc(page_content=t, metadata={"filename": fname, "page": i})
                            for i, t in enumerate(ocr_texts)
                        ]
                
                captions = extract_figure_captions(docs, fname)

            elif fname.lower().endswith((".png", ".jpg", ".jpeg")):
                if ocr_engine:
                    print(f"  🖼️ {fname} is an image, running OCR ...")
                    ocr_texts = ocr_engine.ocr_pdf(fpath)
                    if ocr_texts:
                        docs = [
                            LCDoc(page_content=t, metadata={"filename": fname, "page": i})
                            for i, t in enumerate(ocr_texts)
                        ]
                else:
                    print(f"  ⚠️  {fname} is an image but ocr_engine is not available.")

            for i, doc in enumerate(docs):
                doc.metadata["filename"]    = fname
                doc.metadata.setdefault("source_type", "text")
                doc.metadata.setdefault("page", i)

            documents.extend(docs)
            documents.extend(captions)
            
            if docs or captions:
                successfully_loaded.append(fname)
                print(f"  ✓ {fname} — {len(docs)} pages/chunks, {len(captions)} captions")
            else:
                print(f"  ✗ {fname} — skipped (no text could be extracted)")

        except Exception as e:
            print(f"  ✗ {fname} — skipped ({e})")

    if not documents:
        raise ValueError("No pages could be loaded from the new PDFs.")

    # ── Split ─────────────────────────────────────────────────────────────────
    print("\n[ingest] Splitting documents into chunks ...")
    text_docs    = [d for d in documents if d.metadata.get("source_type") != "figure_caption"]
    caption_docs = [d for d in documents if d.metadata.get("source_type") == "figure_caption"]
    splitter     = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    text_chunks = splitter.split_documents(text_docs)
    all_chunks  = text_chunks + caption_docs
    print(f"[ingest] {len(text_chunks)} text + {len(caption_docs)} captions = {len(all_chunks)} chunks")

    # ── Embed & store ─────────────────────────────────────────────────────────
    print(f"[ingest] Embedding {len(all_chunks)} chunks into ChromaDB ...")
    
    # Inject filename into actual text content so it can be searched by name
    for chunk in all_chunks:
        fname = chunk.metadata.get("filename", "unknown")
        prefix = f"Filename: {fname}\n\n"
        if not chunk.page_content.startswith(prefix):
            chunk.page_content = prefix + chunk.page_content

    vectorstore = _make_vectorstore(embeddings)

    batch_size = 5000
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        print(f"[ingest]   Batch {i//batch_size + 1}: {len(batch)} chunks ...")
        vectorstore.add_documents(batch)

    total = vectorstore._collection.count()
    print(f"\n[ingest] ✅ Done! {len(all_chunks)} chunks added. Total in DB: {total}")

    # ── Update tracking file ──────────────────────────────────────────────────
    # Only mark files as done AFTER they were successfully embedded.
    updated_set = already_ingested | set(successfully_loaded)
    _save_tracked_files(updated_set)
    print(f"[ingest] Tracking file updated — {len(updated_set)} files marked as done.")

    # Release ChromaDB reference (del is enough on all platforms)
    del vectorstore


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest PDFs into the Sarvam RAG database.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete the database and re-ingest ALL PDFs from scratch.")
    args = parser.parse_args()
    ingest(force_rebuild=args.rebuild)