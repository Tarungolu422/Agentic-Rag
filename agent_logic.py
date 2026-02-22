"""
agent_logic.py — LangGraph Agentic RAG Workflow
─────────────────────────────────────────────────────────────────────
Embeddings : HuggingFace sentence-transformers (local, no API key)
LLM        : Sarvam AI  (sarvam-m model via OpenAI-compatible endpoint)

Flow: retrieve → rerank → grade (batched) → generate → faithfulness → END
      └── rewrite (on no relevant docs, up to MAX_REWRITES) ──────────┘
"""

import os
import json
from dotenv import load_dotenv
load_dotenv()

from typing import Generator, List, Optional, TypedDict

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import httpx

# ── Config ────────────────────────────────────────────────────────────────────
DB_DIR          = "./sarvam_db"       # Renamed from db_v2 — must match ingest.py
COLLECTION_NAME = "rag_docs"          # MUST match ingest.py
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
SARVAM_MODEL    = "sarvam-m"          # Correct Sarvam AI model name
SARVAM_BASE_URL = "https://api.sarvam.ai/v1"  # LangChain appends /chat/completions
TOP_K_RETRIEVE  = 10
TOP_K_FINAL     = 5
MAX_REWRITES    = 2
FAITHFULNESS_THRESHOLD = 0.20
# ──────────────────────────────────────────────────────────────────────────────


# ── LLM & Retriever ───────────────────────────────────────────────────────────
def _build_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    # Use explicit PersistentClient to avoid ChromaDB 1.5 tenant errors
    client = chromadb.PersistentClient(path=DB_DIR)
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})


def _build_llm():
    """
    Connect to Sarvam AI via the OpenAI-compatible endpoint.
    Auth: 'api-subscription-key' header injected via httpx.Client default_headers.
    """
    SARVAM_KEY = os.environ.get("SARVAM_API_KEY", "")
    if not SARVAM_KEY:
        raise ValueError("SARVAM_API_KEY is not set in your .env file.")

    # Inject Sarvam auth header and disable SSL verify at the transport level
    http_client = httpx.Client(
        verify=False,
        headers={"api-subscription-key": SARVAM_KEY},
    )

    return ChatOpenAI(
        model=SARVAM_MODEL,
        temperature=0,
        http_client=http_client,
        api_key=SARVAM_KEY,       # used as Authorization: Bearer (Sarvam ignores it)
        base_url=SARVAM_BASE_URL, # .../v1 → LangChain appends /chat/completions ✅
    )


# ── Graph State ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question:      str
    documents:     List[Document]
    generation:    str
    sources:       List[str]
    rewrite_count: int
    context:       str
    chat_history:  List[dict]


# ── Prompts ───────────────────────────────────────────────────────────────────
RERANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a relevance scorer. Given a user question and a document chunk, "
     "output a single integer score 0-10 indicating relevance.\n"
     "10=directly answers, 7-9=discusses in depth, 4-6=references topic, 1-3=tangential, 0=off-topic.\n"
     "Output ONLY the integer, nothing else."),
    ("human", "Question: {question}\n\nChunk:\n{document}"),
])

BATCH_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a relevance grader. Given a question and numbered document chunks, "
     "decide if each chunk contains ANY useful information.\n"
     "Be generous — output true if even partial relevance. False ONLY if completely off-topic.\n"
     "Respond ONLY with a valid JSON boolean array, e.g. [true, false, true]"),
    ("human", "Question: {question}\n\nChunks:\n{chunks}"),
])

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert research assistant. Answer the user's question strictly "
     "using the provided context from research papers.\n"
     "Rules:\n"
     "1. If the answer is not in the context, say so. Do NOT hallucinate.\n"
     "2. Always cite filename(s) and page numbers using [Source: filename.pdf, Page: N].\n"
     "3. Be precise, thorough, and academic in tone.\n"
     "4. [FIGURE DESCRIPTION] chunks are figure/table captions extracted from the paper. "
     "Treat them as visual evidence and cite them.\n"
     "5. CONVERSATION HISTORY: Use prior exchanges to answer follow-ups but always "
     "prioritise retrieved document context for factual claims.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query optimizer for a research paper retrieval system. "
     "The current query did not retrieve relevant documents. "
     "Rewrite it to be more specific using different academic terminology. "
     "Output ONLY the rewritten query."),
    ("human", "Original query: {question}"),
])

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a faithfulness verifier for a RAG system.\n"
     "Score how faithful the answer is to the context on a scale 0.0-1.0.\n"
     "1.0=every claim supported, 0.5-0.9=mostly supported, "
     "0.1-0.4=significant unsupported claims, 0.0=fabricated.\n"
     "Output ONLY the float, e.g. 0.87"),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"),
])


# ── Node functions ─────────────────────────────────────────────────────────────
def retrieve_node(state: AgentState, retriever) -> AgentState:
    print(f"[retrieve] Searching: '{state['question']}'")
    docs = retriever.invoke(state["question"])
    print(f"[retrieve] Found {len(docs)} chunks.")
    return {**state, "documents": docs}


def rerank_node(state: AgentState, llm) -> AgentState:
    docs = state["documents"]
    if not docs:
        return state
    print(f"[rerank] Scoring {len(docs)} chunks ...")
    scorer = RERANK_PROMPT | llm | StrOutputParser()
    scored = []
    for doc in docs:
        try:
            raw = scorer.invoke({"question": state["question"], "document": doc.page_content})
            score = int(raw.strip())
        except (ValueError, AttributeError):
            score = 0
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored[:TOP_K_FINAL]]
    print(f"[rerank] Top-{len(top_docs)} scores: {[s for s, _ in scored[:TOP_K_FINAL]]}")
    return {**state, "documents": top_docs}


def grade_node(state: AgentState, llm) -> AgentState:
    """Batched grading — 1 LLM call for all chunks."""
    docs = state["documents"]
    if not docs:
        return {**state, "documents": []}
    print(f"[grade] Batch-grading {len(docs)} chunks ...")
    chunks_text = "\n\n".join(f"[{i+1}] {doc.page_content[:600]}" for i, doc in enumerate(docs))
    grader = BATCH_GRADE_PROMPT | llm | StrOutputParser()
    try:
        raw = grader.invoke({"question": state["question"], "chunks": chunks_text}).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].strip().lstrip("json").strip()
        verdicts = json.loads(raw)
        if not isinstance(verdicts, list):
            raise ValueError("Not a list")
    except Exception as e:
        print(f"[grade] Warning: could not parse response ({e}). Keeping all chunks.")
        verdicts = [True] * len(docs)
    relevant = [doc for doc, v in zip(docs, verdicts) if v]
    print(f"[grade] {len(relevant)}/{len(docs)} chunks relevant.")
    return {**state, "documents": relevant}


def generate_node(state: AgentState, llm) -> AgentState:
    docs = state["documents"]
    context_parts, sources_metadata = [], []
    for doc in docs:
        fname    = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        page_num = doc.metadata.get("page", 0) + 1
        label    = "[FIGURE DESCRIPTION]" if doc.metadata.get("source_type") == "figure_caption" else "[TEXT]"
        context_parts.append(f"{label} [Source: {fname}, Page: {page_num}]\n{doc.page_content}")
        meta = {"file": fname, "page": page_num}
        if meta not in sources_metadata:
            sources_metadata.append(meta)
    context = "\n\n---\n\n".join(context_parts)

    lc_history = []
    for msg in state.get("chat_history", []):
        role, content = msg.get("role"), msg.get("content", "")
        if role == "user":      lc_history.append(HumanMessage(content=content))
        elif role == "assistant": lc_history.append(AIMessage(content=content))

    generator = GENERATE_PROMPT | llm | StrOutputParser()
    answer = generator.invoke({"question": state["question"], "context": context, "chat_history": lc_history})
    print(f"[generate] Done. Sources: {sources_metadata}")
    return {**state, "generation": answer, "sources": sources_metadata, "context": context}


def faithfulness_node(state: AgentState, llm) -> AgentState:
    answer, context, question = state.get("generation", ""), state.get("context", ""), state["question"]
    if not answer or not context:
        return state
    print("[faithfulness] Verifying ...")
    verifier = FAITHFULNESS_PROMPT | llm | StrOutputParser()
    try:
        score = float(verifier.invoke({"question": question, "context": context[:8000], "answer": answer}).strip())
    except Exception as e:
        print(f"[faithfulness] Could not parse score ({e}). Accepting answer.")
        score = 1.0
    print(f"[faithfulness] Score: {score:.2f} (threshold: {FAITHFULNESS_THRESHOLD})")
    if score < FAITHFULNESS_THRESHOLD:
        return {**state, "generation": (
            "⚠️ **Faithfulness Warning**: The generated answer could not be fully verified "
            f"against the retrieved documents (score: {score:.2f}). "
            "Please try rephrasing your question."
        )}
    return state


def rewrite_node(state: AgentState, llm) -> AgentState:
    count = state.get("rewrite_count", 0) + 1
    print(f"[rewrite] Attempt {count} ...")
    rewriter = REWRITE_PROMPT | llm | StrOutputParser()
    new_q = rewriter.invoke({"question": state["question"]})
    print(f"[rewrite] New query: '{new_q}'")
    return {**state, "question": new_q, "rewrite_count": count}


def no_answer_node(state: AgentState) -> AgentState:
    return {**state, "generation": (
        "I was unable to find relevant information in the provided research documents "
        "to answer your question. Please try rephrasing or upload relevant papers first."
    ), "sources": []}


def route_after_grade(state: AgentState) -> str:
    if state["documents"]:
        return "generate"
    if state.get("rewrite_count", 0) >= MAX_REWRITES:
        return "no_answer"
    return "rewrite"


# ── Graph builder ──────────────────────────────────────────────────────────────
def build_graph():
    retriever = _build_retriever()
    llm       = _build_llm()
    graph     = StateGraph(AgentState)
    graph.add_node("retrieve",     lambda s: retrieve_node(s, retriever))
    graph.add_node("rerank",       lambda s: rerank_node(s, llm))
    graph.add_node("grade",        lambda s: grade_node(s, llm))
    graph.add_node("generate",     lambda s: generate_node(s, llm))
    graph.add_node("faithfulness", lambda s: faithfulness_node(s, llm))
    graph.add_node("rewrite",      lambda s: rewrite_node(s, llm))
    graph.add_node("no_answer",    no_answer_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve",  "rerank")
    graph.add_edge("rerank",    "grade")
    graph.add_conditional_edges("grade", route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "no_answer": "no_answer"})
    graph.add_edge("rewrite",      "retrieve")
    graph.add_edge("generate",     "faithfulness")
    graph.add_edge("faithfulness", END)
    graph.add_edge("no_answer",    END)
    return graph.compile()


# ── Public API ─────────────────────────────────────────────────────────────────
# st.cache_resource ensures ChromaDB is opened ONCE per Streamlit server process.
# This prevents WinError 32/5 file locking on Windows.
try:
    import streamlit as st

    @st.cache_resource(show_spinner="🔄 Loading knowledge base…")
    def get_app():
        return build_graph()

except ImportError:
    _app = None
    def get_app():
        global _app
        if _app is None:
            _app = build_graph()
        return _app


def run_query(question: str, chat_history: Optional[List[dict]] = None) -> dict:
    """Blocking call — returns {answer, sources}."""
    app = get_app()
    final_state = app.invoke({
        "question": question, "documents": [], "generation": "",
        "sources": [], "rewrite_count": 0, "context": "", "chat_history": chat_history or [],
    })
    return {"answer": final_state.get("generation", "No answer generated."),
            "sources": final_state.get("sources", [])}


def run_query_stream(
    question: str,
    chat_history: Optional[List[dict]] = None,
) -> Generator[str, None, dict]:
    """
    Streaming entry point for st.write_stream().
    Yields text tokens progressively; returns sources list via StopIteration.value.
    """
    app = get_app()
    init_state = {
        "question": question, "documents": [], "generation": "",
        "sources": [], "rewrite_count": 0, "context": "", "chat_history": chat_history or [],
    }
    prev_generation = ""
    final_sources   = []
    for snapshot in app.stream(init_state, stream_mode="values"):
        current_gen = snapshot.get("generation", "")
        final_sources = snapshot.get("sources", [])
        if current_gen and current_gen != prev_generation:
            new_tokens = current_gen[len(prev_generation):]
            if new_tokens:
                yield new_tokens
            prev_generation = current_gen
    return final_sources


if __name__ == "__main__":
    result = run_query("What are the main contributions of the papers in this collection?")
    print("\n=== ANSWER ===\n", result["answer"])
    print("\n=== SOURCES ===")
    for s in result["sources"]:
        print(f"  • {s}")