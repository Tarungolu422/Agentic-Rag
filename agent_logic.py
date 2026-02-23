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
TOP_K_FINAL     = 8
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
    from chromadb.config import Settings
    settings = Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True,
    )
    # Use explicit PersistentClient to avoid ChromaDB 1.5 tenant errors
    client = chromadb.PersistentClient(path=DB_DIR, settings=settings)
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
class AgentState(TypedDict, total=False):
    question:      str
    search_query:  str
    documents:     List[Document]
    draft_generation: str
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
     "You are an Advanced Research Assistant and RAG System. Your primary goal is to provide helpful, conversational, and highly accurate answers, similar to ChatGPT, based strictly on the provided documents.\n\n"
     "RULE 0: If the user is just saying hello or greeting you, respond with a friendly greeting and ask how you can help them. Do NOT use context or cite sources for simple greetings.\n\n"
     "INSTRUCTIONS FOR ANSWERING:\n"
     "1. Answer the user's query comprehensively using the information found in the Context.\n"
     "2. Always cite your sources using the format [Source: filename.pdf, Page: X].\n"
     "3. Be conversational, helpful, and clear in your explanations.\n\n"
     "HALLUCINATION GUARD & DIAGNOSTICS (Fallback Only):\n"
     "If the provided Context genuinely DOES NOT contain the answer to the user's question, do NOT hallucinate or guess. Instead, politely inform the user that the information is missing from the retrieved documents. Then, append the following diagnostic block to help the user troubleshoot the retrieval architecture:\n\n"
     "Retrieval Failure Detected.\n"
     "Likely Causes: [Briefly suggest if top_k is too low, chunk sizes are breaking context, or keyword matching is needed].\n"
     "Recommended Fix: Increase top_k (e.g. 6-8), use Hybrid Retrieval, or adjust chunking strategy (e.g. 1000/150).\n\n"
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

OPTIMIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query optimizer for a research paper RAG system. "
     "Your task is to take the user's conversational question and rewrite it into a highly effective search query. "
     "Extract key terms, ignore conversational filler, and include relevant synonyms. "
     "Respond ONLY with the optimized search string, nothing else."),
    ("human", "{question}"),
])

POST_PROCESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict LLM output post-processor. "
     "Your task is to review the drafted 'Answer' given the 'Context' and 'Question'. "
     "1. Fix any formatting issues (ensure good markdown). "
     "2. Remove any claims in the Answer that are NOT supported by the Context (hallucinations). "
     "3. If the Answer is completely unfaithful, rewrite it to simply state that the context lacks the information. "
     "Respond ONLY with the final, polished response text, nothing else."),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nDraft Answer:\n{answer}"),
])

# ── Node functions ─────────────────────────────────────────────────────────────
def query_optimizer_node(state: AgentState, llm) -> AgentState:
    print(f"[optimize] Raw query: '{state['question']}'")
    optimizer = OPTIMIZE_PROMPT | llm | StrOutputParser()
    optimized_q = optimizer.invoke({"question": state["question"]}).strip()
    print(f"[optimize] Optimized search query: '{optimized_q}'")
    return {**state, "search_query": optimized_q}

def retrieve_node(state: AgentState, retriever, bm25_data) -> AgentState:
    sq = state.get("search_query", state["question"])
    print(f"[retrieve] Searching dense & sparse for: '{sq}'")
    
    # 1. Dense Retrieval
    dense_docs = retriever.invoke(sq)
    
    # 2. Sparse Retrieval (BM25)
    sparse_docs = []
    if bm25_data and "bm25" in bm25_data:
        bm25 = bm25_data["bm25"]
        all_chunks = bm25_data["chunks"]
        tokenized_query = sq.lower().split()
        
        sparse_scores = bm25.get_scores(tokenized_query)
        top_k_indices = sorted(range(len(sparse_scores)), key=lambda i: sparse_scores[i], reverse=True)[:TOP_K_RETRIEVE]
        
        for idx in top_k_indices:
            if sparse_scores[idx] > 0:
                sparse_docs.append(all_chunks[idx])
                
    # 3. Reciprocal Rank Fusion
    fused_scores = {}
    
    def add_to_fusion(docs, weight=1.0):
        for rank, doc in enumerate(docs):
            doc_key = doc.page_content
            if doc_key not in fused_scores:
                fused_scores[doc_key] = {"score": 0.0, "doc": doc}
            fused_scores[doc_key]["score"] += weight / (rank + 60)
            
    add_to_fusion(dense_docs, weight=1.0)
    add_to_fusion(sparse_docs, weight=1.0)
    
    reranked = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    fused_docs = [item["doc"] for item in reranked[:TOP_K_RETRIEVE]]
    
    print(f"[retrieve] Found {len(dense_docs)} dense, {len(sparse_docs)} sparse. Fused: {len(fused_docs)} chunks.")
    return {**state, "documents": fused_docs}


def rerank_node(state: AgentState, llm) -> AgentState:
    docs = state["documents"]
    if not docs:
        return state
    print(f"[rerank] Scoring {len(docs)} chunks ...")
    scorer = RERANK_PROMPT | llm | StrOutputParser()
    scored = []
    import re
    for doc in docs:
        try:
            raw = scorer.invoke({"question": state["question"], "document": doc.page_content})
            match = re.search(r'\b([0-9]|10)\b', raw)
            score = int(match.group(1)) if match else 0
        except Exception:
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
    import re
    try:
        raw = grader.invoke({"question": state["question"], "chunks": chunks_text}).strip()
        
        # Try finding the array inside the output if it's chatting
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            raw = match.group(0)
            
        verdicts = json.loads(raw)
        if not isinstance(verdicts, list):
            raise ValueError("Not a list")
            
        # Ensure verdicts match doc count by truncating or padding
        if len(verdicts) < len(docs):
            verdicts += [False] * (len(docs) - len(verdicts))
        elif len(verdicts) > len(docs):
            verdicts = verdicts[:len(docs)]
            
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
    return {**state, "draft_generation": answer, "sources": sources_metadata, "context": context}


def post_processor_node(state: AgentState, llm) -> AgentState:
    draft = state.get("draft_generation", "")
    context = state.get("context", "")
    question = state["question"]
    
    if not draft or not context:
        return {**state, "generation": draft}
        
    print("[post_process] Polishing response and checking faithfulness ...")
    processor = POST_PROCESS_PROMPT | llm | StrOutputParser()
    try:
        final_answer = processor.invoke({"question": question, "context": context[:8000], "answer": draft}).strip()
    except Exception as e:
        print(f"[post_process] Warning: {e}. Passing draft answer as fallback.")
        final_answer = draft
        
    return {**state, "generation": final_answer}


def rewrite_node(state: AgentState, llm) -> AgentState:
    count = state.get("rewrite_count", 0) + 1
    print(f"[rewrite] Attempt {count} ...")
    rewriter = REWRITE_PROMPT | llm | StrOutputParser()
    sq = state.get("search_query", state["question"])
    new_q = rewriter.invoke({"question": sq}).strip()
    print(f"[rewrite] New search query: '{new_q}'")
    return {**state, "search_query": new_q, "rewrite_count": count}


def no_answer_node(state: AgentState) -> AgentState:
    return {**state, "generation": (
        "I was unable to find relevant information in the provided research documents "
        "to answer your question. Please try rephrasing or upload relevant papers first."
    ), "sources": []}


def route_after_grade(state: AgentState) -> str:
    if state["documents"]:
        return "generate"
    if state.get("rewrite_count", 0) >= MAX_REWRITES:
        # Pass empty documents to generation so the LLM can run its diagnostic prompt
        return "generate"
    return "rewrite"


# ── Graph builder ──────────────────────────────────────────────────────────────
def _load_bm25():
    import pickle
    bm25_path = os.path.join(DB_DIR, "bm25_index.pkl")
    if os.path.exists(bm25_path):
        try:
            with open(bm25_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load BM25 index: {e}")
    return None

def build_graph():
    retriever = _build_retriever()
    llm       = _build_llm()
    bm25_data = _load_bm25()
    graph     = StateGraph(AgentState)
    graph.add_node("optimize",     lambda s: query_optimizer_node(s, llm))
    graph.add_node("retrieve",     lambda s: retrieve_node(s, retriever, bm25_data))
    graph.add_node("rerank",       lambda s: rerank_node(s, llm))
    graph.add_node("grade",        lambda s: grade_node(s, llm))
    graph.add_node("generate",     lambda s: generate_node(s, llm))
    graph.add_node("post_process", lambda s: post_processor_node(s, llm))
    graph.add_node("rewrite",      lambda s: rewrite_node(s, llm))
    graph.add_node("no_answer",    no_answer_node)
    
    graph.set_entry_point("optimize")
    graph.add_edge("optimize",  "retrieve")
    graph.add_edge("retrieve",  "rerank")
    graph.add_edge("rerank",    "grade")
    graph.add_conditional_edges("grade", route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "no_answer": "no_answer"})
    graph.add_edge("rewrite",      "retrieve")
    graph.add_edge("generate",     "post_process")
    graph.add_edge("post_process", END)
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
        "question": question, "search_query": question, "documents": [], "generation": "",
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
        "question": question, "search_query": question, "documents": [], "generation": "",
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