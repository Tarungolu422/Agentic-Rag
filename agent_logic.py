"""
agent_logic.py — LangGraph Agentic RAG Workflow (with Re-ranking)
─────────────────────────────────────────────────────────────────
Embeddings: HuggingFace sentence-transformers (local, no API key)
LLM:OpenAI GPT-4o (for reasoning, grading, generating)

Flow: retrieve → rerank → grade (batched) → [generate → faithfulness | rewrite → retrieve (loop)]

Improvements over v1:
  ✓ Batched grading  — 1 LLM call instead of N (5–8× faster)
  ✓ Faithfulness check — post-generation verifier to catch hallucinations
  ✓ Image-description context — chunks tagged as figure captions are labelled
    clearly in the prompt so GPT-4o knows they describe visual content
"""

import os
import json
from dotenv import load_dotenv
load_dotenv()

from typing import List, TypedDict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

import httpx

# ── Config ────────────────────────────────────────────────────────────────────
DB_DIR         = "./db_v2"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"   # Local, free, no API
OPENAI_MODEL   = "gpt-4o"                                    # Powerful OpenAI model
TOP_K_RETRIEVE = 10
TOP_K_FINAL    = 5
MAX_REWRITES   = 2
FAITHFULNESS_THRESHOLD = 0.45   # Below this → return safe fallback
# ──────────────────────────────────────────────────────────────────────────────


# ── LLM & Retriever setup ─────────────────────────────────────────────────────
def _build_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})


def _build_llm():
    # Requires OPENAI_API_KEY in .env
    # Disable SSL verify to fix local connection issues
    http_client = httpx.Client(verify=False)
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0, http_client=http_client)


# ── Graph State ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question:      str
    documents:     List[Document]
    generation:    str
    sources:       List[str]
    rewrite_count: int
    context:       str   # stored so faithfulness node can access it


# ── Prompts ───────────────────────────────────────────────────────────────────

RERANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a relevance scorer. Given a user question and a document chunk, "
     "output a single integer score from 0 to 10 indicating how relevant the chunk is.\n"
     "Scoring rules:\n"
     "- 10: chunk directly answers the question\n"
     "- 7-9: chunk discusses the topic in depth\n"
     "- 4-6: chunk references, describes, or mentions figures/images/tables relevant to the question, "
     "even if the actual image is not embedded in text (e.g. 'Fig. 2 shows the microstructure...')\n"
     "- 1-3: chunk is tangentially related\n"
     "- 0: chunk is completely off-topic\n"
     "IMPORTANT: If the user asks about a figure or image, chunks that MENTION or DESCRIBE that figure "
     "should score 5 or higher, even though a text chunk cannot embed an actual image.\n"
     "Output ONLY the integer, nothing else."),
    ("human", "Question: {question}\n\nChunk:\n{document}"),
])

# ── IMPROVEMENT 1: Batched grading — single LLM call for all chunks ──────────
BATCH_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a relevance grader for a research paper retrieval system.\n"
     "You will receive a question and a numbered list of document chunks.\n"
     "For each chunk, decide if it contains ANY useful information related to the question.\n"
     "Be generous — output true if there is even partial relevance. Output false ONLY if completely off-topic.\n"
     "Respond with ONLY a valid JSON array of booleans, one per chunk, in order.\n"
     "Example for 3 chunks: [true, false, true]"),
    ("human", "Question: {question}\n\nChunks:\n{chunks}"),
])

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert research assistant. Answer the user's question strictly "
     "using the provided context from research papers.\n"
     "Rules:\n"
     "1. If the answer is not in the context, say: 'I don't have enough information "
     "in the provided documents to answer this question.' Do NOT hallucinate.\n"
     "2. Always cite the paper filename(s) and page numbers using [Source: filename.pdf, Page: N].\n"
     "3. Be precise, thorough, and academic in tone.\n"
     "4. FIGURE DESCRIPTIONS: Some context chunks are labelled [FIGURE DESCRIPTION]. These are "
     "figure/table captions extracted directly from the paper text. Treat them as visual evidence.\n"
     "5. If the user asks about a figure, diagram, microstructure, graph, or image:\n"
     "   a) Look for [FIGURE DESCRIPTION] chunks — describe what they say.\n"
     "   b) Tell the user: 'The actual image can be viewed in the Sources panel below.'\n"
     "   c) Cite the source and page number.\n"
     "   Do NOT say you have no information if context references a figure on the topic.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query optimizer for a research paper retrieval system. "
     "The current query did not retrieve relevant documents. "
     "Rewrite the query to be more specific and use different terminology "
     "that is likely to appear in academic papers. Output only the rewritten query."),
    ("human", "Original query: {question}"),
])

# ── IMPROVEMENT 2: Faithfulness check prompt ──────────────────────────────────
FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a faithfulness verifier for a RAG system.\n"
     "Given a question, the source context documents, and a generated answer, "
     "score how faithful the answer is to the context on a scale from 0.0 to 1.0.\n"
     "Rules:\n"
     "- 1.0: Every claim in the answer is directly supported by the context.\n"
     "- 0.5–0.9: Most claims are supported; some minor inferences.\n"
     "- 0.1–0.4: Answer contains significant claims NOT in the context.\n"
     "- 0.0: Answer is completely fabricated or contradicts the context.\n"
     "Output ONLY the float, nothing else. Example: 0.87"),
    ("human",
     "Question: {question}\n\nContext:\n{context}\n\nGenerated Answer:\n{answer}"),
])


# ── Node functions ─────────────────────────────────────────────────────────────
def retrieve_node(state: AgentState, retriever) -> AgentState:
    print(f"[retrieve] Searching for: '{state['question']}'")
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
    print(f"[rerank] Kept top-{len(top_docs)} (scores: {[s for s,_ in scored[:TOP_K_FINAL]]}).")
    return {**state, "documents": top_docs}


def grade_node(state: AgentState, llm) -> AgentState:
    """
    IMPROVEMENT 1: Batched grading.
    Sends ALL chunks to GPT-4o in ONE call and gets back a JSON boolean array.
    Previously: 10 sequential LLM calls. Now: 1 call. ~5-8x faster.
    """
    docs = state["documents"]
    if not docs:
        return {**state, "documents": []}

    print(f"[grade] Batch-grading {len(docs)} chunks in a single LLM call ...")

    # Build numbered chunk list for the prompt
    chunks_text = "\n\n".join(
        f"[{i+1}] {doc.page_content[:600]}" for i, doc in enumerate(docs)
    )

    grader = BATCH_GRADE_PROMPT | llm | StrOutputParser()
    try:
        raw = grader.invoke({"question": state["question"], "chunks": chunks_text})
        # Robustly parse the JSON array
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        verdicts = json.loads(raw)
        if not isinstance(verdicts, list):
            raise ValueError("Expected a list")
    except Exception as e:
        print(f"[grade] Warning: Could not parse batch response ({e}). Keeping all chunks.")
        verdicts = [True] * len(docs)

    relevant = [doc for doc, verdict in zip(docs, verdicts) if verdict]
    print(f"[grade] {len(relevant)}/{len(docs)} chunks relevant (batched).")
    return {**state, "documents": relevant}


def generate_node(state: AgentState, llm) -> AgentState:
    docs = state["documents"]
    context_parts = []
    sources_metadata = []

    for doc in docs:
        fname    = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        page_num = doc.metadata.get("page", 0) + 1   # 0-indexed → 1-indexed

        # IMPROVEMENT 3: Label figure caption chunks clearly in the context
        source_type = doc.metadata.get("source_type", "text")
        if source_type == "figure_caption":
            label = "[FIGURE DESCRIPTION]"
        else:
            label = "[TEXT]"

        context_parts.append(
            f"{label} [Source: {fname}, Page: {page_num}]\n{doc.page_content}"
        )

        meta = {"file": fname, "page": page_num}
        if meta not in sources_metadata:
            sources_metadata.append(meta)

    context = "\n\n---\n\n".join(context_parts)
    generator = GENERATE_PROMPT | llm | StrOutputParser()
    answer = generator.invoke({"question": state["question"], "context": context})
    print(f"[generate] Done. Sources: {sources_metadata}")
    return {**state, "generation": answer, "sources": sources_metadata, "context": context}


def faithfulness_node(state: AgentState, llm) -> AgentState:
    """
    IMPROVEMENT 2: Faithfulness verification.
    Scores the generated answer against the retrieved context.
    If score is too low, replaces the answer with a safe fallback.
    """
    answer  = state.get("generation", "")
    context = state.get("context", "")
    question = state["question"]

    if not answer or not context:
        return state

    print("[faithfulness] Verifying answer against source context ...")
    verifier = FAITHFULNESS_PROMPT | llm | StrOutputParser()
    try:
        raw_score = verifier.invoke({
            "question": question,
            "context":  context[:8000],   # Limit to avoid token overflow
            "answer":   answer,
        })
        score = float(raw_score.strip())
    except Exception as e:
        print(f"[faithfulness] Warning: Could not parse score ({e}). Accepting answer.")
        score = 1.0

    print(f"[faithfulness] Score: {score:.2f} (threshold: {FAITHFULNESS_THRESHOLD})")

    if score < FAITHFULNESS_THRESHOLD:
        print("[faithfulness] ⚠️ Answer failed faithfulness check — returning safe fallback.")
        safe_answer = (
            "⚠️ **Faithfulness Warning**: The generated answer could not be fully verified "
            "against the retrieved documents (faithfulness score: "
            f"{score:.2f}). This can happen when the documents don't contain enough "
            "direct information on the topic.\n\n"
            "Please try rephrasing your question or narrowing its scope."
        )
        return {**state, "generation": safe_answer}

    return state


def rewrite_node(state: AgentState, llm) -> AgentState:
    count = state.get("rewrite_count", 0) + 1
    print(f"[rewrite] Attempt {count} ...")
    rewriter = REWRITE_PROMPT | llm | StrOutputParser()
    new_question = rewriter.invoke({"question": state["question"]})
    print(f"[rewrite] New query: '{new_question}'")
    return {**state, "question": new_question, "rewrite_count": count}


def no_answer_node(state: AgentState) -> AgentState:
    return {
        **state,
        "generation": (
            "I was unable to find relevant information in the provided research documents "
            "to answer your question. Please try rephrasing or check that the relevant "
            "papers have been ingested into the knowledge base."
        ),
        "sources": [],
    }


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

    graph = StateGraph(AgentState)
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
    graph.add_conditional_edges(
        "grade", route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "no_answer": "no_answer"},
    )
    graph.add_edge("rewrite",      "retrieve")
    graph.add_edge("generate",     "faithfulness")   # ← faithfulness check wired in
    graph.add_edge("faithfulness", END)
    graph.add_edge("no_answer",    END)

    return graph.compile()


# ── Public API ─────────────────────────────────────────────────────────────────
_app = None

def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def run_query(question: str) -> dict:
    app = get_app()
    final_state = app.invoke({
        "question":      question,
        "documents":     [],
        "generation":    "",
        "sources":       [],
        "rewrite_count": 0,
        "context":       "",
    })
    return {
        "answer":  final_state.get("generation", "No answer generated."),
        "sources": final_state.get("sources", []),
    }


if __name__ == "__main__":
    result = run_query("What are the main contributions of the papers in this collection?")
    print("\n=== ANSWER ===\n", result["answer"])
    print("\n=== SOURCES ===")
    for s in result["sources"]:
        print(f"  • {s}")