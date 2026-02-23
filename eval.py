"""
eval.py — Evaluation Pipeline for Agentic RAG
Run with: python eval.py

This script runs a set of "Golden" test questions through the RAG pipeline
and uses an LLM-as-a-judge to grade Context Relevance and Answer Faithfulness/Accuracy.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from agent_logic import run_query, _build_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Golden Dataset ────────────────────────────────────────────────────────────
# Replace these with actual questions and expected facts from your PDFs.
GOLDEN_DATASET = [
    {
        "question": "What are the main contributions of the research papers in this collection?",
        "expected_facts": "Should mention key algorithms, methodologies, or findings discussed in the papers."
    },
    {
        "question": "What is the specific performance or accuracy reported?",
        "expected_facts": "Should quote specific numbers or percentage improvements."
    }
]

# ── Evaluator LLM ─────────────────────────────────────────────────────────────
llm = _build_llm()

EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an impartial AI judge evaluating a RAG system.\n"
     "Given a 'Question', an 'Expected Fact', and the 'Generated Answer', grade the answer.\n"
     "Score 1: The Answer contains the expected facts and is accurate.\n"
     "Score 0: The Answer is missing the expected facts, is irrelevant, or is a hallucination.\n"
     "Output ONLY the integer 1 or 0."),
    ("human", "Question: {question}\nExpected: {expected_facts}\nAnswer: {answer}")
])

eval_chain = EVAL_PROMPT | llm | StrOutputParser() if True else None  # Delay initialization
# Fixed compilation error above by just putting EVAL_PROMPT... wait, evaluating in function is safer.

def run_eval_pipeline():
    eval_chain = EVAL_PROMPT | llm | StrOutputParser()
    
    print(f"Starting Evaluation on {len(GOLDEN_DATASET)} queries...\n" + "-"*50)
    
    total_score = 0
    for i, item in enumerate(GOLDEN_DATASET):
        q = item["question"]
        expected = item["expected_facts"]
        
        print(f"Query {i+1}: {q}")
        
        # 1. Run Pipeline
        try:
            result = run_query(q)
            answer = result["answer"]
            sources = result["sources"]
        except Exception as e:
            print(f"  Pipeline failed: {e}")
            continue
            
        print(f"  Sources Retrieved: {len(sources)}")
        
        # 2. Evaluate
        try:
            score_str = eval_chain.invoke({
                "question": q,
                "expected_facts": expected,
                "answer": answer
            })
            # Clean LLM response just in case
            import re
            match = re.search(r'\b(0|1)\b', score_str)
            score = int(match.group(1)) if match else 0
        except Exception as e:
            print(f"  Eval LLM failed, assuming 0. Error: {e}")
            score = 0
            
        total_score += score
        print(f"  Score: {score}/1\n")
        
    accuracy = (total_score / len(GOLDEN_DATASET)) * 100
    print("-" * 50)
    print(f"Final Evaluation Accuracy: {accuracy:.1f}% ({total_score}/{len(GOLDEN_DATASET)})")
    print("If accuracy is low, consider adjusting Top-K, chunk overlap, or adding more documents.")

if __name__ == "__main__":
    run_eval_pipeline()
