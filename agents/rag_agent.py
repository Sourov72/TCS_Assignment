"""
Natural-language policy question -> retrieval from the combined policy
index -> cited, natural-language answer.

Wraps rag/retrieve.py (plain similarity search over every ingested PDF)
and adds:
  - the final answer generation (LLM answers using ONLY the retrieved
    sections, so it can't hallucinate policy details, and is told to say so
    plainly if the sections don't actually cover the question - this is
    also what protects against a totally unrelated question reaching this
    tool, now that there's no category-based out-of-scope check here).
  - source citations built directly from chunk metadata (source_file, page)
    rather than asked from the LLM, so citations are always accurate
    instead of a model's guess.

Run with (manual smoke test, requires DEEPSEEK_API_KEY in .env):
    venv\\Scripts\\python.exe agents/rag_agent.py "What is the current refund policy?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))

from llm import get_chat_model
from retrieve import retrieve_relevant_chunks

RAG_ANSWER_PROMPT = """You are a helpful customer support assistant. Answer \
the user's question using ONLY the policy sections below. If the sections \
don't actually contain the answer, say so plainly - do not make anything up.

Question: "{question}"

Policy sections:
{context}

Answer concisely and in a friendly tone. Do not list citations or source \
file names in your answer text - those are shown separately."""


def _format_context(chunks: list) -> str:
    """Join retrieved chunks into one numbered context block for the prompt."""
    return "\n\n".join(
        f"[Section {i}]\n{chunk.page_content}" for i, chunk in enumerate(chunks, start=1)
    )


def _format_citations(chunks: list) -> list:
    """Build a deduplicated (source_file, page) citation list from chunk metadata.

    Pages from PyPDFLoader are 0-indexed, so +1 here for a human-readable
    page number in the UI.
    """
    seen = set()
    citations = []
    for chunk in chunks:
        source_file = chunk.metadata.get("source_file", "unknown")
        page = chunk.metadata.get("page")
        page_display = page + 1 if isinstance(page, int) else page
        key = (source_file, page_display)
        if key not in seen:
            seen.add(key)
            citations.append({"source_file": source_file, "page": page_display})
    return citations


def generate_answer_from_chunks(question: str, chunks: list) -> str:
    """Ask the LLM to answer the question using only the retrieved sections."""
    prompt = RAG_ANSWER_PROMPT.format(question=question, context=_format_context(chunks))
    response = get_chat_model().invoke(prompt)
    return response.content.strip()


def answer_policy_question(question: str, k: int = 3) -> dict:
    """End-to-end: retrieve from the combined index, then answer with citations."""
    chunks = retrieve_relevant_chunks(question, k=k)
    answer = generate_answer_from_chunks(question, chunks)
    return {
        "question": question,
        "chunks": chunks,
        "sources": _format_citations(chunks),
        "answer": answer,
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]) or "What is the current refund policy?"
    result = answer_policy_question(question)
    print(f"Question: {result['question']}")
    print(f"Answer:\n{result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for source in result["sources"]:
            print(f" - {source['source_file']} (page {source['page']})")
