"""
Given a user's question about company policies, retrieves the most relevant
chunks from the single combined FAISS index (see rag/ingest.py - there is
no category classification step; an earlier per-category design was
removed as an unnecessary invention not asked for by the assignment).

Run with (manual smoke test):
    venv\\Scripts\\python.exe rag/retrieve.py "What is the current refund policy?"
"""

import sys

from ingest import index_exists, load_index


def retrieve_relevant_chunks(query: str, k: int = 3) -> list:
    """Return the top-k most relevant chunks from the combined policy index.
    Returns an empty list if no PDF has been ingested yet."""
    if not index_exists():
        return []
    index = load_index()
    return index.similarity_search(query, k=k)


if __name__ == "__main__":
    # Real-world PDFs can contain Unicode punctuation (smart quotes, special
    # hyphens) that Windows' default console encoding (cp1252) can't print.
    # That's a console-display limitation, not a data problem - replace
    # unprintable characters instead of letting the whole script crash.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    question = " ".join(sys.argv[1:]) or "What is the current refund policy?"
    chunks = retrieve_relevant_chunks(question)
    print(f"Question: {question}")
    for chunk in chunks:
        print("-", chunk.metadata.get("source_file"), "| page", chunk.metadata.get("page"))
        print("  ", chunk.page_content[:200].replace("\n", " "))
