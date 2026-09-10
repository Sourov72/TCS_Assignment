"""
Tests for agents/rag_agent.py: citation formatting.

Run with:
    venv\\Scripts\\python.exe -m pytest tests/test_rag_agent.py -v
"""

import sys
from pathlib import Path

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

import rag_agent  # noqa: E402


def _make_chunk(source_file: str, page: int, text: str) -> Document:
    return Document(page_content=text, metadata={"source_file": source_file, "page": page})


# --- citation formatting ---

def test_format_citations_dedups_same_source_and_page():
    chunks = [
        _make_chunk("company_policies_western_capital.pdf", 0, "text a"),
        _make_chunk("company_policies_western_capital.pdf", 0, "text b"),  # same source+page
    ]
    citations = rag_agent._format_citations(chunks)
    assert citations == [{"source_file": "company_policies_western_capital.pdf", "page": 1}]


def test_format_citations_converts_zero_indexed_page_to_human_readable():
    chunks = [_make_chunk("company_policies_western_capital.pdf", 2, "text")]
    citations = rag_agent._format_citations(chunks)
    assert citations[0]["page"] == 3  # 0-indexed page 2 -> displayed as page 3


def test_format_citations_keeps_distinct_sources_separate():
    chunks = [
        _make_chunk("policy_doc_a.pdf", 0, "text a"),
        _make_chunk("policy_doc_b.pdf", 0, "text b"),
    ]
    citations = rag_agent._format_citations(chunks)
    assert len(citations) == 2
