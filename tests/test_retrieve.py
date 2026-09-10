"""
Tests for rag/retrieve.py - plain similarity search over the policy index.

Run with:
    venv\\Scripts\\python.exe -m pytest tests/test_retrieve.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

import retrieve  # noqa: E402


def test_retrieve_relevant_chunks_returns_empty_list_when_no_index_yet():
    """The real starting state - no PDF ingested yet - should return no
    results, not crash."""
    assert retrieve.retrieve_relevant_chunks("What is the refund policy?") == []


def test_retrieve_relevant_chunks_returns_requested_count(seeded_index):
    chunks = retrieve.retrieve_relevant_chunks("What is the refund policy?", k=2)
    assert len(chunks) == 2


def test_retrieve_relevant_chunks_are_relevant_to_the_query(seeded_index):
    chunks = retrieve.retrieve_relevant_chunks("What personal data do you collect?", k=3)
    assert any("information" in c.page_content.lower() or "data" in c.page_content.lower() for c in chunks)


def test_retrieve_relevant_chunks_have_citation_metadata(seeded_index):
    chunks = retrieve.retrieve_relevant_chunks("shipping and delivery", k=1)
    assert "source_file" in chunks[0].metadata
    assert "page" in chunks[0].metadata
