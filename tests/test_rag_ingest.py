"""
Tests for the RAG ingestion pipeline (rag/ingest.py). data/policies/ ships
empty (see README - no PDF is pre-ingested), so these tests build their own
isolated policies folder from Example_PDF/ instead of depending on
data/policies/ having anything in it.

Run with:
    venv\\Scripts\\python.exe -m pytest tests/test_rag_ingest.py -v
"""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

import ingest
from ingest import build_index

from conftest import EXAMPLE_PDF

EXPECTED_CHUNK_COUNT = 16  # company_policies_western_capital.pdf: 6 pages -> 16 chunks


def _seeded_policies_dir(tmp_path: Path, monkeypatch) -> Path:
    """Copy the example PDF into an isolated temp folder and point
    ingest.POLICIES_DIR at it."""
    policies_dir = tmp_path / "policies"
    policies_dir.mkdir()
    shutil.copy(EXAMPLE_PDF, policies_dir / EXAMPLE_PDF.name)
    monkeypatch.setattr(ingest, "POLICIES_DIR", policies_dir)
    return policies_dir


def test_build_index_raises_clear_error_when_policies_dir_is_empty(tmp_path, monkeypatch):
    """This is the real starting state of data/policies/ - an empty folder
    should fail with a clear message, not a cryptic FAISS crash."""
    monkeypatch.setattr(ingest, "POLICIES_DIR", tmp_path)
    with pytest.raises(ValueError, match="No PDF files found"):
        build_index()


def test_build_index_produces_expected_chunks_and_metadata(tmp_path, monkeypatch):
    policies_dir = _seeded_policies_dir(tmp_path, monkeypatch)
    index = build_index()

    assert index.index.ntotal == EXPECTED_CHUNK_COUNT
    for doc in index.docstore._dict.values():
        assert doc.metadata["source_file"] == EXAMPLE_PDF.name
        assert "page" in doc.metadata
    assert {p.name for p in policies_dir.glob("*.pdf")} == {EXAMPLE_PDF.name}


def test_add_pdf_to_index_creates_index_when_none_exists(tmp_path, monkeypatch):
    """The very first upload (no index yet) should create a fresh index,
    not try to merge into a nonexistent one - this is the real starting
    state a user hits before ever uploading a PDF."""
    monkeypatch.setattr(ingest, "INDEX_DIR", tmp_path / "vectorstore")
    assert ingest.index_exists() is False

    added = ingest.add_pdf_to_index(EXAMPLE_PDF)

    assert added == EXPECTED_CHUNK_COUNT
    assert ingest.index_exists() is True


def test_add_pdf_to_index_merges_into_existing_index(tmp_path, monkeypatch):
    """A second upload should merge into (not replace) the existing index."""
    monkeypatch.setattr(ingest, "INDEX_DIR", tmp_path / "vectorstore")
    ingest.add_pdf_to_index(EXAMPLE_PDF)
    original_count = ingest.load_index().index.ntotal

    added = ingest.add_pdf_to_index(EXAMPLE_PDF)  # stand-in for a second upload

    assert ingest.load_index().index.ntotal == original_count + added


def test_is_duplicate_pdf_detects_same_content(tmp_path, monkeypatch):
    _seeded_policies_dir(tmp_path, monkeypatch)
    same_content = EXAMPLE_PDF.read_bytes()
    assert ingest.is_duplicate_pdf(same_content) is True


def test_is_duplicate_pdf_returns_false_for_new_content(tmp_path, monkeypatch):
    _seeded_policies_dir(tmp_path, monkeypatch)
    assert ingest.is_duplicate_pdf(b"this is not a real pdf, just new bytes") is False


def test_refund_query_retrieves_relevant_chunk_with_citation_metadata(tmp_path, monkeypatch):
    """An actual relevance + citation-metadata check for a refund question."""
    _seeded_policies_dir(tmp_path, monkeypatch)
    index = build_index()
    results = index.similarity_search("What is the current refund policy?", k=1)
    assert "refund" in results[0].page_content.lower()
    assert "page" in results[0].metadata  # needed for citations
