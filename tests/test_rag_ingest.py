"""
Tests for the RAG ingestion pipeline (rag/ingest.py) - a single combined
FAISS index built from every PDF in data/policies/.

Run with:
    venv\\Scripts\\python.exe -m pytest tests/test_rag_ingest.py -v
"""

import sys
from pathlib import Path

# Make rag/ importable as plain modules when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

import ingest  # noqa: E402
from ingest import build_index  # noqa: E402

KNOWN_PDF = "company_policies_western_capital.pdf"
EXPECTED_CHUNK_COUNT = 16  # company_policies_western_capital.pdf: 6 pages -> 16 chunks


def test_index_builds_from_every_pdf_in_policies_dir():
    """The corpus is discovered dynamically (glob), not from a hardcoded
    list - this just confirms every *.pdf currently in data/policies/ ended
    up in the index."""
    index = build_index()
    sources = {
        doc.metadata["source_file"] for doc in index.docstore._dict.values()
    }
    expected_sources = {p.name for p in ingest.POLICIES_DIR.glob("*.pdf")}
    assert sources == expected_sources


def test_known_pdf_chunk_count_matches_expected():
    """Catches silent regressions in chunking parameters, checked against
    the one PDF we know the exact expected count for. Deliberately doesn't
    assert the *total* index size, since data/policies/ - and therefore the
    index - can legitimately grow at any time via the upload feature; a
    hardcoded grand total would break the moment someone uploads a PDF."""
    index = build_index()
    chunks_for_known_pdf = [
        doc for doc in index.docstore._dict.values()
        if doc.metadata["source_file"] == KNOWN_PDF
    ]
    assert len(chunks_for_known_pdf) == EXPECTED_CHUNK_COUNT


def test_every_chunk_has_citation_metadata():
    """Every chunk must carry source_file + page metadata - needed for the
    RAG agent's citations."""
    index = build_index()
    for doc in index.docstore._dict.values():
        assert "source_file" in doc.metadata
        assert "page" in doc.metadata


def test_add_pdf_to_index_increases_chunk_count(tmp_path, monkeypatch):
    """add_pdf_to_index() should merge a new PDF's chunks into the existing
    combined index on disk. Uses an isolated temp copy of the index (via
    monkeypatching INDEX_DIR) so this test never mutates the real,
    persisted demo vectorstore."""
    monkeypatch.setattr(ingest, "INDEX_DIR", tmp_path)
    build_index().save_local(str(tmp_path))

    original_count = ingest.load_index().index.ntotal

    # Re-ingest the same PDF again as a stand-in for a freshly "uploaded" one.
    pdf_path = next(ingest.POLICIES_DIR.glob("*.pdf"))
    added = ingest.add_pdf_to_index(pdf_path)

    updated_count = ingest.load_index().index.ntotal
    assert added > 0
    assert updated_count == original_count + added


def test_is_duplicate_pdf_detects_same_content():
    """Uploading a PDF whose content matches an existing file (even under a
    different filename) should be detected as a duplicate, by hashing
    content rather than comparing filenames."""
    existing_pdf = next(ingest.POLICIES_DIR.glob("*.pdf"))
    same_content = existing_pdf.read_bytes()
    assert ingest.is_duplicate_pdf(same_content) is True


def test_is_duplicate_pdf_returns_false_for_new_content():
    assert ingest.is_duplicate_pdf(b"this is not a real pdf, just new bytes") is False


def test_refund_query_retrieves_relevant_chunk_with_citation_metadata():
    """An actual relevance + citation-metadata check for a refund question,
    against the single combined index."""
    index = build_index()
    results = index.similarity_search("What is the current refund policy?", k=1)
    assert len(results) == 1
    top_result = results[0]
    assert "refund" in top_result.page_content.lower()
    assert top_result.metadata["source_file"].endswith(".pdf")
    assert "page" in top_result.metadata  # needed for citations later
