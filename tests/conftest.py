"""
Shared pytest fixtures.

data/policies/ ships empty on purpose (see README - the app starts with no
policy PDF ingested; a user uploads one via the UI). Tests that need real
RAG content to search use the seeded_index fixture below, which points
rag/ingest.py at an isolated temp index built from Example_PDF/ - the real
rag/vectorstore/ is never touched by the test suite.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

import ingest  # noqa: E402

EXAMPLE_PDF = Path(__file__).resolve().parent.parent / "Example_PDF" / "company_policies_western_capital.pdf"


@pytest.fixture
def seeded_index(tmp_path, monkeypatch):
    """Point ingest.INDEX_DIR at a temp folder seeded with the example PDF.
    Every function that loads the index (directly or via the agents/MCP
    server, since they all share this one `ingest` module) picks this up."""
    monkeypatch.setattr(ingest, "INDEX_DIR", tmp_path / "vectorstore")
    ingest.add_pdf_to_index(EXAMPLE_PDF)
