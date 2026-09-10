"""
Tests for agents/tools.py - the LangChain tool wrappers around the SQL and
RAG agents. Mocks the underlying agent functions (already tested in
tests/test_sql_agent.py / tests/test_rag_agent.py), so these are fast/free
and only check the wrapping/formatting logic itself.

Run with:
    venv\\Scripts\\python.exe -m pytest tests/test_tools.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

import tools


def test_query_customer_database_returns_sql_agent_answer(monkeypatch):
    monkeypatch.setattr(tools, "answer_sql_question", lambda q: {"answer": "Ema is on the Pro plan."})
    result = tools.query_customer_database.invoke({"question": "What plan is Ema on?"})
    assert result == "Ema is on the Pro plan."


def test_search_policy_documents_appends_citations(monkeypatch):
    monkeypatch.setattr(
        tools, "answer_policy_question",
        lambda q: {
            "answer": "Refunds must be requested within a reasonable window.",
            "sources": [{"source_file": "refund_nyc_dca.pdf", "page": 1}],
        },
    )
    result = tools.search_policy_documents.invoke({"question": "What is the refund policy?"})
    assert "Refunds must be requested within a reasonable window." in result
    assert "refund_nyc_dca.pdf (page 1)" in result


def test_search_policy_documents_no_citations_when_out_of_scope(monkeypatch):
    monkeypatch.setattr(
        tools, "answer_policy_question",
        lambda q: {"answer": "I can only answer policy questions.", "sources": []},
    )
    result = tools.search_policy_documents.invoke({"question": "What's the weather?"})
    assert result == "I can only answer policy questions."
    assert "Sources:" not in result
