"""
Tests for agents/rag_agent.py: answer generation and citation formatting.
Mocks the LLM and the retrieve() boundary (which already has its own tests
in tests/test_retrieve.py) so these run fast/free without a real
DEEPSEEK_API_KEY.

Run with:
    venv\\Scripts\\python.exe -m pytest tests/test_rag_agent.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

import rag_agent  # noqa: E402


class _FakeChatModel:
    def __init__(self, reply: str):
        self._reply = reply

    def invoke(self, prompt: str):
        return SimpleNamespace(content=self._reply)


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


# --- generate_answer_from_chunks (mocked LLM) ---

def test_generate_answer_from_chunks_returns_llm_reply(monkeypatch):
    monkeypatch.setattr(rag_agent, "get_chat_model", lambda: _FakeChatModel("Refunds are allowed within 30 days."))
    chunks = [_make_chunk("company_policies_western_capital.pdf", 0, "Refund within 30 days.")]
    answer = rag_agent.generate_answer_from_chunks("What is the refund window?", chunks)
    assert answer == "Refunds are allowed within 30 days."


# --- answer_policy_question end-to-end ---

def test_answer_policy_question(monkeypatch):
    chunks = [_make_chunk("company_policies_western_capital.pdf", 0, "Refund within 30 days.")]
    monkeypatch.setattr(rag_agent, "retrieve_relevant_chunks", lambda q, k=3: chunks)
    monkeypatch.setattr(rag_agent, "get_chat_model", lambda: _FakeChatModel("You can get a refund within 30 days."))

    result = rag_agent.answer_policy_question("What is the refund policy?")

    assert result["answer"] == "You can get a refund within 30 days."
    assert result["sources"] == [{"source_file": "company_policies_western_capital.pdf", "page": 1}]


def test_answer_policy_question_no_relevant_chunks(monkeypatch):
    """Even with no useful chunks, the answer should come from the LLM
    prompt's "say so if the sections don't cover it" instruction, not a
    hardcoded guardrail message - there's no category-based out-of-scope
    check at this layer anymore (that's now the ReAct agent's job)."""
    monkeypatch.setattr(rag_agent, "retrieve_relevant_chunks", lambda q, k=3: [])
    monkeypatch.setattr(
        rag_agent, "get_chat_model",
        lambda: _FakeChatModel("I don't have information about that in our policy documents."),
    )

    result = rag_agent.answer_policy_question("What's the capital of France?")

    assert result["sources"] == []
    assert "don't have information" in result["answer"]
