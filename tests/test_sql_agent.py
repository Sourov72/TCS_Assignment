"""
Tests for agents/sql_agent.py: SQL guardrails and execution against the real
seeded db/support.db, plus generation/summarization with a mocked LLM
(fast/free, no real DEEPSEEK_API_KEY needed - important for CI).

Run with:
    venv\\Scripts\\python.exe -m pytest tests/test_sql_agent.py -v
"""

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "db"))

import sql_agent  


class _FakeChatModel:
    """Stand-in for the real DeepSeek chat model."""

    def __init__(self, reply: str):
        self._reply = reply

    def invoke(self, prompt: str):
        return SimpleNamespace(content=self._reply)


# --- is_safe_select_query guardrail ---

def test_plain_select_is_safe():
    assert sql_agent.is_safe_select_query("SELECT * FROM customers") is True


def test_drop_table_is_rejected():
    assert sql_agent.is_safe_select_query("DROP TABLE customers") is False


def test_stacked_statements_are_rejected():
    assert sql_agent.is_safe_select_query(
        "SELECT * FROM customers; DELETE FROM customers"
    ) is False


def test_legitimate_query_with_created_at_column_is_not_falsely_blocked():
    """Regression test: a naive substring check on 'CREATE' would false-
    positive on the legitimate column name 'created_at'. This is exactly
    what the live smoke test against the real DeepSeek API hit first try."""
    assert sql_agent.is_safe_select_query(
        "SELECT full_name, created_at FROM customers c "
        "JOIN support_tickets t ON c.customer_id = t.customer_id"
    ) is True


def test_forbidden_keyword_anywhere_in_query_is_rejected():
    assert sql_agent.is_safe_select_query(
        "SELECT * FROM customers WHERE customer_id IN "
        "(SELECT customer_id FROM customers); UPDATE customers SET plan_tier='Free'"
    ) is False


# --- execute_sql_query against the real seeded database ---

def test_execute_sql_query_returns_ema():
    rows = sql_agent.execute_sql_query(
        "SELECT full_name, plan_tier FROM customers WHERE full_name = 'Ema Thompson'"
    )
    assert len(rows) == 1
    assert rows[0]["plan_tier"] == "Pro"


def test_execute_sql_query_read_only_connection_blocks_writes():
    """Even a query that somehow slipped past is_safe_select_query should
    still fail at the database layer, since the connection is read-only."""
    with pytest.raises(sqlite3.OperationalError):
        sql_agent.execute_sql_query("DELETE FROM customers")


# --- generate_sql_query / summarize_results (mocked LLM) ---

def test_generate_sql_query_strips_markdown_fences(monkeypatch):
    monkeypatch.setattr(
        sql_agent, "get_chat_model",
        lambda: _FakeChatModel("```sql\nSELECT * FROM customers\n```"),
    )
    sql = sql_agent.generate_sql_query("show me all customers")
    assert sql == "SELECT * FROM customers"


def test_summarize_results_uses_llm_and_returns_text(monkeypatch):
    monkeypatch.setattr(
        sql_agent, "get_chat_model",
        lambda: _FakeChatModel("Ema is on the Pro plan."),
    )
    answer = sql_agent.summarize_results(
        "What plan is Ema on?", [{"full_name": "Ema Thompson", "plan_tier": "Pro"}]
    )
    assert "Pro" in answer


# --- answer_sql_question end-to-end ---

def test_answer_sql_question_end_to_end(monkeypatch):
    responses = iter([
        "SELECT full_name, plan_tier FROM customers WHERE full_name LIKE '%Ema%'",
        "Ema Thompson is on the Pro plan.",
    ])
    monkeypatch.setattr(sql_agent, "get_chat_model", lambda: _FakeChatModel(next(responses)))

    result = sql_agent.answer_sql_question("What plan is Ema on?")

    assert "Ema" in result["sql_query"]
    assert len(result["rows"]) == 1
    assert result["rows"][0]["plan_tier"] == "Pro"
    assert result["answer"] == "Ema Thompson is on the Pro plan."


def test_answer_sql_question_blocks_unsafe_generated_sql(monkeypatch):
    """If the LLM ever generates a destructive statement, it must never
    reach execute_sql_query - the response should explain it was blocked,
    with no rows returned."""
    monkeypatch.setattr(sql_agent, "get_chat_model", lambda: _FakeChatModel("DROP TABLE customers"))

    result = sql_agent.answer_sql_question("delete everything")

    assert result["rows"] == []
    assert "safely run" in result["answer"]
