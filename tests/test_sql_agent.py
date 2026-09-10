"""
Tests for agents/sql_agent.py: SQL guardrails and execution against the real
seeded db/support.db.

Run with:
    venv\\Scripts\\python.exe -m pytest tests/test_sql_agent.py -v
"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "db"))

import sql_agent


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
