"""
Tests for agents/graph.py - the ReAct agent that reasons about which
tool(s) to call and remembers conversation history per thread_id.

These are real, live tests against the actual DeepSeek API (mocking the
LLM wouldn't prove the agent actually behaves safely, just that the code
passes through whatever a fake reply says). Skipped automatically without
a real DEEPSEEK_API_KEY.

Run with (requires DEEPSEEK_API_KEY in .env):
    venv\\Scripts\\python.exe -m pytest tests/test_graph.py -v -s
"""

import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from config import DEEPSEEK_API_KEY  
from graph import ask 

pytestmark = pytest.mark.skipif(not DEEPSEEK_API_KEY, reason="requires a real DEEPSEEK_API_KEY")


def _new_thread() -> str:
    """A fresh thread_id per test, so tests never share conversation memory."""
    return f"test-{uuid.uuid4().hex[:8]}"


def test_ask_answers_a_sql_question():
    answer = ask("What plan is Ema Thompson on?", thread_id=_new_thread())
    assert "pro" in answer.lower()


def test_ask_answers_a_policy_question(seeded_index):
    answer = ask("What is the current refund policy?", thread_id=_new_thread())
    assert "refund" in answer.lower()


def test_ask_declines_out_of_scope_question():
    answer = ask("What's the capital of France?", thread_id=_new_thread())
    assert "customer" in answer.lower() or "polic" in answer.lower()
    assert "paris" not in answer.lower()


def test_ask_asks_for_clarification_on_unresolved_pronoun():
    """A question referring to a customer ambiguously ("she") with no
    customer established yet should get a clarifying question, not a guess."""
    answer = ask(
        "Is she eligible for a refund on her canceled order, based on our policy?",
        thread_id=_new_thread(),
    )
    assert "name, email" in answer.lower()  # asks for an identifier
    assert "sherry" not in answer.lower()  # must not guess a random unrelated customer


def test_ask_remembers_conversation_and_combines_both_tools(seeded_index):
    """After establishing a customer earlier in the conversation, a
    follow-up question needing both customer data and policy content should
    resolve the pronoun and answer using both - not ask for clarification."""
    thread_id = _new_thread()
    ask("Give me an overview of customer Ema's profile.", thread_id=thread_id)
    answer = ask(
        "Is she eligible for a refund on her canceled order, based on our policy?",
        thread_id=thread_id,
    )
    assert "name, email" not in answer.lower()  # should NOT ask for clarification this time
    assert "refund" in answer.lower()  # policy-side content
    assert "ema" in answer.lower()  # customer-side content
