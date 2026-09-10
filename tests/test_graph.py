"""
Tests for agents/graph.py - the ReAct agent that reasons about which
tool(s) to call and remembers conversation history per thread_id.

This agent's key behaviors -
deciding whether it has enough information to call a tool, deciding when to
use both tools together, declining out-of-scope questions - are genuine LLM
reasoning, not deterministic code paths. Mocking the LLM would only prove
"the wrapper passes through whatever the LLM said," not "the system prompt
actually produces safe behavior" - so these are real, live tests against
the actual DeepSeek API, skipped automatically without a real
DEEPSEEK_API_KEY (same pattern as the MCP/Streamlit/eval live tests).

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


def test_ask_answers_a_policy_question():
    answer = ask("What is the current refund policy?", thread_id=_new_thread())
    assert "refund" in answer.lower()


def test_ask_declines_out_of_scope_question():
    answer = ask("What's the capital of France?", thread_id=_new_thread())
    assert "customer" in answer.lower() or "polic" in answer.lower()
    assert "paris" not in answer.lower()


def test_ask_asks_for_clarification_on_unresolved_pronoun():
    """When a question refers to a customer ambiguously (e.g. "she") with
    no customer established in the conversation, the agent should ask for
    clarification instead of guessing - the ReAct agent's system prompt
    explicitly tells it to do this."""
    answer = ask(
        "Is she eligible for a refund on her canceled order, based on our policy?",
        thread_id=_new_thread(),
    )
    # Exact phrasing varies run to run (confirmed across several live runs),
    # but the agent consistently asks for an identifier this way - that's
    # the stable signal to check for, not one exact sentence.
    assert "name, email" in answer.lower()
    assert "sherry" not in answer.lower()  # must not guess a random unrelated customer


def test_ask_remembers_conversation_and_resolves_pronoun_after_context():
    """Same ambiguous question as above, but this time asked *after*
    establishing which customer is being discussed - should now actually
    attempt an answer instead of asking for clarification."""
    thread_id = _new_thread()
    ask("Give me an overview of customer Ema's profile.", thread_id=thread_id)
    answer = ask(
        "Is she eligible for a refund on her canceled order, based on our policy?",
        thread_id=thread_id,
    )
    assert "ema" in answer.lower()
    assert "name, email" not in answer.lower()  # should NOT ask for clarification this time


def test_ask_combines_both_tools_for_cross_referencing_question():
    """A question needing both customer data and policy content should
    reference a concrete fact from each - not just one or the other."""
    thread_id = _new_thread()
    ask("Give me an overview of customer Ema's profile.", thread_id=thread_id)
    answer = ask(
        "Is she eligible for a refund on her canceled order, based on our policy?",
        thread_id=thread_id,
    )
    assert "refund" in answer.lower()  # policy-side content
    assert "ema" in answer.lower() or "ticket" in answer.lower()  # customer-side content
