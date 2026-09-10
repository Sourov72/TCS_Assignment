"""
Tests for app.py using Streamlit's official headless testing framework
(streamlit.testing.v1.AppTest), which actually executes the script and lets
us inspect what it rendered - without needing a real browser.

Kept intentionally light: this checks the app boots cleanly and the missing-
API-key guardrail works. Deeper behavior (chat answers, PDF ingestion) is
already covered by tests/test_graph.py, tests/test_sql_agent.py,
tests/test_rag_agent.py, and tests/test_rag_ingest.py - re-simulating full
chat interactions here would mostly re-test that same logic through a much
slower, more fragile harness.

Run with:
    venv\\Scripts\\python.exe -m pytest tests/test_app.py -v
"""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(PROJECT_ROOT / "app.py")

sys.path.insert(0, str(PROJECT_ROOT / "common"))

from config import DEEPSEEK_API_KEY  # noqa: E402


def test_app_loads_without_exceptions():
    """A basic run should complete with no uncaught exceptions and render
    the expected title - catches import errors and st API misuse."""
    at = AppTest.from_file(APP_PATH)
    # Generous timeout: the first run in a test session pays the cold-import
    # cost of langchain/langgraph/torch, which can comfortably exceed
    # AppTest's 3s default - that's import latency, not an app bug.
    at.run(timeout=60)

    assert at.exception == []
    assert any("Customer Support Copilot" in title.value for title in at.title)


def test_app_shows_error_when_api_key_missing(monkeypatch):
    """If DEEPSEEK_API_KEY isn't configured, the app should show a clear
    error and stop, rather than crash further down with a confusing error
    from the LLM client."""
    import config

    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert at.exception == []
    assert any("DEEPSEEK_API_KEY" in error.value for error in at.error)


@pytest.mark.skipif(not DEEPSEEK_API_KEY, reason="requires a real DEEPSEEK_API_KEY")
def test_app_answers_a_real_chat_question():
    """Real end-to-end check: submit an actual chat message through the
    running app (not just direct function calls) and confirm a sensible
    answer renders, with no exceptions along the way."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)

    at.chat_input[0].set_value("Give me an overview of customer Ema's profile.").run(timeout=60)

    assert at.exception == []
    assistant_messages = [msg for msg in at.chat_message if msg.name == "assistant"]
    assert len(assistant_messages) == 1
    answer_text = assistant_messages[0].markdown[0].value
    assert "Ema" in answer_text
