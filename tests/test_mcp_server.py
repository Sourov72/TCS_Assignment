"""
Tests for mcp_server/server.py.

Two layers:
  - Unit tests call the tool functions directly (mocked underlying agents,
    fast/free, no DEEPSEEK_API_KEY needed) - these always run in CI.
  - One real integration test spawns the server as an actual subprocess and
    talks to it over the real MCP stdio protocol using the MCP client SDK,
    proving the tools are genuinely reachable as an MCP server - not just
    callable as plain Python functions. Needs a real DEEPSEEK_API_KEY and
    is skipped automatically if one isn't configured.

Run with:
    venv\\Scripts\\python.exe -m pytest tests/test_mcp_server.py -v
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp_server"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

import server  # noqa: E402
from config import DEEPSEEK_API_KEY  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --- unit tests: call the tool functions directly, mocked agents ---

def test_query_customer_data_returns_answer(monkeypatch):
    monkeypatch.setattr(server, "answer_sql_question", lambda q: {"answer": "Ema is on the Pro plan."})
    assert server.query_customer_data("What plan is Ema on?") == "Ema is on the Pro plan."


def test_search_policy_documents_appends_sources(monkeypatch):
    monkeypatch.setattr(
        server, "answer_policy_question",
        lambda q: {
            "answer": "Refunds are allowed within 30 days.",
            "sources": [{"source_file": "refund_nyc_dca.pdf", "page": 1}],
        },
    )
    result = server.search_policy_documents("What is the refund policy?")
    assert "Refunds are allowed within 30 days." in result
    assert "refund_nyc_dca.pdf (page 1)" in result


def test_search_policy_documents_no_sources_when_out_of_scope(monkeypatch):
    monkeypatch.setattr(
        server, "answer_policy_question",
        lambda q: {"answer": "I can only answer policy questions.", "sources": []},
    )
    result = server.search_policy_documents("What's the weather?")
    assert result == "I can only answer policy questions."
    assert "Sources:" not in result


def test_ask_support_assistant_passes_session_id(monkeypatch):
    captured = {}

    def fake_ask(question, thread_id):
        captured["question"] = question
        captured["thread_id"] = thread_id
        return "the answer"

    monkeypatch.setattr(server, "ask_assistant", fake_ask)
    result = server.ask_support_assistant("hello", session_id="user-42")
    assert result == "the answer"
    assert captured == {"question": "hello", "thread_id": "user-42"}


# --- integration test: real subprocess + real MCP protocol + real API ---

@pytest.mark.skipif(not DEEPSEEK_API_KEY, reason="requires a real DEEPSEEK_API_KEY")
def test_mcp_server_end_to_end_over_stdio():
    """Spawns the actual server.py as a subprocess and talks MCP over stdio -
    proves the tools are genuinely reachable through the protocol."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(PROJECT_ROOT / "mcp_server" / "server.py")],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                tool_names = {tool.name for tool in tools.tools}
                assert {
                    "query_customer_data",
                    "search_policy_documents",
                    "ask_support_assistant",
                }.issubset(tool_names)

                result = await session.call_tool(
                    "search_policy_documents", {"question": "What is the current refund policy?"}
                )
                answer_text = result.content[0].text
                assert "refund" in answer_text.lower()

    asyncio.run(run())
