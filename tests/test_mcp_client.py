"""
Tests for agents/mcp_client.py - the HTTP MCP client app.py uses to talk to
mcp_server/server.py instead of calling agents/graph.py in-process.

Run with:
    venv\\Scripts\\python.exe -m pytest tests/test_mcp_client.py -v

Note: the live test below calls ensure_server_running(), which will leave a
background MCP HTTP server process running afterward if one wasn't already
up (by design - it's meant to persist across Streamlit reruns, not die when
the caller exits). Re-running this test won't spawn a second one, since
ensure_server_running() checks the port first.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

import mcp_client  # noqa: E402
from config import DEEPSEEK_API_KEY  # noqa: E402


def test_is_server_listening_false_when_nothing_bound(monkeypatch):
    def _raise_connection_error(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(mcp_client.socket, "create_connection", _raise_connection_error)
    assert mcp_client._is_server_listening() is False


def test_is_server_listening_true_when_connection_succeeds(monkeypatch):
    class _FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(mcp_client.socket, "create_connection", lambda *a, **k: _FakeSocket())
    assert mcp_client._is_server_listening() is True


@pytest.mark.skipif(not DEEPSEEK_API_KEY, reason="requires a real DEEPSEEK_API_KEY")
def test_ensure_server_running_and_ask_end_to_end():
    """Real integration test: starts (or reuses) the MCP HTTP server and
    asks it a real question through the full client/server round trip -
    proving app.py's chat path genuinely goes over MCP, not just that the
    underlying agent functions work."""
    mcp_client.ensure_server_running()
    answer = mcp_client.ask("What plan is Ema Thompson on?", thread_id="test-mcp-client")
    assert "pro" in answer.lower()
