"""
MCP client used by app.py to talk to mcp_server/server.py over HTTP,
instead of calling agents/graph.py directly in-process.

Why HTTP (streamable-http) instead of stdio: Streamlit re-runs the whole
script on every user interaction, but a stdio MCP connection needs a
subprocess + its stdin/stdout pipes kept alive across those reruns - messy
to manage inside Streamlit's execution model. Running the MCP server as its
own long-lived HTTP process sidesteps that entirely: app.py just makes an
HTTP-based MCP call per question, no subprocess lifecycle to juggle inside
the UI script itself.

To keep "just run streamlit" a one-command experience, ensure_server_running()
auto-launches the MCP server as a background process the first time it's
needed, if nothing is already listening on its port.

Run with (manual smoke test, requires DEEPSEEK_API_KEY in .env and the
server NOT already running elsewhere):
    venv\\Scripts\\python.exe agents/mcp_client.py "What is the current refund policy?"
"""

import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

SERVER_SCRIPT = Path(__file__).parent.parent / "mcp_server" / "server.py"
HOST = "127.0.0.1"
PORT = 8933  # must match mcp_server/server.py's HTTP_PORT
SERVER_URL = f"http://{HOST}:{PORT}/mcp"

STARTUP_TIMEOUT_SECONDS = 15
STARTUP_POLL_INTERVAL_SECONDS = 0.5


def _is_server_listening() -> bool:
    """Quick TCP-level check - just "is something bound to this port,"
    not a full MCP handshake (that happens per-call in call_tool)."""
    try:
        with socket.create_connection((HOST, PORT), timeout=0.5):
            return True
    except OSError:
        return False


def _start_server_in_background() -> None:
    """Launch mcp_server/server.py as a detached background process, over
    the HTTP transport."""
    subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT), "streamable-http"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def ensure_server_running() -> None:
    """Start the MCP server if it isn't already running, and wait until it's
    reachable. Safe to call every time - it's a no-op if already up."""
    if _is_server_listening():
        return

    _start_server_in_background()

    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _is_server_listening():
            return
        time.sleep(STARTUP_POLL_INTERVAL_SECONDS)

    raise RuntimeError(
        f"MCP server did not start within {STARTUP_TIMEOUT_SECONDS}s "
        f"(expected to be listening on {SERVER_URL})"
    )


async def _call_tool_async(tool_name: str, arguments: dict) -> str:
    """Open one MCP session over HTTP, call a single tool, return its text result."""
    async with streamable_http_client(SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result.content[0].text


def call_tool(tool_name: str, arguments: dict) -> str:
    """Synchronous wrapper around _call_tool_async, for use from Streamlit's
    synchronous script code."""
    return asyncio.run(_call_tool_async(tool_name, arguments))


def ask(question: str, thread_id: str = "default") -> str:
    """Drop-in replacement for agents/graph.py's ask() - same signature,
    but routed through the MCP server's ask_support_assistant tool instead
    of calling the ReAct agent in-process."""
    ensure_server_running()
    return call_tool("ask_support_assistant", {"question": question, "session_id": thread_id})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]) or "What is the current refund policy?"
    print(f"Question: {question}")
    print(f"Answer:\n{ask(question)}")
