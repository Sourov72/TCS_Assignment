"""
MCP server exposing this project's customer-support capabilities as tools,
so any MCP-compatible client (Claude Desktop, an MCP inspector, another
agent) can use them:

  - query_customer_data: NL question over the SQL customer/ticket database.
  - search_policy_documents: NL question over the policy PDF knowledge base.
  - ask_support_assistant: the full multi-agent assistant - automatically
    routes between the two above (or both), remembers conversation history
    per session_id.

Each tool is a thin wrapper around the already-built and independently
tested agents (agents/sql_agent.py, agents/rag_agent.py, agents/graph.py) -
no logic is reimplemented here, so this server exposes exactly the same,
already-verified behavior over the MCP protocol instead of duplicating it.

Run with:
    venv\\Scripts\\python.exe mcp_server/server.py            # stdio (for Claude Desktop, an MCP inspector, etc.)
    venv\\Scripts\\python.exe mcp_server/server.py streamable-http   # HTTP (used by app.py - see agents/mcp_client.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))
sys.path.insert(0, str(Path(__file__).parent.parent / "common"))

from graph import ask as ask_assistant
from mcp.server.mcpserver import MCPServer
from rag_agent import answer_policy_question
from sql_agent import answer_sql_question

mcp_server = MCPServer(
    name="customer-support-copilot",
    description=(
        "Query customer profiles/support ticket history and company policy "
        "documents for customer support."
    ),
)


@mcp_server.tool()
def query_customer_data(question: str) -> str:
    """Answer a natural-language question about a customer's profile or
    their support ticket history, by querying the structured SQL database.

    Example: "Give me an overview of customer Ema's profile and past
    support ticket details."
    """
    result = answer_sql_question(question)
    return result["answer"]


@mcp_server.tool()
def search_policy_documents(question: str) -> str:
    """Answer a natural-language question about company policy by searching
    the policy PDF knowledge base. Includes source document + page
    citations when available.

    Example: "What is the current refund policy?"
    """
    result = answer_policy_question(question)
    answer = result["answer"]
    if result["sources"]:
        citations = ", ".join(
            f"{source['source_file']} (page {source['page']})" for source in result["sources"]
        )
        answer += f"\n\nSources: {citations}"
    return answer


@mcp_server.tool()
def ask_support_assistant(question: str, session_id: str = "default") -> str:
    """Ask the full customer support assistant a question. Automatically
    routes to customer data, policy documents, both, or neither, and
    remembers the conversation history for the given session_id across calls.
    """
    return ask_assistant(question, thread_id=session_id)


HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8933  # arbitrary fixed port, used by agents/mcp_client.py to connect

if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp_server.run(transport=transport, host=HTTP_HOST, port=HTTP_PORT)
