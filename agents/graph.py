"""
The multi-agent system: a ReAct agent that reasons about which tool(s) it
needs (customer database, policy documents, both, or neither) and calls
them itself, rather than a hand-written router classifying the question
up front.

Why ReAct instead of a fixed router -> agent(s) -> combine pipeline: a
fixed pipeline has no way to reconsider or validate before acting - it
would generate a SQL query for whatever standalone question it's given,
even one that refers to a customer ambiguously. A ReAct loop can reason
about whether it actually has enough information *before* calling a tool,
and its system prompt explicitly tells it to ask for clarification instead
of guessing when a question refers to someone ambiguously.

As a bonus, a true reasoning loop can call one tool, read its result, and
then call the second tool informed by what it learned - genuinely
cross-referencing customer data with policy content, which independent
SQL/RAG calls in a fixed pipeline couldn't do.

Still built on LangGraph under the hood (langchain.agents.create_agent
compiles to a LangGraph graph) rather than hand-written nodes/edges - this
prebuilt agent already implements the ReAct tool-calling loop plus
per-thread memory (via a MemorySaver checkpointer), so there's no need to
hand-roll a graph for it.

Run with (manual smoke test, requires DEEPSEEK_API_KEY in .env):
    venv\\Scripts\\python.exe agents/graph.py "Give me an overview of customer Ema's profile and past support ticket details."
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "common"))

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from llm import get_chat_model
from tools import query_customer_database, search_policy_documents

SYSTEM_PROMPT = """You are a customer support assistant with two tools:
- query_customer_database: for questions about a specific customer's \
profile or support ticket history.
- search_policy_documents: for questions about company policy (e.g. \
refunds, shipping, privacy, warranty, or anything else covered by an \
uploaded policy document).

Rules:
1. Use a tool only when you have enough concrete information to do so. If \
a question refers to a person via an unresolved pronoun (he/she/they) and \
no specific customer has been named earlier in this conversation, do NOT \
guess or call a tool - ask the user to clarify who they mean instead.
2. If a question needs both customer data and policy information, call \
both tools (in whichever order makes sense) and combine what you learn \
into one coherent answer - use facts from one tool's result to inform how \
you use the other, where relevant.
3. If a question is unrelated to customer data or company policy (e.g. \
general knowledge, small talk), politely say you can only help with those \
two things - do not call any tool.
4. Never state a fact that isn't supported by a tool's result. If a tool's \
result doesn't answer the question, say so plainly rather than guessing.
"""

TOOLS = [query_customer_database, search_policy_documents]


def _build_agent():
    """Compile the ReAct agent with the shared LLM, tools, system prompt,
    and an in-memory checkpointer (gives each thread_id its own remembered
    conversation history)."""
    return create_agent(
        model=get_chat_model(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )


agent = _build_agent()


def ask(question: str, thread_id: str = "default") -> str:
    """Run one conversation turn for a given thread_id (session) and return
    the final answer text. Prior turns for the same thread_id are
    remembered automatically by the agent's checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}
    result_state = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    return result_state["messages"][-1].content


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]) or (
        "Give me an overview of customer Ema's profile and past support ticket details."
    )
    print(f"Question: {question}")
    print(f"Answer:\n{ask(question)}")
