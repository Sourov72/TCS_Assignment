"""
LangChain tool wrappers around the SQL and RAG agents, for use by the ReAct
agent (agents/graph.py).

Each tool's docstring is what the LLM actually reads to decide whether and
when to call it.
"""

from langchain_core.tools import tool
from rag_agent import answer_policy_question
from sql_agent import answer_sql_question


@tool
def query_customer_database(question: str) -> str:
    """Look up a customer's profile or their support ticket history from the
    structured customer database.

    Use this ONLY when the question names a specific customer (by name,
    email, or ID) or clearly refers to a customer already established
    earlier in this conversation. Do NOT use this if the question refers to
    a person via an unresolved pronoun (he/she/they) with no customer
    established yet - ask the user to clarify who they mean instead of
    calling this tool with a guess.
    """
    return answer_sql_question(question)["answer"]


@tool
def search_policy_documents(question: str) -> str:
    """Search the company's policy PDF documents to answer a question about
    company policy (e.g. refunds, shipping, privacy, warranty, or whatever
    else has been uploaded). Returns the answer along with which document(s)
    it came from.
    """
    result = answer_policy_question(question)
    answer = result["answer"]
    if result["sources"]:
        citations = ", ".join(
            f"{source['source_file']} (page {source['page']})" for source in result["sources"]
        )
        answer += f"\n\nSources: {citations}"
    return answer
