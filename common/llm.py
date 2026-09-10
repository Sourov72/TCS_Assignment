"""
Provides the chat LLM used by all agents (query classification, SQL
generation, RAG answer generation).

Kept as its own tiny module so swapping the LLM provider later (a different
model, a different API) only requires editing this file - nothing else in
rag/ or agents/ needs to change.
"""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


@lru_cache(maxsize=4)
def get_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    """Return a cached DeepSeek chat model, accessed via its OpenAI-compatible API."""
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=temperature,
    )
