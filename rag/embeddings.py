"""
Provides the embedding model used to vectorize policy document chunks and
user queries.

Kept as its own tiny module so the embedding model can be swapped later
(a different HuggingFace model, or a hosted API) by editing only this file -
nothing else in rag/ or agents/ needs to change.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

# Local, free, no API key required. DeepSeek (our LLM provider) has no
# embeddings endpoint, so embedding is handled entirely locally.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a cached embedding model instance (downloaded/loaded once per process)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
