"""
Shared ChromaDB client singleton for CallSense.
All agents import get_collection() from here instead of creating their own client.
"""

import os
import logging

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

_client = None
_collection = None


def get_collection():
    """Return the shared 'rulebook' ChromaDB collection (lazy singleton)."""
    global _client, _collection
    if _collection is not None:
        return _collection

    chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
    _client = chromadb.PersistentClient(path=chroma_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    _collection = _client.get_collection(name="rulebook", embedding_function=ef)
    logger.info("ChromaDB collection 'rulebook' loaded from %s", chroma_path)
    return _collection
