"""
RAG agent node — retrieves relevant policy chunks from ChromaDB and generates
a grounded answer using only the retrieved context.
"""

import logging

from langchain_core.messages import HumanMessage

from prompts.prompt_store import PromptStore
from agents.llm import get_llm
from agents.chroma import get_collection

logger = logging.getLogger(__name__)
_store = PromptStore()


def prepare_rag(query: str) -> tuple[str, list[str]]:
    """Retrieve context and build the RAG prompt. Returns (prompt_text, chunks)."""
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=3)
    chunks = results["documents"][0] if results["documents"] else []
    context = "\n\n---\n\n".join(chunks) if chunks else "No relevant context found."
    prompt_text = _store.get_active_prompt("rag_agent").format(context=context, query=query)
    return prompt_text, chunks


def rag_agent_node(state: dict) -> dict:
    """Retrieve policy chunks and generate a grounded answer."""
    prompt_text, chunks = prepare_rag(state["query"])
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt_text)])
    return {
        **state,
        "rag_context": chunks,
        "response": response.content.strip(),
        "agent_used": "rag",
    }
