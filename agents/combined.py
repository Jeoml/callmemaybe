"""
Combined agent node — runs RAG retrieval and web search together, then
synthesizes a single response using both policy context and live information.
Used when the orchestrator determines the query needs both sources.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from prompts.prompt_store import PromptStore
from agents.llm import get_llm
from agents.chroma import get_collection

logger = logging.getLogger(__name__)
_store = PromptStore()


def _fetch_rag(query: str) -> tuple[list[str], str]:
    """Retrieve RAG chunks and format context."""
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=3)
    chunks = results["documents"][0] if results["documents"] else []
    context = "\n\n---\n\n".join(chunks) if chunks else "No relevant policy context found."
    return chunks, context


def _fetch_web(query: str) -> tuple[list[str], str]:
    """Run web search and format results."""
    search_tool = TavilySearchResults(max_results=3)
    raw_results = search_tool.invoke(query)
    snippets = []
    for r in raw_results:
        if isinstance(r, dict):
            snippets.append(f"- {r.get('content', str(r))}")
        else:
            snippets.append(f"- {str(r)}")
    results_text = "\n".join(snippets) if snippets else "No search results found."
    return snippets, results_text


def prepare_combined(query: str) -> tuple[str, list[str], list[str]]:
    """Retrieve RAG context + web results in parallel and build the prompt."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        rag_future = executor.submit(_fetch_rag, query)
        web_future = executor.submit(_fetch_web, query)

        chunks, context = rag_future.result(timeout=30)
        snippets, results_text = web_future.result(timeout=30)

    prompt_text = _store.get_active_prompt("combined_agent").format(
        context=context, results=results_text, query=query
    )
    return prompt_text, chunks, snippets


def combined_node(state: dict) -> dict:
    """Retrieve from both sources and synthesize a unified answer."""
    prompt_text, chunks, snippets = prepare_combined(state["query"])
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt_text)])
    return {
        **state,
        "rag_context": chunks,
        "web_results": snippets,
        "response": response.content.strip(),
        "agent_used": "both",
    }
