"""
Web agent node — runs a Tavily web search and generates an answer
from the search results for queries requiring live/external information.
"""

import logging

from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from prompts.prompt_store import PromptStore
from agents.llm import get_llm

logger = logging.getLogger(__name__)
_store = PromptStore()
_search_tool = TavilySearchResults(max_results=3)


def prepare_web(query: str) -> tuple[str, list[str]]:
    """Run web search and build the prompt. Returns (prompt_text, snippets)."""
    try:
        raw_results = _search_tool.invoke(query)
    except Exception as e:
        logger.warning("Tavily search failed: %s", e)
        raw_results = []

    snippets = []
    for r in raw_results:
        if isinstance(r, dict):
            snippets.append(f"- {r.get('content', str(r))}")
        else:
            snippets.append(f"- {str(r)}")
    results_text = "\n".join(snippets) if snippets else "No search results found."
    prompt_text = _store.get_active_prompt("web_agent").format(results=results_text, query=query)
    return prompt_text, snippets


def web_agent_node(state: dict) -> dict:
    """Search the web and synthesize an answer with citations."""
    prompt_text, snippets = prepare_web(state["query"])
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt_text)])
    return {
        **state,
        "web_results": snippets,
        "response": response.content.strip(),
        "agent_used": "web",
    }
