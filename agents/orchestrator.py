"""
Orchestrator node — classifies inbound queries using a two-step CoT approach.
Step 1: LLM reasons through the query and outputs structured JSON on the last line.
Step 2: Parse the reasoning and routing decision from the response.
"""

import uuid
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from prompts.prompt_store import PromptStore
from agents.llm import get_llm
from agents.parsing import parse_orchestrator_output, ESCALATE_RESPONSE

_store = PromptStore()


def orchestrator_node(state: dict) -> dict:
    """
    Classify the customer query into rag/web/escalate with chain-of-thought reasoning.
    Prompt is fetched from the DB at call time so live updates take effect without restart.
    """
    query = state["query"]

    trace_id = state.get("trace_id") or str(uuid.uuid4())
    timestamp = state.get("timestamp") or datetime.now(timezone.utc).isoformat()

    prompt_template = _store.get_active_prompt("orchestrator")
    prompt_text = prompt_template.format(query=query)

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt_text)])
    parsed = parse_orchestrator_output(response.content.strip())

    response_text = state.get("response", "")
    agent_used = state.get("agent_used", "")
    if parsed["intent"] == "escalate":
        response_text = ESCALATE_RESPONSE
        agent_used = "escalate"

    return {
        **state,
        **parsed,
        "trace_id": trace_id,
        "timestamp": timestamp,
        "response": response_text,
        "agent_used": agent_used,
    }
