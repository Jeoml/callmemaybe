"""
LangGraph StateGraph definition for CallSense.
Defines the CallState schema and wires orchestrator → agent → eval pipeline
with conditional routing based on the orchestrator's intent classification.
Supports four intents: rag, web, both (rag+web combined), and escalate.
"""

import uuid
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.orchestrator import orchestrator_node
from agents.rag_agent import rag_agent_node
from agents.web_agent import web_agent_node
from agents.combined import combined_node
from eval.judge import eval_node


class CallState(TypedDict):
    query: str                  # original caller query
    intent: str                 # "rag" | "web" | "both" | "escalate"
    cot_reasoning: str          # orchestrator's full chain-of-thought
    routing_confidence: str     # "high" | "medium" | "low"
    rag_context: list[str]      # retrieved chunks (empty if not used)
    web_results: list[str]      # search snippets (empty if not used)
    response: str               # final answer to caller
    agent_used: str             # which agent produced the answer
    eval_scores: dict           # populated by judge after response
    trace_id: str               # uuid4 for this call
    timestamp: str              # ISO timestamp


def create_initial_state(query: str) -> CallState:
    """Factory for a blank CallState with trace metadata populated."""
    return {
        "query": query,
        "intent": "",
        "cot_reasoning": "",
        "routing_confidence": "",
        "rag_context": [],
        "web_results": [],
        "response": "",
        "agent_used": "",
        "eval_scores": {},
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _route_by_intent(state: CallState) -> str:
    """
    Conditional edge: directs flow based on orchestrator classification.
    - rag: policy questions answered from the DSS/HRA rulebook
    - web: live/current info from web search
    - both: needs policy context + live info combined
    - escalate: distress/legal/safety — skip agents, go straight to eval
    """
    intent = state.get("intent", "rag")
    if intent == "web":
        return "web_agent"
    elif intent == "both":
        return "combined_agent"
    elif intent == "escalate":
        return "eval"
    # Default to RAG for unknown intents — safer for policy questions
    return "rag_agent"


def build_graph() -> StateGraph:
    """Construct and compile the CallSense processing graph."""
    graph = StateGraph(CallState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("web_agent", web_agent_node)
    graph.add_node("combined_agent", combined_node)
    graph.add_node("eval", eval_node)

    graph.set_entry_point("orchestrator")

    # Orchestrator decides which agent handles the query
    graph.add_conditional_edges(
        "orchestrator",
        _route_by_intent,
        {
            "rag_agent": "rag_agent",
            "web_agent": "web_agent",
            "combined_agent": "combined_agent",
            "eval": "eval",
        },
    )

    graph.add_edge("rag_agent", "eval")
    graph.add_edge("web_agent", "eval")
    graph.add_edge("combined_agent", "eval")
    graph.add_edge("eval", END)

    return graph.compile()
