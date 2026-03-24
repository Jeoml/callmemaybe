"""
LLM-as-judge evaluation node — scores each call on routing accuracy,
faithfulness (RAG only), and response confidence. Writes the full trace
to PostgreSQL atomically.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage

from prompts.prompt_store import PromptStore
from agents.llm import get_llm
from db.trace_store import write_trace

logger = logging.getLogger(__name__)
_store = PromptStore()


def _score(llm, prompt_text: str, label: str) -> dict:
    """
    Run a single judge prompt and parse the JSON score.
    On parse failure, returns score=0 so the trace is still written
    and the graph never crashes from an eval issue.
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt_text)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Failed to parse %s judge response: %s", label, e)
        return {"score": 0, "reason": "parse error"}
    except Exception as e:
        logger.warning("Judge %s failed: %s", label, e)
        return {"score": 0, "reason": f"judge error: {str(e)}"}


def eval_node(state: dict) -> dict:
    """
    Score the completed call on three dimensions in parallel and persist
    the full trace to PostgreSQL.
    """
    llm = get_llm()

    # Build all judge prompts
    routing_prompt = _store.get_active_prompt("judge_routing").format(
        agent_used=state.get("agent_used", "unknown"),
        cot_reasoning=state.get("cot_reasoning", ""),
        query=state.get("query", ""),
        response=state.get("response", ""),
    )

    conf_prompt = _store.get_active_prompt("judge_confidence").format(
        query=state.get("query", ""),
        response=state.get("response", ""),
    )

    needs_faithfulness = state.get("agent_used") in ("rag", "both")
    faith_prompt = None
    if needs_faithfulness:
        context_str = "\n".join(state.get("rag_context", []))
        faith_prompt = _store.get_active_prompt("judge_faithfulness").format(
            rag_context=context_str,
            response=state.get("response", ""),
        )

    # Run judges in parallel
    scores = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        routing_future = executor.submit(_score, llm, routing_prompt, "routing")
        conf_future = executor.submit(_score, llm, conf_prompt, "confidence")
        faith_future = None
        if faith_prompt:
            faith_future = executor.submit(_score, llm, faith_prompt, "faithfulness")

        scores["routing_accuracy"] = routing_future.result(timeout=30)
        scores["resolution_confidence"] = conf_future.result(timeout=30)
        if faith_future:
            scores["faithfulness"] = faith_future.result(timeout=30)
        else:
            scores["faithfulness"] = {"score": None, "reason": "not applicable (non-RAG agent)"}

    updated_state = {**state, "eval_scores": scores}

    # Write trace to PostgreSQL
    trace_data = {
        "trace_id": state.get("trace_id", "unknown"),
        "timestamp": state.get("timestamp", ""),
        "query": state.get("query", ""),
        "intent": state.get("intent", ""),
        "cot_reasoning": state.get("cot_reasoning", ""),
        "routing_confidence": state.get("routing_confidence", ""),
        "rag_context": state.get("rag_context", []),
        "web_results": state.get("web_results", []),
        "response": state.get("response", ""),
        "agent_used": state.get("agent_used", ""),
        "eval_scores": scores,
    }

    try:
        write_trace(trace_data)
    except Exception as e:
        logger.error("Failed to write trace %s: %s", trace_data["trace_id"], e)

    return updated_state
