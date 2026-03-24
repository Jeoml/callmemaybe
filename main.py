"""
FastAPI entry point for CallSense — exposes call processing and prompt management endpoints.
Run migrations on startup to ensure the DB schema and seed prompts exist.
"""

import json
import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from db.migrations import run_migrations
from db.pool import close_pool
from db.trace_store import list_traces as db_list_traces, get_trace as db_get_trace
from prompts.prompt_store import PromptStore
from graph import build_graph, create_initial_state
from agents.parsing import parse_orchestrator_output, ESCALATE_RESPONSE
from agents.llm import get_llm
from agents.rag_agent import prepare_rag
from agents.web_agent import prepare_web
from agents.combined import prepare_combined
from eval.judge import eval_node

# ── Startup ─────────────────────────────────────────────────────────────────
try:
    run_migrations()
except Exception as e:
    logger.error("Migration failed: %s", e)
    raise

app = FastAPI(title="CallSense", description="Multi-agent call centre assistant")

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prompt_store = PromptStore()
graph = build_graph()


@app.on_event("shutdown")
def shutdown():
    close_pool()


# ── Request/Response models ─────────────────────────────────────────────────

class CallRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class PromptCreateRequest(BaseModel):
    prompt_text: str
    created_by: str = "api"
    activate: bool = False


# ── Call endpoints ──────────────────────────────────────────────────────────

@app.post("/call")
def handle_call(req: CallRequest):
    """Process an inbound customer query through the full agent pipeline."""
    result = graph.invoke(create_initial_state(req.query))
    return {
        "trace_id": result.get("trace_id"),
        "response": result.get("response"),
        "agent_used": result.get("agent_used"),
        "intent": result.get("intent"),
        "cot_reasoning": result.get("cot_reasoning"),
        "routing_confidence": result.get("routing_confidence"),
        "eval_scores": result.get("eval_scores"),
    }


@app.post("/call/stream")
def handle_call_stream(req: CallRequest):
    """
    Stream call processing via SSE with token-level CoT streaming.
    Event types:
      - cot_delta: a single token of orchestrator reasoning (streamed live)
      - node: a graph node completed (contains updated state fields)
      - error: pipeline failure with message
      - [DONE]: stream finished
    """

    def event_generator():
        state = create_initial_state(req.query)

        try:
            # ── Phase 1: Stream orchestrator CoT token-by-token ────────────
            llm = get_llm()
            prompt_text = prompt_store.get_active_prompt("orchestrator").format(query=req.query)

            chunks = []
            for chunk in llm.stream([HumanMessage(content=prompt_text)]):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    chunks.append(token)
                    yield f"data: {json.dumps({'type': 'cot_delta', 'content': token})}\n\n"

            parsed = parse_orchestrator_output("".join(chunks))
            state.update(parsed)

            if state["intent"] == "escalate":
                state["response"] = ESCALATE_RESPONSE
                state["agent_used"] = "escalate"

            yield f"data: {json.dumps({'type': 'node', 'node': 'orchestrator', 'intent': state['intent'], 'routing_confidence': state['routing_confidence'], 'trace_id': state['trace_id']})}\n\n"

            # ── Phase 2: Stream agent response token-by-token ──────────────
            intent = state["intent"]
            agent_prompt = None

            if intent == "rag":
                agent_prompt, rag_chunks = prepare_rag(req.query)
                state["rag_context"] = rag_chunks
                state["agent_used"] = "rag"
            elif intent == "web":
                agent_prompt, web_snippets = prepare_web(req.query)
                state["web_results"] = web_snippets
                state["agent_used"] = "web"
            elif intent == "both":
                agent_prompt, rag_chunks, web_snippets = prepare_combined(req.query)
                state["rag_context"] = rag_chunks
                state["web_results"] = web_snippets
                state["agent_used"] = "both"

            if agent_prompt:
                yield f"data: {json.dumps({'type': 'node', 'node': state['agent_used'], 'response': ''})}\n\n"

                resp_chunks = []
                for chunk in llm.stream([HumanMessage(content=agent_prompt)]):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        resp_chunks.append(token)
                        yield f"data: {json.dumps({'type': 'response_delta', 'content': token})}\n\n"

                state["response"] = "".join(resp_chunks).strip()

            # ── Phase 3: Run eval ──────────────────────────────────────────
            state = eval_node(state)

            yield f"data: {json.dumps({'type': 'node', 'node': 'eval', 'eval_scores': state.get('eval_scores', {}), 'trace_id': state.get('trace_id', '')})}\n\n"

        except Exception as e:
            logger.exception("Stream error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Trace endpoints (PostgreSQL-backed) ────────────────────────────────────

@app.get("/traces")
def list_traces(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    intent: Optional[str] = None,
):
    """Return paginated trace summaries, newest first."""
    return db_list_traces(limit=limit, offset=offset, intent=intent)


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str):
    """Return full trace data for a specific trace."""
    trace = db_get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return trace


# ── Prompt management endpoints ─────────────────────────────────────────────

@app.get("/prompts/{prompt_key}")
def get_active_prompt(prompt_key: str):
    """Return the active prompt version for a given key."""
    try:
        versions = prompt_store.list_prompt_versions(prompt_key)
        active = next((v for v in versions if v["is_active"]), None)
        if not active:
            raise HTTPException(status_code=404, detail=f"No active prompt for key: {prompt_key}")
        active["created_at"] = str(active["created_at"])
        return active
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/prompts/{prompt_key}/history")
def get_prompt_history(prompt_key: str):
    """Return all versions for a prompt key."""
    versions = prompt_store.list_prompt_versions(prompt_key)
    if not versions:
        raise HTTPException(status_code=404, detail=f"No prompts found for key: {prompt_key}")
    for v in versions:
        v["created_at"] = str(v["created_at"])
    return versions


@app.post("/prompts/{prompt_key}")
def create_prompt(prompt_key: str, req: PromptCreateRequest):
    """Create a new prompt version. If activate=true, immediately activate it."""
    try:
        new_version = prompt_store.create_prompt_version(
            prompt_key, req.prompt_text, req.created_by
        )
        if req.activate:
            new_version = prompt_store.activate_prompt_version(prompt_key, new_version["version"])
        new_version["created_at"] = str(new_version["created_at"])
        return new_version
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prompts/{prompt_key}/activate/{version}")
def activate_prompt(prompt_key: str, version: int):
    """Activate a specific version number for a prompt key."""
    try:
        row = prompt_store.activate_prompt_version(prompt_key, version)
        row["created_at"] = str(row["created_at"])
        return row
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prompts/{prompt_key}/rollback")
def rollback_prompt(prompt_key: str):
    """Roll back to the previously active version."""
    try:
        row = prompt_store.rollback_prompt(prompt_key)
        row["created_at"] = str(row["created_at"])
        return row
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
