"""
TraceStore — PostgreSQL-backed trace storage for CallSense.
Replaces flat-file JSONL storage with indexed, paginated database queries.
"""

import json
import logging

from db.pool import get_conn

logger = logging.getLogger(__name__)


def write_trace(trace_data: dict):
    """Insert a trace record into PostgreSQL."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO traces (trace_id, timestamp, query, intent, cot_reasoning,
                    routing_confidence, rag_context, web_results, response, agent_used, eval_scores)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (trace_id) DO UPDATE SET
                    eval_scores = EXCLUDED.eval_scores,
                    response = EXCLUDED.response
                """,
                (
                    trace_data.get("trace_id"),
                    trace_data.get("timestamp"),
                    trace_data.get("query", ""),
                    trace_data.get("intent", ""),
                    trace_data.get("cot_reasoning", ""),
                    trace_data.get("routing_confidence", ""),
                    json.dumps(trace_data.get("rag_context", [])),
                    json.dumps(trace_data.get("web_results", [])),
                    trace_data.get("response", ""),
                    trace_data.get("agent_used", ""),
                    json.dumps(trace_data.get("eval_scores", {}), default=str),
                ),
            )
        conn.commit()
        logger.info("Trace written to DB: %s", trace_data.get("trace_id"))


def list_traces(limit: int = 50, offset: int = 0, intent: str = None) -> list[dict]:
    """Return trace summaries, newest first, with pagination."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            if intent:
                cur.execute(
                    """
                    SELECT trace_id, timestamp, query, intent, routing_confidence,
                           agent_used, eval_scores
                    FROM traces
                    WHERE intent = %s
                    ORDER BY timestamp DESC
                    LIMIT %s OFFSET %s
                    """,
                    (intent, limit, offset),
                )
            else:
                cur.execute(
                    """
                    SELECT trace_id, timestamp, query, intent, routing_confidence,
                           agent_used, eval_scores
                    FROM traces
                    ORDER BY timestamp DESC
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset),
                )
            cols = [d[0] for d in cur.description]
            rows = []
            for row in cur.fetchall():
                d = dict(zip(cols, row))
                d["timestamp"] = d["timestamp"].isoformat() if d["timestamp"] else None
                return_scores = d.get("eval_scores")
                if isinstance(return_scores, str):
                    d["eval_scores"] = json.loads(return_scores)
                rows.append(d)
            return rows


def get_trace(trace_id: str) -> dict | None:
    """Return full trace data for a specific trace_id."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trace_id, timestamp, query, intent, cot_reasoning,
                       routing_confidence, rag_context, web_results,
                       response, agent_used, eval_scores
                FROM traces
                WHERE trace_id = %s
                """,
                (trace_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            d = dict(zip(cols, row))
            d["timestamp"] = d["timestamp"].isoformat() if d["timestamp"] else None
            for key in ("rag_context", "web_results", "eval_scores"):
                if isinstance(d.get(key), str):
                    d[key] = json.loads(d[key])
            return d


def count_traces(intent: str = None) -> int:
    """Return total trace count, optionally filtered by intent."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            if intent:
                cur.execute("SELECT COUNT(*) FROM traces WHERE intent = %s", (intent,))
            else:
                cur.execute("SELECT COUNT(*) FROM traces")
            return cur.fetchone()[0]
