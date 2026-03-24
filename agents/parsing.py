"""
Shared parsing utilities for orchestrator output.
Extracted to avoid circular imports between graph.py and orchestrator.py.
"""

import json
import logging

logger = logging.getLogger(__name__)

VALID_INTENTS = ("rag", "web", "both", "escalate")
VALID_CONFIDENCES = ("high", "medium", "low")

ESCALATE_RESPONSE = "I understand your concern. Let me connect you with a supervisor who can assist you directly. Please hold."


def parse_orchestrator_output(raw_output: str) -> dict:
    """
    Parse the orchestrator's LLM output into cot_reasoning, intent, and confidence.
    The LLM puts a JSON object on the last line with free-form reasoning above it.
    Falls back to intent=rag on parse failure (safest for policy questions).
    """
    lines = raw_output.strip().split("\n")
    last_line = ""
    cot_lines = []

    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped:
            last_line = stripped
            cot_lines = lines[:i]
            break

    cot_reasoning = "\n".join(cot_lines).strip()
    intent = "rag"
    confidence = "low"

    try:
        parsed = json.loads(last_line)
        intent = parsed.get("intent", "rag")
        confidence = parsed.get("confidence", "low")
        summary = parsed.get("summary", "")
        if summary:
            cot_reasoning += f"\n\nRouting summary: {summary}"
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse orchestrator JSON from: %s", last_line)
        cot_reasoning = raw_output

    if intent not in VALID_INTENTS:
        logger.warning("Unknown intent '%s', defaulting to 'rag'", intent)
        intent = "rag"
    if confidence not in VALID_CONFIDENCES:
        confidence = "low"

    return {"cot_reasoning": cot_reasoning, "intent": intent, "routing_confidence": confidence}
