"""
Database migrations for CallSense prompt versioning system.
Creates the prompt_versions table and seeds default prompts on first run.
Connects to NeonDB (PostgreSQL) via NEONDB_URL with SSL required.
"""

import logging

from db.pool import get_conn

logger = logging.getLogger(__name__)

# ── Default prompts seeded on first run ──────────────────────────────────────
# These define the baseline behavior for every agent and judge in the graph.
# Each key maps to a node or eval dimension; updates go through the prompt store.

DEFAULT_PROMPTS = {
    "orchestrator": (
        "You are the routing brain for an NYC HRA/DSS call centre. "
        "Your job is to read the caller's query, reason through it out loud, and decide which specialist should handle it.\n\n"
        "THINK OUT LOUD — your reasoning is shown to supervisors in real time, so write clearly and naturally:\n"
        "- Start by restating what the caller needs in your own words.\n"
        "- Consider which knowledge source fits best:\n"
        "    rag  → policy/program rules (CityFHEPS, FHEPS, SCRIE, DRIE, SNAP, Cash Assistance, One-Shot Deals, shelter, fair hearings, grievances)\n"
        "    web  → live/current info (processing times, office status, news, anything not in policy docs)\n"
        "    both → caller needs policy rules AND current info together\n"
        "    escalate → distress, threats, abuse, safety, or repeated escalation requests → needs a human now\n"
        "- Explain WHY you chose that route and how confident you are.\n"
        "- Keep it concise — 3 to 6 sentences of reasoning, not a checklist.\n\n"
        "After your reasoning, output ONLY a JSON object on the very last line:\n"
        '{{"intent": "<rag|web|both|escalate>", "confidence": "<high|medium|low>", '
        '"summary": "<one sentence explaining the routing decision>"}}\n\n'
        "Caller: {query}"
    ),
    "rag_agent": (
        "You are a helpful and empathetic call centre agent for the NYC Department of Social Services (DSS) / Human Resources Administration (HRA).\n"
        "Answer the caller's query using ONLY the policy context below. Be specific with program names, eligibility requirements, addresses, and phone numbers when available.\n"
        "If the answer is not clearly supported by the context, say: "
        '"I don\'t have that specific information in our policy documents. Let me connect you with a caseworker who can help."\n\n'
        "Context:\n{context}\n\n"
        "Caller's query: {query}\n\n"
        "Answer:"
    ),
    "web_agent": (
        "You are a helpful and empathetic call centre agent for the NYC Department of Social Services (DSS) / Human Resources Administration (HRA).\n"
        "Answer the caller's query using the search results below. Be concise and factual.\n"
        "Cite which result supports your answer. If results are insufficient, say so clearly and recommend the caller visit ACCESS HRA (accesshra.nyc.gov) or call 311.\n\n"
        "Search results:\n{results}\n\n"
        "Caller's query: {query}\n\n"
        "Answer:"
    ),
    "combined_agent": (
        "You are a helpful and empathetic call centre agent for the NYC Department of Social Services (DSS) / Human Resources Administration (HRA).\n"
        "Answer the caller's query using BOTH the internal policy context AND the web search results below.\n"
        "Use policy context for program rules, eligibility, and procedures. Use web results for current/live information.\n"
        "Clearly distinguish between established policy and current information when relevant.\n"
        "If either source is insufficient, acknowledge what you can confirm and what needs follow-up.\n\n"
        "Policy context:\n{context}\n\n"
        "Web search results:\n{results}\n\n"
        "Caller's query: {query}\n\n"
        "Answer:"
    ),
    "judge_routing": (
        'Was routing this query to agent "{agent_used}" the correct decision?\n'
        "The orchestrator's reasoning was: {cot_reasoning}\n"
        "Query: {query}\n"
        "Response produced: {response}\n"
        "Consider: rag is for HRA/DSS policy questions, web is for live/current info, both is for queries needing policy + live info, escalate is for distress/legal/safety.\n"
        'Score 1 (wrong agent) to 5 (clearly correct agent).\n'
        'Reply with ONLY valid JSON: {{"score": <int>, "reason": "<one sentence>"}}'
    ),
    "judge_faithfulness": (
        "Does the following response faithfully reflect only what is in the provided context, without hallucinating facts?\n"
        "Context: {rag_context}\n"
        "Response: {response}\n"
        'Score 1 (fabricated) to 5 (fully grounded).\n'
        'Reply with ONLY valid JSON: {{"score": <int>, "reason": "<one sentence>"}}'
    ),
    "judge_confidence": (
        "Does the following response definitively answer the caller's query, or does it hedge or deflect?\n"
        "Query: {query}\n"
        "Response: {response}\n"
        'Score 1 (no answer / fully deflected) to 5 (clear complete answer).\n'
        'Reply with ONLY valid JSON: {{"score": <int>, "reason": "<one sentence>"}}'
    ),
}


def run_migrations():
    """Create the prompt_versions table and seed defaults if the table is empty."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            # ── Schema creation ─────────────────────────────────────────
            # The unique partial index guarantees at most one active version
            # per prompt_key at the database level, preventing accidental
            # double-activation even under concurrent requests.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id            SERIAL PRIMARY KEY,
                    prompt_key    VARCHAR(100) NOT NULL,
                    version       INTEGER NOT NULL,
                    prompt_text   TEXT NOT NULL,
                    created_at    TIMESTAMP DEFAULT NOW(),
                    created_by    VARCHAR(100) DEFAULT 'system',
                    is_active     BOOLEAN DEFAULT FALSE,
                    rollback_of   INTEGER DEFAULT NULL
                );
            """)
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS unique_active_prompt
                    ON prompt_versions (prompt_key)
                    WHERE is_active = TRUE;
            """)

            # ── Traces table ─────────────────────────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id            SERIAL PRIMARY KEY,
                    trace_id      VARCHAR(64) UNIQUE NOT NULL,
                    timestamp     TIMESTAMPTZ DEFAULT NOW(),
                    query         TEXT NOT NULL,
                    intent        VARCHAR(20),
                    cot_reasoning TEXT,
                    routing_confidence VARCHAR(20),
                    rag_context   JSONB DEFAULT '[]'::jsonb,
                    web_results   JSONB DEFAULT '[]'::jsonb,
                    response      TEXT,
                    agent_used    VARCHAR(20),
                    eval_scores   JSONB DEFAULT '{}'::jsonb
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces (timestamp DESC);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_intent ON traces (intent);
            """)

            # ── Seed default prompts if table is empty ──────────────────
            # Only runs on first deployment; subsequent starts skip seeding
            # so that user-edited prompts are preserved.
            cur.execute("SELECT COUNT(*) FROM prompt_versions;")
            count = cur.fetchone()[0]

            if count == 0:
                logger.info("Seeding %d default prompts into prompt_versions", len(DEFAULT_PROMPTS))
                for key, text in DEFAULT_PROMPTS.items():
                    cur.execute(
                        """
                        INSERT INTO prompt_versions (prompt_key, version, prompt_text, created_by, is_active)
                        VALUES (%s, 1, %s, 'system', TRUE)
                        """,
                        (key, text),
                    )
                logger.info("Default prompts seeded successfully")
            else:
                logger.info("prompt_versions already has %d rows — skipping seed", count)

        conn.commit()
        logger.info("Migrations complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_migrations()
    print("Migrations applied successfully.")
