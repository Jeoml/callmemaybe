# CallSense — Multi-Agent Call Centre AI Assistant

## Project Summary

CallSense is a production-grade, multi-agent AI system that automates inbound call handling for government social services. It uses a LangGraph state machine to intelligently route customer queries through specialized agents, then self-evaluates every response with an LLM-as-judge pipeline. The system ships with a FastAPI backend, SSE streaming, and a full-featured Streamlit ops dashboard.

---

## What I Built

### Intelligent Query Router (Orchestrator)
- Chain-of-thought reasoning engine that classifies inbound queries in real time
- 4-way intent classification: **RAG** (policy lookup), **Web** (live info), **Combined** (both), **Escalate** (human handoff for safety/distress)
- Confidence scoring (high/medium/low) on every routing decision
- Transparent reasoning — CoT is streamed live to supervisors, not hidden

### Specialized Agent Pipeline
- **RAG Agent** — Semantic search over a policy rulebook using ChromaDB (all-MiniLM-L6-v2 embeddings), top-3 chunk retrieval, grounded answer generation
- **Web Agent** — Real-time web search via Tavily API, LLM synthesis with source citations
- **Combined Agent** — Merges RAG retrieval + web search into a unified answer when queries need both policy context and live information
- **Escalate Path** — Immediate human handoff for distress, threats, or safety situations (no AI response)

### Automated Quality Evaluation (LLM-as-Judge)
- Every single response is scored on 3 dimensions, automatically, with no human in the loop:
  - **Routing Accuracy** — did the orchestrator pick the right agent?
  - **Faithfulness** — is the RAG response grounded in retrieved context (no hallucination)?
  - **Resolution Confidence** — did the response actually answer the caller's question?
- Judges run in parallel (ThreadPoolExecutor) to minimize latency
- Scores are 1-5 with written reasoning, stored as structured JSON

### Full Trace Persistence
- Every call produces a complete trace: query, CoT reasoning, routing decision, retrieved context, web results, final response, and all eval scores
- Traces stored in PostgreSQL (NeonDB) with indexed queries by timestamp and intent
- Traces are queryable via REST API and browsable in the dashboard

### Hot-Swappable Prompt Versioning System
- All 7 system prompts (orchestrator, 3 agents, 3 judges) are stored in PostgreSQL, not hardcoded
- Full version history with create/activate/rollback operations
- 60-second LRU in-memory cache — prompts update without server restart
- Unique partial index enforces exactly one active version per prompt key at the DB level

### Real-Time Streaming
- SSE endpoint streams orchestrator reasoning token-by-token, then agent response token-by-token
- Dashboard renders thinking + response as they generate (like ChatGPT/Claude UX)
- Event types: `cot_delta`, `response_delta`, `node` completions, `error`

### Operations Dashboard (Streamlit)
- **Chat UI** — multi-conversation, streaming responses, inline CoT display, eval score bars with reasoning
- **Trace Explorer** — search/browse all historical traces, view full pipeline metadata
- **Batch Eval** — aggregate score charts (mean routing accuracy, faithfulness, resolution confidence), agent distribution, confidence breakdown
- **Prompt Manager** — edit, version, activate, and rollback any prompt in the system live

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (StateGraph with conditional edges) |
| LLM | Groq (Llama 3.3 70B) / OpenAI (swappable via env var) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB (local persistence) |
| Web Search | Tavily API |
| Backend | FastAPI + Uvicorn |
| Database | PostgreSQL (NeonDB, serverless) |
| Frontend | Streamlit |
| Frameworks | LangChain Core, LangChain Community |

---

## Architecture

```
Inbound Query
     |
     v
Orchestrator (CoT reasoning -> intent classification)
     |
     |--- rag ---------> RAG Agent (ChromaDB retrieval + grounded generation)
     |--- web ---------> Web Agent (Tavily search + cited synthesis)
     |--- both --------> Combined Agent (RAG + Web merged)
     |--- escalate ----> Canned human-handoff response
     |
     v
Eval Node (3 LLM judges in parallel)
     |
     v
Response + Eval Scores + Full Trace (persisted to PostgreSQL)
```

---

## ROI / Value Delivered

### Automation
- **Handles inbound queries end-to-end** — from classification to response to quality scoring, zero human intervention for standard queries
- **Escalation detection** catches safety/distress cases and routes to humans immediately, reducing risk

### Quality Assurance Without Manual QA
- Every response is auto-scored on 3 dimensions — equivalent to having a QA analyst review 100% of calls
- Traces provide full audit trail for compliance and incident review
- Batch eval dashboard gives management aggregate quality metrics at a glance

### Operational Agility
- Prompts can be tuned, A/B tested, and rolled back in production without code deploys or restarts
- Swapping LLM providers (Groq <-> OpenAI) is a single env var change
- Adding a new agent type = one new node + one conditional edge in the graph

### Cost Efficiency
- Groq inference (Llama 3.3 70B) as default — significantly cheaper than GPT-4 class models
- Parallel judge execution reduces eval latency by ~3x vs sequential
- 60s prompt cache eliminates redundant DB reads under load

### Observability
- Token-level streaming gives supervisors real-time visibility into AI reasoning
- Trace explorer enables root-cause analysis on any individual call
- Batch eval surfaces systemic quality issues (e.g., low faithfulness = retrieval problem, low routing accuracy = orchestrator prompt needs work)

---

## API Surface

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/call` | POST | Synchronous full pipeline |
| `/call/stream` | POST | SSE streaming (token-level CoT + response) |
| `/traces` | GET | Paginated trace list (filterable by intent) |
| `/traces/{id}` | GET | Full trace detail |
| `/prompts/{key}` | GET | Active prompt for a key |
| `/prompts/{key}` | POST | Create new prompt version |
| `/prompts/{key}/history` | GET | All versions for a key |
| `/prompts/{key}/activate/{v}` | POST | Activate specific version |
| `/prompts/{key}/rollback` | POST | Rollback to previous version |

---

## Key Engineering Decisions

- **LangGraph over plain LangChain chains** — state machine gives explicit control over routing, makes the pipeline testable and debuggable node-by-node
- **DB-backed prompts over hardcoded strings** — enables A/B testing and incident response without deploys
- **Parallel judge execution** — 3 LLM calls at once instead of sequential, cuts eval time to ~1/3
- **Atomic trace writes** — no partial reads, no corrupt traces
- **Graceful degradation everywhere** — judge parse failure returns score=0 (never crashes), unknown intents default to RAG (safest for policy), web search failure returns empty results (never blocks the pipeline)
