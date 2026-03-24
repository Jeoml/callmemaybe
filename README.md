# CallSense — Multi-Agent Call Centre Assistant

A LangGraph-powered call centre assistant that classifies inbound customer queries, routes them to specialized agents (RAG or Web), evaluates responses with an LLM-as-judge, and provides full execution tracing — all with hot-swappable prompts managed through PostgreSQL.

## Architecture

```
Customer Query
       │
       ▼
┌─────────────┐
│ Orchestrator │  ← CoT reasoning → intent classification
│  (LLM node)  │
└──────┬──────┘
       │
  ┌────┼────┐
  ▼    ▼    ▼
 RAG  Web  Escalate
  │    │    │
  └────┼────┘
       ▼
┌─────────────┐
│  Eval Judge  │  ← routing accuracy, faithfulness, confidence
└──────┬──────┘
       ▼
   Trace File
```

**Orchestrator** uses two-step chain-of-thought: first reasons about the query, then outputs a structured JSON routing decision.

**RAG Agent** retrieves policy chunks from ChromaDB (embedded with all-MiniLM-L6-v2) and generates grounded answers.

**Web Agent** searches the web via Tavily and synthesizes factual answers with citations.

**Eval Judge** scores each interaction on 3 dimensions using LLM-as-judge prompts stored in PostgreSQL.

**Prompt Store** provides versioned prompt management with activation, rollback, and 60s caching — prompts update live without restart.

## Setup

### 1. Prerequisites
- Python 3.11+
- A NeonDB (PostgreSQL) database
- API keys: Groq or OpenAI, Tavily

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual credentials
```

### 4. Run database migrations

```bash
python db/migrations.py
```

This creates the `prompt_versions` table and seeds 6 default prompts.

### 5. Ingest the rulebook

```bash
python ingest.py
```

With simulation (runs 10 test queries):
```bash
python ingest.py --simulate
```

### 6. Start the API server

```bash
uvicorn main:app --reload --port 8000
```

### 7. Launch the dashboard

```bash
streamlit run dashboard.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/call` | Process a customer query |
| GET | `/traces` | List all trace summaries |
| GET | `/traces/{trace_id}` | Get full trace |
| GET | `/prompts/{key}` | Get active prompt |
| GET | `/prompts/{key}/history` | List all versions |
| POST | `/prompts/{key}` | Create new version |
| POST | `/prompts/{key}/activate/{v}` | Activate version |
| POST | `/prompts/{key}/rollback` | Rollback to previous |

## Sample CoT Output

```
Query: "What is your return policy?"

Orchestrator reasoning:
1. The customer is asking about the return policy — a standard policy question.
2. This is clearly something covered in the company policy document.
3. No signals of distress or legal threats.
4. High confidence this should go to the RAG agent.

Routing summary: Policy question about returns — routing to RAG agent for documented answer.

→ Intent: rag | Confidence: high
```

## Sample Eval Scores

| Metric | Score | Reason |
|--------|-------|--------|
| Routing Accuracy | 5/5 | Return policy is a clear RAG query, correctly routed |
| Faithfulness | 5/5 | Response accurately reflects the 30-day return window from context |
| Resolution Confidence | 4/5 | Clear answer with specific details, minor hedge about edge cases |

## Prompt Update Workflow

```bash
# 1. Create a new version (inactive)
curl -X POST http://localhost:8000/prompts/orchestrator \
  -H "Content-Type: application/json" \
  -d '{"prompt_text": "Updated prompt...", "created_by": "joel"}'

# 2. Test with the current active prompt still in place

# 3. Activate the new version
curl -X POST http://localhost:8000/prompts/orchestrator/activate/2

# 4. If something goes wrong, rollback
curl -X POST http://localhost:8000/prompts/orchestrator/rollback
```

Prompts propagate within 60 seconds (cache TTL) without restarting the server.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `NEONDB_URL` | PostgreSQL connection string (sslmode=require) |
| `GROQ_API_KEY` | Groq API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `TAVILY_API_KEY` | Tavily search API key |
| `LLM_PROVIDER` | `groq` or `openai` |
| `LLM_MODEL` | Model name (e.g., `llama3-70b-8192`) |
| `CHROMA_PATH` | ChromaDB persistence path |
| `TRACES_PATH` | Trace output directory |
