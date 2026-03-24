"""
Ingest script — embeds the NovaMart rulebook into ChromaDB and optionally
runs a simulation of 10 sample queries through the full graph pipeline.
"""

import os
import sys
import argparse
import logging

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RULEBOOK_PATH = os.path.join(os.path.dirname(__file__), "data", "rulebook.txt")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split text into ~chunk_size token chunks with overlap.
    Uses whitespace-based word splitting as a proxy for tokens —
    close enough for embedding and avoids a tokenizer dependency.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def ingest():
    """Read the rulebook, chunk it, embed with all-MiniLM-L6-v2, and persist to ChromaDB."""
    with open(RULEBOOK_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    logger.info("Split rulebook into %d chunks", len(chunks))

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection if re-ingesting to avoid stale/duplicate chunks
    try:
        client.delete_collection("rulebook")
    except Exception:
        pass

    collection = client.create_collection(name="rulebook", embedding_function=ef)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

    logger.info("Ingested %d chunks into ChromaDB at %s", len(chunks), CHROMA_PATH)
    return len(chunks)


# ── Simulation queries covering all routing paths ───────────────────────────
SIMULATION_QUERIES = [
    # RAG queries (HRA/DSS policy-based)
    "What are the eligibility requirements for CityFHEPS?",
    "How do I apply for SNAP benefits?",
    "What is the income limit for SCRIE rent freeze?",
    # Web queries (live/current info)
    "What are the current CityFHEPS processing times in 2026?",
    "Are there any changes to HRA office hours this week?",
    "What is the latest news about NYC rent subsidy programs?",
    # Combined queries (need both policy + live info)
    "Am I eligible for a One-Shot Deal and how long is the current wait?",
    "What does CityFHEPS cover and are landlords actually accepting vouchers right now?",
    # Escalation queries (distress/legal/threats)
    "I'm going to sue HRA if you don't fix my case immediately! My kids are going to be on the street!",
    "I want to speak to a supervisor right now, my caseworker has been ignoring me for weeks and I'm contacting my lawyer.",
]


def simulate():
    """Run sample queries through the graph and print results summary."""
    from graph import build_graph

    graph = build_graph()
    results = []

    for i, query in enumerate(SIMULATION_QUERIES):
        logger.info("Simulating query %d/%d: %s", i + 1, len(SIMULATION_QUERIES), query[:60])
        try:
            initial_state = {
                "query": query,
                "intent": "",
                "cot_reasoning": "",
                "routing_confidence": "",
                "rag_context": [],
                "web_results": [],
                "response": "",
                "agent_used": "",
                "eval_scores": {},
                "trace_id": "",
                "timestamp": "",
            }
            result = graph.invoke(initial_state)
            results.append(result)
            logger.info(
                "  → intent=%s  agent=%s  confidence=%s",
                result.get("intent"),
                result.get("agent_used"),
                result.get("routing_confidence"),
            )
        except Exception as e:
            logger.error("  → FAILED: %s", e)
            results.append({"query": query, "error": str(e)})

    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    agents = {"rag": 0, "web": 0, "escalate": 0}
    for r in results:
        agent = r.get("agent_used", "error")
        agents[agent] = agents.get(agent, 0) + 1
    for agent, count in agents.items():
        print(f"  {agent:>10}: {count} queries")
    print(f"  {'total':>10}: {len(results)} queries")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest rulebook into ChromaDB")
    parser.add_argument("--simulate", action="store_true", help="Run sample queries after ingest")
    args = parser.parse_args()

    chunk_count = ingest()
    print(f"Ingested {chunk_count} chunks into ChromaDB.")

    if args.simulate:
        simulate()
