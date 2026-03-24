"""
Shared LLM factory for CallSense.
Reads LLM_PROVIDER from env to switch between Groq and OpenAI at deploy time.
Returns a cached singleton so multiple calls per request reuse the same client.
"""

import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_llm():
    """Return a cached LangChain chat model based on LLM_PROVIDER env var."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    timeout = int(os.getenv("LLM_TIMEOUT", "30"))

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0, request_timeout=timeout)
    else:
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=0, request_timeout=timeout)
