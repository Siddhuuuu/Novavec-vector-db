"""
NovaVec RAG Demo — Retrieval-Augmented Generation over Wikipedia.

Retrieve relevant chunks from the ingested Wikipedia collection
via NovaVec semantic search, then generate a grounded answer using
an LLM (OpenAI GPT-4o-mini or local Ollama/Mistral fallback).

Usage
-----
    # After running demo/ingest.py:
    uvicorn demo.rag_demo:app --port 8001 --reload

    # Ask a question:
    curl -X POST http://localhost:8001/ask \\
         -H "Content-Type: application/json" \\
         -d '{"question": "What is the speed of light?"}'

Environment
-----------
    OPENAI_API_KEY — if set, uses gpt-4o-mini (OpenAI).
                     If unset, falls back to Ollama (local Mistral).
    NOVAVEC_DATA   — path to the saved collection (default: ./wikipedia_novavec_data).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

import novavec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---- App configuration -------------------------------------

DATA_DIR = os.environ.get("NOVAVEC_DATA", "./wikipedia_novavec_data")


# ---- FastAPI app -------------------------------------------

app = FastAPI(
    title="NovaVec RAG Demo",
    description=(
        "Retrieval-Augmented Generation demo. "
        "Embeds questions with all-MiniLM-L6-v2, retrieves the "
        "most relevant Wikipedia passages from NovaVec, and "
        "generates a grounded answer via GPT-4o-mini or Ollama."
    ),
    version="1.0.0",
)


# ---- Lazy-loaded globals -----------------------------------
# Loaded once at first request to avoid startup failure when
# the collection hasn't been ingested yet.

_model:      Optional[SentenceTransformer] = None
_collection: Optional[novavec.Collection]  = None


def get_model() -> SentenceTransformer:
    global _model  # noqa: PLW0603
    if _model is None:
        logger.info("Loading embedding model…")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded.")
    return _model


def get_collection() -> novavec.Collection:
    global _collection  # noqa: PLW0603
    if _collection is None:
        logger.info("Loading NovaVec collection from '%s'…", DATA_DIR)
        try:
            _collection = novavec.load_collection(DATA_DIR)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load collection from '{DATA_DIR}'. "
                f"Run 'python demo/ingest.py' first. Error: {exc}"
            ) from exc
        logger.info("Collection loaded: %d vectors.", len(_collection))
    return _collection


# ---- System prompt -----------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "Answer the question using ONLY the provided context passages. "
    "If the answer is not clearly stated in the context, say "
    "'I don't have enough information about this.' "
    "Do not invent facts or speculate beyond the context."
)


# ---- LLM call ----------------------------------------------

def call_llm(question: str, context: str) -> str:
    """
    Generate a grounded answer using whichever LLM provider is configured.

    Priority order (first matching env var wins):
      OPENAI_API_KEY   → OpenAI  (gpt-4o-mini, or OPENAI_MODEL override)
      GEMINI_API_KEY   → Google Gemini  (gemini-1.5-flash, or GEMINI_MODEL override)
      GROK_API_KEY     → xAI Grok  (grok-3-mini, or GROK_MODEL override)
      ANTHROPIC_API_KEY→ Anthropic Claude  (claude-haiku-4-5, or ANTHROPIC_MODEL override)
      OPENROUTER_API_KEY→ OpenRouter  (any model via OPENROUTER_MODEL)
      (none set)       → Ollama local  (mistral, or OLLAMA_MODEL override)

    All providers accept SYSTEM_PROMPT + user prompt identically.
    """
    prompt = (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    # ── OpenAI ────────────────────────────────────────────────
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        from openai import OpenAI
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=openai_key)
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=512, temperature=0.1)
        return resp.choices[0].message.content.strip()

    # ── Google Gemini ─────────────────────────────────────────
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        import google.generativeai as genai
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
        )
        resp = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=512, temperature=0.1),
        )
        return resp.text.strip()

    # ── xAI Grok ──────────────────────────────────────────────
    # Grok exposes an OpenAI-compatible API at https://api.x.ai/v1
    grok_key = os.environ.get("GROK_API_KEY")
    if grok_key:
        from openai import OpenAI
        model = os.environ.get("GROK_MODEL", "grok-3-mini")
        client = OpenAI(api_key=grok_key, base_url="https://api.x.ai/v1")
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=512, temperature=0.1)
        return resp.choices[0].message.content.strip()

    # ── Anthropic Claude ──────────────────────────────────────
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        import anthropic
        model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        client = anthropic.Anthropic(api_key=anthropic_key)
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    # ── OpenRouter (any model via single key) ─────────────────
    # https://openrouter.ai — also OpenAI-compatible
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        from openai import OpenAI
        model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
        client = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=512, temperature=0.1)
        return resp.choices[0].message.content.strip()

    # ── Ollama local fallback ──────────────────────────────────
    try:
        import ollama
        model = os.environ.get("OLLAMA_MODEL", "mistral")
        resp = ollama.chat(model=model, messages=messages)
        return resp["message"]["content"].strip()
    except Exception as exc:
        logger.error("All LLM providers failed. Last error: %s", exc)
        return (
            "No LLM provider configured. Set one of: OPENAI_API_KEY, "
            "GEMINI_API_KEY, GROK_API_KEY, ANTHROPIC_API_KEY, "
            "OPENROUTER_API_KEY, or run 'ollama serve'.\n\n"
            f"Retrieved context:\n{context[:400]}…"
        )


# ---- Request / response models -----------------------------

class AskRequest(BaseModel):
    question: str  = Field(
        description="The question to answer from Wikipedia context.",
        examples=["What is the speed of light?"],
    )
    top_k: int     = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context passages to retrieve.",
    )


class SourceChunk(BaseModel):
    title: str
    text:  str
    score: float


class AskResponse(BaseModel):
    question: str
    answer:   str
    sources:  list[SourceChunk]


# ---- Endpoints ---------------------------------------------

@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question",
    response_description="Grounded answer with source passages",
)
async def ask(req: AskRequest) -> AskResponse:
    """
    Answer a natural language question using Wikipedia context retrieved
    from NovaVec, then passed to an LLM for answer generation.
    """
    try:
        model = get_model()
        col   = get_collection()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Embed question
    query_emb = np.array(
        model.encode(req.question, normalize_embeddings=False),
        dtype=np.float32,
    )

    # Retrieve top-k passages from NovaVec
    results = col.search(query=query_emb, top_k=req.top_k)

    # Build source chunks
    sources: list[SourceChunk] = []
    for r in results:
        meta = r.metadata()
        sources.append(SourceChunk(
            title = meta.get("title", "Unknown"),
            text  = meta.get("text",  ""),
            score = float(r.score),
        ))

    # Assemble context string for LLM
    context = "\n\n".join(
        f"[{i + 1}] {s.title}: {s.text}"
        for i, s in enumerate(sources)
    )

    # Generate answer
    answer = call_llm(req.question, context)

    return AskResponse(
        question = req.question,
        answer   = answer,
        sources  = sources,
    )


@app.get("/health", summary="Health check")
async def health():
    return {
        "status":          "ok",
        "collection_size": len(get_collection()) if _collection else 0,
    }


# ---- Entry point -------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
