"""
Ingest Wikipedia Simple English into NovaVec.

Downloads the Wikipedia Simple English dataset via HuggingFace Datasets,
chunks articles into overlapping word windows, embeds each chunk with
all-MiniLM-L6-v2 (384-dim), and upserts into a NovaVec collection.

Usage
-----
    python demo/ingest.py [--max-chunks 50000] [--batch-size 64]

Output
------
    ./wikipedia_novavec_data/   — persisted collection
"""

import argparse
import logging
import sys
import time
from typing import Generator

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import novavec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---- Text chunking ------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int    = 50,
) -> list[str]:
    """
    Word-level chunking with sliding-window overlap.

    No external library — splits on whitespace, builds chunks from
    contiguous word windows. Overlap preserves cross-boundary context.

    Parameters
    ----------
    text       : Source text.
    chunk_size : Window size in words.
    overlap    : Number of words shared between consecutive chunks.

    Returns
    -------
    List of chunk strings. Last chunk may be shorter than chunk_size.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step  = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks


# ---- Batch generator ----------------------------------------

def iter_batches(
    dataset,
    model: SentenceTransformer,
    max_chunks: int,
    batch_size: int,
) -> Generator[tuple[list[np.ndarray], list[dict]], None, None]:
    """
    Yield (embeddings, metadata) batches from the Wikipedia dataset.
    Stops after max_chunks total chunks have been processed.
    """
    texts:  list[str]  = []
    metas:  list[dict] = []
    total = 0

    for article in dataset:
        if total >= max_chunks:
            break

        title = article.get("title", "")
        text  = article.get("text", "")
        if not text.strip():
            continue

        for chunk_i, chunk in enumerate(chunk_text(text)):
            texts.append(chunk)
            metas.append({
                "title":    title,
                "chunk_id": chunk_i,
                # Store first 500 chars to keep metadata compact
                "text":     chunk[:500],
            })
            total += 1

            if len(texts) >= batch_size or total >= max_chunks:
                embeddings = model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=False,  # NovaVec handles normalization
                )
                yield [np.array(e, dtype=np.float32) for e in embeddings], metas
                texts, metas = [], []

        if total >= max_chunks:
            break

    # Flush remaining partial batch
    if texts:
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        yield [np.array(e, dtype=np.float32) for e in embeddings], metas


# ---- Main ---------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Wikipedia Simple English into NovaVec."
    )
    parser.add_argument("--max-chunks", type=int, default=50_000,
                        help="Maximum number of chunks to ingest (default 50000).")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Embedding batch size (default 64).")
    args = parser.parse_args()

    COLLECTION_NAME = "wikipedia"
    EMBED_DIM       = 384  # all-MiniLM-L6-v2 output dimensionality

    # ---- Load embedding model ----
    logger.info("Loading sentence-transformer model (all-MiniLM-L6-v2)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Model loaded.")

    # ---- Create NovaVec collection ----
    logger.info("Creating NovaVec collection '%s' (dim=%d)…", COLLECTION_NAME, EMBED_DIM)
    col = novavec.Collection(
        name            = COLLECTION_NAME,
        dim             = EMBED_DIM,
        metric          = "cosine",
        index_type      = "hnsw",
        M               = 16,
        ef_construction = 200,
    )

    # ---- Load dataset ----
    logger.info("Downloading Wikipedia Simple English dataset…")
    dataset = load_dataset(
        "wikipedia",
        "20220301.simple",
        split="train",
        trust_remote_code=True,
    )
    logger.info("Dataset loaded: %d articles.", len(dataset))

    # ---- Ingest ----
    doc_id     = 0
    t0         = time.perf_counter()
    log_every  = 1000

    for embeddings, metas in iter_batches(
        dataset, model, args.max_chunks, args.batch_size
    ):
        for emb, meta in zip(embeddings, metas):
            col.upsert(doc_id, emb, meta)
            doc_id += 1

        if doc_id % log_every < args.batch_size:
            elapsed = time.perf_counter() - t0
            rate    = doc_id / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d chunks (%.0f chunks/s)",
                doc_id, rate,
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Ingestion complete. %d chunks in %.1f s (%.0f chunks/s).",
        doc_id, elapsed, doc_id / elapsed if elapsed > 0 else 0,
    )

    # ---- Persist ----
    logger.info("Saving collection to disk…")
    col.save()
    logger.info(
        "Collection saved. Run 'uvicorn demo.rag_demo:app --port 8001' to start the RAG demo."
    )


if __name__ == "__main__":
    main()
