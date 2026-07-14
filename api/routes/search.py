"""
Search endpoint.
POST /collections/{name}/points/search — nearest-neighbor search
"""

import json
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas import SearchRequest, SearchResponse, SearchResultResponse
from api.main import collections_registry, logger

router = APIRouter()


@router.post(
    "/{name}/points/search",
    response_model=SearchResponse,
    summary="Nearest-neighbor search",
    response_description="Top-k nearest neighbors with optional payload",
)
async def search_points(name: str, req: SearchRequest):
    """
    Semantic nearest-neighbor search with optional metadata filtering.

    **Query planner** automatically selects between:
    - **PRE_FILTER** — bitmap mask during HNSW traversal (low selectivity filters)
    - **POST_FILTER** — overfetch then discard (high selectivity filters)
    - **NONE** — unfiltered full search

    The `filter.must[0]` condition is used (single field=value match).

    **ef** controls the HNSW beam width — larger values trade latency for recall.
    """
    if name not in collections_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{name}' not found.",
        )

    col = collections_registry[name]

    # Build query vector — contiguous float32
    query_vec = np.array(req.vector, dtype=np.float32)

    # Extract filter from Qdrant-compatible FilterCondition
    filter_dict: Optional[dict] = None
    if req.filter and req.filter.must:
        # Take first 'must' condition — single-field exact-match filter
        cond = req.filter.must[0]
        filter_dict = {cond.key: str(cond.match.value)}

    start = time.perf_counter()

    try:
        raw_results = col.search(
            query    = query_vec,
            top_k    = req.top_k,
            ef_search= req.params.ef,
            filter   = filter_dict,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Search error",
            extra={"collection": name, "error": str(exc)},
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Build response objects
    results: list[SearchResultResponse] = []
    for r in raw_results:
        payload: dict = {}
        if req.with_payload and r.metadata_json:
            try:
                payload = json.loads(r.metadata_json)
            except (json.JSONDecodeError, ValueError):
                payload = {}
        results.append(
            SearchResultResponse(
                id      = r.id,
                score   = float(r.score),
                payload = payload,
            )
        )

    logger.info(
        "Search completed",
        extra={
            "collection":  name,
            "top_k":       req.top_k,
            "results":     len(results),
            "latency_ms":  round(elapsed_ms, 2),
            "has_filter":  filter_dict is not None,
        },
    )

    return SearchResponse(
        results  = results,
        time_ms  = round(elapsed_ms, 3),
    )
