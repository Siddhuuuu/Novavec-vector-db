"""
Collection management endpoints.
POST   /collections/{name}         — create collection
DELETE /collections/{name}         — delete collection
GET    /collections/{name}/info    — collection info
GET    /collections/               — list all collections
"""

from fastapi import APIRouter, HTTPException

import novavec

from api.schemas import CreateCollectionRequest
from api.main import collections_registry, logger

router = APIRouter()


@router.post(
    "/{name}",
    status_code=201,
    summary="Create a collection",
    response_description="Collection created successfully",
)
async def create_collection(name: str, req: CreateCollectionRequest):
    """
    Create a new vector collection.

    - **name**: URL-safe collection identifier (e.g. `my-docs`)
    - **dim**: Vector dimensionality — must match all inserted vectors
    - **metric**: Distance metric (`cosine`, `l2`, `inner_product`)
    - **index_type**: `hnsw` (default), `ivf`, `ivf_hnsw`, or `flat`
    """
    if name in collections_registry:
        raise HTTPException(
            status_code=409,
            detail=f"Collection '{name}' already exists. "
                   f"DELETE it first to recreate.",
        )

    try:
        col = novavec.Collection(
            name           = name,
            dim            = req.dim,
            metric         = req.metric.value,
            index_type     = req.index_type.value,
            M              = req.hnsw_config.M,
            ef_construction= req.hnsw_config.ef_construction,
            nlist          = req.ivf_config.nlist,
            nprobe         = req.ivf_config.nprobe,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    collections_registry[name] = col

    logger.info(
        "Collection created",
        extra={
            "collection": name,
            "dim":        req.dim,
            "metric":     req.metric.value,
            "index_type": req.index_type.value,
        },
    )

    return {
        "status":     "ok",
        "name":       name,
        "dim":        req.dim,
        "metric":     req.metric.value,
        "index_type": req.index_type.value,
    }


@router.delete(
    "/{name}",
    summary="Delete a collection",
)
async def delete_collection(name: str):
    """
    Delete a collection and all its vectors from memory.
    Does **not** delete persisted data from disk — use with care.
    """
    if name not in collections_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{name}' not found.",
        )
    del collections_registry[name]
    logger.info("Collection deleted", extra={"collection": name})
    return {"status": "ok", "deleted": name}


@router.get(
    "/{name}/info",
    summary="Get collection info",
)
async def collection_info(name: str):
    """Return metadata about a collection including vector count and config."""
    if name not in collections_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{name}' not found.",
        )
    col = collections_registry[name]
    cfg = col.config
    return {
        "name":          name,
        "vectors_count": len(col),
        "config": {
            "dim":             cfg["dim"],
            "M":               cfg["M"],
            "ef_construction": cfg["ef_construction"],
            "nlist":           cfg["nlist"],
            "nprobe":          cfg["nprobe"],
        },
    }


@router.get(
    "/",
    summary="List all collections",
)
async def list_collections():
    """Return a summary of all live collections."""
    return {
        "collections": [
            {
                "name":          n,
                "vectors_count": len(c),
            }
            for n, c in collections_registry.items()
        ]
    }
