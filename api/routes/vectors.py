"""
Vector mutation endpoints.
POST   /collections/{name}/points           — upsert vectors
DELETE /collections/{name}/points/{point_id} — delete a vector
"""

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas import UpsertRequest
from api.main import collections_registry, logger

router = APIRouter()


@router.post(
    "/{name}/points",
    summary="Upsert vectors",
    response_description="Number of points upserted",
)
async def upsert_points(name: str, req: UpsertRequest):
    """
    Insert or update vectors in a collection.

    - Vectors are identified by their **id** field.
    - If a vector with the same id already exists it is replaced.
    - For **cosine** collections, vectors are L2-normalized automatically.
    - **payload** values with string or numeric types are indexed for filtering.
    """
    if name not in collections_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{name}' not found.",
        )
    col = collections_registry[name]

    upserted = 0
    errors   = []

    for point in req.points:
        try:
            # Convert list[float] to contiguous float32 numpy array.
            # numpy ensures alignment and dtype before we pass to C++.
            vec = np.array(point.vector, dtype=np.float32)
            col.upsert(
                point.id,
                vec,
                point.payload if point.payload else None,
            )
            upserted += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"id": point.id, "error": str(exc)})

    if errors:
        logger.warning(
            "Partial upsert failure",
            extra={"collection": name, "errors": errors},
        )

    logger.info(
        "Points upserted",
        extra={"collection": name, "count": upserted},
    )

    response = {"status": "ok", "upserted": upserted}
    if errors:
        response["errors"] = errors  # type: ignore[assignment]
    return response


@router.delete(
    "/{name}/points/{point_id}",
    summary="Delete a vector",
)
async def delete_point(name: str, point_id: int):
    """
    Soft-delete a vector by its DocId.

    The vector is marked as deleted (tombstone) and excluded from
    future search results. Graph structure is not compacted —
    the node remains in memory to preserve HNSW graph connectivity.
    """
    if name not in collections_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{name}' not found.",
        )
    try:
        collections_registry[name].remove(point_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info(
        "Point deleted",
        extra={"collection": name, "point_id": point_id},
    )
    return {"status": "ok", "deleted": point_id}
