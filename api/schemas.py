"""
Pydantic v2 schema models for the NovaVec REST API.
Shape is Qdrant-compatible to ease migration for existing users.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---- Enumerations -------------------------------------------

class MetricType(str, Enum):
    COSINE        = "cosine"
    L2            = "l2"
    INNER_PRODUCT = "inner_product"


class IndexType(str, Enum):
    HNSW     = "hnsw"
    IVF      = "ivf"
    IVF_HNSW = "ivf_hnsw"
    FLAT     = "flat"


# ---- Index configuration sub-models -------------------------

class HNSWConfig(BaseModel):
    """HNSW graph construction parameters."""
    M: int = Field(
        default=16,
        ge=2,
        le=128,
        description=(
            "Maximum number of bidirectional connections per node "
            "in upper layers. Layer 0 uses 2*M. "
            "Higher M → better recall, more memory."
        ),
    )
    ef_construction: int = Field(
        default=200,
        ge=10,
        le=2000,
        description=(
            "Beam width during graph construction. "
            "Higher ef_construction → better recall, slower indexing."
        ),
    )


class IVFConfig(BaseModel):
    """Inverted File index parameters."""
    nlist: int = Field(
        default=256,
        ge=1,
        description="Number of Voronoi cells (k-means clusters).",
    )
    nprobe: int = Field(
        default=32,
        ge=1,
        description=(
            "Number of cells scanned at query time. "
            "Higher nprobe → better recall, more latency."
        ),
    )


# ---- Collection management ----------------------------------

class CreateCollectionRequest(BaseModel):
    """Request body for POST /collections/{name}."""
    dim: int = Field(
        ge=1,
        le=65536,
        description="Dimensionality of vectors in this collection.",
    )
    metric: MetricType = Field(
        default=MetricType.COSINE,
        description="Distance metric used for similarity search.",
    )
    index_type: IndexType = Field(
        default=IndexType.HNSW,
        description="Index algorithm.",
    )
    hnsw_config: HNSWConfig = Field(
        default_factory=HNSWConfig,
        description="HNSW-specific parameters (ignored for FLAT/IVF).",
    )
    ivf_config: IVFConfig = Field(
        default_factory=IVFConfig,
        description="IVF-specific parameters (ignored for FLAT/HNSW).",
    )


# ---- Vector upsert ------------------------------------------

class PointStruct(BaseModel):
    """One vector point to upsert."""
    id: int = Field(description="Unique integer document ID.")
    vector: list[float] = Field(
        description="Dense float32 vector of length == collection.dim.",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary key-value metadata. String and numeric values "
            "are indexed for metadata filtering."
        ),
    )


class UpsertRequest(BaseModel):
    """Request body for POST /collections/{name}/points."""
    points: list[PointStruct] = Field(
        description="List of points to insert or update.",
    )


# ---- Search -------------------------------------------------

class MatchValue(BaseModel):
    """Exact-match condition value."""
    value: Any = Field(description="The value to match against.")


class FieldCondition(BaseModel):
    """Condition on a single metadata field."""
    key:   str        = Field(description="Metadata field name.")
    match: MatchValue = Field(description="Exact-match value wrapper.")


class FilterCondition(BaseModel):
    """Compound filter — currently supports 'must' (AND) conditions."""
    must: list[FieldCondition] = Field(
        default_factory=list,
        description=(
            "All conditions must be satisfied (AND semantics). "
            "Only the first condition is used in the current implementation."
        ),
    )


class SearchParams(BaseModel):
    """Index-specific search parameters."""
    ef: int = Field(
        default=100,
        ge=1,
        description="HNSW ef_search beam width.",
    )
    nprobe: int = Field(
        default=32,
        ge=1,
        description="IVF nprobe — number of cells to scan.",
    )


class SearchRequest(BaseModel):
    """Request body for POST /collections/{name}/points/search."""
    vector: list[float] = Field(
        description="Query vector (float32, length == collection.dim).",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=10000,
        description="Number of nearest neighbors to return.",
    )
    filter: Optional[FilterCondition] = Field(
        default=None,
        description=(
            "Optional metadata filter. "
            "Strategy (pre/post-filter) is chosen automatically "
            "based on estimated selectivity."
        ),
    )
    params: SearchParams = Field(
        default_factory=SearchParams,
        description="Index-specific search parameters.",
    )
    with_payload: bool = Field(
        default=True,
        description="Whether to include metadata payload in results.",
    )


# ---- Search response ----------------------------------------

class SearchResultResponse(BaseModel):
    """One result from a nearest-neighbor search."""
    id:      int              = Field(description="Document ID.")
    score:   float            = Field(
        description="Distance score. Lower is more similar for L2/cosine.",
    )
    payload: dict[str, Any]   = Field(
        default_factory=dict,
        description="Metadata payload (empty if with_payload=False).",
    )


class SearchResponse(BaseModel):
    """Response body for POST /collections/{name}/points/search."""
    results: list[SearchResultResponse] = Field(
        description="List of nearest neighbors, sorted by ascending score.",
    )
    time_ms: float = Field(
        description="Server-side search latency in milliseconds.",
    )
