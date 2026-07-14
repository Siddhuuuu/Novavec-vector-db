"""
NovaVec REST API — FastAPI application.
Thin HTTP layer over the C++ engine via pybind11 bindings.
Auto-generated Swagger docs at /docs, ReDoc at /redoc.
Prometheus metrics at /metrics.
"""

import time
import uuid
import logging
import json
from contextlib import asynccontextmanager
from typing import Dict

import novavec
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ---- Structured JSON logging --------------------------------

class JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON for log aggregation pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        extra = getattr(record, "extra", {})
        base.update(extra)
        return json.dumps(base)


_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
logging.root.handlers = [_handler]
logging.root.setLevel(logging.INFO)

logger = logging.getLogger("novavec.api")

# ---- Prometheus metrics -------------------------------------

REQUEST_COUNT = Counter(
    "novavec_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "novavec_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

# ---- Global collection registry ----------------------------
# Maps collection name -> novavec.Collection instance.
# All routes share this dict — no external state backend needed
# for single-node deployment.
collections_registry: Dict[str, novavec.Collection] = {}


# ---- Application lifespan -----------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    logger.info("NovaVec API starting up")
    yield
    # Graceful shutdown: persist all live collections
    for name, col in collections_registry.items():
        try:
            col.save()
            logger.info("Saved collection on shutdown",
                        extra={"collection": name})
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save collection on shutdown",
                         extra={"collection": name, "error": str(exc)})
    logger.info("NovaVec API shut down cleanly")


# ---- FastAPI app --------------------------------------------

app = FastAPI(
    title="NovaVec",
    description=(
        "Vector database built from first principles in C++. "
        "Qdrant-compatible REST API over a hand-rolled HNSW engine "
        "with AVX2 SIMD distance kernels, WAL durability, "
        "and selectivity-based query planning."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Observability middleware --------------------------------

@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()

    response = await call_next(request)

    latency   = time.perf_counter() - start
    path      = request.url.path
    method    = request.method
    status    = response.status_code

    REQUEST_COUNT.labels(
        method=method, path=path, status=str(status)
    ).inc()
    REQUEST_LATENCY.labels(method=method, path=path).observe(latency)

    logger.info(
        "request",
        extra={
            "request_id": request_id,
            "method":     method,
            "path":       path,
            "status":     status,
            "latency_ms": round(latency * 1000, 2),
        },
    )
    return response


# ---- Include routers ----------------------------------------
# Import here (after collections_registry is defined) to avoid
# circular import when routes import collections_registry.
from api.routes import collections, vectors, search  # noqa: E402

app.include_router(collections.router,
                   prefix="/collections",
                   tags=["collections"])
app.include_router(vectors.router,
                   prefix="/collections",
                   tags=["vectors"])
app.include_router(search.router,
                   prefix="/collections",
                   tags=["search"])


# ---- Utility endpoints --------------------------------------

@app.get("/health", tags=["system"],
         summary="Health check")
async def health():
    """Returns 200 OK when the API is ready to serve requests."""
    return {
        "status":      "ok",
        "collections": len(collections_registry),
    }


@app.get("/metrics", tags=["system"],
         summary="Prometheus metrics",
         response_class=PlainTextResponse)
async def metrics():
    """Expose Prometheus metrics in text exposition format."""
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/", tags=["system"], include_in_schema=False)
async def root():
    return {
        "name":    "NovaVec",
        "version": "1.0.0",
        "docs":    "/docs",
    }
