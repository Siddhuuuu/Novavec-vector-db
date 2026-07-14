#pragma once

#include <cstdint>
#include <string>
#include <optional>
#include <limits>

// ============================================================
// Fundamental scalar types
// ============================================================
using DocId    = uint32_t;
using Distance = float;

// Sentinel value for "no document" / uninitialized
inline constexpr DocId   INVALID_DOC_ID  = std::numeric_limits<DocId>::max();
inline constexpr Distance MAX_DISTANCE   = std::numeric_limits<Distance>::max();

// ============================================================
// Enumerations
// ============================================================
enum class DistanceMetric {
    L2,
    COSINE,
    INNER_PRODUCT
};

enum class IndexType {
    FLAT,
    HNSW,
    IVF,
    IVF_HNSW
};

enum class FsyncPolicy {
    SYNC_ALWAYS,    // fsync after every write — maximum durability
    SYNC_PERIODIC,  // fsync every 200 ms — balanced
    SYNC_NEVER      // never fsync — maximum throughput, data loss on crash
};

enum class FilterStrategy {
    NONE,        // no metadata filter
    PRE_FILTER,  // apply bitmap mask during graph traversal
    POST_FILTER  // fetch more results, apply filter after search
};

// ============================================================
// Core structs
// ============================================================

struct SearchResult {
    DocId       id            = INVALID_DOC_ID;
    Distance    score         = MAX_DISTANCE;
    std::string metadata_json = "{}";

    bool operator<(const SearchResult& o) const { return score < o.score; }
    bool operator>(const SearchResult& o) const { return score > o.score; }
};

struct CollectionConfig {
    int           dim                         = 128;
    DistanceMetric metric                     = DistanceMetric::COSINE;
    IndexType      index_type                 = IndexType::HNSW;

    // HNSW parameters
    int           M                           = 16;
    int           ef_construction             = 200;

    // IVF parameters
    int           nlist                       = 256;
    int           nprobe                      = 32;

    // Durability
    FsyncPolicy   fsync_policy                = FsyncPolicy::SYNC_PERIODIC;

    // Query planner selectivity threshold.
    // When filter selectivity > threshold, use POST_FILTER (overfetch then filter).
    // When filter selectivity <= threshold, use PRE_FILTER (bitmap mask in graph traversal).
    float         filter_selectivity_threshold = 0.2f;

    // Segment size: seal and compact when a segment exceeds this many vectors
    size_t        segment_size_threshold      = 100000;
};

// ============================================================
// Named constants — no magic numbers anywhere in the codebase
// ============================================================

// HNSW layer-0 has 2*M max connections (denser graph at ground layer)
inline constexpr int HNSW_M0_MULTIPLIER         = 2;

// Default HNSW beam width during construction
inline constexpr int HNSW_DEFAULT_EF_SEARCH      = 100;

// Initial visited-set capacity multiplier
inline constexpr int HNSW_VISITED_INITIAL_FACTOR = 2;

// Cosine normalization check tolerance
inline constexpr float NORM_TOLERANCE            = 1e-5f;

// WAL entry magic bytes: "NVWC"
inline constexpr uint8_t WAL_MAGIC[4]            = {0x4E, 0x56, 0x57, 0x43};
inline constexpr uint64_t WAL_SEQ_START          = 1;

// WAL periodic sync interval in milliseconds
inline constexpr int WAL_SYNC_INTERVAL_MS        = 200;

// mmap header size: num_vectors (uint32_t) + dim (uint32_t) = 8 bytes
inline constexpr size_t MMAP_HEADER_BYTES        = sizeof(uint32_t) * 2;

// Query planner post-filter overfetch multiplier cap
inline constexpr int QUERY_PLANNER_MAX_OVERFETCH = 10;

// K-means default max iterations and convergence tolerance
inline constexpr int    KMEANS_DEFAULT_MAX_ITER   = 300;
inline constexpr float  KMEANS_DEFAULT_TOL        = 1e-4f;
inline constexpr uint64_t KMEANS_DEFAULT_SEED     = 42;

// Minimum cluster size before reinitializing empty centroid
inline constexpr int KMEANS_MIN_CLUSTER_SIZE      = 1;

// IVF-HNSW parallel search uses std::async — each cluster searched concurrently
// Maximum parallel cluster searches launched at once
inline constexpr int IVF_HNSW_MAX_PARALLEL       = 64;

// Segment manager: initial mutable segment id
inline constexpr int SEGMENT_INITIAL_ID          = 0;
