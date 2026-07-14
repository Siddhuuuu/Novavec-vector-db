#pragma once

#include "core/types.hpp"
#include "query/query_engine.hpp"
#include "query/query_planner.hpp"
#include "storage/metadata_store.hpp"
#include "storage/segment_manager.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

// ============================================================
// Collection — top-level user-facing C++ class
// ============================================================
// This is the single entry point for all user operations.
// pybind11 wraps this class directly.
//
// Responsibilities:
//   - Validate and normalize incoming vectors (L2-normalize for COSINE).
//   - Delegate inserts to SegmentManager + MetadataStore.
//   - Delegate searches to QueryEngine.
//   - Coordinate persistence (save/load).
//
// Normalization:
//   For COSINE metric, vectors are L2-normalized at insertion time.
//   This means cosine_distance(a, b) == 1 - dot(a, b) since
//   both |a| = |b| = 1, avoiding expensive sqrt + division at
//   query time. The query vector is also normalized at search time.
// ============================================================
class Collection {
public:
    Collection(const std::string& name, const CollectionConfig& config);

    // Insert or update a vector with optional JSON metadata.
    // If metric == COSINE, the vector is L2-normalized in-place
    // on a local copy before insertion (caller's array not modified).
    void upsert(DocId              id,
                const float*       vec,
                int                dim,
                const std::string& metadata_json = "{}");

    // Semantic search with optional metadata filter.
    // ef_search: HNSW beam width (larger = higher recall, more latency).
    // filter: optional {field: value} pair — routed by QueryPlanner.
    std::vector<SearchResult> search(
        const float*                     query,
        int                              dim,
        int                              top_k     = 10,
        int                              ef_search = HNSW_DEFAULT_EF_SEARCH,
        const std::optional<FilterSpec>& filter    = std::nullopt) const;

    // Soft-delete a vector by DocId.
    void remove(DocId id);

    // Persist collection to data_dir_.
    void save() const;

    // Load a previously saved collection from data_dir.
    static Collection load(const std::string& data_dir);

    // Accessors
    int              size()   const;
    CollectionConfig config() const { return config_; }
    std::string      name()   const { return name_; }

private:
    std::string      name_;
    std::string      data_dir_;
    CollectionConfig config_;

    // Owned storage and query components
    std::unique_ptr<MetadataStore>  metadata_store_;
    std::unique_ptr<SegmentManager> segment_manager_;
    std::unique_ptr<QueryEngine>    query_engine_;

    // Private constructor used by load()
    Collection() = default;

    // Config serialization path
    std::string config_path() const;
    std::string metadata_path() const;

    void save_config() const;
    static CollectionConfig load_config(const std::string& data_dir);
};
