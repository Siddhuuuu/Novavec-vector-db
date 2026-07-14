#pragma once

#include "core/types.hpp"

#include <roaring.hh>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================
// MetadataStore — inverted index for structured metadata filters
// ============================================================
// Architecture:
//   index_[field][value] = roaring::Roaring bitmap of DocIds
//
// CRoaring bitmaps compress sets of DocIds efficiently using
// run-length encoding for dense ranges and hash maps for sparse
// sets. Intersection (AND) of two bitmaps is O(output_size)
// using SIMD-accelerated roaring operations.
//
// Selectivity estimation:
//   selectivity(field, value) = |bitmap| / total_docs_
//   Used by the QueryPlanner to choose between PRE_FILTER and
//   POST_FILTER strategies. A value of 0.1 means 10% of all
//   documents match this field/value — pre-filter is beneficial.
//
// Thread safety:
//   std::shared_mutex — shared_lock for reads, unique_lock for writes.
// ============================================================
class MetadataStore {
public:
    // Insert document metadata.
    // Parses JSON and indexes every string/number field value.
    void insert(DocId id, const std::string& metadata_json);

    // Remove document from all bitmaps.
    void remove(DocId id);

    // Get bitmap for a specific field=value match.
    // Returns empty bitmap if field or value not found.
    roaring::Roaring get_bitmap(
        const std::string& field,
        const std::string& value) const;

    // Intersect multiple field=value filters (AND semantics).
    // Returns bitmap of DocIds matching ALL conditions.
    roaring::Roaring intersect_filters(
        const std::vector<std::pair<std::string, std::string>>& filters) const;

    // Selectivity: fraction of total documents matching field=value.
    // Returns 1.0 if field/value not found (conservative — treat as full scan).
    float selectivity(
        const std::string& field,
        const std::string& value) const;

    // Retrieve full metadata JSON for a document.
    std::string get_metadata(DocId id) const;

    // Serialize to file using nlohmann/json.
    void save(const std::string& path) const;

    // Deserialize from file.
    void load(const std::string& path);

    int total_docs() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return total_docs_;
    }

private:
    // field -> value -> set of DocIds
    std::unordered_map<
        std::string,
        std::unordered_map<std::string, roaring::Roaring>> index_;

    // DocId -> full metadata JSON
    std::unordered_map<DocId, std::string> doc_metadata_;

    int total_docs_ = 0;

    mutable std::shared_mutex mutex_;
};
