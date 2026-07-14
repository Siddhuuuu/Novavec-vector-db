#pragma once

#include "core/types.hpp"
#include "storage/segment.hpp"
#include <roaring.hh>

#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

// ============================================================
// SegmentManager — manages the lifecycle of all segments
// ============================================================
// A Collection has exactly one SegmentManager.
//
// At any time there is:
//   - One mutable segment accepting writes
//   - Zero or more sealed (read-only) segments
//
// Sealing flow:
//   After each insert, maybe_seal_mutable() checks if the
//   mutable segment has reached the size threshold. If so:
//   1. Seal the current mutable segment (persist to disk).
//   2. Create a new mutable segment with id+1.
//   3. Atomically update the segments list under unique_lock.
//
// Parallel search:
//   Fan out to all segments using std::async(std::launch::async).
//   Each segment's search runs concurrently. Results are collected
//   via future.get() and merged using a global max-heap.
//
// Thread safety:
//   segments_ list: shared_mutex — shared for reads, unique for
//                   segment seal/create transitions.
// ============================================================
class SegmentManager {
public:
    SegmentManager(const std::string& data_dir,
                   const CollectionConfig& config);

    void insert(DocId id,
                const float* vec,
                const std::string& metadata_json);

    // Soft-delete: writes a DELETE WAL entry and tombstones the vector in the index.
    void remove(DocId id);

    std::vector<SearchResult> search(
        const float* query,
        int top_k,
        int ef_search,
        const roaring::Roaring* bitmap = nullptr) const;

    int total_size() const;

    // Persist all sealed segments (mutable segment is already WAL-backed)
    void save() const;

    // Load sealed segments from data directory on startup
    void load_sealed_segments();

private:
    std::vector<std::shared_ptr<Segment>> segments_;       // sealed segments
    std::shared_ptr<Segment>              mutable_segment_;
    mutable std::shared_mutex             segments_mutex_;
    std::string                           data_dir_;
    CollectionConfig                      config_;
    int                                   next_segment_id_ = SEGMENT_INITIAL_ID;

    // Check if mutable segment needs sealing; seal and create new if so.
    void maybe_seal_mutable();

    // Merge results from multiple segments using a global max-heap.
    static std::vector<SearchResult> merge_results(
        std::vector<std::vector<SearchResult>>& all_results,
        int top_k);
};
