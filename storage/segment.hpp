#pragma once

#include "core/types.hpp"
#include "indexes/base_index.hpp"
#include "storage/metadata_store.hpp"
#include "storage/wal.hpp"
#include <roaring.hh>

#include <memory>
#include <string>

// ============================================================
// Segment — a unit of storage combining index + metadata + WAL
// ============================================================
// A Collection contains one mutable segment (accepting writes)
// and zero or more sealed (immutable) segments.
//
// Lifecycle:
//   MUTABLE  — accepts inserts, backed by an in-memory index
//              and a WAL for crash recovery.
//   SEALED   — no new inserts; data flushed to mmap file.
//              HNSW graph compacted to disk.
//
// Sealing:
//   When vector_count_ >= size_threshold_, the segment is sealed:
//   1. WAL is closed.
//   2. Index data is persisted.
//   3. State transitions to SEALED.
//   After sealing, the segment accepts no new inserts.
//
// Query fan-out:
//   SegmentManager sends every search to all segments in parallel.
//   Each segment searches independently with its own index and
//   bitmap filter; the SegmentManager merges the results.
// ============================================================

enum class SegmentState {
    MUTABLE,
    SEALED
};

class Segment {
public:
    Segment(int segment_id,
            const CollectionConfig& config,
            const std::string& data_dir);

    // Insert a vector with metadata into this segment.
    void insert(DocId id,
                const float* vec,
                const std::string& metadata_json);

    // Soft-delete: WAL DELETE entry + index tombstone.
    void remove(DocId id);

    // Search this segment.
    std::vector<SearchResult> search(
        const float* query,
        int top_k,
        int ef_search,
        const roaring::Roaring* bitmap = nullptr) const;

    // Returns true when vector count reaches the size threshold.
    bool should_seal() const;

    // Transition to SEALED: persist index and metadata to disk.
    void seal();

    // Accessors
    int            id()           const { return segment_id_; }
    int            size()         const { return static_cast<int>(vector_count_); }
    SegmentState   state()        const { return state_; }
    bool           is_sealed()    const { return state_ == SegmentState::SEALED; }

    // Recover state from WAL (called on startup for mutable segments).
    void replay_wal();

private:
    SegmentState                 state_;
    std::unique_ptr<BaseIndex>   index_;
    MetadataStore                metadata_;
    std::unique_ptr<WALWriter>   wal_;

    int                          segment_id_;
    size_t                       vector_count_    = 0;
    size_t                       size_threshold_;
    std::string                  data_dir_;
    CollectionConfig             config_;

    // Construct the appropriate index type from config
    std::unique_ptr<BaseIndex> make_index() const;

    // Paths for persistence
    std::string wal_path()      const;
    std::string index_path()    const;
    std::string metadata_path() const;
};
