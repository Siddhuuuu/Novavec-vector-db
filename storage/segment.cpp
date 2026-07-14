#include "storage/segment.hpp"

#include "indexes/flat_index.hpp"
#include "indexes/hnsw/hnsw_index.hpp"
#include "indexes/ivf/ivf_index.hpp"
#include "indexes/ivf_hnsw/ivf_hnsw_index.hpp"
#include <roaring.hh>

#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

// ============================================================
// Construction
// ============================================================

Segment::Segment(int segment_id,
                 const CollectionConfig& config,
                 const std::string& data_dir)
    : state_(SegmentState::MUTABLE)
    , segment_id_(segment_id)
    , size_threshold_(config.segment_size_threshold)
    , data_dir_(data_dir)
    , config_(config)
{
    fs::create_directories(data_dir_);
    index_ = make_index();
    wal_   = std::make_unique<WALWriter>(wal_path(), config_.fsync_policy);
}

// ============================================================
// make_index — instantiate correct index type
// ============================================================

std::unique_ptr<BaseIndex> Segment::make_index() const {
    switch (config_.index_type) {
    case IndexType::FLAT:
        return std::make_unique<FlatIndex>(config_.dim, config_.metric);
    case IndexType::HNSW:
        return std::make_unique<HNSWIndex>(
            config_.dim, config_.M, config_.ef_construction, config_.metric);
    case IndexType::IVF:
        return std::make_unique<IVFIndex>(
            config_.dim, config_.nlist, config_.nprobe, config_.metric);
    case IndexType::IVF_HNSW:
        return std::make_unique<IVFHNSWIndex>(config_);
    default:
        throw std::runtime_error("Segment::make_index: unknown index type");
    }
}

// ============================================================
// Path helpers
// ============================================================

std::string Segment::wal_path() const {
    return data_dir_ + "/seg_" + std::to_string(segment_id_) + ".wal";
}

std::string Segment::index_path() const {
    return data_dir_ + "/seg_" + std::to_string(segment_id_) + ".idx";
}

std::string Segment::metadata_path() const {
    return data_dir_ + "/seg_" + std::to_string(segment_id_) + "_meta.json";
}

// ============================================================
// insert
// ============================================================

void Segment::insert(DocId id,
                     const float* vec,
                     const std::string& metadata_json)
{
    if (state_ == SegmentState::SEALED) {
        throw std::runtime_error("Segment::insert: cannot insert into sealed segment");
    }

    // 1. Write to WAL first (durability guarantee)
    wal_->append(WalOpType::INSERT, id, vec, config_.dim, metadata_json);

    // 2. Insert into in-memory index
    index_->insert(id, vec);

    // 3. Index metadata
    metadata_.insert(id, metadata_json);

    ++vector_count_;
}

// ============================================================
// remove — WAL DELETE + index tombstone
// ============================================================

void Segment::remove(DocId id) {
    if (state_ == SegmentState::SEALED) return; // sealed segments are immutable

    // Write DELETE to WAL first for crash durability.
    // dim=0, vec=nullptr — DELETE entries carry no vector payload.
    if (wal_) {
        wal_->append(WalOpType::DELETE, id, nullptr, 0, "{}");
    }

    // Tombstone in index
    if (auto* flat = dynamic_cast<FlatIndex*>(index_.get())) {
        flat->remove(id);
    } else if (auto* hnsw = dynamic_cast<HNSWIndex*>(index_.get())) {
        hnsw->remove(id);
    }

    // Remove from per-segment metadata
    metadata_.remove(id);

    if (vector_count_ > 0) --vector_count_;
}

// ============================================================
// search
// ============================================================

std::vector<SearchResult> Segment::search(
    const float* query,
    int top_k,
    int ef_search,
    const roaring::Roaring* bitmap) const
{
    return index_->search(query, top_k, ef_search, bitmap);
}

// ============================================================
// should_seal
// ============================================================

bool Segment::should_seal() const {
    return vector_count_ >= size_threshold_;
}

// ============================================================
// seal — persist and transition to immutable
// ============================================================

void Segment::seal() {
    if (state_ == SegmentState::SEALED) return;

    // Final WAL sync
    wal_->sync();
    wal_.reset(); // close WAL file

    // Persist index
    index_->save(index_path());

    // Persist metadata
    metadata_.save(metadata_path());

    state_ = SegmentState::SEALED;
}

// ============================================================
// replay_wal — reconstruct state from WAL on startup
// ============================================================

void Segment::replay_wal() {
    WALReader reader;
    auto entries = reader.replay(wal_path());

    for (const auto& entry : entries) {
        switch (entry.op_type) {
        case WalOpType::INSERT:
            if (!entry.vector.empty()) {
                index_->insert(entry.doc_id, entry.vector.data());
                metadata_.insert(entry.doc_id, entry.metadata_json);
                ++vector_count_;
            }
            break;
        case WalOpType::DELETE:
            // Attempt soft-delete — dynamic_cast to concrete types
            if (auto* flat = dynamic_cast<FlatIndex*>(index_.get())) {
                flat->remove(entry.doc_id);
            } else if (auto* hnsw = dynamic_cast<HNSWIndex*>(index_.get())) {
                hnsw->remove(entry.doc_id);
            }
            metadata_.remove(entry.doc_id);
            if (vector_count_ > 0) --vector_count_;
            break;
        }
    }
}
