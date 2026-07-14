#include "storage/segment_manager.hpp"
#include <mutex>
#include <shared_mutex>

#include <roaring.hh>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <queue>
#include <stdexcept>

namespace fs = std::filesystem;

// ============================================================
// Construction
// ============================================================

SegmentManager::SegmentManager(const std::string& data_dir,
                               const CollectionConfig& config)
    : data_dir_(data_dir)
    , config_(config)
{
    fs::create_directories(data_dir_);
    // Always create the initial mutable segment.
    // When loading a collection, Collection::load() calls load_sealed_segments()
    // immediately after construction, which replaces mutable_segment_ with the
    // WAL-replayed version. The segment created here gets overwritten cleanly.
    mutable_segment_ = std::make_shared<Segment>(
        next_segment_id_++, config_, data_dir_);
}

// ============================================================
// insert
// ============================================================

void SegmentManager::insert(DocId id,
                             const float* vec,
                             const std::string& metadata_json)
{
    // Write to mutable segment (no lock needed — only one writer at a time)
    mutable_segment_->insert(id, vec, metadata_json);

    // Check if we need to seal
    maybe_seal_mutable();
}

// ============================================================
// remove — soft-delete via WAL + index tombstone
// ============================================================

void SegmentManager::remove(DocId id) {
    // Write DELETE to the mutable segment's WAL for crash durability.
    // Pass dim=0 and nullptr for vector (not needed for delete ops).
    mutable_segment_->remove(id);
}

// ============================================================
// maybe_seal_mutable — seal and rotate mutable segment
// ============================================================

void SegmentManager::maybe_seal_mutable() {
    if (!mutable_segment_->should_seal()) return;

    // Seal the current mutable segment
    mutable_segment_->seal();

    // Create new mutable segment
    auto new_mutable = std::make_shared<Segment>(
        next_segment_id_++, config_, data_dir_);

    // Atomic swap under unique_lock
    std::unique_lock<std::shared_mutex> lock(segments_mutex_);
    segments_.push_back(std::move(mutable_segment_));
    mutable_segment_ = std::move(new_mutable);
}

// ============================================================
// search — fan out to all segments in parallel
// ============================================================

std::vector<SearchResult> SegmentManager::search(
    const float* query,
    int top_k,
    int ef_search,
    const roaring::Roaring* bitmap) const
{
    // Snapshot the segment list under shared lock
    std::vector<std::shared_ptr<Segment>> all_segments;
    {
        std::shared_lock<std::shared_mutex> lock(segments_mutex_);
        all_segments = segments_;
        all_segments.push_back(mutable_segment_);
    }

    // Launch parallel searches
    std::vector<std::future<std::vector<SearchResult>>> futures;
    futures.reserve(all_segments.size());

    for (const auto& seg : all_segments) {
        if (seg->size() == 0) continue;
        futures.push_back(std::async(
            std::launch::async,
            [&seg, query, top_k, ef_search, bitmap]() {
                return seg->search(query, top_k, ef_search, bitmap);
            }));
    }

    // Collect and merge
    std::vector<std::vector<SearchResult>> all_results;
    all_results.reserve(futures.size());
    for (auto& fut : futures) {
        all_results.push_back(fut.get());
    }

    return merge_results(all_results, top_k);
}

// ============================================================
// merge_results — global top-k via max-heap
// ============================================================

std::vector<SearchResult> SegmentManager::merge_results(
    std::vector<std::vector<SearchResult>>& all_results,
    int top_k)
{
    std::priority_queue<SearchResult,
                        std::vector<SearchResult>,
                        std::less<SearchResult>> heap;

    for (auto& seg_results : all_results) {
        for (auto& sr : seg_results) {
            if (static_cast<int>(heap.size()) < top_k) {
                heap.push(sr);
            } else if (sr.score < heap.top().score) {
                heap.pop();
                heap.push(sr);
            }
        }
    }

    std::vector<SearchResult> merged;
    merged.reserve(heap.size());
    while (!heap.empty()) {
        merged.push_back(heap.top());
        heap.pop();
    }
    std::sort(merged.begin(), merged.end());
    return merged;
}

// ============================================================
// total_size
// ============================================================

int SegmentManager::total_size() const {
    std::shared_lock<std::shared_mutex> lock(segments_mutex_);
    int total = mutable_segment_->size();
    for (const auto& seg : segments_) {
        total += seg->size();
    }
    return total;
}

// ============================================================
// save — seal and persist everything
// ============================================================

void SegmentManager::save() const {
    std::shared_lock<std::shared_mutex> lock(segments_mutex_);
    // Sealed segments already persisted — no-op for them
    // Mutable segment: WAL provides durability
    // Save segment count metadata for load_sealed_segments
    std::ofstream manifest(data_dir_ + "/segments.manifest");
    if (!manifest) return;
    manifest << "next_segment_id=" << next_segment_id_ << "\n";
    manifest << "sealed_count=" << segments_.size() << "\n";
    for (const auto& seg : segments_) {
        manifest << "sealed=" << seg->id() << "\n";
    }
    manifest << "mutable=" << mutable_segment_->id() << "\n";
}

// ============================================================
// load_sealed_segments — reconstruct from disk on startup
// ============================================================

void SegmentManager::load_sealed_segments() {
    std::string manifest_path = data_dir_ + "/segments.manifest";
    std::ifstream manifest(manifest_path);

    if (!manifest) {
        // No manifest: crash before first save(), or genuinely fresh start.
        // The constructor already created mutable_segment_ at id=0.
        // Replay its WAL to recover any inserts that survived the crash.
        mutable_segment_->replay_wal();
        return;
    }

    std::string line;
    int saved_mutable_id = -1;

    while (std::getline(manifest, line)) {
        if (line.substr(0, 7) == "sealed=") {
            int seg_id = std::stoi(line.substr(7));
            auto seg = std::make_shared<Segment>(seg_id, config_, data_dir_);
            seg->replay_wal();
            segments_.push_back(seg);
            if (seg_id >= next_segment_id_) {
                next_segment_id_ = seg_id + 1;
            }
        } else if (line.substr(0, 16) == "next_segment_id=") {
            int saved_next = std::stoi(line.substr(16));
            if (saved_next > next_segment_id_) {
                next_segment_id_ = saved_next;
            }
        } else if (line.substr(0, 8) == "mutable=") {
            saved_mutable_id = std::stoi(line.substr(8));
        }
    }

    // Replace the blank constructor segment with the WAL-recovered one.
    // saved_mutable_id is the segment id written by the last save().
    int mutable_id = (saved_mutable_id >= 0) ? saved_mutable_id
                                              : (next_segment_id_ - 1);
    mutable_segment_ = std::make_shared<Segment>(mutable_id, config_, data_dir_);
    mutable_segment_->replay_wal();

    if (mutable_id >= next_segment_id_) {
        next_segment_id_ = mutable_id + 1;
    }
}
