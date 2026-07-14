#include "indexes/flat_index.hpp"

#include <roaring.hh>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

FlatIndex::FlatIndex(int dim, DistanceMetric metric)
    : dim_(dim), metric_(metric) {}

void FlatIndex::insert(DocId id, const float* vec) {
    // Append vector to flat storage
    const size_t offset = static_cast<size_t>(num_vectors_) * dim_;
    data_.resize(offset + dim_);
    std::memcpy(data_.data() + offset, vec, dim_ * sizeof(float));

    // Track id mapping
    id_to_idx_[id] = num_vectors_;
    ids_.push_back(id);
    ++num_vectors_;
}

std::vector<SearchResult> FlatIndex::search(
    const float* query,
    int top_k,
    int /*ef_search*/,
    const roaring::Roaring* filter_bitmap)
{
    // Fixed-size max-heap: O(N log k) — better than O(N log N) sort
    // for small k and large N. The max element is always at the top,
    // allowing cheap comparison and replacement.
    std::priority_queue<SearchResult,
                        std::vector<SearchResult>,
                        std::less<SearchResult>> heap; // max-heap

    for (int i = 0; i < num_vectors_; ++i) {
        const DocId doc_id = ids_[i];
        if (doc_id == INVALID_DOC_ID) continue; // soft-deleted

        // Pre-filter: skip vectors not in the filter bitmap
        if (filter_bitmap && !filter_bitmap->contains(doc_id)) continue;

        const float* vec = data_.data() + static_cast<size_t>(i) * dim_;
        float dist = compute_distance(query, vec, dim_, metric_);

        if (static_cast<int>(heap.size()) < top_k) {
            heap.push({doc_id, dist, ""});
        } else if (dist < heap.top().score) {
            heap.pop();
            heap.push({doc_id, dist, ""});
        }
    }

    // Extract results in ascending distance order
    std::vector<SearchResult> results;
    results.reserve(heap.size());
    while (!heap.empty()) {
        results.push_back(heap.top());
        heap.pop();
    }
    std::sort(results.begin(), results.end());
    return results;
}

void FlatIndex::remove(DocId id) {
    auto it = id_to_idx_.find(id);
    if (it == id_to_idx_.end()) return;
    // Tombstone: mark slot as invalid; storage not compacted
    ids_[it->second] = INVALID_DOC_ID;
    id_to_idx_.erase(it);
    --num_vectors_;
}

void FlatIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("FlatIndex::save: cannot open " + path);

    // Header: num_vectors (int32_t), dim (int32_t)
    const int32_t n = num_vectors_;
    const int32_t d = dim_;
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&d), sizeof(d));

    // Flat float array (includes tombstoned slots — they are skipped on load)
    out.write(reinterpret_cast<const char*>(data_.data()),
              static_cast<std::streamsize>(data_.size() * sizeof(float)));

    // DocId array
    out.write(reinterpret_cast<const char*>(ids_.data()),
              static_cast<std::streamsize>(ids_.size() * sizeof(DocId)));
}

void FlatIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("FlatIndex::load: cannot open " + path);

    int32_t n = 0, d = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&d), sizeof(d));

    if (d != dim_) {
        throw std::runtime_error(
            "FlatIndex::load: dim mismatch: file=" + std::to_string(d) +
            " index=" + std::to_string(dim_));
    }

    const size_t total_floats = static_cast<size_t>(n) * d;
    data_.resize(total_floats);
    in.read(reinterpret_cast<char*>(data_.data()),
            static_cast<std::streamsize>(total_floats * sizeof(float)));

    ids_.resize(n);
    in.read(reinterpret_cast<char*>(ids_.data()),
            static_cast<std::streamsize>(n * sizeof(DocId)));

    num_vectors_ = 0;
    id_to_idx_.clear();
    for (int i = 0; i < n; ++i) {
        if (ids_[i] != INVALID_DOC_ID) {
            id_to_idx_[ids_[i]] = i;
            ++num_vectors_;
        }
    }
}
