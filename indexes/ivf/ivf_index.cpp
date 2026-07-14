#include "indexes/ivf/ivf_index.hpp"

#include <roaring.hh>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <queue>
#include <stdexcept>

IVFIndex::IVFIndex(int dim, int nlist, int nprobe, DistanceMetric metric)
    : dim_(dim), nlist_(nlist), nprobe_(nprobe), metric_(metric)
{
    lists_.resize(nlist_);
}

void IVFIndex::train(const float* data, int n) {
    if (n < nlist_) {
        throw std::runtime_error(
            "IVFIndex::train: need at least nlist=" + std::to_string(nlist_) +
            " training vectors, got " + std::to_string(n));
    }
    auto result = kmeans_train(data, n, dim_, nlist_);
    centroids_  = std::move(result.centroids);
    trained_    = true;
    // Reset inverted lists
    for (auto& lst : lists_) lst.clear();
}

void IVFIndex::insert(DocId id, const float* vec) {
    if (!trained_) {
        throw std::runtime_error(
            "IVFIndex::insert: must call train() before insert()");
    }

    // Assign to nearest centroid (full scan over nlist centroids)
    int c = assign_to_nearest_centroid(
        vec, centroids_.data(), nlist_, dim_, metric_);

    // Append to flat storage
    const size_t offset = static_cast<size_t>(num_vectors_) * dim_;
    vectors_.resize(offset + dim_);
    std::memcpy(vectors_.data() + offset, vec, dim_ * sizeof(float));

    doc_ids_.push_back(id);
    lists_[c].push_back(num_vectors_);
    ++num_vectors_;
}

std::vector<SearchResult> IVFIndex::search(
    const float* query,
    int top_k,
    int /*ef_search*/,
    const roaring::Roaring* filter_bitmap)
{
    if (!trained_) {
        throw std::runtime_error("IVFIndex::search: index not trained");
    }

    // Find nprobe nearest centroids
    auto probe_cells = find_nearest_centroids(
        query, centroids_.data(), nlist_, dim_, nprobe_, metric_);

    // Fixed-size max-heap for O(candidates * log k) collection
    std::priority_queue<SearchResult,
                        std::vector<SearchResult>,
                        std::less<SearchResult>> heap;

    for (int cell : probe_cells) {
        for (int internal_idx : lists_[cell]) {
            const DocId doc_id = doc_ids_[internal_idx];

            if (filter_bitmap && !filter_bitmap->contains(doc_id)) continue;

            const float* v =
                vectors_.data() + static_cast<size_t>(internal_idx) * dim_;
            float dist = compute_distance(query, v, dim_, metric_);

            if (static_cast<int>(heap.size()) < top_k) {
                heap.push({doc_id, dist, ""});
            } else if (dist < heap.top().score) {
                heap.pop();
                heap.push({doc_id, dist, ""});
            }
        }
    }

    // Extract and sort
    std::vector<SearchResult> results;
    results.reserve(heap.size());
    while (!heap.empty()) {
        results.push_back(heap.top());
        heap.pop();
    }
    std::sort(results.begin(), results.end());
    return results;
}

void IVFIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("IVFIndex::save: cannot open " + path);

    auto write_val = [&](const auto& v) {
        out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    };

    write_val(dim_);
    write_val(nlist_);
    write_val(nprobe_);
    write_val(num_vectors_);
    uint8_t tr = trained_ ? 1 : 0;
    write_val(tr);

    // Centroids
    out.write(reinterpret_cast<const char*>(centroids_.data()),
              static_cast<std::streamsize>(
                  static_cast<size_t>(nlist_) * dim_ * sizeof(float)));

    // Flat vector storage
    out.write(reinterpret_cast<const char*>(vectors_.data()),
              static_cast<std::streamsize>(
                  static_cast<size_t>(num_vectors_) * dim_ * sizeof(float)));

    // DocId array
    out.write(reinterpret_cast<const char*>(doc_ids_.data()),
              static_cast<std::streamsize>(num_vectors_ * sizeof(DocId)));

    // Inverted lists
    for (int c = 0; c < nlist_; ++c) {
        uint32_t sz = static_cast<uint32_t>(lists_[c].size());
        write_val(sz);
        out.write(reinterpret_cast<const char*>(lists_[c].data()),
                  static_cast<std::streamsize>(sz * sizeof(int)));
    }
}

void IVFIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("IVFIndex::load: cannot open " + path);

    auto read_val = [&](auto& v) {
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
    };

    read_val(dim_);
    read_val(nlist_);
    read_val(nprobe_);
    read_val(num_vectors_);
    uint8_t tr = 0;
    read_val(tr);
    trained_ = (tr != 0);

    centroids_.resize(static_cast<size_t>(nlist_) * dim_);
    in.read(reinterpret_cast<char*>(centroids_.data()),
            static_cast<std::streamsize>(
                static_cast<size_t>(nlist_) * dim_ * sizeof(float)));

    vectors_.resize(static_cast<size_t>(num_vectors_) * dim_);
    in.read(reinterpret_cast<char*>(vectors_.data()),
            static_cast<std::streamsize>(
                static_cast<size_t>(num_vectors_) * dim_ * sizeof(float)));

    doc_ids_.resize(num_vectors_);
    in.read(reinterpret_cast<char*>(doc_ids_.data()),
            static_cast<std::streamsize>(num_vectors_ * sizeof(DocId)));

    lists_.resize(nlist_);
    for (int c = 0; c < nlist_; ++c) {
        uint32_t sz = 0;
        read_val(sz);
        lists_[c].resize(sz);
        in.read(reinterpret_cast<char*>(lists_[c].data()),
                static_cast<std::streamsize>(sz * sizeof(int)));
    }
}
