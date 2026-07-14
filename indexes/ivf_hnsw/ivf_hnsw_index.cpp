#include "indexes/ivf_hnsw/ivf_hnsw_index.hpp"

#include <roaring.hh>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <queue>
#include <stdexcept>

namespace fs = std::filesystem;

IVFHNSWIndex::IVFHNSWIndex(const CollectionConfig& config)
    : config_(config)
    , nlist_(config.nlist)
    , nprobe_(config.nprobe)
{}

void IVFHNSWIndex::train(const float* data, int n) {
    if (n < nlist_) {
        throw std::runtime_error(
            "IVFHNSWIndex::train: need at least nlist=" +
            std::to_string(nlist_) + " training vectors, got " +
            std::to_string(n));
    }

    // K-means quantization
    auto result = kmeans_train(data, n, config_.dim, nlist_);
    centroids_  = std::move(result.centroids);

    // Initialize one HNSW index per cluster
    cluster_indexes_.clear();
    cluster_indexes_.reserve(nlist_);
    for (int c = 0; c < nlist_; ++c) {
        cluster_indexes_.push_back(
            std::make_unique<HNSWIndex>(
                config_.dim,
                config_.M,
                config_.ef_construction,
                config_.metric));
    }

    trained_ = true;
}

void IVFHNSWIndex::insert(DocId id, const float* vec) {
    if (!trained_) {
        throw std::runtime_error(
            "IVFHNSWIndex::insert: must call train() before insert()");
    }

    // Assign to nearest centroid
    int c = assign_to_nearest_centroid(
        vec, centroids_.data(), nlist_, config_.dim, config_.metric);

    cluster_indexes_[c]->insert(id, vec);
    ++num_vectors_;
}

std::vector<SearchResult> IVFHNSWIndex::search(
    const float* query,
    int top_k,
    int ef_search,
    const roaring::Roaring* filter_bitmap)
{
    if (!trained_) {
        throw std::runtime_error("IVFHNSWIndex::search: index not trained");
    }

    // Find nprobe nearest centroids
    auto probe_cells = find_nearest_centroids(
        query, centroids_.data(), nlist_, config_.dim, nprobe_,
        config_.metric);

    // Launch parallel HNSW searches — one async task per cluster
    // Cap at IVF_HNSW_MAX_PARALLEL to avoid spawning too many threads
    const int n_probes = static_cast<int>(probe_cells.size());
    std::vector<std::future<std::vector<SearchResult>>> futures;
    futures.reserve(n_probes);

    for (int i = 0; i < n_probes; ++i) {
        int cluster_idx = probe_cells[i];
        if (cluster_indexes_[cluster_idx]->size() == 0) continue;

        futures.push_back(std::async(
            std::launch::async,
            [this, cluster_idx, query, top_k, ef_search, filter_bitmap]() {
                return cluster_indexes_[cluster_idx]->search(
                    query, top_k, ef_search, filter_bitmap);
            }));
    }

    // Collect all results and merge with global max-heap
    std::priority_queue<SearchResult,
                        std::vector<SearchResult>,
                        std::less<SearchResult>> heap;

    for (auto& fut : futures) {
        auto cluster_results = fut.get();
        for (auto& sr : cluster_results) {
            if (static_cast<int>(heap.size()) < top_k) {
                heap.push(sr);
            } else if (sr.score < heap.top().score) {
                heap.pop();
                heap.push(sr);
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

void IVFHNSWIndex::save(const std::string& path) const {
    fs::create_directories(path);

    // Save centroids and metadata
    std::ofstream meta(path + "/ivf_hnsw_meta.bin", std::ios::binary);
    if (!meta) throw std::runtime_error("IVFHNSWIndex::save: cannot open meta file");

    auto write_val = [&](const auto& v) {
        meta.write(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    write_val(nlist_);
    write_val(nprobe_);
    write_val(num_vectors_);
    write_val(config_.dim);
    write_val(config_.M);
    write_val(config_.ef_construction);
    uint8_t tr = trained_ ? 1 : 0;
    write_val(tr);

    meta.write(reinterpret_cast<const char*>(centroids_.data()),
               static_cast<std::streamsize>(
                   static_cast<size_t>(nlist_) * config_.dim * sizeof(float)));

    // Save each cluster HNSW
    for (int c = 0; c < nlist_; ++c) {
        cluster_indexes_[c]->save(
            path + "/cluster_" + std::to_string(c) + ".hnsw");
    }
}

void IVFHNSWIndex::load(const std::string& path) {
    std::ifstream meta(path + "/ivf_hnsw_meta.bin", std::ios::binary);
    if (!meta) throw std::runtime_error("IVFHNSWIndex::load: cannot open meta file");

    auto read_val = [&](auto& v) {
        meta.read(reinterpret_cast<char*>(&v), sizeof(v));
    };
    read_val(nlist_);
    read_val(nprobe_);
    read_val(num_vectors_);
    read_val(config_.dim);
    read_val(config_.M);
    read_val(config_.ef_construction);
    uint8_t tr = 0;
    read_val(tr);
    trained_ = (tr != 0);

    centroids_.resize(static_cast<size_t>(nlist_) * config_.dim);
    meta.read(reinterpret_cast<char*>(centroids_.data()),
              static_cast<std::streamsize>(
                  static_cast<size_t>(nlist_) * config_.dim * sizeof(float)));

    cluster_indexes_.clear();
    cluster_indexes_.reserve(nlist_);
    for (int c = 0; c < nlist_; ++c) {
        auto hnsw = std::make_unique<HNSWIndex>(
            config_.dim, config_.M, config_.ef_construction, config_.metric);
        std::string cluster_path =
            path + "/cluster_" + std::to_string(c) + ".hnsw";
        if (fs::exists(cluster_path)) {
            hnsw->load(cluster_path);
        }
        cluster_indexes_.push_back(std::move(hnsw));
    }
}
