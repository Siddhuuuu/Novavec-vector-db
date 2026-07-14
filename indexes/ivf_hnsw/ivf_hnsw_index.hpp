#pragma once

#include "indexes/base_index.hpp"
#include "indexes/hnsw/hnsw_index.hpp"
#include "indexes/ivf/kmeans.hpp"
#include "core/types.hpp"
#include <roaring.hh>

#include <memory>
#include <vector>
#include <string>

// ============================================================
// IVFHNSWIndex — IVF partitioning with per-cluster HNSW indexes
// ============================================================
// Combines two ideas:
//   - IVF: coarse quantization partitions vectors into nlist
//     Voronoi cells, reducing the search space to nprobe cells.
//   - HNSW: within each cell, an HNSW graph provides sub-linear
//     search instead of IVF's flat linear scan.
//
// When this beats pure IVF:
//   For large nlist (many clusters) and large individual clusters,
//   the HNSW per-cluster sub-linear traversal amortizes the overhead
//   of maintaining nlist separate HNSW graphs. At 1M vectors with
//   nlist=256, each cluster has ~3906 vectors — HNSW search within
//   a cluster is O(log 3906 * M * ef) ≈ O(12 * 16 * 50) = O(9600)
//   vs IVF flat scan O(3906 * dim) = O(3906 * 128) = O(500k) ops.
//
// Parallel cluster search:
//   Each of the nprobe cluster HNSW searches is launched as a
//   std::async(std::launch::async) future, giving wall-clock
//   time O(single_cluster_search_time) rather than O(nprobe *
//   single_cluster_search_time).
// ============================================================
class IVFHNSWIndex : public BaseIndex {
public:
    IVFHNSWIndex(const CollectionConfig& config);

    // Train the IVF quantizer on n vectors.
    // data: n x dim float array.
    void train(const float* data, int n);

    bool is_trained() const { return trained_; }

    void insert(DocId id, const float* vec) override;

    // Parallel HNSW search across nprobe clusters.
    std::vector<SearchResult> search(
        const float* query,
        int top_k,
        int ef_search,
        const roaring::Roaring* filter_bitmap = nullptr) override;

    void save(const std::string& path) const override;
    void load(const std::string& path) override;

    int size() const override { return num_vectors_; }
    int dim()  const override { return config_.dim; }

private:
    CollectionConfig                            config_;
    std::vector<float>                          centroids_;       // flat: nlist * dim
    std::vector<std::unique_ptr<HNSWIndex>>     cluster_indexes_; // one per cluster
    int                                         nlist_;
    int                                         nprobe_;
    bool                                        trained_     = false;
    int                                         num_vectors_ = 0;
};
