#pragma once

#include "indexes/base_index.hpp"
#include "indexes/ivf/kmeans.hpp"
#include "core/distance.hpp"
#include <roaring.hh>

#include <vector>
#include <string>

// ============================================================
// IVFIndex — Inverted File Index
// ============================================================
// Partitions the vector space into nlist Voronoi cells via
// k-means clustering. At search time, only the nprobe nearest
// cells are scanned, reducing work from O(N) to O(N/nlist * nprobe).
//
// Workflow:
//   1. train(data, n) — run k-means to obtain nlist centroids
//   2. insert(id, vec) — assign to nearest centroid, append to list
//   3. search(query, ...) — scan nprobe nearest cells
//
// Must call train() before insert(). Attempts to insert before
// training will throw std::runtime_error.
// ============================================================
class IVFIndex : public BaseIndex {
public:
    IVFIndex(int dim, int nlist, int nprobe, DistanceMetric metric);

    // Train the IVF quantizer on n vectors.
    // data: n x dim float array (flat, vector i at data + i*dim).
    void train(const float* data, int n);

    bool is_trained() const { return trained_; }

    void insert(DocId id, const float* vec) override;

    // Search nprobe nearest cells, collect candidates, return top_k.
    // Applies filter_bitmap if non-null.
    std::vector<SearchResult> search(
        const float* query,
        int top_k,
        int ef_search,          // ignored for IVF
        const roaring::Roaring* filter_bitmap = nullptr) override;

    void save(const std::string& path) const override;
    void load(const std::string& path) override;

    int size() const override { return num_vectors_; }
    int dim()  const override { return dim_; }

private:
    std::vector<float>              centroids_;  // flat: centroid i at i*dim_
    std::vector<std::vector<int>>   lists_;      // lists_[c] = internal indices in cell c
    std::vector<float>              vectors_;    // flat contiguous vector storage
    std::vector<DocId>              doc_ids_;    // internal_idx -> DocId

    int            dim_;
    int            nlist_;
    int            nprobe_;
    int            num_vectors_ = 0;
    bool           trained_     = false;
    DistanceMetric metric_;
};
