#pragma once

#include "indexes/base_index.hpp"
#include "core/distance.hpp"
#include <roaring.hh>

#include <vector>
#include <unordered_map>
#include <queue>
#include <string>

// ============================================================
// FlatIndex — exact brute-force nearest neighbor search
// ============================================================
// Storage layout:
//   data_      — flat contiguous float array; vector i lives at
//                data_.data() + i * dim_
//   ids_       — parallel DocId array; ids_[i] is the external
//                document ID for internal index i
//   id_to_idx_ — reverse map for O(1) soft-delete
//
// Search complexity: O(N * dim) per query.
// Used as the correctness oracle in benchmarks.
// ============================================================
class FlatIndex : public BaseIndex {
public:
    explicit FlatIndex(int dim, DistanceMetric metric);

    void insert(DocId id, const float* vec) override;

    // O(N log k) search using a fixed-size max-heap.
    // The heap stores at most top_k elements and discards the
    // worst result when it overflows — better than sorting N
    // elements when k << N.
    std::vector<SearchResult> search(
        const float* query,
        int top_k,
        int ef_search,                          // ignored for FLAT
        const roaring::Roaring* filter_bitmap = nullptr) override;

    // Soft-delete by DocId: marks slot as deleted (sets ids_[i] = INVALID_DOC_ID).
    // Does not compact storage.
    void remove(DocId id);

    void save(const std::string& path) const override;
    void load(const std::string& path) override;

    int size() const override { return num_vectors_; }
    int dim()  const override { return dim_; }

private:
    std::vector<float>  data_;        // flat contiguous vector storage
    std::vector<DocId>  ids_;         // internal_idx -> DocId
    std::unordered_map<DocId, int> id_to_idx_; // DocId -> internal_idx

    int            dim_;
    int            num_vectors_ = 0;
    DistanceMetric metric_;
};
