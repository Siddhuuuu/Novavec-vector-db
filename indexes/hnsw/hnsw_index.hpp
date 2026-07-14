#pragma once

#include "indexes/base_index.hpp"
#include "indexes/hnsw/hnsw_node.hpp"
#include "core/distance.hpp"
#include <roaring.hh>

#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include <atomic>
#include <random>
#include <string>
#include <limits>

// ============================================================
// HNSWIndex — Hierarchical Navigable Small World graph index
// ============================================================
// Reference: Malkov & Yashunin, "Efficient and Robust Approximate
// Nearest Neighbor Search Using Hierarchical Navigable Small World
// Graphs", IEEE TPAMI 2020.
//
// Key implementation decisions:
//
// 1. Flat contiguous vector storage (vectors_ is a plain float
//    array, NOT std::vector<float> per node). This keeps all
//    vectors in a single allocation, improving cache locality
//    during distance computation.
//
// 2. Generation counter visited set: a single std::vector<uint32_t>
//    visited_ is shared across all queries. A query increments
//    current_gen_ before starting, then marks visited_[idx] =
//    current_gen_. Checking visited is O(1) with no heap allocation
//    or clear() call per query. Memory: O(N) total, zero per query.
//
// 3. Diverse neighbor selection heuristic (Algorithm 4 in paper):
//    Instead of naive k-nearest, neighbors are selected to be
//    "diverse" — each new candidate must be closer to the query
//    than to any already-selected neighbor. This improves recall
//    by ensuring the neighborhood covers multiple directions.
//
// 4. std::shared_mutex for reader-writer concurrency: reads
//    (search) take shared_lock, writes (insert) take unique_lock.
//
// 5. thread_local RNG in assign_level(): each thread has its own
//    Mersenne Twister, avoiding lock contention in parallel inserts.
// ============================================================
class HNSWIndex : public BaseIndex {
public:
    HNSWIndex(int dim, int M, int ef_construction, DistanceMetric metric);

    void insert(DocId id, const float* vec) override;

    std::vector<SearchResult> search(
        const float* query,
        int top_k,
        int ef_search,
        const roaring::Roaring* filter_bitmap = nullptr) override;

    void save(const std::string& path) const override;
    void load(const std::string& path) override;

    int size() const override { return num_vectors_; }
    int dim()  const override { return dim_; }

    // Soft-delete by DocId
    void remove(DocId id);

private:
    // ---- Graph structure ----------------------------------------
    std::vector<HNSWNode> nodes_;          // graph nodes, one per vector
    std::vector<float>    vectors_;        // flat storage: vector i at i*dim_
    std::vector<DocId>    doc_ids_;        // internal_idx -> external DocId
    std::unordered_map<DocId, int> id_map_; // external DocId -> internal_idx

    // ---- Index parameters ---------------------------------------
    int           dim_;
    int           M_;               // max connections per node at upper layers
    int           M0_;              // max connections at layer 0 = 2*M
    int           ef_construction_; // beam width during graph construction
    float         mL_;              // level normalization: 1/ln(M)
    DistanceMetric metric_;

    // ---- Graph state --------------------------------------------
    int           entry_point_ = -1; // internal idx of global entry point
    int           max_layer_   = 0;  // current highest layer in the graph
    int           num_vectors_ = 0;

    // ---- Concurrency --------------------------------------------
    mutable std::shared_mutex rw_mutex_;

    // ---- Generation counter visited set -------------------------
    // Generation counter visited set.
    // visited_[i] == current_gen_ means node i was visited by the current query.
    // Unvisited nodes have visited_[i] < current_gen_ (default-initialized to 0).
    // current_gen_ starts at 1 so the zero-initialized array means "unvisited".
    // Both are atomic: multiple concurrent shared-lock searches each fetch-add
    // current_gen_ to get a unique generation value. A stale write from one
    // query uses a different generation and is invisible to other searches.
    mutable std::vector<std::atomic<uint32_t>> visited_;
    mutable std::atomic<uint32_t> current_gen_{1};

    // ---- Private helpers ----------------------------------------

    // Exponential layer assignment from the HNSW paper.
    // P(level >= l) = (1/M)^l — ensures the expected number of
    // nodes at each level decreases exponentially, giving O(log N)
    // layers with high probability.
    int assign_level() const;

    // Beam search within a single layer.
    // Returns up to ef candidate results closest to query at the given layer.
    // When filter_bitmap is non-null (PRE_FILTER mode): non-matching nodes
    // are still traversed for graph exploration but excluded from results.
    std::vector<SearchResult> search_layer(
        const float* query,
        int entry_point_idx,
        int ef,
        int layer,
        const roaring::Roaring* filter_bitmap = nullptr) const;

    // Diverse neighbor selection heuristic (Algorithm 4, Malkov 2020).
    // Selects up to M_max diverse neighbors from candidates.
    // candidates must be pre-sorted by ascending distance.
    std::vector<int32_t> select_neighbors_heuristic(
        const float* query_vec,
        std::vector<SearchResult>& candidates,
        int M_max,
        int layer) const;

    // Add a bidirectional edge between node a and node b at the given layer.
    // If either node already has M_max connections, prunes via heuristic.
    void connect_bidirectional(int a, int b, int layer, int M_max);
};
