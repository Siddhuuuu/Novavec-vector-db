#include "indexes/hnsw/hnsw_index.hpp"

#include <roaring.hh>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <queue>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <limits>

// ============================================================
// Construction
// ============================================================

HNSWIndex::HNSWIndex(int dim, int M, int ef_construction, DistanceMetric metric)
    : dim_(dim)
    , M_(M)
    , M0_(M * HNSW_M0_MULTIPLIER)
    , ef_construction_(ef_construction)
    , mL_(1.0f / std::log(static_cast<float>(M)))
    , metric_(metric)
{}

// ============================================================
// Layer assignment
// ============================================================

int HNSWIndex::assign_level() const {
    // thread_local RNG: each thread has its own Mersenne Twister,
    // so parallel inserts never contend on a shared RNG lock.
    thread_local std::mt19937 rng{std::random_device{}()};
    // Exclude exact 0 to prevent -log(0) = +inf
    thread_local std::uniform_real_distribution<float> dist{
        std::numeric_limits<float>::min(), 1.0f};

    // Formula from Malkov & Yashunin 2018:
    // level = floor(-ln(uniform(0,1)) * mL)
    // Since mL = 1/ln(M), this gives P(level >= l) = (1/M)^l —
    // an exponentially decaying probability distribution ensuring
    // O(log N) expected layers.
    return static_cast<int>(-std::log(dist(rng)) * mL_);
}

// ============================================================
// search_layer — core beam search within one layer
// ============================================================

std::vector<SearchResult> HNSWIndex::search_layer(
    const float* query,
    int entry_point_idx,
    int ef,
    int layer,
    const roaring::Roaring* filter_bitmap) const
{
    // Increment generation counter — "clears" visited set in O(1)
    // without touching the vector memory.
    // Fetch-add gives this search a unique generation value.
    // Each concurrent search gets a distinct gen, so their visited marks
    // do not interfere even though visited_ is a shared array.
    const uint32_t gen = current_gen_.fetch_add(1, std::memory_order_relaxed) + 1;

    // Candidates: min-heap (best candidate at top)
    // Results:    max-heap (worst result at top, popped when overflow)
    using MaxHeap = std::priority_queue<SearchResult,
                                        std::vector<SearchResult>,
                                        std::less<SearchResult>>;
    using MinHeap = std::priority_queue<SearchResult,
                                        std::vector<SearchResult>,
                                        std::greater<SearchResult>>;

    float entry_dist = compute_distance(
        query, vectors_.data() + static_cast<size_t>(entry_point_idx) * dim_,
        dim_, metric_);

    // Mark entry point visited using relaxed atomics — correctness
    // requires only that this thread sees its own writes, which relaxed guarantees.
    visited_[entry_point_idx].store(gen, std::memory_order_relaxed);

    MinHeap candidates;
    MaxHeap results;

    SearchResult entry_sr{doc_ids_[entry_point_idx], entry_dist, ""};
    candidates.push(entry_sr);

    // Only add to results if not filtered out (or no filter)
    bool entry_in_filter = !filter_bitmap ||
                           filter_bitmap->contains(doc_ids_[entry_point_idx]);
    bool entry_deleted   = nodes_[entry_point_idx].deleted;
    if (entry_in_filter && !entry_deleted) {
        results.push(entry_sr);
    }

    while (!candidates.empty()) {
        const SearchResult best_candidate = candidates.top();
        candidates.pop();

        // Termination condition: best unexplored candidate is worse
        // than the worst result already in the heap.
        // If results is full (size == ef), no better result can come.
        if (!results.empty() &&
            static_cast<int>(results.size()) >= ef &&
            best_candidate.score > results.top().score)
        {
            break;
        }

        // Find the internal index of this candidate
        // (best_candidate.id is DocId; we need the internal idx)
        auto map_it = id_map_.find(best_candidate.id);
        if (map_it == id_map_.end()) continue;
        int c_idx = map_it->second;

        // Explore neighbors at this layer
        const auto& nbrs = nodes_[c_idx].neighbors;
        if (layer >= static_cast<int>(nbrs.size())) continue;
        const auto& layer_nbrs = nbrs[layer];

        for (int32_t nbr_idx : layer_nbrs) {
            if (nbr_idx < 0 || nbr_idx >= static_cast<int>(visited_.size())) continue;
            if (visited_[nbr_idx].load(std::memory_order_relaxed) == gen) continue;
            visited_[nbr_idx].store(gen, std::memory_order_relaxed);

            float nbr_dist = compute_distance(
                query,
                vectors_.data() + static_cast<size_t>(nbr_idx) * dim_,
                dim_, metric_);

            bool should_explore =
                results.empty() ||
                static_cast<int>(results.size()) < ef ||
                nbr_dist < results.top().score;

            if (should_explore) {
                SearchResult nbr_sr{doc_ids_[nbr_idx], nbr_dist, ""};
                candidates.push(nbr_sr);

                // PRE_FILTER: only add to results if the node passes the filter
                // and is not deleted. We still explored the neighbor above to
                // maintain graph connectivity — skipping traversal of non-matching
                // nodes would disconnect the graph and destroy recall.
                bool nbr_in_filter = !filter_bitmap ||
                                     filter_bitmap->contains(doc_ids_[nbr_idx]);
                bool nbr_deleted   = nodes_[nbr_idx].deleted;
                if (nbr_in_filter && !nbr_deleted) {
                    results.push(nbr_sr);
                    if (static_cast<int>(results.size()) > ef) {
                        results.pop(); // discard worst
                    }
                }
            }
        }
    }

    // Extract results into a vector (max-heap order, unsorted)
    std::vector<SearchResult> out;
    out.reserve(results.size());
    while (!results.empty()) {
        out.push_back(results.top());
        results.pop();
    }
    return out;
}

// ============================================================
// select_neighbors_heuristic — Algorithm 4 from Malkov 2020
// ============================================================

std::vector<int32_t> HNSWIndex::select_neighbors_heuristic(
    const float* query_vec,
    std::vector<SearchResult>& candidates,
    int M_max,
    int /*layer*/) const
{
    // Sort candidates by ascending distance to the query
    std::sort(candidates.begin(), candidates.end());

    std::vector<int32_t> selected;
    selected.reserve(M_max);

    for (const auto& cand : candidates) {
        if (static_cast<int>(selected.size()) >= M_max) break;

        // Look up the internal index for this candidate
        auto it = id_map_.find(cand.id);
        if (it == id_map_.end()) continue;
        int cand_idx = it->second;

        const float* cand_vec =
            vectors_.data() + static_cast<size_t>(cand_idx) * dim_;

        // Diverse selection criterion:
        // Add candidate only if it is the nearest neighbor from the
        // query for at least one direction not covered by already-selected
        // neighbors. Formally: cand is added if for ALL already-selected
        // neighbors s, dist(query, cand) < dist(cand, s).
        // This ensures that each neighbor "covers a different direction"
        // from the query — preventing all neighbors from clustering together.
        bool is_diverse = true;
        for (int32_t sel_idx : selected) {
            const float* sel_vec =
                vectors_.data() + static_cast<size_t>(sel_idx) * dim_;
            float dist_cand_to_selected =
                compute_distance(cand_vec, sel_vec, dim_, metric_);
            if (dist_cand_to_selected < cand.score) {
                // Another selected neighbor is closer to cand than cand is
                // to the query — cand doesn't add a new direction
                is_diverse = false;
                break;
            }
        }

        if (is_diverse || selected.empty()) {
            selected.push_back(cand_idx);
        }
    }

    return selected;
}

// ============================================================
// connect_bidirectional — add edge a<->b, pruning if necessary
// ============================================================

void HNSWIndex::connect_bidirectional(int a, int b, int layer, int M_max) {
    // Add b as neighbor of a
    auto& a_nbrs = nodes_[a].neighbors[layer];
    if (std::find(a_nbrs.begin(), a_nbrs.end(), b) == a_nbrs.end()) {
        a_nbrs.push_back(b);
    }

    // Add a as neighbor of b
    auto& b_nbrs = nodes_[b].neighbors[layer];
    if (std::find(b_nbrs.begin(), b_nbrs.end(), a) == b_nbrs.end()) {
        b_nbrs.push_back(a);
    }

    // Prune a's neighbors if over capacity
    if (static_cast<int>(a_nbrs.size()) > M_max) {
        const float* a_vec =
            vectors_.data() + static_cast<size_t>(a) * dim_;
        std::vector<SearchResult> cands;
        cands.reserve(a_nbrs.size());
        for (int32_t n : a_nbrs) {
            float d = compute_distance(
                a_vec,
                vectors_.data() + static_cast<size_t>(n) * dim_,
                dim_, metric_);
            cands.push_back({doc_ids_[n], d, ""});
        }
        auto pruned = select_neighbors_heuristic(a_vec, cands, M_max, layer);
        a_nbrs = pruned;
    }

    // Prune b's neighbors if over capacity
    if (static_cast<int>(b_nbrs.size()) > M_max) {
        const float* b_vec =
            vectors_.data() + static_cast<size_t>(b) * dim_;
        std::vector<SearchResult> cands;
        cands.reserve(b_nbrs.size());
        for (int32_t n : b_nbrs) {
            float d = compute_distance(
                b_vec,
                vectors_.data() + static_cast<size_t>(n) * dim_,
                dim_, metric_);
            cands.push_back({doc_ids_[n], d, ""});
        }
        auto pruned = select_neighbors_heuristic(b_vec, cands, M_max, layer);
        b_nbrs = pruned;
    }
}

// ============================================================
// insert — multi-layer graph construction
// ============================================================

void HNSWIndex::insert(DocId id, const float* vec) {
    const int new_idx = num_vectors_;

    // Step 1: copy vector into flat storage
    vectors_.resize(static_cast<size_t>(new_idx + 1) * dim_);
    std::memcpy(vectors_.data() + static_cast<size_t>(new_idx) * dim_,
                vec, dim_ * sizeof(float));

    doc_ids_.push_back(id);
    ++num_vectors_;

    // Step 2: grow visited set if needed (each element is a std::atomic<uint32_t>)
    if (new_idx >= static_cast<int>(visited_.size())) {
        // atomic<uint32_t> is not copyable; resize requires default-constructible,
        // which atomic is (default value 0 = "unvisited"). We use a fresh vector
        // and move it in to avoid the copy issue.
        std::vector<std::atomic<uint32_t>> new_vis(
            static_cast<size_t>(new_idx + 1) * HNSW_VISITED_INITIAL_FACTOR);
        // Copy existing values into new vector
        for (size_t i = 0; i < visited_.size(); ++i) {
            new_vis[i].store(visited_[i].load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
        }
        visited_ = std::move(new_vis);
    }

    // Step 3: assign level — cap at max_layer_+1 to prevent isolated nodes
    int level = assign_level();
    if (entry_point_ >= 0) {
        level = std::min(level, max_layer_ + 1);
    }

    // Step 4: create node with appropriate neighbor lists
    HNSWNode node;
    node.top_layer = level;
    node.neighbors.resize(level + 1);
    for (int l = 0; l <= level; ++l) {
        const int cap = (l == 0) ? M0_ : M_;
        node.neighbors[l].reserve(cap);
    }
    nodes_.push_back(std::move(node));

    // Step 5: first node — initialize graph
    if (entry_point_ == -1) {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        id_map_[id] = new_idx;
        entry_point_ = new_idx;
        max_layer_   = level;
        return;
    }

    // Step 6: take write lock for graph modification
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    id_map_[id] = new_idx;

    int curr_ep = entry_point_;

    // Step 7: coarse descent — navigate from max_layer_ to level+1
    // using greedy single-neighbor search (ef=1) to find good entry point
    for (int l = max_layer_; l > level; --l) {
        auto layer_results = search_layer(vec, curr_ep, 1, l);
        if (!layer_results.empty()) {
            auto it = id_map_.find(layer_results[0].id);
            if (it != id_map_.end()) curr_ep = it->second;
        }
    }

    // Step 8: fine descent — for each layer from min(level,max_layer_) down to 0,
    // find ef_construction_ candidates and connect new node to selected neighbors
    for (int l = std::min(level, max_layer_); l >= 0; --l) {
        const int M_max = (l == 0) ? M0_ : M_;

        auto candidates = search_layer(vec, curr_ep, ef_construction_, l);

        // Update entry point for next lower layer
        if (!candidates.empty()) {
            std::sort(candidates.begin(), candidates.end());
            auto it = id_map_.find(candidates[0].id);
            if (it != id_map_.end()) curr_ep = it->second;
        }

        // Select diverse neighbors for the new node
        auto selected = select_neighbors_heuristic(vec, candidates, M_max, l);

        // Add bidirectional edges
        for (int32_t nbr_idx : selected) {
            connect_bidirectional(new_idx, nbr_idx, l, M_max);
        }
    }

    // Step 9: update global entry point if new node is on a higher layer
    if (level > max_layer_) {
        max_layer_   = level;
        entry_point_ = new_idx;
    }
}

// ============================================================
// search — full multi-layer nearest neighbor search
// ============================================================

std::vector<SearchResult> HNSWIndex::search(
    const float* query,
    int top_k,
    int ef_search,
    const roaring::Roaring* filter_bitmap)
{
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);

    if (entry_point_ == -1) return {};

    int curr_ep = entry_point_;

    // Coarse descent: from max_layer_ down to layer 1
    // ef=1 — greedy single-step to approach the query region
    for (int l = max_layer_; l >= 1; --l) {
        auto results = search_layer(query, curr_ep, 1, l);
        if (!results.empty()) {
            std::sort(results.begin(), results.end());
            auto it = id_map_.find(results[0].id);
            if (it != id_map_.end()) curr_ep = it->second;
        }
    }

    // Fine search at layer 0: ef = max(ef_search, top_k) to ensure
    // enough candidates even when ef_search < top_k
    const int ef = std::max(ef_search, top_k);
    auto results = search_layer(query, curr_ep, ef, 0, filter_bitmap);

    // Sort by ascending distance and return top_k
    std::sort(results.begin(), results.end());
    if (static_cast<int>(results.size()) > top_k) {
        results.resize(top_k);
    }
    return results;
}

// ============================================================
// remove — soft delete
// ============================================================

void HNSWIndex::remove(DocId id) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto it = id_map_.find(id);
    if (it == id_map_.end()) return;
    nodes_[it->second].deleted = true;
    id_map_.erase(it);
}

// ============================================================
// save — binary serialization
// ============================================================

void HNSWIndex::save(const std::string& path) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("HNSWIndex::save: cannot open " + path);

    auto write = [&](const auto& val) {
        out.write(reinterpret_cast<const char*>(&val), sizeof(val));
    };

    write(M_);
    write(M0_);
    write(ef_construction_);
    write(mL_);
    write(dim_);
    write(num_vectors_);
    write(entry_point_);
    write(max_layer_);

    // Flat vector array
    out.write(reinterpret_cast<const char*>(vectors_.data()),
              static_cast<std::streamsize>(
                  static_cast<size_t>(num_vectors_) * dim_ * sizeof(float)));

    // DocId array
    out.write(reinterpret_cast<const char*>(doc_ids_.data()),
              static_cast<std::streamsize>(num_vectors_ * sizeof(DocId)));

    // Node data
    for (int i = 0; i < num_vectors_; ++i) {
        const auto& node = nodes_[i];
        write(node.top_layer);
        uint8_t del = node.deleted ? 1 : 0;
        write(del);
        for (int l = 0; l <= node.top_layer; ++l) {
            uint32_t cnt = static_cast<uint32_t>(node.neighbors[l].size());
            write(cnt);
            out.write(reinterpret_cast<const char*>(node.neighbors[l].data()),
                      static_cast<std::streamsize>(cnt * sizeof(int32_t)));
        }
    }
}

// ============================================================
// load — binary deserialization
// ============================================================

void HNSWIndex::load(const std::string& path) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("HNSWIndex::load: cannot open " + path);

    auto read = [&](auto& val) {
        in.read(reinterpret_cast<char*>(&val), sizeof(val));
    };

    int saved_M0 = 0;
    read(M_);
    read(saved_M0);
    M0_ = saved_M0;
    read(ef_construction_);
    read(mL_);
    read(dim_);
    read(num_vectors_);
    read(entry_point_);
    read(max_layer_);

    const size_t total_floats =
        static_cast<size_t>(num_vectors_) * dim_;
    vectors_.resize(total_floats);
    in.read(reinterpret_cast<char*>(vectors_.data()),
            static_cast<std::streamsize>(total_floats * sizeof(float)));

    doc_ids_.resize(num_vectors_);
    in.read(reinterpret_cast<char*>(doc_ids_.data()),
            static_cast<std::streamsize>(num_vectors_ * sizeof(DocId)));

    nodes_.resize(num_vectors_);
    id_map_.clear();
    for (int i = 0; i < num_vectors_; ++i) {
        auto& node = nodes_[i];
        read(node.top_layer);
        uint8_t del = 0;
        read(del);
        node.deleted = (del != 0);
        node.neighbors.resize(node.top_layer + 1);
        for (int l = 0; l <= node.top_layer; ++l) {
            uint32_t cnt = 0;
            read(cnt);
            node.neighbors[l].resize(cnt);
            in.read(reinterpret_cast<char*>(node.neighbors[l].data()),
                    static_cast<std::streamsize>(cnt * sizeof(int32_t)));
        }
        if (!node.deleted) {
            id_map_[doc_ids_[i]] = i;
        }
    }

    visited_ = std::vector<std::atomic<uint32_t>>(
        static_cast<size_t>(num_vectors_) * HNSW_VISITED_INITIAL_FACTOR);
    current_gen_.store(1, std::memory_order_relaxed);
}
