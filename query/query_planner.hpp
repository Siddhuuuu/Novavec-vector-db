#pragma once

#include "core/types.hpp"
#include "storage/metadata_store.hpp"
#include <roaring.hh>

#include <optional>
#include <string>

// ============================================================
// QueryPlanner — selectivity-based filter strategy selection
// ============================================================
// The planner inspects the metadata index to estimate what
// fraction of documents match the filter (selectivity), then
// chooses the cheapest execution strategy:
//
// NONE (no filter):
//   Search top_k vectors, return as-is.
//
// POST_FILTER (high selectivity — filter matches many docs):
//   Fetch effective_top_k > top_k candidates, then filter.
//   effective_top_k = top_k * ceil(1 / selectivity) to guarantee
//   top_k valid results survive the filter.
//   Example: selectivity=0.5 → fetch 2*top_k, filter half,
//   expect top_k survivors.
//   Capped at QUERY_PLANNER_MAX_OVERFETCH * top_k.
//
// PRE_FILTER (low selectivity — filter matches few docs):
//   Pass a roaring bitmap of matching DocIds to the index.
//   During HNSW traversal, non-matching nodes are still
//   explored for graph connectivity but excluded from results.
//   This avoids scanning the entire dataset when only 5% of
//   vectors are relevant.
//
// Threshold rationale (default 0.2 = 20%):
//   Post-filter requires fetching top_k / selectivity candidates
//   to guarantee top_k valid results. When selectivity < 0.2,
//   this overfetch exceeds 5x top_k and hurts latency —
//   pre-filter with bitmap constraint is faster.
//   At selectivity = 0.2: overfetch = 5x (borderline).
//   At selectivity = 0.05: overfetch = 20x (very expensive post-filter).
// ============================================================

struct FilterSpec {
    std::string field;
    std::string value;
};

struct QueryPlan {
    FilterStrategy              strategy;
    float                       estimated_selectivity;
    int                         effective_top_k;   // may be inflated for post-filter
    std::optional<roaring::Roaring> filter_bitmap; // populated for PRE_FILTER/POST_FILTER
};

class QueryPlanner {
public:
    QueryPlan plan(
        const std::optional<FilterSpec>& filter,
        int top_k,
        const MetadataStore& meta,
        float threshold = 0.2f) const;
};
