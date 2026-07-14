#include "query/query_planner.hpp"

#include <roaring.hh>
#include <algorithm>
#include <cmath>

QueryPlan QueryPlanner::plan(
    const std::optional<FilterSpec>& filter,
    int top_k,
    const MetadataStore& meta,
    float threshold) const
{
    // ---- No filter: trivial plan --------------------------------
    if (!filter.has_value()) {
        return QueryPlan{
            FilterStrategy::NONE,
            1.0f,           // selectivity irrelevant
            top_k,
            std::nullopt
        };
    }

    const std::string& field = filter->field;
    const std::string& value = filter->value;

    // ---- Estimate selectivity from metadata inverted index ------
    float sel = meta.selectivity(field, value);

    // ---- Choose strategy based on selectivity vs threshold ------
    //
    // Post-filter requires fetching top_k/selectivity candidates
    // to guarantee top_k valid results. When selectivity < 0.2,
    // this over-fetch exceeds 5x and hurts latency — pre-filter
    // with bitmap constraint is faster.
    //
    // Example:
    //   selectivity=0.5 → need 2*top_k candidates (cheap post-filter)
    //   selectivity=0.1 → need 10*top_k candidates (expensive — use pre-filter)
    //   selectivity=0.01→ need 100*top_k candidates (very expensive — use pre-filter)

    if (sel > threshold) {
        // POST_FILTER: high selectivity — many results survive the filter.
        // Overfetch by ceil(1/sel), capped at QUERY_PLANNER_MAX_OVERFETCH * top_k.
        int overfetch = static_cast<int>(
            std::ceil(1.0f / std::max(sel, 1e-6f)));
        overfetch = std::max(overfetch, 1);

        const int effective_top_k = std::min(
            top_k * overfetch,
            top_k * QUERY_PLANNER_MAX_OVERFETCH);

        // Build bitmap now for post-filtering the results
        roaring::Roaring bitmap = meta.get_bitmap(field, value);

        return QueryPlan{
            FilterStrategy::POST_FILTER,
            sel,
            effective_top_k,
            std::move(bitmap)
        };
    } else {
        // PRE_FILTER: low selectivity — only a small fraction of vectors
        // are relevant. Build bitmap and pass it to the index so graph
        // traversal can avoid adding non-matching nodes to the result heap.
        // We still explore non-matching neighbors during HNSW traversal
        // to preserve graph connectivity — they are just excluded from results.
        //
        // Small buffer (2x top_k) ensures we have enough candidates even
        // if the bitmap filtering removes some near-boundary results.
        const int effective_top_k = top_k * 2;

        roaring::Roaring bitmap = meta.get_bitmap(field, value);

        return QueryPlan{
            FilterStrategy::PRE_FILTER,
            sel,
            effective_top_k,
            std::move(bitmap)
        };
    }
}
