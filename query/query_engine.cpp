#include "query/query_engine.hpp"

#include <roaring.hh>
#include <algorithm>

QueryEngine::QueryEngine(SegmentManager& segment_manager,
                         MetadataStore&  metadata_store)
    : segment_manager_(segment_manager)
    , metadata_store_(metadata_store)
{}

std::vector<SearchResult> QueryEngine::search(
    const float*                     query,
    int                              top_k,
    int                              ef_search,
    const std::optional<FilterSpec>& filter) const
{
    // Step 1: Determine execution strategy
    QueryPlan plan = planner_.plan(
        filter,
        top_k,
        metadata_store_,
        /*threshold=*/0.2f);

    std::vector<SearchResult> results;

    switch (plan.strategy) {

    case FilterStrategy::NONE: {
        // No filter — straightforward search
        results = segment_manager_.search(
            query, top_k, ef_search, /*bitmap=*/nullptr);
        break;
    }

    case FilterStrategy::POST_FILTER: {
        // Fetch more candidates than needed, then filter by bitmap.
        // effective_top_k is inflated by 1/selectivity to guarantee
        // at least top_k valid results survive the filter.
        results = segment_manager_.search(
            query, plan.effective_top_k, ef_search, /*bitmap=*/nullptr);

        // Apply bitmap filter to the over-fetched results
        if (plan.filter_bitmap.has_value()) {
            const auto& bm = plan.filter_bitmap.value();
            std::vector<SearchResult> filtered;
            filtered.reserve(top_k);
            for (auto& r : results) {
                if (bm.contains(static_cast<uint32_t>(r.id))) {
                    filtered.push_back(std::move(r));
                    if (static_cast<int>(filtered.size()) == top_k) break;
                }
            }
            results = std::move(filtered);
        }
        break;
    }

    case FilterStrategy::PRE_FILTER: {
        // Pass bitmap directly to the index so graph traversal
        // only includes matching nodes in the results heap.
        // Non-matching nodes are still traversed for connectivity.
        const roaring::Roaring* bm_ptr =
            plan.filter_bitmap.has_value()
            ? &plan.filter_bitmap.value()
            : nullptr;

        results = segment_manager_.search(
            query, plan.effective_top_k, ef_search, bm_ptr);

        // Trim to top_k (effective_top_k may be 2*top_k)
        if (static_cast<int>(results.size()) > top_k) {
            results.resize(top_k);
        }
        break;
    }
    }

    // Step 2: Enrich results with metadata from the metadata store.
    // The index stores no metadata — only DocId and distance.
    for (auto& r : results) {
        r.metadata_json = metadata_store_.get_metadata(r.id);
    }

    return results;
}
