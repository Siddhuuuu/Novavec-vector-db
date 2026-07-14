#pragma once

#include "core/types.hpp"
#include "query/query_planner.hpp"
#include "storage/metadata_store.hpp"
#include "storage/segment_manager.hpp"

#include <optional>
#include <vector>

// ============================================================
// QueryEngine — orchestrates planning, execution, and enrichment
// ============================================================
// Sits between the Collection API and the SegmentManager.
// Responsibilities:
//   1. Ask QueryPlanner for the execution strategy.
//   2. Execute the search via SegmentManager.
//   3. Apply post-filter if strategy == POST_FILTER.
//   4. Enrich each SearchResult with its metadata_json from
//      MetadataStore.
//   5. Return exactly top_k results.
// ============================================================
class QueryEngine {
public:
    QueryEngine(SegmentManager& segment_manager,
                MetadataStore&  metadata_store);

    std::vector<SearchResult> search(
        const float*                     query,
        int                              top_k,
        int                              ef_search,
        const std::optional<FilterSpec>& filter = std::nullopt) const;

private:
    SegmentManager& segment_manager_;
    MetadataStore&  metadata_store_;
    QueryPlanner    planner_;
};
