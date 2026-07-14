#pragma once

#include <vector>
#include <cstdint>

// ============================================================
// HNSWNode — one node in the Hierarchical Navigable Small World graph
// ============================================================
// Memory layout reasoning:
//   neighbors[l] stores the adjacency list at layer l.
//   Layer 0 has up to M0 = 2*M connections (denser base layer
//   improves recall without significantly increasing memory).
//   Upper layers (l > 0) have at most M connections each.
//
//   The neighbors are stored as internal integer indices into
//   the HNSWIndex::nodes_ and HNSWIndex::vectors_ arrays —
//   NOT as DocIds. This keeps adjacency lists small (int32_t
//   vs uint64_t) and avoids an indirection through id_map_
//   during graph traversal.
//
// Soft delete:
//   deleted = true marks the node as a tombstone. It is still
//   present in the graph and reachable during traversal (removing
//   it would break graph connectivity), but it is excluded from
//   search results. Full compaction is out of scope for this
//   implementation — see DESIGN.md §7 for discussion.
// ============================================================
struct HNSWNode {
    // neighbors[layer][k] = internal index of k-th neighbor at that layer
    // neighbors.size() == top_layer + 1
    std::vector<std::vector<int32_t>> neighbors;

    // Highest layer this node participates in (0-indexed)
    int32_t top_layer = 0;

    // Tombstone flag — node excluded from results but stays in graph
    bool deleted = false;
};
