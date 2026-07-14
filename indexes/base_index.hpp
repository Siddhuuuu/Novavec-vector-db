#pragma once

#include "core/types.hpp"
#include <vector>
#include <string>

// Forward-declare roaring::Roaring to avoid pulling in all of CRoaring here
namespace roaring {
class Roaring;
}

// ============================================================
// BaseIndex — pure abstract interface for all index types
// ============================================================
// Every index type (FLAT, HNSW, IVF, IVF-HNSW) inherits from
// this interface. The QueryEngine and SegmentManager work
// exclusively through this interface, enabling runtime dispatch.
//
// filter_bitmap parameter:
//   When non-null, enables PRE_FILTER mode — only DocIds present
//   in the bitmap are eligible for the result set. During graph
//   traversal (HNSW), non-matching nodes are still explored to
//   maintain graph connectivity; they are simply excluded from
//   the results heap. For brute-force (FLAT, IVF), non-matching
//   vectors are skipped entirely.
// ============================================================
class BaseIndex {
public:
    // Insert a vector with its external document ID.
    // dim must match the index's configured dimensionality.
    virtual void insert(DocId id, const float* vec) = 0;

    // k-nearest neighbor search.
    // ef_search: beam width (HNSW only, ignored for FLAT).
    // filter_bitmap: optional pre-filter bitmap (see above).
    // Returns at most top_k results sorted by ascending distance.
    virtual std::vector<SearchResult> search(
        const float* query,
        int top_k,
        int ef_search,
        const roaring::Roaring* filter_bitmap = nullptr) = 0;

    // Serialize index to a directory or file path.
    virtual void save(const std::string& path) const = 0;

    // Deserialize index from a directory or file path.
    virtual void load(const std::string& path) = 0;

    // Number of vectors currently in the index.
    virtual int size() const = 0;

    // Dimensionality of vectors this index operates on.
    virtual int dim() const = 0;

    virtual ~BaseIndex() = default;
};
