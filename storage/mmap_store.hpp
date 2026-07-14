#pragma once

#include "core/types.hpp"
#include <string>

// ============================================================
// MmapVectorStore — immutable memory-mapped vector storage
// ============================================================
// Used for sealed (immutable) segments. Once a segment is sealed,
// its vector data is written to a flat binary file and accessed
// via mmap().
//
// Design rationale vs read():
//   mmap() maps the file into the process virtual address space.
//   Random reads go through the OS page cache without a separate
//   kernel buffer (no double buffering). For large vector datasets,
//   this reduces memory overhead and lets the OS page replacement
//   algorithm (LRU) manage which pages are hot.
//
// File format:
//   [num_vectors: uint32_t]  — 4 bytes
//   [dim:         uint32_t]  — 4 bytes
//   [vectors:     num_vectors * dim * sizeof(float) bytes]
//
// After mmap, madvise(MADV_RANDOM) hints to the OS that access
// is non-sequential, disabling readahead prefetch that would
// waste memory bandwidth for random vector lookups.
// ============================================================
class MmapVectorStore {
public:
    MmapVectorStore() = default;
    ~MmapVectorStore();

    // Disable copy — mmap region is tied to this object's lifetime
    MmapVectorStore(const MmapVectorStore&)            = delete;
    MmapVectorStore& operator=(const MmapVectorStore&) = delete;

    // Move is allowed
    MmapVectorStore(MmapVectorStore&& o) noexcept;
    MmapVectorStore& operator=(MmapVectorStore&& o) noexcept;

    // Write a flat float array to disk in mmap-compatible format.
    // Static — does not require an open store instance.
    static void save(const std::string& path,
                     const float* data,
                     int num_vectors,
                     int dim);

    // Map the file into virtual memory.
    void load(const std::string& path);

    // Zero-copy access — returns a raw pointer into the mmap'd region.
    // Caller must not write through this pointer (PROT_READ).
    const float* get(int internal_idx) const;

    int num_vectors() const { return num_vectors_; }
    int dim()         const { return dim_; }
    bool is_loaded()  const { return mmap_ptr_ != nullptr; }

private:
    void*  mmap_ptr_    = nullptr;
    size_t mmap_size_   = 0;
    int    fd_          = -1;
    int    num_vectors_ = 0;
    int    dim_         = 0;
};
