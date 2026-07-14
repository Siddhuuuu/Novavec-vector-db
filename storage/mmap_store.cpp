#include "storage/mmap_store.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ============================================================
// Destructor — unmap and close
// ============================================================

MmapVectorStore::~MmapVectorStore() {
    if (mmap_ptr_ != nullptr && mmap_ptr_ != MAP_FAILED) {
        munmap(mmap_ptr_, mmap_size_);
        mmap_ptr_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

// ============================================================
// Move semantics
// ============================================================

MmapVectorStore::MmapVectorStore(MmapVectorStore&& o) noexcept
    : mmap_ptr_(o.mmap_ptr_)
    , mmap_size_(o.mmap_size_)
    , fd_(o.fd_)
    , num_vectors_(o.num_vectors_)
    , dim_(o.dim_)
{
    o.mmap_ptr_    = nullptr;
    o.mmap_size_   = 0;
    o.fd_          = -1;
    o.num_vectors_ = 0;
    o.dim_         = 0;
}

MmapVectorStore& MmapVectorStore::operator=(MmapVectorStore&& o) noexcept {
    if (this != &o) {
        // Clean up existing resources
        if (mmap_ptr_ != nullptr && mmap_ptr_ != MAP_FAILED) {
            munmap(mmap_ptr_, mmap_size_);
        }
        if (fd_ >= 0) close(fd_);

        mmap_ptr_    = o.mmap_ptr_;
        mmap_size_   = o.mmap_size_;
        fd_          = o.fd_;
        num_vectors_ = o.num_vectors_;
        dim_         = o.dim_;

        o.mmap_ptr_    = nullptr;
        o.mmap_size_   = 0;
        o.fd_          = -1;
        o.num_vectors_ = 0;
        o.dim_         = 0;
    }
    return *this;
}

// ============================================================
// save — write header + float data to disk
// ============================================================

void MmapVectorStore::save(
    const std::string& path,
    const float* data,
    int num_vectors,
    int dim)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error(
            "MmapVectorStore::save: cannot open " + path);
    }

    // Header: num_vectors and dim as uint32_t
    const uint32_t n = static_cast<uint32_t>(num_vectors);
    const uint32_t d = static_cast<uint32_t>(dim);
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&d), sizeof(d));

    // Flat float array
    const size_t bytes =
        static_cast<size_t>(num_vectors) * dim * sizeof(float);
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(bytes));

    if (!out) {
        throw std::runtime_error(
            "MmapVectorStore::save: write failed for " + path);
    }
}

// ============================================================
// load — mmap the file
// ============================================================

void MmapVectorStore::load(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error(
            "MmapVectorStore::load: cannot open " + path +
            ": " + std::strerror(errno));
    }

    // Get file size
    struct stat st{};
    if (fstat(fd_, &st) < 0) {
        close(fd_); fd_ = -1;
        throw std::runtime_error(
            "MmapVectorStore::load: fstat failed: " +
            std::string(std::strerror(errno)));
    }
    mmap_size_ = static_cast<size_t>(st.st_size);

    if (mmap_size_ < MMAP_HEADER_BYTES) {
        close(fd_); fd_ = -1;
        throw std::runtime_error(
            "MmapVectorStore::load: file too small: " + path);
    }

    // Map file into virtual address space — read-only, shared
    mmap_ptr_ = mmap(nullptr,
                     mmap_size_,
                     PROT_READ,
                     MAP_SHARED,
                     fd_,
                     0);
    if (mmap_ptr_ == MAP_FAILED) {
        close(fd_); fd_ = -1; mmap_ptr_ = nullptr;
        throw std::runtime_error(
            "MmapVectorStore::load: mmap failed: " +
            std::string(std::strerror(errno)));
    }

    // Hint: random access pattern — suppress readahead prefetch
    madvise(mmap_ptr_, mmap_size_, MADV_RANDOM);

    // Read header
    const uint32_t* header =
        static_cast<const uint32_t*>(mmap_ptr_);
    num_vectors_ = static_cast<int>(header[0]);
    dim_         = static_cast<int>(header[1]);
}

// ============================================================
// get — zero-copy pointer arithmetic into mmap'd region
// ============================================================

const float* MmapVectorStore::get(int internal_idx) const {
    // Skip the 8-byte header (two uint32_t fields)
    // Then offset by internal_idx * dim_ floats
    const auto* base =
        static_cast<const float*>(mmap_ptr_) +
        (MMAP_HEADER_BYTES / sizeof(float));
    return base + static_cast<size_t>(internal_idx) * dim_;
}
