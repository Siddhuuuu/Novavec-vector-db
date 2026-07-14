#include "storage/wal.hpp"
#include <mutex>
#include <shared_mutex>

#include <zlib.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <vector>

// ============================================================
// WALWriter — construction and destruction
// ============================================================

WALWriter::WALWriter(const std::string& path, FsyncPolicy policy)
    : seq_counter_(WAL_SEQ_START)
    , policy_(policy)
{
    // Open or create WAL file — append-only
    fd_ = open(path.c_str(),
               O_WRONLY | O_CREAT | O_APPEND,
               0644);
    if (fd_ < 0) {
        throw std::runtime_error(
            "WALWriter: cannot open " + path +
            ": " + std::strerror(errno));
    }

    if (policy_ == FsyncPolicy::SYNC_PERIODIC) {
        sync_thread_ = std::thread(&WALWriter::run_periodic_sync, this);
    }
}

WALWriter::~WALWriter() {
    if (policy_ == FsyncPolicy::SYNC_PERIODIC) {
        // Wake periodic sync thread and join
        {
            std::unique_lock<std::mutex> lk(sync_cv_mutex_);
            stop_sync_.store(true);
        }
        sync_cv_.notify_all();
        if (sync_thread_.joinable()) sync_thread_.join();
    }
    // Final sync regardless of policy
    if (fd_ >= 0) {
        fsync(fd_);
        close(fd_);
        fd_ = -1;
    }
}

// ============================================================
// Periodic sync background thread
// ============================================================

void WALWriter::run_periodic_sync() {
    while (!stop_sync_.load()) {
        std::unique_lock<std::mutex> lk(sync_cv_mutex_);
        sync_cv_.wait_for(lk,
            std::chrono::milliseconds(WAL_SYNC_INTERVAL_MS),
            [this] { return stop_sync_.load(); });
        if (fd_ >= 0) {
            fsync(fd_);
        }
    }
}

// ============================================================
// WALWriter::sync — explicit fsync
// ============================================================

void WALWriter::sync() {
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (fd_ >= 0) fsync(fd_);
}

// ============================================================
// WALWriter::append — write one entry
// ============================================================

void WALWriter::append(
    WalOpType op,
    DocId id,
    const float* vec,
    int dim,
    const std::string& metadata_json)
{
    std::lock_guard<std::mutex> lock(write_mutex_);

    // Accumulate all bytes into a buffer so we can compute CRC32
    // over the entire entry before writing.
    std::vector<uint8_t> buf;

    // Reserve approximate size upfront
    const size_t approx = 4 + 8 + 1 + 4 + 4 +
                          static_cast<size_t>(dim) * sizeof(float) +
                          4 + metadata_json.size() + 4;
    buf.reserve(approx);

    auto push_bytes = [&](const void* src, size_t len) {
        const auto* p = static_cast<const uint8_t*>(src);
        buf.insert(buf.end(), p, p + len);
    };

    auto push_u8  = [&](uint8_t  v) { buf.push_back(v); };
    auto push_u32 = [&](uint32_t v) { push_bytes(&v, sizeof(v)); };
    auto push_u64 = [&](uint64_t v) { push_bytes(&v, sizeof(v)); };

    // Magic
    push_bytes(WAL_MAGIC, 4);

    // seq_id (little-endian on x86-64 by default)
    const uint64_t seq = seq_counter_++;
    push_u64(seq);

    // op_type
    push_u8(static_cast<uint8_t>(op));

    // doc_id
    push_u32(static_cast<uint32_t>(id));

    // dim
    push_u32(static_cast<uint32_t>(dim));

    // vector bytes — only written for INSERT ops (dim > 0 and vec != nullptr)
    const size_t vec_bytes = (dim > 0 && vec != nullptr)
                             ? static_cast<size_t>(dim) * sizeof(float)
                             : 0;
    if (vec_bytes > 0) {
        push_bytes(vec, vec_bytes);
    }

    // metadata
    const uint32_t meta_len =
        static_cast<uint32_t>(metadata_json.size());
    push_u32(meta_len);
    if (meta_len > 0) {
        push_bytes(metadata_json.data(), meta_len);
    }

    // CRC32 over all preceding bytes
    uLong crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc,
                reinterpret_cast<const Bytef*>(buf.data()),
                static_cast<uInt>(buf.size()));
    uint32_t crc32_val = static_cast<uint32_t>(crc);
    push_u32(crc32_val);

    // Write entire entry atomically (single write syscall)
    ssize_t written = write(fd_,
                            buf.data(),
                            static_cast<ssize_t>(buf.size()));
    if (written < 0 || static_cast<size_t>(written) != buf.size()) {
        throw std::runtime_error(
            "WALWriter::append: write failed: " + std::string(std::strerror(errno)));
    }

    if (policy_ == FsyncPolicy::SYNC_ALWAYS) {
        fsync(fd_);
    }
}

// ============================================================
// WALReader::replay — parse and validate all entries
// ============================================================

std::vector<WalEntry> WALReader::replay(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        // WAL doesn't exist yet — normal on first startup
        return {};
    }

    // Read entire file into memory for efficient parsing
    in.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);

    std::vector<uint8_t> raw(file_size);
    in.read(reinterpret_cast<char*>(raw.data()),
            static_cast<std::streamsize>(file_size));

    std::vector<WalEntry> entries;
    size_t pos = 0;

    while (pos < file_size) {
        const size_t entry_start = pos;

        // Need at least magic(4) + seq(8) + op(1) + id(4) + dim(4) + meta_len(4) + crc(4) = 29 bytes
        constexpr size_t MIN_ENTRY = 4 + 8 + 1 + 4 + 4 + 4 + 4;
        if (pos + MIN_ENTRY > file_size) break; // truncated

        // Validate magic
        if (std::memcmp(raw.data() + pos, WAL_MAGIC, 4) != 0) {
            std::cerr << "WALReader: bad magic at offset " << pos
                      << " — stopping replay\n";
            break;
        }

        size_t field_pos = pos;

        // We'll parse fields and accumulate bytes for CRC validation
        // Parse magic (already validated)
        field_pos += 4;

        // seq_id
        uint64_t seq_id = 0;
        std::memcpy(&seq_id, raw.data() + field_pos, 8);
        field_pos += 8;

        // op_type
        uint8_t op_byte = raw[field_pos++];
        WalOpType op_type = static_cast<WalOpType>(op_byte);

        // doc_id
        uint32_t doc_id = 0;
        std::memcpy(&doc_id, raw.data() + field_pos, 4);
        field_pos += 4;

        // dim
        uint32_t dim = 0;
        std::memcpy(&dim, raw.data() + field_pos, 4);
        field_pos += 4;

        // vector bytes
        const size_t vec_bytes = static_cast<size_t>(dim) * sizeof(float);
        if (field_pos + vec_bytes > file_size) break; // truncated
        const float* vec_ptr =
            reinterpret_cast<const float*>(raw.data() + field_pos);
        field_pos += vec_bytes;

        // meta_len
        if (field_pos + 4 > file_size) break;
        uint32_t meta_len = 0;
        std::memcpy(&meta_len, raw.data() + field_pos, 4);
        field_pos += 4;

        // metadata string
        if (field_pos + meta_len > file_size) break;
        std::string metadata_json(
            reinterpret_cast<const char*>(raw.data() + field_pos),
            meta_len);
        field_pos += meta_len;

        // crc32
        if (field_pos + 4 > file_size) break;
        uint32_t stored_crc = 0;
        std::memcpy(&stored_crc, raw.data() + field_pos, 4);
        field_pos += 4;

        // Validate CRC32 over bytes from entry_start to before crc field
        uLong crc = crc32(0L, Z_NULL, 0);
        crc = crc32(crc,
                    reinterpret_cast<const Bytef*>(raw.data() + entry_start),
                    static_cast<uInt>(field_pos - 4 - entry_start));
        uint32_t computed_crc = static_cast<uint32_t>(crc);

        if (computed_crc != stored_crc) {
            std::cerr << "WALReader: CRC mismatch at offset " << entry_start
                      << " (expected " << std::hex << stored_crc
                      << ", got " << computed_crc << std::dec
                      << ") — skipping entry\n";
            pos = field_pos; // advance past corrupt entry and continue
            continue;
        }

        WalEntry entry;
        entry.seq_id        = seq_id;
        entry.op_type       = op_type;
        entry.doc_id        = static_cast<DocId>(doc_id);
        entry.dim           = static_cast<int>(dim);
        entry.metadata_json = std::move(metadata_json);

        if (dim > 0) {
            entry.vector.assign(vec_ptr, vec_ptr + dim);
        }

        entries.push_back(std::move(entry));
        pos = field_pos;
    }

    return entries;
}
