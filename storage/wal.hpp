#pragma once

#include "core/types.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ============================================================
// Write-Ahead Log (WAL)
// ============================================================
// Binary append-only log providing crash recovery. Before any
// mutation reaches the index, it is durably written to the WAL.
// On restart, WALReader::replay() replays all valid entries.
//
// Entry binary format (little-endian, written in this order):
//
//   magic:    4 bytes  = {0x4E, 0x56, 0x57, 0x43}  ("NVWC")
//   seq_id:   8 bytes  uint64_t
//   op_type:  1 byte   (0x01=INSERT, 0x02=DELETE)
//   doc_id:   4 bytes  uint32_t
//   dim:      4 bytes  uint32_t
//   vector:   dim * sizeof(float) bytes
//   meta_len: 4 bytes  uint32_t
//   metadata: meta_len bytes  (UTF-8 JSON string)
//   crc32:    4 bytes  uint32_t  (CRC32 over ALL preceding bytes)
//
// CRC32 is computed with zlib's crc32() function over the
// concatenation of all preceding fields, protecting against
// torn writes and storage corruption.
//
// Fsync policies:
//   SYNC_ALWAYS   — fsync after every write. Maximum durability,
//                   ~1-2 ms per write (disk-limited).
//   SYNC_PERIODIC — background thread calls fsync every 200 ms.
//                   ~200 ms maximum data loss on crash.
//   SYNC_NEVER    — never fsync. OS page cache handles writes.
//                   Data loss on power failure. Maximum throughput.
// ============================================================

enum class WalOpType : uint8_t {
    INSERT = 0x01,
    DELETE = 0x02
};

struct WalEntry {
    uint64_t            seq_id;
    WalOpType           op_type;
    DocId               doc_id;
    int                 dim;
    std::vector<float>  vector;
    std::string         metadata_json;
};

// ---- Writer ------------------------------------------------

class WALWriter {
public:
    WALWriter(const std::string& path, FsyncPolicy policy);
    ~WALWriter();

    // Append one entry to the WAL.
    // Thread-safe — protected by write_mutex_.
    void append(WalOpType op,
                DocId id,
                const float* vec,
                int dim,
                const std::string& metadata_json);

    // Force fsync to disk (callable externally for checkpointing).
    void sync();

private:
    int                       fd_;
    uint64_t                  seq_counter_;
    FsyncPolicy               policy_;
    std::mutex                write_mutex_;

    // SYNC_PERIODIC background thread
    std::thread               sync_thread_;
    std::atomic<bool>         stop_sync_{false};
    std::condition_variable   sync_cv_;
    std::mutex                sync_cv_mutex_;

    void run_periodic_sync();
};

// ---- Reader ------------------------------------------------

class WALReader {
public:
    // Replay all valid entries from a WAL file.
    // Entries with invalid CRC32 or truncated data are logged
    // and skipped — partial writes at crash boundary are expected.
    std::vector<WalEntry> replay(const std::string& path);
};
