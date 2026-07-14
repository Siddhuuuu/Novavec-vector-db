/**
 * bench_runner.cpp — NovaVec index benchmark tool
 *
 * Usage:
 *   ./bench_runner \
 *     --dataset  sift-128           \
 *     --index    hnsw                \
 *     --M        16                  \
 *     --ef-construction 200          \
 *     --ef-search       100          \
 *     --nlist    256                 \
 *     --nprobe   32                  \
 *     --queries  1000                \
 *     --output   /tmp/result.json
 *
 * HDF5 dataset format (ann-benchmarks convention):
 *   /train       float32[N, D]
 *   /test        float32[Q, D]
 *   /neighbors   int32[Q, K]      (ground-truth nearest neighbor IDs)
 *   /distances   float32[Q, K]    (optional)
 *
 * Download datasets from: http://ann-benchmarks.com/
 *   wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
 *   wget http://ann-benchmarks.com/glove-100-angular.hdf5
 */

#include <highfive/H5Easy.hpp>
#include <nlohmann/json.hpp>

#include "indexes/base_index.hpp"
#include "indexes/flat_index.hpp"
#include "indexes/hnsw/hnsw_index.hpp"
#include "indexes/ivf/ivf_index.hpp"
#include "indexes/ivf_hnsw/ivf_hnsw_index.hpp"
#include "core/distance.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

using json = nlohmann::json;

// ============================================================
// CLI argument parsing — hand-rolled, no external library
// ============================================================

struct BenchConfig {
    std::string dataset       = "sift-128";
    std::string index_type    = "hnsw";
    std::string hdf5_path;          // derived from dataset if not set
    std::string output_path   = "/tmp/bench_result.json";
    int         M             = 16;
    int         ef_construction = 200;
    int         ef_search     = 100;
    int         nlist         = 256;
    int         nprobe        = 32;
    int         num_queries   = 1000;
    int         top_k         = 10;
};

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  --dataset         sift-128 | glove-100 | <custom.hdf5>\n"
              << "  --index           flat | hnsw | ivf | ivf_hnsw\n"
              << "  --M               HNSW M (default 16)\n"
              << "  --ef-construction HNSW ef_construction (default 200)\n"
              << "  --ef-search       HNSW ef_search (default 100)\n"
              << "  --nlist           IVF nlist (default 256)\n"
              << "  --nprobe          IVF nprobe (default 32)\n"
              << "  --queries         Number of query vectors (default 1000)\n"
              << "  --output          Output JSON path (default /tmp/bench_result.json)\n";
}

static BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;
    std::map<std::string, std::string> args_map;

    for (int i = 1; i + 1 < argc; i += 2) {
        std::string key = argv[i];
        std::string val = argv[i + 1];
        // Strip leading "--"
        if (key.size() > 2 && key[0] == '-' && key[1] == '-') {
            key = key.substr(2);
        }
        args_map[key] = val;
    }

    if (args_map.count("dataset"))         cfg.dataset          = args_map["dataset"];
    if (args_map.count("index"))           cfg.index_type       = args_map["index"];
    if (args_map.count("M"))              cfg.M               = std::stoi(args_map["M"]);
    if (args_map.count("ef-construction"))cfg.ef_construction = std::stoi(args_map["ef-construction"]);
    if (args_map.count("ef-search"))      cfg.ef_search       = std::stoi(args_map["ef-search"]);
    if (args_map.count("nlist"))          cfg.nlist           = std::stoi(args_map["nlist"]);
    if (args_map.count("nprobe"))         cfg.nprobe          = std::stoi(args_map["nprobe"]);
    if (args_map.count("queries"))        cfg.num_queries     = std::stoi(args_map["queries"]);
    if (args_map.count("output"))         cfg.output_path     = args_map["output"];

    // Derive HDF5 path from dataset name if not explicitly given
    if (cfg.hdf5_path.empty()) {
        // Map short name -> filename
        static const std::map<std::string,std::string> KNOWN = {
            {"sift-128",  "sift-128-euclidean.hdf5"},
            {"glove-100", "glove-100-angular.hdf5"},
        };
        auto it = KNOWN.find(cfg.dataset);
        if (it != KNOWN.end()) {
            cfg.hdf5_path = it->second;
        } else {
            cfg.hdf5_path = cfg.dataset; // treat as direct path
        }
    }

    return cfg;
}

// ============================================================
// Timing utilities
// ============================================================

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double elapsed_ms(Clock::time_point start) {
    return Ms(Clock::now() - start).count();
}

static void l2_normalize_vec(std::vector<float>& v) {
    float norm_sq = 0.0f;
    for (float x : v) norm_sq += x * x;
    if (norm_sq < 1e-10f) return;

    const float inv_norm = 1.0f / std::sqrt(norm_sq);
    for (float& x : v) x *= inv_norm;
}



// ============================================================
// Recall@K computation
// ============================================================

static float compute_recall_at_k(
    const std::vector<std::vector<int>>&           ground_truth,
    const std::vector<std::vector<SearchResult>>&  results,
    int                                             k)
{
    if (ground_truth.empty()) return 0.0f;

    int hits  = 0;
    int total = 0;

    for (size_t q = 0; q < std::min(ground_truth.size(), results.size()); ++q) {
        const auto& gt = ground_truth[q];
        const auto& res = results[q];

        // Build set of ground-truth IDs (first k)
        const int gt_k = std::min(k, static_cast<int>(gt.size()));
        std::vector<int> gt_ids(gt.begin(), gt.begin() + gt_k);
        std::sort(gt_ids.begin(), gt_ids.end());

        // Count how many result IDs are in ground truth
        const int res_k = std::min(k, static_cast<int>(res.size()));
        for (int i = 0; i < res_k; ++i) {
            if (std::binary_search(gt_ids.begin(), gt_ids.end(),
                                   static_cast<int>(res[i].id))) {
                ++hits;
            }
        }
        total += gt_k;
    }

    return total > 0 ? static_cast<float>(hits) / static_cast<float>(total) : 0.0f;
}

// ============================================================
// Percentile from sorted vector
// ============================================================

static double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t idx = static_cast<size_t>(p / 100.0 * (v.size() - 1));
    return v[std::min(idx, v.size() - 1)];
}

// ============================================================
// Build index
// ============================================================


static std::unique_ptr<BaseIndex> build_index(
    const BenchConfig& cfg,
    const std::vector<std::vector<float>>& train,
    int dim)
{
    std::cout << "Building " << cfg.index_type << " index on "
              << train.size() << " vectors (dim=" << dim << ")…\n";

           DistanceMetric metric =
        (cfg.hdf5_path.find("angular") != std::string::npos)
            ? DistanceMetric::COSINE
            : DistanceMetric::L2;



    // Flatten training data into a contiguous array
    const int n = static_cast<int>(train.size());
    std::vector<float> flat(static_cast<size_t>(n) * dim);
    for (int i = 0; i < n; ++i) {
        std::copy(train[i].begin(), train[i].end(),
                  flat.data() + static_cast<size_t>(i) * dim);
    }

        if (metric == DistanceMetric::COSINE) {
        for (int i = 0; i < n; ++i) {
            l2_normalize(flat.data() + static_cast<size_t>(i) * dim, dim);
        }
    }

    auto t0 = Clock::now();

    std::unique_ptr<BaseIndex> index;

    if (cfg.index_type == "flat") {
        index = std::make_unique<FlatIndex>(dim, metric);
        for (int i = 0; i < n; ++i) {
            index->insert(static_cast<DocId>(i),
                          flat.data() + static_cast<size_t>(i) * dim);
        }

    } else if (cfg.index_type == "hnsw") {
        index = std::make_unique<HNSWIndex>(
    dim, cfg.M, cfg.ef_construction, metric);
        for (int i = 0; i < n; ++i) {
            index->insert(static_cast<DocId>(i),
                          flat.data() + static_cast<size_t>(i) * dim);
        }

    } else if (cfg.index_type == "ivf") {
        auto ivf = std::make_unique<IVFIndex>(
    dim, cfg.nlist, cfg.nprobe, metric);
        ivf->train(flat.data(), n);
        for (int i = 0; i < n; ++i) {
            ivf->insert(static_cast<DocId>(i),
                        flat.data() + static_cast<size_t>(i) * dim);
        }
        index = std::move(ivf);

    } else if (cfg.index_type == "ivf_hnsw") {
        CollectionConfig colcfg;
        colcfg.dim             = dim;
        colcfg.metric = metric;
        colcfg.M               = cfg.M;
        colcfg.ef_construction = cfg.ef_construction;
        colcfg.nlist           = cfg.nlist;
        colcfg.nprobe          = cfg.nprobe;

        auto ih = std::make_unique<IVFHNSWIndex>(colcfg);
        ih->train(flat.data(), n);
        for (int i = 0; i < n; ++i) {
            ih->insert(static_cast<DocId>(i),
                       flat.data() + static_cast<size_t>(i) * dim);
        }
        index = std::move(ih);
    } else {
        throw std::invalid_argument("Unknown index type: " + cfg.index_type);
    }

    double build_ms = elapsed_ms(t0);
    std::cout << "Index built in " << std::fixed << std::setprecision(1)
              << build_ms << " ms.\n";

    return index;
}

// ============================================================
// Run queries and collect latencies
// ============================================================

static void run_benchmark(
    BaseIndex& index,
    const std::vector<std::vector<float>>& test,
    const std::vector<std::vector<int>>&   ground_truth,
    DistanceMetric metric,
    const BenchConfig& cfg,
    json& result_json)
{
    const int n_queries = std::min(
        cfg.num_queries, static_cast<int>(test.size()));
    const int dim       = index.dim();

    std::cout << "Running " << n_queries << " queries (ef_search="
              << cfg.ef_search << ", top_k=" << cfg.top_k << ")…\n";

    std::vector<std::vector<SearchResult>> all_results(n_queries);
    std::vector<double> latencies_ms(n_queries);

        for (int q = 0; q < n_queries; ++q) {
        std::vector<float> query = test[q];
        if (metric == DistanceMetric::COSINE) {
            l2_normalize_vec(query);
        }

        auto t0 = Clock::now();
        all_results[q] = index.search(query.data(), cfg.top_k, cfg.ef_search);
        latencies_ms[q] = elapsed_ms(t0);
    }

    // ---- Recall ----
    float recall = compute_recall_at_k(ground_truth, all_results, cfg.top_k);

    // ---- Latency percentiles ----
    // Use nth_element for O(n) p50, then sort for p95/p99
    const double p50 = percentile(latencies_ms, 50.0);
    const double p95 = percentile(latencies_ms, 95.0);
    const double p99 = percentile(latencies_ms, 99.0);

    double total_ms = std::accumulate(
        latencies_ms.begin(), latencies_ms.end(), 0.0);
    double mean_ms  = total_ms / n_queries;
    double qps      = (n_queries / total_ms) * 1000.0;

    // ---- Print table ----
    std::cout << "\n========== Benchmark Results ==========\n"
              << "Index:      " << cfg.index_type << "\n"
              << "Dataset:    " << cfg.dataset << "\n"
              << "Vectors:    " << index.size() << "\n"
              << "Queries:    " << n_queries << "\n"
              << "ef_search:  " << cfg.ef_search << "\n"
              << "top_k:      " << cfg.top_k << "\n"
              << "---------------------------------------\n"
              << "Recall@" << cfg.top_k << ":   "
              << std::fixed << std::setprecision(4) << recall << "\n"
              << "QPS:        "
              << std::fixed << std::setprecision(1) << qps << "\n"
              << "p50 (ms):   "
              << std::fixed << std::setprecision(3) << p50 << "\n"
              << "p95 (ms):   "
              << std::fixed << std::setprecision(3) << p95 << "\n"
              << "p99 (ms):   "
              << std::fixed << std::setprecision(3) << p99 << "\n"
              << "mean (ms):  "
              << std::fixed << std::setprecision(3) << mean_ms << "\n"
              << "=======================================\n";

    // ---- JSON output ----
    result_json["dataset"]      = cfg.dataset;
    result_json["index_type"]   = cfg.index_type;
    result_json["n_vectors"]    = index.size();
    result_json["n_queries"]    = n_queries;
    result_json["top_k"]        = cfg.top_k;
    result_json["ef_search"]    = cfg.ef_search;
    result_json["M"]            = cfg.M;
    result_json["nlist"]        = cfg.nlist;
    result_json["nprobe"]       = cfg.nprobe;
    result_json["recall_at_10"] = static_cast<double>(recall);
    result_json["qps"]          = qps;
    result_json["p50_ms"]       = p50;
    result_json["p95_ms"]       = p95;
    result_json["p99_ms"]       = p99;
    result_json["mean_ms"]      = mean_ms;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    BenchConfig cfg = parse_args(argc, argv);

    // ---- Load HDF5 dataset ----
    if (!std::filesystem::exists(cfg.hdf5_path)) {
        std::cerr << "ERROR: HDF5 file not found: " << cfg.hdf5_path << "\n"
                  << "Download from http://ann-benchmarks.com/\n"
                  << "  wget http://ann-benchmarks.com/sift-128-euclidean.hdf5\n";
        return 1;
    }

    std::cout << "Loading HDF5 dataset: " << cfg.hdf5_path << "\n";
    HighFive::File f(cfg.hdf5_path, HighFive::File::ReadOnly);

    std::vector<std::vector<float>> train, test;
    std::vector<std::vector<int>>   neighbors;

    try {
        train     = f.getDataSet("train").read<std::vector<std::vector<float>>>();
        test      = f.getDataSet("test").read<std::vector<std::vector<float>>>();
        neighbors = f.getDataSet("neighbors").read<std::vector<std::vector<int>>>();
    } catch (const std::exception& e) {
        std::cerr << "ERROR reading HDF5: " << e.what() << "\n";
        return 1;
    }

    if (train.empty() || test.empty()) {
        std::cerr << "ERROR: empty dataset\n";
        return 1;
    }

    const int dim = static_cast<int>(train[0].size());
    std::cout << "Train: " << train.size()
              << " vectors, Test: " << test.size()
              << " queries, Dim: " << dim << "\n";

    // ---- Build index ----
    std::unique_ptr<BaseIndex> index;
    try {
        index = build_index(cfg, train, dim);
    } catch (const std::exception& e) {
        std::cerr << "ERROR building index: " << e.what() << "\n";
        return 1;
    }

    // ---- Run benchmark ----
    json result;
        DistanceMetric metric =
        (cfg.hdf5_path.find("angular") != std::string::npos)
            ? DistanceMetric::COSINE
            : DistanceMetric::L2;

    run_benchmark(*index, test, neighbors, metric, cfg, result);


    // ---- Write JSON output ----
    std::filesystem::create_directories(
        std::filesystem::path(cfg.output_path).parent_path());
    std::ofstream out(cfg.output_path);
    if (!out) {
        std::cerr << "WARNING: cannot write output to " << cfg.output_path << "\n";
    } else {
        out << result.dump(2);
        std::cout << "Results written to: " << cfg.output_path << "\n";
    }

    return 0;
}
