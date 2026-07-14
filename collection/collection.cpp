#include "collection/collection.hpp"
#include "core/distance.hpp"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ============================================================
// Construction
// ============================================================

Collection::Collection(const std::string& name, const CollectionConfig& config)
    : name_(name)
    , config_(config)
{
    data_dir_ = "./" + name + "_novavec_data";
    fs::create_directories(data_dir_);

    metadata_store_  = std::make_unique<MetadataStore>();
    segment_manager_ = std::make_unique<SegmentManager>(data_dir_, config_);
    query_engine_    = std::make_unique<QueryEngine>(
        *segment_manager_, *metadata_store_);

    // Write config immediately so WAL replay can work even without an
    // explicit save() call (e.g. after a crash between inserts).
    save_config();
}

// ============================================================
// Path helpers
// ============================================================

std::string Collection::config_path() const {
    return data_dir_ + "/collection_config.json";
}

std::string Collection::metadata_path() const {
    return data_dir_ + "/metadata_store.json";
}

// ============================================================
// upsert
// ============================================================

void Collection::upsert(DocId              id,
                        const float*       vec,
                        int                dim,
                        const std::string& metadata_json)
{
    if (dim != config_.dim) {
        throw std::invalid_argument(
            "Collection::upsert: dim mismatch — expected " +
            std::to_string(config_.dim) + ", got " + std::to_string(dim));
    }

    // For COSINE metric, L2-normalize before insertion.
    // Normalization is on a local copy — the caller's array is unchanged.
    // This enables cheaper cosine computation (dot product of unit vectors)
    // at query time, avoiding sqrt + division per distance call.
    if (config_.metric == DistanceMetric::COSINE) {
        std::vector<float> norm_vec(vec, vec + dim);
        l2_normalize(norm_vec.data(), dim);
        segment_manager_->insert(id, norm_vec.data(), metadata_json);
    } else {
        segment_manager_->insert(id, vec, metadata_json);
    }

    // Index metadata separately (shared across all segments for selectivity estimation)
    metadata_store_->insert(id, metadata_json);
}

// ============================================================
// search
// ============================================================

std::vector<SearchResult> Collection::search(
    const float*                     query,
    int                              dim,
    int                              top_k,
    int                              ef_search,
    const std::optional<FilterSpec>& filter) const
{
    if (dim != config_.dim) {
        throw std::invalid_argument(
            "Collection::search: dim mismatch — expected " +
            std::to_string(config_.dim) + ", got " + std::to_string(dim));
    }

    // Normalize query for COSINE metric (same transform as at insertion)
    if (config_.metric == DistanceMetric::COSINE) {
        std::vector<float> norm_query(query, query + dim);
        l2_normalize(norm_query.data(), dim);
        return query_engine_->search(
            norm_query.data(), top_k, ef_search, filter);
    }

    return query_engine_->search(query, top_k, ef_search, filter);
}

// ============================================================
// remove
// ============================================================

void Collection::remove(DocId id) {
    // Remove from the collection-level metadata store (used by QueryPlanner
    // for selectivity estimation and QueryEngine for result enrichment).
    metadata_store_->remove(id);

    // Propagate delete to SegmentManager — writes a WAL DELETE entry
    // to the mutable segment and tombstones the vector in the index.
    // This ensures the delete survives a restart via WAL replay.
    segment_manager_->remove(id);
}

// ============================================================
// size
// ============================================================

int Collection::size() const {
    return segment_manager_->total_size();
}

// ============================================================
// save_config / load_config
// ============================================================

void Collection::save_config() const {
    json j;
    j["name"]                         = name_;
    j["dim"]                          = config_.dim;
    j["metric"]                       = static_cast<int>(config_.metric);
    j["index_type"]                   = static_cast<int>(config_.index_type);
    j["M"]                            = config_.M;
    j["ef_construction"]              = config_.ef_construction;
    j["nlist"]                        = config_.nlist;
    j["nprobe"]                       = config_.nprobe;
    j["fsync_policy"]                 = static_cast<int>(config_.fsync_policy);
    j["filter_selectivity_threshold"] = config_.filter_selectivity_threshold;
    j["segment_size_threshold"]       = config_.segment_size_threshold;

    std::ofstream f(config_path());
    if (!f) {
        throw std::runtime_error(
            "Collection::save_config: cannot open " + config_path());
    }
    f << j.dump(2);
}

CollectionConfig Collection::load_config(const std::string& data_dir) {
    std::ifstream f(data_dir + "/collection_config.json");
    if (!f) {
        throw std::runtime_error(
            "Collection::load_config: config not found in " + data_dir);
    }
    json j;
    f >> j;

    CollectionConfig cfg;
    cfg.dim                         = j.at("dim").get<int>();
    cfg.metric                      = static_cast<DistanceMetric>(j.at("metric").get<int>());
    cfg.index_type                  = static_cast<IndexType>(j.at("index_type").get<int>());
    cfg.M                           = j.value("M", 16);
    cfg.ef_construction             = j.value("ef_construction", 200);
    cfg.nlist                       = j.value("nlist", 256);
    cfg.nprobe                      = j.value("nprobe", 32);
    cfg.fsync_policy                = static_cast<FsyncPolicy>(j.value("fsync_policy", 1));
    cfg.filter_selectivity_threshold= j.value("filter_selectivity_threshold", 0.2f);
    cfg.segment_size_threshold      = j.value("segment_size_threshold", static_cast<size_t>(100000));
    return cfg;
}

// ============================================================
// save
// ============================================================

void Collection::save() const {
    fs::create_directories(data_dir_);
    save_config();
    metadata_store_->save(metadata_path());
    segment_manager_->save();
}

// ============================================================
// load — static factory
// ============================================================

Collection Collection::load(const std::string& data_dir) {
    Collection col;
    col.data_dir_ = data_dir;
    col.config_   = load_config(data_dir);

    // Reconstruct name from data_dir (strip suffix if present)
    fs::path p(data_dir);
    col.name_ = p.filename().string();

    col.metadata_store_  = std::make_unique<MetadataStore>();
    col.segment_manager_ = std::make_unique<SegmentManager>(
        data_dir, col.config_);
    col.query_engine_    = std::make_unique<QueryEngine>(
        *col.segment_manager_, *col.metadata_store_);

    // Reload persisted metadata
    std::string meta_path = data_dir + "/metadata_store.json";
    if (fs::exists(meta_path)) {
        col.metadata_store_->load(meta_path);
    }

    // Reload segments
    col.segment_manager_->load_sealed_segments();

    return col;
}
