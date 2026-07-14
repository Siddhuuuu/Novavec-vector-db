#include "storage/metadata_store.hpp"
#include <mutex>
#include <shared_mutex>

#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

// ============================================================
// insert
// ============================================================

void MetadataStore::insert(DocId id, const std::string& metadata_json) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Remove old entry if this DocId is being updated (upsert semantics)
    if (doc_metadata_.count(id)) {
        lock.unlock();
        remove(id);
        lock.lock();
    }

    // Parse JSON and index each string/number field
    try {
        auto j = json::parse(metadata_json);
        if (j.is_object()) {
            for (auto& [key, val] : j.items()) {
                std::string val_str;
                if (val.is_string()) {
                    val_str = val.get<std::string>();
                } else if (val.is_number()) {
                    // Represent numbers as their string form for indexing
                    val_str = val.dump();
                } else {
                    // Skip arrays, objects, booleans, null
                    continue;
                }
                index_[key][val_str].add(static_cast<uint32_t>(id));
            }
        }
    } catch (const json::parse_error&) {
        // Non-JSON metadata: store raw string, no field indexing
    }

    doc_metadata_[id] = metadata_json;
    ++total_docs_;
}

// ============================================================
// remove
// ============================================================

void MetadataStore::remove(DocId id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    if (!doc_metadata_.count(id)) return;

    // Remove from inverted index
    try {
        auto j = json::parse(doc_metadata_[id]);
        if (j.is_object()) {
            for (auto& [key, val] : j.items()) {
                std::string val_str;
                if (val.is_string())      val_str = val.get<std::string>();
                else if (val.is_number()) val_str = val.dump();
                else continue;

                auto fit = index_.find(key);
                if (fit != index_.end()) {
                    auto vit = fit->second.find(val_str);
                    if (vit != fit->second.end()) {
                        vit->second.remove(static_cast<uint32_t>(id));
                        if (vit->second.isEmpty()) {
                            fit->second.erase(vit);
                        }
                    }
                }
            }
        }
    } catch (...) {}

    doc_metadata_.erase(id);
    --total_docs_;
}

// ============================================================
// get_bitmap
// ============================================================

roaring::Roaring MetadataStore::get_bitmap(
    const std::string& field,
    const std::string& value) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto fit = index_.find(field);
    if (fit == index_.end()) return roaring::Roaring{};

    auto vit = fit->second.find(value);
    if (vit == fit->second.end()) return roaring::Roaring{};

    return vit->second; // copy
}

// ============================================================
// intersect_filters
// ============================================================

roaring::Roaring MetadataStore::intersect_filters(
    const std::vector<std::pair<std::string, std::string>>& filters) const
{
    if (filters.empty()) return roaring::Roaring{};

    roaring::Roaring result = get_bitmap(filters[0].first, filters[0].second);

    for (size_t i = 1; i < filters.size(); ++i) {
        roaring::Roaring next =
            get_bitmap(filters[i].first, filters[i].second);
        result &= next; // in-place AND using CRoaring SIMD
    }

    return result;
}

// ============================================================
// selectivity
// ============================================================

float MetadataStore::selectivity(
    const std::string& field,
    const std::string& value) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);

    if (total_docs_ == 0) return 1.0f;

    auto fit = index_.find(field);
    if (fit == index_.end()) return 1.0f;

    auto vit = fit->second.find(value);
    if (vit == fit->second.end()) return 1.0f;

    return static_cast<float>(vit->second.cardinality()) /
           static_cast<float>(total_docs_);
}

// ============================================================
// get_metadata
// ============================================================

std::string MetadataStore::get_metadata(DocId id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = doc_metadata_.find(id);
    if (it == doc_metadata_.end()) return "{}";
    return it->second;
}

// ============================================================
// save / load — JSON serialization
// ============================================================

void MetadataStore::save(const std::string& path) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    json out;
    out["total_docs"] = total_docs_;

    // Serialize doc_metadata_
    json doc_meta_j = json::object();
    for (const auto& [id, meta] : doc_metadata_) {
        doc_meta_j[std::to_string(id)] = meta;
    }
    out["doc_metadata"] = doc_meta_j;

    // Serialize inverted index as field -> value -> [docid, ...]
    json index_j = json::object();
    for (const auto& [field, vals] : index_) {
        json field_j = json::object();
        for (const auto& [val, bitmap] : vals) {
            // Serialize bitmap as list of uint32_t
            std::vector<uint32_t> ids;
            ids.reserve(bitmap.cardinality());
            for (uint32_t x : bitmap) ids.push_back(x);
            field_j[val] = ids;
        }
        index_j[field] = field_j;
    }
    out["index"] = index_j;

    std::ofstream f(path);
    if (!f) throw std::runtime_error("MetadataStore::save: cannot open " + path);
    f << out.dump(2);
}

void MetadataStore::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) return; // file doesn't exist — fresh start

    json in;
    try {
        f >> in;
    } catch (const json::parse_error& e) {
        throw std::runtime_error(
            "MetadataStore::load: JSON parse error: " + std::string(e.what()));
    }

    std::unique_lock<std::shared_mutex> lock(mutex_);

    total_docs_ = in.value("total_docs", 0);

    // Restore doc_metadata_
    doc_metadata_.clear();
    if (in.contains("doc_metadata")) {
        for (auto& [id_str, meta] : in["doc_metadata"].items()) {
            DocId id = static_cast<DocId>(std::stoul(id_str));
            doc_metadata_[id] = meta.get<std::string>();
        }
    }

    // Restore inverted index
    index_.clear();
    if (in.contains("index")) {
        for (auto& [field, vals_j] : in["index"].items()) {
            for (auto& [val, ids_j] : vals_j.items()) {
                roaring::Roaring bm;
                for (uint32_t id : ids_j.get<std::vector<uint32_t>>()) {
                    bm.add(id);
                }
                index_[field][val] = std::move(bm);
            }
        }
    }
}
