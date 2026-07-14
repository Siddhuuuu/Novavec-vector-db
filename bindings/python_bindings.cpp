#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <optional>
#include "collection/collection.hpp"
#include "query/query_planner.hpp"

namespace py = pybind11;

// ============================================================
// NovaVec Python module
// ============================================================
// Zero-copy numpy interop: py::array_t<float, py::array::c_style>
// gives us a raw float* pointer into the numpy buffer without copying.
// This is critical for performance — a 1536-dim OpenAI embedding is
// 6 KB; copying it on every call would add measurable overhead.
// ============================================================

PYBIND11_MODULE(novavec, m) {
    m.doc() =
        "NovaVec — vector database from first principles. "
        "C++ engine exposed via pybind11 with zero-copy numpy interop.";

    // ---- FilterSpec ---------------------------------------------
    py::class_<FilterSpec>(m, "FilterSpec",
        "A single field=value metadata filter.")
        .def(py::init<std::string, std::string>(),
             py::arg("field"), py::arg("value"))
        .def_readwrite("field", &FilterSpec::field)
        .def_readwrite("value", &FilterSpec::value)
        .def("__repr__", [](const FilterSpec& f) {
            return "FilterSpec(field='" + f.field +
                   "', value='" + f.value + "')";
        });

    // ---- SearchResult -------------------------------------------
    py::class_<SearchResult>(m, "SearchResult",
        "One result from a nearest-neighbor search.")
        .def_readonly("id",            &SearchResult::id,
                      "External DocId of the matched vector.")
        .def_readonly("score",         &SearchResult::score,
                      "Distance score (lower = more similar for L2/cosine).")
        .def_readonly("metadata_json", &SearchResult::metadata_json,
                      "Raw metadata JSON string.")
        .def("metadata", [](const SearchResult& r) {
            // Parse metadata_json and return as a Python dict.
            // This avoids requiring the user to call json.loads() manually.
            return py::module_::import("json")
                       .attr("loads")(r.metadata_json);
        }, "Parse and return metadata as a Python dict.")
        .def("__repr__", [](const SearchResult& r) {
            return "SearchResult(id=" + std::to_string(r.id) +
                   ", score=" + std::to_string(r.score) + ")";
        });

    // ---- Collection ---------------------------------------------
    py::class_<Collection>(m, "Collection",
        "A named collection of vectors with HNSW/IVF indexing and "
        "metadata filtering. Wraps the C++ engine with zero-copy "
        "numpy buffer access.")
        .def(py::init([](
                const std::string& name,
                int                dim,
                const std::string& metric,
                const std::string& index_type,
                int                M,
                int                ef_construction,
                int                nlist,
                int                nprobe) {

            CollectionConfig cfg;
            cfg.dim            = dim;
            cfg.M              = M;
            cfg.ef_construction = ef_construction;
            cfg.nlist          = nlist;
            cfg.nprobe         = nprobe;

            if      (metric == "cosine")        cfg.metric = DistanceMetric::COSINE;
            else if (metric == "l2")            cfg.metric = DistanceMetric::L2;
            else if (metric == "inner_product") cfg.metric = DistanceMetric::INNER_PRODUCT;
            else throw std::invalid_argument(
                "metric must be 'cosine', 'l2', or 'inner_product'");

            if      (index_type == "hnsw")     cfg.index_type = IndexType::HNSW;
            else if (index_type == "ivf")      cfg.index_type = IndexType::IVF;
            else if (index_type == "ivf_hnsw") cfg.index_type = IndexType::IVF_HNSW;
            else if (index_type == "flat")     cfg.index_type = IndexType::FLAT;
            else throw std::invalid_argument(
                "index_type must be 'hnsw', 'ivf', 'ivf_hnsw', or 'flat'");

            return Collection(name, cfg);
        }),
        py::arg("name"),
        py::arg("dim"),
        py::arg("metric")           = "cosine",
        py::arg("index_type")       = "hnsw",
        py::arg("M")                = 16,
        py::arg("ef_construction")  = 200,
        py::arg("nlist")            = 256,
        py::arg("nprobe")           = 32,
        "Create a new Collection.\n\n"
        "Parameters\n----------\n"
        "name           : Collection identifier.\n"
        "dim            : Vector dimensionality.\n"
        "metric         : 'cosine', 'l2', or 'inner_product'.\n"
        "index_type     : 'hnsw', 'ivf', 'ivf_hnsw', or 'flat'.\n"
        "M              : HNSW max connections per layer.\n"
        "ef_construction: HNSW beam width during construction.\n"
        "nlist          : IVF/IVF-HNSW number of Voronoi cells.\n"
        "nprobe         : IVF/IVF-HNSW cells probed at query time.")

        // ---- upsert -------------------------------------------
        .def("upsert",
            [](Collection& self,
               uint32_t     id,
               py::array_t<float, py::array::c_style> vec,
               py::object   metadata) {

                // Zero-copy: request raw buffer pointer from numpy array.
                // No data is copied — we read directly from the numpy buffer.
                py::buffer_info buf = vec.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error(
                        "upsert: vector must be a 1-D numpy array");
                }
                float* ptr = static_cast<float*>(buf.ptr);
                int    dim  = static_cast<int>(buf.shape[0]);

                std::string meta_json = "{}";
                if (!metadata.is_none()) {
                    meta_json = py::str(
                        py::module_::import("json").attr("dumps")(metadata))
                        .cast<std::string>();
                }
                self.upsert(id, ptr, dim, meta_json);
            },
            py::arg("id"),
            py::arg("vector"),
            py::arg("metadata") = py::none(),
            "Insert or update a vector.\n\n"
            "Parameters\n----------\n"
            "id       : Unique integer document ID.\n"
            "vector   : 1-D numpy float32 array of length dim.\n"
            "metadata : Optional dict of scalar field values for filtering.")

        // ---- search -------------------------------------------
        .def("search",
            [](Collection& self,
               py::array_t<float, py::array::c_style> query,
               int        top_k,
               int        ef_search,
               py::object filter_dict) -> std::vector<SearchResult> {

                py::buffer_info buf = query.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error(
                        "search: query must be a 1-D numpy array");
                }
                float* ptr = static_cast<float*>(buf.ptr);
                int    dim  = static_cast<int>(buf.shape[0]);

                std::optional<FilterSpec> filter = std::nullopt;
                if (!filter_dict.is_none()) {
                    auto d = filter_dict.cast<py::dict>();
                    if (d.size() != 1) {
                        throw std::runtime_error(
                            "filter dict must have exactly one key-value pair, "
                            "e.g. {'category': 'science'}");
                    }
                    auto it = d.begin();
                    filter = FilterSpec{
                        it->first.cast<std::string>(),
                        py::str(it->second).cast<std::string>()
                    };
                }
                return self.search(ptr, dim, top_k, ef_search, filter);
            },
            py::arg("query"),
            py::arg("top_k")     = 10,
            py::arg("ef_search") = 100,
            py::arg("filter")    = py::none(),
            "Search for nearest neighbors.\n\n"
            "Parameters\n----------\n"
            "query    : 1-D numpy float32 query vector.\n"
            "top_k    : Number of results to return.\n"
            "ef_search: HNSW beam width (larger = higher recall).\n"
            "filter   : Optional dict like {'category': 'science'}.\n\n"
            "Returns\n-------\n"
            "List of SearchResult objects sorted by ascending distance.")

        // ---- remove -------------------------------------------
        .def("remove", &Collection::remove,
             py::arg("id"),
             "Soft-delete a vector by DocId.")

        // ---- save ---------------------------------------------
        .def("save", &Collection::save,
             "Persist the collection to disk (data_dir_).")

        // ---- size / len ---------------------------------------
        .def("size", &Collection::size,
             "Return the total number of vectors in the collection.")
        .def("__len__", &Collection::size)

        // ---- config accessor ----------------------------------
        .def_property_readonly("config",
            [](const Collection& c) {
                const auto& cfg = c.config();
                return py::dict(
                    py::arg("dim")              = cfg.dim,
                    py::arg("M")                = cfg.M,
                    py::arg("ef_construction")  = cfg.ef_construction,
                    py::arg("nlist")            = cfg.nlist,
                    py::arg("nprobe")           = cfg.nprobe
                );
            }, "Collection configuration as a dict.")

        // ---- repr ---------------------------------------------
        .def("__repr__", [](const Collection& c) {
            return "novavec.Collection(name='" + c.name() +
                   "', size="  + std::to_string(c.size()) +
                   ", dim="    + std::to_string(c.config().dim) + ")";
        });

    // ---- Module-level load_collection --------------------------
    m.def("load_collection",
          &Collection::load,
          py::arg("data_dir"),
          "Load a previously saved collection from disk.\n\n"
          "Parameters\n----------\n"
          "data_dir : Path to the directory written by Collection.save().");
}
