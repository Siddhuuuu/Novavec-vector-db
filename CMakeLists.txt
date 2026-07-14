cmake_minimum_required(VERSION 3.18)
project(novavec CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fix for CRoaring's old cmake_policy on newer CMake versions
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Performance flags — AVX2 SIMD, fast math
# MSVC uses different flags than GCC/Clang
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast /W3)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-O3 -mavx2 -march=native -ffast-math -Wall -Wextra)
endif()

include(FetchContent)

# CRoaring — compressed roaring bitmaps for metadata inverted index
FetchContent_Declare(roaring
    GIT_REPOSITORY https://github.com/RoaringBitmap/CRoaring.git
    GIT_TAG v2.0.4)
FetchContent_MakeAvailable(roaring)
# CRoaring provides C headers under include/ and C++ headers under cpp/.
# Expose the cpp/ directory so we can #include <roaring/roaring.hh>.
set_target_properties(roaring PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    "$<BUILD_INTERFACE:${roaring_SOURCE_DIR}/include>;$<BUILD_INTERFACE:${roaring_SOURCE_DIR}/cpp>")

# nlohmann/json — header-only JSON parsing for metadata
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3)
FetchContent_MakeAvailable(json)

# pybind11 — Python bindings with zero-copy numpy interop
FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1)
FetchContent_MakeAvailable(pybind11)

find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

# HDF5 + HighFive are only needed for the benchmark runner — make them optional
find_package(HDF5 QUIET)
if(HDF5_FOUND)
    FetchContent_Declare(highfive
        GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
        GIT_TAG v2.9.0)
    FetchContent_MakeAvailable(highfive)
    set(BUILD_BENCH ON)
    message(STATUS "HDF5 found — benchmark runner will be built")
else()
    set(BUILD_BENCH OFF)
    message(STATUS "HDF5 not found — benchmark runner skipped (install libhdf5-dev to enable)")
endif()

# Collect all C++ source files for the core engine
set(NOVAVEC_SOURCES
    core/distance.cpp
    indexes/flat_index.cpp
    indexes/hnsw/hnsw_index.cpp
    indexes/ivf/kmeans.cpp
    indexes/ivf/ivf_index.cpp
    indexes/ivf_hnsw/ivf_hnsw_index.cpp
    storage/wal.cpp
    storage/mmap_store.cpp
    storage/metadata_store.cpp
    storage/segment.cpp
    storage/segment_manager.cpp
    query/query_planner.cpp
    query/query_engine.cpp
    collection/collection.cpp)

# Static library for the core engine
add_library(novavec_core STATIC ${NOVAVEC_SOURCES})
# -fPIC is required because novavec_core is linked into a shared object (pybind11 .so).
# Without it, thread_local variables with dynamic initializers produce
# R_X86_64_TPOFF32 relocations that are invalid in shared libraries.
set_target_properties(novavec_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(novavec_core PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(novavec_core PUBLIC
    roaring
    nlohmann_json::nlohmann_json
    ZLIB::ZLIB
    Threads::Threads)
if(HDF5_FOUND)
    target_link_libraries(novavec_core PUBLIC HighFive)
endif()

# pybind11 Python extension module
pybind11_add_module(novavec bindings/python_bindings.cpp)
target_link_libraries(novavec PRIVATE novavec_core)

# Benchmark executable — only built when HDF5 is available
if(BUILD_BENCH)
    add_executable(bench_runner benchmarks/bench_runner.cpp)
    target_link_libraries(bench_runner PRIVATE novavec_core ${HDF5_LIBRARIES})
    target_include_directories(bench_runner PRIVATE ${HDF5_INCLUDE_DIRS})
endif()