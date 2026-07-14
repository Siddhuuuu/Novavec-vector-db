#pragma once

#include "core/types.hpp"
#include <vector>
#include <cstdint>

// ============================================================
// K-Means clustering with k-means++ initialization
// ============================================================
// Used by IVFIndex and IVFHNSWIndex to partition the vector
// space into nlist Voronoi cells (inverted lists).
//
// k-means++ initialization:
//   Instead of random centroid selection (which can lead to poor
//   clustering and slow convergence), k-means++ selects initial
//   centroids with probability proportional to D²(x) — the
//   squared distance from x to the nearest already-chosen centroid.
//   This spreads centroids across the space, reducing expected
//   iterations by O(log k) compared to random initialization.
//
// Lloyd's algorithm:
//   1. Assign each vector to its nearest centroid (AVX2 distance)
//   2. Update centroids as the mean of assigned vectors
//   3. Repeat until convergence (change_fraction < tol) or max_iter
//   Empty cluster handling: reinitialize to a random data point.
// ============================================================

struct KMeansResult {
    std::vector<float> centroids;    // flat: centroid i at offset i*dim
    std::vector<int>   assignments;  // assignments[j] = centroid index for vector j
    int                iterations;
    float              final_inertia; // sum of squared distances to assigned centroids
};

// k-means++ initialization.
// Returns flat centroid array (k * dim floats).
// seed: random seed for reproducibility.
std::vector<float> kmeans_plus_plus_init(
    const float* data,
    int n,           // number of training vectors
    int dim,
    int k,           // number of clusters
    uint64_t seed = KMEANS_DEFAULT_SEED);

// Full k-means training.
// data: n x dim column-major float array (vector i at data + i*dim).
KMeansResult kmeans_train(
    const float* data,
    int n,
    int dim,
    int k,
    int max_iter   = KMEANS_DEFAULT_MAX_ITER,
    float tol      = KMEANS_DEFAULT_TOL,
    uint64_t seed  = KMEANS_DEFAULT_SEED);

// Assign a single query vector to its nearest centroid.
// Returns centroid index in [0, k).
int assign_to_nearest_centroid(
    const float* query,
    const float* centroids,
    int k,
    int dim,
    DistanceMetric metric = DistanceMetric::L2);

// Return indices of the nprobe nearest centroids to query, sorted ascending.
std::vector<int> find_nearest_centroids(
    const float* query,
    const float* centroids,
    int k,
    int dim,
    int nprobe,
    DistanceMetric metric = DistanceMetric::L2);
