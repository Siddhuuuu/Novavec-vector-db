#include "indexes/ivf/kmeans.hpp"
#include "core/distance.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

// ============================================================
// k-means++ initialization
// ============================================================

std::vector<float> kmeans_plus_plus_init(
    const float* data,
    int n,
    int dim,
    int k,
    uint64_t seed)
{
    if (k <= 0 || n < k) {
        throw std::invalid_argument("kmeans++: k must be > 0 and <= n");
    }

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> uniform_idx(0, n - 1);

    std::vector<float> centroids(static_cast<size_t>(k) * dim);

    // Step 1: choose first centroid uniformly at random
    int first = uniform_idx(rng);
    std::memcpy(centroids.data(), data + static_cast<size_t>(first) * dim,
                dim * sizeof(float));

    // D²(x) for each point: min squared distance to any chosen centroid
    std::vector<float> d2(n, std::numeric_limits<float>::max());

    for (int c = 1; c < k; ++c) {
        // Update D²(x): only need to consider the newly added centroid
        const float* new_cent =
            centroids.data() + static_cast<size_t>(c - 1) * dim;
        for (int i = 0; i < n; ++i) {
            float dist = l2_squared_scalar(
                data + static_cast<size_t>(i) * dim, new_cent, dim);
            if (dist < d2[i]) d2[i] = dist;
        }

        // Sample next centroid with probability proportional to D²(x)
        // std::discrete_distribution handles the weighted sampling correctly
        std::discrete_distribution<int> weighted(d2.begin(), d2.end());
        int next = weighted(rng);
        std::memcpy(centroids.data() + static_cast<size_t>(c) * dim,
                    data + static_cast<size_t>(next) * dim,
                    dim * sizeof(float));
    }

    return centroids;
}

// ============================================================
// Lloyd's algorithm
// ============================================================

KMeansResult kmeans_train(
    const float* data,
    int n,
    int dim,
    int k,
    int max_iter,
    float tol,
    uint64_t seed)
{
    if (n < k) {
        throw std::invalid_argument(
            "kmeans_train: number of training points (" + std::to_string(n) +
            ") must be >= k (" + std::to_string(k) + ")");
    }

    std::mt19937_64 rng(seed);

    // Initialize centroids with k-means++
    std::vector<float> centroids = kmeans_plus_plus_init(data, n, dim, k, seed);
    std::vector<int>   assignments(n, 0);
    std::vector<int>   cluster_sizes(k, 0);

    KMeansResult result;
    result.iterations = 0;
    result.final_inertia = 0.0f;

    for (int iter = 0; iter < max_iter; ++iter) {
        // ---- Assign step: each vector -> nearest centroid ----
        int changes = 0;
        double inertia = 0.0;

        for (int i = 0; i < n; ++i) {
            const float* v = data + static_cast<size_t>(i) * dim;
            float best_dist = std::numeric_limits<float>::max();
            int   best_c    = 0;

            for (int c = 0; c < k; ++c) {
                // Use AVX2 L2 dispatch for speed
                float d = compute_distance(
                    v, centroids.data() + static_cast<size_t>(c) * dim,
                    dim, DistanceMetric::L2);
                if (d < best_dist) {
                    best_dist = d;
                    best_c    = c;
                }
            }

            if (assignments[i] != best_c) {
                assignments[i] = best_c;
                ++changes;
            }
            inertia += static_cast<double>(best_dist);
        }

        result.final_inertia = static_cast<float>(inertia);
        result.iterations    = iter + 1;

        // Convergence check: fraction of vectors that changed cluster
        if (static_cast<float>(changes) < tol * static_cast<float>(n)) {
            break;
        }

        // ---- Update step: recompute centroids as cluster means ----
        std::fill(centroids.begin(), centroids.end(), 0.0f);
        std::fill(cluster_sizes.begin(), cluster_sizes.end(), 0);

        for (int i = 0; i < n; ++i) {
            int c = assignments[i];
            ++cluster_sizes[c];
            float* cent = centroids.data() + static_cast<size_t>(c) * dim;
            const float* v = data + static_cast<size_t>(i) * dim;
            for (int d = 0; d < dim; ++d) {
                cent[d] += v[d];
            }
        }

        for (int c = 0; c < k; ++c) {
            if (cluster_sizes[c] >= KMEANS_MIN_CLUSTER_SIZE) {
                float inv = 1.0f / static_cast<float>(cluster_sizes[c]);
                float* cent = centroids.data() + static_cast<size_t>(c) * dim;
                for (int d = 0; d < dim; ++d) {
                    cent[d] *= inv;
                }
            } else {
                // Empty cluster: reinitialize to a random data point
                // This prevents degenerate solutions where some centroids
                // have no assigned vectors and never converge.
                std::uniform_int_distribution<int> rand_idx(0, n - 1);
                int pick = rand_idx(rng);
                std::memcpy(centroids.data() + static_cast<size_t>(c) * dim,
                            data + static_cast<size_t>(pick) * dim,
                            dim * sizeof(float));
            }
        }
    }

    result.centroids   = std::move(centroids);
    result.assignments = std::move(assignments);
    return result;
}

// ============================================================
// Centroid assignment helpers
// ============================================================

int assign_to_nearest_centroid(
    const float* query,
    const float* centroids,
    int k,
    int dim,
    DistanceMetric metric)
{
    float best_dist = std::numeric_limits<float>::max();
    int   best_c    = 0;
    for (int c = 0; c < k; ++c) {
        float d = compute_distance(
            query,
            centroids + static_cast<size_t>(c) * dim,
            dim, metric);
        if (d < best_dist) {
            best_dist = d;
            best_c    = c;
        }
    }
    return best_c;
}

std::vector<int> find_nearest_centroids(
    const float* query,
    const float* centroids,
    int k,
    int dim,
    int nprobe,
    DistanceMetric metric)
{
    // Collect (distance, index) pairs
    std::vector<std::pair<float, int>> dists(k);
    for (int c = 0; c < k; ++c) {
        dists[c] = {
            compute_distance(
                query,
                centroids + static_cast<size_t>(c) * dim,
                dim, metric),
            c
        };
    }

    // Partial sort: bring nprobe smallest to the front
    const int np = std::min(nprobe, k);
    std::partial_sort(dists.begin(), dists.begin() + np, dists.end());

    std::vector<int> result(np);
    for (int i = 0; i < np; ++i) {
        result[i] = dists[i].second;
    }
    return result;
}
