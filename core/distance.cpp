#include "core/distance.hpp"

#include <cmath>
#include <stdexcept>

#ifdef __AVX2__
#include <immintrin.h>
#endif

// ============================================================
// Scalar implementations
// ============================================================

float l2_squared_scalar(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float dot_product_scalar(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float cosine_scalar(const float* a, const float* b, int dim) {
    // General cosine: dot(a,b) / (|a| * |b|)
    // When vectors are pre-normalized at insertion (COSINE metric),
    // |a|=|b|=1 so this reduces to dot product. This function handles
    // the general case for correctness verification purposes.
    float dot  = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-10f) return 0.0f;
    // Return 1 - similarity so smaller = more similar (consistent with L2)
    return 1.0f - (dot / denom);
}

// ============================================================
// AVX2 implementations
// ============================================================
#ifdef __AVX2__

// Horizontal sum of a __m256 register.
// Correct shuffle sequence per Intel intrinsics guide:
//   1. Split 256-bit register into two 128-bit halves
//   2. Add halves together (4 x float)
//   3. Two rounds of hadd_ps to reduce to scalar
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 r  = _mm_add_ps(lo, hi);
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    return _mm_cvtss_f32(r);
}

float l2_squared_avx2(const float* a, const float* b, int dim) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    // Process 8 floats per iteration
    for (; i + 8 <= dim; i += 8) {
        __m256 va   = _mm256_loadu_ps(a + i);
        __m256 vb   = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        // fmadd: sum += diff * diff
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    float result = hsum256_ps(sum);
    // Scalar fallback for remainder
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

float dot_product_avx2(const float* a, const float* b, int dim) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        // fmadd: sum += va * vb
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    float result = hsum256_ps(sum);
    // Scalar fallback for remainder
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

#endif // __AVX2__

// ============================================================
// Main dispatcher
// ============================================================

float compute_distance(const float* a, const float* b,
                       int dim, DistanceMetric metric) {
    switch (metric) {
    case DistanceMetric::L2:
#ifdef __AVX2__
        return l2_squared_avx2(a, b, dim);
#else
        return l2_squared_scalar(a, b, dim);
#endif

    case DistanceMetric::INNER_PRODUCT:
        // Negate so that higher similarity = lower "distance" —
        // consistent with the min-heap search interface.
#ifdef __AVX2__
        return -dot_product_avx2(a, b, dim);
#else
        return -dot_product_scalar(a, b, dim);
#endif

    case DistanceMetric::COSINE:
        // OPTIMIZATION: vectors are L2-normalized at insertion time
        // (see Collection::upsert). For unit-norm vectors,
        // cosine_similarity(a,b) == dot_product(a,b), and
        // cosine_distance = 1 - cosine_similarity.
        // We use dot_product directly here, negated and shifted:
        // distance = 1 - dot(a,b).
        // This avoids sqrt and division in the general cosine formula,
        // saving ~3-4x computation vs cosine_scalar().
#ifdef __AVX2__
        return 1.0f - dot_product_avx2(a, b, dim);
#else
        return 1.0f - dot_product_scalar(a, b, dim);
#endif

    default:
        throw std::invalid_argument("Unknown distance metric");
    }
}

// ============================================================
// Normalization utilities
// ============================================================

void l2_normalize(float* vec, int dim) {
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        norm_sq += vec[i] * vec[i];
    }
    if (norm_sq < 1e-10f) {
        // Zero vector — cannot normalize, leave as-is
        return;
    }
    float inv_norm = 1.0f / std::sqrt(norm_sq);
    for (int i = 0; i < dim; ++i) {
        vec[i] *= inv_norm;
    }
}

bool is_normalized(const float* vec, int dim, float tolerance) {
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        norm_sq += vec[i] * vec[i];
    }
    return std::abs(norm_sq - 1.0f) < tolerance;
}
