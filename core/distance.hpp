#pragma once

#include "core/types.hpp"

// ============================================================
// Distance computation interface
// ============================================================
// All functions operate on raw float pointers for zero-copy
// compatibility with the flat contiguous vector storage.
//
// Vectors for COSINE distance are L2-normalized at insertion
// time, so cosine similarity reduces to a dot product — this
// enables reuse of the faster dot product kernel.
// ============================================================

// Main dispatcher: selects AVX2 or scalar based on compile-time __AVX2__
float compute_distance(const float* a, const float* b,
                       int dim, DistanceMetric metric);

// ---- Scalar implementations --------------------------------
float l2_squared_scalar(const float* a, const float* b, int dim);
float dot_product_scalar(const float* a, const float* b, int dim);
float cosine_scalar(const float* a, const float* b, int dim);

// ---- AVX2 implementations (guarded with #ifdef __AVX2__) ----
// Process 8 floats per iteration using 256-bit YMM registers.
// Falls back to scalar for remainder when dim % 8 != 0.
float l2_squared_avx2(const float* a, const float* b, int dim);
float dot_product_avx2(const float* a, const float* b, int dim);

// ---- Normalization ------------------------------------------
// Normalize vec in-place to unit L2 norm.
// No-op if vec is the zero vector (avoids divide-by-zero).
void l2_normalize(float* vec, int dim);

// Returns true if the vector's L2 norm is within tolerance of 1.0
bool is_normalized(const float* vec, int dim,
                   float tolerance = NORM_TOLERANCE);
