//! Inline SIMD distance functions with zero overhead
//!
//! Provides optimized distance calculations for vector similarity search.
//! Supports:
//! - x86-64: AVX2 + FMA instructions
//! - ARM64: NEON instructions (Apple Silicon, ARM servers)
//! - Fallback: Scalar with loop unrolling
//!
//! All functions are `#[inline(always)]` for hot path optimization.

#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// =============================================================================
// DOT PRODUCT
// =============================================================================

/// Inline dot product with compile-time SIMD selection
///
/// For normalized vectors, dot product equals cosine similarity.
/// Higher values = more similar.
#[inline(always)]
pub fn dot_product_inline(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "avx2")]
        unsafe {
            return dot_product_avx2_inline(a, b);
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            dot_product_scalar_inline(a, b)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(target_feature = "neon")]
        unsafe {
            return dot_product_neon_inline(a, b);
        }

        #[cfg(not(target_feature = "neon"))]
        {
            dot_product_scalar_inline(a, b)
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        dot_product_scalar_inline(a, b)
    }
}

/// AVX2 + FMA dot product (x86-64)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn dot_product_avx2_inline(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let simd_len = len & !7; // Round down to multiple of 8

    let mut sum = _mm256_setzero_ps();

    // Process 16 elements at a time (2x unroll)
    let mut i = 0;
    while i + 16 <= simd_len {
        let va1 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i));
        let va2 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb2 = _mm256_loadu_ps(b.as_ptr().add(i + 8));

        sum = _mm256_fmadd_ps(va1, vb1, sum);
        sum = _mm256_fmadd_ps(va2, vb2, sum);
        i += 16;
    }

    // Handle remaining blocks of 8
    while i < simd_len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum = _mm256_fmadd_ps(va, vb, sum);
        i += 8;
    }

    // Horizontal sum: extract and sum all 8 lanes
    let sum_array = std::mem::transmute::<__m256, [f32; 8]>(sum);
    let mut result = sum_array[0]
        + sum_array[1]
        + sum_array[2]
        + sum_array[3]
        + sum_array[4]
        + sum_array[5]
        + sum_array[6]
        + sum_array[7];

    // Handle remaining elements
    for j in simd_len..len {
        result += a[j] * b[j];
    }

    result
}

/// NEON dot product (ARM64 - Apple Silicon, ARM servers)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot_product_neon_inline(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let simd_len = len & !3; // Round down to multiple of 4

    let mut sum = vdupq_n_f32(0.0);

    // Process 8 elements at a time (2x unroll)
    let mut i = 0;
    while i + 8 <= simd_len {
        let va1 = vld1q_f32(a.as_ptr().add(i));
        let vb1 = vld1q_f32(b.as_ptr().add(i));
        let va2 = vld1q_f32(a.as_ptr().add(i + 4));
        let vb2 = vld1q_f32(b.as_ptr().add(i + 4));

        sum = vfmaq_f32(sum, va1, vb1);
        sum = vfmaq_f32(sum, va2, vb2);
        i += 8;
    }

    // Handle remaining blocks of 4
    while i < simd_len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        sum = vfmaq_f32(sum, va, vb);
        i += 4;
    }

    // Horizontal sum: add all 4 lanes
    let mut result = vaddvq_f32(sum);

    // Handle remaining elements
    for j in simd_len..len {
        result += a[j] * b[j];
    }

    result
}

/// Scalar dot product with 4x unrolling
#[inline(always)]
fn dot_product_scalar_inline(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let unroll_len = len & !3;
    let mut sum = 0.0;

    let mut i = 0;
    while i < unroll_len {
        sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        i += 4;
    }

    for j in unroll_len..len {
        sum += a[j] * b[j];
    }

    sum
}

// =============================================================================
// EUCLIDEAN DISTANCE
// =============================================================================

/// Inline Euclidean distance squared (no sqrt for comparisons)
///
/// Returns ||a - b||^2. Use this for distance comparisons where
/// the actual distance value isn't needed (sqrt is monotonic).
#[inline(always)]
pub fn euclidean_squared_inline(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "avx2")]
        unsafe {
            return euclidean_squared_avx2_inline(a, b);
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            euclidean_squared_scalar_inline(a, b)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(target_feature = "neon")]
        unsafe {
            return euclidean_squared_neon_inline(a, b);
        }

        #[cfg(not(target_feature = "neon"))]
        {
            euclidean_squared_scalar_inline(a, b)
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        euclidean_squared_scalar_inline(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn euclidean_squared_avx2_inline(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let simd_len = len & !7;

    let mut sum = _mm256_setzero_ps();

    let mut i = 0;
    while i < simd_len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
        i += 8;
    }

    let sum_array = std::mem::transmute::<__m256, [f32; 8]>(sum);
    let mut result = sum_array[0]
        + sum_array[1]
        + sum_array[2]
        + sum_array[3]
        + sum_array[4]
        + sum_array[5]
        + sum_array[6]
        + sum_array[7];

    for j in simd_len..len {
        let diff = a[j] - b[j];
        result += diff * diff;
    }

    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn euclidean_squared_neon_inline(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let simd_len = len & !3;

    let mut sum = vdupq_n_f32(0.0);

    let mut i = 0;
    while i < simd_len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff);
        i += 4;
    }

    let mut result = vaddvq_f32(sum);

    for j in simd_len..len {
        let diff = a[j] - b[j];
        result += diff * diff;
    }

    result
}

#[inline(always)]
fn euclidean_squared_scalar_inline(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let unroll_len = len & !3;
    let mut sum = 0.0;

    let mut i = 0;
    while i < unroll_len {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }

    for j in unroll_len..len {
        let diff = a[j] - b[j];
        sum += diff * diff;
    }

    sum
}

// =============================================================================
// L2 NORM
// =============================================================================

/// Compute L2 norm (magnitude) of a vector
///
/// Returns ||a|| = sqrt(sum(a_i^2))
#[inline(always)]
pub fn l2_norm_inline(a: &[f32]) -> f32 {
    l2_norm_squared_inline(a).sqrt()
}

/// Compute L2 norm squared (no sqrt)
///
/// Returns ||a||^2 = sum(a_i^2)
#[inline(always)]
pub fn l2_norm_squared_inline(a: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "avx2")]
        unsafe {
            return l2_norm_squared_avx2_inline(a);
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            l2_norm_squared_scalar_inline(a)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(target_feature = "neon")]
        unsafe {
            return l2_norm_squared_neon_inline(a);
        }

        #[cfg(not(target_feature = "neon"))]
        {
            l2_norm_squared_scalar_inline(a)
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        l2_norm_squared_scalar_inline(a)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn l2_norm_squared_avx2_inline(a: &[f32]) -> f32 {
    let len = a.len();
    let simd_len = len & !7;

    let mut sum = _mm256_setzero_ps();

    let mut i = 0;
    while i < simd_len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        sum = _mm256_fmadd_ps(va, va, sum);
        i += 8;
    }

    let sum_array = std::mem::transmute::<__m256, [f32; 8]>(sum);
    let mut result = sum_array[0]
        + sum_array[1]
        + sum_array[2]
        + sum_array[3]
        + sum_array[4]
        + sum_array[5]
        + sum_array[6]
        + sum_array[7];

    for j in simd_len..len {
        result += a[j] * a[j];
    }

    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn l2_norm_squared_neon_inline(a: &[f32]) -> f32 {
    let len = a.len();
    let simd_len = len & !3;

    let mut sum = vdupq_n_f32(0.0);

    let mut i = 0;
    while i < simd_len {
        let va = vld1q_f32(a.as_ptr().add(i));
        sum = vfmaq_f32(sum, va, va);
        i += 4;
    }

    let mut result = vaddvq_f32(sum);

    for j in simd_len..len {
        result += a[j] * a[j];
    }

    result
}

#[inline(always)]
fn l2_norm_squared_scalar_inline(a: &[f32]) -> f32 {
    let len = a.len();
    let unroll_len = len & !3;
    let mut sum = 0.0;

    let mut i = 0;
    while i < unroll_len {
        sum += a[i] * a[i] + a[i + 1] * a[i + 1] + a[i + 2] * a[i + 2] + a[i + 3] * a[i + 3];
        i += 4;
    }

    for j in unroll_len..len {
        sum += a[j] * a[j];
    }

    sum
}

// =============================================================================
// COSINE SIMILARITY
// =============================================================================

/// Compute cosine similarity between two vectors
///
/// Returns dot(a, b) / (||a|| * ||b||)
/// Range: [-1, 1] where 1 = identical direction, -1 = opposite, 0 = orthogonal
///
/// For pre-normalized vectors, use `dot_product_inline` directly (faster).
#[inline(always)]
pub fn cosine_similarity_inline(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let dot = dot_product_inline(a, b);
    let norm_a = l2_norm_squared_inline(a);
    let norm_b = l2_norm_squared_inline(b);

    let denominator = (norm_a * norm_b).sqrt();

    if denominator < 1e-10 {
        return 0.0; // Handle zero vectors
    }

    dot / denominator
}

/// Compute cosine distance (1 - cosine_similarity)
///
/// Range: [0, 2] where 0 = identical, 2 = opposite
#[inline(always)]
pub fn cosine_distance_inline(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity_inline(a, b)
}

// =============================================================================
// NORMALIZED VECTOR DISTANCE
// =============================================================================

/// Distance for normalized vectors using negative dot product
///
/// For L2-normalized vectors: ||a-b||^2 = 2 - 2*dot(a,b)
/// So -dot(a,b) gives correct distance ordering (smaller = closer)
///
/// This is faster than euclidean_squared for normalized vectors.
#[inline(always)]
pub fn normalized_distance_inline(a: &[f32], b: &[f32]) -> f32 {
    -dot_product_inline(a, b)
}

/// Verify vector is approximately L2-normalized
///
/// Returns true if ||a|| is within epsilon of 1.0
#[inline(always)]
pub fn is_normalized(a: &[f32], epsilon: f32) -> bool {
    let norm_sq = l2_norm_squared_inline(a);
    (norm_sq - 1.0).abs() < epsilon
}

/// Normalize a vector in-place
#[inline]
pub fn normalize_inplace(a: &mut [f32]) {
    let norm = l2_norm_inline(a);
    if norm > 1e-10 {
        let inv_norm = 1.0 / norm;
        for x in a.iter_mut() {
            *x *= inv_norm;
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_dot_product_correctness() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let result = dot_product_inline(&a, &b);
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

        assert!(
            (result - expected).abs() < EPSILON,
            "dot product: got {}, expected {}",
            result,
            expected
        );
    }

    #[test]
    fn test_euclidean_squared_correctness() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = euclidean_squared_inline(&a, &b);
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum();

        assert!(
            (result - expected).abs() < EPSILON,
            "euclidean squared: got {}, expected {}",
            result,
            expected
        );
    }

    #[test]
    fn test_l2_norm() {
        let a = vec![3.0, 4.0]; // 3-4-5 triangle
        let result = l2_norm_inline(&a);
        assert!(
            (result - 5.0).abs() < EPSILON,
            "l2 norm: got {}, expected 5.0",
            result
        );
    }

    #[test]
    fn test_cosine_similarity() {
        // Same direction
        let a = vec![1.0, 0.0];
        let b = vec![2.0, 0.0];
        let sim = cosine_similarity_inline(&a, &b);
        assert!(
            (sim - 1.0).abs() < EPSILON,
            "same direction: got {}, expected 1.0",
            sim
        );

        // Opposite direction
        let c = vec![-1.0, 0.0];
        let sim2 = cosine_similarity_inline(&a, &c);
        assert!(
            (sim2 - (-1.0)).abs() < EPSILON,
            "opposite direction: got {}, expected -1.0",
            sim2
        );

        // Orthogonal
        let d = vec![0.0, 1.0];
        let sim3 = cosine_similarity_inline(&a, &d);
        assert!(
            sim3.abs() < EPSILON,
            "orthogonal: got {}, expected 0.0",
            sim3
        );
    }

    #[test]
    fn test_is_normalized() {
        let mut v = vec![3.0, 4.0];
        assert!(!is_normalized(&v, 0.01));

        normalize_inplace(&mut v);
        assert!(is_normalized(&v, 0.01));
    }

    #[test]
    fn test_normalized_distance() {
        // For normalized vectors, -dot gives distance ordering
        let a = vec![1.0, 0.0];
        let b = vec![0.6, 0.8]; // normalized

        let dist = normalized_distance_inline(&a, &b);
        assert!(dist < 0.0); // Negative because vectors are similar (positive dot product)

        // More similar vectors should have more negative distance
        let c = vec![0.8, 0.6]; // closer to a
        let dist_c = normalized_distance_inline(&a, &c);
        assert!(dist_c < dist); // c is closer to a than b
    }

    #[test]
    fn test_large_vectors() {
        // Test with 384-dim vectors (MiniLM size)
        let a: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..384).map(|i| ((384 - i) as f32) * 0.01).collect();

        let dot_result = dot_product_inline(&a, &b);
        let expected_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!(
            (dot_result - expected_dot).abs() < 0.01,
            "384-dim dot: got {}, expected {}",
            dot_result,
            expected_dot
        );

        let eucl_result = euclidean_squared_inline(&a, &b);
        let expected_eucl: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum();
        assert!(
            (eucl_result - expected_eucl).abs() < 0.01,
            "384-dim euclidean: got {}, expected {}",
            eucl_result,
            expected_eucl
        );
    }
}
