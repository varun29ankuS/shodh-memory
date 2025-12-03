//! Inline SIMD distance functions with zero overhead
//! These are optimized for hot paths where function call overhead matters

#![allow(dead_code)]

use std::arch::x86_64::*;

/// Inline dot product with compile-time SIMD selection
#[inline(always)]
pub fn dot_product_inline(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        // Compile-time feature detection - zero overhead!
        #[cfg(target_feature = "avx2")]
        unsafe {
            return dot_product_avx2_inline(a, b);
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            dot_product_scalar_inline(a, b)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        dot_product_scalar_inline(a, b)
    }
}

/// Inline AVX2 dot product - no function call overhead
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]  // Cannot use always with target_feature
unsafe fn dot_product_avx2_inline(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let simd_len = len & !7;

    let mut sum = _mm256_setzero_ps();

    // Unrolled by 2 for better performance
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

    // Horizontal sum
    let sum_array = std::mem::transmute::<__m256, [f32; 8]>(sum);
    let mut result = sum_array.iter().sum::<f32>();

    // Handle remaining elements
    for j in simd_len..len {
        result += a[j] * b[j];
    }

    result
}

/// Inline scalar dot product - always inlined
#[inline(always)]
fn dot_product_scalar_inline(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let unroll_len = len & !3;
    let mut sum = 0.0;

    // Unrolled by 4 for better performance
    let mut i = 0;
    while i < unroll_len {
        sum += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3];
        i += 4;
    }

    // Handle remaining
    for j in unroll_len..len {
        sum += a[j] * b[j];
    }

    sum
}

/// Inline Euclidean distance squared (no sqrt for comparisons)
#[inline(always)]
pub fn euclidean_squared_inline(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        return euclidean_squared_avx2_inline(a, b);
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        euclidean_squared_scalar_inline(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]  // Cannot use always with target_feature
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

    // Horizontal sum
    let sum_array = std::mem::transmute::<__m256, [f32; 8]>(sum);
    let mut result = sum_array.iter().sum::<f32>();

    // Handle remaining
    for j in simd_len..len {
        let diff = a[j] - b[j];
        result += diff * diff;
    }

    result
}

#[inline(always)]
fn euclidean_squared_scalar_inline(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_correctness() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let result = dot_product_inline(&a, &b);
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

        assert!((result - expected).abs() < 1e-5);
    }
}
