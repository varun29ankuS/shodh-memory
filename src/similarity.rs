//! Vector similarity search for semantic retrieval
//!
//! Uses SIMD-optimized operations from distance_inline for single comparisons.

use ordered_float::OrderedFloat;
use crate::vector_db::distance_inline::dot_product_inline;

/// Compute cosine similarity between two vectors (SIMD-optimized)
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot = dot_product_inline(a, b);
    let norm_a = dot_product_inline(a, a).sqrt();
    let norm_b = dot_product_inline(b, b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Find top-k most similar vectors
pub fn top_k_similar<T>(
    query: &[f32],
    candidates: &[(Vec<f32>, T)],
    k: usize,
) -> Vec<(f32, T)>
where
    T: Clone,
{
    let mut scored: Vec<(OrderedFloat<f32>, T)> = candidates
        .iter()
        .map(|(vec, item)| {
            let score = cosine_similarity(query, vec);
            (OrderedFloat(score), item.clone())
        })
        .collect();

    // Sort by score descending
    scored.sort_by(|a, b| b.0.cmp(&a.0));

    // Take top k
    scored
        .into_iter()
        .take(k)
        .map(|(score, item)| (score.0, item))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);

        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }
}
