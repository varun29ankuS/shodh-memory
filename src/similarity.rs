//! Vector similarity search for semantic retrieval
//!
//! Uses SIMD-optimized operations from distance_inline for single comparisons.

use crate::vector_db::distance_inline::dot_product_inline;
use ordered_float::OrderedFloat;

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

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Find top-k most similar vectors
pub fn top_k_similar<T>(query: &[f32], candidates: &[(Vec<f32>, T)], k: usize) -> Vec<(f32, T)>
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

    #[test]
    fn test_cosine_similarity_edge_cases() {
        // Mismatched lengths are treated as non-comparable.
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        // Any zero vector should yield 0 to avoid NaN propagation.
        let zero = vec![0.0, 0.0, 0.0];
        let vec3 = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&zero, &vec3), 0.0);
        assert_eq!(cosine_similarity(&vec3, &zero), 0.0);
    }

    #[test]
    fn test_cosine_similarity_negative_correlation() {
        let a = vec![1.0, -1.0];
        let b = vec![-1.0, 1.0];
        let score = cosine_similarity(&a, &b);
        assert!((score + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_top_k_similar_orders_and_limits() {
        let query = vec![1.0, 0.0];
        let candidates = vec![
            (vec![1.0, 0.0], "perfect"),
            (vec![0.7, 0.7], "diagonal"),
            (vec![0.0, 1.0], "orthogonal"),
            (vec![-1.0, 0.0], "opposite"),
        ];

        let top2 = top_k_similar(&query, &candidates, 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].1, "perfect");
        assert_eq!(top2[1].1, "diagonal");
        assert!(top2[0].0 >= top2[1].0);
    }

    #[test]
    fn test_top_k_similar_k_larger_than_candidates() {
        let query = vec![1.0, 0.0];
        let candidates = vec![(vec![1.0, 0.0], 1u8), (vec![0.0, 1.0], 2u8)];

        let result = top_k_similar(&query, &candidates, 10);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_top_k_similar_empty_candidates() {
        let query = vec![1.0, 0.0];
        let candidates: Vec<(Vec<f32>, &str)> = Vec::new();
        let result = top_k_similar(&query, &candidates, 3);
        assert!(result.is_empty());
    }
}
