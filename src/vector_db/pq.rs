//! Product Quantization (PQ) for vector compression
//!
//! Compresses high-dimensional vectors by splitting them into subvectors
//! and quantizing each subvector to its nearest centroid.
//!
//! For 384-dim MiniLM: 1536 bytes → 48 bytes (32x compression)
//! For 768-dim CLIP: 3072 bytes → 96 bytes (32x compression)
//!
//! Trade-off: ~95% recall accuracy for 32x storage reduction

use anyhow::{anyhow, Result};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Number of centroids per subspace (2^8 = 256, fits in u8)
pub const NUM_CENTROIDS: usize = 256;

/// Default subvector dimension (8 floats per subvector)
pub const DEFAULT_SUBVEC_DIM: usize = 8;

/// Product Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    /// Total vector dimension (e.g., 384 for MiniLM, 768 for CLIP)
    pub dimension: usize,
    /// Number of subvectors (dimension / subvec_dim)
    pub num_subvectors: usize,
    /// Dimension of each subvector
    pub subvec_dim: usize,
    /// Number of centroids per subspace (default 256)
    pub num_centroids: usize,
    /// Number of k-means iterations for training
    pub kmeans_iterations: usize,
}

impl PQConfig {
    /// Create PQ config for a given vector dimension
    pub fn for_dimension(dimension: usize) -> Self {
        let subvec_dim = DEFAULT_SUBVEC_DIM;
        let num_subvectors = dimension / subvec_dim;

        assert!(
            dimension % subvec_dim == 0,
            "Dimension {} must be divisible by subvec_dim {}",
            dimension,
            subvec_dim
        );

        Self {
            dimension,
            num_subvectors,
            subvec_dim,
            num_centroids: NUM_CENTROIDS,
            kmeans_iterations: 20,
        }
    }

    /// Create config for MiniLM embeddings (384 dims)
    pub fn minilm() -> Self {
        Self::for_dimension(384)
    }

    /// Create config for CLIP embeddings (768 dims)
    pub fn clip() -> Self {
        Self::for_dimension(768)
    }
}

/// Trained Product Quantizer
///
/// Contains centroids for each subspace, learned from training data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    /// Configuration
    pub config: PQConfig,
    /// Centroids for each subspace: [num_subvectors][num_centroids][subvec_dim]
    pub centroids: Vec<Vec<Vec<f32>>>,
    /// Whether the quantizer has been trained
    pub trained: bool,
}

impl ProductQuantizer {
    /// Create a new untrained product quantizer
    pub fn new(config: PQConfig) -> Self {
        Self {
            config,
            centroids: Vec::new(),
            trained: false,
        }
    }

    /// Create and train a product quantizer on given vectors
    pub fn train(config: PQConfig, training_vectors: &[Vec<f32>]) -> Result<Self> {
        if training_vectors.is_empty() {
            return Err(anyhow!("No training vectors provided"));
        }

        let first_dim = training_vectors[0].len();
        if first_dim != config.dimension {
            return Err(anyhow!(
                "Training vector dimension {} doesn't match config {}",
                first_dim,
                config.dimension
            ));
        }

        let mut pq = Self::new(config);
        pq.fit(training_vectors)?;
        Ok(pq)
    }

    /// Train centroids on a set of vectors using k-means
    fn fit(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        let n_vectors = vectors.len();
        let n_subvectors = self.config.num_subvectors;
        let subvec_dim = self.config.subvec_dim;
        let n_centroids = self.config.num_centroids.min(n_vectors);
        let iterations = self.config.kmeans_iterations;

        tracing::info!(
            "Training PQ: {} vectors, {} subvectors, {} centroids, {} iterations",
            n_vectors, n_subvectors, n_centroids, iterations
        );

        // Initialize centroids storage
        self.centroids = Vec::with_capacity(n_subvectors);

        // Train k-means for each subspace independently
        for subvec_idx in 0..n_subvectors {
            let start = subvec_idx * subvec_dim;
            let end = start + subvec_dim;

            // Extract subvectors for this subspace
            let subvectors: Vec<Vec<f32>> = vectors
                .iter()
                .map(|v| v[start..end].to_vec())
                .collect();

            // Run k-means clustering
            let centroids = self.kmeans(&subvectors, n_centroids, iterations)?;
            self.centroids.push(centroids);
        }

        self.trained = true;
        tracing::info!("PQ training complete");
        Ok(())
    }

    /// Simple k-means clustering
    fn kmeans(&self, vectors: &[Vec<f32>], k: usize, iterations: usize) -> Result<Vec<Vec<f32>>> {
        let dim = vectors[0].len();
        let n = vectors.len();

        // Initialize centroids by random sampling
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let mut centroids: Vec<Vec<f32>> = indices
            .iter()
            .take(k)
            .map(|&i| vectors[i].clone())
            .collect();

        // Pad with random vectors if needed (when n < k)
        while centroids.len() < k {
            let idx = indices[centroids.len() % n];
            centroids.push(vectors[idx].clone());
        }

        let mut assignments = vec![0usize; n];

        // K-means iterations
        for _ in 0..iterations {
            // Assign each vector to nearest centroid
            for (i, vec) in vectors.iter().enumerate() {
                let mut best_centroid = 0;
                let mut best_dist = f32::MAX;

                for (c, centroid) in centroids.iter().enumerate() {
                    let dist = squared_l2_distance(vec, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_centroid = c;
                    }
                }
                assignments[i] = best_centroid;
            }

            // Update centroids
            let mut new_centroids: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
            let mut counts = vec![0usize; k];

            for (i, vec) in vectors.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                for (j, &v) in vec.iter().enumerate() {
                    new_centroids[c][j] += v;
                }
            }

            // Average and handle empty clusters
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..dim {
                        new_centroids[c][j] /= counts[c] as f32;
                    }
                    centroids[c] = new_centroids[c].clone();
                }
                // Keep old centroid if cluster is empty
            }
        }

        Ok(centroids)
    }

    /// Encode a vector to PQ codes (one u8 per subvector)
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        if !self.trained {
            return Err(anyhow!("ProductQuantizer not trained"));
        }

        if vector.len() != self.config.dimension {
            return Err(anyhow!(
                "Vector dimension {} doesn't match config {}",
                vector.len(),
                self.config.dimension
            ));
        }

        let mut codes = Vec::with_capacity(self.config.num_subvectors);
        let subvec_dim = self.config.subvec_dim;

        for (subvec_idx, subspace_centroids) in self.centroids.iter().enumerate() {
            let start = subvec_idx * subvec_dim;
            let end = start + subvec_dim;
            let subvector = &vector[start..end];

            // Find nearest centroid
            let mut best_centroid = 0u8;
            let mut best_dist = f32::MAX;

            for (c, centroid) in subspace_centroids.iter().enumerate() {
                let dist = squared_l2_distance_slice(subvector, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_centroid = c as u8;
                }
            }

            codes.push(best_centroid);
        }

        Ok(codes)
    }

    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, codes: &[u8]) -> Result<Vec<f32>> {
        if !self.trained {
            return Err(anyhow!("ProductQuantizer not trained"));
        }

        if codes.len() != self.config.num_subvectors {
            return Err(anyhow!(
                "Code length {} doesn't match num_subvectors {}",
                codes.len(),
                self.config.num_subvectors
            ));
        }

        let mut vector = Vec::with_capacity(self.config.dimension);

        for (subvec_idx, &code) in codes.iter().enumerate() {
            let centroid = &self.centroids[subvec_idx][code as usize];
            vector.extend_from_slice(centroid);
        }

        Ok(vector)
    }

    /// Compute asymmetric distance between query vector and encoded vector
    ///
    /// ADC (Asymmetric Distance Computation) is more accurate than SDC
    /// because the query is not quantized.
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> Result<f32> {
        if !self.trained {
            return Err(anyhow!("ProductQuantizer not trained"));
        }

        let subvec_dim = self.config.subvec_dim;
        let mut total_dist = 0.0f32;

        for (subvec_idx, &code) in codes.iter().enumerate() {
            let start = subvec_idx * subvec_dim;
            let end = start + subvec_dim;
            let query_subvec = &query[start..end];
            let centroid = &self.centroids[subvec_idx][code as usize];

            total_dist += squared_l2_distance_slice(query_subvec, centroid);
        }

        Ok(total_dist)
    }

    /// Build distance lookup table for a query (ADC optimization)
    ///
    /// Pre-computes distances from each query subvector to all centroids.
    /// This allows O(M) distance computation per encoded vector instead of O(D).
    pub fn build_distance_table(&self, query: &[f32]) -> Result<Vec<Vec<f32>>> {
        if !self.trained {
            return Err(anyhow!("ProductQuantizer not trained"));
        }

        let subvec_dim = self.config.subvec_dim;
        let n_centroids = self.config.num_centroids;
        let mut table = Vec::with_capacity(self.config.num_subvectors);

        for (subvec_idx, subspace_centroids) in self.centroids.iter().enumerate() {
            let start = subvec_idx * subvec_dim;
            let end = start + subvec_dim;
            let query_subvec = &query[start..end];

            let mut distances = Vec::with_capacity(n_centroids);
            for centroid in subspace_centroids {
                distances.push(squared_l2_distance_slice(query_subvec, centroid));
            }
            table.push(distances);
        }

        Ok(table)
    }

    /// Fast distance computation using pre-built lookup table
    #[inline]
    pub fn distance_with_table(&self, table: &[Vec<f32>], codes: &[u8]) -> f32 {
        let mut total = 0.0f32;
        for (subvec_idx, &code) in codes.iter().enumerate() {
            total += table[subvec_idx][code as usize];
        }
        total
    }

    /// Batch encode multiple vectors
    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<u8>>> {
        vectors.iter().map(|v| self.encode(v)).collect()
    }

    /// Compressed size in bytes for one vector
    pub fn compressed_size(&self) -> usize {
        self.config.num_subvectors // One byte per subvector
    }

    /// Original size in bytes for one vector
    pub fn original_size(&self) -> usize {
        self.config.dimension * std::mem::size_of::<f32>()
    }

    /// Compression ratio
    pub fn compression_ratio(&self) -> f32 {
        self.original_size() as f32 / self.compressed_size() as f32
    }
}

/// Squared L2 distance between two vectors
#[inline]
fn squared_l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

/// Squared L2 distance for slices (same as above but clearer intent)
#[inline]
fn squared_l2_distance_slice(a: &[f32], b: &[f32]) -> f32 {
    squared_l2_distance(a, b)
}

/// Compressed vector storage for PQ-encoded vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedVectorStore {
    /// The trained quantizer
    pub quantizer: ProductQuantizer,
    /// Encoded vectors: vector_id -> PQ codes
    pub codes: HashMap<u32, Vec<u8>>,
}

impl CompressedVectorStore {
    /// Create a new compressed vector store
    pub fn new(quantizer: ProductQuantizer) -> Self {
        Self {
            quantizer,
            codes: HashMap::new(),
        }
    }

    /// Train quantizer and create store from training vectors
    pub fn train_and_create(config: PQConfig, training_vectors: &[Vec<f32>]) -> Result<Self> {
        let quantizer = ProductQuantizer::train(config, training_vectors)?;
        Ok(Self::new(quantizer))
    }

    /// Add a vector to the store
    pub fn add(&mut self, vector_id: u32, vector: &[f32]) -> Result<()> {
        let codes = self.quantizer.encode(vector)?;
        self.codes.insert(vector_id, codes);
        Ok(())
    }

    /// Get PQ codes for a vector
    pub fn get_codes(&self, vector_id: u32) -> Option<&Vec<u8>> {
        self.codes.get(&vector_id)
    }

    /// Decode a vector back to approximate floats
    pub fn decode(&self, vector_id: u32) -> Result<Vec<f32>> {
        let codes = self.codes.get(&vector_id)
            .ok_or_else(|| anyhow!("Vector {} not found", vector_id))?;
        self.quantizer.decode(codes)
    }

    /// Search for k nearest neighbors using PQ distance
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        // Build distance table for fast lookup
        let table = self.quantizer.build_distance_table(query)?;

        // Compute distances to all vectors
        let mut distances: Vec<(u32, f32)> = self.codes
            .iter()
            .map(|(&id, codes)| (id, self.quantizer.distance_with_table(&table, codes)))
            .collect();

        // Sort by distance and take top k
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);

        Ok(distances)
    }

    /// Number of vectors in store
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Total compressed storage size in bytes
    pub fn storage_bytes(&self) -> usize {
        self.codes.len() * self.quantizer.compressed_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn test_pq_encode_decode() {
        let vectors = generate_random_vectors(1000, 384);
        let config = PQConfig::minilm();
        let pq = ProductQuantizer::train(config, &vectors).unwrap();

        // Test encode/decode
        let original = &vectors[0];
        let codes = pq.encode(original).unwrap();
        let decoded = pq.decode(&codes).unwrap();

        // Check dimensions
        assert_eq!(codes.len(), 48); // 384 / 8 = 48 subvectors
        assert_eq!(decoded.len(), 384);

        // Decoded should be close to original (not exact due to quantization)
        let mse: f32 = original.iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / 384.0;

        // MSE should be reasonably low
        assert!(mse < 0.1, "MSE too high: {}", mse);
    }

    #[test]
    fn test_compression_ratio() {
        let config = PQConfig::minilm();
        let pq = ProductQuantizer::new(config);

        assert_eq!(pq.original_size(), 384 * 4); // 1536 bytes
        assert_eq!(pq.compressed_size(), 48);    // 48 bytes
        assert!((pq.compression_ratio() - 32.0).abs() < 0.01);
    }

    #[test]
    fn test_distance_table() {
        let vectors = generate_random_vectors(100, 384);
        let config = PQConfig::minilm();
        let pq = ProductQuantizer::train(config, &vectors).unwrap();

        let query = &vectors[0];
        let codes = pq.encode(&vectors[1]).unwrap();

        // Direct distance
        let direct_dist = pq.asymmetric_distance(query, &codes).unwrap();

        // Table-based distance
        let table = pq.build_distance_table(query).unwrap();
        let table_dist = pq.distance_with_table(&table, &codes);

        // Should be identical
        assert!((direct_dist - table_dist).abs() < 1e-6);
    }

    #[test]
    fn test_compressed_store_search() {
        let vectors = generate_random_vectors(1000, 384);
        let config = PQConfig::minilm();

        let mut store = CompressedVectorStore::train_and_create(config, &vectors).unwrap();

        // Add all vectors
        for (i, v) in vectors.iter().enumerate() {
            store.add(i as u32, v).unwrap();
        }

        // Search
        let results = store.search(&vectors[0], 10).unwrap();

        // First result should be the query itself or very close vector
        // Note: PQ has quantization error, so distance won't be exactly 0
        assert_eq!(results.len(), 10);
        // The query should be among the top results (within first few)
        let query_in_top_results = results.iter().take(5).any(|(id, _)| *id == 0);
        assert!(query_in_top_results, "Query vector not found in top 5 results");
    }
}
