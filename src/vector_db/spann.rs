//! SPANN - Scalable Proximity-graph ANN for billion-scale vector search
//!
//! Implements disk-based IVF (Inverted File Index) with optional PQ compression.
//! Designed for datasets that don't fit in RAM.
//!
//! # Architecture
//!
//! SPANN partitions vectors into √n clusters using k-means:
//! - **Centroids**: Cluster centers stored in RAM for fast routing
//! - **Posting Lists**: Vector assignments stored on disk, mmap'd on demand
//! - **PQ Compression**: Optional product quantization for 32x storage reduction
//!
//! # File Format (v1)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │ Header (128 bytes)                                      │
//! │ ├── magic: [u8; 4] = "SPAN"                             │
//! │ ├── version: u32 = 1                                    │
//! │ ├── num_vectors: u64                                    │
//! │ ├── num_partitions: u32                                 │
//! │ ├── dimension: u32                                      │
//! │ ├── pq_enabled: u8 (0 or 1)                             │
//! │ ├── pq_subvectors: u32                                  │
//! │ ├── distance_metric: u8                                 │
//! │ ├── checksum: u64                                       │
//! │ ├── centroids_offset: u64                               │
//! │ ├── codebook_offset: u64                                │
//! │ ├── posting_index_offset: u64                           │
//! │ ├── posting_data_offset: u64                            │
//! │ └── reserved: [u8; 60]                                  │
//! ├─────────────────────────────────────────────────────────┤
//! │ Centroids Section (aligned to 64 bytes)                 │
//! │ └── [[f32; dimension]; num_partitions]                  │
//! ├─────────────────────────────────────────────────────────┤
//! │ PQ Codebook Section (if pq_enabled, aligned to 64)      │
//! │ ├── num_subvectors: u32                                 │
//! │ ├── num_centroids: u32 (always 256)                     │
//! │ ├── subvec_dim: u32                                     │
//! │ └── codebook: [[f32; subvec_dim]; 256] × num_subvectors │
//! ├─────────────────────────────────────────────────────────┤
//! │ Posting List Index (12 bytes per partition)             │
//! │ └── [(offset: u64, count: u32); num_partitions]         │
//! ├─────────────────────────────────────────────────────────┤
//! │ Posting List Data                                       │
//! │ └── For each partition:                                 │
//! │     └── entries: [PostingEntry; count]                  │
//! │         where PostingEntry =                            │
//! │           - vector_id: u32                              │
//! │           - pq_codes: [u8; num_subvectors] (if PQ)      │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Query Flow
//!
//! 1. Compute distances from query to all centroids
//! 2. Select top-k nearest partitions (multi-probe)
//! 3. Load posting lists for selected partitions
//! 4. Compute PQ distances (ADC) or exact distances
//! 5. Return top-k results across all probed partitions

use anyhow::{anyhow, Result};
use memmap2::{Mmap, MmapMut};
use parking_lot::RwLock;
use rand::seq::SliceRandom;
use std::collections::BinaryHeap;
use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::info;

use super::pq::{PQConfig, ProductQuantizer, NUM_CENTROIDS};
use super::vamana::DistanceMetric;

const MAGIC: [u8; 4] = *b"SPAN";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 128;
const ALIGNMENT: usize = 64;
const POSTING_INDEX_ENTRY_SIZE: usize = 12; // u64 offset + u32 count

/// SPANN configuration
#[derive(Debug, Clone)]
pub struct SpannConfig {
    /// Vector dimension
    pub dimension: usize,
    /// Number of partitions (default: √n, set during build)
    pub num_partitions: Option<usize>,
    /// Enable PQ compression (32x storage reduction, ~5% recall loss)
    pub use_pq: bool,
    /// Number of partitions to probe during search (default: 10)
    pub num_probes: usize,
    /// K-means iterations for clustering
    pub kmeans_iterations: usize,
    /// Distance metric
    pub distance_metric: DistanceMetric,
    /// Minimum vectors per partition before merge
    pub min_partition_size: usize,
    /// Maximum vectors per partition before split
    pub max_partition_size: usize,
}

impl Default for SpannConfig {
    fn default() -> Self {
        Self {
            dimension: 384,
            num_partitions: None, // Auto-compute as √n
            use_pq: true,
            num_probes: 10,
            kmeans_iterations: 25,
            distance_metric: DistanceMetric::NormalizedDotProduct,
            min_partition_size: 100,
            max_partition_size: 10000,
        }
    }
}

impl SpannConfig {
    /// Create config for MiniLM embeddings (384 dims)
    pub fn minilm() -> Self {
        Self {
            dimension: 384,
            ..Default::default()
        }
    }

    /// Create config for CLIP embeddings (768 dims)
    pub fn clip() -> Self {
        Self {
            dimension: 768,
            ..Default::default()
        }
    }

    /// Compute optimal partition count for dataset size
    pub fn compute_partitions(&self, num_vectors: usize) -> usize {
        self.num_partitions
            .unwrap_or_else(|| ((num_vectors as f64).sqrt().ceil() as usize).max(1))
    }
}

/// Posting list entry - vector ID with optional PQ codes
#[derive(Debug, Clone)]
pub struct PostingEntry {
    /// Vector ID (maps back to original storage)
    pub vector_id: u32,
    /// PQ codes (if PQ enabled)
    pub pq_codes: Option<Vec<u8>>,
}

impl PostingEntry {
    /// Size in bytes when serialized
    pub fn serialized_size(pq_subvectors: usize) -> usize {
        4 + pq_subvectors // u32 + pq_codes
    }
}

/// Partition metadata
#[derive(Debug, Clone)]
pub struct Partition {
    /// Partition ID
    pub id: u32,
    /// Centroid vector
    pub centroid: Vec<f32>,
    /// Entries in this partition
    pub entries: Vec<PostingEntry>,
}

/// File header for SPANN index
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
struct SpannHeader {
    magic: [u8; 4],
    version: u32,
    num_vectors: u64,
    num_partitions: u32,
    dimension: u32,
    pq_enabled: u8,
    pq_subvectors: u32,
    distance_metric: u8,
    checksum: u64,
    centroids_offset: u64,
    codebook_offset: u64,
    posting_index_offset: u64,
    posting_data_offset: u64,
    reserved: [u8; 60],
}

impl SpannHeader {
    fn new(
        num_vectors: usize,
        num_partitions: usize,
        dimension: usize,
        pq_enabled: bool,
        pq_subvectors: usize,
        distance_metric: DistanceMetric,
    ) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            num_vectors: num_vectors as u64,
            num_partitions: num_partitions as u32,
            dimension: dimension as u32,
            pq_enabled: if pq_enabled { 1 } else { 0 },
            pq_subvectors: pq_subvectors as u32,
            distance_metric: match distance_metric {
                DistanceMetric::NormalizedDotProduct => 0,
                DistanceMetric::Euclidean => 1,
                DistanceMetric::Cosine => 2,
            },
            checksum: 0,
            centroids_offset: 0,
            codebook_offset: 0,
            posting_index_offset: 0,
            posting_data_offset: 0,
            reserved: [0u8; 60],
        }
    }

    fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];
        let mut offset = 0;

        bytes[offset..offset + 4].copy_from_slice(&self.magic);
        offset += 4;
        bytes[offset..offset + 4].copy_from_slice(&self.version.to_le_bytes());
        offset += 4;
        bytes[offset..offset + 8].copy_from_slice(&self.num_vectors.to_le_bytes());
        offset += 8;
        bytes[offset..offset + 4].copy_from_slice(&self.num_partitions.to_le_bytes());
        offset += 4;
        bytes[offset..offset + 4].copy_from_slice(&self.dimension.to_le_bytes());
        offset += 4;
        bytes[offset] = self.pq_enabled;
        offset += 1;
        bytes[offset..offset + 4].copy_from_slice(&self.pq_subvectors.to_le_bytes());
        offset += 4;
        bytes[offset] = self.distance_metric;
        offset += 1;
        bytes[offset..offset + 8].copy_from_slice(&self.checksum.to_le_bytes());
        offset += 8;
        bytes[offset..offset + 8].copy_from_slice(&self.centroids_offset.to_le_bytes());
        offset += 8;
        bytes[offset..offset + 8].copy_from_slice(&self.codebook_offset.to_le_bytes());
        offset += 8;
        bytes[offset..offset + 8].copy_from_slice(&self.posting_index_offset.to_le_bytes());
        offset += 8;
        bytes[offset..offset + 8].copy_from_slice(&self.posting_data_offset.to_le_bytes());
        // Reserved bytes already 0
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(anyhow!("Header too small"));
        }

        let magic: [u8; 4] = bytes[0..4].try_into()?;
        if magic != MAGIC {
            return Err(anyhow!("Invalid magic bytes: {:?}", magic));
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into()?);
        if version != VERSION {
            return Err(anyhow!("Unsupported version: {}", version));
        }

        let mut offset = 8;
        let num_vectors = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?);
        offset += 8;
        let num_partitions = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
        offset += 4;
        let dimension = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
        offset += 4;
        let pq_enabled = bytes[offset];
        offset += 1;
        let pq_subvectors = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
        offset += 4;
        let distance_metric = bytes[offset];
        offset += 1;
        let checksum = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?);
        offset += 8;
        let centroids_offset = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?);
        offset += 8;
        let codebook_offset = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?);
        offset += 8;
        let posting_index_offset = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?);
        offset += 8;
        let posting_data_offset = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?);

        Ok(Self {
            magic,
            version,
            num_vectors,
            num_partitions,
            dimension,
            pq_enabled,
            pq_subvectors,
            distance_metric,
            checksum,
            centroids_offset,
            codebook_offset,
            posting_index_offset,
            posting_data_offset,
            reserved: [0u8; 60],
        })
    }

    fn distance_metric_enum(&self) -> DistanceMetric {
        match self.distance_metric {
            0 => DistanceMetric::NormalizedDotProduct,
            1 => DistanceMetric::Euclidean,
            2 => DistanceMetric::Cosine,
            _ => DistanceMetric::NormalizedDotProduct,
        }
    }
}

/// SPANN Index - Scalable disk-based ANN
pub struct SpannIndex {
    /// Configuration
    pub config: SpannConfig,
    /// Cluster centroids (kept in RAM for fast routing)
    centroids: Arc<RwLock<Vec<Vec<f32>>>>,
    /// PQ quantizer (if enabled)
    quantizer: Arc<RwLock<Option<ProductQuantizer>>>,
    /// Partitions (in-memory for building, cleared after save)
    partitions: Arc<RwLock<Vec<Partition>>>,
    /// Memory-mapped file for disk access
    mmap: Arc<RwLock<Option<Mmap>>>,
    /// Number of vectors indexed
    num_vectors: AtomicUsize,
    /// Number of partitions
    num_partitions: AtomicUsize,
    /// Posting index offsets (from header)
    posting_index_offset: AtomicUsize,
    /// Posting data offset (from header)
    posting_data_offset: AtomicUsize,
}

impl SpannIndex {
    /// Create a new empty SPANN index
    pub fn new(config: SpannConfig) -> Self {
        Self {
            config,
            centroids: Arc::new(RwLock::new(Vec::new())),
            quantizer: Arc::new(RwLock::new(None)),
            partitions: Arc::new(RwLock::new(Vec::new())),
            mmap: Arc::new(RwLock::new(None)),
            num_vectors: AtomicUsize::new(0),
            num_partitions: AtomicUsize::new(0),
            posting_index_offset: AtomicUsize::new(0),
            posting_data_offset: AtomicUsize::new(0),
        }
    }

    /// Build index from vectors
    ///
    /// 1. Cluster vectors using k-means
    /// 2. Assign each vector to nearest centroid
    /// 3. Optionally train PQ and encode vectors
    pub fn build(&mut self, vectors: Vec<Vec<f32>>) -> Result<()> {
        if vectors.is_empty() {
            return Err(anyhow!("Cannot build index from empty vectors"));
        }

        let n = vectors.len();
        let dim = vectors[0].len();

        if dim != self.config.dimension {
            return Err(anyhow!(
                "Vector dimension {} doesn't match config {}",
                dim,
                self.config.dimension
            ));
        }

        let num_partitions = self.config.compute_partitions(n);
        info!(
            "Building SPANN index: {} vectors, {} partitions, PQ={}",
            n, num_partitions, self.config.use_pq
        );

        let start = std::time::Instant::now();

        // Step 1: K-means clustering to find centroids
        let centroids = self.kmeans_cluster(&vectors, num_partitions)?;

        // Step 2: Train PQ on vectors (if enabled)
        let quantizer = if self.config.use_pq {
            info!("Training PQ quantizer...");
            let pq_config = PQConfig::for_dimension(dim);
            let pq = ProductQuantizer::train(pq_config, &vectors)?;
            Some(pq)
        } else {
            None
        };

        // Step 3: Assign vectors to partitions
        let mut partitions: Vec<Partition> = centroids
            .iter()
            .enumerate()
            .map(|(i, c)| Partition {
                id: i as u32,
                centroid: c.clone(),
                entries: Vec::new(),
            })
            .collect();

        for (vec_id, vector) in vectors.iter().enumerate() {
            let partition_id = self.find_nearest_centroid(vector, &centroids);

            let pq_codes = if let Some(ref pq) = quantizer {
                Some(pq.encode(vector)?)
            } else {
                None
            };

            partitions[partition_id].entries.push(PostingEntry {
                vector_id: vec_id as u32,
                pq_codes,
            });
        }

        // Log partition distribution
        let sizes: Vec<usize> = partitions.iter().map(|p| p.entries.len()).collect();
        let min_size = sizes.iter().min().copied().unwrap_or(0);
        let max_size = sizes.iter().max().copied().unwrap_or(0);
        let avg_size = if !sizes.is_empty() {
            sizes.iter().sum::<usize>() / sizes.len()
        } else {
            0
        };
        info!(
            "Partition distribution: min={}, max={}, avg={}",
            min_size, max_size, avg_size
        );

        // Store results
        *self.centroids.write() = centroids;
        *self.quantizer.write() = quantizer;
        *self.partitions.write() = partitions;
        self.num_vectors.store(n, Ordering::Release);
        self.num_partitions.store(num_partitions, Ordering::Release);

        info!("SPANN build complete in {:?}", start.elapsed());

        Ok(())
    }

    /// K-means clustering
    fn kmeans_cluster(&self, vectors: &[Vec<f32>], k: usize) -> Result<Vec<Vec<f32>>> {
        let n = vectors.len();
        let dim = vectors[0].len();
        let iterations = self.config.kmeans_iterations;

        // Initialize centroids by random sampling
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let mut centroids: Vec<Vec<f32>> = indices
            .iter()
            .take(k)
            .map(|&i| vectors[i].clone())
            .collect();

        // Pad if needed
        while centroids.len() < k {
            let idx = indices[centroids.len() % n];
            centroids.push(vectors[idx].clone());
        }

        let mut assignments = vec![0usize; n];

        // K-means iterations
        for iter in 0..iterations {
            let mut changed = 0usize;

            // Assign vectors to nearest centroid
            for (i, vec) in vectors.iter().enumerate() {
                let new_assignment = self.find_nearest_centroid(vec, &centroids);
                if new_assignment != assignments[i] {
                    changed += 1;
                }
                assignments[i] = new_assignment;
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

            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..dim {
                        new_centroids[c][j] /= counts[c] as f32;
                    }
                    centroids[c] = new_centroids[c].clone();
                }
            }

            if iter % 5 == 0 {
                info!(
                    "K-means iter {}/{}: {} assignments changed",
                    iter + 1,
                    iterations,
                    changed
                );
            }

            // Early termination if converged
            if changed == 0 {
                info!("K-means converged at iteration {}", iter + 1);
                break;
            }
        }

        Ok(centroids)
    }

    /// Find nearest centroid for a vector
    #[inline]
    fn find_nearest_centroid(&self, vector: &[f32], centroids: &[Vec<f32>]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for (i, centroid) in centroids.iter().enumerate() {
            let dist = self.compute_distance(vector, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Compute distance between two vectors
    #[inline]
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.distance_metric {
            DistanceMetric::Euclidean => a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum(),
            DistanceMetric::NormalizedDotProduct | DistanceMetric::Cosine => {
                // For normalized vectors, 1 - dot_product gives distance
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                1.0 - dot
            }
        }
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        let centroids = self.centroids.read();
        if centroids.is_empty() {
            return Ok(Vec::new());
        }

        // Step 1: Find top-nprobes nearest partitions
        let mut partition_distances: Vec<(usize, f32)> = centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.compute_distance(query, c)))
            .collect();

        partition_distances
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let probe_partitions: Vec<usize> = partition_distances
            .iter()
            .take(self.config.num_probes)
            .map(|(i, _)| *i)
            .collect();

        // Step 2: Search in selected partitions
        let quantizer = self.quantizer.read();
        let partitions = self.partitions.read();

        // Build distance table for PQ (required for SPANN search)
        let distance_table = if let Some(ref pq) = *quantizer {
            Some(pq.build_distance_table(query)?)
        } else {
            anyhow::bail!(
                "SPANN search requires PQ quantizer but use_pq is disabled. \
                 PostingEntry stores only PQ codes, not original vectors."
            );
        };

        // Collect candidates from all probed partitions
        // Use max-heap to keep k smallest distances: pop() removes largest (worst) match
        let mut heap: BinaryHeap<(ordered_float::OrderedFloat<f32>, u32)> = BinaryHeap::new();

        if !partitions.is_empty() {
            // In-memory search (before save or after full load)
            for &partition_id in &probe_partitions {
                if partition_id >= partitions.len() {
                    continue;
                }

                for entry in &partitions[partition_id].entries {
                    let dist = if let (Some(ref table), Some(ref codes)) =
                        (&distance_table, &entry.pq_codes)
                    {
                        quantizer
                            .as_ref()
                            .unwrap()
                            .distance_with_table(table, codes)
                    } else {
                        // Would need original vectors - skip for now
                        continue;
                    };

                    heap.push((ordered_float::OrderedFloat(dist), entry.vector_id));
                    if heap.len() > k {
                        heap.pop(); // Removes largest distance (worst match)
                    }
                }
            }
        } else {
            // Disk-based search (after load_from_file)
            let mmap_guard = self.mmap.read();
            if let Some(ref mmap) = *mmap_guard {
                let pq_subvectors = if quantizer.is_some() {
                    self.config.dimension / 8 // PQ subvector count
                } else {
                    0
                };

                for &partition_id in &probe_partitions {
                    let entries = self.read_posting_list(mmap, partition_id, pq_subvectors)?;

                    for entry in entries {
                        let dist = if let (Some(ref table), Some(ref codes)) =
                            (&distance_table, &entry.pq_codes)
                        {
                            quantizer
                                .as_ref()
                                .unwrap()
                                .distance_with_table(table, codes)
                        } else {
                            continue;
                        };

                        heap.push((ordered_float::OrderedFloat(dist), entry.vector_id));
                        if heap.len() > k {
                            heap.pop(); // Removes largest distance (worst match)
                        }
                    }
                }
            }
        }

        // Convert heap to sorted results (smallest distance first)
        let mut results: Vec<(u32, f32)> = heap.into_iter().map(|(d, id)| (id, d.0)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Read posting list from mmap'd file
    fn read_posting_list(
        &self,
        mmap: &Mmap,
        partition_id: usize,
        pq_subvectors: usize,
    ) -> Result<Vec<PostingEntry>> {
        let index_offset = self.posting_index_offset.load(Ordering::Acquire);
        let data_offset = self.posting_data_offset.load(Ordering::Acquire);

        // Read posting index entry
        let entry_offset = index_offset + partition_id * POSTING_INDEX_ENTRY_SIZE;
        if entry_offset + POSTING_INDEX_ENTRY_SIZE > mmap.len() {
            return Err(anyhow!("Posting index out of bounds"));
        }

        let list_offset =
            u64::from_le_bytes(mmap[entry_offset..entry_offset + 8].try_into()?) as usize;
        let count =
            u32::from_le_bytes(mmap[entry_offset + 8..entry_offset + 12].try_into()?) as usize;

        // Read posting list entries
        let entry_size = PostingEntry::serialized_size(pq_subvectors);
        let list_start = data_offset + list_offset;
        let list_end = list_start + count * entry_size;

        if list_end > mmap.len() {
            return Err(anyhow!("Posting list data out of bounds"));
        }

        let mut entries = Vec::with_capacity(count);
        let mut offset = list_start;

        for _ in 0..count {
            let vector_id = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
            offset += 4;

            let pq_codes = if pq_subvectors > 0 {
                let codes = mmap[offset..offset + pq_subvectors].to_vec();
                offset += pq_subvectors;
                Some(codes)
            } else {
                None
            };

            entries.push(PostingEntry {
                vector_id,
                pq_codes,
            });
        }

        Ok(entries)
    }

    /// Save index to file
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let start = std::time::Instant::now();

        let centroids = self.centroids.read();
        let quantizer = self.quantizer.read();
        let partitions = self.partitions.read();

        if centroids.is_empty() {
            return Err(anyhow!("Cannot save empty index"));
        }

        let num_vectors = self.num_vectors.load(Ordering::Acquire);
        let num_partitions = centroids.len();
        let dimension = self.config.dimension;
        let pq_enabled = quantizer.is_some();
        let pq_subvectors = if pq_enabled { dimension / 8 } else { 0 };

        // Calculate section offsets
        let centroids_offset = align_to(HEADER_SIZE, ALIGNMENT);
        let centroids_size = num_partitions * dimension * 4;

        let codebook_offset = align_to(centroids_offset + centroids_size, ALIGNMENT);
        let codebook_size = if pq_enabled {
            // num_subvectors + num_centroids + subvec_dim + codebook data
            4 + 4 + 4 + (pq_subvectors * NUM_CENTROIDS * 8 * 4)
        } else {
            0
        };

        let posting_index_offset = align_to(codebook_offset + codebook_size, ALIGNMENT);
        let posting_index_size = num_partitions * POSTING_INDEX_ENTRY_SIZE;

        let posting_data_offset = align_to(posting_index_offset + posting_index_size, ALIGNMENT);

        // Calculate posting data size
        let entry_size = PostingEntry::serialized_size(pq_subvectors);
        let posting_data_size: usize = partitions
            .iter()
            .map(|p| p.entries.len() * entry_size)
            .sum();

        let total_size = posting_data_offset + posting_data_size;

        // Create file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(total_size as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Create header
        let mut header = SpannHeader::new(
            num_vectors,
            num_partitions,
            dimension,
            pq_enabled,
            pq_subvectors,
            self.config.distance_metric,
        );
        header.centroids_offset = centroids_offset as u64;
        header.codebook_offset = codebook_offset as u64;
        header.posting_index_offset = posting_index_offset as u64;
        header.posting_data_offset = posting_data_offset as u64;

        // Write centroids
        let mut offset = centroids_offset;
        for centroid in centroids.iter() {
            for &val in centroid {
                mmap[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
                offset += 4;
            }
        }

        // Write PQ codebook
        if let Some(ref pq) = *quantizer {
            offset = codebook_offset;
            mmap[offset..offset + 4].copy_from_slice(&(pq_subvectors as u32).to_le_bytes());
            offset += 4;
            mmap[offset..offset + 4].copy_from_slice(&(NUM_CENTROIDS as u32).to_le_bytes());
            offset += 4;
            mmap[offset..offset + 4].copy_from_slice(&8u32.to_le_bytes()); // subvec_dim
            offset += 4;

            for subspace_centroids in &pq.centroids {
                for centroid in subspace_centroids {
                    for &val in centroid {
                        mmap[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
                        offset += 4;
                    }
                }
            }
        }

        // Write posting index and data
        let mut data_write_offset: usize = 0;
        for (partition_id, partition) in partitions.iter().enumerate() {
            // Write index entry
            let index_entry_offset = posting_index_offset + partition_id * POSTING_INDEX_ENTRY_SIZE;
            mmap[index_entry_offset..index_entry_offset + 8]
                .copy_from_slice(&(data_write_offset as u64).to_le_bytes());
            mmap[index_entry_offset + 8..index_entry_offset + 12]
                .copy_from_slice(&(partition.entries.len() as u32).to_le_bytes());

            // Write posting list entries
            offset = posting_data_offset + data_write_offset;
            for entry in &partition.entries {
                mmap[offset..offset + 4].copy_from_slice(&entry.vector_id.to_le_bytes());
                offset += 4;

                if let Some(ref codes) = entry.pq_codes {
                    mmap[offset..offset + codes.len()].copy_from_slice(codes);
                    offset += codes.len();
                }
            }

            data_write_offset += partition.entries.len() * entry_size;
        }

        // Compute checksum and write header
        let checksum = compute_checksum(&mmap[HEADER_SIZE..]);
        header.checksum = checksum;
        mmap[..HEADER_SIZE].copy_from_slice(&header.to_bytes());

        mmap.flush()?;

        info!(
            "Saved SPANN index: {} vectors, {} partitions, {} bytes in {:?}",
            num_vectors,
            num_partitions,
            total_size,
            start.elapsed()
        );

        Ok(())
    }

    /// Load index from file
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let start = std::time::Instant::now();

        if !path.exists() {
            return Err(anyhow!("Index file not found: {:?}", path));
        }

        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Read and verify header
        let header = SpannHeader::from_bytes(&mmap[..HEADER_SIZE])?;

        let stored_checksum = header.checksum;
        let computed_checksum = compute_checksum(&mmap[HEADER_SIZE..]);
        if stored_checksum != computed_checksum {
            return Err(anyhow!(
                "Checksum mismatch: stored={}, computed={}",
                stored_checksum,
                computed_checksum
            ));
        }

        let num_vectors = header.num_vectors as usize;
        let num_partitions = header.num_partitions as usize;
        let dimension = header.dimension as usize;
        let pq_enabled = header.pq_enabled == 1;
        let _pq_subvectors = header.pq_subvectors as usize;

        // Read centroids
        let mut centroids = Vec::with_capacity(num_partitions);
        let mut offset = header.centroids_offset as usize;
        for _ in 0..num_partitions {
            let mut centroid = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                let val = f32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
                centroid.push(val);
                offset += 4;
            }
            centroids.push(centroid);
        }

        // Read PQ codebook
        let quantizer = if pq_enabled {
            offset = header.codebook_offset as usize;
            let num_subvectors = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?) as usize;
            offset += 4;
            let num_centroids = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?) as usize;
            offset += 4;
            let subvec_dim = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?) as usize;
            offset += 4;

            let mut pq_centroids = Vec::with_capacity(num_subvectors);
            for _ in 0..num_subvectors {
                let mut subspace = Vec::with_capacity(num_centroids);
                for _ in 0..num_centroids {
                    let mut centroid = Vec::with_capacity(subvec_dim);
                    for _ in 0..subvec_dim {
                        let val = f32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
                        centroid.push(val);
                        offset += 4;
                    }
                    subspace.push(centroid);
                }
                pq_centroids.push(subspace);
            }

            let config = PQConfig {
                dimension,
                num_subvectors,
                subvec_dim,
                num_centroids,
                kmeans_iterations: 20,
            };

            Some(ProductQuantizer {
                config,
                centroids: pq_centroids,
                trained: true,
            })
        } else {
            None
        };

        let config = SpannConfig {
            dimension,
            num_partitions: Some(num_partitions),
            use_pq: pq_enabled,
            distance_metric: header.distance_metric_enum(),
            ..Default::default()
        };

        let index = SpannIndex {
            config,
            centroids: Arc::new(RwLock::new(centroids)),
            quantizer: Arc::new(RwLock::new(quantizer)),
            partitions: Arc::new(RwLock::new(Vec::new())), // Not loaded - use mmap
            mmap: Arc::new(RwLock::new(Some(mmap))),
            num_vectors: AtomicUsize::new(num_vectors),
            num_partitions: AtomicUsize::new(num_partitions),
            posting_index_offset: AtomicUsize::new(header.posting_index_offset as usize),
            posting_data_offset: AtomicUsize::new(header.posting_data_offset as usize),
        };

        info!(
            "Loaded SPANN index: {} vectors, {} partitions in {:?}",
            num_vectors,
            num_partitions,
            start.elapsed()
        );

        Ok(index)
    }

    /// Insert a single vector into the index
    pub fn insert(&mut self, vector_id: u32, vector: &[f32]) -> Result<()> {
        let centroids = self.centroids.read();
        if centroids.is_empty() {
            return Err(anyhow!("Cannot insert into empty index - build first"));
        }

        let partition_id = self.find_nearest_centroid(vector, &centroids);
        drop(centroids);

        let pq_codes = {
            let quantizer = self.quantizer.read();
            if let Some(ref pq) = *quantizer {
                Some(pq.encode(vector)?)
            } else {
                None
            }
        };

        let mut partitions = self.partitions.write();
        if partition_id < partitions.len() {
            partitions[partition_id].entries.push(PostingEntry {
                vector_id,
                pq_codes,
            });
            self.num_vectors.fetch_add(1, Ordering::Release);
        }

        Ok(())
    }

    /// Number of vectors in the index
    pub fn len(&self) -> usize {
        self.num_vectors.load(Ordering::Acquire)
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of partitions
    pub fn num_partitions(&self) -> usize {
        self.num_partitions.load(Ordering::Acquire)
    }

    /// Verify index file integrity
    pub fn verify_index_file(path: &Path) -> Result<bool> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Ok(false);
        }

        let header = SpannHeader::from_bytes(&mmap[..HEADER_SIZE])?;
        let stored_checksum = header.checksum;
        let computed_checksum = compute_checksum(&mmap[HEADER_SIZE..]);

        Ok(stored_checksum == computed_checksum)
    }
}

/// Align offset to boundary
fn align_to(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

/// Compute FNV-1a checksum
fn compute_checksum(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn generate_random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| {
                let mut vec: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
                // Normalize for cosine distance
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    vec.iter_mut().for_each(|x| *x /= norm);
                }
                vec
            })
            .collect()
    }

    #[test]
    fn test_spann_build_and_search() {
        let vectors = generate_random_vectors(1000, 384);

        let config = SpannConfig {
            dimension: 384,
            use_pq: true,
            num_probes: 20, // More probes for better recall
            ..Default::default()
        };

        let mut index = SpannIndex::new(config);
        index.build(vectors.clone()).unwrap();

        // Search - stringent test: query should be in top 10
        let results = index.search(&vectors[0], 10).unwrap();

        assert!(!results.is_empty(), "Search should return results");
        assert_eq!(results.len(), 10, "Should return exactly k results");

        // Query vector should be #1 or very close (top 3) with PQ
        // Self-distance should be near zero even with quantization
        let query_position = results.iter().position(|(id, _)| *id == 0);
        assert!(
            query_position.is_some() && query_position.unwrap() < 3,
            "Query vector should be in top 3 results, found at {:?}, results: {:?}",
            query_position,
            results.iter().take(5).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_spann_save_and_load() {
        let temp_dir = tempdir().unwrap();
        let index_path = temp_dir.path().join("test.spann");

        let vectors = generate_random_vectors(500, 384);

        let config = SpannConfig {
            dimension: 384,
            use_pq: true,
            ..Default::default()
        };

        let mut index = SpannIndex::new(config);
        index.build(vectors.clone()).unwrap();

        // Save
        index.save_to_file(&index_path).unwrap();
        assert!(index_path.exists());

        // Verify
        assert!(SpannIndex::verify_index_file(&index_path).unwrap());

        // Load
        let loaded = SpannIndex::load_from_file(&index_path).unwrap();
        assert_eq!(loaded.len(), 500);
        assert!(loaded.num_partitions() > 0);

        // Search loaded index
        let results = loaded.search(&vectors[0], 10).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_partition_count() {
        let config = SpannConfig::default();

        assert_eq!(config.compute_partitions(100), 10);
        assert_eq!(config.compute_partitions(10000), 100);
        assert_eq!(config.compute_partitions(1000000), 1000);
    }

    #[test]
    fn test_no_pq() {
        let vectors = generate_random_vectors(100, 384);

        let config = SpannConfig {
            dimension: 384,
            use_pq: false,
            ..Default::default()
        };

        let mut index = SpannIndex::new(config);
        index.build(vectors).unwrap();

        assert!(index.quantizer.read().is_none());
    }
}
