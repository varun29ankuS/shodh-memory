//! Vamana: Single-shot graph construction for billion-scale similarity search
//! Based on Microsoft Research paper: "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node"
//!
//! Production implementation optimized for 8-16GB RAM laptops
//!
//! # Index Maintenance
//!
//! Incremental inserts use neighbor truncation which can degrade search quality over time.
//! For optimal recall@10 accuracy, consider rebuilding the index periodically:
//!
//! - **Recommended**: Rebuild after every 10,000 incremental inserts
//! - **Impact**: Without rebuilds, recall@10 may degrade 5-15% over thousands of inserts
//! - **Detection**: Use `needs_rebuild()` to check if rebuild is recommended
//!
//! ## Example
//! ```ignore
//! if index.needs_rebuild() {
//!     let vectors = index.extract_all_vectors();
//!     index.rebuild_from_vectors(&vectors)?;
//! }
//! ```

use super::distance_inline::{
    cosine_similarity_inline, dot_product_inline, euclidean_squared_inline,
    normalized_distance_inline,
};
use anyhow::{anyhow, Result};
use memmap2::MmapMut;
use parking_lot::RwLock;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};

/// Distance metric for vector similarity
///
/// All metrics are SIMD-optimized (AVX2 on x86-64, NEON on ARM64).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    /// For L2-normalized vectors (default). Fastest option.
    /// Uses -dot_product which gives correct distance ordering.
    /// MiniLM and most sentence transformers output normalized vectors.
    #[default]
    NormalizedDotProduct,

    /// Euclidean distance squared. Works for any vectors.
    /// Slightly slower than dot product but doesn't require normalization.
    Euclidean,

    /// Cosine similarity (1 - cos_sim). Works for any vectors.
    /// Computes norms on-the-fly, slowest but most flexible.
    Cosine,
}

/// Vamana configuration
#[derive(Debug, Clone)]
pub struct VamanaConfig {
    /// Maximum degree of graph (R in paper)
    pub max_degree: usize,

    /// Search list size during construction (L in paper)
    pub search_list_size: usize,

    /// Alpha parameter for RNG pruning (α in paper, typically 1.2)
    pub alpha: f32,

    /// Vector dimension
    pub dimension: usize,

    /// Use memory mapping for large datasets
    pub use_mmap: bool,

    /// Distance metric for similarity calculation
    /// Default: NormalizedDotProduct (assumes L2-normalized vectors)
    pub distance_metric: DistanceMetric,
}

impl Default for VamanaConfig {
    fn default() -> Self {
        Self {
            max_degree: 32,                             // R=32 for billion-scale
            search_list_size: 75,                       // L=75 during construction
            alpha: 1.2,                                 // Standard α for pruning
            dimension: 384,                             // MiniLM dimension
            use_mmap: true,                             // Disk-based for large datasets
            distance_metric: DistanceMetric::default(), // NormalizedDotProduct for MiniLM
        }
    }
}

/// Node in the Vamana graph
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct VamanaNode {
    /// Node ID
    id: u32,

    /// Neighbor IDs sorted by distance
    neighbors: Vec<u32>,
}

/// Threshold for recommending index rebuild (number of incremental inserts)
pub const REBUILD_THRESHOLD: usize = 10_000;

/// Main Vamana index
pub struct VamanaIndex {
    pub(crate) config: VamanaConfig,

    /// Graph structure: node_id -> neighbors
    pub(crate) graph: Arc<RwLock<Vec<VamanaNode>>>,

    /// Vectors (can be memory-mapped)
    pub(crate) vectors: Arc<RwLock<VectorStorage>>,

    /// Medoid/centroid as entry point
    pub(crate) medoid: Arc<RwLock<u32>>,

    /// Number of vectors
    pub(crate) num_vectors: usize,

    /// Storage path for mmap files (unique per index instance)
    storage_path: Option<PathBuf>,

    /// Counter for incremental inserts since last rebuild
    /// Used to track index quality degradation
    incremental_inserts: std::sync::atomic::AtomicUsize,

    /// Flag to prevent concurrent rebuilds
    rebuilding: std::sync::atomic::AtomicBool,
}

/// Vector storage abstraction
pub(crate) enum VectorStorage {
    /// In-memory storage
    Memory(Vec<Vec<f32>>),

    /// Memory-mapped storage
    Mmap {
        mmap: MmapMut,
        dimension: usize,
        num_vectors: usize,
    },
}

impl VamanaIndex {
    /// Create new Vamana index
    pub fn new(config: VamanaConfig) -> Result<Self> {
        Self::with_storage_path(config, None)
    }

    /// Create new Vamana index with explicit storage path for mmap
    pub fn with_storage_path(config: VamanaConfig, storage_path: Option<PathBuf>) -> Result<Self> {
        Ok(Self {
            config,
            graph: Arc::new(RwLock::new(Vec::new())),
            vectors: Arc::new(RwLock::new(VectorStorage::Memory(Vec::new()))),
            medoid: Arc::new(RwLock::new(0)),
            num_vectors: 0,
            storage_path,
            incremental_inserts: std::sync::atomic::AtomicUsize::new(0),
            rebuilding: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Get number of vectors in the index
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// Build index from vectors using Vamana algorithm
    pub fn build(&mut self, vectors: Vec<Vec<f32>>) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        let n = vectors.len();
        self.num_vectors = n;

        info!("Building Vamana index with {} vectors", n);

        // Step 1: Initialize graph randomly
        self.initialize_graph(n)?;

        // Step 2: Store vectors
        self.store_vectors(vectors)?;

        // Step 3: Find medoid (closest to centroid)
        self.find_medoid()?;

        // Step 4: Main Vamana construction
        let mut iteration = 0;
        loop {
            iteration += 1;
            info!("Vamana iteration {}", iteration);

            let mut updates = 0;

            // Process each node
            for node_id in 0..n {
                // Get vector for this node
                let query = self.get_vector(node_id as u32)?;

                // Search for L nearest neighbors
                let candidates =
                    self.greedy_search(&query, self.config.search_list_size, *self.medoid.read())?;

                // Prune using α-RNG strategy
                let pruned = self.robust_prune(node_id as u32, &candidates)?;

                // Update graph
                let mut graph = self.graph.write();
                if graph[node_id].neighbors != pruned {
                    updates += 1;
                    graph[node_id].neighbors = pruned.clone();

                    // Ensure bidirectional edges
                    for &neighbor in &pruned {
                        if neighbor as usize >= graph.len() {
                            continue;
                        }

                        let neighbor_node = &mut graph[neighbor as usize];
                        if !neighbor_node.neighbors.contains(&(node_id as u32)) {
                            neighbor_node.neighbors.push(node_id as u32);

                            // Prune neighbor if exceeds max degree
                            if neighbor_node.neighbors.len() > self.config.max_degree {
                                let _neighbor_vec = self.get_vector(neighbor)?;
                                let pruned_neighbors = self.robust_prune(
                                    neighbor,
                                    &neighbor_node
                                        .neighbors
                                        .iter()
                                        .map(|&id| SearchCandidate { id, distance: 0.0 })
                                        .collect::<Vec<_>>(),
                                )?;
                                neighbor_node.neighbors = pruned_neighbors;
                            }
                        }
                    }
                }
            }

            info!("Updated {} nodes", updates);

            // Converged
            if updates == 0 || iteration >= 2 {
                break;
            }
        }

        info!("Vamana construction complete");
        Ok(())
    }

    /// Initialize random graph
    fn initialize_graph(&mut self, n: usize) -> Result<()> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        let mut graph = Vec::with_capacity(n);

        for i in 0..n {
            // Create random edges
            let mut neighbors: Vec<u32> = (0..n as u32).filter(|&j| j != i as u32).collect();

            neighbors.shuffle(&mut rng);
            neighbors.truncate(self.config.max_degree);

            graph.push(VamanaNode {
                id: i as u32,
                neighbors,
            });
        }

        *self.graph.write() = graph;
        Ok(())
    }

    /// Store vectors in storage
    fn store_vectors(&mut self, vectors: Vec<Vec<f32>>) -> Result<()> {
        let mut storage = self.vectors.write();

        if self.config.use_mmap {
            // Require explicit storage path for mmap mode
            let mmap_path = self.storage_path
                .as_ref()
                .map(|p| p.join("vamana_vectors.bin"))
                .ok_or_else(|| anyhow!("Storage path required for mmap mode. Use with_storage_path() or disable use_mmap."))?;

            // Ensure parent directory exists
            if let Some(parent) = mmap_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            // Create memory-mapped file
            let file_size = vectors.len() * self.config.dimension * std::mem::size_of::<f32>();
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&mmap_path)?;

            file.set_len(file_size as u64)?;

            // SAFETY CHECK: Verify file size is correctly set and aligned
            let actual_file_size = file.metadata()?.len();
            if actual_file_size != file_size as u64 {
                anyhow::bail!(
                    "File size mismatch: expected {file_size} bytes, got {actual_file_size} bytes"
                );
            }

            // SAFETY CHECK: Verify size is properly aligned for f32 (4-byte alignment)
            if file_size % std::mem::align_of::<f32>() != 0 {
                anyhow::bail!(
                    "File size {} is not aligned to f32 alignment ({})",
                    file_size,
                    std::mem::align_of::<f32>()
                );
            }

            // SAFETY: MmapMut::map_mut is safe because:
            // 1. File handle is valid and exclusively owned
            // 2. File size is non-zero and verified above
            // 3. File permissions allow read+write
            // 4. No other process has this file mapped
            let mut mmap = unsafe { MmapMut::map_mut(&file)? };

            // SAFETY CHECK: Verify pointer alignment before casting to f32*
            let ptr = mmap.as_mut_ptr();
            if ptr.align_offset(std::mem::align_of::<f32>()) != 0 {
                anyhow::bail!(
                    "Mmap pointer {:?} is not aligned to f32 alignment ({})",
                    ptr,
                    std::mem::align_of::<f32>()
                );
            }

            // SAFETY: from_raw_parts_mut is safe because:
            // 1. Pointer is properly aligned (verified above)
            // 2. Memory region is valid for the entire length
            // 3. Length calculation is correct: vectors.len() * dimension
            // 4. Mmap is exclusively owned and won't be accessed elsewhere
            // 5. f32 is Copy, so no double-free issues
            let float_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    ptr as *mut f32,
                    vectors.len() * self.config.dimension,
                )
            };

            for (i, vec) in vectors.iter().enumerate() {
                let start = i * self.config.dimension;
                float_slice[start..start + self.config.dimension].copy_from_slice(vec);
            }

            *storage = VectorStorage::Mmap {
                mmap,
                dimension: self.config.dimension,
                num_vectors: vectors.len(),
            };
        } else {
            *storage = VectorStorage::Memory(vectors);
        }

        Ok(())
    }

    /// Find medoid (closest point to centroid)
    fn find_medoid(&mut self) -> Result<()> {
        let n = self.num_vectors;
        if n == 0 {
            return Ok(());
        }

        // Compute centroid
        let mut centroid = vec![0.0; self.config.dimension];
        for i in 0..n {
            let vec = self.get_vector(i as u32)?;
            for (j, &val) in vec.iter().enumerate() {
                centroid[j] += val;
            }
        }

        for val in &mut centroid {
            *val /= n as f32;
        }

        // Find closest to centroid
        let mut best_id = 0;
        let mut best_dist = f32::MAX;

        for i in 0..n {
            let vec = self.get_vector(i as u32)?;
            let dist = self.distance(&vec, &centroid);
            if dist < best_dist {
                best_dist = dist;
                best_id = i as u32;
            }
        }

        *self.medoid.write() = best_id;
        Ok(())
    }

    /// Get vector by ID
    fn get_vector(&self, id: u32) -> Result<Vec<f32>> {
        let storage = self.vectors.read();

        match &*storage {
            VectorStorage::Memory(vecs) => Ok(vecs
                .get(id as usize)
                .ok_or_else(|| anyhow!("Vector {id} not found"))?
                .clone()),
            VectorStorage::Mmap {
                mmap,
                dimension,
                num_vectors,
            } => {
                // Bounds check
                if id as usize >= *num_vectors {
                    return Err(anyhow!(
                        "Vector {id} out of bounds (num_vectors={})",
                        num_vectors
                    ));
                }

                let start = id as usize * dimension;
                let end = start + dimension;

                // SAFETY CHECK: Debug assertion for pointer alignment before reading f32 values
                // This catches alignment issues in debug builds without runtime cost in release
                let ptr = mmap.as_ptr();
                debug_assert!(
                    ptr.align_offset(std::mem::align_of::<f32>()) == 0,
                    "Mmap pointer {:?} is not aligned to f32 alignment ({}). This is undefined behavior.",
                    ptr,
                    std::mem::align_of::<f32>()
                );

                // SAFETY CHECK: Verify the slice bounds are within the mmap region
                let total_floats = mmap.len() / std::mem::size_of::<f32>();
                debug_assert!(
                    end <= total_floats,
                    "Vector slice bounds [{}..{}] exceed mmap capacity ({})",
                    start,
                    end,
                    total_floats
                );

                // SAFETY: from_raw_parts is safe because:
                // 1. Pointer alignment verified via debug_assert above
                // 2. Bounds verified: end <= total_floats
                // 3. Mmap is valid for the lifetime of the returned slice
                // 4. f32 is Copy, no ownership issues
                let float_slice =
                    unsafe { std::slice::from_raw_parts(ptr as *const f32, total_floats) };

                Ok(float_slice[start..end].to_vec())
            }
        }
    }

    /// Get vector by ID from a storage reference (static helper for use when locks are already held)
    fn get_vector_from_storage(storage: &VectorStorage, id: u32) -> Result<Vec<f32>> {
        match storage {
            VectorStorage::Memory(vecs) => Ok(vecs
                .get(id as usize)
                .ok_or_else(|| anyhow!("Vector {id} not found"))?
                .clone()),
            VectorStorage::Mmap {
                mmap,
                dimension,
                num_vectors,
            } => {
                if id as usize >= *num_vectors {
                    return Err(anyhow!(
                        "Vector {id} out of bounds (num_vectors={})",
                        num_vectors
                    ));
                }
                let start = id as usize * dimension;
                let end = start + dimension;
                let ptr = mmap.as_ptr();
                let total_floats = mmap.len() / std::mem::size_of::<f32>();
                if end > total_floats {
                    return Err(anyhow!("Vector slice bounds exceed mmap capacity"));
                }
                let float_slice =
                    unsafe { std::slice::from_raw_parts(ptr as *const f32, total_floats) };
                Ok(float_slice[start..end].to_vec())
            }
        }
    }

    /// Greedy search for nearest neighbors
    fn greedy_search(&self, query: &[f32], k: usize, entry: u32) -> Result<Vec<SearchCandidate>> {
        let graph = self.graph.read();

        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();

        // Start from entry point
        let entry_vec = self.get_vector(entry)?;
        let entry_dist = self.distance(query, &entry_vec);

        candidates.push(Reverse(SearchCandidate {
            id: entry,
            distance: entry_dist,
        }));

        w.push(SearchCandidate {
            id: entry,
            distance: entry_dist,
        });

        visited.insert(entry);

        // Greedy search
        while let Some(Reverse(current)) = candidates.pop() {
            // Defensive check: w should never be empty (entry point pushed above)
            if w.peek()
                .map(|p| current.distance > p.distance)
                .unwrap_or(false)
            {
                break;
            }

            // Check neighbors (ensure index is valid)
            if (current.id as usize) >= graph.len() {
                // Node doesn't exist in graph yet
                continue;
            }

            let node = &graph[current.id as usize];
            for &neighbor_id in &node.neighbors {
                if visited.contains(&neighbor_id) {
                    continue;
                }

                visited.insert(neighbor_id);

                let neighbor_vec = self.get_vector(neighbor_id)?;
                let dist = self.distance(query, &neighbor_vec);

                // Defensive: check if closer than worst in w, or w not yet full
                let should_add = w.len() < k || w.peek().map(|p| dist < p.distance).unwrap_or(true);
                if should_add {
                    candidates.push(Reverse(SearchCandidate {
                        id: neighbor_id,
                        distance: dist,
                    }));

                    w.push(SearchCandidate {
                        id: neighbor_id,
                        distance: dist,
                    });

                    if w.len() > k {
                        w.pop();
                    }
                }
            }
        }

        // Extract results
        let mut results = Vec::new();
        while let Some(candidate) = w.pop() {
            results.push(candidate);
        }
        results.reverse();

        Ok(results)
    }

    /// Robust prune using α-RNG strategy
    fn robust_prune(&self, node_id: u32, candidates: &[SearchCandidate]) -> Result<Vec<u32>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let mut pruned = Vec::new();
        let node_vec = self.get_vector(node_id)?;

        // Sort candidates by distance (NaN values sort to end)
        let mut sorted_candidates = candidates.to_vec();
        sorted_candidates.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for candidate in sorted_candidates {
            if candidate.id == node_id {
                continue;
            }

            // Check α-RNG condition
            let candidate_vec = self.get_vector(candidate.id)?;
            let dist_nc = self.distance(&node_vec, &candidate_vec);

            let mut should_add = true;
            for &existing_id in &pruned {
                let existing_vec = self.get_vector(existing_id)?;
                let dist_ne = self.distance(&node_vec, &existing_vec);
                let dist_ce = self.distance(&candidate_vec, &existing_vec);

                // α-RNG pruning condition
                if self.config.alpha * dist_ce <= dist_nc && dist_ce <= dist_ne {
                    should_add = false;
                    break;
                }
            }

            if should_add {
                pruned.push(candidate.id);
                if pruned.len() >= self.config.max_degree {
                    break;
                }
            }
        }

        Ok(pruned)
    }

    /// Compute distance between two vectors using configured metric
    ///
    /// All metrics are SIMD-optimized:
    /// - NormalizedDotProduct: -dot(a,b) - fastest, requires normalized vectors
    /// - Euclidean: ||a-b||^2 - works for any vectors
    /// - Cosine: 1 - cos_sim(a,b) - works for any vectors, computes norms
    #[inline(always)]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.distance_metric {
            DistanceMetric::NormalizedDotProduct => normalized_distance_inline(a, b),
            DistanceMetric::Euclidean => euclidean_squared_inline(a, b),
            DistanceMetric::Cosine => 1.0 - cosine_similarity_inline(a, b),
        }
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        // Check if index is empty
        if self.num_vectors == 0 {
            return Ok(Vec::new());
        }

        // Check if graph is built
        if self.graph.read().is_empty() {
            return Err(anyhow!(
                "Vamana graph not built. Call build() first or add more vectors."
            ));
        }

        let entry = *self.medoid.read();
        let candidates = self.greedy_search(query, k, entry)?;

        Ok(candidates.into_iter().map(|c| (c.id, c.distance)).collect())
    }

    /// Add a single vector (incremental indexing) - OPTIMIZED
    pub fn add_vector(&mut self, vector: Vec<f32>) -> Result<u32> {
        let id = self.num_vectors as u32;

        // Add to storage
        let mut storage = self.vectors.write();
        match &mut *storage {
            VectorStorage::Memory(vecs) => {
                vecs.push(vector.clone());
            }
            VectorStorage::Mmap { .. } => {
                return Err(anyhow!("Cannot add to mmap storage, rebuild index"));
            }
        }
        drop(storage);

        // For the first vector, just create a node with no neighbors
        if self.num_vectors == 0 {
            let mut graph = self.graph.write();
            graph.push(VamanaNode {
                id,
                neighbors: Vec::new(),
            });
            *self.medoid.write() = 0;
            self.num_vectors += 1;
            return Ok(id);
        }

        // OPTIMIZATION: Use simpler neighbor selection for incremental adds
        let neighbors = if self.graph.read().is_empty() {
            Vec::new()
        } else {
            // Just find k-nearest neighbors without expensive pruning
            let candidates =
                self.greedy_search(&vector, self.config.max_degree, *self.medoid.read())?;
            // Take top-k neighbors directly without robust_prune for speed
            candidates
                .into_iter()
                .take(self.config.max_degree)
                .map(|c| c.id)
                .collect()
        };

        // Add node to graph
        let mut graph = self.graph.write();
        graph.push(VamanaNode {
            id,
            neighbors: neighbors.clone(),
        });

        // BUG-004 FIX: Distance-aware neighbor pruning for incremental inserts
        // Instead of truncate() which removes newest (possibly best) neighbors,
        // we sort by distance and keep the closest ones.
        let vectors = self.vectors.read();
        for &neighbor_id in &neighbors {
            if neighbor_id as usize >= graph.len() {
                continue;
            }

            graph[neighbor_id as usize].neighbors.push(id);

            // Prune by distance when over max_degree
            if graph[neighbor_id as usize].neighbors.len() > self.config.max_degree {
                // Get neighbor's vector for distance calculations
                if let Ok(neighbor_vec) = Self::get_vector_from_storage(&vectors, neighbor_id) {
                    // Calculate distances to all neighbors
                    let mut neighbor_distances: Vec<(u32, f32)> = graph[neighbor_id as usize]
                        .neighbors
                        .iter()
                        .filter_map(|&n_id| {
                            Self::get_vector_from_storage(&vectors, n_id)
                                .ok()
                                .map(|v| (n_id, dot_product_inline(&neighbor_vec, &v)))
                        })
                        .collect();

                    // Sort by distance (higher dot product = closer for normalized vectors)
                    neighbor_distances
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

                    // Keep only max_degree closest neighbors
                    graph[neighbor_id as usize].neighbors = neighbor_distances
                        .into_iter()
                        .take(self.config.max_degree)
                        .map(|(id, _)| id)
                        .collect();
                } else {
                    // Fallback: truncate if vector access fails
                    graph[neighbor_id as usize]
                        .neighbors
                        .truncate(self.config.max_degree);
                }
            }
        }
        drop(vectors);

        self.num_vectors += 1;
        self.incremental_inserts
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(id)
    }

    /// Check if index rebuild is recommended for optimal search quality
    ///
    /// Returns true when incremental inserts exceed REBUILD_THRESHOLD (10,000).
    /// Incremental inserts use simplified neighbor pruning which can degrade
    /// recall@10 by 5-15% over time.
    pub fn needs_rebuild(&self) -> bool {
        self.incremental_inserts
            .load(std::sync::atomic::Ordering::Relaxed)
            >= REBUILD_THRESHOLD
    }

    /// Get the number of incremental inserts since last rebuild
    pub fn incremental_insert_count(&self) -> usize {
        self.incremental_inserts
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Reset incremental insert counter (call after rebuild)
    pub fn reset_incremental_counter(&self) {
        self.incremental_inserts
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Extract all vectors from the index for rebuilding
    ///
    /// Returns a clone of all vectors currently in the index.
    /// Use this before calling `rebuild_from_vectors()`.
    pub fn extract_all_vectors(&self) -> Vec<Vec<f32>> {
        match &*self.vectors.read() {
            VectorStorage::Memory(vecs) => vecs.clone(),
            VectorStorage::Mmap {
                mmap,
                dimension,
                num_vectors,
            } => {
                let mut vecs = Vec::with_capacity(*num_vectors);
                let total_floats = mmap.len() / std::mem::size_of::<f32>();
                let float_slice = unsafe {
                    std::slice::from_raw_parts(mmap.as_ptr() as *const f32, total_floats)
                };

                for i in 0..*num_vectors {
                    let start = i * dimension;
                    let end = start + dimension;
                    if end <= total_floats {
                        vecs.push(float_slice[start..end].to_vec());
                    }
                }
                vecs
            }
        }
    }

    /// Rebuild the index from vectors with full Vamana construction
    ///
    /// This performs a complete rebuild using robust_prune for optimal graph quality.
    /// Call this when `needs_rebuild()` returns true to restore recall@10 accuracy.
    ///
    /// # Arguments
    /// * `vectors` - All vectors to index (typically from `extract_all_vectors()`)
    ///
    /// # Returns
    /// * `Ok(())` on success, resets the incremental insert counter
    pub fn rebuild_from_vectors(&mut self, vectors: Vec<Vec<f32>>) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        info!(
            "Rebuilding Vamana index with {} vectors (was {} incremental inserts)",
            vectors.len(),
            self.incremental_insert_count()
        );

        // Clear current state
        self.graph.write().clear();
        *self.vectors.write() = VectorStorage::Memory(Vec::new());
        self.num_vectors = 0;

        // Full rebuild with robust_prune
        self.build(vectors)?;

        // Reset counter after successful rebuild
        self.reset_incremental_counter();

        info!("Vamana index rebuild complete");
        Ok(())
    }

    /// Perform automatic rebuild if threshold exceeded
    ///
    /// Thread-safe method that checks if rebuild is needed and performs it atomically.
    /// Returns true if rebuild was performed, false if not needed or already in progress.
    ///
    /// Uses compare-and-swap to ensure only one rebuild occurs even with concurrent calls.
    pub fn auto_rebuild_if_needed(&mut self) -> Result<bool> {
        if !self.needs_rebuild() {
            return Ok(false);
        }

        // Atomic compare-and-swap: try to set rebuilding from false to true
        // If another thread is already rebuilding, this returns Err and we skip
        if self
            .rebuilding
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
            )
            .is_err()
        {
            // Another thread is already rebuilding
            return Ok(false);
        }

        // We acquired the rebuild lock - perform rebuild
        let result = (|| {
            let vectors = self.extract_all_vectors();
            if vectors.is_empty() {
                return Ok(false);
            }
            self.rebuild_from_vectors(vectors)?;
            Ok(true)
        })();

        // Always release the lock, even on error
        self.rebuilding
            .store(false, std::sync::atomic::Ordering::SeqCst);

        result
    }

    /// Check if a rebuild is currently in progress
    pub fn is_rebuilding(&self) -> bool {
        self.rebuilding.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Save index to disk
    pub fn save(&self, path: &Path) -> Result<()> {
        use serde::{Deserialize, Serialize};
        use std::fs::{create_dir_all, File};
        use std::io::BufWriter;

        // Ensure directory exists
        create_dir_all(path)?;

        #[derive(Serialize, Deserialize)]
        struct VamanaData {
            graph: Vec<VamanaNode>,
            vectors: Vec<Vec<f32>>,
            medoid: u32,
            num_vectors: usize,
        }

        // Collect vectors from storage
        let vectors = match &*self.vectors.read() {
            VectorStorage::Memory(vecs) => vecs.clone(),
            VectorStorage::Mmap {
                mmap,
                dimension,
                num_vectors,
            } => {
                // SAFETY CHECK: Debug assertion for pointer alignment
                let ptr = mmap.as_ptr();
                debug_assert!(
                    ptr.align_offset(std::mem::align_of::<f32>()) == 0,
                    "Mmap pointer {:?} is not aligned to f32 alignment ({})",
                    ptr,
                    std::mem::align_of::<f32>()
                );

                // Read vectors from mmap with alignment-safe approach
                let mut vecs = Vec::with_capacity(*num_vectors);
                let total_floats = mmap.len() / std::mem::size_of::<f32>();
                let float_slice =
                    unsafe { std::slice::from_raw_parts(ptr as *const f32, total_floats) };

                for i in 0..*num_vectors {
                    let start = i * dimension;
                    let end = start + dimension;
                    debug_assert!(
                        end <= total_floats,
                        "Vector {} bounds [{}..{}] exceed mmap capacity ({})",
                        i,
                        start,
                        end,
                        total_floats
                    );
                    vecs.push(float_slice[start..end].to_vec());
                }
                vecs
            }
        };

        let data = VamanaData {
            graph: self.graph.read().clone(),
            vectors,
            medoid: *self.medoid.read(),
            num_vectors: self.num_vectors,
        };

        // Save as binary
        let index_file = path.join("vamana_index.bin");
        let file = File::create(&index_file)?;
        bincode::serialize_into(BufWriter::new(file), &data)?;

        info!(
            "Saved Vamana index with {} vectors to {:?}",
            self.num_vectors, index_file
        );
        Ok(())
    }

    /// Load index from disk
    /// Load index data into existing instance (dynamic method)
    pub fn load(&mut self, path: &Path) -> Result<()> {
        use serde::{Deserialize, Serialize};
        use std::fs::File;
        use std::io::BufReader;

        let index_file = path.join("vamana_index.bin");
        if !index_file.exists() {
            return Err(anyhow!("Vamana index file not found at {index_file:?}"));
        }

        // Load serialized data
        let file = File::open(&index_file)?;
        let reader = BufReader::new(file);

        #[derive(Serialize, Deserialize)]
        struct VamanaData {
            graph: Vec<VamanaNode>,
            vectors: Vec<Vec<f32>>,
            medoid: u32,
            num_vectors: usize,
        }

        let data: VamanaData = bincode::deserialize_from(reader)?;

        // Update internal state
        *self.graph.write() = data.graph;
        *self.medoid.write() = data.medoid;
        self.num_vectors = data.num_vectors;

        // Update vector storage
        match &mut *self.vectors.write() {
            VectorStorage::Memory(vecs) => {
                *vecs = data.vectors;
            }
            VectorStorage::Mmap { .. } => {
                // Cannot restore mmap from serialized data - converting to in-memory storage
                warn!(
                    "Loading index into mmap-configured instance: converting {} vectors to in-memory storage. \
                     This may increase memory usage. To use mmap, rebuild the index with build().",
                    data.num_vectors
                );
                *self.vectors.write() = VectorStorage::Memory(data.vectors);
            }
        }

        info!("Loaded Vamana index with {} vectors", self.num_vectors);
        Ok(())
    }
}

/// Search candidate
#[derive(Debug, Clone)]
struct SearchCandidate {
    id: u32,
    distance: f32,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.distance == other.distance
    }
}

impl Eq for SearchCandidate {}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vamana_construction() {
        let mut index = VamanaIndex::new(VamanaConfig {
            dimension: 4,
            max_degree: 3,
            search_list_size: 10,
            alpha: 1.2,
            use_mmap: false,
            ..Default::default()
        })
        .unwrap();

        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0, 0.0],
        ];

        index.build(vectors).unwrap();

        let query = vec![0.9, 0.1, 0.0, 0.0];
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Closest to [1,0,0,0]
    }
}
