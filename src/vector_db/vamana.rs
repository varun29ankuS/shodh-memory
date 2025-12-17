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

/// Threshold for recommending index rebuild based on deletion ratio
/// When 30% or more of vectors are soft-deleted, compaction is recommended
pub const DELETION_RATIO_THRESHOLD: f32 = 0.30;

/// Main Vamana index
pub struct VamanaIndex {
    pub(crate) config: VamanaConfig,

    /// Graph structure: node_id -> neighbors
    pub(crate) graph: Arc<RwLock<Vec<VamanaNode>>>,

    /// Vectors (can be memory-mapped)
    pub(crate) vectors: Arc<RwLock<VectorStorage>>,

    /// Medoid/centroid as entry point
    pub(crate) medoid: Arc<RwLock<u32>>,

    /// Number of vectors (atomic for lock-free reads during background rebuild)
    pub(crate) num_vectors: std::sync::atomic::AtomicUsize,

    /// Storage path for mmap files (unique per index instance)
    storage_path: Option<PathBuf>,

    /// Counter for incremental inserts since last rebuild
    /// Used to track index quality degradation
    incremental_inserts: std::sync::atomic::AtomicUsize,

    /// Flag to prevent concurrent rebuilds
    rebuilding: std::sync::atomic::AtomicBool,

    /// Soft-deleted vector IDs (filtered from search results)
    /// These vectors remain in the graph but are excluded from results.
    /// Physically removed on next rebuild.
    deleted_ids: Arc<RwLock<HashSet<u32>>>,
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

impl Default for VectorStorage {
    fn default() -> Self {
        VectorStorage::Memory(Vec::new())
    }
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
            num_vectors: std::sync::atomic::AtomicUsize::new(0),
            storage_path,
            incremental_inserts: std::sync::atomic::AtomicUsize::new(0),
            rebuilding: std::sync::atomic::AtomicBool::new(false),
            deleted_ids: Arc::new(RwLock::new(HashSet::new())),
        })
    }

    /// Get number of vectors in the index
    pub fn len(&self) -> usize {
        self.num_vectors.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.num_vectors.load(std::sync::atomic::Ordering::Acquire) == 0
    }

    /// Build index from vectors using Vamana algorithm
    pub fn build(&mut self, vectors: Vec<Vec<f32>>) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        let n = vectors.len();
        self.num_vectors
            .store(n, std::sync::atomic::Ordering::Release);

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
        let n = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
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

    /// Get vector slice by ID from storage reference (zero-copy, no allocation)
    ///
    /// This is the performance-critical path for search operations.
    /// Returns a borrowed slice instead of cloning the vector data.
    #[inline]
    fn get_slice_from_storage(storage: &VectorStorage, id: u32) -> Result<&[f32]> {
        match storage {
            VectorStorage::Memory(vecs) => vecs
                .get(id as usize)
                .map(|v| v.as_slice())
                .ok_or_else(|| anyhow!("Vector {id} not found")),
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
                // SAFETY: Pointer alignment verified during store_vectors().
                // Bounds checked above. Mmap lifetime outlives returned slice.
                let float_slice =
                    unsafe { std::slice::from_raw_parts(ptr as *const f32, total_floats) };
                Ok(&float_slice[start..end])
            }
        }
    }

    /// Greedy search for nearest neighbors
    ///
    /// Optimized to use zero-copy slice access for vector data.
    /// Holds both graph and vector storage locks for the duration of the search
    /// to avoid per-neighbor lock acquisition overhead.
    fn greedy_search(&self, query: &[f32], k: usize, entry: u32) -> Result<Vec<SearchCandidate>> {
        let graph = self.graph.read();
        let storage = self.vectors.read(); // Hold lock for entire search (zero-copy access)

        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();

        // Start from entry point (zero-copy slice access)
        let entry_slice = Self::get_slice_from_storage(&storage, entry)?;
        let entry_dist = self.distance(query, entry_slice);

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

                // Zero-copy slice access - no allocation per neighbor
                let neighbor_slice = Self::get_slice_from_storage(&storage, neighbor_id)?;
                let dist = self.distance(query, neighbor_slice);

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
    ///
    /// Optimized with:
    /// - Zero-copy slice access for vector data
    /// - Cached dist_ne (node to existing) to avoid O(n²) distance recomputation
    /// - Pre-loaded candidate vectors to minimize storage lookups
    fn robust_prune(&self, node_id: u32, candidates: &[SearchCandidate]) -> Result<Vec<u32>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let storage = self.vectors.read(); // Hold lock for entire prune operation
        let node_slice = Self::get_slice_from_storage(&storage, node_id)?;

        // Sort candidates by distance (NaN values sort to end)
        let mut sorted_candidates = candidates.to_vec();
        sorted_candidates.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Pre-load all candidate vectors to avoid repeated storage lookups
        // This trades memory for CPU - worth it for the O(n²) inner loop
        let candidate_vectors: Vec<_> = sorted_candidates
            .iter()
            .filter_map(|c| {
                if c.id == node_id {
                    None
                } else {
                    Self::get_slice_from_storage(&storage, c.id)
                        .ok()
                        .map(|slice| (c.id, slice.to_vec(), c.distance))
                }
            })
            .collect();

        let mut pruned_ids = Vec::with_capacity(self.config.max_degree);
        // Cache dist_ne (distance from node to each pruned neighbor)
        // When we add candidate C to pruned, dist_nc becomes the dist_ne for C
        let mut pruned_dist_ne: Vec<f32> = Vec::with_capacity(self.config.max_degree);
        // Cache existing vectors for O(1) access in inner loop
        let mut pruned_vectors: Vec<&[f32]> = Vec::with_capacity(self.config.max_degree);

        for (candidate_id, candidate_vec, _candidate_dist) in &candidate_vectors {
            let dist_nc = self.distance(node_slice, candidate_vec);

            let mut should_add = true;
            for i in 0..pruned_ids.len() {
                let dist_ne = pruned_dist_ne[i]; // Cached - no recomputation!
                let dist_ce = self.distance(candidate_vec, pruned_vectors[i]);

                // α-RNG pruning condition
                if self.config.alpha * dist_ce <= dist_nc && dist_ce <= dist_ne {
                    should_add = false;
                    break;
                }
            }

            if should_add {
                pruned_ids.push(*candidate_id);
                // dist_nc is the distance from node to this candidate
                // It becomes dist_ne when this candidate is used as "existing" in future iterations
                pruned_dist_ne.push(dist_nc);
                pruned_vectors.push(candidate_vec);
                if pruned_ids.len() >= self.config.max_degree {
                    break;
                }
            }
        }

        Ok(pruned_ids)
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

    /// Search for k nearest neighbors (excludes soft-deleted vectors)
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        // Check if index is empty
        if self.num_vectors.load(std::sync::atomic::Ordering::Acquire) == 0 {
            return Ok(Vec::new());
        }

        // Check if graph is built
        if self.graph.read().is_empty() {
            return Err(anyhow!(
                "Vamana graph not built. Call build() first or add more vectors."
            ));
        }

        let entry = *self.medoid.read();
        let deleted = self.deleted_ids.read();
        let deleted_count = deleted.len();

        // Request extra candidates to account for deleted vectors
        let search_k = if deleted_count > 0 {
            k + deleted_count.min(k * 2)
        } else {
            k
        };

        let candidates = self.greedy_search(query, search_k, entry)?;

        // Filter out deleted vectors and take k results
        let results: Vec<(u32, f32)> = candidates
            .into_iter()
            .filter(|c| !deleted.contains(&c.id))
            .take(k)
            .map(|c| (c.id, c.distance))
            .collect();

        Ok(results)
    }

    /// Mark a vector as deleted (soft delete)
    /// The vector remains in the graph but is excluded from search results.
    /// It will be physically removed on the next rebuild.
    pub fn mark_deleted(&self, vector_id: u32) -> bool {
        if (vector_id as usize) < self.num_vectors.load(std::sync::atomic::Ordering::Acquire) {
            self.deleted_ids.write().insert(vector_id);
            true
        } else {
            false
        }
    }

    /// Check if a vector is marked as deleted
    pub fn is_deleted(&self, vector_id: u32) -> bool {
        self.deleted_ids.read().contains(&vector_id)
    }

    /// Get the number of soft-deleted vectors
    pub fn deleted_count(&self) -> usize {
        self.deleted_ids.read().len()
    }

    /// Get the deletion ratio (deleted / total vectors)
    /// Returns 0.0 if index is empty
    pub fn deletion_ratio(&self) -> f32 {
        let n = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        if n == 0 {
            return 0.0;
        }
        self.deleted_count() as f32 / n as f32
    }

    /// Check if compaction is needed based on deletion ratio
    pub fn needs_compaction(&self) -> bool {
        self.deletion_ratio() >= DELETION_RATIO_THRESHOLD
    }

    /// Clear all deleted markers (use after rebuild)
    pub fn clear_deleted(&self) {
        self.deleted_ids.write().clear();
    }

    /// Add a single vector (incremental indexing) - OPTIMIZED
    pub fn add_vector(&mut self, vector: Vec<f32>) -> Result<u32> {
        let current_count = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        let id = current_count as u32;

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
        if current_count == 0 {
            let mut graph = self.graph.write();
            graph.push(VamanaNode {
                id,
                neighbors: Vec::new(),
            });
            *self.medoid.write() = 0;
            self.num_vectors
                .fetch_add(1, std::sync::atomic::Ordering::Release);
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

        self.num_vectors
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        self.incremental_inserts
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(id)
    }

    /// Check if index rebuild is recommended for optimal search quality
    ///
    /// Returns true when:
    /// - Incremental inserts exceed REBUILD_THRESHOLD (10,000), OR
    /// - Deletion ratio exceeds DELETION_RATIO_THRESHOLD (30%)
    ///
    /// Incremental inserts use simplified neighbor pruning which can degrade
    /// recall@10 by 5-15% over time. High deletion ratios waste memory and
    /// slow down search (must filter more orphaned entries).
    pub fn needs_rebuild(&self) -> bool {
        let needs_insert_rebuild = self
            .incremental_inserts
            .load(std::sync::atomic::Ordering::Relaxed)
            >= REBUILD_THRESHOLD;
        let needs_compaction = self.needs_compaction();

        needs_insert_rebuild || needs_compaction
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

    /// Extract only live (non-deleted) vectors for compaction rebuild
    ///
    /// Returns vectors that are NOT marked as deleted.
    /// Use this for compaction to physically remove deleted vectors.
    pub fn extract_live_vectors(&self) -> Vec<Vec<f32>> {
        let deleted = self.deleted_ids.read();
        match &*self.vectors.read() {
            VectorStorage::Memory(vecs) => vecs
                .iter()
                .enumerate()
                .filter(|(i, _)| !deleted.contains(&(*i as u32)))
                .map(|(_, v)| v.clone())
                .collect(),
            VectorStorage::Mmap {
                mmap,
                dimension,
                num_vectors,
            } => {
                let total_floats = mmap.len() / std::mem::size_of::<f32>();
                let float_slice = unsafe {
                    std::slice::from_raw_parts(mmap.as_ptr() as *const f32, total_floats)
                };

                let mut vecs = Vec::with_capacity(num_vectors - deleted.len());
                for i in 0..*num_vectors {
                    if deleted.contains(&(i as u32)) {
                        continue;
                    }
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
        self.num_vectors
            .store(0, std::sync::atomic::Ordering::Release);

        // Full rebuild with robust_prune
        self.build(vectors)?;

        // Reset counter after successful rebuild
        self.reset_incremental_counter();

        info!("Vamana index rebuild complete");
        Ok(())
    }

    /// Perform automatic rebuild if threshold exceeded (non-blocking)
    ///
    /// Thread-safe method that checks if rebuild is needed and performs it without
    /// blocking concurrent reads or writes. Uses a background build followed by
    /// atomic swap of index internals.
    ///
    /// Returns true if rebuild was performed, false if not needed or already in progress.
    ///
    /// ## Concurrency Model
    ///
    /// - **Reads**: Continue uninterrupted on the old index during rebuild
    /// - **Writes**: Continue on the old index but will be lost when swap occurs
    /// - **Swap**: Brief write locks acquired only during the final atomic swap
    ///
    /// Uses compare-and-swap to ensure only one rebuild occurs even with concurrent calls.
    /// Compacts deleted vectors by extracting only live vectors.
    ///
    /// ## Note on Write Handling
    ///
    /// Any vectors added between `extract_live_vectors()` and the final swap will be
    /// lost. This is acceptable for periodic maintenance rebuilds. For write-intensive
    /// workloads, consider using `rebuild_from_vectors()` which takes `&mut self` and
    /// blocks writes during rebuild.
    pub fn auto_rebuild_if_needed(&self) -> Result<bool> {
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

        // Log reason for rebuild
        let deleted_count = self.deleted_count();
        let deletion_ratio = self.deletion_ratio();
        let total_vectors = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        if deletion_ratio >= DELETION_RATIO_THRESHOLD {
            info!(
                "Compacting index: {} deleted vectors ({:.1}% of {})",
                deleted_count,
                deletion_ratio * 100.0,
                total_vectors
            );
        }

        // We acquired the rebuild lock - perform background rebuild with atomic swap
        let result = (|| {
            // 1. Extract live vectors (read-only, doesn't block writes)
            let vectors = self.extract_live_vectors();
            let compacted = deleted_count;
            if vectors.is_empty() {
                self.clear_deleted();
                return Ok(false);
            }

            info!(
                "Background rebuilding Vamana index with {} vectors (was {} incremental inserts)",
                vectors.len(),
                self.incremental_insert_count()
            );

            // 2. Build completely new index (expensive, but doesn't hold any locks on self)
            let config = self.config.clone();
            let mut new_index = VamanaIndex::new(config)?;
            new_index.build(vectors)?;

            // 3. Atomic swap - acquire all write locks briefly
            // Lock ordering: graph -> vectors -> medoid (consistent with struct field order)
            {
                let mut old_graph = self.graph.write();
                let mut old_vectors = self.vectors.write();
                let mut old_medoid = self.medoid.write();

                // Swap graph
                let new_graph = std::mem::take(&mut *new_index.graph.write());
                *old_graph = new_graph;

                // Swap vectors
                let new_vectors = std::mem::take(&mut *new_index.vectors.write());
                *old_vectors = new_vectors;

                // Swap medoid
                *old_medoid = *new_index.medoid.read();
            }

            // Update num_vectors atomically (after releasing locks)
            self.num_vectors.store(
                new_index
                    .num_vectors
                    .load(std::sync::atomic::Ordering::Acquire),
                std::sync::atomic::Ordering::Release,
            );

            // Clear deleted markers and reset counter
            self.clear_deleted();
            self.reset_incremental_counter();

            if compacted > 0 {
                info!("Compaction complete: removed {} deleted vectors", compacted);
            }
            info!("Background Vamana index rebuild complete");

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

        let num_vecs = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        let data = VamanaData {
            graph: self.graph.read().clone(),
            vectors,
            medoid: *self.medoid.read(),
            num_vectors: num_vecs,
        };

        // Save as binary
        let index_file = path.join("vamana_index.bin");
        let file = File::create(&index_file)?;
        bincode::serialize_into(BufWriter::new(file), &data)?;

        info!(
            "Saved Vamana index with {} vectors to {:?}",
            num_vecs, index_file
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
        self.num_vectors
            .store(data.num_vectors, std::sync::atomic::Ordering::Release);

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

        info!("Loaded Vamana index with {} vectors", data.num_vectors);
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
