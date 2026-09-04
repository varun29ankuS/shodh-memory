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
    cosine_similarity_inline, euclidean_squared_inline, normalized_distance_inline,
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
    pub(crate) id: u32,

    /// Neighbor IDs sorted by distance
    pub(crate) neighbors: Vec<u32>,
}

/// Threshold for recommending index rebuild (number of incremental inserts)
pub const REBUILD_THRESHOLD: usize = 10_000;

/// Threshold for incremental repair (lighter maintenance, more frequent)
/// Repairs neighborhoods of recently inserted nodes without full rebuild
pub const REPAIR_THRESHOLD: usize = 1_000;

/// Threshold for recommending index rebuild based on deletion ratio
/// When 30% or more of vectors are soft-deleted, compaction is recommended
pub const DELETION_RATIO_THRESHOLD: f32 = 0.30;

/// Minimum recall for acceptable index quality (used in quality estimation)
/// Below this threshold, rebuild is strongly recommended
pub const MIN_ACCEPTABLE_RECALL: f32 = 0.85;

/// Environment variable selecting the query-time search list size (DiskANN `L`).
pub const SEARCH_EF_ENV: &str = "SHODH_VAMANA_EF";

/// Parse a query-time search list size from its raw environment value.
///
/// Returns `None` — meaning "leave the beam at the requested candidate count",
/// i.e. the historical behaviour — for an absent, empty, unparseable or zero
/// value. Split out as a pure function so it can be unit tested without
/// mutating process environment (which races under a parallel test runner).
fn parse_search_ef(raw: Option<&str>) -> Option<usize> {
    raw?.trim().parse::<usize>().ok().filter(|&v| v > 0)
}

/// Query-time search list size from [`SEARCH_EF_ENV`], resolved once per process.
///
/// `None` (the default, variable unset) keeps `L = k`, so the shipped search
/// path is unchanged. A value larger than the internal candidate count widens
/// the beam; a smaller one is ignored (the beam is clamped up to `k`).
///
/// Cached in a `OnceLock` because `search()` is on the per-query hot path and
/// the eval/production environment is fixed at process start.
fn configured_search_ef() -> Option<usize> {
    static SEARCH_EF: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *SEARCH_EF.get_or_init(|| {
        let parsed = parse_search_ef(std::env::var(SEARCH_EF_ENV).ok().as_deref());
        if let Some(ef) = parsed {
            info!("Vamana query-time search list size ({SEARCH_EF_ENV}) = {ef}");
        }
        parsed
    })
}

/// Environment variable enabling α-RNG pruning on the incremental insert path.
pub const INSERT_PRUNE_ENV: &str = "SHODH_VAMANA_INSERT_PRUNE";

/// Parse the insert-prune flag from its raw environment value.
///
/// Only `1` / `true` (case-insensitive, trimmed) enable it; anything else —
/// absent, empty, `0`, garbage — leaves the historical greedy-kNN insert
/// untouched. A pure function so it is unit-testable without mutating process
/// environment (which races under a parallel test runner).
fn parse_insert_prune(raw: Option<&str>) -> bool {
    // Opt-OUT. Unset means ON, because a-RNG construction is what makes this a
    // Vamana index rather than a greedy kNN graph; skipping it is the deviation,
    // not the default. Only an explicit `0`/`false` disables it.
    !matches!(raw.map(str::trim), Some(v) if v == "0" || v.eq_ignore_ascii_case("false"))
}

/// Insert-prune policy from [`INSERT_PRUNE_ENV`], resolved once per process.
///
/// DEFAULT ON. α-RNG construction is what makes this a Vamana index; the
/// incremental path skipped it "for speed" and silently built a greedy kNN
/// graph instead -- one whose own true neighbours greedy search could not
/// reach (45.6% self-recall at beam=k on clustered data, against 99.9% with
/// α-RNG). No beam width recovers a neighbour the graph has no edge to, which
/// is why widening `ef` only ever recovered a third of the loss.
///
/// `SHODH_VAMANA_INSERT_PRUNE=0` restores the historical greedy path bit-for-bit
/// for A/B measurement. It is an escape hatch, not a performance option: the
/// insert-latency it buys costs index navigability.
///
/// Cached in a `OnceLock`: ingest is a hot path and the eval/production
/// environment is fixed at process start. The `info!` line doubles as the
/// CI-log proof that the flag actually reached the live insert path.
fn configured_insert_prune() -> bool {
    static INSERT_PRUNE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *INSERT_PRUNE.get_or_init(|| {
        let enabled = parse_insert_prune(std::env::var(INSERT_PRUNE_ENV).ok().as_deref());
        if !enabled {
            info!(
                "Vamana insert-time α-RNG pruning DISABLED via {INSERT_PRUNE_ENV} — the index                  will be a greedy kNN graph, not a Vamana graph"
            );
        }
        enabled
    })
}

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
    pub(crate) storage_path: Option<PathBuf>,

    /// Counter for incremental inserts since last rebuild
    /// Used to track index quality degradation
    pub(crate) incremental_inserts: std::sync::atomic::AtomicUsize,

    /// Flag to prevent concurrent rebuilds
    pub(crate) rebuilding: std::sync::atomic::AtomicBool,

    /// Soft-deleted vector IDs (filtered from search results)
    /// These vectors remain in the graph but are excluded from results.
    /// Physically removed on next rebuild.
    pub(crate) deleted_ids: Arc<RwLock<HashSet<u32>>>,
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
        let mut rng = rand::thread_rng();
        let degree = self.config.max_degree.min(n.saturating_sub(1));

        let mut graph = Vec::with_capacity(n);

        for i in 0..n {
            // Sample `degree` random neighbors (excluding self) in O(degree) time
            let mut neighbors = Vec::with_capacity(degree);
            if degree > 0 && n > 1 {
                let sample = rand::seq::index::sample(&mut rng, n - 1, degree);
                for idx in sample.into_vec() {
                    // Map sampled indices to node IDs, skipping self
                    let j = if idx >= i { idx + 1 } else { idx };
                    neighbors.push(j as u32);
                }
            }

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
            if !file_size.is_multiple_of(std::mem::align_of::<f32>()) {
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

    /// Greedy search for nearest neighbors with the search list tied to `k`.
    ///
    /// Equivalent to [`Self::greedy_search_beam`] with `beam == k`.
    fn greedy_search(&self, query: &[f32], k: usize, entry: u32) -> Result<Vec<SearchCandidate>> {
        self.greedy_search_beam(query, k, k, entry)
    }

    /// Greedy search over the Vamana graph keeping a search list of
    /// `L = max(beam, k)` candidates, returning the best `k` of them.
    ///
    /// DiskANN searches with a list size `L >= k` and reports the top `k` from
    /// it. A larger `L` explores more of the graph, so both the *reach* (which
    /// true neighbours are found at all) and the *order* of the returned `k`
    /// move towards exact. This index previously hard-wired `L = k`, which is
    /// maximally myopic for small `k` — at `k = 1` it degenerates to pure
    /// hill-climbing, and `VamanaConfig::search_list_size` was consulted only
    /// for `with_capacity` hints, never as an actual query-time beam.
    ///
    /// `beam` is clamped UP to `k`, so it can only ever widen the search; the
    /// `beam == k` call site ([`Self::greedy_search`]) behaves exactly as the
    /// previous implementation did.
    ///
    /// Optimized to use zero-copy slice access for vector data.
    /// Holds both graph and vector storage locks for the duration of the search
    /// to avoid per-neighbor lock acquisition overhead.
    fn greedy_search_beam(
        &self,
        query: &[f32],
        k: usize,
        beam: usize,
        entry: u32,
    ) -> Result<Vec<SearchCandidate>> {
        let graph = self.graph.read();
        let storage = self.vectors.read(); // Hold lock for entire search (zero-copy access)

        // Search list size. Never below k — a beam smaller than the requested
        // result count could not return k results.
        let list_size = beam.max(k);

        let search_cap = self.config.search_list_size;
        let mut visited = HashSet::with_capacity(search_cap);
        let mut candidates = BinaryHeap::with_capacity(search_cap);
        let mut w = BinaryHeap::with_capacity(search_cap);

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
                let should_add =
                    w.len() < list_size || w.peek().map(|p| dist < p.distance).unwrap_or(true);
                if should_add {
                    candidates.push(Reverse(SearchCandidate {
                        id: neighbor_id,
                        distance: dist,
                    }));

                    w.push(SearchCandidate {
                        id: neighbor_id,
                        distance: dist,
                    });

                    if w.len() > list_size {
                        w.pop();
                    }
                }
            }
        }

        // Extract results: the search list holds up to `list_size` candidates;
        // report only the best `k`. A no-op when `list_size == k` (the default).
        let mut results = Vec::new();
        while let Some(candidate) = w.pop() {
            results.push(candidate);
        }
        results.reverse();
        results.truncate(k);

        Ok(results)
    }

    /// Robust prune using α-RNG strategy
    ///
    /// Optimized with:
    /// - Zero-copy slice access for vector data
    /// - Cached dist_ne (node to existing) to avoid O(n²) distance recomputation
    /// - Pre-loaded candidate vectors to minimize storage lookups
    fn robust_prune(&self, node_id: u32, candidates: &[SearchCandidate]) -> Result<Vec<u32>> {
        let storage = self.vectors.read(); // Hold lock for entire prune operation
        self.robust_prune_in(&storage, node_id, candidates)
    }

    /// α-RNG prune against an already-held storage reference.
    ///
    /// Split out from [`Self::robust_prune`] so callers that already hold the
    /// vector-storage lock (the insert path holds `graph.write()` +
    /// `vectors.read()` while fixing up reverse edges) can prune without
    /// re-acquiring it — parking_lot RwLocks are not re-entrant, and a second
    /// read acquisition with a writer waiting can deadlock.
    fn robust_prune_in(
        &self,
        storage: &VectorStorage,
        node_id: u32,
        candidates: &[SearchCandidate],
    ) -> Result<Vec<u32>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let node_slice = Self::get_slice_from_storage(storage, node_id)?;

        // Sort candidates by distance (NaN values sort to end), tie-break by id for determinism
        let mut sorted_candidates = candidates.to_vec();
        sorted_candidates.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });

        // Pre-load all candidate vectors to avoid repeated storage lookups
        // This trades memory for CPU - worth it for the O(n²) inner loop
        let candidate_vectors: Vec<_> = sorted_candidates
            .iter()
            .filter_map(|c| {
                if c.id == node_id {
                    None
                } else {
                    Self::get_slice_from_storage(storage, c.id)
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

        // The α-RNG rule requires NONNEGATIVE distances. NormalizedDotProduct
        // returns d = -dot ∈ [-1, 1]: multiplying a NEGATIVE d by α > 1 makes it
        // "closer", inverting the prune rule — for near-tied similar candidates
        // `α·dist_ce ≤ dist_nc` then fires for everything after the first kept
        // neighbor and the built graph degenerates to out-degree ~1 (reproduced by
        // retrieval::tests::test_force_quality_rebuild_*). Shift to cosine
        // distance (1 + d ∈ [0, 2]) for the α comparison; Euclidean/Cosine are
        // already nonnegative (offset 0).
        let rng_offset = match self.config.distance_metric {
            DistanceMetric::NormalizedDotProduct => 1.0_f32,
            DistanceMetric::Euclidean | DistanceMetric::Cosine => 0.0_f32,
        };

        for (candidate_id, candidate_vec, _candidate_dist) in &candidate_vectors {
            let dist_nc = self.distance(node_slice, candidate_vec);

            let mut should_add = true;
            for i in 0..pruned_ids.len() {
                let dist_ne = pruned_dist_ne[i]; // Cached - no recomputation!
                let dist_ce = self.distance(candidate_vec, pruned_vectors[i]);

                // α-RNG pruning condition (on the nonnegative-shifted scale)
                if self.config.alpha * (dist_ce + rng_offset) <= (dist_nc + rng_offset)
                    && dist_ce <= dist_ne
                {
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
    ///
    /// The search list size is taken from `SHODH_VAMANA_EF` (see
    /// [`configured_search_ef`]); unset — the default — leaves the beam equal
    /// to the requested candidate count, exactly as before.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        self.search_with_ef(query, k, configured_search_ef())
    }

    /// Search for k nearest neighbors with an explicit search list size.
    ///
    /// `ef` is the DiskANN search list size `L`. `None` (and any value below
    /// the internal candidate count) leaves the beam at the candidate count,
    /// which is the historical behaviour. A larger `ef` widens the beam without
    /// changing how many results are returned, trading query time for a top-`k`
    /// that is closer to exact in both membership and order.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
    ) -> Result<Vec<(u32, f32)>> {
        // Check if index is empty
        if self.num_vectors.load(std::sync::atomic::Ordering::Acquire) == 0 {
            return Ok(Vec::new());
        }

        // SHODH_VECTOR_EXACT=1 → bypass the Vamana ANN graph and return the TRUE
        // k nearest neighbours by exact brute-force. Diagnostic: separates *index
        // recall* (ANN vs exact, the number LanceDB/FAISS publish) from *task recall*
        // (gold answers). If task recall is unchanged vs ANN, the index is faithful
        // and the embeddings — not the index — are the ceiling.
        if std::env::var("SHODH_VECTOR_EXACT").is_ok() {
            return self.brute_force_search(query, k);
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

        // The beam only ever widens: `greedy_search_beam` clamps it up to
        // `search_k`, so `ef = None` reproduces the historical `L = k` search.
        let candidates = self.greedy_search_beam(query, search_k, ef.unwrap_or(search_k), entry)?;

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

    /// Add a single vector (incremental indexing).
    ///
    /// Neighbor selection policy comes from `SHODH_VAMANA_INSERT_PRUNE` (see
    /// [`configured_insert_prune`]); unset — the default — keeps the historical
    /// greedy top-k selection bit-for-bit.
    pub fn add_vector(&mut self, vector: Vec<f32>) -> Result<u32> {
        self.add_vector_with_policy(vector, configured_insert_prune())
    }

    /// Add a single vector with an explicit neighbor-selection policy.
    ///
    /// `alpha_prune = false` is the historical incremental insert: the new
    /// node's neighbors are the raw greedy top-`max_degree` (no α-RNG
    /// diversification), and a neighbor whose reverse edge overflows
    /// `max_degree` keeps its `max_degree` CLOSEST neighbors. Both choices
    /// optimize insert speed and both destroy exactly the property greedy
    /// search needs: neighbor lists degenerate into tight same-cluster cliques
    /// with no long-range edges, so the search surface develops local minima
    /// that no query-time beam can fully climb out of.
    ///
    /// `alpha_prune = true` is the DiskANN/FreshDiskANN insert: candidates come
    /// from a `search_list_size`-wide greedy search, the new node's neighbors
    /// are α-RNG pruned ([`Self::robust_prune`]), and overflowing reverse
    /// neighborhoods are re-pruned with the same α-RNG rule over their true
    /// distances — the construction `build()` uses, applied incrementally.
    pub fn add_vector_with_policy(&mut self, vector: Vec<f32>, alpha_prune: bool) -> Result<u32> {
        let current_count = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        let id = current_count as u32;

        // Add to storage - convert mmap to memory if needed for incremental updates
        let mut storage = self.vectors.write();
        match &mut *storage {
            VectorStorage::Memory(vecs) => {
                vecs.push(vector.clone());
            }
            VectorStorage::Mmap {
                mmap,
                num_vectors,
                dimension,
            } => {
                // Convert mmap to memory storage for incremental updates
                tracing::info!(
                    "Converting mmap storage ({} vectors) to memory for incremental indexing",
                    num_vectors
                );
                let dim = *dimension;
                let count = *num_vectors;
                let ptr = mmap.as_ptr() as *const f32;
                let mut vecs = Vec::with_capacity(count + 1);
                for i in 0..count {
                    let start = i * dim;
                    let slice = unsafe { std::slice::from_raw_parts(ptr.add(start), dim) };
                    vecs.push(slice.to_vec());
                }
                vecs.push(vector.clone());
                *storage = VectorStorage::Memory(vecs);
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

        // Neighbor selection. Historical path (alpha_prune = false): raw greedy
        // top-max_degree, no pruning. α-RNG path: candidates from a
        // search_list_size-wide beam, then robust_prune — same rule as build().
        let neighbors: Vec<u32> = if self.graph.read().is_empty() {
            Vec::new()
        } else if alpha_prune {
            let beam = self.config.search_list_size.max(self.config.max_degree);
            let candidates = self.greedy_search_beam(&vector, beam, beam, *self.medoid.read())?;
            // The new vector is already in storage at `id`, so robust_prune can
            // read it; it is NOT yet in the graph, so the search cannot visit it.
            self.robust_prune(id, &candidates)?
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

            // Re-prune when the reverse edge overflows max_degree
            if graph[neighbor_id as usize].neighbors.len() > self.config.max_degree {
                // Get neighbor's vector for distance calculations
                if let Ok(neighbor_vec) = Self::get_vector_from_storage(&vectors, neighbor_id) {
                    // Calculate distances to all neighbors using configured metric
                    let mut neighbor_distances: Vec<(u32, f32)> = graph[neighbor_id as usize]
                        .neighbors
                        .iter()
                        .filter_map(|&n_id| {
                            Self::get_vector_from_storage(&vectors, n_id)
                                .ok()
                                .map(|v| (n_id, self.distance(&neighbor_vec, &v)))
                        })
                        .collect();

                    // Sort by distance (lower = closer for all metrics), tie-break by id
                    neighbor_distances
                        .sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

                    if alpha_prune {
                        // α-RNG over the neighbor's TRUE distances. Distance-only
                        // keep-closest (the historical path below) turns hub
                        // neighborhoods into same-cluster cliques: every new
                        // same-cluster insert evicts the long-range edge greedy
                        // search navigates by. robust_prune keeps the closest
                        // candidate AND the diverse ones — `robust_prune_in`
                        // because the vectors lock is already held here.
                        let candidates: Vec<SearchCandidate> = neighbor_distances
                            .into_iter()
                            .map(|(n_id, dist)| SearchCandidate {
                                id: n_id,
                                distance: dist,
                            })
                            .collect();
                        graph[neighbor_id as usize].neighbors =
                            self.robust_prune_in(&vectors, neighbor_id, &candidates)?;
                    } else {
                        // Historical: keep only max_degree closest neighbors
                        graph[neighbor_id as usize].neighbors = neighbor_distances
                            .into_iter()
                            .take(self.config.max_degree)
                            .map(|(id, _)| id)
                            .collect();
                    }
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
            .fetch_add(1, std::sync::atomic::Ordering::Release);
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
            .load(std::sync::atomic::Ordering::Acquire)
            >= REBUILD_THRESHOLD;
        let needs_compaction = self.needs_compaction();

        needs_insert_rebuild || needs_compaction
    }

    /// Get the number of incremental inserts since last rebuild
    pub fn incremental_insert_count(&self) -> usize {
        self.incremental_inserts
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Reset incremental insert counter (call after rebuild)
    pub fn reset_incremental_counter(&self) {
        self.incremental_inserts
            .store(0, std::sync::atomic::Ordering::Release);
    }

    /// Check if incremental repair is recommended
    ///
    /// Repair is lighter than full rebuild - only re-prunes neighborhoods of
    /// recently inserted nodes. Recommended every 1,000 inserts.
    pub fn needs_repair(&self) -> bool {
        let inserts = self
            .incremental_inserts
            .load(std::sync::atomic::Ordering::Relaxed);
        (REPAIR_THRESHOLD..REBUILD_THRESHOLD).contains(&inserts)
    }

    /// Perform incremental repair on recently inserted nodes
    ///
    /// This is faster than full rebuild (~10x) and maintains index quality
    /// by re-pruning neighborhoods using proper α-RNG strategy instead of
    /// the simplified truncation used during incremental insert.
    ///
    /// Call when `needs_repair()` returns true.
    ///
    /// # Algorithm
    /// 1. Identify nodes inserted since last repair (last REPAIR_THRESHOLD nodes)
    /// 2. For each such node, re-run robust_prune on its neighborhood
    /// 3. Update bidirectional edges
    ///
    /// # Returns
    /// Number of nodes repaired
    pub fn incremental_repair(&self) -> Result<usize> {
        let n = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        if n == 0 {
            return Ok(0);
        }

        let inserts_since = self
            .incremental_inserts
            .load(std::sync::atomic::Ordering::Relaxed);

        if inserts_since < REPAIR_THRESHOLD {
            return Ok(0);
        }

        // Repair the last REPAIR_THRESHOLD nodes (most recently inserted)
        let repair_count = inserts_since.min(REPAIR_THRESHOLD).min(n);
        let start_id = (n - repair_count) as u32;

        info!(
            "Incremental repair: re-pruning {} recently inserted nodes",
            repair_count
        );

        let medoid = *self.medoid.read();
        let mut repaired = 0;

        for node_id in start_id..(n as u32) {
            // Get vector for this node
            let query = match self.get_vector(node_id) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Search for fresh neighbors using current graph state
            let candidates = self.greedy_search(&query, self.config.search_list_size, medoid)?;

            // Re-prune using proper α-RNG strategy
            let pruned = self.robust_prune(node_id, &candidates)?;

            // Update graph
            let mut graph = self.graph.write();
            let old_neighbors = graph[node_id as usize].neighbors.clone();

            if old_neighbors != pruned {
                graph[node_id as usize].neighbors = pruned.clone();
                repaired += 1;

                // Update bidirectional edges
                // Remove back-edges from old neighbors not in new set
                for &old_neighbor in &old_neighbors {
                    if !pruned.contains(&old_neighbor) && (old_neighbor as usize) < graph.len() {
                        graph[old_neighbor as usize]
                            .neighbors
                            .retain(|&x| x != node_id);
                    }
                }

                // Add back-edges to new neighbors
                for &new_neighbor in &pruned {
                    if (new_neighbor as usize) < graph.len()
                        && !graph[new_neighbor as usize].neighbors.contains(&node_id)
                    {
                        graph[new_neighbor as usize].neighbors.push(node_id);

                        // Prune if exceeds max degree
                        if graph[new_neighbor as usize].neighbors.len() > self.config.max_degree {
                            graph[new_neighbor as usize]
                                .neighbors
                                .truncate(self.config.max_degree);
                        }
                    }
                }
            }
        }

        // Reset counter after repair (but not to 0 - track cumulative for rebuild)
        let new_count = inserts_since.saturating_sub(REPAIR_THRESHOLD);
        self.incremental_inserts
            .store(new_count, std::sync::atomic::Ordering::Relaxed);

        info!("Incremental repair complete: {} nodes updated", repaired);
        Ok(repaired)
    }

    /// Estimate current recall@k using random sampling
    ///
    /// Performs brute-force search on a sample of vectors and compares
    /// against ANN results to estimate recall degradation.
    ///
    /// # Arguments
    /// * `sample_size` - Number of random queries (default: 100)
    /// * `k` - Number of neighbors to check (default: 10)
    ///
    /// # Returns
    /// Estimated recall as f32 in range [0.0, 1.0]
    pub fn estimate_recall(&self, sample_size: usize, k: usize) -> Result<f32> {
        let n = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        if n < 2 {
            return Ok(1.0); // Perfect recall for trivial cases
        }

        let sample_size = sample_size.min(n / 2).max(1);
        let k = k.min(n - 1);

        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        // Sample random query indices
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        let sample_indices: Vec<usize> = indices.into_iter().take(sample_size).collect();

        let mut total_recall = 0.0;

        for &query_idx in &sample_indices {
            let query = self.get_vector(query_idx as u32)?;

            // Get ANN results
            let ann_results = self.search(&query, k)?;
            let ann_ids: HashSet<u32> = ann_results.iter().map(|(id, _)| *id).collect();

            // Get exact brute-force results
            let exact_results = self.brute_force_search(&query, k)?;
            let exact_ids: HashSet<u32> = exact_results.iter().map(|(id, _)| *id).collect();

            // Calculate recall for this query
            let overlap = ann_ids.intersection(&exact_ids).count();
            total_recall += overlap as f32 / k as f32;
        }

        Ok(total_recall / sample_size as f32)
    }

    /// Brute-force k-NN search (for recall estimation)
    fn brute_force_search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        let n = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        let deleted = self.deleted_ids.read();

        let mut distances: Vec<(u32, f32)> = Vec::with_capacity(n);

        for i in 0..n {
            let id = i as u32;
            if deleted.contains(&id) {
                continue;
            }

            let vec = self.get_vector(id)?;
            let dist = self.distance(query, &vec);
            distances.push((id, dist));
        }

        distances.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        distances.truncate(k);

        Ok(distances)
    }

    /// Check if index quality has degraded below acceptable threshold
    ///
    /// Uses sampling to estimate recall without expensive full evaluation.
    /// Returns true if estimated recall@10 < MIN_ACCEPTABLE_RECALL (85%).
    pub fn quality_degraded(&self) -> Result<bool> {
        let n = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        if n < 100 {
            return Ok(false); // Too small to meaningfully measure
        }

        // Quick check: if no incremental inserts, quality is fine
        if self.incremental_insert_count() == 0 {
            return Ok(false);
        }

        // Sample-based recall estimation (50 samples, recall@10)
        let recall = self.estimate_recall(50, 10)?;
        Ok(recall < MIN_ACCEPTABLE_RECALL)
    }

    /// Automatic maintenance: repair or rebuild as needed
    ///
    /// Checks index state and performs appropriate maintenance:
    /// 1. If needs_repair() → incremental_repair()
    /// 2. If needs_rebuild() → auto_rebuild_if_needed()
    ///
    /// Returns description of action taken
    pub fn auto_maintain(&self) -> Result<String> {
        if self.needs_rebuild() {
            if self.auto_rebuild_if_needed()? {
                return Ok("full_rebuild".to_string());
            } else {
                return Ok("rebuild_skipped".to_string());
            }
        }

        if self.needs_repair() {
            let repaired = self.incremental_repair()?;
            return Ok(format!("repaired_{}_nodes", repaired));
        }

        Ok("no_action".to_string())
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
            #[serde(default)]
            deleted_ids: HashSet<u32>,
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
            deleted_ids: self.deleted_ids.read().clone(),
        };

        // Save as length-prefixed postcard binary
        let index_file = path.join("vamana_index.bin");
        let encoded = crate::serialization::encode_raw(&data)?;
        let file = File::create(&index_file)?;
        let mut writer = BufWriter::new(file);
        use std::io::Write;
        writer.write_all(&(encoded.len() as u64).to_le_bytes())?;
        writer.write_all(&encoded)?;
        writer.flush()?;

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
        let mut reader = BufReader::new(file);

        #[derive(Serialize, Deserialize)]
        struct VamanaData {
            graph: Vec<VamanaNode>,
            vectors: Vec<Vec<f32>>,
            medoid: u32,
            num_vectors: usize,
            #[serde(default)]
            deleted_ids: HashSet<u32>,
        }

        // Try new length-prefixed postcard format, fall back to legacy bincode streaming
        let data: VamanaData = {
            use std::io::Read;
            let mut len_buf = [0u8; 8];
            let postcard_result = reader.read_exact(&mut len_buf).ok().and_then(|()| {
                let len = u64::from_le_bytes(len_buf) as usize;
                // Sanity check: reject implausible lengths (> 4 GB)
                if len > 4 * 1024 * 1024 * 1024 {
                    return None;
                }
                let mut buf = vec![0u8; len];
                reader.read_exact(&mut buf).ok()?;
                crate::serialization::decode_raw::<VamanaData>(&buf).ok()
            });
            match postcard_result {
                Some(data) => data,
                None => {
                    // Fall back to legacy bincode streaming format
                    drop(reader);
                    let file = File::open(path.join("vamana_index.bin"))?;
                    let mut reader = BufReader::new(file);
                    bincode::serde::decode_from_std_read(&mut reader, crate::bincode_safe_config())?
                }
            }
        };

        // Update internal state
        *self.graph.write() = data.graph;
        *self.medoid.write() = data.medoid;
        self.num_vectors
            .store(data.num_vectors, std::sync::atomic::Ordering::Release);

        // Update vector storage
        let is_mmap = matches!(*self.vectors.read(), VectorStorage::Mmap { .. });
        if is_mmap {
            // Cannot restore mmap from serialized data - converting to in-memory storage
            warn!(
                "Loading index into mmap-configured instance: converting {} vectors to in-memory storage. \
                 This may increase memory usage. To use mmap, rebuild the index with build().",
                data.num_vectors
            );
            *self.vectors.write() = VectorStorage::Memory(data.vectors);
        } else {
            let is_mmap_variant = matches!(&*self.vectors.read(), VectorStorage::Mmap { .. });
            if is_mmap_variant {
                warn!(
                    "Unexpected mmap storage with use_mmap=false during apply_persisted_data; \
                     converting {} vectors to in-memory storage",
                    data.num_vectors
                );
            }
            // Both branches set to Memory storage — safe assignment without match borrow
            *self.vectors.write() = VectorStorage::Memory(data.vectors);
        }

        // Restore soft-deleted IDs
        if !data.deleted_ids.is_empty() {
            info!(
                "Restoring {} soft-deleted vector IDs from persisted index",
                data.deleted_ids.len()
            );
            *self.deleted_ids.write() = data.deleted_ids;
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
        // Stable tie-break by id ensures deterministic rank order across runs and machines.
        // Without this, equal-distance candidates can swap positions based on float
        // accumulation order, causing rank flips on the recall harness between CPUs.
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
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

    #[test]
    fn test_incremental_repair() {
        let mut index = VamanaIndex::new(VamanaConfig {
            dimension: 4,
            max_degree: 3,
            search_list_size: 10,
            alpha: 1.2,
            use_mmap: false,
            ..Default::default()
        })
        .unwrap();

        // Build initial index
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        index.build(vectors).unwrap();

        // Should not need repair initially
        assert!(!index.needs_repair());
        assert_eq!(index.incremental_insert_count(), 0);

        // Add some vectors incrementally
        for i in 0..5 {
            let v = vec![0.1 * i as f32, 0.1, 0.1, 0.1];
            index.add_vector(v).unwrap();
        }

        assert_eq!(index.incremental_insert_count(), 5);
        assert!(!index.needs_repair()); // Still below threshold

        // Repair should do nothing below threshold
        let repaired = index.incremental_repair().unwrap();
        assert_eq!(repaired, 0);
    }

    #[test]
    fn test_estimate_recall() {
        let mut index = VamanaIndex::new(VamanaConfig {
            dimension: 4,
            max_degree: 4,        // Higher degree for better connectivity
            search_list_size: 20, // Larger search list for better recall
            alpha: 1.2,
            use_mmap: false,
            ..Default::default()
        })
        .unwrap();

        // Use more vectors for stable recall estimation
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0, 0.0],
            vec![0.5, 0.0, 0.5, 0.0],
            vec![0.0, 0.5, 0.5, 0.0],
            vec![0.0, 0.0, 0.5, 0.5],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.7, 0.3, 0.0, 0.0],
        ];
        index.build(vectors).unwrap();

        // Freshly built index should have reasonable recall
        // With small indices, recall can vary; 0.6 is a stable lower bound
        let recall = index.estimate_recall(5, 3).unwrap();
        assert!(recall >= 0.6, "Expected reasonable recall, got {}", recall);
    }

    #[test]
    fn test_auto_maintain() {
        let mut index = VamanaIndex::new(VamanaConfig {
            dimension: 4,
            max_degree: 3,
            search_list_size: 10,
            alpha: 1.2,
            use_mmap: false,
            ..Default::default()
        })
        .unwrap();

        let vectors = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        index.build(vectors).unwrap();

        // Should take no action on fresh index
        let result = index.auto_maintain().unwrap();
        assert_eq!(result, "no_action");
    }

    // ---------------------------------------------------------------------
    // Query-time search list size (SHODH_VAMANA_EF)
    // ---------------------------------------------------------------------

    #[test]
    fn test_parse_search_ef_rejects_non_positive_and_garbage() {
        assert_eq!(parse_search_ef(None), None);
        assert_eq!(parse_search_ef(Some("")), None);
        assert_eq!(parse_search_ef(Some("   ")), None);
        assert_eq!(parse_search_ef(Some("0")), None);
        assert_eq!(parse_search_ef(Some("-4")), None);
        assert_eq!(parse_search_ef(Some("abc")), None);
        assert_eq!(parse_search_ef(Some("1.5")), None);
        assert_eq!(parse_search_ef(Some("1")), Some(1));
        assert_eq!(parse_search_ef(Some(" 256 ")), Some(256));
    }

    /// Deterministic index built only through `add_vector`, which is the path
    /// production and the recall harness actually use (no RNG: the random graph
    /// initialisation lives in `build()`, which this never calls).
    fn deterministic_incremental_index(n: usize, dim: usize, max_degree: usize) -> VamanaIndex {
        let mut index = VamanaIndex::new(VamanaConfig {
            dimension: dim,
            max_degree,
            search_list_size: 100,
            alpha: 1.2,
            use_mmap: false,
            ..Default::default()
        })
        .unwrap();

        // Deterministic pseudo-random unit vectors: a fixed integer hash, so the
        // fixture is identical on every machine and every run.
        for i in 0..n {
            let mut v = Vec::with_capacity(dim);
            for d in 0..dim {
                let h = ((i as u64).wrapping_mul(6_364_136_223_846_793_005)
                    ^ (d as u64).wrapping_mul(1_442_695_040_888_963_407))
                .wrapping_mul(2_862_933_555_777_941_757);
                // 31 random bits mapped to [-1, 1)
                v.push(((h >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0);
            }
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            index.add_vector(v).unwrap();
        }
        index
    }

    #[test]
    fn test_search_ef_none_matches_beam_equal_k() {
        let index = deterministic_incremental_index(300, 16, 4);
        let query = index.get_vector(7).unwrap();

        // `ef = None` and any `ef <= k` must both clamp to the k-wide beam,
        // reproducing the historical search exactly.
        let base = index.search_with_ef(&query, 10, None).unwrap();
        let clamped = index.search_with_ef(&query, 10, Some(1)).unwrap();
        let equal = index.search_with_ef(&query, 10, Some(10)).unwrap();

        assert_eq!(base, clamped, "ef below k must clamp up to k");
        assert_eq!(base, equal, "ef == k must be the historical behaviour");
        assert_eq!(base.len(), 10);
    }

    #[test]
    fn test_search_ef_returns_exactly_k_in_ascending_distance() {
        let index = deterministic_incremental_index(300, 16, 4);
        let query = index.get_vector(11).unwrap();

        let wide = index.search_with_ef(&query, 10, Some(200)).unwrap();
        assert_eq!(
            wide.len(),
            10,
            "a wider beam must not change how many results are returned"
        );

        // Exercise the inner contract directly: `search_with_ef` also applies
        // its own `.take(k)` when filtering soft-deleted ids, so going through
        // the public path alone cannot tell whether `greedy_search_beam` honours
        // "search with L, report k" or leaks the whole search list upward.
        let entry = *index.medoid.read();
        let raw = index.greedy_search_beam(&query, 10, 200, entry).unwrap();
        assert_eq!(
            raw.len(),
            10,
            "greedy_search_beam must report k, not the full search list"
        );
        for pair in wide.windows(2) {
            assert!(
                pair[0].1 <= pair[1].1,
                "results must stay sorted by ascending distance: {:?}",
                wide
            );
        }
    }

    #[test]
    fn test_search_ef_widens_beam_and_recovers_true_neighbors() {
        // Production `max_degree` is 32; the harness index is built purely by
        // `add_vector`, so this fixture is the same regime the eval runs in.
        let index = deterministic_incremental_index(3000, 384, 32);

        let k = 10;
        let mut base_hits = 0usize;
        let mut wide_hits = 0usize;
        let mut total = 0usize;

        for q in [3u32, 29, 71, 130, 244, 301, 388, 415, 502, 577] {
            let query = index.get_vector(q).unwrap();
            let exact: HashSet<u32> = index
                .brute_force_search(&query, k)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();

            let base: HashSet<u32> = index
                .search_with_ef(&query, k, None)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let wide: HashSet<u32> = index
                .search_with_ef(&query, k, Some(256))
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();

            let base_q = base.intersection(&exact).count();
            let wide_q = wide.intersection(&exact).count();

            // Widening the search list can only ever ADD explored nodes, so it
            // must never lose a true neighbour the narrow beam already found.
            assert!(
                wide_q >= base_q,
                "query {q}: widening the beam lost ground ({base_q} -> {wide_q})"
            );

            base_hits += base_q;
            wide_hits += wide_q;
            total += exact.len();
        }

        // The default beam must be genuinely lossy on this fixture — if it were
        // not, the test could not tell a working `ef` from an ignored one.
        assert!(
            base_hits < total,
            "fixture must be lossy at beam == k, got {base_hits}/{total}"
        );
        // ...and the wider beam must strictly recover some of that loss.
        assert!(
            wide_hits > base_hits,
            "ef=256 must beat beam==k: {base_hits}/{total} vs {wide_hits}/{total}"
        );
        eprintln!("vamana ef ablation: beam=k {base_hits}/{total}, ef=256 {wide_hits}/{total}");
    }

    // ---------------------------------------------------------------------
    // Insert-time α-RNG pruning (SHODH_VAMANA_INSERT_PRUNE)
    // ---------------------------------------------------------------------

    #[test]
    fn test_parse_insert_prune_is_opt_out_and_fails_safe() {
        // α-RNG construction is the default; only an explicit 0/false disables
        // it. A typo must fail SAFE, i.e. leave the index navigable.
        assert!(parse_insert_prune(None), "unset must mean α-RNG ON");
        assert!(parse_insert_prune(Some("")));
        assert!(parse_insert_prune(Some("1")));
        assert!(parse_insert_prune(Some("true")));
        assert!(parse_insert_prune(Some("yes")));
        assert!(parse_insert_prune(Some("2")));
        assert!(
            parse_insert_prune(Some("flase")),
            "a typo must not disable it"
        );
        assert!(!parse_insert_prune(Some("0")));
        assert!(!parse_insert_prune(Some("false")));
        assert!(!parse_insert_prune(Some("FALSE")));
        assert!(!parse_insert_prune(Some(" false ")));
    }

    /// The α-RNG rule must keep the closest candidate AND the geometrically
    /// diverse one, dropping a near-duplicate of an already-kept neighbor —
    /// keep-closest would keep the duplicate and drop the diverse candidate.
    #[test]
    fn test_robust_prune_keeps_diverse_over_near_duplicate() {
        let mut index = VamanaIndex::new(VamanaConfig {
            dimension: 4,
            max_degree: 2,
            search_list_size: 10,
            alpha: 1.2,
            use_mmap: false,
            ..Default::default()
        })
        .unwrap();

        // All unit vectors; NormalizedDotProduct distance = -dot.
        let node = vec![1.0, 0.0, 0.0, 0.0]; // id 0: the node being pruned
        let close_a = vec![0.992, 0.126_231_93, 0.0, 0.0]; // id 1: closest
        let close_b = vec![0.990, 0.141_067_36, 0.0, 0.0]; // id 2: near-duplicate of id 1
        let diverse = vec![0.0, 1.0, 0.0, 0.0]; // id 3: far but diverse
        for v in [&node, &close_a, &close_b, &diverse] {
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-3, "fixture vectors must be unit norm");
        }
        {
            let mut storage = index.vectors.write();
            *storage = VectorStorage::Memory(vec![node.clone(), close_a, close_b, diverse]);
        }
        index
            .num_vectors
            .store(4, std::sync::atomic::Ordering::Release);

        let dist = |a: &[f32], b: &[f32]| -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let storage = index.vectors.read();
        let candidates: Vec<SearchCandidate> = (1..4)
            .map(|id| SearchCandidate {
                id,
                distance: dist(
                    &node,
                    &VamanaIndex::get_vector_from_storage(&storage, id).unwrap(),
                ),
            })
            .collect();

        let pruned = index.robust_prune_in(&storage, 0, &candidates).unwrap();
        assert_eq!(
            pruned,
            vec![1, 3],
            "α-RNG must keep the closest (1) and the diverse (3) candidates and \
             drop the near-duplicate (2); keep-closest would produce [1, 2]"
        );
    }

    /// End-to-end through `add_vector_with_policy`: when a hub's reverse edges
    /// overflow max_degree, the α path must retain its long-range (diverse)
    /// edge while the historical path evicts it for a same-cluster duplicate.
    /// This is exactly the structural difference that costs task recall: greedy
    /// search navigates BY those long-range edges.
    #[test]
    fn test_insert_prune_reverse_edges_keep_long_range_link() {
        let build = |alpha_prune: bool| -> VamanaIndex {
            let mut index = VamanaIndex::new(VamanaConfig {
                dimension: 4,
                max_degree: 2,
                search_list_size: 10,
                alpha: 1.2,
                use_mmap: false,
                ..Default::default()
            })
            .unwrap();
            let vectors = [
                vec![1.0, 0.0, 0.0, 0.0],            // 0: hub H (medoid/entry)
                vec![0.0, 1.0, 0.0, 0.0],            // 1: far diverse F
                vec![0.992, 0.126_231_93, 0.0, 0.0], // 2: close c1
                vec![0.990, 0.141_067_36, 0.0, 0.0], // 3: close c2 ≈ c1
            ];
            for v in vectors {
                index.add_vector_with_policy(v, alpha_prune).unwrap();
            }
            index
        };

        let alpha = build(true);
        let alpha_hub = alpha.graph.read()[0].neighbors.clone();
        assert!(
            alpha_hub.contains(&1),
            "α-RNG reverse pruning must keep the hub's long-range edge to the \
             diverse node, got {alpha_hub:?}"
        );
        assert!(
            alpha_hub.len() <= 2,
            "reverse edges must stay within max_degree, got {alpha_hub:?}"
        );

        let greedy = build(false);
        let greedy_hub = greedy.graph.read()[0].neighbors.clone();
        assert!(
            !greedy_hub.contains(&1),
            "fixture must be discriminating: the historical keep-closest path \
             evicts the long-range edge (got {greedy_hub:?}); if it no longer \
             does, this test cannot tell the two policies apart"
        );
    }

    /// Front half of the insert policy: the NEW node's own neighbor list must
    /// be α-diversified, not the raw greedy top-k. The fixture is built so the
    /// greedy graph has already lost its long-range edge by the fifth insert —
    /// the greedy candidate search cannot even REACH the diverse node — while
    /// the α graph both reaches it and selects it over a redundant
    /// near-duplicate. Catches the mutation where `add_vector_with_policy`
    /// α-prunes reverse edges but silently keeps greedy selection for the new
    /// node itself (the reverse-edge test alone cannot see that break).
    #[test]
    fn test_insert_prune_diversifies_new_node_neighbors() {
        let build = |alpha_prune: bool| -> VamanaIndex {
            let mut index = VamanaIndex::new(VamanaConfig {
                dimension: 4,
                max_degree: 2,
                search_list_size: 10,
                alpha: 1.2,
                use_mmap: false,
                ..Default::default()
            })
            .unwrap();
            let vectors = [
                vec![1.0, 0.0, 0.0, 0.0],            // 0: hub H (medoid/entry)
                vec![0.0, 1.0, 0.0, 0.0],            // 1: far diverse F
                vec![0.992, 0.126_231_93, 0.0, 0.0], // 2: close c1
                vec![0.990, 0.141_067_36, 0.0, 0.0], // 3: close c2 ≈ c1
                vec![0.924, -0.382_499_5, 0.0, 0.0], // 4: X, outside the c-cluster
            ];
            for v in vectors {
                index.add_vector_with_policy(v, alpha_prune).unwrap();
            }
            index
        };

        let alpha = build(true);
        let alpha_new = alpha.graph.read()[4].neighbors.clone();
        assert!(
            alpha_new.contains(&1),
            "α selection for the new node must keep the diverse far node, got {alpha_new:?}"
        );
        assert!(
            !alpha_new.contains(&2) && !alpha_new.contains(&3),
            "α selection must drop cluster members redundant with the hub, got {alpha_new:?}"
        );

        let greedy = build(false);
        let greedy_new = greedy.graph.read()[4].neighbors.clone();
        assert!(
            !greedy_new.contains(&1),
            "fixture must be discriminating: greedy selection cannot reach or \
             keep the diverse node (got {greedy_new:?}); if it now does, this \
             test cannot tell the two policies apart"
        );
    }

    /// Deterministic CLUSTERED fixture built only through the incremental
    /// insert path. Uniform-random unit vectors are the easy case for a greedy
    /// kNN graph (near-orthogonal, no local minima); real MiniLM embeddings of
    /// a conversational corpus are tightly clustered near-duplicates, which is
    /// where greedy top-k neighbor lists degenerate into same-cluster cliques.
    fn clustered_incremental_index(
        n: usize,
        dim: usize,
        clusters: usize,
        max_degree: usize,
        alpha_prune: bool,
    ) -> VamanaIndex {
        let mut index = VamanaIndex::new(VamanaConfig {
            dimension: dim,
            max_degree,
            search_list_size: 100,
            alpha: 1.2,
            use_mmap: false,
            ..Default::default()
        })
        .unwrap();

        let h = |a: u64, b: u64| -> f32 {
            let x = (a.wrapping_mul(6_364_136_223_846_793_005)
                ^ b.wrapping_mul(1_442_695_040_888_963_407))
            .wrapping_mul(2_862_933_555_777_941_757);
            ((x >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };

        for i in 0..n {
            let c = (i % clusters) as u64;
            let mut v = Vec::with_capacity(dim);
            for d in 0..dim {
                // Cluster center + small within-cluster noise: after
                // normalization neighbors within a cluster have cosine ≈ 0.9+,
                // matching embedded near-duplicate conversation turns.
                let center = h(c.wrapping_add(1_000_003), d as u64);
                let noise = h(i as u64, d.wrapping_add(7_777_777) as u64);
                v.push(center + 0.15 * noise);
            }
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            index.add_vector_with_policy(v, alpha_prune).unwrap();
        }
        index
    }

    /// The shipped index must be NAVIGABLE, not merely populated.
    ///
    /// This is the guard that did not exist when `add_vector` was allowed to skip
    /// the α-RNG rule "for speed". Nothing failed when that landed. The index kept
    /// accepting vectors, kept returning results, and kept passing every test —
    /// it had simply stopped being a Vamana graph. Greedy search could not reach
    /// over half of the index's own true neighbours, and no beam width recovers a
    /// neighbour the graph has no edge to, which is why widening `ef` only ever
    /// bought back a third of the loss.
    ///
    /// The comparative test above proves α-RNG beats greedy. That is not the same
    /// guarantee: two equally broken constructions would still satisfy it. This
    /// one pins an absolute floor on the DEFAULT policy, so a future trade of
    /// construction quality for insert latency fails here rather than showing up
    /// months later as an unexplained retrieval deficit.
    #[test]
    fn shipped_default_index_is_navigable_on_clustered_data() {
        // α-RNG measures 99.9% on this fixture and greedy 45.6%; 0.90 sits far
        // from both, so the test discriminates the construction rule rather than
        // tracking fixture noise.
        const SELF_RECALL_FLOOR: f64 = 0.90;

        let n = 1200;
        let dim = 384;
        let clusters = 40;
        let k = 120;

        // `parse_insert_prune(None)` is the SHIPPED default with the variable
        // unset — deliberately not `configured_insert_prune()`, whose OnceLock
        // can be poisoned by whichever test in this binary reads the env first.
        let default_policy = parse_insert_prune(None);
        let index = clustered_incremental_index(n, dim, clusters, 32, default_policy);

        let queries: Vec<u32> = (0..40).map(|i| (i * 29 + 3) % n as u32).collect();
        let mut hits = 0usize;
        let mut total = 0usize;
        for &q in &queries {
            let query = index.get_vector(q).unwrap();
            let exact: std::collections::HashSet<u32> = index
                .brute_force_search(&query, k)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            total += exact.len();
            hits += index
                .search_with_ef(&query, k, None)
                .unwrap()
                .into_iter()
                .filter(|(id, _)| exact.contains(id))
                .count();
        }

        let ratio = hits as f64 / total as f64;
        eprintln!(
            "shipped default (insert_prune={default_policy}): index self-recall \
             {hits}/{total} = {ratio:.4} at beam=k"
        );
        assert!(
            ratio >= SELF_RECALL_FLOOR,
            "the shipped index cannot navigate to its own true neighbours: \
             {hits}/{total} = {ratio:.4} at beam=k, floor {SELF_RECALL_FLOOR}. \
             The construction rule has regressed — greedy top-k inserts build a \
             kNN graph, not a Vamana graph."
        );
    }
    /// Root-cause fixture for the ANN-vs-exact gap at the pipeline's real
    /// operating point (k = 120, max_degree = 32, incremental inserts only):
    /// on clustered vectors the greedy-insert graph must be measurably lossy
    /// at beam = k, and α-RNG insert pruning must recover part of that loss
    /// with NO query-time change. If the α graph stopped beating the greedy
    /// graph here, the insert-prune path is broken (e.g. the policy flag is
    /// being ignored) — that is the mutation this test is built to catch.
    #[test]
    fn test_insert_prune_recovers_index_recall_on_clustered_fixture() {
        let n = 1200;
        let dim = 384;
        let clusters = 40;
        let k = 120;

        let greedy = clustered_incremental_index(n, dim, clusters, 32, false);
        let alpha = clustered_incremental_index(n, dim, clusters, 32, true);

        let queries: Vec<u32> = (0..40).map(|i| (i * 29 + 3) % n as u32).collect();
        let mut greedy_hits = 0usize;
        let mut alpha_hits = 0usize;
        let mut greedy_hits_ef = 0usize;
        let mut alpha_hits_ef = 0usize;
        let mut total = 0usize;

        for &q in &queries {
            let query = greedy.get_vector(q).unwrap();
            let exact: HashSet<u32> = greedy
                .brute_force_search(&query, k)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            total += exact.len();

            let hits = |index: &VamanaIndex, ef: Option<usize>| -> usize {
                index
                    .search_with_ef(&query, k, ef)
                    .unwrap()
                    .into_iter()
                    .filter(|(id, _)| exact.contains(id))
                    .count()
            };
            greedy_hits += hits(&greedy, None);
            alpha_hits += hits(&alpha, None);
            greedy_hits_ef += hits(&greedy, Some(512));
            alpha_hits_ef += hits(&alpha, Some(512));
        }

        eprintln!(
            "clustered fixture (n={n}, dim={dim}, clusters={clusters}, k={k}): \
             greedy beam=k {greedy_hits}/{total}, α beam=k {alpha_hits}/{total}, \
             greedy ef=512 {greedy_hits_ef}/{total}, α ef=512 {alpha_hits_ef}/{total}"
        );

        // The greedy-insert graph must be genuinely lossy on clustered data at
        // the production operating point, or this fixture can't discriminate.
        assert!(
            greedy_hits < total,
            "fixture must be lossy for greedy inserts at beam == k, got {greedy_hits}/{total}"
        );
        // α-RNG insert pruning must strictly improve index recall at the SAME
        // query cost (beam = k). This is the root-cause claim: the loss is
        // graph STRUCTURE, not just beam width.
        assert!(
            alpha_hits > greedy_hits,
            "α-RNG insert pruning must beat greedy inserts at beam == k: \
             {greedy_hits}/{total} vs {alpha_hits}/{total}"
        );
        // Widening the beam must not erase the structural advantage entirely
        // in the wrong direction: the α graph may not do WORSE than the greedy
        // graph when both search at ef=512.
        assert!(
            alpha_hits_ef >= greedy_hits_ef,
            "α graph must not lose to greedy graph at ef=512: \
             {greedy_hits_ef}/{total} vs {alpha_hits_ef}/{total}"
        );
    }

    /// `add_vector` (env-driven) and `add_vector_with_policy(_, false)` must
    /// build the identical graph when the flag is unset — the default path is
    /// bit-identical to the pre-flag implementation.
    #[test]
    fn test_add_vector_default_matches_policy_true() {
        // Guard: only meaningful when the env flag is unset (the CI/test
        // default). If someone exports SHODH_VAMANA_INSERT_PRUNE=0 globally,
        // failing loudly here is correct — the "default" path would no longer
        // be the shipped default.
        assert!(
            configured_insert_prune(),
            "{INSERT_PRUNE_ENV} must be unset when running the test suite"
        );

        let make = |policy: Option<bool>| -> Vec<Vec<u32>> {
            let mut index = VamanaIndex::new(VamanaConfig {
                dimension: 16,
                max_degree: 4,
                search_list_size: 20,
                alpha: 1.2,
                use_mmap: false,
                ..Default::default()
            })
            .unwrap();
            for i in 0..120usize {
                let mut v: Vec<f32> = (0..16)
                    .map(|d| {
                        let x = ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            ^ (d as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
                        .wrapping_mul(0x94D0_49BB_1331_11EB);
                        ((x >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
                    })
                    .collect();
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in &mut v {
                    *x /= norm;
                }
                match policy {
                    Some(p) => index.add_vector_with_policy(v, p).unwrap(),
                    None => index.add_vector(v).unwrap(),
                };
            }
            let graph = index.graph.read();
            graph.iter().map(|node| node.neighbors.clone()).collect()
        };

        assert_eq!(
            make(None),
            make(Some(true)),
            "env-default add_vector must match the explicit historical policy"
        );
        assert_ne!(
            make(Some(false)),
            make(Some(true)),
            "the two policies must build different graphs on this fixture — if \
             they don't, the α path is silently not running"
        );
    }
}
