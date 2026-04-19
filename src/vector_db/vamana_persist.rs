//! Vamana index persistence for instant startup
//!
//! Binary file format for persisting Vamana graph to disk.
//! Uses mmap for zero-copy loading.
//!
//! # File Format (v1)
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │ Header (64 bytes)                       │
//! │ ├── magic: [u8; 4] = "VAMA"             │
//! │ ├── version: u32 = 1                    │
//! │ ├── num_vectors: u64                    │
//! │ ├── dimension: u32                      │
//! │ ├── max_degree: u32                     │
//! │ ├── medoid: u32                         │
//! │ ├── distance_metric: u8                 │
//! │ ├── deleted_count: u32                  │
//! │ ├── incremental_inserts: u64            │
//! │ ├── checksum: u64                       │
//! │ └── reserved: [u8; 15]                  │
//! ├─────────────────────────────────────────┤
//! │ Deleted IDs Section                     │
//! │ └── [u32; deleted_count]                │
//! ├─────────────────────────────────────────┤
//! │ Graph Section                           │
//! │ ├── For each node:                      │
//! │ │   ├── neighbor_count: u16             │
//! │ │   └── neighbors: [u32; neighbor_count]│
//! ├─────────────────────────────────────────┤
//! │ Vectors Section (aligned to 64 bytes)   │
//! │ └── [[f32; dimension]; num_vectors]     │
//! └─────────────────────────────────────────┘
//! ```

use anyhow::{anyhow, Result};
use memmap2::{Mmap, MmapMut};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read};
use std::path::Path;
use tracing::info;

use super::vamana::{DistanceMetric, VamanaConfig, VamanaIndex, VamanaNode, VectorStorage};

const MAGIC: [u8; 4] = *b"VAMA";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;
const ALIGNMENT: usize = 64;

/// Header for persisted Vamana index
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
struct VamanaHeader {
    magic: [u8; 4],
    version: u32,
    num_vectors: u64,
    dimension: u32,
    max_degree: u32,
    medoid: u32,
    distance_metric: u8,
    deleted_count: u32,
    incremental_inserts: u64,
    checksum: u64,
    reserved: [u8; 15],
}

impl VamanaHeader {
    fn new(
        num_vectors: usize,
        dimension: usize,
        max_degree: usize,
        medoid: u32,
        distance_metric: DistanceMetric,
        deleted_count: usize,
        incremental_inserts: usize,
    ) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            num_vectors: num_vectors as u64,
            dimension: dimension as u32,
            max_degree: max_degree as u32,
            medoid,
            distance_metric: match distance_metric {
                DistanceMetric::NormalizedDotProduct => 0,
                DistanceMetric::Euclidean => 1,
                DistanceMetric::Cosine => 2,
            },
            deleted_count: deleted_count as u32,
            incremental_inserts: incremental_inserts as u64,
            checksum: 0, // Computed after serialization
            reserved: [0u8; 15],
        }
    }

    #[allow(clippy::wrong_self_convention)] // &self avoids copying 56-byte header struct
    fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..8].copy_from_slice(&self.version.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.num_vectors.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.dimension.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.max_degree.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.medoid.to_le_bytes());
        bytes[28] = self.distance_metric;
        bytes[29..33].copy_from_slice(&self.deleted_count.to_le_bytes());
        bytes[33..41].copy_from_slice(&self.incremental_inserts.to_le_bytes());
        bytes[41..49].copy_from_slice(&self.checksum.to_le_bytes());
        // reserved bytes already 0
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

        Ok(Self {
            magic,
            version,
            num_vectors: u64::from_le_bytes(bytes[8..16].try_into()?),
            dimension: u32::from_le_bytes(bytes[16..20].try_into()?),
            max_degree: u32::from_le_bytes(bytes[20..24].try_into()?),
            medoid: u32::from_le_bytes(bytes[24..28].try_into()?),
            distance_metric: bytes[28],
            deleted_count: u32::from_le_bytes(bytes[29..33].try_into()?),
            incremental_inserts: u64::from_le_bytes(bytes[33..41].try_into()?),
            checksum: u64::from_le_bytes(bytes[41..49].try_into()?),
            reserved: [0u8; 15],
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

/// Compute checksum for data integrity
fn compute_checksum(data: &[u8]) -> u64 {
    // Simple xxhash-style checksum
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    hash
}

/// Align offset to boundary
fn align_to(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

impl VamanaIndex {
    /// Save index to file
    ///
    /// Serializes the entire index (graph + vectors) to a binary file.
    /// Can be loaded back with `load_from_file()`.
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let start = std::time::Instant::now();

        let graph = self.graph.read();
        let vectors = self.vectors.read();
        let deleted_ids = self.deleted_ids.read();
        let medoid = *self.medoid.read();
        let num_vectors = self.num_vectors.load(std::sync::atomic::Ordering::Acquire);
        let incremental_inserts = self
            .incremental_inserts
            .load(std::sync::atomic::Ordering::Acquire);

        // Extract vectors based on storage type
        let vector_data: Vec<f32> = match &*vectors {
            VectorStorage::Memory(vecs) => vecs.iter().flatten().copied().collect(),
            VectorStorage::Mmap {
                mmap,
                dimension,
                num_vectors,
            } => {
                let total_floats = dimension * num_vectors;
                let bytes = &mmap[..total_floats * 4];
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            }
        };

        // Create header
        let mut header = VamanaHeader::new(
            num_vectors,
            self.config.dimension,
            self.config.max_degree,
            medoid,
            self.config.distance_metric,
            deleted_ids.len(),
            incremental_inserts,
        );

        // Calculate sizes
        let deleted_section_size = deleted_ids.len() * 4;
        let mut graph_section_size = 0;
        for node in graph.iter() {
            graph_section_size += 2 + node.neighbors.len() * 4; // u16 count + u32 per neighbor
        }
        let vectors_offset = align_to(
            HEADER_SIZE + deleted_section_size + graph_section_size,
            ALIGNMENT,
        );
        let vectors_section_size = vector_data.len() * 4;
        let total_size = vectors_offset + vectors_section_size;

        // Create file and write
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(total_size as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Write header (without checksum first)
        let header_bytes = header.to_bytes();
        mmap[..HEADER_SIZE].copy_from_slice(&header_bytes);

        // Write deleted IDs
        let mut offset = HEADER_SIZE;
        for &id in deleted_ids.iter() {
            mmap[offset..offset + 4].copy_from_slice(&id.to_le_bytes());
            offset += 4;
        }

        // Write graph
        for node in graph.iter() {
            let count = node.neighbors.len() as u16;
            mmap[offset..offset + 2].copy_from_slice(&count.to_le_bytes());
            offset += 2;
            for &neighbor in &node.neighbors {
                mmap[offset..offset + 4].copy_from_slice(&neighbor.to_le_bytes());
                offset += 4;
            }
        }

        // Write vectors (aligned)
        offset = vectors_offset;
        for val in &vector_data {
            mmap[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
            offset += 4;
        }

        // Compute and write checksum
        let checksum = compute_checksum(&mmap[HEADER_SIZE..]);
        header.checksum = checksum;
        let header_bytes = header.to_bytes();
        mmap[..HEADER_SIZE].copy_from_slice(&header_bytes);

        mmap.flush()?;

        info!(
            "Saved Vamana index: {} vectors, {} bytes in {:?}",
            num_vectors,
            total_size,
            start.elapsed()
        );

        Ok(())
    }

    /// Load index from file
    ///
    /// Uses mmap for zero-copy access to vectors.
    /// Returns immediately - vectors are demand-paged by OS.
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let start = std::time::Instant::now();

        if !path.exists() {
            return Err(anyhow!("Index file not found: {:?}", path));
        }

        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Read header
        let header = VamanaHeader::from_bytes(&mmap[..HEADER_SIZE])?;

        // Verify checksum
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
        let dimension = header.dimension as usize;
        let deleted_count = header.deleted_count as usize;

        // Read deleted IDs
        let mut offset = HEADER_SIZE;
        let mut deleted_ids = HashSet::with_capacity(deleted_count);
        for _ in 0..deleted_count {
            let id = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
            deleted_ids.insert(id);
            offset += 4;
        }

        // Read graph
        let mut graph = Vec::with_capacity(num_vectors);
        for node_id in 0..num_vectors {
            let count = u16::from_le_bytes(mmap[offset..offset + 2].try_into()?) as usize;
            offset += 2;
            let mut neighbors = Vec::with_capacity(count);
            for _ in 0..count {
                let neighbor = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
                neighbors.push(neighbor);
                offset += 4;
            }
            graph.push(VamanaNode {
                id: node_id as u32,
                neighbors,
            });
        }

        // Calculate vectors offset (aligned)
        let vectors_offset = align_to(offset, ALIGNMENT);

        // Read vectors into memory (could also keep mmap'd)
        let vectors_bytes = &mmap[vectors_offset..];
        let mut vectors = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            let start = i * dimension * 4;
            let end = start + dimension * 4;
            let vec: Vec<f32> = vectors_bytes[start..end]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            vectors.push(vec);
        }

        let config = VamanaConfig {
            max_degree: header.max_degree as usize,
            search_list_size: 75, // Default, not persisted
            alpha: 1.2,           // Default, not persisted
            dimension,
            use_mmap: false, // Loaded into memory
            distance_metric: header.distance_metric_enum(),
        };

        let index = VamanaIndex {
            config,
            graph: std::sync::Arc::new(parking_lot::RwLock::new(graph)),
            vectors: std::sync::Arc::new(parking_lot::RwLock::new(VectorStorage::Memory(vectors))),
            medoid: std::sync::Arc::new(parking_lot::RwLock::new(header.medoid)),
            num_vectors: std::sync::atomic::AtomicUsize::new(num_vectors),
            storage_path: Some(path.to_path_buf()),
            incremental_inserts: std::sync::atomic::AtomicUsize::new(
                header.incremental_inserts as usize,
            ),
            rebuilding: std::sync::atomic::AtomicBool::new(false),
            deleted_ids: std::sync::Arc::new(parking_lot::RwLock::new(deleted_ids)),
        };

        info!(
            "Loaded Vamana index: {} vectors in {:?}",
            num_vectors,
            start.elapsed()
        );

        Ok(index)
    }

    /// Check if a persisted index exists and is valid
    pub fn index_file_exists(path: &Path) -> bool {
        if !path.exists() {
            return false;
        }

        // Try to read and validate header
        if let Ok(file) = File::open(path) {
            let mut header_bytes = [0u8; HEADER_SIZE];
            let mut reader = BufReader::new(file);
            if reader.read_exact(&mut header_bytes).is_ok() {
                return VamanaHeader::from_bytes(&header_bytes).is_ok();
            }
        }
        false
    }

    /// Verify index file integrity without fully loading
    pub fn verify_index_file(path: &Path) -> Result<bool> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Ok(false);
        }

        let header = VamanaHeader::from_bytes(&mmap[..HEADER_SIZE])?;
        let stored_checksum = header.checksum;
        let computed_checksum = compute_checksum(&mmap[HEADER_SIZE..]);

        Ok(stored_checksum == computed_checksum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_save_and_load() {
        let temp_dir = tempdir().unwrap();
        let index_path = temp_dir.path().join("test.vamana");

        // Create index with some vectors (disable mmap for test)
        let config = VamanaConfig {
            dimension: 4,
            max_degree: 8,
            use_mmap: false,
            ..Default::default()
        };
        let mut index = VamanaIndex::new(config).unwrap();

        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0, 0.0],
        ];
        index.build(vectors.clone()).unwrap();

        // Save
        index.save_to_file(&index_path).unwrap();
        assert!(index_path.exists());

        // Verify
        assert!(VamanaIndex::verify_index_file(&index_path).unwrap());

        // Load
        let loaded = VamanaIndex::load_from_file(&index_path).unwrap();
        assert_eq!(loaded.len(), 5);

        // Search should work
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = loaded.search(&query, 3).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0); // Should find vector 0 first
    }

    #[test]
    fn test_checksum_detects_corruption() {
        let temp_dir = tempdir().unwrap();
        let index_path = temp_dir.path().join("corrupt.vamana");

        // Create and save index (disable mmap for test)
        let config = VamanaConfig {
            dimension: 4,
            use_mmap: false,
            ..Default::default()
        };
        let mut index = VamanaIndex::new(config).unwrap();
        index
            .build(vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]])
            .unwrap();
        index.save_to_file(&index_path).unwrap();

        // Corrupt the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&index_path)
            .unwrap();
        let mut mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        mmap[HEADER_SIZE + 10] ^= 0xFF; // Flip some bits
        mmap.flush().unwrap();

        // Verify should fail
        assert!(!VamanaIndex::verify_index_file(&index_path).unwrap());
    }

    #[test]
    fn test_header_serialization() {
        let header = VamanaHeader::new(
            1000,
            384,
            32,
            42,
            DistanceMetric::NormalizedDotProduct,
            5,
            100,
        );

        let bytes = header.to_bytes();
        let restored = VamanaHeader::from_bytes(&bytes).unwrap();

        // Copy values to avoid packed struct reference issues
        let num_vectors = restored.num_vectors;
        let dimension = restored.dimension;
        let max_degree = restored.max_degree;
        let medoid = restored.medoid;
        let deleted_count = restored.deleted_count;
        let incremental_inserts = restored.incremental_inserts;

        assert_eq!(num_vectors, 1000);
        assert_eq!(dimension, 384);
        assert_eq!(max_degree, 32);
        assert_eq!(medoid, 42);
        assert_eq!(deleted_count, 5);
        assert_eq!(incremental_inserts, 100);
    }
}
