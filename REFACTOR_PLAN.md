# COMPREHENSIVE REFACTORING PLAN

## P0: CRITICAL STABILITY FIXES

### P0.1: Error Handling - COMPLETED ITEMS
- ✅ Added LockPoisoned and LockAcquisitionFailed to AppError
- ✅ Created lock_utils.rs with safe lock helpers
- ✅ Changed memory/mod.rs to use parking_lot::RwLock
- ✅ Added helper methods to AppError

### P0.1: Error Handling - REMAINING ITEMS
Since we're now using `parking_lot::RwLock` which NEVER poisons:
- **ACTION**: Remove all `.expect("Failed to acquire lock")` calls
- **RATIONALE**: parking_lot locks never fail, so `.read()` and `.write()` are infallible
- **IMPACT**: Code becomes cleaner and more idiomatic

**Files to update:**
1. `src/memory/mod.rs` - 29 `.expect()` calls → simple `.read()` or `.write()`
2. `src/memory/retrieval.rs` - 2 `.expect()` calls
3. `src/memory/context.rs` - 2 `.expect()` calls
4. `src/embeddings/minilm.rs` - 2 `.expect()` calls
5. `src/vector_db/vamana.rs` - 10 `.expect()` calls
6. `src/vector_db/distance.rs` - 2 `.expect()` calls
7. `src/main.rs` - 4 `.expect()` calls (keep signal handler expects, fix others)

### P0.2: Memory Cloning Elimination

**Current Problem:**
```rust
// 122 clone() calls - each Memory has Vec<f32> embeddings (1.5-6KB)
self.working_memory.add(memory.clone())  // WASTEFUL
memories.into_iter().map(|m| m.clone())  // UNNECESSARY
```

**Solution:**
1. Wrap Memory in Arc for shared ownership:
   ```rust
   pub type SharedMemory = Arc<Memory>;
   ```

2. Update storage to use `Arc<Memory>`:
   ```rust
   pub struct WorkingMemory {
       memories: HashMap<MemoryId, SharedMemory>, // was: Memory
   }
   ```

3. Update APIs to return references or Arc:
   ```rust
   pub fn retrieve(&self, query: &Query) -> Result<Vec<SharedMemory>>
   ```

**Files to update:**
- `src/memory/types.rs` - Add SharedMemory type alias
- `src/memory/storage.rs` - Use Arc<Memory>
- `src/memory/mod.rs` - Use Arc<Memory> in tiers
- `src/memory/retrieval.rs` - Return Arc<Memory>
- `src/main.rs` - Handle Arc<Memory> in responses

### P0.3: Blocking I/O in Async

**Problem:**
```rust
// RocksDB operations block the Tokio executor!
self.audit_db.put(key.as_bytes(), serialized); // BLOCKS 100ms+
self.long_term_memory.store(&memory)?; // BLOCKS
```

**Solution:**
Wrap all RocksDB operations in `tokio::task::spawn_blocking`:
```rust
let db = self.audit_db.clone();
tokio::task::spawn_blocking(move || {
    db.put(key.as_bytes(), serialized)
}).await??;
```

**Files to update:**
- `src/main.rs` - All audit_db operations
- `src/memory/storage.rs` - Make methods async or document blocking
- `src/memory/retrieval.rs` - Vamana index operations

### P0.4: Race Condition in Audit Logs

**Problem:**
```rust
let mut counter = self.audit_log_counter.lock();
*counter += 1;
if *counter >= AUDIT_ROTATION_CHECK_INTERVAL {
    *counter = 0;
    drop(counter); // TOCTOU: Other threads can now increment!
    self.rotate_user_audit_logs(user_id)?;
}
```

**Solution:**
Use `AtomicUsize` with `fetch_add`:
```rust
use std::sync::atomic::{AtomicUsize, Ordering};

struct MultiUserMemoryManager {
    audit_log_counter: Arc<AtomicUsize>,
}

// In log_event:
let count = self.audit_log_counter.fetch_add(1, Ordering::Relaxed);
if count % AUDIT_ROTATION_CHECK_INTERVAL == 0 {
    let _ = self.rotate_user_audit_logs(user_id);
}
```

## P1: HIGH-PRIORITY IMPROVEMENTS

### P1.1: God Object Refactor

Split `MultiUserMemoryManager` (800+ lines) into:

```rust
/// Manages user memory systems
pub struct UserMemoryManager {
    user_memories: DashMap<String, Arc<RwLock<MemorySystem>>>,
    base_path: PathBuf,
    default_config: MemoryConfig,
}

/// Manages audit logs
pub struct AuditManager {
    audit_logs: DashMap<String, Arc<RwLock<Vec<AuditEvent>>>>,
    audit_db: Arc<rocksdb::DB>,
    counter: Arc<AtomicUsize>,
}

/// Manages knowledge graphs
pub struct GraphMemoryManager {
    graph_memories: DashMap<String, Arc<RwLock<GraphMemory>>>,
    entity_extractor: Arc<EntityExtractor>,
}

/// Top-level coordinator
pub struct MemoryService {
    user_manager: Arc<UserMemoryManager>,
    audit_manager: Arc<AuditManager>,
    graph_manager: Arc<GraphMemoryManager>,
}
```

### P1.2: Proper Error Hierarchy

Current `anyhow::Error` loses type information. Create:

```rust
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("Index error: {0}")]
    Index(#[from] IndexError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("RocksDB error: {0}")]
    RocksDB(#[from] rocksdb::Error),

    #[error("Key not found: {0}")]
    NotFound(String),
}
```

### P1.3: Connection Pooling

RocksDB doesn't need pooling, but we should:
1. Lazy-load user memory systems
2. Add LRU eviction for inactive users
3. Periodic flush to disk

## P2: FEATURE IMPROVEMENTS

### P2.1: NLP-based Entity Extraction

Replace primitive extraction with proper NER:

```rust
use rust_bert::pipelines::ner::NERModel;

pub struct EntityExtractor {
    ner_model: NERModel,
}

impl EntityExtractor {
    pub fn extract(&self, text: &str) -> Vec<(String, EntityLabel, f32)> {
        let entities = self.ner_model.predict(&[text]);
        // Returns (text, label, confidence score)
    }
}
```

### P2.2: Graph-Aware Search

Fuse vector search with graph traversal:

```rust
pub fn graph_aware_search(
    &self,
    query: &Query,
    graph: &GraphMemory,
) -> Result<Vec<Memory>> {
    // 1. Vector search for initial candidates
    let initial = self.vector_search(query)?;

    // 2. Expand using graph relationships
    let expanded = graph.expand_with_related(&initial)?;

    // 3. Re-rank using graph + vector signals
    self.hybrid_rank(&expanded, query)
}
```

## IMPLEMENTATION ORDER

1. **Week 1 - P0 (Stability)**
   - Day 1-2: Remove .expect() calls (P0.1)
   - Day 3-4: Arc<Memory> refactor (P0.2)
   - Day 5: Async I/O fixes (P0.3)
   - Day 5: Atomic counter (P0.4)

2. **Week 2 - P1 (Architecture)**
   - Day 1-3: God Object split (P1.1)
   - Day 4-5: Error hierarchy (P1.2)

3. **Week 3 - P2 (Features)**
   - Day 1-3: NLP entity extraction (P2.1)
   - Day 4-5: Graph-aware search (P2.2)

## SUCCESS METRICS

- **P0 Complete**: Zero panics in production, 10x better memory efficiency
- **P1 Complete**: <500 lines per module, typed error handling
- **P2 Complete**: >80% entity extraction accuracy, 2x search relevance
