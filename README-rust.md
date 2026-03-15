# shodh-memory

**Persistent cognitive memory for AI agents. Local-first. Runs offline.**

[![crates.io](https://img.shields.io/crates/v/shodh-memory.svg)](https://crates.io/crates/shodh-memory)
[![Downloads](https://img.shields.io/crates/d/shodh-memory.svg)](https://crates.io/crates/shodh-memory)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

Give your AI agents memory that persists across sessions, learns from experience, and runs entirely on your hardware.

## Installation

```toml
[dependencies]
shodh-memory = "0.1"
```

On first use, models (~37MB) download automatically to `~/.cache/shodh-memory/`.

## Quick Start

```rust
use shodh_memory::memory::{
    MemorySystem, MemoryConfig, Experience, ExperienceType, Query, ForgetCriteria
};
use anyhow::Result;

fn main() -> Result<()> {
    // Create memory system with default config
    let config = MemoryConfig {
        storage_path: "./my_agent_data".into(),
        ..Default::default()
    };
    let memory = MemorySystem::new(config)?;

    // Store a memory
    let experience = Experience {
        content: "User prefers dark mode for all applications".to_string(),
        experience_type: ExperienceType::Decision,
        ..Default::default()
    };
    let memory_id = memory.remember(experience, None)?;
    println!("Stored memory: {:?}", memory_id);

    // Semantic search
    let query = Query::builder()
        .query_text("user interface preferences")
        .max_results(5)
        .build();

    let results = memory.recall(&query)?;
    for mem in results {
        println!("{} (importance: {:.2})", mem.experience.content, mem.importance());
    }

    // Flush to disk
    memory.flush_storage()?;

    Ok(())
}
```

## Features

- **Semantic search** - MiniLM-L6 embeddings (384-dim) for meaning-based retrieval
- **Hebbian learning** - Connections strengthen when memories co-activate
- **Activation decay** - Unused memories fade naturally (exponential decay)
- **Entity extraction** - TinyBERT NER extracts people, orgs, locations
- **Knowledge graph** - Entity relationships with spreading activation
- **3-tier architecture** - Working → Session → Long-term memory (Cowan's model)
- **100% offline** - Works on air-gapped systems after initial model download

## Experience Types

```rust
pub enum ExperienceType {
    Conversation,  // Chat interactions
    Decision,      // Choices made (+0.30 importance)
    Error,         // Failures/bugs (+0.25)
    Learning,      // New knowledge (+0.25)
    Discovery,     // Found something (+0.20)
    Pattern,       // Recurring behavior (+0.20)
    Task,          // Work items (+0.15)
    Context,       // Background info (+0.10)
    CodeEdit,      // Code changes
    FileAccess,    // File operations
    Search,        // Search queries
    Command,       // Commands executed
    Observation,   // General notes (+0.05, default)
}
```

## API Overview

### Store Memories (`remember`)

```rust
// Basic storage
let experience = Experience {
    content: "The API uses JWT for authentication".to_string(),
    experience_type: ExperienceType::Learning,
    ..Default::default()
};
let id = memory.remember(experience, None)?;

// With metadata and tags
let experience = Experience {
    content: "User prefers Rust over Python".to_string(),
    experience_type: ExperienceType::Observation,
    metadata: [("source".to_string(), "conversation".to_string())].into(),
    ..Default::default()
};
memory.remember(experience, None)?;

// With custom timestamp
use chrono::Utc;
memory.remember(experience, Some(Utc::now()))?;
```

### Retrieve Memories (`recall`)

```rust
// Semantic search
let query = Query::builder()
    .query_text("authentication methods")
    .max_results(10)
    .build();
let results = memory.recall(&query)?;

// Filter by type
let query = Query::builder()
    .query_text("errors")
    .experience_types(vec![ExperienceType::Error])
    .build();
let results = memory.recall(&query)?;

// Filter by importance
let query = Query::builder()
    .importance_threshold(0.5)
    .max_results(20)
    .build();
let results = memory.recall(&query)?;

// Convenience methods for common queries
// Recall by tags (returns all memories with ANY matching tag)
let results = memory.recall_by_tags(&["auth".to_string(), "security".to_string()], 20)?;

// Recall by date range
use chrono::{Utc, Duration};
let now = Utc::now();
let start = now - Duration::days(7);
let results = memory.recall_by_date(start, now, 50)?;

// Pagination
let query = Query::builder()
    .query_text("preferences")
    .max_results(10)
    .offset(20)  // Skip first 20
    .build();
let paginated = memory.paginated_recall(&query)?;
println!("Has more: {}", paginated.has_more);
```

### Forget Memories

```rust
use shodh_memory::memory::types::MemoryId;

// Delete by ID
memory.forget(ForgetCriteria::ById(memory_id))?;

// Delete old memories (older than N days)
memory.forget(ForgetCriteria::OlderThan(30))?;

// Delete low-importance memories
memory.forget(ForgetCriteria::LowImportance(0.1))?;

// Delete by regex pattern
memory.forget(ForgetCriteria::Pattern(r"test.*".to_string()))?;

// Delete by tags
memory.forget(ForgetCriteria::ByTags(vec!["temporary".to_string()]))?;

// Delete by date range
use chrono::{Utc, Duration};
let end = Utc::now();
let start = end - Duration::days(7);
memory.forget(ForgetCriteria::ByDateRange { start, end })?;

// GDPR: Delete everything
memory.forget(ForgetCriteria::All)?;
```

### Statistics & Maintenance

```rust
// Get stats
let stats = memory.stats();
println!("Total memories: {}", stats.total_memories);
println!("Long-term: {}", stats.long_term_memory_count);
println!("Vector indexed: {}", stats.vector_index_count);

// Storage stats
let storage_stats = memory.get_storage_stats()?;

// Flush to disk
memory.flush_storage()?;

// Run maintenance (decay old memories)
let processed = memory.run_maintenance(0.95)?;

// Index health
let health = memory.index_health();
println!("Healthy: {}", health.healthy);
```

## Configuration

```rust
let config = MemoryConfig {
    // Storage location
    storage_path: "./data".into(),

    // Working memory: hot cache for recent memories
    working_memory_size: 100,  // entries

    // Session memory: warm cache
    session_memory_size_mb: 500,  // MB

    // Per-user heap limit (prevents OOM)
    max_heap_per_user_mb: 256,  // MB

    // Auto-compress old memories
    auto_compress: true,
    compression_age_days: 30,

    // Importance threshold for long-term storage
    importance_threshold: 0.3,
};
```

Environment variables:

```bash
SHODH_MEMORY_PATH=./data
SHODH_OFFLINE=true  # Disable auto-download
RUST_LOG=info
```

## Architecture

```
Working Memory ──overflow──> Session Memory ──importance──> Long-Term Memory
   (100 items)                  (500 MB)                      (RocksDB)
```

**Cognitive processing:**
- Spreading activation retrieval
- Exponential activation decay: A(t) = A₀ · e^(-λt)
- Hebbian strengthening on co-retrieval
- Long-term potentiation (permanent connections)
- Memory replay during maintenance
- Retroactive interference detection

## Performance

Measured on Intel i7-1355U (10 cores, 1.7GHz):

| Operation | Latency |
|-----------|---------|
| `remember()` | 55-60ms |
| `recall()` semantic | 34-58ms |
| `recall_by_tags()` | ~1ms |
| `recall_by_date()` | ~2ms |
| Entity lookup | 763ns |
| Hebbian strengthen | 5.7µs |
| Graph traversal (3-hop) | 30µs |

Content-hash dedup (SHA-256) ensures identical content is never stored twice.

## Platform Support

| Platform | Status |
|----------|--------|
| Linux x86_64 | Supported |
| macOS ARM64 | Supported |
| macOS x86_64 | Supported |
| Windows x86_64 | Supported |
| Linux ARM64 | Coming soon |

## Links

- [GitHub](https://github.com/varun29ankuS/shodh-memory)
- [Documentation](https://www.shodh-rag.com/memory)
- [PyPI (Python)](https://pypi.org/project/shodh-memory/)
- [npm (MCP Server)](https://www.npmjs.com/package/@shodh/memory-mcp)

## License

Apache 2.0
