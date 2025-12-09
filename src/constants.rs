//! Documented constants for the memory system
//!
//! This module contains all tunable parameters with justification for their values.
//! Centralizing constants prevents magic numbers and makes tuning easier.

// =============================================================================
// HEBBIAN LEARNING CONSTANTS
// Based on synaptic plasticity research: small incremental changes over time
// produce stable learning. Large changes cause instability.
// =============================================================================

/// Importance boost for helpful memories (+5%)
///
/// When a memory helps complete a task successfully (RetrievalOutcome::Helpful),
/// its importance is increased by this amount.
///
/// Justification:
/// - 5% is small enough to require multiple successful uses for significant impact
/// - Matches biological synaptic strengthening (~3-7% per successful activation)
/// - 20 successful uses → importance increases from 0.5 to ~0.8 (compound effect)
/// - Prevents single lucky retrieval from dominating future searches
///
/// Reference: Bi & Poo (1998) "Synaptic Modifications in Cultured Hippocampal Neurons"
pub const HEBBIAN_BOOST_HELPFUL: f32 = 0.05;

/// Importance decay for misleading memories (-10%)
///
/// When a memory misleads or causes errors (RetrievalOutcome::Misleading),
/// its importance is reduced by this multiplicative factor.
///
/// Justification:
/// - 10% is aggressive enough to quickly demote bad memories
/// - Asymmetric with boost (2:1 ratio) because false positives are more costly
///   than false negatives in retrieval systems
/// - After 7 consecutive misleading uses: 0.5 → 0.24 (memory becomes low-priority)
/// - Multiplicative decay ensures importance never goes negative
pub const HEBBIAN_DECAY_MISLEADING: f32 = 0.10;

/// Minimum importance floor after decay
///
/// Importance never drops below this value, allowing recovery if the memory
/// becomes useful again in a different context.
///
/// Justification:
/// - 5% floor prevents complete forgetting of potentially useful memories
/// - Matches "savings" effect in human memory (relearning is faster than learning)
/// - Allows for context-dependent recovery
pub const IMPORTANCE_FLOOR: f32 = 0.05;

// =============================================================================
// MEMORY GRAPH EDGE CONSTANTS
// =============================================================================

/// Initial strength for new memory associations
///
/// When two memories are first co-activated, their edge starts at this strength.
///
/// Justification:
/// - 0.5 is neutral - not too strong, not too weak
/// - Allows for both strengthening and weakening based on usage patterns
/// - Matches the "initial synaptic strength" concept in neuroscience
pub const EDGE_INITIAL_STRENGTH: f32 = 0.5;

/// Minimum edge strength before pruning
///
/// Edges below this strength are removed during maintenance.
///
/// Justification:
/// - 5% threshold prevents memory graph from growing unboundedly
/// - Weak associations are unlikely to be useful for retrieval
/// - Matches IMPORTANCE_FLOOR for consistency
pub const EDGE_MIN_STRENGTH: f32 = 0.05;

/// Edge half-life base in hours (for time-based decay)
///
/// Associations decay exponentially with this half-life.
/// Stronger edges decay slower (adjusted by strength).
///
/// Justification:
/// - 24 hours base ensures daily cleanup of weak associations
/// - Strong edges (0.9 strength) decay much slower: ~96 hours effective half-life
/// - Matches circadian rhythms in memory consolidation
pub const EDGE_HALF_LIFE_HOURS: f64 = 24.0;

// =============================================================================
// COMPRESSION THRESHOLDS
// Based on information theory and cognitive psychology research on memory decay
// =============================================================================

/// Importance threshold for LZ4 (lossless) compression
///
/// High-importance memories (above this threshold) use only lossless compression
/// to preserve all information.
///
/// Justification:
/// - 80% importance indicates frequently accessed or explicitly important memory
/// - Top 20% of memories by importance should be preserved fully
/// - Aligns with Pareto principle: 20% of memories likely serve 80% of queries
pub const COMPRESSION_IMPORTANCE_HIGH: f32 = 0.8;

/// Importance threshold for semantic (lossy) compression
///
/// Low-importance memories below this threshold can be semantically compressed.
///
/// Justification:
/// - 50% is the median - half of memories by importance
/// - Combined with age (>30 days), targets truly obsolete information
/// - Semantic compression preserves keywords and summary for future retrieval
pub const COMPRESSION_IMPORTANCE_LOW: f32 = 0.5;

/// Age threshold in days for aggressive compression
///
/// Memories older than this AND below importance threshold get semantic compression.
///
/// Justification:
/// - 30 days matches Ebbinghaus forgetting curve plateau
/// - After 30 days, episodic details naturally fade in human memory
/// - System mimics human memory consolidation (episodic → semantic)
pub const COMPRESSION_AGE_DAYS: i64 = 30;

/// Access count threshold to skip compression
///
/// Frequently accessed memories (above this count) stay uncompressed.
///
/// Justification:
/// - 10 accesses indicates ongoing utility
/// - Avoids compressing memories that are still being actively used
/// - Cost of decompression outweighs storage savings at this usage level
pub const COMPRESSION_ACCESS_THRESHOLD: u32 = 10;

// =============================================================================
// RESOURCE LIMITS
// Based on memory profiling of typical experiences with embeddings
// =============================================================================

/// Estimated bytes per memory entry
///
/// Used for resource limit calculations. Intentionally conservative.
///
/// Breakdown:
/// - Experience content: ~2-5KB (text, metadata)
/// - Embeddings (384 dims): 1.5KB
/// - Memory struct overhead: ~500 bytes
/// - Serialization overhead: ~200 bytes
/// - Buffer for large experiences: ~4KB
///
/// Total conservative estimate: 8-10KB average
/// We use 250KB as a 25x safety margin for edge cases and future growth.
///
/// Note: This is intentionally high. Real-world measurements should tune this.
pub const ESTIMATED_BYTES_PER_MEMORY: usize = 250 * 1024;

/// Vector search candidate multiplier
///
/// When searching for N results, we retrieve N * this multiplier candidates
/// then filter down to N.
///
/// Justification:
/// - 2x accounts for ~50% filter rejection rate in typical queries
/// - Higher values waste compute; lower values may miss results
/// - Adaptive systems should tune this based on observed filter selectivity
pub const VECTOR_SEARCH_CANDIDATE_MULTIPLIER: usize = 2;

// =============================================================================
// SALIENCE SCORING WEIGHTS
// Based on cognitive psychology research on memory retrieval
// =============================================================================

/// Weight for recency in salience scoring
///
/// How much recent memories are boosted over older ones.
///
/// Justification:
/// - 1.0 baseline - recent memories have full weight
/// - Decays logarithmically with age
/// - Matches primacy/recency effects in human recall
pub const SALIENCE_RECENCY_WEIGHT: f32 = 1.0;

/// Time ranges for recency scoring (days)
///
/// - 0-7 days: Full relevance (1.0)
/// - 8-30 days: High relevance (0.7)
/// - 31-90 days: Medium relevance (0.4)
/// - 90+ days: Low relevance (0.1)
///
/// Justification:
/// - Matches Ebbinghaus forgetting curve decay rates
/// - Weekly work cycle for immediate relevance
/// - Monthly cycles for project/task context
/// - Quarterly for long-term reference
pub const RECENCY_FULL_DAYS: i64 = 7;
pub const RECENCY_HIGH_DAYS: i64 = 30;
pub const RECENCY_MEDIUM_DAYS: i64 = 90;
pub const RECENCY_HIGH_WEIGHT: f32 = 0.7;
pub const RECENCY_MEDIUM_WEIGHT: f32 = 0.4;
pub const RECENCY_LOW_WEIGHT: f32 = 0.1;

// =============================================================================
// HYBRID RETRIEVAL WEIGHTS
// =============================================================================

/// Weight for semantic similarity in hybrid retrieval
///
/// Justification:
/// - 0.5 (50%) gives semantic search primary role
/// - Semantic search handles "what" questions
pub const HYBRID_SEMANTIC_WEIGHT: f32 = 0.5;

/// Weight for graph-based activation in hybrid retrieval
///
/// Justification:
/// - 0.35 (35%) for associative context
/// - Handles "related to" and "when I was working on X" queries
pub const HYBRID_GRAPH_WEIGHT: f32 = 0.35;

/// Weight for linguistic overlap in hybrid retrieval
///
/// Justification:
/// - 0.15 (15%) for exact term matching
/// - Handles specific entity/keyword queries
pub const HYBRID_LINGUISTIC_WEIGHT: f32 = 0.15;

// =============================================================================
// SEMANTIC CONSOLIDATION THRESHOLDS
// =============================================================================

/// Minimum supporting memories to extract a semantic fact
///
/// Justification:
/// - 2 minimum ensures pattern isn't a one-off
/// - Higher values (3-5) for more confidence but slower learning
pub const CONSOLIDATION_MIN_SUPPORT: usize = 2;

/// Minimum age in days before consolidation
///
/// Justification:
/// - 7 days allows patterns to emerge through repeated use
/// - Matches weekly work cycles
pub const CONSOLIDATION_MIN_AGE_DAYS: i64 = 7;

/// Base decay period for semantic facts (days)
///
/// Unreinforced facts start decaying after this period.
///
/// Justification:
/// - 30 days base (one month without use)
/// - Extended by confidence and support count
pub const FACT_DECAY_BASE_DAYS: i64 = 30;

/// Days added to decay per support count
///
/// Justification:
/// - 7 days per supporting memory
/// - Fact with 5 supporters: 30 + 35 = 65 days before decay
pub const FACT_DECAY_PER_SUPPORT_DAYS: i64 = 7;

// =============================================================================
// DEFAULT CONFIGURATION VALUES
// =============================================================================

/// Default working memory capacity (entries)
pub const DEFAULT_WORKING_MEMORY_SIZE: usize = 100;

/// Default session memory size (MB)
pub const DEFAULT_SESSION_MEMORY_SIZE_MB: usize = 100;

/// Default max heap per user (MB)
///
/// Justification:
/// - 500MB is generous for edge devices with 4GB+ RAM
/// - Prevents single user from OOMing multi-tenant system
/// - Adjust down for Raspberry Pi (256MB) or up for servers (2GB)
pub const DEFAULT_MAX_HEAP_PER_USER_MB: usize = 500;

/// Default importance threshold for long-term storage
pub const DEFAULT_IMPORTANCE_THRESHOLD: f32 = 0.7;

/// Default compression age (days)
pub const DEFAULT_COMPRESSION_AGE_DAYS: u32 = 7;

/// Default max results for queries
pub const DEFAULT_MAX_RESULTS: usize = 10;

// =============================================================================
// SPREADING ACTIVATION CONSTANTS
// Based on Anderson & Pirolli (1984) "Spread of Activation"
// =============================================================================

/// Activation decay rate for spreading activation
///
/// Formula: A(d) = A₀ × e^(-λd) where λ = SPREADING_DECAY_RATE
///
/// Justification:
/// - 0.5 provides moderate decay - activation halves every ~1.4 hops
/// - Lower values (0.3) spread further but may activate irrelevant nodes
/// - Higher values (0.7) focus on immediate neighbors only
///
/// Reference: Anderson & Pirolli (1984), ACT-R cognitive architecture
pub const SPREADING_DECAY_RATE: f32 = 0.5;

/// Maximum hops for spreading activation
///
/// Justification:
/// - 3 hops captures "friend of a friend of a friend" relationships
/// - Beyond 3 hops, activation typically falls below threshold anyway
/// - Limits computational cost for large graphs
pub const SPREADING_MAX_HOPS: usize = 3;

/// Activation threshold for pruning weak activations
///
/// Justification:
/// - 0.01 (1%) prunes noise while preserving meaningful spread
/// - Matches MIN_STRENGTH for consistency
/// - Below this, activation contributes negligibly to final score
pub const SPREADING_ACTIVATION_THRESHOLD: f32 = 0.01;

// =============================================================================
// LONG-TERM POTENTIATION (LTP) CONSTANTS
// Based on synaptic plasticity and Hebbian learning theory
// =============================================================================

/// Learning rate for Hebbian edge strengthening
///
/// How much edge strength increases per co-activation.
///
/// Justification:
/// - 0.1 (10%) is moderate - requires ~10 co-activations to saturate
/// - Matches empirical synaptic LTP rates
///
/// Reference: Bi & Poo (1998), Hebbian learning
pub const LTP_LEARNING_RATE: f32 = 0.1;

/// Half-life in days for synapse decay without use
///
/// Justification:
/// - 14 days matches typical project/task cycles
/// - Unused associations fade but don't disappear immediately
/// - LTP edges decay 10x slower (see LTP_DECAY_FACTOR)
pub const LTP_DECAY_HALF_LIFE_DAYS: f64 = 14.0;

/// Activation count threshold for Long-Term Potentiation
///
/// After this many co-activations, the synapse becomes "potentiated"
/// and decays much slower.
///
/// Justification:
/// - 10 activations indicates consistent pattern, not coincidence
/// - Matches biological LTP threshold (~10-100 activations)
pub const LTP_THRESHOLD: u32 = 10;

/// Decay factor for potentiated synapses
///
/// Potentiated synapses decay at this fraction of the normal rate.
///
/// Justification:
/// - 0.1 means potentiated edges decay 10x slower
/// - Important associations persist longer
pub const LTP_DECAY_FACTOR: f32 = 0.1;

/// Minimum synapse strength floor
///
/// Synapses never decay below this value.
///
/// Justification:
/// - 0.01 (1%) allows recovery if pattern re-emerges
/// - Matches SPREADING_ACTIVATION_THRESHOLD
pub const LTP_MIN_STRENGTH: f32 = 0.01;

// =============================================================================
// INFORMATION CONTENT (IC) WEIGHTS
// Based on linguistic analysis for query parsing
// Reference: Lioma & Ounis (2006) "Information Content Weighting"
// =============================================================================

/// Information content weight for nouns
///
/// Nouns are the most discriminative in queries.
///
/// Justification:
/// - 2.3 matches empirical IC measurements for English nouns
/// - Nouns carry the core semantic meaning ("Rust", "database", "memory")
pub const IC_NOUN: f32 = 2.3;

/// Information content weight for adjectives
///
/// Adjectives provide discriminative context.
///
/// Justification:
/// - 1.7 reflects moderate discriminative power
/// - Adjectives narrow down ("fast database" vs "reliable database")
pub const IC_ADJECTIVE: f32 = 1.7;

/// Information content weight for verbs
///
/// Verbs are less discriminative ("bus stops" in IR terminology).
///
/// Justification:
/// - 1.0 baseline weight
/// - Common verbs like "is", "has", "uses" add little discriminative value
pub const IC_VERB: f32 = 1.0;

// =============================================================================
// SERVER TIMEOUT CONSTANTS
// =============================================================================

/// Graceful shutdown timeout in seconds
///
/// Maximum time to wait for active requests to drain.
///
/// Justification:
/// - 30 seconds allows long-running requests to complete
/// - Beyond this, force shutdown to avoid hanging
pub const GRACEFUL_SHUTDOWN_TIMEOUT_SECS: u64 = 30;

/// Database flush timeout in seconds
///
/// Maximum time to wait for RocksDB flush on shutdown.
///
/// Justification:
/// - 10 seconds is sufficient for typical write buffers
/// - Prevents data loss on clean shutdown
pub const DATABASE_FLUSH_TIMEOUT_SECS: u64 = 10;

/// Vector index save timeout in seconds
///
/// Maximum time to wait for HNSW index persistence.
///
/// Justification:
/// - 10 seconds handles typical index sizes
/// - Large indices may need longer, but startup rebuild is fallback
pub const VECTOR_INDEX_SAVE_TIMEOUT_SECS: u64 = 10;

// =============================================================================
// COMPRESSION SAFETY LIMITS
// =============================================================================

/// Maximum decompressed size in bytes (safety limit)
///
/// Prevents zip bomb attacks and memory exhaustion.
///
/// Justification:
/// - 10MB is generous for any single memory entry
/// - Larger content should be chunked at ingestion time
pub const MAX_DECOMPRESSED_SIZE: i32 = 10 * 1024 * 1024;

// =============================================================================
// PREFETCH RECENCY BOOST CONSTANTS
// For anticipatory prefetch relevance scoring
// =============================================================================

/// Hours threshold for full recency boost in prefetch
///
/// Memories younger than this get the full recency boost.
///
/// Justification:
/// - 24 hours captures "today's context"
/// - Recent memories are most likely to be relevant
pub const PREFETCH_RECENCY_FULL_HOURS: i64 = 24;

/// Hours threshold for partial recency boost in prefetch
///
/// Memories between FULL and this threshold get partial boost.
///
/// Justification:
/// - 168 hours = 1 week
/// - Weekly work cycles are common patterns
pub const PREFETCH_RECENCY_PARTIAL_HOURS: i64 = 168;

/// Recency boost for very recent memories (< 24h)
pub const PREFETCH_RECENCY_FULL_BOOST: f32 = 0.1;

/// Recency boost for recent memories (24h - 1 week)
pub const PREFETCH_RECENCY_PARTIAL_BOOST: f32 = 0.05;

// =============================================================================
// CONSTANTS USAGE DOCUMENTATION
// =============================================================================
//
// This section documents where each constant is used in the codebase.
// Updated: 2025-12-09
//
// ## Hebbian Learning Constants
// | Constant                  | File                      | Function/Context                    |
// |---------------------------|---------------------------|-------------------------------------|
// | HEBBIAN_BOOST_HELPFUL     | memory/types.rs           | Memory::boost_importance()          |
// | HEBBIAN_DECAY_MISLEADING  | memory/types.rs           | Memory::decay_importance()          |
// | IMPORTANCE_FLOOR          | memory/types.rs           | Memory::decay_importance() - floor  |
//
// ## Memory Graph Edge Constants
// | Constant                  | File                      | Function/Context                    |
// |---------------------------|---------------------------|-------------------------------------|
// | EDGE_INITIAL_STRENGTH     | memory/retrieval.rs       | EdgeWeight::default()               |
// | EDGE_MIN_STRENGTH         | memory/retrieval.rs       | EdgeWeight::decay(), find_assoc()   |
// | EDGE_HALF_LIFE_HOURS      | memory/retrieval.rs       | EdgeWeight::decay()                 |
//
// ## Compression Constants
// | Constant                      | File                  | Function/Context                    |
// |-------------------------------|---------------------- |-------------------------------------|
// | COMPRESSION_IMPORTANCE_HIGH   | memory/compression.rs | should_compress() - LZ4 threshold   |
// | COMPRESSION_IMPORTANCE_LOW    | memory/compression.rs | should_compress() - semantic thresh |
// | COMPRESSION_AGE_DAYS          | memory/compression.rs | should_compress() - age check       |
// | COMPRESSION_ACCESS_THRESHOLD  | memory/compression.rs | should_compress() - access count    |
// | MAX_DECOMPRESSED_SIZE         | memory/compression.rs | decompress() - safety limit         |
//
// ## Vector Search Constants
// | Constant                          | File                | Function/Context                  |
// |-----------------------------------|---------------------|-----------------------------------|
// | VECTOR_SEARCH_CANDIDATE_MULTIPLIER| memory/retrieval.rs | search_ids(), similarity_search() |
// |                                   | main.rs             | hybrid recall query building      |
// | ESTIMATED_BYTES_PER_MEMORY        | (unused)            | Resource estimation               |
//
// ## Salience/Recency Scoring Constants (Ebbinghaus Forgetting Curve)
// | Constant                  | File                | Function/Context                      |
// |---------------------------|---------------------|---------------------------------------|
// | SALIENCE_RECENCY_WEIGHT   | memory/types.rs     | Memory::salience_score()              |
// | RECENCY_FULL_DAYS         | memory/types.rs     | Memory::salience_score() - 7 day tier |
// | RECENCY_HIGH_DAYS         | memory/types.rs     | Memory::salience_score() - 30 day     |
// | RECENCY_MEDIUM_DAYS       | memory/types.rs     | Memory::salience_score() - 90 day     |
// | RECENCY_HIGH_WEIGHT       | memory/types.rs     | Memory::salience_score() - 0.7        |
// | RECENCY_MEDIUM_WEIGHT     | memory/types.rs     | Memory::salience_score() - 0.4        |
// | RECENCY_LOW_WEIGHT        | memory/types.rs     | Memory::salience_score() - 0.1        |
//
// ## Hybrid Retrieval Weights
// | Constant                  | File                      | Function/Context                  |
// |---------------------------|---------------------------|-----------------------------------|
// | HYBRID_SEMANTIC_WEIGHT    | memory/graph_retrieval.rs | spreading_activation_retrieve()   |
// | HYBRID_GRAPH_WEIGHT       | memory/graph_retrieval.rs | spreading_activation_retrieve()   |
// | HYBRID_LINGUISTIC_WEIGHT  | memory/graph_retrieval.rs | spreading_activation_retrieve()   |
//
// ## Semantic Consolidation Constants
// | Constant                      | File                  | Function/Context                  |
// |-------------------------------|---------------------- |-----------------------------------|
// | CONSOLIDATION_MIN_SUPPORT     | memory/compression.rs | consolidate_semantic_facts()      |
// | CONSOLIDATION_MIN_AGE_DAYS    | memory/compression.rs | consolidate_semantic_facts()      |
// | FACT_DECAY_BASE_DAYS          | memory/compression.rs | fact decay calculation            |
// | FACT_DECAY_PER_SUPPORT_DAYS   | memory/compression.rs | fact decay per support            |
//
// ## Default Configuration Constants
// | Constant                      | File                | Function/Context                    |
// |-------------------------------|---------------------|-------------------------------------|
// | DEFAULT_WORKING_MEMORY_SIZE   | memory/types.rs     | MemoryConfig::default()             |
// | DEFAULT_SESSION_MEMORY_SIZE_MB| memory/types.rs     | MemoryConfig::default()             |
// | DEFAULT_MAX_HEAP_PER_USER_MB  | memory/types.rs     | MemoryConfig::default()             |
// | DEFAULT_IMPORTANCE_THRESHOLD  | memory/types.rs     | MemoryConfig::default()             |
// | DEFAULT_COMPRESSION_AGE_DAYS  | memory/types.rs     | MemoryConfig::default()             |
// | DEFAULT_MAX_RESULTS           | memory/types.rs     | Query::default()                    |
//
// ## Spreading Activation Constants (Anderson & Pirolli 1984)
// | Constant                      | File                      | Function/Context                |
// |-------------------------------|---------------------------|---------------------------------|
// | SPREADING_DECAY_RATE          | memory/graph_retrieval.rs | spreading_activation_retrieve() |
// | SPREADING_MAX_HOPS            | memory/graph_retrieval.rs | spreading_activation_retrieve() |
// | SPREADING_ACTIVATION_THRESHOLD| memory/graph_retrieval.rs | spreading_activation_retrieve() |
//
// ## Long-Term Potentiation (LTP) Constants
// | Constant                  | File              | Function/Context                      |
// |---------------------------|-------------------|---------------------------------------|
// | LTP_LEARNING_RATE         | graph_memory.rs   | Synapse::activate()                   |
// | LTP_DECAY_HALF_LIFE_DAYS  | graph_memory.rs   | Synapse::decay()                      |
// | LTP_THRESHOLD             | graph_memory.rs   | Synapse::is_potentiated()             |
// | LTP_DECAY_FACTOR          | graph_memory.rs   | Synapse::decay() - potentiated rate   |
// | LTP_MIN_STRENGTH          | graph_memory.rs   | Synapse::decay() - minimum floor      |
//
// ## Information Content (IC) Weights (Lioma & Ounis 2006)
// | Constant      | File                    | Function/Context                        |
// |---------------|-------------------------|-----------------------------------------|
// | IC_NOUN       | memory/query_parser.rs  | analyze_query() - noun IC weight        |
// | IC_ADJECTIVE  | memory/query_parser.rs  | analyze_query() - adjective IC weight   |
// | IC_VERB       | memory/query_parser.rs  | analyze_query() - verb IC weight        |
//
// ## Server Timeout Constants
// | Constant                      | File      | Function/Context                        |
// |-------------------------------|-----------|-----------------------------------------|
// | GRACEFUL_SHUTDOWN_TIMEOUT_SECS| main.rs   | graceful_shutdown() - request drain     |
// | DATABASE_FLUSH_TIMEOUT_SECS   | main.rs   | graceful_shutdown() - RocksDB flush     |
// | VECTOR_INDEX_SAVE_TIMEOUT_SECS| main.rs   | graceful_shutdown() - HNSW persist      |
//
// ## Prefetch Recency Constants
// | Constant                      | File                | Function/Context                    |
// |-------------------------------|---------------------|-------------------------------------|
// | PREFETCH_RECENCY_FULL_HOURS   | memory/retrieval.rs | AnticipatoryPrefetch::relevance()   |
// | PREFETCH_RECENCY_PARTIAL_HOURS| memory/retrieval.rs | AnticipatoryPrefetch::relevance()   |
// | PREFETCH_RECENCY_FULL_BOOST   | memory/retrieval.rs | AnticipatoryPrefetch::relevance()   |
// | PREFETCH_RECENCY_PARTIAL_BOOST| memory/retrieval.rs | AnticipatoryPrefetch::relevance()   |
//
// =============================================================================
