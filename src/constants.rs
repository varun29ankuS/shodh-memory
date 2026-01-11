//! Documented constants for the memory system
//!
//! This module contains all tunable parameters with justification for their values.
//! Centralizing constants prevents magic numbers and makes tuning easier.

// =============================================================================
// HEBBIAN LEARNING CONSTANTS
// Based on synaptic plasticity research: small incremental changes over time
// produce stable learning. Large changes cause instability.
// =============================================================================

/// Importance boost for helpful memories (+2.5%)
///
/// When a memory helps complete a task successfully (RetrievalOutcome::Helpful),
/// its importance is increased by this amount.
///
/// Justification:
/// - 2.5% is conservative to require many successful uses for significant impact
/// - Below biological synaptic strengthening (~3-7% per successful activation)
/// - 40 successful uses → importance increases from 0.5 to ~0.7 (compound effect)
/// - Prevents "rich get richer" effect where early memories dominate
///
/// Reference: Bi & Poo (1998) "Synaptic Modifications in Cultured Hippocampal Neurons"
pub const HEBBIAN_BOOST_HELPFUL: f32 = 0.025;

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
/// Used for resource limit calculations. Uses 2x safety margin.
///
/// Breakdown:
/// - Experience content: ~2-5KB (text, metadata)
/// - Embeddings (384 dims): 1.5KB (384 * 4 bytes)
/// - Memory struct overhead: ~500 bytes
/// - Serialization overhead: ~200 bytes
/// - Buffer for large experiences: ~4KB
///
/// Total realistic estimate: 8-10KB average
/// We use 20KB (2x safety margin) to prevent false positive resource limits
/// while still protecting against runaway memory usage.
///
/// Math: 500MB default limit / 20KB = ~25,000 memories per user
/// This is sufficient for most use cases while preventing OOM.
pub const ESTIMATED_BYTES_PER_MEMORY: usize = 20 * 1024;

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
// DENSITY-DEPENDENT RETRIEVAL WEIGHTS (SHO-26)
// Based on GraphRAG Survey (arXiv 2408.08921) - hybrid KG-Vector improves 13.1%
// Graph weight scales with density: sparse graphs get less trust, dense graphs more
// =============================================================================

/// Minimum graph weight for sparse graphs (few associations)
///
/// Justification:
/// - 0.1 (10%) for graphs with < 0.5 edges per memory
/// - Sparse graphs have low-confidence associations
/// - Semantic similarity dominates when graph is underdeveloped
pub const DENSITY_GRAPH_WEIGHT_MIN: f32 = 0.1;

/// Maximum graph weight for dense graphs (rich associations)
///
/// Justification:
/// - 0.5 (50%) for graphs with > 2.0 edges per memory
/// - Dense graphs have high-confidence Hebbian associations
/// - Graph traversal becomes primary retrieval signal
///
/// Reference: GraphRAG Survey (arXiv 2408.08921)
pub const DENSITY_GRAPH_WEIGHT_MAX: f32 = 0.5;

/// Linguistic weight for density-dependent retrieval
///
/// Justification:
/// - 0.15 (15%) fixed - exact term matching always useful
/// - Semantic weight = 1.0 - graph_weight - linguistic_weight
pub const DENSITY_LINGUISTIC_WEIGHT: f32 = 0.15;

/// Density threshold for minimum graph weight
///
/// Below this edges-per-memory ratio, use DENSITY_GRAPH_WEIGHT_MIN
pub const DENSITY_THRESHOLD_MIN: f32 = 0.5;

/// Density threshold for maximum graph weight
///
/// Above this edges-per-memory ratio, use DENSITY_GRAPH_WEIGHT_MAX
pub const DENSITY_THRESHOLD_MAX: f32 = 2.0;

// =============================================================================
// IMPORTANCE-WEIGHTED DECAY (SHO-26)
// Based on spreadr R package (Siew, 2019) and ACT-R cognitive architecture
// Important memories spread activation slower (preserve signal)
// Weak memories spread faster but decay quickly (exploratory)
// =============================================================================

/// Minimum decay rate for high-importance memories
///
/// Justification:
/// - 0.05 decay preserves ~95% activation per hop for important nodes
/// - High-importance memories (decisions, learnings) maintain strong signal
/// - Enables 6-hop traversal with meaningful activation at destination
/// - At hop 6: e^(-0.05*6) = 0.74 retention (vs 0.55 at 0.1)
///
/// Reference: spreadr R package (Siew, 2019), decay range 0.1-0.3
/// Tuning (2026-01): Lowered from 0.1 to 0.05 for deep traversal
pub const IMPORTANCE_DECAY_MIN: f32 = 0.05;

/// Maximum decay rate for low-importance memories
///
/// Justification:
/// - 0.15 decay for transient observations/context
/// - At hop 6: e^(-0.15*6) = 0.41 retention (enough signal)
/// - Preserves signal for 6-hop chains even on weak edges
/// - Still differentiates important vs transient memories
///
/// Tuning (2026-01): Lowered from 0.4 to 0.15 for deep traversal
pub const IMPORTANCE_DECAY_MAX: f32 = 0.15;

/// Type-based importance: Decision memories
///
/// Justification:
/// - 0.30 weight - highest importance
/// - Decisions represent explicit choices and preferences
/// - Critical for agent memory consistency
pub const IMPORTANCE_TYPE_DECISION: f32 = 0.30;

/// Type-based importance: Learning memories
pub const IMPORTANCE_TYPE_LEARNING: f32 = 0.25;

/// Type-based importance: Error memories
pub const IMPORTANCE_TYPE_ERROR: f32 = 0.25;

/// Type-based importance: Discovery/Pattern memories
pub const IMPORTANCE_TYPE_DISCOVERY: f32 = 0.20;

/// Type-based importance: Task memories
pub const IMPORTANCE_TYPE_TASK: f32 = 0.15;

/// Type-based importance: Context/Observation memories
pub const IMPORTANCE_TYPE_OBSERVATION: f32 = 0.10;

/// Entity presence boost for importance calculation
///
/// Justification:
/// - 0.04 per entity (max ~0.12 for 3 entities)
/// - Named entities indicate factual, retrievable content
pub const IMPORTANCE_ENTITY_BOOST: f32 = 0.04;

/// Max entities for importance boost calculation
pub const IMPORTANCE_ENTITY_MAX: usize = 3;

/// Graph connectivity boost for importance
///
/// Justification:
/// - 0.03 per connected memory (max ~0.15 for 5 connections)
/// - Well-connected memories are central to knowledge graph
pub const IMPORTANCE_CONNECTIVITY_BOOST: f32 = 0.03;

/// Max connections for importance boost calculation
pub const IMPORTANCE_CONNECTIVITY_MAX: usize = 5;

/// Recency boost for importance (within threshold)
///
/// Justification:
/// - 0.20 boost for memories within IMPORTANCE_RECENCY_DAYS
/// - Recent memories more likely to be contextually relevant
pub const IMPORTANCE_RECENCY_BOOST: f32 = 0.20;

/// Days threshold for recency importance boost
pub const IMPORTANCE_RECENCY_DAYS: f64 = 7.0;

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

// =============================================================================
// COWAN'S MODEL TIER PROMOTION CONSTANTS
// Based on Cowan (1988) "Evolving conceptions of memory storage"
// and memory consolidation research (Rasch & Born, 2013)
// Tier promotion is based on importance + time, not size
// =============================================================================

/// Minimum importance for Working → Session promotion
/// Memories must reach this threshold through Hebbian strengthening
/// or initial high-importance assignment before promotion
///
/// Justification:
/// - 0.35 allows moderately important memories to consolidate
/// - Combined with time threshold, prevents noise from entering session memory
/// - Matches ~65th percentile of memories by initial importance
pub const TIER_PROMOTION_WORKING_IMPORTANCE: f32 = 0.35;

/// Minimum age in seconds for Working → Session promotion
/// Based on early consolidation window in hippocampal memory formation
///
/// Justification:
/// - 30 minutes (1800s) matches synaptic consolidation window
/// - Allows for replay/rehearsal before promotion
/// - Short enough for practical use, long enough for consolidation
///
/// Reference: McGaugh (2000) "Memory - a century of consolidation"
pub const TIER_PROMOTION_WORKING_AGE_SECS: i64 = 1800; // 30 minutes

/// Minimum importance for Session → LongTerm promotion
/// Higher threshold ensures only well-consolidated memories persist
///
/// Justification:
/// - 0.5 requires either high initial importance or Hebbian strengthening
/// - Memories must prove their value through access patterns
/// - Matches median importance threshold for durable memories
pub const TIER_PROMOTION_SESSION_IMPORTANCE: f32 = 0.5;

/// Minimum age in seconds for Session → LongTerm promotion
/// Based on hippocampal-cortical memory transfer timeline
///
/// Justification:
/// - 24 hours (86400s) matches sleep-dependent consolidation cycle
/// - Allows for multiple replay cycles before permanent storage
/// - Hippocampal → cortical transfer primarily occurs during sleep
///
/// Reference: Rasch & Born (2013) "About Sleep's Role in Memory"
pub const TIER_PROMOTION_SESSION_AGE_SECS: i64 = 86400; // 24 hours

/// Potentiation boost applied during each maintenance cycle
/// Applied to ALL memories based on access count (Hebbian strengthening)
///
/// Justification:
/// - 0.5% per cycle is gradual (requires ~40 cycles for noticeable effect)
/// - Prevents runaway importance inflation
/// - Matches slow synaptic strengthening in biological systems
pub const POTENTIATION_MAINTENANCE_BOOST: f32 = 0.005; // 0.5% per cycle

/// Access count threshold for potentiation boost
/// Memories accessed more than this get a maintenance boost
///
/// Justification:
/// - 3 accesses indicates pattern of use, not single retrieval
/// - Prevents potentiation of noise/rarely-used memories
pub const POTENTIATION_ACCESS_THRESHOLD: u32 = 3;

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

/// Maximum hops for spreading activation (upper bound)
///
/// Justification:
/// - 6 hops captures deep conceptual chains and distant associations
/// - With aggressive decay tuning, signal survives to hop 5-6
/// - Enables discovery of non-obvious relationships
/// - Adaptive algorithm may terminate earlier (see SPREADING_EARLY_TERMINATION_*)
///
/// Tuning (2026-01): Increased from 3 to 6 for deep traversal
pub const SPREADING_MAX_HOPS: usize = 6;

/// Minimum hops before early termination is allowed
///
/// Ensures at least some spreading even when initial activation is high.
/// Prevents returning only directly connected entities.
///
/// Tuning (2026-01): Increased from 1 to 3 to guarantee deep exploration
pub const SPREADING_MIN_HOPS: usize = 3;

/// Activation threshold for pruning weak activations (initial/strict)
///
/// Justification:
/// - 0.005 allows weak but meaningful signals through
/// - Enables 6-hop traversal even with moderate edge strengths
/// - Below this, activation is truly noise
///
/// Tuning (2026-01): Lowered from 0.01 to 0.005 for deep traversal
pub const SPREADING_ACTIVATION_THRESHOLD: f32 = 0.005;

/// Relaxed activation threshold when too few candidates found
///
/// If fewer than SPREADING_MIN_CANDIDATES are activated, the threshold
/// is lowered to this value to allow more exploration.
///
/// Tuning (2026-01): Lowered from 0.005 to 0.001 for deep traversal
pub const SPREADING_RELAXED_THRESHOLD: f32 = 0.001;

/// Minimum candidates before relaxing threshold
///
/// If fewer than this many entities are activated after a hop,
/// the activation threshold is relaxed to explore more.
pub const SPREADING_MIN_CANDIDATES: usize = 5;

/// Early termination threshold - ratio of new activations
///
/// If (new_activations / total_activations) < this ratio,
/// spreading has saturated and we terminate early.
/// Value of 0.05 = less than 5% new activations → terminate.
///
/// Tuning (2026-01): Lowered from 0.1 to 0.05 to resist early termination
pub const SPREADING_EARLY_TERMINATION_RATIO: f32 = 0.05;

/// Early termination threshold - minimum candidate count
///
/// If we have at least this many candidates after minimum hops,
/// we can terminate early (we have enough coverage).
///
/// Tuning (2026-01): Increased from 20 to 50 for richer exploration
pub const SPREADING_EARLY_TERMINATION_CANDIDATES: usize = 50;

/// Activation normalization factor per hop
///
/// Prevents unbounded activation growth by normalizing per hop.
/// After each hop, activations are scaled so max = 1.0 × this factor.
/// Value > 1.0 allows some accumulation while preventing explosion.
///
/// Tuning (2026-01): Increased from 1.5 to 2.0 for more signal preservation
pub const SPREADING_NORMALIZATION_FACTOR: f32 = 2.0;

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
// HYBRID DECAY MODEL CONSTANTS (SHO-103)
// Based on Wixted & Ebbesen (1991) - power-law forgetting matches human memory
// Hybrid model: exponential for consolidation, power-law for long-term retention
// =============================================================================

/// Crossover point in days from exponential to power-law decay
///
/// Below this threshold: exponential decay (fast consolidation)
/// Above this threshold: power-law decay (slow long-term forgetting)
///
/// Justification:
/// - 3 days matches memory consolidation window in neuroscience
/// - Hippocampal-cortical transfer takes ~72 hours
/// - Short-term plasticity is exponential, long-term follows power-law
///
/// Reference: Wixted (2004) "The psychology and neuroscience of forgetting"
pub const DECAY_CROSSOVER_DAYS: f64 = 3.0;

/// Power-law exponent (β) for long-term forgetting
///
/// Formula: A(t) = A_cross × (t / t_cross)^(-β)
///
/// Justification:
/// - β = 0.5 produces moderate long-term retention
/// - Lower β (0.3) = slower forgetting, heavier tail
/// - Higher β (0.7) = faster forgetting, lighter tail
/// - 0.5 matches empirical human forgetting curves
///
/// Reference: Wixted & Ebbesen (1991), Anderson & Schooler (1991)
pub const POWERLAW_BETA: f64 = 0.5;

/// Power-law exponent for potentiated/important memories
///
/// Potentiated synapses and high-importance memories use lower β
/// for even slower forgetting (heavier tail).
///
/// Justification:
/// - 0.3 exponent means 50% retention at ~11 days vs ~4 days for β=0.5
/// - Matches LTP protection ratio (10x slower decay)
pub const POWERLAW_BETA_POTENTIATED: f64 = 0.3;

/// Exponential decay rate (λ) for consolidation phase
///
/// Used during t < DECAY_CROSSOVER_DAYS.
/// λ = ln(2) / half_life, where half_life ≈ 1 day for consolidation.
///
/// Justification:
/// - Fast initial decay clears noise and weak associations
/// - Matches short-term synaptic depression rates
/// - After 3 days at this rate: ~12.5% retention → power-law takes over
pub const DECAY_LAMBDA_CONSOLIDATION: f64 = 0.693; // ln(2) / 1.0 day

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

/// Maximum allowed compression ratio (decompressed / compressed)
///
/// Prevents zip bomb attacks where small payloads decompress to huge sizes.
/// Normal LZ4 compression ratios are typically 2:1 to 5:1 for text.
/// Ratios above 100:1 are suspicious and indicate potential attack.
///
/// Justification:
/// - LZ4 typical ratio for text: 2-5x
/// - LZ4 maximum theoretical ratio: ~255x (all zeros)
/// - 100:1 is generous for legitimate data while catching attacks
/// - Combined with MAX_DECOMPRESSED_SIZE for defense in depth
pub const MAX_COMPRESSION_RATIO: usize = 100;

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
// MEMORY REPLAY CONSTANTS (SHO-105)
// Based on sleep consolidation research: hippocampal replay during rest
// strengthens important memory traces through co-activation
// =============================================================================

/// Minimum importance score for a memory to be eligible for replay
///
/// Justification:
/// - 0.3 threshold selects top ~30% of memories by importance
/// - Low-importance memories don't benefit from replay (noise)
/// - Combined with recency, targets recent + important memories
///
/// Reference: Rasch & Born (2013) "About Sleep's Role in Memory"
pub const REPLAY_IMPORTANCE_THRESHOLD: f32 = 0.3;

/// Maximum age in days for memories eligible for replay
///
/// Justification:
/// - 7 days matches sleep consolidation window in neuroscience
/// - Older memories are already consolidated (in power-law phase)
/// - Focus replay resources on recent, unconsolidated memories
pub const REPLAY_MAX_AGE_DAYS: i64 = 7;

/// Minimum emotional arousal for priority replay
///
/// High-arousal memories (emotional events) get priority in replay queue.
///
/// Justification:
/// - 0.6 threshold selects emotionally significant events
/// - Matches amygdala modulation threshold for enhanced encoding
/// - Emotional memories benefit most from replay consolidation
///
/// Reference: LaBar & Cabeza (2006) "Cognitive neuroscience of emotional memory"
pub const REPLAY_AROUSAL_THRESHOLD: f32 = 0.6;

/// Strength boost per replay cycle for memory activation
///
/// Justification:
/// - 0.05 (5%) is small enough to require multiple replays for significant effect
/// - Matches biological synaptic strengthening rates
/// - 10 replay cycles → ~60% increase (compound effect)
pub const REPLAY_STRENGTH_BOOST: f32 = 0.05;

/// Strength boost for edges during replay co-activation
///
/// Justification:
/// - 0.08 (8%) strengthens associations during replay
/// - Slightly higher than memory boost (edges benefit more from co-activation)
/// - Simulates sleep spindle-mediated synaptic strengthening
pub const REPLAY_EDGE_BOOST: f32 = 0.08;

/// Maximum memories to replay per cycle
///
/// Justification:
/// - 50 memories per cycle limits computational cost
/// - Typical replay session processes 30-50 memories
/// - Higher values may cause maintenance cycle delays
pub const REPLAY_BATCH_SIZE: usize = 50;

/// Minimum connected memories required for replay network
///
/// Justification:
/// - 2 minimum ensures replay involves co-activation (not isolated memories)
/// - Replay benefits from network effects
/// - Isolated memories should wait for more connections
pub const REPLAY_MIN_CONNECTIONS: usize = 2;

// =============================================================================
// MEMORY INTERFERENCE CONSTANTS (SHO-106)
// Based on interference theory: similar memories compete and can disrupt each other
// =============================================================================

/// Similarity threshold for interference detection
///
/// When a new memory exceeds this similarity to an existing memory,
/// interference effects are triggered.
///
/// Justification:
/// - 0.85 cosine similarity indicates highly similar content
/// - Below this, memories are distinct enough to coexist
/// - Above this, memories compete for the same retrieval cues
///
/// Reference: Postman & Underwood (1973) "Critical issues in interference theory"
pub const INTERFERENCE_SIMILARITY_THRESHOLD: f32 = 0.85;

/// Similarity threshold for severe interference (conflicting memories)
///
/// Justification:
/// - 0.95 indicates nearly identical content
/// - At this level, one memory should dominate
/// - Retroactive interference is strongest
pub const INTERFERENCE_SEVERE_THRESHOLD: f32 = 0.95;

/// Strength reduction per interference event (retroactive)
///
/// When new memory interferes with old, old memory loses this much strength.
///
/// Justification:
/// - 0.1 (10%) is moderate - requires multiple interferences to significantly weaken
/// - Matches empirical retroactive interference rates
/// - Allows recovery if new memory is later found to be incorrect
pub const INTERFERENCE_RETROACTIVE_DECAY: f32 = 0.1;

/// Strength reduction for proactive interference
///
/// Strong old memories can suppress encoding of similar new memories.
///
/// Justification:
/// - 0.05 (5%) is weaker than retroactive (new info typically wins)
/// - Only applies when old memory is very strong (>0.8 importance)
/// - Prevents over-learning effects
pub const INTERFERENCE_PROACTIVE_DECAY: f32 = 0.05;

/// Importance threshold for proactive interference
///
/// Only memories above this importance can cause proactive interference.
///
/// Justification:
/// - 0.8 selects only very strong, well-established memories
/// - Weak memories shouldn't block new learning
/// - Matches schema-based learning effects
pub const INTERFERENCE_PROACTIVE_THRESHOLD: f32 = 0.8;

/// Competition factor during retrieval
///
/// How much similar memories suppress each other during retrieval.
///
/// Justification:
/// - 0.15 provides moderate competition
/// - Higher values cause winner-take-all behavior
/// - Lower values allow more co-retrieval
///
/// Reference: Anderson & Neely (1996) "Interference and inhibition in memory retrieval"
pub const INTERFERENCE_COMPETITION_FACTOR: f32 = 0.15;

/// Time window for interference sensitivity (hours)
///
/// New memories are most vulnerable to interference within this window.
///
/// Justification:
/// - 24 hours matches synaptic consolidation window
/// - After this, memories become more resistant to interference
/// - Combined with DECAY_CROSSOVER_DAYS for full model
pub const INTERFERENCE_VULNERABILITY_HOURS: i64 = 24;

/// Maximum interference events to track per memory
///
/// Justification:
/// - 10 events provides sufficient history
/// - Beyond this, early interference events are less relevant
/// - Limits memory overhead
pub const INTERFERENCE_MAX_TRACKED: usize = 10;

// =============================================================================
// 3-TIER GRAPH DENSITY CONSTANTS (SHO-200)
// Based on neuroscience research on hippocampal-cortical memory consolidation
// =============================================================================
//
// Research basis:
// - Dentate Gyrus (L1): 2-4% active neurons (dense, fast encoding)
// - CA1/CA3 (L2): 0.5-2.5% active (moderate, pattern separation)
// - Neocortex (L3): <1% active (sparse, long-term storage)
//
// References:
// - Engram Memory Encoding (arXiv:2506.01659)
// - Population sparseness & Hebbian plasticity (bioRxiv:2025.06.16.659837)
// - Sparse coding of episodic memory (PNAS 2014)
// =============================================================================

// === L1: WORKING MEMORY (Hippocampus/Dentate Gyrus-like) ===
// Dense connections, fast encoding, aggressive pruning

/// Target edge density for L1 (working memory tier)
///
/// 5% of possible edges should be active at any time.
/// Dense enough for rich associations, sparse enough to avoid noise.
pub const L1_TARGET_DENSITY: f32 = 0.05;

/// Initial weight for new edges in L1
///
/// New edges start weak and must prove their value through co-activation.
pub const L1_INITIAL_WEIGHT: f32 = 0.3;

/// Decay rate per hour for unused L1 edges
///
/// Aggressive pruning: 15% decay per hour if not accessed.
/// This clears noise and spurious connections quickly.
pub const L1_DECAY_PER_HOUR: f32 = 0.15;

/// Maximum age in hours before L1 edge must promote or die
///
/// After 4 hours, edges either strengthen enough to promote to L2 or get pruned.
pub const L1_MAX_AGE_HOURS: u32 = 4;

/// Minimum weight threshold for L1 edges
///
/// Edges below this are immediately pruned.
pub const L1_PRUNE_THRESHOLD: f32 = 0.1;

/// Minimum weight required to promote from L1 to L2
///
/// Only edges that reach this strength survive to episodic memory.
pub const L1_PROMOTION_THRESHOLD: f32 = 0.5;

// === L2: EPISODIC MEMORY (CA1/CA3-like) ===
// Moderate density, Hebbian learning determines survival

/// Target edge density for L2 (episodic memory tier)
///
/// 2.5% of possible edges - sparser than L1 but still associative.
pub const L2_TARGET_DENSITY: f32 = 0.025;

/// Initial weight for edges promoted to L2 from L1
///
/// Promoted edges start at 0.5 (they already proved value in L1).
pub const L2_PROMOTION_WEIGHT: f32 = 0.5;

/// Decay rate per day for unused L2 edges
///
/// Moderate decay: 10% per day if not accessed.
pub const L2_DECAY_PER_DAY: f32 = 0.10;

/// Maximum age in days before L2 edge must promote or die
///
/// After 14 days, edges either consolidate to L3 or get pruned.
pub const L2_MAX_AGE_DAYS: u32 = 14;

/// Minimum weight threshold for L2 edges
///
/// Higher bar than L1 - weak edges don't survive episodic tier.
pub const L2_PRUNE_THRESHOLD: f32 = 0.2;

/// Minimum weight required to promote from L2 to L3
///
/// Only strongly reinforced edges become permanent semantic knowledge.
pub const L2_PROMOTION_THRESHOLD: f32 = 0.7;

// === L3: SEMANTIC MEMORY (Neocortex-like) ===
// Very sparse, near-permanent, abstract associations

/// Target edge density for L3 (semantic memory tier)
///
/// <1% of possible edges - only the most important survive.
pub const L3_TARGET_DENSITY: f32 = 0.008;

/// Initial weight for edges promoted to L3 from L2
///
/// Consolidated edges are strong (0.7) and resistant to decay.
pub const L3_PROMOTION_WEIGHT: f32 = 0.7;

/// Decay rate per month for unused L3 edges
///
/// Very slow decay: 2% per month. Near-permanent.
pub const L3_DECAY_PER_MONTH: f32 = 0.02;

/// Minimum weight threshold for L3 edges
///
/// Highest bar - only strongest associations remain.
pub const L3_PRUNE_THRESHOLD: f32 = 0.3;

// === HEBBIAN STRENGTHENING ===
// Co-activation and retrieval success boost edge weights

/// Weight boost per co-access event
///
/// When two memories are retrieved together, their edge strengthens by 15%.
pub const TIER_CO_ACCESS_BOOST: f32 = 0.15;

/// Weight boost when edge contributes to successful retrieval
///
/// If traversing an edge helped answer a query correctly, +25% strength.
pub const TIER_RETRIEVAL_SUCCESS_BOOST: f32 = 0.25;

/// Threshold for Long-Term Potentiation (LTP) status
///
/// Edges above 0.8 weight are considered "potentiated" and decay even slower.
pub const TIER_LTP_THRESHOLD: f32 = 0.8;

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
