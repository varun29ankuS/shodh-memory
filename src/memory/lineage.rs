//! Decision Lineage Graph - Causal Memory System (SHO-118)
//!
//! Transforms Shodh from a memory system into a reasoning system by tracking
//! causal relationships between memories. Enables:
//!
//! 1. "Why" Audit Trail - Trace decisions back to root causes
//! 2. Lineage Branching - Git-like branches when projects pivot
//! 3. Automatic Post-Mortems - Synthesize learnings on task completion
//!
//! Storage schema:
//! - `lineage:edges:{user_id}:{edge_id}` - Causal edges between memories
//! - `lineage:by_from:{user_id}:{from_id}:{edge_id}` - Index by source memory
//! - `lineage:by_to:{user_id}:{to_id}:{edge_id}` - Index by target memory
//! - `lineage:branches:{user_id}:{branch_id}` - Branch metadata

use anyhow::Result;
use chrono::{DateTime, Utc};
use rocksdb::{IteratorMode, DB};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use uuid::Uuid;

use super::types::{ExperienceType, Memory, MemoryId};

/// Causal relationship types between memories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CausalRelation {
    /// Error/Bug caused a Todo to be created
    Caused,
    /// Todo was resolved by a Learning/Fix
    ResolvedBy,
    /// Decision was informed by a Learning/Discovery
    InformedBy,
    /// Old decision/pattern was superseded by new one
    SupersededBy,
    /// Discovery/Learning triggered a Todo
    TriggeredBy,
    /// Memory branched from another (project pivot)
    BranchedFrom,
    /// Generic relation when type is unclear
    RelatedTo,
}

impl CausalRelation {
    /// Get the inverse relation for bidirectional traversal.
    ///
    /// When edge A→B has relation R, traversing B→A should show R.inverse().
    /// True pairs: Caused↔ResolvedBy. Directional relations (InformedBy,
    /// TriggeredBy) are self-inverse because the from/to already encodes
    /// directionality — "A informed-by B" reversed is "B informed A".
    pub fn inverse(&self) -> Self {
        match self {
            CausalRelation::Caused => CausalRelation::ResolvedBy,
            CausalRelation::ResolvedBy => CausalRelation::Caused,
            CausalRelation::InformedBy => CausalRelation::InformedBy,
            CausalRelation::TriggeredBy => CausalRelation::TriggeredBy,
            CausalRelation::SupersededBy => CausalRelation::SupersededBy,
            CausalRelation::BranchedFrom => CausalRelation::BranchedFrom,
            CausalRelation::RelatedTo => CausalRelation::RelatedTo,
        }
    }

    /// Human-readable description of the relationship
    pub fn description(&self) -> &'static str {
        match self {
            CausalRelation::Caused => "caused",
            CausalRelation::ResolvedBy => "was resolved by",
            CausalRelation::InformedBy => "was informed by",
            CausalRelation::SupersededBy => "was superseded by",
            CausalRelation::TriggeredBy => "triggered",
            CausalRelation::BranchedFrom => "branched from",
            CausalRelation::RelatedTo => "is related to",
        }
    }
}

/// Source of a lineage edge
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineageSource {
    /// Automatically inferred by the system
    Inferred,
    /// Explicitly confirmed by user/agent
    Confirmed,
    /// Manually added by user/agent
    Explicit,
}

/// A causal edge between two memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageEdge {
    /// Unique edge identifier
    pub id: String,
    /// Source memory (cause)
    pub from: MemoryId,
    /// Target memory (effect)
    pub to: MemoryId,
    /// Type of causal relationship
    pub relation: CausalRelation,
    /// Confidence in this causal link (0.0-1.0)
    pub confidence: f32,
    /// How this edge was created
    pub source: LineageSource,
    /// Branch this edge belongs to (None = main branch)
    pub branch_id: Option<String>,
    /// When the edge was created
    pub created_at: DateTime<Utc>,
    /// Last time this edge was reinforced/confirmed
    pub last_reinforced: DateTime<Utc>,
    /// Number of times this edge was reinforced
    pub reinforcement_count: u32,
}

impl LineageEdge {
    /// Create a new inferred edge
    pub fn inferred(
        from: MemoryId,
        to: MemoryId,
        relation: CausalRelation,
        confidence: f32,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            from,
            to,
            relation,
            confidence,
            source: LineageSource::Inferred,
            branch_id: None,
            created_at: now,
            last_reinforced: now,
            reinforcement_count: 1,
        }
    }

    /// Create a new explicit edge
    pub fn explicit(from: MemoryId, to: MemoryId, relation: CausalRelation) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            from,
            to,
            relation,
            confidence: 1.0, // Explicit edges have full confidence
            source: LineageSource::Explicit,
            branch_id: None,
            created_at: now,
            last_reinforced: now,
            reinforcement_count: 1,
        }
    }

    /// Confirm an inferred edge
    pub fn confirm(&mut self) {
        self.source = LineageSource::Confirmed;
        self.confidence = 1.0;
        self.last_reinforced = Utc::now();
        self.reinforcement_count += 1;
    }

    /// Reinforce this edge (increase confidence)
    pub fn reinforce(&mut self) {
        self.confidence = (self.confidence + 0.1).min(1.0);
        self.last_reinforced = Utc::now();
        self.reinforcement_count += 1;
    }

    /// Weaken this edge (decrease confidence via multiplicative decay).
    ///
    /// Asymmetric with reinforce: weakening is multiplicative (×0.90, −10%) while
    /// strengthening is additive (+0.1). This follows van Rossum et al. (2000)'s
    /// multiplicative LTD model where depression scales with current weight.
    ///
    /// The 10% per event sits at the conservative end of Dudek & Bear (1992)'s
    /// validated range of 10–30% long-term depression per induction. At w=1.0
    /// the depression/potentiation ratio is 1.0:1 (symmetric), rising to 2.25:1
    /// at w=0.45 — closer to Song, Miller & Abbott (2000)'s steady-state ratio
    /// of ~1.05:1 than the previous 0.85 multiplier which produced 1.5:1 at w=1.0.
    ///
    /// Pruning threshold 0.05 is in the 5th percentile range suggested by
    /// Chechik (1998) for optimal memory capacity in sparse networks.
    ///
    /// Returns true if the edge should be pruned (confidence dropped below 0.05).
    /// Callers should delete pruned edges to prevent the lineage graph from
    /// accumulating zombie edges with negligible confidence.
    ///
    /// References:
    /// - Dudek & Bear (1992) "Homosynaptic long-term depression" — 10-30% LTD
    /// - van Rossum et al. (2000) "Stable Hebbian learning from spike timing" — multiplicative LTD
    /// - Song, Miller & Abbott (2000) "Competitive Hebbian learning" — ~1.05:1 ratio
    /// - Chechik (1998) "Synaptic pruning in development" — 5-50th percentile threshold
    pub fn weaken(&mut self) -> bool {
        self.confidence *= 0.90;
        self.last_reinforced = Utc::now();
        self.confidence < 0.05
    }
}

/// A branch in the lineage graph (for project pivots)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageBranch {
    /// Unique branch identifier
    pub id: String,
    /// Human-readable branch name
    pub name: String,
    /// Description of what this branch represents
    pub description: Option<String>,
    /// Parent branch (None for main branch)
    pub parent_branch: Option<String>,
    /// Memory where this branch diverged from parent
    pub branch_point: Option<MemoryId>,
    /// When the branch was created
    pub created_at: DateTime<Utc>,
    /// Whether this branch is currently active
    pub active: bool,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl LineageBranch {
    /// Create the main branch
    pub fn main() -> Self {
        Self {
            id: "main".to_string(),
            name: "Main".to_string(),
            description: Some("Primary project lineage".to_string()),
            parent_branch: None,
            branch_point: None,
            created_at: Utc::now(),
            active: true,
            tags: vec![],
        }
    }

    /// Create a new branch from a parent
    pub fn new(
        name: &str,
        parent: &str,
        branch_point: MemoryId,
        description: Option<&str>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            description: description.map(|s| s.to_string()),
            parent_branch: Some(parent.to_string()),
            branch_point: Some(branch_point),
            created_at: Utc::now(),
            active: true,
            tags: vec![],
        }
    }
}

/// Result of lineage trace operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageTrace {
    /// The memory we started from
    pub root: MemoryId,
    /// Direction of traversal
    pub direction: TraceDirection,
    /// Edges in the trace (ordered by distance from root)
    pub edges: Vec<LineageEdge>,
    /// Memory IDs in traversal order
    pub path: Vec<MemoryId>,
    /// Total depth traversed
    pub depth: usize,
}

/// Direction for lineage traversal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceDirection {
    /// Trace backward to find causes
    Backward,
    /// Trace forward to find effects
    Forward,
    /// Trace in both directions
    Both,
}

/// Configuration for lineage inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Maximum days between memories for causal inference
    pub max_temporal_gap_days: i64,
    /// Minimum entity overlap for causal inference
    pub min_entity_overlap: f32,
    /// Confidence thresholds for each relation type
    pub relation_confidence: HashMap<CausalRelation, f32>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        use crate::constants::*;

        let mut relation_confidence = HashMap::new();
        relation_confidence.insert(CausalRelation::Caused, LINEAGE_CONFIDENCE_CAUSED);
        relation_confidence.insert(CausalRelation::ResolvedBy, LINEAGE_CONFIDENCE_RESOLVED_BY);
        relation_confidence.insert(CausalRelation::InformedBy, LINEAGE_CONFIDENCE_INFORMED_BY);
        relation_confidence.insert(
            CausalRelation::SupersededBy,
            LINEAGE_CONFIDENCE_SUPERSEDED_BY,
        );
        relation_confidence.insert(CausalRelation::TriggeredBy, LINEAGE_CONFIDENCE_TRIGGERED_BY);
        relation_confidence.insert(
            CausalRelation::BranchedFrom,
            LINEAGE_CONFIDENCE_BRANCHED_FROM,
        );
        relation_confidence.insert(CausalRelation::RelatedTo, LINEAGE_CONFIDENCE_RELATED_TO);

        Self {
            max_temporal_gap_days: LINEAGE_MAX_TEMPORAL_GAP_DAYS,
            min_entity_overlap: LINEAGE_MIN_ENTITY_OVERLAP,
            relation_confidence,
        }
    }
}

/// Statistics about the lineage graph
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LineageStats {
    pub total_edges: usize,
    pub inferred_edges: usize,
    pub confirmed_edges: usize,
    pub explicit_edges: usize,
    pub total_branches: usize,
    pub active_branches: usize,
    pub edges_by_relation: HashMap<String, usize>,
    pub avg_confidence: f32,
}

/// The Lineage Graph - stores and infers causal relationships
pub struct LineageGraph {
    db: Arc<DB>,
    config: InferenceConfig,
}

impl LineageGraph {
    /// Create a new lineage graph backed by RocksDB
    pub fn new(db: Arc<DB>) -> Self {
        Self {
            db,
            config: InferenceConfig::default(),
        }
    }

    /// Create with custom inference config
    pub fn with_config(db: Arc<DB>, config: InferenceConfig) -> Self {
        Self { db, config }
    }

    // =========================================================================
    // EDGE STORAGE
    // =========================================================================

    /// Store a lineage edge
    pub fn store_edge(&self, user_id: &str, edge: &LineageEdge) -> Result<()> {
        // Primary storage
        let key = format!("lineage:edges:{}:{}", user_id, edge.id);
        let value = crate::serialization::encode(edge)?;
        self.db.put(key.as_bytes(), &value)?;

        // Index by source (from)
        let from_key = format!("lineage:by_from:{}:{}:{}", user_id, edge.from.0, edge.id);
        self.db.put(from_key.as_bytes(), edge.id.as_bytes())?;

        // Index by target (to)
        let to_key = format!("lineage:by_to:{}:{}:{}", user_id, edge.to.0, edge.id);
        self.db.put(to_key.as_bytes(), edge.id.as_bytes())?;

        Ok(())
    }

    /// Get an edge by ID
    pub fn get_edge(&self, user_id: &str, edge_id: &str) -> Result<Option<LineageEdge>> {
        let key = format!("lineage:edges:{}:{}", user_id, edge_id);
        match self.db.get(key.as_bytes())? {
            Some(data) => {
                let (edge, _) = crate::serialization::try_decode::<LineageEdge>(&data)?;
                Ok(Some(edge))
            }
            None => Ok(None),
        }
    }

    /// Delete an edge (for rejection)
    pub fn delete_edge(&self, user_id: &str, edge_id: &str) -> Result<bool> {
        if let Some(edge) = self.get_edge(user_id, edge_id)? {
            // Delete indices
            let from_key = format!("lineage:by_from:{}:{}:{}", user_id, edge.from.0, edge_id);
            self.db.delete(from_key.as_bytes())?;

            let to_key = format!("lineage:by_to:{}:{}:{}", user_id, edge.to.0, edge_id);
            self.db.delete(to_key.as_bytes())?;

            // Delete primary
            let key = format!("lineage:edges:{}:{}", user_id, edge_id);
            self.db.delete(key.as_bytes())?;

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get all edges from a memory (outgoing)
    pub fn get_edges_from(&self, user_id: &str, memory_id: &MemoryId) -> Result<Vec<LineageEdge>> {
        let prefix = format!("lineage:by_from:{}:{}:", user_id, memory_id.0);
        self.get_edges_by_prefix(user_id, &prefix)
    }

    /// Get all edges to a memory (incoming)
    pub fn get_edges_to(&self, user_id: &str, memory_id: &MemoryId) -> Result<Vec<LineageEdge>> {
        let prefix = format!("lineage:by_to:{}:{}:", user_id, memory_id.0);
        self.get_edges_by_prefix(user_id, &prefix)
    }

    /// Helper to get edges by index prefix
    fn get_edges_by_prefix(&self, user_id: &str, prefix: &str) -> Result<Vec<LineageEdge>> {
        let mut edges = Vec::new();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(prefix) {
                break;
            }

            let edge_id = String::from_utf8_lossy(&value);
            if let Some(edge) = self.get_edge(user_id, &edge_id)? {
                edges.push(edge);
            }
        }

        Ok(edges)
    }

    /// List all edges for a user
    pub fn list_edges(&self, user_id: &str, limit: usize) -> Result<Vec<LineageEdge>> {
        let prefix = format!("lineage:edges:{}:", user_id);
        let mut edges = Vec::new();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            if let Ok((edge, _)) = crate::serialization::try_decode::<LineageEdge>(&value) {
                edges.push(edge);
                if edges.len() >= limit {
                    break;
                }
            }
        }

        // Sort by creation time (newest first)
        edges.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(edges)
    }

    // =========================================================================
    // BRANCH MANAGEMENT
    // =========================================================================

    /// Store a branch
    pub fn store_branch(&self, user_id: &str, branch: &LineageBranch) -> Result<()> {
        let key = format!("lineage:branches:{}:{}", user_id, branch.id);
        let value = crate::serialization::encode(branch)?;
        self.db.put(key.as_bytes(), &value)?;
        Ok(())
    }

    /// Get a branch by ID
    pub fn get_branch(&self, user_id: &str, branch_id: &str) -> Result<Option<LineageBranch>> {
        let key = format!("lineage:branches:{}:{}", user_id, branch_id);
        match self.db.get(key.as_bytes())? {
            Some(data) => {
                let (branch, _) = crate::serialization::try_decode::<LineageBranch>(&data)?;
                Ok(Some(branch))
            }
            None => Ok(None),
        }
    }

    /// List all branches for a user
    pub fn list_branches(&self, user_id: &str) -> Result<Vec<LineageBranch>> {
        let prefix = format!("lineage:branches:{}:", user_id);
        let mut branches = Vec::new();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            if let Ok((branch, _)) = crate::serialization::try_decode::<LineageBranch>(&value) {
                branches.push(branch);
            }
        }

        // Sort by creation time (newest first)
        branches.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(branches)
    }

    /// Create a new branch from current state
    pub fn create_branch(
        &self,
        user_id: &str,
        name: &str,
        parent_branch: &str,
        branch_point: MemoryId,
        description: Option<&str>,
    ) -> Result<LineageBranch> {
        let branch = LineageBranch::new(name, parent_branch, branch_point, description);
        self.store_branch(user_id, &branch)?;
        Ok(branch)
    }

    /// Ensure main branch exists for user
    pub fn ensure_main_branch(&self, user_id: &str) -> Result<()> {
        if self.get_branch(user_id, "main")?.is_none() {
            self.store_branch(user_id, &LineageBranch::main())?;
        }
        Ok(())
    }

    // =========================================================================
    // LINEAGE INFERENCE ENGINE
    // =========================================================================

    /// Infer causal relationship between two memories.
    ///
    /// Uses three complementary signals:
    /// 1. **Type-pair rules**: ExperienceType combinations → CausalRelation + base confidence
    /// 2. **Semantic overlap**: max(Jaccard entity overlap, cosine embedding similarity)
    /// 3. **Temporal proximity**: linear decay over the temporal gap
    ///
    /// The semantic overlap combines entity tags (precise but brittle — exact string match)
    /// with embedding similarity (fuzzy but robust — captures synonyms and paraphrases).
    /// Using `max()` ensures we never regress when entities work well, while rescuing
    /// cases where NER fails or different surface forms describe the same concept.
    pub fn infer_relation(&self, from: &Memory, to: &Memory) -> Option<(CausalRelation, f32)> {
        // Must be in temporal order (from before to)
        if from.created_at >= to.created_at {
            return None;
        }

        // Check temporal gap
        let gap = to.created_at.signed_duration_since(from.created_at);
        if gap.num_days() > self.config.max_temporal_gap_days {
            return None;
        }

        // Signal 1: Entity overlap (Jaccard on entity tags)
        let entity_overlap =
            Self::calculate_entity_overlap(&from.experience.entities, &to.experience.entities);

        // Signal 2: Embedding similarity (cosine between content embeddings)
        let embedding_sim = match (&from.experience.embeddings, &to.experience.embeddings) {
            (Some(emb_a), Some(emb_b)) if emb_a.len() == emb_b.len() && !emb_a.is_empty() => {
                Self::cosine_similarity(emb_a, emb_b).max(0.0) // clamp negatives
            }
            _ => 0.0,
        };

        // Combined semantic signal: best of entity overlap and embedding similarity.
        // This ensures neither signal pathway can suppress the other.
        let semantic_signal = entity_overlap.max(embedding_sim);

        // Gate: when we have both entities AND embeddings, require minimum semantic signal.
        // When either is missing, let inference proceed at whatever signal we have.
        let has_entities =
            !from.experience.entities.is_empty() && !to.experience.entities.is_empty();
        let has_embeddings =
            from.experience.embeddings.is_some() && to.experience.embeddings.is_some();

        if has_entities && !has_embeddings && entity_overlap < self.config.min_entity_overlap {
            // Only entities available, and they don't overlap enough
            return None;
        }
        if has_embeddings && !has_entities && embedding_sim < self.config.min_entity_overlap {
            // Only embeddings available, and similarity too low
            return None;
        }
        if has_entities && has_embeddings && semantic_signal < self.config.min_entity_overlap {
            // Both signals available, but neither reaches threshold
            return None;
        }

        // Infer based on memory types
        let (relation, base_confidence) = self.infer_by_types(
            &from.experience.experience_type,
            &to.experience.experience_type,
        )?;

        // Compute effective overlap for confidence scaling.
        // When no entities and no embeddings, use a floor of 0.3 to avoid zeroing out.
        let effective_overlap = if semantic_signal > 0.0 {
            semantic_signal
        } else if has_entities {
            entity_overlap
        } else {
            0.3 // floor for memories with no semantic signals
        };

        // Signal 3: Temporal proximity (linear decay)
        let temporal_factor =
            1.0 - (gap.num_days() as f32 / self.config.max_temporal_gap_days as f32);
        let confidence = base_confidence * effective_overlap * (0.5 + 0.5 * temporal_factor);

        Some((relation, confidence))
    }

    /// Cosine similarity between two embedding vectors.
    /// Delegates to the SIMD-optimized implementation in similarity.rs.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        crate::similarity::cosine_similarity(a, b)
    }

    /// Infer relation based on memory types.
    ///
    /// Type-pair modifiers (0.70–0.90) for bridge types (Observation, Conversation)
    /// encode source reliability following Johnson & Raye (1981)'s Source Monitoring
    /// Framework: memories from different sources carry different diagnostic weight
    /// for causal attribution. Specific modifier rationale:
    ///
    /// - **0.90** (Conversation → Decision): Conversations are high-fidelity causal
    ///   sources — they carry explicit reasoning and intent. Closest to "direct
    ///   experience" in the SMF hierarchy.
    ///
    /// - **0.85** (Observation/Conversation → Task/Learning/Discovery): Moderate
    ///   reliability — observations and discussions often precede work but the causal
    ///   link is indirect (correlation, not demonstrated causation).
    ///
    /// - **0.75** (Observation → Error): Observations rarely *cause* errors directly;
    ///   they *reveal* pre-existing conditions. The lower modifier reflects this
    ///   weaker causal claim (closer to "associated with" than "caused by").
    ///
    /// - **0.70** (Conversation → Error): Weakest bridge — conversations surfacing
    ///   bugs is informational, not causal. Bovens & Hartmann (2003) show that
    ///   indirect testimony requires larger discounts than direct observation.
    ///
    /// These are engineering choices grounded in the SMF taxonomy but not
    /// empirically calibrated. The confidence formula (base × modifier × overlap ×
    /// temporal) means modifiers affect the starting point, not the final ranking,
    /// which is dominated by entity overlap and temporal proximity.
    ///
    /// References:
    /// - Johnson & Raye (1981) "Reality monitoring" — source monitoring framework
    /// - Bovens & Hartmann (2003) "Bayesian Epistemology" — testimony reliability
    fn infer_by_types(
        &self,
        from_type: &ExperienceType,
        to_type: &ExperienceType,
    ) -> Option<(CausalRelation, f32)> {
        use ExperienceType::*;

        match (from_type, to_type) {
            // Error → Todo = Caused
            (Error, Task) => Some((
                CausalRelation::Caused,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::Caused)
                    .unwrap_or(&0.8),
            )),

            // Todo → Learning = ResolvedBy (when todo leads to learning)
            (Task, Learning) => Some((
                CausalRelation::ResolvedBy,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::ResolvedBy)
                    .unwrap_or(&0.85),
            )),

            // Learning → Decision = InformedBy
            (Learning, Decision) | (Discovery, Decision) => Some((
                CausalRelation::InformedBy,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::InformedBy)
                    .unwrap_or(&0.7),
            )),

            // Decision → Decision = SupersededBy (newer supersedes older)
            (Decision, Decision) => Some((
                CausalRelation::SupersededBy,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::SupersededBy)
                    .unwrap_or(&0.6),
            )),

            // Discovery/Learning → Todo = TriggeredBy
            (Discovery, Task) | (Learning, Task) => Some((
                CausalRelation::TriggeredBy,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::TriggeredBy)
                    .unwrap_or(&0.75),
            )),

            // Pattern → Learning = InformedBy (patterns inform learnings)
            (Pattern, Learning) | (Pattern, Decision) => Some((
                CausalRelation::InformedBy,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::InformedBy)
                    .unwrap_or(&0.7),
            )),

            // Error → Learning = ResolvedBy (error led to learning)
            (Error, Learning) => Some((
                CausalRelation::ResolvedBy,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::ResolvedBy)
                    .unwrap_or(&0.85),
            )),

            // Observation → Discovery = TriggeredBy
            (Observation, Discovery) => Some((
                CausalRelation::TriggeredBy,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::TriggeredBy)
                    .unwrap_or(&0.75),
            )),

            // Observation → Task = TriggeredBy (observation led to work)
            (Observation, Task) => Some((
                CausalRelation::TriggeredBy,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::TriggeredBy)
                    .unwrap_or(&0.75)
                    * 0.85, // slightly lower — observations are less direct triggers than discoveries
            )),

            // Observation → Decision = InformedBy (observation informed a decision)
            (Observation, Decision) => Some((
                CausalRelation::InformedBy,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::InformedBy)
                    .unwrap_or(&0.7)
                    * 0.85,
            )),

            // Observation → Error = Caused (observation of a problem → error report)
            (Observation, Error) => Some((
                CausalRelation::Caused,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::Caused)
                    .unwrap_or(&0.8)
                    * 0.75, // observations rarely directly cause errors; lower confidence
            )),

            // Observation → Learning = TriggeredBy (observation sparked learning)
            (Observation, Learning) => Some((
                CausalRelation::TriggeredBy,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::TriggeredBy)
                    .unwrap_or(&0.75)
                    * 0.85,
            )),

            // Conversation → Decision = InformedBy (discussion informed a decision)
            (Conversation, Decision) => Some((
                CausalRelation::InformedBy,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::InformedBy)
                    .unwrap_or(&0.7)
                    * 0.9, // conversations are strong informational sources
            )),

            // Conversation → Task = TriggeredBy (discussion spawned work)
            (Conversation, Task) => Some((
                CausalRelation::TriggeredBy,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::TriggeredBy)
                    .unwrap_or(&0.75)
                    * 0.85,
            )),

            // Conversation → Learning = InformedBy (discussion led to learning)
            (Conversation, Learning) => Some((
                CausalRelation::InformedBy,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::InformedBy)
                    .unwrap_or(&0.7)
                    * 0.85,
            )),

            // Conversation → Error = Caused (discussion surfaced a bug)
            (Conversation, Error) => Some((
                CausalRelation::Caused,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::Caused)
                    .unwrap_or(&0.8)
                    * 0.7, // conversations weakly cause error reports
            )),

            // Conversation → Discovery = TriggeredBy (discussion led to discovery)
            (Conversation, Discovery) => Some((
                CausalRelation::TriggeredBy,
                *self
                    .config
                    .relation_confidence
                    .get(&CausalRelation::TriggeredBy)
                    .unwrap_or(&0.75)
                    * 0.85,
            )),

            // Default: RelatedTo if same type or generic relation
            _ => {
                // Only suggest RelatedTo for semantically related types
                if Self::are_types_related(from_type, to_type) {
                    Some((
                        CausalRelation::RelatedTo,
                        *self
                            .config
                            .relation_confidence
                            .get(&CausalRelation::RelatedTo)
                            .unwrap_or(&0.5),
                    ))
                } else {
                    None
                }
            }
        }
    }

    /// Check if two experience types are semantically related.
    ///
    /// Returns true for same-group pairs AND for cross-group bridging types.
    /// Observation and Conversation are "bridge" types — they can relate to
    /// any other type because observations and conversations often span domains.
    fn are_types_related(a: &ExperienceType, b: &ExperienceType) -> bool {
        use ExperienceType::*;

        // Same type is always related
        if std::mem::discriminant(a) == std::mem::discriminant(b) {
            return true;
        }

        // Bridge types: Observation and Conversation can relate to anything.
        // These are the most common types in production (segmentation fallback
        // and hook ingestion), so they must bridge across groups to form chains.
        let is_bridge = |t: &ExperienceType| matches!(t, Observation | Conversation);
        if is_bridge(a) || is_bridge(b) {
            return true;
        }

        // Define related type groups for non-bridge types
        let knowledge_types = [Learning, Discovery, Pattern];
        let action_types = [Task, Decision, Command, CodeEdit];
        let context_types = [Context, FileAccess, Search];

        let in_knowledge = |t: &ExperienceType| {
            knowledge_types
                .iter()
                .any(|k| std::mem::discriminant(k) == std::mem::discriminant(t))
        };
        let in_action = |t: &ExperienceType| {
            action_types
                .iter()
                .any(|k| std::mem::discriminant(k) == std::mem::discriminant(t))
        };
        let in_context = |t: &ExperienceType| {
            context_types
                .iter()
                .any(|k| std::mem::discriminant(k) == std::mem::discriminant(t))
        };

        // Types in same group are related
        (in_knowledge(a) && in_knowledge(b))
            || (in_action(a) && in_action(b))
            || (in_context(a) && in_context(b))
    }

    /// Calculate entity overlap using Jaccard similarity
    fn calculate_entity_overlap(tags_a: &[String], tags_b: &[String]) -> f32 {
        if tags_a.is_empty() && tags_b.is_empty() {
            return 0.0;
        }

        let set_a: HashSet<&str> = tags_a.iter().map(|s| s.as_str()).collect();
        let set_b: HashSet<&str> = tags_b.iter().map(|s| s.as_str()).collect();

        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Detect branch point from memory content (pivot language).
    ///
    /// Uses strong phrase-level signals to avoid false positives from common words
    /// like "actually" or "instead" which appear in normal discourse.
    /// Requires either one strong signal or two weak signals to trigger.
    pub fn detect_branch_signal(content: &str) -> bool {
        let content_lower = content.to_lowercase();

        // Strong signals: unambiguous pivot language
        let strong_signals = [
            "pivot to",
            "change direction",
            "start fresh",
            "start over",
            "complete rewrite",
            "should rewrite",
            "need to rewrite",
            "scrap this",
            "scrap the",
            "different strategy",
            "new strategy",
            "abandon",
        ];

        // Weak signals: common words that only indicate a pivot when combined
        let weak_signals = ["instead", "new approach", "rethink", "rewrite", "pivot"];

        let strong_count = strong_signals
            .iter()
            .filter(|s| content_lower.contains(*s))
            .count();
        let weak_count = weak_signals
            .iter()
            .filter(|s| content_lower.contains(*s))
            .count();

        strong_count >= 1 || weak_count >= 2
    }

    // =========================================================================
    // LINEAGE TRAVERSAL
    // =========================================================================

    /// Trace lineage from a memory
    pub fn trace(
        &self,
        user_id: &str,
        memory_id: &MemoryId,
        direction: TraceDirection,
        max_depth: usize,
    ) -> Result<LineageTrace> {
        let mut visited = HashSet::new();
        let mut edges = Vec::new();
        let mut path = vec![memory_id.clone()];
        let mut queue: VecDeque<(MemoryId, usize)> = VecDeque::new();

        queue.push_back((memory_id.clone(), 0));
        visited.insert(memory_id.clone());

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let next_edges = match direction {
                TraceDirection::Backward => self.get_edges_to(user_id, &current_id)?,
                TraceDirection::Forward => self.get_edges_from(user_id, &current_id)?,
                TraceDirection::Both => {
                    let mut all = self.get_edges_to(user_id, &current_id)?;
                    all.extend(self.get_edges_from(user_id, &current_id)?);
                    all
                }
            };

            for edge in next_edges {
                let next_id = match direction {
                    TraceDirection::Backward => edge.from.clone(),
                    TraceDirection::Forward => edge.to.clone(),
                    TraceDirection::Both => {
                        if edge.from == current_id {
                            edge.to.clone()
                        } else {
                            edge.from.clone()
                        }
                    }
                };

                if !visited.contains(&next_id) {
                    visited.insert(next_id.clone());
                    path.push(next_id.clone());
                    edges.push(edge);
                    queue.push_back((next_id, depth + 1));
                }
            }
        }

        let depth = path.len().saturating_sub(1);
        Ok(LineageTrace {
            root: memory_id.clone(),
            direction,
            edges,
            path,
            depth,
        })
    }

    /// Find the root cause of a memory (trace all the way back)
    ///
    /// Returns `None` if the memory has no ancestors (is itself a root).
    pub fn find_root_cause(&self, user_id: &str, memory_id: &MemoryId) -> Result<Option<MemoryId>> {
        let trace = self.trace(user_id, memory_id, TraceDirection::Backward, 100)?;
        // path[0] is the starting memory — only return a root if we found ancestors
        if trace.path.len() <= 1 {
            Ok(None)
        } else {
            Ok(trace.path.last().cloned())
        }
    }

    /// Find all effects of a memory (trace all the way forward)
    ///
    /// Returns effects only (excludes the starting memory itself).
    pub fn find_effects(
        &self,
        user_id: &str,
        memory_id: &MemoryId,
        max_depth: usize,
    ) -> Result<Vec<MemoryId>> {
        let trace = self.trace(user_id, memory_id, TraceDirection::Forward, max_depth)?;
        // Skip the first element (the starting memory itself)
        Ok(trace.path.into_iter().skip(1).collect())
    }

    // =========================================================================
    // USER OPERATIONS
    // =========================================================================

    /// Confirm an inferred edge
    pub fn confirm_edge(&self, user_id: &str, edge_id: &str) -> Result<bool> {
        if let Some(mut edge) = self.get_edge(user_id, edge_id)? {
            edge.confirm();
            self.store_edge(user_id, &edge)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Reject (delete) an inferred edge
    pub fn reject_edge(&self, user_id: &str, edge_id: &str) -> Result<bool> {
        self.delete_edge(user_id, edge_id)
    }

    /// Add an explicit edge
    pub fn add_explicit_edge(
        &self,
        user_id: &str,
        from: MemoryId,
        to: MemoryId,
        relation: CausalRelation,
    ) -> Result<LineageEdge> {
        let edge = LineageEdge::explicit(from, to, relation);
        self.store_edge(user_id, &edge)?;
        Ok(edge)
    }

    /// Check if an edge already exists between two memories
    pub fn edge_exists(&self, user_id: &str, from: &MemoryId, to: &MemoryId) -> Result<bool> {
        let edges = self.get_edges_from(user_id, from)?;
        Ok(edges.iter().any(|e| &e.to == to))
    }

    // =========================================================================
    // STATISTICS
    // =========================================================================

    /// Get lineage statistics for a user.
    ///
    /// Caps the scan at 10,000 edges. For users with more edges, the stats
    /// will be approximate (counts capped, averages computed over the sample).
    pub fn stats(&self, user_id: &str) -> Result<LineageStats> {
        const STATS_SCAN_LIMIT: usize = 10_000;
        let edges = self.list_edges(user_id, STATS_SCAN_LIMIT)?;
        let branches = self.list_branches(user_id)?;

        let mut stats = LineageStats {
            total_edges: edges.len(),
            total_branches: branches.len(),
            active_branches: branches.iter().filter(|b| b.active).count(),
            ..Default::default()
        };

        let mut total_confidence: f32 = 0.0;

        for edge in &edges {
            match edge.source {
                LineageSource::Inferred => stats.inferred_edges += 1,
                LineageSource::Confirmed => stats.confirmed_edges += 1,
                LineageSource::Explicit => stats.explicit_edges += 1,
            }

            let relation_name = format!("{:?}", edge.relation);
            *stats.edges_by_relation.entry(relation_name).or_insert(0) += 1;

            total_confidence += edge.confidence;
        }

        if !edges.is_empty() {
            stats.avg_confidence = total_confidence / edges.len() as f32;
        }

        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::Experience;
    use chrono::Duration;
    use tempfile::TempDir;

    fn create_test_graph() -> (LineageGraph, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db = Arc::new(DB::open_default(temp_dir.path()).unwrap());
        (LineageGraph::new(db), temp_dir)
    }

    fn create_test_memory(exp_type: ExperienceType, entities: Vec<&str>) -> Memory {
        let experience = Experience {
            experience_type: exp_type,
            content: "Test memory".to_string(),
            entities: entities.into_iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        };
        Memory::new(
            MemoryId(Uuid::new_v4()),
            experience,
            0.5,  // importance
            None, // agent_id
            None, // run_id
            None, // actor_id
            None, // created_at (uses Utc::now())
        )
    }

    #[test]
    fn test_store_and_get_edge() {
        let (graph, _dir) = create_test_graph();
        let from = MemoryId(Uuid::new_v4());
        let to = MemoryId(Uuid::new_v4());

        let edge = LineageEdge::explicit(from.clone(), to.clone(), CausalRelation::Caused);
        graph.store_edge("user-1", &edge).unwrap();

        let retrieved = graph.get_edge("user-1", &edge.id).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().relation, CausalRelation::Caused);
    }

    #[test]
    fn test_get_edges_from_and_to() {
        let (graph, _dir) = create_test_graph();
        let from = MemoryId(Uuid::new_v4());
        let to1 = MemoryId(Uuid::new_v4());
        let to2 = MemoryId(Uuid::new_v4());

        let edge1 = LineageEdge::explicit(from.clone(), to1.clone(), CausalRelation::Caused);
        let edge2 = LineageEdge::explicit(from.clone(), to2.clone(), CausalRelation::TriggeredBy);

        graph.store_edge("user-1", &edge1).unwrap();
        graph.store_edge("user-1", &edge2).unwrap();

        let from_edges = graph.get_edges_from("user-1", &from).unwrap();
        assert_eq!(from_edges.len(), 2);

        let to_edges = graph.get_edges_to("user-1", &to1).unwrap();
        assert_eq!(to_edges.len(), 1);
    }

    #[test]
    fn test_infer_error_to_task() {
        let (graph, _dir) = create_test_graph();

        // Use same entities for high overlap
        let error = create_test_memory(ExperienceType::Error, vec!["auth", "login"]);
        let mut task = create_test_memory(ExperienceType::Task, vec!["auth", "login"]);
        task.created_at = error.created_at + Duration::days(1);

        let result = graph.infer_relation(&error, &task);
        assert!(result.is_some());
        let (relation, confidence) = result.unwrap();
        assert_eq!(relation, CausalRelation::Caused);
        // With perfect overlap (1.0) and 1 day gap: 0.8 * 1.0 * 0.93 ≈ 0.74
        assert!(confidence > 0.4, "confidence was {}", confidence);
    }

    #[test]
    fn test_infer_learning_to_decision() {
        let (graph, _dir) = create_test_graph();

        let learning = create_test_memory(ExperienceType::Learning, vec!["react", "hooks"]);
        let mut decision =
            create_test_memory(ExperienceType::Decision, vec!["react", "hooks", "state"]);
        decision.created_at = learning.created_at + Duration::days(2);

        let result = graph.infer_relation(&learning, &decision);
        assert!(result.is_some());
        let (relation, _) = result.unwrap();
        assert_eq!(relation, CausalRelation::InformedBy);
    }

    #[test]
    fn test_no_inference_wrong_order() {
        let (graph, _dir) = create_test_graph();

        let task = create_test_memory(ExperienceType::Task, vec!["auth"]);
        let mut error = create_test_memory(ExperienceType::Error, vec!["auth"]);
        error.created_at = task.created_at - Duration::days(1); // Error BEFORE task

        // Task to Error (wrong causal direction) should not infer Caused
        let result = graph.infer_relation(&task, &error);
        assert!(result.is_none());
    }

    #[test]
    fn test_branch_creation() {
        let (graph, _dir) = create_test_graph();
        let branch_point = MemoryId(Uuid::new_v4());

        graph.ensure_main_branch("user-1").unwrap();

        let branch = graph
            .create_branch(
                "user-1",
                "v2-rewrite",
                "main",
                branch_point,
                Some("Complete rewrite"),
            )
            .unwrap();

        let retrieved = graph.get_branch("user-1", &branch.id).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "v2-rewrite");
    }

    #[test]
    fn test_detect_branch_signal() {
        assert!(LineageGraph::detect_branch_signal(
            "Let's pivot to a new approach"
        ));
        assert!(LineageGraph::detect_branch_signal(
            "Actually, we should rewrite this"
        ));
        assert!(LineageGraph::detect_branch_signal(
            "I think we need to start fresh"
        ));
        assert!(!LineageGraph::detect_branch_signal("Fixed the bug in auth"));
    }

    #[test]
    fn test_confirm_and_reject_edge() {
        let (graph, _dir) = create_test_graph();
        let from = MemoryId(Uuid::new_v4());
        let to = MemoryId(Uuid::new_v4());

        let edge = LineageEdge::inferred(from.clone(), to.clone(), CausalRelation::Caused, 0.7);
        graph.store_edge("user-1", &edge).unwrap();

        // Confirm
        assert!(graph.confirm_edge("user-1", &edge.id).unwrap());
        let confirmed = graph.get_edge("user-1", &edge.id).unwrap().unwrap();
        assert_eq!(confirmed.source, LineageSource::Confirmed);
        assert_eq!(confirmed.confidence, 1.0);

        // Reject another edge
        let edge2 = LineageEdge::inferred(from, to, CausalRelation::RelatedTo, 0.5);
        graph.store_edge("user-1", &edge2).unwrap();
        assert!(graph.reject_edge("user-1", &edge2.id).unwrap());
        assert!(graph.get_edge("user-1", &edge2.id).unwrap().is_none());
    }

    #[test]
    fn test_lineage_stats() {
        let (graph, _dir) = create_test_graph();

        let from = MemoryId(Uuid::new_v4());
        let to = MemoryId(Uuid::new_v4());

        graph
            .store_edge(
                "user-1",
                &LineageEdge::inferred(from.clone(), to.clone(), CausalRelation::Caused, 0.8),
            )
            .unwrap();
        graph
            .store_edge(
                "user-1",
                &LineageEdge::explicit(from.clone(), to.clone(), CausalRelation::InformedBy),
            )
            .unwrap();
        graph.ensure_main_branch("user-1").unwrap();

        let stats = graph.stats("user-1").unwrap();
        assert_eq!(stats.total_edges, 2);
        assert_eq!(stats.inferred_edges, 1);
        assert_eq!(stats.explicit_edges, 1);
        assert_eq!(stats.total_branches, 1);
    }

    // =========================================================================
    // Observation type inference tests (#205)
    // =========================================================================

    #[test]
    fn test_infer_observation_to_task() {
        let (graph, _dir) = create_test_graph();
        let obs = create_test_memory(ExperienceType::Observation, vec!["auth", "login"]);
        let mut task = create_test_memory(ExperienceType::Task, vec!["auth", "login"]);
        task.created_at = obs.created_at + Duration::days(1);

        let result = graph.infer_relation(&obs, &task);
        assert!(result.is_some(), "Observation → Task should infer");
        let (relation, confidence) = result.unwrap();
        assert_eq!(relation, CausalRelation::TriggeredBy);
        assert!(confidence > 0.3, "confidence was {}", confidence);
    }

    #[test]
    fn test_infer_observation_to_decision() {
        let (graph, _dir) = create_test_graph();
        let obs = create_test_memory(ExperienceType::Observation, vec!["perf", "latency"]);
        let mut decision =
            create_test_memory(ExperienceType::Decision, vec!["perf", "latency", "cache"]);
        decision.created_at = obs.created_at + Duration::days(1);

        let result = graph.infer_relation(&obs, &decision);
        assert!(result.is_some(), "Observation → Decision should infer");
        let (relation, _) = result.unwrap();
        assert_eq!(relation, CausalRelation::InformedBy);
    }

    #[test]
    fn test_infer_observation_to_error() {
        let (graph, _dir) = create_test_graph();
        let obs = create_test_memory(ExperienceType::Observation, vec!["mutex", "deadlock"]);
        let mut error = create_test_memory(ExperienceType::Error, vec!["mutex", "deadlock"]);
        error.created_at = obs.created_at + Duration::days(1);

        let result = graph.infer_relation(&obs, &error);
        assert!(result.is_some(), "Observation → Error should infer");
        let (relation, _) = result.unwrap();
        assert_eq!(relation, CausalRelation::Caused);
    }

    #[test]
    fn test_infer_observation_to_learning() {
        let (graph, _dir) = create_test_graph();
        let obs = create_test_memory(ExperienceType::Observation, vec!["rocksdb", "column"]);
        let mut learning = create_test_memory(
            ExperienceType::Learning,
            vec!["rocksdb", "column", "family"],
        );
        learning.created_at = obs.created_at + Duration::days(1);

        let result = graph.infer_relation(&obs, &learning);
        assert!(result.is_some(), "Observation → Learning should infer");
        let (relation, _) = result.unwrap();
        assert_eq!(relation, CausalRelation::TriggeredBy);
    }

    // =========================================================================
    // Conversation type inference tests (#205)
    // =========================================================================

    #[test]
    fn test_infer_conversation_to_decision() {
        let (graph, _dir) = create_test_graph();
        let conv = create_test_memory(ExperienceType::Conversation, vec!["api", "design"]);
        let mut decision =
            create_test_memory(ExperienceType::Decision, vec!["api", "design", "rest"]);
        decision.created_at = conv.created_at + Duration::days(1);

        let result = graph.infer_relation(&conv, &decision);
        assert!(result.is_some(), "Conversation → Decision should infer");
        let (relation, _) = result.unwrap();
        assert_eq!(relation, CausalRelation::InformedBy);
    }

    #[test]
    fn test_infer_conversation_to_task() {
        let (graph, _dir) = create_test_graph();
        let conv = create_test_memory(ExperienceType::Conversation, vec!["bug", "deploy"]);
        let mut task = create_test_memory(ExperienceType::Task, vec!["bug", "deploy"]);
        task.created_at = conv.created_at + Duration::days(1);

        let result = graph.infer_relation(&conv, &task);
        assert!(result.is_some(), "Conversation → Task should infer");
        let (relation, _) = result.unwrap();
        assert_eq!(relation, CausalRelation::TriggeredBy);
    }

    #[test]
    fn test_infer_conversation_to_learning() {
        let (graph, _dir) = create_test_graph();
        let conv = create_test_memory(ExperienceType::Conversation, vec!["hebbian", "memory"]);
        let mut learning =
            create_test_memory(ExperienceType::Learning, vec!["hebbian", "memory", "decay"]);
        learning.created_at = conv.created_at + Duration::days(1);

        let result = graph.infer_relation(&conv, &learning);
        assert!(result.is_some(), "Conversation → Learning should infer");
        let (relation, _) = result.unwrap();
        assert_eq!(relation, CausalRelation::InformedBy);
    }

    #[test]
    fn test_infer_conversation_to_error() {
        let (graph, _dir) = create_test_graph();
        let conv = create_test_memory(ExperienceType::Conversation, vec!["auth", "token"]);
        let mut error = create_test_memory(ExperienceType::Error, vec!["auth", "token"]);
        error.created_at = conv.created_at + Duration::days(1);

        let result = graph.infer_relation(&conv, &error);
        assert!(result.is_some(), "Conversation → Error should infer");
        let (relation, _) = result.unwrap();
        assert_eq!(relation, CausalRelation::Caused);
    }

    #[test]
    fn test_infer_conversation_to_discovery() {
        let (graph, _dir) = create_test_graph();
        let conv = create_test_memory(ExperienceType::Conversation, vec!["graph", "edge"]);
        let mut discovery =
            create_test_memory(ExperienceType::Discovery, vec!["graph", "edge", "weight"]);
        discovery.created_at = conv.created_at + Duration::days(1);

        let result = graph.infer_relation(&conv, &discovery);
        assert!(result.is_some(), "Conversation → Discovery should infer");
        let (relation, _) = result.unwrap();
        assert_eq!(relation, CausalRelation::TriggeredBy);
    }

    // =========================================================================
    // Bridge type tests — reverse direction (action → Observation/Conversation)
    // =========================================================================

    #[test]
    fn test_infer_decision_to_observation_related() {
        let (graph, _dir) = create_test_graph();
        let decision = create_test_memory(ExperienceType::Decision, vec!["cache", "redis"]);
        let mut obs = create_test_memory(ExperienceType::Observation, vec!["cache", "redis"]);
        obs.created_at = decision.created_at + Duration::days(1);

        let result = graph.infer_relation(&decision, &obs);
        assert!(
            result.is_some(),
            "Decision → Observation should produce RelatedTo via bridge"
        );
        let (relation, _) = result.unwrap();
        assert_eq!(relation, CausalRelation::RelatedTo);
    }

    #[test]
    fn test_infer_task_to_conversation_related() {
        let (graph, _dir) = create_test_graph();
        let task = create_test_memory(ExperienceType::Task, vec!["deploy", "staging"]);
        let mut conv = create_test_memory(ExperienceType::Conversation, vec!["deploy", "staging"]);
        conv.created_at = task.created_at + Duration::days(1);

        let result = graph.infer_relation(&task, &conv);
        assert!(
            result.is_some(),
            "Task → Conversation should produce RelatedTo via bridge"
        );
        let (relation, _) = result.unwrap();
        assert_eq!(relation, CausalRelation::RelatedTo);
    }

    #[test]
    fn test_observation_no_entities_still_bridges() {
        let (graph, _dir) = create_test_graph();
        // No entities — tests the 0.3 floor path
        let obs = create_test_memory(ExperienceType::Observation, vec![]);
        let mut task = create_test_memory(ExperienceType::Task, vec![]);
        task.created_at = obs.created_at + Duration::days(1);

        let result = graph.infer_relation(&obs, &task);
        assert!(
            result.is_some(),
            "Observation → Task should infer even without entities"
        );
        let (relation, confidence) = result.unwrap();
        assert_eq!(relation, CausalRelation::TriggeredBy);
        // With no entities: effective_overlap=0.3, temporal_factor≈0.97
        // base * 0.85 * 0.3 * (0.5 + 0.5*0.97) ≈ 0.75 * 0.85 * 0.3 * 0.985 ≈ 0.19
        assert!(
            confidence > 0.1,
            "low-entity confidence should still be nonzero, was {}",
            confidence
        );
    }

    // =========================================================================
    // Embedding similarity tests (#208)
    // =========================================================================

    fn create_test_memory_with_embeddings(
        exp_type: ExperienceType,
        entities: Vec<&str>,
        embeddings: Option<Vec<f32>>,
    ) -> Memory {
        let experience = Experience {
            experience_type: exp_type,
            content: "Test memory".to_string(),
            entities: entities.into_iter().map(|s| s.to_string()).collect(),
            embeddings,
            ..Default::default()
        };
        Memory::new(
            MemoryId(Uuid::new_v4()),
            experience,
            0.5,
            None,
            None,
            None,
            None,
        )
    }

    #[test]
    fn test_infer_with_high_embedding_similarity() {
        let (graph, _dir) = create_test_graph();

        // High cosine similarity embeddings (identical), no entity overlap
        let emb = vec![0.1, 0.5, 0.8, 0.3, 0.9];
        let obs = create_test_memory_with_embeddings(
            ExperienceType::Observation,
            vec![],
            Some(emb.clone()),
        );
        let mut task = create_test_memory_with_embeddings(
            ExperienceType::Task,
            vec![],
            Some(emb),
        );
        task.created_at = obs.created_at + Duration::days(1);

        let result = graph.infer_relation(&obs, &task);
        assert!(
            result.is_some(),
            "High embedding similarity should produce inference even without entities"
        );
        let (relation, confidence) = result.unwrap();
        assert_eq!(relation, CausalRelation::TriggeredBy);
        // cosine_sim=1.0, entity_overlap=0.0, semantic_signal=max(0,1)=1.0
        assert!(confidence > 0.4, "confidence was {}", confidence);
    }

    #[test]
    fn test_infer_embedding_rescues_low_entity_overlap() {
        let (graph, _dir) = create_test_graph();

        // Low entity overlap but high embedding similarity
        let emb_a = vec![0.1, 0.5, 0.8, 0.3, 0.9];
        let emb_b = vec![0.12, 0.48, 0.82, 0.28, 0.88]; // very similar
        let learning = create_test_memory_with_embeddings(
            ExperienceType::Learning,
            vec!["rust"],
            Some(emb_a),
        );
        let mut decision = create_test_memory_with_embeddings(
            ExperienceType::Decision,
            vec!["go", "lang", "rust"],
            Some(emb_b),
        );
        decision.created_at = learning.created_at + Duration::days(1);

        let result = graph.infer_relation(&learning, &decision);
        assert!(
            result.is_some(),
            "Embedding similarity should rescue low entity overlap"
        );
        let (relation, _) = result.unwrap();
        assert_eq!(relation, CausalRelation::InformedBy);
    }

    #[test]
    fn test_infer_low_embedding_similarity_blocked() {
        let (graph, _dir) = create_test_graph();

        // Low embedding similarity, no entities
        let emb_a = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let emb_b = vec![0.0, 0.0, 0.0, 0.0, 1.0]; // orthogonal
        let obs = create_test_memory_with_embeddings(
            ExperienceType::Observation,
            vec![],
            Some(emb_a),
        );
        let mut task = create_test_memory_with_embeddings(
            ExperienceType::Task,
            vec![],
            Some(emb_b),
        );
        task.created_at = obs.created_at + Duration::days(1);

        let result = graph.infer_relation(&obs, &task);
        assert!(
            result.is_none(),
            "Orthogonal embeddings with no entities should block inference"
        );
    }

    #[test]
    fn test_weaken_uses_090_multiplier() {
        let mut edge = LineageEdge::inferred(
            MemoryId(Uuid::new_v4()),
            MemoryId(Uuid::new_v4()),
            CausalRelation::Caused,
            1.0,
        );

        // First weakening: 1.0 * 0.90 = 0.90
        let should_prune = edge.weaken();
        assert!(!should_prune);
        assert!((edge.confidence - 0.90).abs() < 0.001, "expected ~0.90, got {}", edge.confidence);

        // After many weakenings, should eventually prune
        for _ in 0..50 {
            edge.weaken();
        }
        assert!(edge.confidence < 0.05, "should be below prune threshold after 50 weakenings");
    }
}
