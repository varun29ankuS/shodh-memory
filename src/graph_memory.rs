//! Graph Memory System - Inspired by Graphiti
//!
//! Temporal knowledge graph for tracking entities, relationships, and episodic memories.
//! Implements bi-temporal tracking and hybrid retrieval (semantic + graph traversal).

use anyhow::Result;
use chrono::{DateTime, Utc};
use rocksdb::{Options, WriteBatch, DB};
use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use uuid::Uuid;

use crate::constants::{LTP_LEARNING_RATE, LTP_MIN_STRENGTH, LTP_THRESHOLD};
use crate::decay::hybrid_decay_factor;

/// Entity node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    /// Unique identifier
    pub uuid: Uuid,

    /// Entity name (e.g., "John", "Paris", "Rust programming")
    pub name: String,

    /// Entity labels/types (e.g., ["Person"], ["Location", "City"])
    pub labels: Vec<EntityLabel>,

    /// When this entity was first created in the graph
    pub created_at: DateTime<Utc>,

    /// When this entity was last observed
    pub last_seen_at: DateTime<Utc>,

    /// How many times this entity has been mentioned
    pub mention_count: usize,

    /// Summary of this entity's context (built from surrounding edges)
    pub summary: String,

    /// Additional attributes based on entity type
    pub attributes: HashMap<String, String>,

    /// Semantic embedding of the entity name (for similarity search)
    pub name_embedding: Option<Vec<f32>>,

    /// Salience score (0.0 - 1.0): How important is this entity?
    /// Higher salience = larger gravitational well in the memory universe
    /// Factors: proper noun status, mention frequency, recency, user-defined importance
    #[serde(default = "default_salience")]
    pub salience: f32,

    /// Whether this is a proper noun (names, places, products)
    /// Proper nouns have higher base salience than common nouns
    #[serde(default)]
    pub is_proper_noun: bool,
}

fn default_salience() -> f32 {
    0.5 // Default middle salience
}

/// Entity labels following Graphiti's categorization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityLabel {
    Person,
    Organization,
    Location,
    Technology,
    Concept,
    Event,
    Date,
    Product,
    Skill,
    Other(String),
}

impl EntityLabel {
    /// Get string representation of the entity label
    #[allow(unused)] // Public API for serialization/display
    pub fn as_str(&self) -> &str {
        match self {
            Self::Person => "Person",
            Self::Organization => "Organization",
            Self::Location => "Location",
            Self::Technology => "Technology",
            Self::Concept => "Concept",
            Self::Event => "Event",
            Self::Date => "Date",
            Self::Product => "Product",
            Self::Skill => "Skill",
            Self::Other(s) => s.as_str(),
        }
    }
}

/// Relationship edge between entities
///
/// Implements Hebbian synaptic plasticity: "Neurons that fire together, wire together"
/// - Strength increases with co-activation (strengthen method)
/// - Strength decays over time without use (decay method)
/// - Long-Term Potentiation (LTP): After threshold activations, becomes permanent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipEdge {
    /// Unique identifier for this edge
    pub uuid: Uuid,

    /// Source entity UUID
    pub from_entity: Uuid,

    /// Target entity UUID
    pub to_entity: Uuid,

    /// Type of relationship
    pub relation_type: RelationType,

    /// Confidence/strength of this relationship (0.0 to 1.0)
    /// Dynamic: increases with co-activation, decays without use
    pub strength: f32,

    /// When this relationship was created
    pub created_at: DateTime<Utc>,

    /// When this relationship was last observed (temporal tracking)
    pub valid_at: DateTime<Utc>,

    /// Whether this relationship has been invalidated (temporal edge invalidation)
    pub invalidated_at: Option<DateTime<Utc>>,

    /// Source episode that created this relationship
    pub source_episode_id: Option<Uuid>,

    /// Additional context about the relationship
    pub context: String,

    // === Hebbian Synaptic Plasticity Fields ===
    /// When this synapse was last activated (used in retrieval/traversal)
    /// Used to calculate time-based decay
    #[serde(default = "default_last_activated")]
    pub last_activated: DateTime<Utc>,

    /// Number of times both entities were co-accessed (Hebbian co-activation)
    /// Higher count = stronger learned association
    #[serde(default)]
    pub activation_count: u32,

    /// Long-Term Potentiation flag: synapse becomes permanent after threshold
    /// Once potentiated, decay is dramatically reduced (like biological LTP)
    #[serde(default)]
    pub potentiated: bool,
}

fn default_last_activated() -> DateTime<Utc> {
    Utc::now()
}

// Hebbian learning constants now imported from crate::constants:
// - LTP_LEARNING_RATE (0.1): η for strength increase per co-activation
// - LTP_DECAY_HALF_LIFE_DAYS (14.0): λ for time-based decay
// - LTP_THRESHOLD (10): Activations needed for Long-Term Potentiation
// - LTP_DECAY_FACTOR (0.1): Potentiated synapses decay 10x slower
// - LTP_MIN_STRENGTH (0.01): Floor to prevent complete forgetting

impl RelationshipEdge {
    /// Strengthen this synapse (Hebbian learning)
    ///
    /// Called when both connected entities are accessed together.
    /// Formula: w_new = w_old + η × (1 - w_old) × co_activation_boost
    ///
    /// The (1 - w_old) term ensures asymptotic approach to 1.0,
    /// preventing unbounded growth while allowing strong associations.
    pub fn strengthen(&mut self) {
        self.activation_count += 1;
        self.last_activated = Utc::now();

        // Hebbian strengthening: diminishing returns as strength approaches 1.0
        let boost = LTP_LEARNING_RATE * (1.0 - self.strength);
        self.strength = (self.strength + boost).min(1.0);

        // Check for Long-Term Potentiation threshold
        if !self.potentiated && self.activation_count >= LTP_THRESHOLD {
            self.potentiated = true;
            // LTP bonus: immediate strength boost
            self.strength = (self.strength + 0.2).min(1.0);
        }
    }

    /// Apply time-based decay to this synapse
    ///
    /// Uses hybrid decay model (SHO-103):
    /// - t < 3 days: Exponential decay (fast consolidation filtering)
    /// - t ≥ 3 days: Power-law decay (heavy tail for long-term retention)
    ///
    /// Potentiated synapses use lower β exponent for even slower decay.
    ///
    /// **Important:** Updates `last_activated` to prevent double-decay on
    /// repeated calls. Also caps max decay at 365 days to protect against
    /// clock jumps.
    ///
    /// Returns true if synapse should be pruned (strength below threshold)
    pub fn decay(&mut self) -> bool {
        let now = Utc::now();
        let elapsed = now.signed_duration_since(self.last_activated);
        let mut days_elapsed = elapsed.num_seconds() as f64 / 86400.0;

        if days_elapsed <= 0.0 {
            return false;
        }

        // Cap max decay to protect against clock jumps (max 1 year per call)
        const MAX_DECAY_DAYS: f64 = 365.0;
        if days_elapsed > MAX_DECAY_DAYS {
            days_elapsed = MAX_DECAY_DAYS;
        }

        // Hybrid decay: exponential for consolidation, power-law for long-term
        let decay_factor = hybrid_decay_factor(days_elapsed, self.potentiated);
        self.strength *= decay_factor;

        // Update last_activated to prevent double-decay on repeated calls
        self.last_activated = now;

        // Apply floor to prevent complete forgetting
        if self.strength < LTP_MIN_STRENGTH {
            self.strength = LTP_MIN_STRENGTH;
        }

        // Return whether this synapse should be pruned
        // Non-potentiated synapses with minimal strength can be removed
        !self.potentiated && self.strength <= LTP_MIN_STRENGTH
    }

    /// Get the effective strength considering recency
    ///
    /// This is a read-only version that calculates what the strength
    /// would be after decay, without modifying the edge.
    /// Uses hybrid decay model (exponential → power-law).
    pub fn effective_strength(&self) -> f32 {
        let now = Utc::now();
        let elapsed = now.signed_duration_since(self.last_activated);
        let days_elapsed = elapsed.num_seconds() as f64 / 86400.0;

        if days_elapsed <= 0.0 {
            return self.strength;
        }

        let decay_factor = hybrid_decay_factor(days_elapsed, self.potentiated);
        (self.strength * decay_factor).max(LTP_MIN_STRENGTH)
    }
}

/// Relationship types following Graphiti's semantic model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelationType {
    /// Work relationships
    WorksWith,
    WorksAt,
    EmployedBy,

    /// Structural relationships
    PartOf,
    Contains,
    OwnedBy,

    /// Location relationships
    LocatedIn,
    LocatedAt,

    /// Usage relationships
    Uses,
    CreatedBy,
    DevelopedBy,

    /// Causal relationships
    Causes,
    ResultsIn,

    /// Learning relationships
    Learned,
    Knows,
    Teaches,

    /// Generic relationships
    RelatedTo,
    AssociatedWith,

    /// Memory co-retrieval (Hebbian association between memories)
    CoRetrieved,

    /// Custom relationship
    Custom(String),
}

impl RelationType {
    /// Get string representation of the relation type
    #[allow(unused)] // Public API for serialization/display
    pub fn as_str(&self) -> &str {
        match self {
            Self::WorksWith => "WorksWith",
            Self::WorksAt => "WorksAt",
            Self::EmployedBy => "EmployedBy",
            Self::PartOf => "PartOf",
            Self::Contains => "Contains",
            Self::OwnedBy => "OwnedBy",
            Self::LocatedIn => "LocatedIn",
            Self::LocatedAt => "LocatedAt",
            Self::Uses => "Uses",
            Self::CreatedBy => "CreatedBy",
            Self::DevelopedBy => "DevelopedBy",
            Self::Causes => "Causes",
            Self::ResultsIn => "ResultsIn",
            Self::Learned => "Learned",
            Self::Knows => "Knows",
            Self::Teaches => "Teaches",
            Self::RelatedTo => "RelatedTo",
            Self::AssociatedWith => "AssociatedWith",
            Self::CoRetrieved => "CoRetrieved",
            Self::Custom(s) => s.as_str(),
        }
    }
}

/// Episodic node representing a discrete experience/memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicNode {
    /// Unique identifier
    pub uuid: Uuid,

    /// Human-readable name/title
    pub name: String,

    /// Episode content (the actual experience data)
    pub content: String,

    /// When the original event occurred (event time)
    pub valid_at: DateTime<Utc>,

    /// When this was ingested into the system (ingestion time)
    pub created_at: DateTime<Utc>,

    /// Entities extracted from this episode
    pub entity_refs: Vec<Uuid>,

    /// Source type
    pub source: EpisodeSource,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Episode source types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EpisodeSource {
    Message,
    Document,
    Event,
    Observation,
}

/// Graph memory storage and operations
pub struct GraphMemory {
    /// RocksDB storage for entities
    entities_db: Arc<DB>,

    /// RocksDB storage for relationships
    relationships_db: Arc<DB>,

    /// RocksDB storage for episodes
    episodes_db: Arc<DB>,

    /// RocksDB storage for entity -> relationships index
    entity_edges_db: Arc<DB>,

    /// RocksDB storage for entity -> episodes index (inverted index for fast lookup)
    entity_episodes_db: Arc<DB>,

    /// RocksDB storage for entity name -> UUID index (persisted, O(1) startup)
    entity_name_index_db: Arc<DB>,

    /// RocksDB storage for lowercase name -> UUID index (for O(1) case-insensitive lookup)
    entity_lowercase_index_db: Arc<DB>,

    /// RocksDB storage for stemmed name -> UUID index (for linguistic matching)
    /// Maps Porter-stemmed words to entity UUIDs for "running" -> "run" type matching
    entity_stemmed_index_db: Arc<DB>,

    /// In-memory entity name index for fast lookups (loaded from entity_name_index_db)
    entity_name_index: Arc<parking_lot::RwLock<HashMap<String, Uuid>>>,

    /// In-memory lowercase name index for O(1) case-insensitive lookups
    entity_lowercase_index: Arc<parking_lot::RwLock<HashMap<String, Uuid>>>,

    /// In-memory stemmed name index for O(1) linguistic lookups
    /// Key: Porter-stemmed lowercase name, Value: Entity UUID
    entity_stemmed_index: Arc<parking_lot::RwLock<HashMap<String, Uuid>>>,

    // === Atomic counters for O(1) stats (P1 fix) ===
    /// Entity count - initialized from entity_name_index.len(), updated on add
    entity_count: Arc<AtomicUsize>,

    /// Relationship count - initialized on startup, updated on add
    relationship_count: Arc<AtomicUsize>,

    /// Episode count - initialized on startup, updated on add
    episode_count: Arc<AtomicUsize>,

    /// Mutex for serializing synapse updates to prevent race conditions (SHO-64)
    /// Uses parking_lot::Mutex for better performance than std::sync::Mutex
    synapse_update_lock: Arc<parking_lot::Mutex<()>>,
}

impl GraphMemory {
    /// Create a new graph memory system
    pub fn new(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)?;

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        let entities_db = Arc::new(DB::open(&opts, path.join("graph_entities"))?);
        let relationships_db = Arc::new(DB::open(&opts, path.join("graph_relationships"))?);
        let episodes_db = Arc::new(DB::open(&opts, path.join("graph_episodes"))?);
        let entity_edges_db = Arc::new(DB::open(&opts, path.join("graph_entity_edges"))?);
        let entity_episodes_db = Arc::new(DB::open(&opts, path.join("graph_entity_episodes"))?);
        let entity_name_index_db = Arc::new(DB::open(&opts, path.join("graph_entity_name_index"))?);
        let entity_lowercase_index_db =
            Arc::new(DB::open(&opts, path.join("graph_entity_lowercase_index"))?);
        let entity_stemmed_index_db =
            Arc::new(DB::open(&opts, path.join("graph_entity_stemmed_index"))?);

        // Load entity name index from persisted DB (O(n) but faster than deserializing entities)
        // If empty, migrate from entities_db (one-time migration for existing data)
        let entity_name_index =
            Self::load_or_migrate_name_index(&entity_name_index_db, &entities_db)?;

        // Load/migrate lowercase index for O(1) case-insensitive lookup
        let entity_lowercase_index =
            Self::load_or_migrate_lowercase_index(&entity_lowercase_index_db, &entity_name_index)?;

        // Load/migrate stemmed index for O(1) linguistic lookup
        let entity_stemmed_index =
            Self::load_or_migrate_stemmed_index(&entity_stemmed_index_db, &entity_name_index)?;

        let entity_count = entity_name_index.len();

        // Count relationships and episodes during startup (one-time cost)
        // This is O(n) at startup, but get_stats() will be O(1) at runtime
        let relationship_count = Self::count_db_entries(&relationships_db);
        let episode_count = Self::count_db_entries(&episodes_db);

        let graph = Self {
            entities_db,
            relationships_db,
            episodes_db,
            entity_edges_db,
            entity_episodes_db,
            entity_name_index_db,
            entity_lowercase_index_db,
            entity_stemmed_index_db,
            entity_name_index: Arc::new(parking_lot::RwLock::new(entity_name_index)),
            entity_lowercase_index: Arc::new(parking_lot::RwLock::new(entity_lowercase_index)),
            entity_stemmed_index: Arc::new(parking_lot::RwLock::new(entity_stemmed_index)),
            entity_count: Arc::new(AtomicUsize::new(entity_count)),
            relationship_count: Arc::new(AtomicUsize::new(relationship_count)),
            episode_count: Arc::new(AtomicUsize::new(episode_count)),
            synapse_update_lock: Arc::new(parking_lot::Mutex::new(())),
        };

        if entity_count > 0 || relationship_count > 0 || episode_count > 0 {
            tracing::info!(
                "Loaded graph with {} entities, {} relationships, {} episodes",
                entity_count,
                relationship_count,
                episode_count
            );
        }

        Ok(graph)
    }

    /// Load entity name->UUID index from persisted DB, or migrate from entities_db if empty
    fn load_or_migrate_name_index(
        index_db: &DB,
        entities_db: &DB,
    ) -> Result<HashMap<String, Uuid>> {
        let mut index = HashMap::new();

        // Try to load from dedicated index DB first
        let iter = index_db.iterator(rocksdb::IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if let (Ok(name), Ok(uuid_bytes)) = (
                std::str::from_utf8(&key),
                <[u8; 16]>::try_from(value.as_ref()),
            ) {
                index.insert(name.to_string(), Uuid::from_bytes(uuid_bytes));
            }
        }

        // If index DB is empty but entities exist, migrate (one-time operation)
        if index.is_empty() {
            let entity_iter = entities_db.iterator(rocksdb::IteratorMode::Start);
            let mut migrated_count = 0;
            for (_, value) in entity_iter.flatten() {
                if let Ok(entity) = bincode::serde::decode_from_slice::<EntityNode, _>(
                    &value,
                    bincode::config::standard(),
                )
                .map(|(v, _)| v)
                {
                    // Store in index DB: name -> UUID bytes
                    index_db.put(entity.name.as_bytes(), entity.uuid.as_bytes())?;
                    index.insert(entity.name.clone(), entity.uuid);
                    migrated_count += 1;
                }
            }
            if migrated_count > 0 {
                tracing::info!("Migrated {} entities to name index DB", migrated_count);
            }
        }

        Ok(index)
    }

    /// Load lowercase name->UUID index, or migrate from name_index if empty
    ///
    /// This enables O(1) case-insensitive entity lookup instead of O(n) linear search.
    fn load_or_migrate_lowercase_index(
        lowercase_db: &DB,
        name_index: &HashMap<String, Uuid>,
    ) -> Result<HashMap<String, Uuid>> {
        let mut index = HashMap::new();

        // Try to load from dedicated lowercase index DB
        let iter = lowercase_db.iterator(rocksdb::IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if let (Ok(name), Ok(uuid_bytes)) = (
                std::str::from_utf8(&key),
                <[u8; 16]>::try_from(value.as_ref()),
            ) {
                index.insert(name.to_string(), Uuid::from_bytes(uuid_bytes));
            }
        }

        // If empty but name_index has data, migrate (one-time operation)
        if index.is_empty() && !name_index.is_empty() {
            for (name, uuid) in name_index {
                let lowercase_name = name.to_lowercase();
                lowercase_db.put(lowercase_name.as_bytes(), uuid.as_bytes())?;
                index.insert(lowercase_name, *uuid);
            }
            tracing::info!(
                "Migrated {} entities to lowercase index DB",
                name_index.len()
            );
        }

        Ok(index)
    }

    /// Load stemmed name->UUID index, or migrate from name_index if empty
    ///
    /// This enables O(1) linguistic entity lookup: "running" matches "run"
    /// Uses Porter2 stemmer for English language stemming.
    fn load_or_migrate_stemmed_index(
        stemmed_db: &DB,
        name_index: &HashMap<String, Uuid>,
    ) -> Result<HashMap<String, Uuid>> {
        let mut index = HashMap::new();

        // Try to load from dedicated stemmed index DB
        let iter = stemmed_db.iterator(rocksdb::IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if let (Ok(name), Ok(uuid_bytes)) = (
                std::str::from_utf8(&key),
                <[u8; 16]>::try_from(value.as_ref()),
            ) {
                index.insert(name.to_string(), Uuid::from_bytes(uuid_bytes));
            }
        }

        // If empty but name_index has data, migrate (one-time operation)
        if index.is_empty() && !name_index.is_empty() {
            let stemmer = Stemmer::create(Algorithm::English);
            for (name, uuid) in name_index {
                let stemmed_name = Self::stem_entity_name(&stemmer, name);
                stemmed_db.put(stemmed_name.as_bytes(), uuid.as_bytes())?;
                index.insert(stemmed_name, *uuid);
            }
            tracing::info!(
                "Migrated {} entities to stemmed index DB",
                name_index.len()
            );
        }

        Ok(index)
    }

    /// Stem an entity name for linguistic matching
    ///
    /// For multi-word names (e.g., "New York City"), stems each word and joins.
    /// Returns lowercase stemmed version for consistent matching.
    fn stem_entity_name(stemmer: &Stemmer, name: &str) -> String {
        name.split_whitespace()
            .map(|word| stemmer.stem(&word.to_lowercase()).to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Count entries in a RocksDB database (one-time startup cost)
    fn count_db_entries(db: &DB) -> usize {
        db.iterator(rocksdb::IteratorMode::Start).count()
    }

    /// Add or update an entity node
    /// Salience is updated using the formula: salience = base_salience * (1 + 0.1 * ln(mention_count))
    /// This means frequently mentioned entities grow in salience (gravitational wells get heavier)
    ///
    /// BUG-002 FIX: Handles crash recovery for orphaned entities/stale indices
    pub fn add_entity(&self, mut entity: EntityNode) -> Result<Uuid> {
        // Check if entity already exists by name
        let existing_uuid = {
            let index = self.entity_name_index.read();
            index.get(&entity.name).cloned()
        };

        let is_new_entity;
        if let Some(uuid) = existing_uuid {
            // BUG-002 FIX: Verify entity actually exists in DB (handles stale index)
            if let Some(existing) = self.get_entity(&uuid)? {
                // Update existing entity
                entity.uuid = uuid;
                entity.mention_count = existing.mention_count + 1;
                entity.last_seen_at = Utc::now();
                entity.created_at = existing.created_at; // Preserve original creation time
                entity.is_proper_noun = existing.is_proper_noun || entity.is_proper_noun;

                // Update salience with frequency boost
                // Formula: salience = base_salience * (1 + 0.1 * ln(mention_count))
                // This caps at about 1.3x boost at 20 mentions
                let frequency_boost = 1.0 + 0.1 * (entity.mention_count as f32).ln();
                entity.salience = (existing.salience * frequency_boost).min(1.0);
                is_new_entity = false;
            } else {
                // BUG-002 FIX: Stale index entry - entity in index but not in DB
                // Treat as new entity (index will be updated below)
                tracing::warn!(
                    "Stale index entry for entity '{}' (uuid={}), recreating",
                    entity.name,
                    uuid
                );
                entity.uuid = Uuid::new_v4();
                entity.created_at = Utc::now();
                entity.last_seen_at = entity.created_at;
                entity.mention_count = 1;
                is_new_entity = true;
            }
        } else {
            // New entity
            entity.uuid = Uuid::new_v4();
            entity.created_at = Utc::now();
            entity.last_seen_at = entity.created_at;
            entity.mention_count = 1;
            // Salience stays at base_salience for new entities
            is_new_entity = true;
        }

        // BUG-002 FIX: Write index FIRST, then entity
        // Rationale: If crash after index write but before entity write,
        // next add_entity call will detect stale index (above) and recover.
        // This is safer than orphaned entities with no index reference.

        let lowercase_name = entity.name.to_lowercase();
        let stemmer = Stemmer::create(Algorithm::English);
        let stemmed_name = Self::stem_entity_name(&stemmer, &entity.name);

        // Update in-memory indices first
        {
            let mut index = self.entity_name_index.write();
            index.insert(entity.name.clone(), entity.uuid);
        }
        {
            let mut lowercase_index = self.entity_lowercase_index.write();
            lowercase_index.insert(lowercase_name.clone(), entity.uuid);
        }
        {
            let mut stemmed_index = self.entity_stemmed_index.write();
            stemmed_index.insert(stemmed_name.clone(), entity.uuid);
        }

        // Persist name->UUID mappings
        self.entity_name_index_db
            .put(entity.name.as_bytes(), entity.uuid.as_bytes())?;
        self.entity_lowercase_index_db
            .put(lowercase_name.as_bytes(), entity.uuid.as_bytes())?;
        self.entity_stemmed_index_db
            .put(stemmed_name.as_bytes(), entity.uuid.as_bytes())?;

        // Now store entity in database
        let key = entity.uuid.as_bytes();
        let value = bincode::serde::encode_to_vec(&entity, bincode::config::standard())?;
        self.entities_db.put(key, value)?;

        // Increment counter only for truly new entities
        if is_new_entity {
            self.entity_count.fetch_add(1, Ordering::Relaxed);
        }

        Ok(entity.uuid)
    }

    /// Get entity by UUID
    pub fn get_entity(&self, uuid: &Uuid) -> Result<Option<EntityNode>> {
        let key = uuid.as_bytes();
        match self.entities_db.get(key)? {
            Some(value) => {
                let (entity, _): (EntityNode, _) =
                    bincode::serde::decode_from_slice(&value, bincode::config::standard())?;
                Ok(Some(entity))
            }
            None => Ok(None),
        }
    }

    /// Find entity by name (case-insensitive, O(1) lookup)
    ///
    /// Uses a multi-tier matching strategy:
    /// 1. Exact match (O(1)) - fastest
    /// 2. Case-insensitive match (O(1)) - common case
    /// 3. Stemmed match (O(1)) - "running" matches "run"
    /// 4. Substring match - "York" matches "New York City"
    /// 5. Word-level match - "York" matches "New York"
    pub fn find_entity_by_name(&self, name: &str) -> Result<Option<EntityNode>> {
        // Tier 1: Exact match (O(1))
        let uuid = {
            let index = self.entity_name_index.read();
            index.get(name).copied()
        };

        if let Some(uuid) = uuid {
            return self.get_entity(&uuid);
        }

        // Tier 2: Case-insensitive match (O(1))
        let name_lower = name.to_lowercase();
        let uuid = {
            let lowercase_index = self.entity_lowercase_index.read();
            lowercase_index.get(&name_lower).copied()
        };

        if let Some(uuid) = uuid {
            return self.get_entity(&uuid);
        }

        // Tier 3: Stemmed match (O(1)) - "running" matches "run", "conversations" matches "conversation"
        let stemmer = Stemmer::create(Algorithm::English);
        let stemmed_name = Self::stem_entity_name(&stemmer, name);
        let uuid = {
            let stemmed_index = self.entity_stemmed_index.read();
            stemmed_index.get(&stemmed_name).copied()
        };

        if let Some(uuid) = uuid {
            return self.get_entity(&uuid);
        }

        // Tier 4 & 5: Fuzzy matching (O(n) but bounded)
        // Only do fuzzy matching for names >= 3 chars to avoid noise
        if name.len() >= 3 {
            let lowercase_index = self.entity_lowercase_index.read();

            // Tier 4: Substring match - query is substring of entity
            // e.g., "York" matches "New York City"
            for (entity_name, uuid) in lowercase_index.iter() {
                if entity_name.contains(&name_lower) {
                    return self.get_entity(uuid);
                }
            }

            // Tier 5: Word-level match - query matches a word in entity
            // e.g., "York" matches "New York" (word boundary)
            let query_words: Vec<&str> = name_lower.split_whitespace().collect();
            for (entity_name, uuid) in lowercase_index.iter() {
                let entity_words: Vec<&str> = entity_name.split_whitespace().collect();
                // Check if any query word matches any entity word
                for qw in &query_words {
                    if entity_words.iter().any(|ew| ew == qw || ew.starts_with(qw)) {
                        return self.get_entity(uuid);
                    }
                }
            }
        }

        Ok(None)
    }

    /// Find all entities matching a name with fuzzy matching
    ///
    /// Returns multiple matches ranked by match quality.
    /// Useful for spreading activation across related entities.
    pub fn find_entities_fuzzy(&self, name: &str, max_results: usize) -> Result<Vec<EntityNode>> {
        let mut results = Vec::new();
        let name_lower = name.to_lowercase();

        // Skip very short queries
        if name.len() < 2 {
            return Ok(results);
        }

        let lowercase_index = self.entity_lowercase_index.read();

        // Score and collect matches
        let mut scored: Vec<(Uuid, f32)> = Vec::new();

        for (entity_name, uuid) in lowercase_index.iter() {
            let score = if entity_name == &name_lower {
                1.0 // Exact match
            } else if entity_name.starts_with(&name_lower) {
                0.9 // Prefix match
            } else if entity_name.contains(&name_lower) {
                0.7 // Substring match
            } else {
                // Word-level match
                let entity_words: Vec<&str> = entity_name.split_whitespace().collect();
                let query_words: Vec<&str> = name_lower.split_whitespace().collect();

                let mut word_score: f32 = 0.0;
                for qw in &query_words {
                    for ew in &entity_words {
                        if ew == qw {
                            word_score += 0.5;
                        } else if ew.starts_with(qw) {
                            word_score += 0.3;
                        }
                    }
                }
                word_score.min(0.6) // Cap word-level score
            };

            if score > 0.0 {
                scored.push((*uuid, score));
            }
        }

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top results
        for (uuid, _score) in scored.into_iter().take(max_results) {
            if let Some(entity) = self.get_entity(&uuid)? {
                results.push(entity);
            }
        }

        Ok(results)
    }

    /// Find existing relationship between two entities (either direction)
    pub fn find_relationship_between(
        &self,
        entity_a: &Uuid,
        entity_b: &Uuid,
    ) -> Result<Option<RelationshipEdge>> {
        // Check edges from entity_a
        let edges_a = self.get_entity_relationships(entity_a)?;
        for edge in edges_a {
            if (edge.from_entity == *entity_a && edge.to_entity == *entity_b)
                || (edge.from_entity == *entity_b && edge.to_entity == *entity_a)
            {
                return Ok(Some(edge));
            }
        }
        Ok(None)
    }

    /// Add a relationship edge (or strengthen existing one)
    ///
    /// If an edge already exists between the two entities, strengthens it
    /// instead of creating a duplicate. This implements proper Hebbian learning:
    /// "neurons that fire together, wire together" - repeated co-occurrence
    /// strengthens the same synapse rather than creating parallel connections.
    pub fn add_relationship(&self, mut edge: RelationshipEdge) -> Result<Uuid> {
        // Check for existing relationship between these entities
        if let Some(mut existing) = self.find_relationship_between(&edge.from_entity, &edge.to_entity)? {
            // Strengthen existing edge instead of creating duplicate
            existing.strengthen();
            existing.last_activated = Utc::now();

            // Update context if new context is more informative
            if edge.context.len() > existing.context.len() {
                existing.context = edge.context;
            }

            // Persist the strengthened edge
            let key = existing.uuid.as_bytes();
            let value = bincode::serde::encode_to_vec(&existing, bincode::config::standard())?;
            self.relationships_db.put(key, value)?;

            return Ok(existing.uuid);
        }

        // No existing edge - create new one
        edge.uuid = Uuid::new_v4();
        edge.created_at = Utc::now();

        // Store relationship
        let key = edge.uuid.as_bytes();
        let value = bincode::serde::encode_to_vec(&edge, bincode::config::standard())?;
        self.relationships_db.put(key, value)?;

        // Increment relationship counter
        self.relationship_count.fetch_add(1, Ordering::Relaxed);

        // Update entity->edges index for both entities
        self.index_entity_edge(&edge.from_entity, &edge.uuid)?;
        self.index_entity_edge(&edge.to_entity, &edge.uuid)?;

        Ok(edge.uuid)
    }

    /// Index an edge for an entity
    fn index_entity_edge(&self, entity_uuid: &Uuid, edge_uuid: &Uuid) -> Result<()> {
        let key = format!("{entity_uuid}:{edge_uuid}");
        self.entity_edges_db.put(key.as_bytes(), b"1")?;
        Ok(())
    }

    /// Get all relationships for an entity
    pub fn get_entity_relationships(&self, entity_uuid: &Uuid) -> Result<Vec<RelationshipEdge>> {
        let mut edges = Vec::new();
        let prefix = format!("{entity_uuid}:");

        let iter = self.entity_edges_db.prefix_iterator(prefix.as_bytes());
        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }

                // Extract edge UUID
                if let Some(edge_uuid_str) = key_str.split(':').nth(1) {
                    if let Ok(edge_uuid) = Uuid::parse_str(edge_uuid_str) {
                        if let Some(edge) = self.get_relationship(&edge_uuid)? {
                            edges.push(edge);
                        }
                    }
                }
            }
        }

        Ok(edges)
    }

    /// Get relationship by UUID (raw, without decay applied)
    pub fn get_relationship(&self, uuid: &Uuid) -> Result<Option<RelationshipEdge>> {
        let key = uuid.as_bytes();
        match self.relationships_db.get(key)? {
            Some(value) => {
                let (edge, _): (RelationshipEdge, _) =
                    bincode::serde::decode_from_slice(&value, bincode::config::standard())?;
                Ok(Some(edge))
            }
            None => Ok(None),
        }
    }

    /// Get relationship by UUID with effective strength (lazy decay calculation)
    ///
    /// Returns the edge with strength reflecting time-based decay.
    /// This doesn't persist the decay - just calculates what the strength would be.
    /// Use this for API responses to show accurate current strength.
    pub fn get_relationship_with_effective_strength(
        &self,
        uuid: &Uuid,
    ) -> Result<Option<RelationshipEdge>> {
        let key = uuid.as_bytes();
        match self.relationships_db.get(key)? {
            Some(value) => {
                let (mut edge, _): (RelationshipEdge, _) =
                    bincode::serde::decode_from_slice(&value, bincode::config::standard())?;
                // Apply effective strength calculation (doesn't persist)
                edge.strength = edge.effective_strength();
                Ok(Some(edge))
            }
            None => Ok(None),
        }
    }

    /// Delete a relationship by UUID
    ///
    /// Removes the relationship from storage and decrements the counter.
    /// Returns true if the relationship was found and deleted.
    pub fn delete_relationship(&self, uuid: &Uuid) -> Result<bool> {
        let key = uuid.as_bytes();

        // Get the edge first to find the from_entity for index cleanup
        let edge = match self.get_relationship(uuid)? {
            Some(e) => e,
            None => return Ok(false),
        };

        // Delete from main storage
        self.relationships_db.delete(key)?;
        self.relationship_count.fetch_sub(1, Ordering::Relaxed);

        // Also remove from entity_edges index
        let index_key = format!("{}:{}", edge.from_entity, uuid);
        let _ = self.entity_edges_db.delete(index_key.as_bytes());

        Ok(true)
    }

    /// Add an episodic node
    pub fn add_episode(&self, episode: EpisodicNode) -> Result<Uuid> {
        let key = episode.uuid.as_bytes();
        let value = bincode::serde::encode_to_vec(&episode, bincode::config::standard())?;
        self.episodes_db.put(key, value)?;

        // Increment episode counter
        self.episode_count.fetch_add(1, Ordering::Relaxed);

        // Update inverted index: entity_uuid -> episode_uuid
        for entity_uuid in &episode.entity_refs {
            self.index_entity_episode(entity_uuid, &episode.uuid)?;
        }

        Ok(episode.uuid)
    }

    /// Index an episode for an entity (inverted index)
    fn index_entity_episode(&self, entity_uuid: &Uuid, episode_uuid: &Uuid) -> Result<()> {
        let key = format!("{entity_uuid}:{episode_uuid}");
        self.entity_episodes_db.put(key.as_bytes(), b"1")?;
        Ok(())
    }

    /// Get episode by UUID
    pub fn get_episode(&self, uuid: &Uuid) -> Result<Option<EpisodicNode>> {
        let key = uuid.as_bytes();
        match self.episodes_db.get(key)? {
            Some(value) => {
                let (episode, _): (EpisodicNode, _) =
                    bincode::serde::decode_from_slice(&value, bincode::config::standard())?;
                Ok(Some(episode))
            }
            None => Ok(None),
        }
    }

    /// Get all episodes that contain a specific entity
    ///
    /// Uses inverted index for O(k) lookup instead of O(n) full scan.
    /// Crucial for spreading activation algorithm.
    pub fn get_episodes_by_entity(&self, entity_uuid: &Uuid) -> Result<Vec<EpisodicNode>> {
        let mut episodes = Vec::new();
        let prefix = format!("{entity_uuid}:");

        // Use inverted index: entity_uuid -> episode_uuids
        let iter = self.entity_episodes_db.prefix_iterator(prefix.as_bytes());
        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break; // Prefix iterator exhausted
                }

                // Extract episode UUID from key
                if let Some(episode_uuid_str) = key_str.split(':').nth(1) {
                    if let Ok(episode_uuid) = Uuid::parse_str(episode_uuid_str) {
                        if let Some(episode) = self.get_episode(&episode_uuid)? {
                            episodes.push(episode);
                        }
                    }
                }
            }
        }

        Ok(episodes)
    }

    /// Traverse graph starting from an entity (breadth-first)
    ///
    /// Implements Hebbian learning: edges traversed during retrieval are strengthened.
    /// This means frequently accessed pathways become stronger over time.
    ///
    /// Returns `TraversedEntity` with hop distance and decay factor for proper scoring:
    /// - hop 0 (start entity): decay = 1.0
    /// - hop 1: decay = 0.7
    /// - hop 2: decay = 0.49
    /// - etc.
    pub fn traverse_from_entity(
        &self,
        start_uuid: &Uuid,
        max_depth: usize,
    ) -> Result<GraphTraversal> {
        // Use tuned decay from constants (0.15 max decay → ~86% retention per hop)
        // This enables deeper traversal than the old 0.7 factor
        use crate::constants::IMPORTANCE_DECAY_MAX;
        let hop_decay_factor: f32 = (-IMPORTANCE_DECAY_MAX).exp(); // e^(-0.15) ≈ 0.86

        let mut visited_entities = HashSet::new();
        let mut visited_edges = HashSet::new();
        let mut current_level: Vec<(Uuid, usize)> = vec![(*start_uuid, 0)]; // (uuid, hop_distance)
        let mut all_entities: Vec<TraversedEntity> = Vec::new();
        let mut all_edges = Vec::new();
        let mut edges_to_strengthen = Vec::new();

        visited_entities.insert(*start_uuid);
        if let Some(entity) = self.get_entity(start_uuid)? {
            all_entities.push(TraversedEntity {
                entity,
                hop_distance: 0,
                decay_factor: 1.0,
            });
        }

        for depth in 0..max_depth {
            let mut next_level = Vec::new();

            for (entity_uuid, _hop) in &current_level {
                let edges = self.get_entity_relationships(entity_uuid)?;

                for edge in edges {
                    if visited_edges.contains(&edge.uuid) {
                        continue;
                    }

                    visited_edges.insert(edge.uuid);

                    // Only traverse non-invalidated edges
                    if edge.invalidated_at.is_some() {
                        continue;
                    }

                    // Collect edge UUID for Hebbian strengthening
                    edges_to_strengthen.push(edge.uuid);

                    // Return edge with effective strength (lazy decay calculation)
                    let mut edge_with_decay = edge.clone();
                    edge_with_decay.strength = edge_with_decay.effective_strength();
                    all_edges.push(edge_with_decay);

                    // Add connected entity
                    let connected_uuid = if edge.from_entity == *entity_uuid {
                        edge.to_entity
                    } else {
                        edge.from_entity
                    };

                    if !visited_entities.contains(&connected_uuid) {
                        visited_entities.insert(connected_uuid);
                        let next_hop = depth + 1;
                        let decay = hop_decay_factor.powi(next_hop as i32);

                        if let Some(entity) = self.get_entity(&connected_uuid)? {
                            all_entities.push(TraversedEntity {
                                entity,
                                hop_distance: next_hop,
                                decay_factor: decay,
                            });
                        }
                        next_level.push((connected_uuid, next_hop));
                    }
                }
            }

            if next_level.is_empty() {
                break;
            }

            current_level = next_level;
        }

        // Apply Hebbian strengthening to all traversed edges atomically (SHO-65)
        // "Neurons that fire together, wire together"
        // Uses batch update for efficiency instead of individual writes
        if !edges_to_strengthen.is_empty() {
            match self.batch_strengthen_synapses(&edges_to_strengthen) {
                Ok(count) => {
                    if count > 0 {
                        tracing::trace!("Strengthened {} synapses during traversal", count);
                    }
                }
                Err(e) => {
                    tracing::debug!("Failed to batch strengthen synapses: {}", e);
                }
            }
        }

        Ok(GraphTraversal {
            entities: all_entities,
            relationships: all_edges,
        })
    }

    /// Invalidate a relationship (temporal edge invalidation)
    pub fn invalidate_relationship(&self, edge_uuid: &Uuid) -> Result<()> {
        if let Some(mut edge) = self.get_relationship(edge_uuid)? {
            edge.invalidated_at = Some(Utc::now());

            let key = edge.uuid.as_bytes();
            let value = bincode::serde::encode_to_vec(&edge, bincode::config::standard())?;
            self.relationships_db.put(key, value)?;
        }

        Ok(())
    }

    /// Strengthen a synapse (Hebbian learning)
    ///
    /// Called when an edge is traversed during memory retrieval.
    /// Implements "neurons that fire together, wire together".
    ///
    /// Uses a mutex to prevent race conditions during concurrent updates (SHO-64).
    pub fn strengthen_synapse(&self, edge_uuid: &Uuid) -> Result<()> {
        // Lock to prevent concurrent read-modify-write race conditions
        let _guard = self.synapse_update_lock.lock();

        if let Some(mut edge) = self.get_relationship(edge_uuid)? {
            edge.strengthen();

            let key = edge.uuid.as_bytes();
            let value = bincode::serde::encode_to_vec(&edge, bincode::config::standard())?;
            self.relationships_db.put(key, value)?;
        }

        Ok(())
    }

    /// Batch strengthen multiple synapses atomically (SHO-65)
    ///
    /// More efficient than calling strengthen_synapse individually for each edge.
    /// Uses RocksDB WriteBatch for atomic multi-write and a single lock acquisition.
    ///
    /// Returns the number of synapses successfully strengthened.
    pub fn batch_strengthen_synapses(&self, edge_uuids: &[Uuid]) -> Result<usize> {
        if edge_uuids.is_empty() {
            return Ok(0);
        }

        // Single lock acquisition for entire batch
        let _guard = self.synapse_update_lock.lock();

        let mut batch = WriteBatch::default();
        let mut strengthened = 0;

        for edge_uuid in edge_uuids {
            if let Some(mut edge) = self.get_relationship(edge_uuid)? {
                edge.strengthen();

                let key = edge.uuid.as_bytes();
                match bincode::serde::encode_to_vec(&edge, bincode::config::standard()) {
                    Ok(value) => {
                        batch.put(key, value);
                        strengthened += 1;
                    }
                    Err(e) => {
                        tracing::debug!("Failed to serialize edge {}: {}", edge_uuid, e);
                    }
                }
            }
        }

        // Atomic write of all updates
        if strengthened > 0 {
            self.relationships_db.write(batch)?;
        }

        Ok(strengthened)
    }

    /// Record co-retrieval of memories (Hebbian learning between memories)
    ///
    /// When memories are retrieved together, they form associations.
    /// This creates or strengthens CoRetrieved edges between all pairs of memories.
    ///
    /// Note: Limits to top N memories to avoid O(n²) explosion on large retrievals.
    /// Returns the number of edges created/strengthened.
    pub fn record_memory_coactivation(&self, memory_ids: &[Uuid]) -> Result<usize> {
        const MAX_COACTIVATION_SIZE: usize = 20;

        // Limit to top N to bound worst-case complexity
        let memories_to_process = if memory_ids.len() > MAX_COACTIVATION_SIZE {
            &memory_ids[..MAX_COACTIVATION_SIZE]
        } else {
            memory_ids
        };

        if memories_to_process.len() < 2 {
            return Ok(0);
        }

        let _guard = self.synapse_update_lock.lock();
        let mut batch = WriteBatch::default();
        let mut edges_updated = 0;

        // Process all pairs
        for i in 0..memories_to_process.len() {
            for j in (i + 1)..memories_to_process.len() {
                let mem_a = memories_to_process[i];
                let mem_b = memories_to_process[j];

                // Try to find existing edge between these memories
                let existing_edge = self.find_edge_between_entities(&mem_a, &mem_b)?;

                if let Some(mut edge) = existing_edge {
                    // Strengthen existing edge
                    edge.strengthen();
                    let key = edge.uuid.as_bytes();
                    if let Ok(value) =
                        bincode::serde::encode_to_vec(&edge, bincode::config::standard())
                    {
                        batch.put(key, value);
                        edges_updated += 1;
                    }
                } else {
                    // Create new CoRetrieved edge (bidirectional represented as single edge)
                    let edge = RelationshipEdge {
                        uuid: Uuid::new_v4(),
                        from_entity: mem_a,
                        to_entity: mem_b,
                        relation_type: RelationType::CoRetrieved,
                        strength: 0.5, // Initial strength
                        created_at: Utc::now(),
                        valid_at: Utc::now(),
                        invalidated_at: None,
                        source_episode_id: None,
                        context: String::new(),
                        last_activated: Utc::now(),
                        activation_count: 1,
                        potentiated: false,
                    };

                    let key = edge.uuid.as_bytes();
                    if let Ok(value) =
                        bincode::serde::encode_to_vec(&edge, bincode::config::standard())
                    {
                        batch.put(key, value);

                        // Also index in the reverse direction for lookup
                        let idx_key_fwd = format!("mem_edge:{}:{}", mem_a, mem_b);
                        let idx_key_rev = format!("mem_edge:{}:{}", mem_b, mem_a);
                        batch.put(idx_key_fwd.as_bytes(), edge.uuid.as_bytes());
                        batch.put(idx_key_rev.as_bytes(), edge.uuid.as_bytes());

                        edges_updated += 1;
                    }
                }
            }
        }

        if edges_updated > 0 {
            self.relationships_db.write(batch)?;
        }

        Ok(edges_updated)
    }

    /// Find an edge between two entities/memories (in either direction)
    fn find_edge_between_entities(
        &self,
        entity_a: &Uuid,
        entity_b: &Uuid,
    ) -> Result<Option<RelationshipEdge>> {
        // Check forward index
        let idx_key = format!("mem_edge:{}:{}", entity_a, entity_b);
        if let Some(edge_uuid_bytes) = self.relationships_db.get(idx_key.as_bytes())? {
            if edge_uuid_bytes.len() == 16 {
                let edge_uuid = Uuid::from_slice(&edge_uuid_bytes)?;
                return self.get_relationship(&edge_uuid);
            }
        }

        // Check reverse index
        let idx_key_rev = format!("mem_edge:{}:{}", entity_b, entity_a);
        if let Some(edge_uuid_bytes) = self.relationships_db.get(idx_key_rev.as_bytes())? {
            if edge_uuid_bytes.len() == 16 {
                let edge_uuid = Uuid::from_slice(&edge_uuid_bytes)?;
                return self.get_relationship(&edge_uuid);
            }
        }

        Ok(None)
    }

    /// Batch strengthen edges between memory pairs from replay consolidation
    ///
    /// Takes edge boosts from memory replay and applies Hebbian strengthening.
    /// Creates edges if they don't exist, strengthens if they do.
    /// Returns the number of edges successfully strengthened.
    pub fn strengthen_memory_edges(&self, edge_boosts: &[(String, String, f32)]) -> Result<usize> {
        if edge_boosts.is_empty() {
            return Ok(0);
        }

        let _guard = self.synapse_update_lock.lock();
        let mut batch = WriteBatch::default();
        let mut strengthened = 0;

        for (from_id_str, to_id_str, _boost) in edge_boosts {
            // Parse UUIDs
            let from_uuid = match Uuid::parse_str(from_id_str) {
                Ok(u) => u,
                Err(_) => {
                    tracing::debug!("Invalid from_id UUID: {}", from_id_str);
                    continue;
                }
            };
            let to_uuid = match Uuid::parse_str(to_id_str) {
                Ok(u) => u,
                Err(_) => {
                    tracing::debug!("Invalid to_id UUID: {}", to_id_str);
                    continue;
                }
            };

            // Find or create edge
            let existing_edge = self.find_edge_between_entities(&from_uuid, &to_uuid)?;

            if let Some(mut edge) = existing_edge {
                // Strengthen existing edge
                edge.strengthen();
                let key = edge.uuid.as_bytes();
                if let Ok(value) = bincode::serde::encode_to_vec(&edge, bincode::config::standard())
                {
                    batch.put(key, value);
                    strengthened += 1;
                }
            } else {
                // Create new ReplayStrengthened edge
                let edge = RelationshipEdge {
                    uuid: Uuid::new_v4(),
                    from_entity: from_uuid,
                    to_entity: to_uuid,
                    relation_type: RelationType::CoRetrieved, // Replay strengthens co-retrieval associations
                    strength: 0.5,                            // Initial strength
                    created_at: Utc::now(),
                    valid_at: Utc::now(),
                    invalidated_at: None,
                    source_episode_id: None,
                    context: "replay_strengthened".to_string(),
                    last_activated: Utc::now(),
                    activation_count: 1,
                    potentiated: false,
                };

                let key = edge.uuid.as_bytes();
                if let Ok(value) = bincode::serde::encode_to_vec(&edge, bincode::config::standard())
                {
                    batch.put(key, value);

                    // Index both directions
                    let idx_key_fwd = format!("mem_edge:{}:{}", from_uuid, to_uuid);
                    let idx_key_rev = format!("mem_edge:{}:{}", to_uuid, from_uuid);
                    batch.put(idx_key_fwd.as_bytes(), edge.uuid.as_bytes());
                    batch.put(idx_key_rev.as_bytes(), edge.uuid.as_bytes());

                    strengthened += 1;
                }
            }
        }

        if strengthened > 0 {
            self.relationships_db.write(batch)?;
            tracing::debug!(
                "Applied {} edge boosts from replay consolidation",
                strengthened
            );
        }

        Ok(strengthened)
    }

    /// Find memories associated with a given memory through co-retrieval
    ///
    /// Uses weighted graph traversal prioritizing stronger associations.
    /// Returns memory UUIDs sorted by association strength.
    pub fn find_memory_associations(
        &self,
        memory_id: &Uuid,
        max_results: usize,
    ) -> Result<Vec<(Uuid, f32)>> {
        let mut associations: Vec<(Uuid, f32)> = Vec::new();

        // Scan for edges involving this memory
        let prefix_fwd = format!("mem_edge:{}:", memory_id);

        let iter = self.relationships_db.prefix_iterator(prefix_fwd.as_bytes());
        for item in iter {
            let (key, value) = item?;

            // Check if this is our prefix (RocksDB prefix_iterator may return extra)
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix_fwd) {
                break;
            }

            // Get edge UUID from value and look up edge
            if value.len() == 16 {
                let edge_uuid = Uuid::from_slice(&value)?;
                if let Some(edge) = self.get_relationship(&edge_uuid)? {
                    // Get the other memory in this edge
                    let other_id = if edge.from_entity == *memory_id {
                        edge.to_entity
                    } else {
                        edge.from_entity
                    };

                    // Get effective strength with decay
                    let effective_strength = edge.effective_strength();
                    if effective_strength > LTP_MIN_STRENGTH {
                        associations.push((other_id, effective_strength));
                    }
                }
            }
        }

        // Sort by strength descending and limit
        associations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        associations.truncate(max_results);

        Ok(associations)
    }

    /// Get average Hebbian strength for a memory based on its entity relationships
    ///
    /// This looks up the entities referenced by the memory and averages their
    /// relationship strengths in the graph. Used for composite relevance scoring.
    ///
    /// The algorithm:
    /// 1. Look up memory's EpisodicNode (memory_id.0 == episode UUID)
    /// 2. Get entity_refs from the episode
    /// 3. For each entity, get relationships using get_entity_relationships
    /// 4. Filter to edges where both endpoints are in the memory's entity set
    /// 5. Return average effective_strength of these intra-memory edges
    ///
    /// Returns 0.5 (neutral) if no entities or relationships found.
    pub fn get_memory_hebbian_strength(&self, memory_id: &crate::memory::MemoryId) -> Option<f32> {
        // 1. Look up EpisodicNode for this memory (memory_id.0 == episode UUID)
        let episode = match self.get_episode(&memory_id.0) {
            Ok(Some(ep)) => ep,
            Ok(None) => return Some(0.5), // No episode found - neutral
            Err(_) => return Some(0.5),   // Error - neutral fallback
        };

        // 2. Get entity references from the episode
        if episode.entity_refs.is_empty() {
            return Some(0.5); // No entities - neutral
        }

        // Build a set of entity UUIDs for fast lookup
        let entity_set: std::collections::HashSet<Uuid> =
            episode.entity_refs.iter().cloned().collect();

        // 3. Collect all intra-memory relationship strengths
        let mut strengths: Vec<f32> = Vec::new();
        let mut seen_edges: std::collections::HashSet<Uuid> = std::collections::HashSet::new();

        for entity_uuid in &episode.entity_refs {
            if let Ok(edges) = self.get_entity_relationships(entity_uuid) {
                for edge in edges {
                    // Skip if already processed (edges are bidirectional in lookup)
                    if seen_edges.contains(&edge.uuid) {
                        continue;
                    }
                    seen_edges.insert(edge.uuid);

                    // 4. Only count edges where BOTH endpoints are in this memory's entities
                    if entity_set.contains(&edge.from_entity)
                        && entity_set.contains(&edge.to_entity)
                    {
                        // Skip invalidated edges
                        if edge.invalidated_at.is_some() {
                            continue;
                        }
                        // Use effective_strength which applies time-based decay
                        strengths.push(edge.effective_strength());
                    }
                }
            }
        }

        // 5. Return average strength, or neutral if no intra-memory edges
        if strengths.is_empty() {
            Some(0.5)
        } else {
            let avg = strengths.iter().sum::<f32>() / strengths.len() as f32;
            Some(avg)
        }
    }

    /// Apply decay to a synapse
    ///
    /// Returns true if the synapse should be pruned (non-potentiated and below threshold)
    ///
    /// Uses a mutex to prevent race conditions during concurrent updates (SHO-64).
    pub fn decay_synapse(&self, edge_uuid: &Uuid) -> Result<bool> {
        // Lock to prevent concurrent read-modify-write race conditions
        let _guard = self.synapse_update_lock.lock();

        if let Some(mut edge) = self.get_relationship(edge_uuid)? {
            let should_prune = edge.decay();

            let key = edge.uuid.as_bytes();
            let value = bincode::serde::encode_to_vec(&edge, bincode::config::standard())?;
            self.relationships_db.put(key, value)?;

            return Ok(should_prune);
        }

        Ok(false)
    }

    /// Batch decay multiple synapses atomically
    ///
    /// Returns a vector of edge UUIDs that should be pruned.
    pub fn batch_decay_synapses(&self, edge_uuids: &[Uuid]) -> Result<Vec<Uuid>> {
        if edge_uuids.is_empty() {
            return Ok(Vec::new());
        }

        // Single lock acquisition for entire batch
        let _guard = self.synapse_update_lock.lock();

        let mut batch = WriteBatch::default();
        let mut to_prune = Vec::new();

        for edge_uuid in edge_uuids {
            if let Some(mut edge) = self.get_relationship(edge_uuid)? {
                let should_prune = edge.decay();

                let key = edge.uuid.as_bytes();
                match bincode::serde::encode_to_vec(&edge, bincode::config::standard()) {
                    Ok(value) => {
                        batch.put(key, value);
                        if should_prune {
                            to_prune.push(*edge_uuid);
                        }
                    }
                    Err(e) => {
                        tracing::debug!("Failed to serialize edge {}: {}", edge_uuid, e);
                    }
                }
            }
        }

        // Atomic write of all updates
        self.relationships_db.write(batch)?;

        Ok(to_prune)
    }

    /// Apply decay to all synapses and prune weak edges (AUD-2)
    ///
    /// Called during maintenance cycle to:
    /// 1. Apply time-based decay to all edge strengths
    /// 2. Remove edges that have decayed below threshold
    ///
    /// Returns the number of edges pruned.
    pub fn apply_decay(&self) -> Result<usize> {
        // Get all edge UUIDs
        let edge_uuids: Vec<Uuid> = self
            .get_all_relationships()?
            .into_iter()
            .map(|e| e.uuid)
            .collect();

        if edge_uuids.is_empty() {
            return Ok(0);
        }

        // Apply decay and get edges to prune
        let to_prune = self.batch_decay_synapses(&edge_uuids)?;

        // Delete pruned edges
        let mut pruned_count = 0;
        for edge_uuid in &to_prune {
            if self.delete_relationship(edge_uuid)? {
                pruned_count += 1;
            }
        }

        if pruned_count > 0 {
            tracing::debug!(
                "Graph decay: {} edges pruned (of {} total)",
                pruned_count,
                edge_uuids.len()
            );
        }

        Ok(pruned_count)
    }

    /// Get graph statistics - O(1) using atomic counters
    pub fn get_stats(&self) -> Result<GraphStats> {
        Ok(GraphStats {
            entity_count: self.entity_count.load(Ordering::Relaxed),
            relationship_count: self.relationship_count.load(Ordering::Relaxed),
            episode_count: self.episode_count.load(Ordering::Relaxed),
        })
    }

    /// Clear all graph data (entities, relationships, episodes)
    /// Returns the number of items cleared (entities, relationships, episodes)
    pub fn clear_all(&self) -> Result<(usize, usize, usize)> {
        let entity_count = self.entity_count.load(Ordering::Relaxed);
        let relationship_count = self.relationship_count.load(Ordering::Relaxed);
        let episode_count = self.episode_count.load(Ordering::Relaxed);

        // Clear all databases by iterating and deleting
        // Entities
        let keys: Vec<_> = self
            .entities_db
            .iterator(rocksdb::IteratorMode::Start)
            .flatten()
            .map(|(k, _)| k.to_vec())
            .collect();
        for key in &keys {
            self.entities_db.delete(key)?;
        }

        // Relationships
        let keys: Vec<_> = self
            .relationships_db
            .iterator(rocksdb::IteratorMode::Start)
            .flatten()
            .map(|(k, _)| k.to_vec())
            .collect();
        for key in &keys {
            self.relationships_db.delete(key)?;
        }

        // Episodes
        let keys: Vec<_> = self
            .episodes_db
            .iterator(rocksdb::IteratorMode::Start)
            .flatten()
            .map(|(k, _)| k.to_vec())
            .collect();
        for key in &keys {
            self.episodes_db.delete(key)?;
        }

        // Entity edges index
        let keys: Vec<_> = self
            .entity_edges_db
            .iterator(rocksdb::IteratorMode::Start)
            .flatten()
            .map(|(k, _)| k.to_vec())
            .collect();
        for key in &keys {
            self.entity_edges_db.delete(key)?;
        }

        // Entity episodes index
        let keys: Vec<_> = self
            .entity_episodes_db
            .iterator(rocksdb::IteratorMode::Start)
            .flatten()
            .map(|(k, _)| k.to_vec())
            .collect();
        for key in &keys {
            self.entity_episodes_db.delete(key)?;
        }

        // Entity name index
        let keys: Vec<_> = self
            .entity_name_index_db
            .iterator(rocksdb::IteratorMode::Start)
            .flatten()
            .map(|(k, _)| k.to_vec())
            .collect();
        for key in &keys {
            self.entity_name_index_db.delete(key)?;
        }

        // Entity lowercase index
        let keys: Vec<_> = self
            .entity_lowercase_index_db
            .iterator(rocksdb::IteratorMode::Start)
            .flatten()
            .map(|(k, _)| k.to_vec())
            .collect();
        for key in &keys {
            self.entity_lowercase_index_db.delete(key)?;
        }

        // Entity stemmed index
        let keys: Vec<_> = self
            .entity_stemmed_index_db
            .iterator(rocksdb::IteratorMode::Start)
            .flatten()
            .map(|(k, _)| k.to_vec())
            .collect();
        for key in &keys {
            self.entity_stemmed_index_db.delete(key)?;
        }

        // Reset in-memory indexes
        self.entity_name_index.write().clear();
        self.entity_lowercase_index.write().clear();
        self.entity_stemmed_index.write().clear();

        // Reset counters
        self.entity_count.store(0, Ordering::Relaxed);
        self.relationship_count.store(0, Ordering::Relaxed);
        self.episode_count.store(0, Ordering::Relaxed);

        Ok((entity_count, relationship_count, episode_count))
    }

    /// Get all entities in the graph
    pub fn get_all_entities(&self) -> Result<Vec<EntityNode>> {
        let mut entities = Vec::new();

        let iter = self.entities_db.iterator(rocksdb::IteratorMode::Start);
        for (_, value) in iter.flatten() {
            if let Ok(entity) = bincode::serde::decode_from_slice::<EntityNode, _>(
                &value,
                bincode::config::standard(),
            )
            .map(|(v, _)| v)
            {
                entities.push(entity);
            }
        }

        // Sort by mention count (most mentioned first)
        entities.sort_by(|a, b| b.mention_count.cmp(&a.mention_count));

        Ok(entities)
    }

    /// Get all relationships in the graph
    pub fn get_all_relationships(&self) -> Result<Vec<RelationshipEdge>> {
        let mut relationships = Vec::new();

        let iter = self.relationships_db.iterator(rocksdb::IteratorMode::Start);
        for (_, value) in iter.flatten() {
            if let Ok(edge) = bincode::serde::decode_from_slice::<RelationshipEdge, _>(
                &value,
                bincode::config::standard(),
            )
            .map(|(v, _)| v)
            {
                // Only include non-invalidated relationships
                if edge.invalidated_at.is_none() {
                    relationships.push(edge);
                }
            }
        }

        // Sort by strength (strongest first)
        relationships.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(relationships)
    }

    /// Get the Memory Universe visualization data
    /// Returns entities as "stars" with positions based on their relationships,
    /// sized by salience, and colored by entity type.
    pub fn get_universe(&self) -> Result<MemoryUniverse> {
        let entities = self.get_all_entities()?;
        let relationships = self.get_all_relationships()?;

        // Create entity UUID to index mapping for position calculation
        let entity_indices: HashMap<Uuid, usize> = entities
            .iter()
            .enumerate()
            .map(|(i, e)| (e.uuid, i))
            .collect();

        // Calculate 3D positions using a force-directed layout approximation
        // High-salience entities are positioned more centrally
        let mut stars: Vec<UniverseStar> = entities
            .iter()
            .enumerate()
            .map(|(i, entity)| {
                // Use a spiral galaxy layout with salience affecting radius
                // Higher salience = closer to center
                let angle = (i as f32) * 2.4; // Golden angle for even distribution
                let base_radius = 1.0 - entity.salience; // High salience = small radius
                let radius = base_radius * 100.0 + 10.0; // 10-110 range

                let x = radius * angle.cos();
                let y = radius * angle.sin();
                let z = ((i as f32) * 0.1).sin() * 20.0; // Slight z variation

                UniverseStar {
                    id: entity.uuid.to_string(),
                    name: entity.name.clone(),
                    entity_type: entity.labels.first().map(|l| l.as_str().to_string()),
                    salience: entity.salience,
                    mention_count: entity.mention_count,
                    is_proper_noun: entity.is_proper_noun,
                    position: Position3D { x, y, z },
                    color: entity_type_color(entity.labels.first()),
                    size: 5.0 + entity.salience * 20.0, // Size 5-25 based on salience
                }
            })
            .collect();

        // Apply gravitational forces FIRST, before creating connections
        // This ensures connection positions match final star positions
        for rel in &relationships {
            if let (Some(from_idx), Some(to_idx)) = (
                entity_indices.get(&rel.from_entity),
                entity_indices.get(&rel.to_entity),
            ) {
                // Apply small gravitational pull based on connection strength
                let pull_factor = rel.strength * 0.05;

                let from_pos = stars[*from_idx].position.clone();
                let to_pos = stars[*to_idx].position.clone();

                let dx = (to_pos.x - from_pos.x) * pull_factor;
                let dy = (to_pos.y - from_pos.y) * pull_factor;
                let dz = (to_pos.z - from_pos.z) * pull_factor;

                stars[*from_idx].position.x += dx;
                stars[*from_idx].position.y += dy;
                stars[*from_idx].position.z += dz;

                stars[*to_idx].position.x -= dx;
                stars[*to_idx].position.y -= dy;
                stars[*to_idx].position.z -= dz;
            }
        }

        // Create gravitational connections AFTER star positions are finalized
        // This ensures from_position/to_position match current star positions
        let connections: Vec<GravitationalConnection> = relationships
            .iter()
            .filter_map(|rel| {
                let from_idx = entity_indices.get(&rel.from_entity)?;
                let to_idx = entity_indices.get(&rel.to_entity)?;

                Some(GravitationalConnection {
                    id: rel.uuid.to_string(),
                    from_id: rel.from_entity.to_string(),
                    to_id: rel.to_entity.to_string(),
                    relation_type: rel.relation_type.as_str().to_string(),
                    strength: rel.strength,
                    from_position: stars[*from_idx].position.clone(),
                    to_position: stars[*to_idx].position.clone(),
                })
            })
            .collect();

        // Calculate universe bounds
        let (min_x, max_x, min_y, max_y, min_z, max_z) = stars.iter().fold(
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN, f32::MAX, f32::MIN),
            |(min_x, max_x, min_y, max_y, min_z, max_z), star| {
                (
                    min_x.min(star.position.x),
                    max_x.max(star.position.x),
                    min_y.min(star.position.y),
                    max_y.max(star.position.y),
                    min_z.min(star.position.z),
                    max_z.max(star.position.z),
                )
            },
        );

        Ok(MemoryUniverse {
            stars,
            connections,
            total_entities: entities.len(),
            total_connections: relationships.len(),
            bounds: UniverseBounds {
                min: Position3D {
                    x: min_x,
                    y: min_y,
                    z: min_z,
                },
                max: Position3D {
                    x: max_x,
                    y: max_y,
                    z: max_z,
                },
            },
        })
    }
}

/// Helper function to get color for entity type
fn entity_type_color(label: Option<&EntityLabel>) -> String {
    match label {
        Some(EntityLabel::Person) => "#FF6B6B".to_string(), // Coral red
        Some(EntityLabel::Organization) => "#4ECDC4".to_string(), // Teal
        Some(EntityLabel::Location) => "#45B7D1".to_string(), // Sky blue
        Some(EntityLabel::Technology) => "#96CEB4".to_string(), // Sage green
        Some(EntityLabel::Product) => "#FFEAA7".to_string(), // Soft yellow
        Some(EntityLabel::Event) => "#DDA0DD".to_string(),  // Plum
        Some(EntityLabel::Skill) => "#98D8C8".to_string(),  // Mint
        Some(EntityLabel::Concept) => "#F7DC6F".to_string(), // Gold
        Some(EntityLabel::Date) => "#BB8FCE".to_string(),   // Light purple
        Some(EntityLabel::Other(_)) => "#AEB6BF".to_string(), // Gray
        None => "#AEB6BF".to_string(),                      // Gray default
    }
}

/// 3D position in the memory universe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// A star in the memory universe (represents an entity)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniverseStar {
    pub id: String,
    pub name: String,
    pub entity_type: Option<String>,
    pub salience: f32,
    pub mention_count: usize,
    pub is_proper_noun: bool,
    pub position: Position3D,
    pub color: String,
    pub size: f32,
}

/// A gravitational connection between stars (represents a relationship)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GravitationalConnection {
    pub id: String,
    pub from_id: String,
    pub to_id: String,
    pub relation_type: String,
    pub strength: f32,
    pub from_position: Position3D,
    pub to_position: Position3D,
}

/// Bounds of the memory universe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniverseBounds {
    pub min: Position3D,
    pub max: Position3D,
}

/// The complete memory universe visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUniverse {
    pub stars: Vec<UniverseStar>,
    pub connections: Vec<GravitationalConnection>,
    pub total_entities: usize,
    pub total_connections: usize,
    pub bounds: UniverseBounds,
}

/// Entity with hop distance from traversal origin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversedEntity {
    pub entity: EntityNode,
    /// Number of hops from the starting entity (0 = start entity)
    pub hop_distance: usize,
    /// Decay factor based on hop distance: 1.0 at hop 0, decays with each hop
    pub decay_factor: f32,
}

/// Result of graph traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTraversal {
    /// Entities found during traversal with hop distance info
    pub entities: Vec<TraversedEntity>,
    pub relationships: Vec<RelationshipEdge>,
}

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub entity_count: usize,
    pub relationship_count: usize,
    pub episode_count: usize,
}

/// Extracted entity with salience information
#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    pub name: String,
    pub label: EntityLabel,
    pub base_salience: f32,
}

/// Simple entity extraction (rule-based NER) with salience detection
pub struct EntityExtractor {
    /// Common person name indicators
    person_indicators: HashSet<String>,

    /// Common organization indicators (suffixes like Inc, Corp)
    org_indicators: HashSet<String>,

    /// Known organization names (direct matches)
    org_keywords: HashSet<String>,

    /// Known location names (cities, countries, regions)
    location_keywords: HashSet<String>,

    /// Common technology keywords
    tech_keywords: HashSet<String>,

    /// Common words that should NOT be extracted as entities
    /// (stop words that start with capitals at sentence beginning)
    stop_words: HashSet<String>,
}

impl EntityExtractor {
    pub fn new() -> Self {
        let person_indicators: HashSet<String> =
            vec!["mr", "mrs", "ms", "dr", "prof", "sir", "madam"]
                .into_iter()
                .map(String::from)
                .collect();

        let org_indicators: HashSet<String> = vec![
            "inc",
            "corp",
            "ltd",
            "llc",
            "company",
            "corporation",
            "university",
            "institute",
            "foundation",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let tech_keywords: HashSet<String> = vec![
            "rust",
            "python",
            "java",
            "javascript",
            "typescript",
            "react",
            "vue",
            "angular",
            "docker",
            "kubernetes",
            "aws",
            "azure",
            "gcp",
            "sql",
            "nosql",
            "mongodb",
            "postgresql",
            "redis",
            "kafka",
            "api",
            "rest",
            "graphql",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Known organization names (global - India-first, then worldwide)
        let org_keywords: HashSet<String> = vec![
            // Indian Companies - IT/Tech
            "tcs",
            "infosys",
            "wipro",
            "hcl",
            "tech mahindra",
            "cognizant",
            "mindtree",
            "mphasis",
            "ltimindtree",
            "persistent",
            "zensar",
            "cyient",
            "hexaware",
            "coforge",
            "birlasoft",
            "sonata software",
            "mastek",
            "newgen",
            // Indian Companies - Startups/Unicorns
            "flipkart",
            "paytm",
            "zomato",
            "swiggy",
            "ola",
            "oyo",
            "byju's",
            "byjus",
            "razorpay",
            "phonepe",
            "cred",
            "zerodha",
            "groww",
            "upstox",
            "policybazaar",
            "nykaa",
            "meesho",
            "udaan",
            "delhivery",
            "freshworks",
            "zoho",
            "postman",
            "browserstack",
            "chargebee",
            "clevertap",
            "druva",
            "hasura",
            "innovaccer",
            "lenskart",
            "mamaearth",
            "unacademy",
            "vedantu",
            "physicswallah",
            "dream11",
            "mpl",
            "winzo",
            "slice",
            "jupiter",
            "fi",
            "niyo",
            "smallcase",
            "koo",
            "sharechat",
            "dailyhunt",
            "pratilipi",
            "inshorts",
            "rapido",
            "urban company",
            "dunzo",
            "bigbasket",
            "grofers",
            "blinkit",
            "jiomart",
            "tata neu",
            // Indian Conglomerates
            "tata",
            "reliance",
            "jio",
            "adani",
            "birla",
            "mahindra",
            "godrej",
            "bajaj",
            "hdfc",
            "icici",
            "kotak",
            "axis",
            "sbi",
            "bharti",
            "airtel",
            "vodafone",
            "idea",
            "hero",
            "tvs",
            "maruti",
            "suzuki",
            "hyundai",
            "kia",
            "mg",
            "tata motors",
            "larsen",
            "toubro",
            "l&t",
            "itc",
            "hindustan unilever",
            "hul",
            "nestle",
            "britannia",
            "parle",
            "amul",
            "dabur",
            "patanjali",
            "emami",
            "marico",
            // Indian Banks & Finance
            "rbi",
            "sebi",
            "nse",
            "bse",
            "npci",
            "upi",
            "bhim",
            "paisa",
            "mswipe",
            "pine labs",
            "billdesk",
            "ccavenue",
            "instamojo",
            "cashfree",
            // Indian Institutions
            "iit",
            "iim",
            "iisc",
            "nit",
            "bits",
            "isro",
            "drdo",
            "barc",
            "tifr",
            "aiims",
            "iiser",
            "iiit",
            "srm",
            "vit",
            "manipal",
            "amity",
            "lovely",
            // Global Tech Giants
            "microsoft",
            "google",
            "apple",
            "amazon",
            "meta",
            "facebook",
            "netflix",
            "alphabet",
            "youtube",
            "instagram",
            "whatsapp",
            "tiktok",
            "snapchat",
            "twitter",
            "x",
            "linkedin",
            "pinterest",
            "reddit",
            "discord",
            "telegram",
            // Global Enterprise Tech
            "salesforce",
            "oracle",
            "ibm",
            "sap",
            "vmware",
            "dell",
            "hp",
            "hpe",
            "cisco",
            "juniper",
            "palo alto",
            "crowdstrike",
            "fortinet",
            "splunk",
            "servicenow",
            "workday",
            "atlassian",
            "jira",
            "confluence",
            "trello",
            "asana",
            "monday",
            "notion",
            "airtable",
            "figma",
            "canva",
            "miro",
            // Global Cloud & Infrastructure
            "aws",
            "azure",
            "gcp",
            "digitalocean",
            "linode",
            "vultr",
            "cloudflare",
            "akamai",
            "fastly",
            "vercel",
            "netlify",
            "heroku",
            "render",
            "railway",
            // Global Hardware/Chip
            "intel",
            "amd",
            "nvidia",
            "qualcomm",
            "broadcom",
            "arm",
            "tsmc",
            "samsung",
            "mediatek",
            "apple silicon",
            "marvell",
            "micron",
            "sk hynix",
            "western digital",
            // Global AI/ML Companies
            "openai",
            "anthropic",
            "deepmind",
            "cohere",
            "stability",
            "midjourney",
            "hugging face",
            "databricks",
            "snowflake",
            "palantir",
            "c3ai",
            "datarobot",
            // Global Fintech
            "stripe",
            "square",
            "block",
            "paypal",
            "venmo",
            "wise",
            "revolut",
            "robinhood",
            "coinbase",
            "binance",
            "kraken",
            "gemini",
            "ftx",
            "blockchain",
            "ripple",
            // Global Dev Tools
            "github",
            "gitlab",
            "bitbucket",
            "jetbrains",
            "vscode",
            "sublime",
            "vim",
            "docker",
            "kubernetes",
            "terraform",
            "ansible",
            "puppet",
            "chef",
            // Global Consulting
            "accenture",
            "deloitte",
            "pwc",
            "kpmg",
            "ey",
            "mckinsey",
            "bcg",
            "bain",
            // Global Auto/EV
            "tesla",
            "rivian",
            "lucid",
            "nio",
            "byd",
            "xpeng",
            "volkswagen",
            "bmw",
            "mercedes",
            "audi",
            "porsche",
            "toyota",
            "honda",
            "nissan",
            "ford",
            "gm",
            // Global Aerospace
            "spacex",
            "boeing",
            "airbus",
            "lockheed",
            "northrop",
            "raytheon",
            "nasa",
            "esa",
            "jaxa",
            "isro",
            "blue origin",
            "virgin galactic",
            // Universities - India
            "delhi university",
            "jnu",
            "bhu",
            "amu",
            "jadavpur",
            "presidency",
            "st stephens",
            "loyola",
            "xavier",
            "symbiosis",
            "nmims",
            "sp jain",
            "xlri",
            "fms",
            "iift",
            "mdi",
            "great lakes",
            "ism dhanbad",
            // Universities - Global
            "mit",
            "stanford",
            "harvard",
            "yale",
            "princeton",
            "caltech",
            "berkeley",
            "oxford",
            "cambridge",
            "imperial",
            "eth zurich",
            "epfl",
            "tsinghua",
            "peking",
            "nus",
            "nanyang",
            "kaist",
            "university of tokyo",
            "melbourne",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Known location names (global - India-first, then worldwide)
        let location_keywords: HashSet<String> = vec![
            // Indian Metro Cities
            "mumbai",
            "delhi",
            "bangalore",
            "bengaluru",
            "hyderabad",
            "chennai",
            "kolkata",
            "pune",
            "ahmedabad",
            "surat",
            "jaipur",
            "lucknow",
            // Indian Tier-1 Cities
            "kochi",
            "cochin",
            "thiruvananthapuram",
            "trivandrum",
            "coimbatore",
            "madurai",
            "visakhapatnam",
            "vizag",
            "vijayawada",
            "nagpur",
            "indore",
            "bhopal",
            "chandigarh",
            "mohali",
            "panchkula",
            "noida",
            "gurgaon",
            "gurugram",
            "faridabad",
            "ghaziabad",
            "greater noida",
            "dwarka",
            // Indian Tier-2 Cities
            "mysore",
            "mangalore",
            "hubli",
            "belgaum",
            "nashik",
            "aurangabad",
            "rajkot",
            "vadodara",
            "baroda",
            "gandhinagar",
            "kanpur",
            "varanasi",
            "allahabad",
            "prayagraj",
            "agra",
            "meerut",
            "dehradun",
            "rishikesh",
            "haridwar",
            "amritsar",
            "jalandhar",
            "ludhiana",
            "shimla",
            "manali",
            "dharamshala",
            "jammu",
            "srinagar",
            "ranchi",
            "jamshedpur",
            "patna",
            "guwahati",
            "shillong",
            "imphal",
            "kohima",
            "gangtok",
            "darjeeling",
            "bhubaneswar",
            "cuttack",
            "rourkela",
            "raipur",
            "bilaspur",
            // Indian States & UTs
            "maharashtra",
            "karnataka",
            "tamil nadu",
            "telangana",
            "andhra pradesh",
            "kerala",
            "gujarat",
            "rajasthan",
            "uttar pradesh",
            "madhya pradesh",
            "west bengal",
            "bihar",
            "odisha",
            "jharkhand",
            "chhattisgarh",
            "punjab",
            "haryana",
            "himachal pradesh",
            "uttarakhand",
            "goa",
            "assam",
            "meghalaya",
            "manipur",
            "nagaland",
            "tripura",
            "mizoram",
            "arunachal pradesh",
            "sikkim",
            "jammu and kashmir",
            "ladakh",
            // Indian Regions
            "silicon valley of india",
            "electronic city",
            "whitefield",
            "marathahalli",
            "koramangala",
            "indiranagar",
            "hsr layout",
            "jayanagar",
            "malleshwaram",
            "bandra",
            "andheri",
            "powai",
            "lower parel",
            "bkc",
            "navi mumbai",
            "thane",
            "connaught place",
            "nehru place",
            "saket",
            "cyber city",
            "dlf",
            "hitech city",
            "madhapur",
            "gachibowli",
            "ecr",
            "omr",
            "it corridor",
            // Asian Cities
            "singapore",
            "hong kong",
            "tokyo",
            "osaka",
            "seoul",
            "busan",
            "beijing",
            "shanghai",
            "shenzhen",
            "guangzhou",
            "hangzhou",
            "taipei",
            "bangkok",
            "kuala lumpur",
            "jakarta",
            "manila",
            "ho chi minh",
            "hanoi",
            "dubai",
            "abu dhabi",
            "doha",
            "riyadh",
            "tel aviv",
            "istanbul",
            // European Cities
            "london",
            "paris",
            "berlin",
            "munich",
            "frankfurt",
            "amsterdam",
            "rotterdam",
            "brussels",
            "zurich",
            "geneva",
            "vienna",
            "prague",
            "warsaw",
            "budapest",
            "barcelona",
            "madrid",
            "milan",
            "rome",
            "lisbon",
            "dublin",
            "edinburgh",
            "manchester",
            "stockholm",
            "oslo",
            "helsinki",
            "copenhagen",
            "athens",
            "moscow",
            "st petersburg",
            // North American Cities
            "new york",
            "los angeles",
            "san francisco",
            "seattle",
            "boston",
            "chicago",
            "austin",
            "denver",
            "portland",
            "miami",
            "atlanta",
            "dallas",
            "houston",
            "phoenix",
            "san diego",
            "san jose",
            "oakland",
            "palo alto",
            "mountain view",
            "cupertino",
            "menlo park",
            "redwood city",
            "washington dc",
            "philadelphia",
            "detroit",
            "toronto",
            "vancouver",
            "montreal",
            "calgary",
            "ottawa",
            "mexico city",
            "guadalajara",
            // South American Cities
            "sao paulo",
            "rio de janeiro",
            "buenos aires",
            "santiago",
            "bogota",
            "lima",
            "medellin",
            "cartagena",
            // African Cities
            "johannesburg",
            "cape town",
            "lagos",
            "nairobi",
            "cairo",
            "casablanca",
            "accra",
            "addis ababa",
            "kigali",
            // Australian/NZ Cities
            "sydney",
            "melbourne",
            "brisbane",
            "perth",
            "auckland",
            "wellington",
            // Countries - Asia
            "india",
            "china",
            "japan",
            "south korea",
            "korea",
            "taiwan",
            "singapore",
            "malaysia",
            "thailand",
            "vietnam",
            "indonesia",
            "philippines",
            "bangladesh",
            "pakistan",
            "sri lanka",
            "nepal",
            "bhutan",
            "myanmar",
            "cambodia",
            "laos",
            // Countries - Middle East
            "uae",
            "emirates",
            "saudi arabia",
            "qatar",
            "bahrain",
            "kuwait",
            "oman",
            "israel",
            "turkey",
            "iran",
            "iraq",
            "jordan",
            "lebanon",
            "egypt",
            // Countries - Europe
            "uk",
            "united kingdom",
            "britain",
            "england",
            "scotland",
            "wales",
            "ireland",
            "france",
            "germany",
            "italy",
            "spain",
            "portugal",
            "netherlands",
            "belgium",
            "switzerland",
            "austria",
            "poland",
            "czech",
            "hungary",
            "romania",
            "bulgaria",
            "greece",
            "sweden",
            "norway",
            "finland",
            "denmark",
            "russia",
            "ukraine",
            // Countries - Americas
            "usa",
            "united states",
            "america",
            "canada",
            "mexico",
            "brazil",
            "argentina",
            "chile",
            "colombia",
            "peru",
            "venezuela",
            // Countries - Africa/Oceania
            "south africa",
            "nigeria",
            "kenya",
            "ghana",
            "ethiopia",
            "rwanda",
            "australia",
            "new zealand",
            // Famous Tech Hubs
            "silicon valley",
            "bay area",
            "wall street",
            "tech city",
            "shoreditch",
            "station f",
            "blockchain island",
            "crypto valley",
            "startup nation",
            "innovation district",
            "tech park",
            "it park",
            "sez",
            "special economic zone",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Stop words: common words that appear capitalized at sentence start
        // These aren't named entities even when capitalized
        let stop_words: HashSet<String> = vec![
            // Articles & pronouns
            "the", "a", "an", "this", "that", "these", "those", "i", "we", "you", "he", "she", "it",
            "they", // Common verbs (appear at sentence start)
            "is", "are", "was", "were", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", // Question words
            "if", "when", "where", "what", "why", "how",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        Self {
            person_indicators,
            org_indicators,
            org_keywords,
            location_keywords,
            tech_keywords,
            stop_words,
        }
    }

    /// Calculate base salience for an entity based on its type and detection confidence
    ///
    /// Salience values by entity type:
    /// - Person: 0.8 (highest - people are key context)
    /// - Organization/Product: 0.7
    /// - Location/Technology/Event: 0.6
    /// - Skill: 0.5
    /// - Concept: 0.4
    /// - Date/Other: 0.3
    ///
    /// Proper nouns receive a 20% boost (capped at 1.0).
    pub fn calculate_base_salience(label: &EntityLabel, is_proper_noun: bool) -> f32 {
        let type_salience = match label {
            EntityLabel::Person => 0.8,       // People are highly salient
            EntityLabel::Organization => 0.7, // Organizations are important
            EntityLabel::Location => 0.6,     // Locations matter for context
            EntityLabel::Technology => 0.6,   // Tech keywords matter for dev context
            EntityLabel::Product => 0.7,      // Products are specific entities
            EntityLabel::Event => 0.6,        // Events are temporal anchors
            EntityLabel::Skill => 0.5,        // Skills are somewhat important
            EntityLabel::Concept => 0.4,      // Concepts are more generic
            EntityLabel::Date => 0.3,         // Dates are structural, not salient
            EntityLabel::Other(_) => 0.3,     // Unknown types get low salience
        };

        // Proper nouns get a 20% boost
        if is_proper_noun {
            (type_salience * 1.2_f32).min(1.0_f32)
        } else {
            type_salience
        }
    }

    /// Check if a word is likely a proper noun (not just capitalized at sentence start)
    fn is_likely_proper_noun(&self, word: &str, position: usize, prev_char: Option<char>) -> bool {
        // If it's not at position 0 and is capitalized, it's likely a proper noun
        if position > 0 {
            return true;
        }

        // At position 0, check if previous character was punctuation (sentence start)
        // If previous char was '.', '!', '?' then this might just be sentence capitalization
        if let Some(c) = prev_char {
            if c == '.' || c == '!' || c == '?' {
                // It's at sentence start - could be either
                // Check if it's a common word
                let lower = word.to_lowercase();
                return !self.stop_words.contains(&lower);
            }
        }

        // Default to proper noun for capitalized words
        true
    }

    /// Extract entities from text with salience information
    pub fn extract_with_salience(&self, text: &str) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();
        let mut seen = HashSet::new();
        let mut skip_until_index = 0; // For skipping sub-spans of multi-word entities

        // Split into words and detect capitalized sequences
        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            // Skip if this word is part of a multi-word entity we already extracted
            if i < skip_until_index {
                continue;
            }

            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());

            if clean_word.is_empty() {
                continue;
            }

            let lower = clean_word.to_lowercase();

            // Skip common stop words
            if self.stop_words.contains(&lower) {
                continue;
            }

            // Check for known organization keywords (direct match)
            if self.org_keywords.contains(&lower) && !seen.contains(&lower) {
                let entity = ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Organization,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Organization, true),
                };
                entities.push(entity);
                seen.insert(lower.clone());
                continue;
            }

            // Check for known location keywords (direct match)
            if self.location_keywords.contains(&lower) && !seen.contains(&lower) {
                let entity = ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Location,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Location, true),
                };
                entities.push(entity);
                seen.insert(lower.clone());
                continue;
            }

            // Check for technology keywords (always proper nouns in tech context)
            if self.tech_keywords.contains(&lower) && !seen.contains(&lower) {
                let entity = ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Technology,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Technology, true),
                };
                entities.push(entity);
                seen.insert(lower.clone());
                continue;
            }

            // Check for capitalized words (potential entities)
            if clean_word
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
            {
                let mut entity_name = clean_word.to_string();
                let mut entity_label = EntityLabel::Other("Unknown".to_string());

                // Determine previous character for proper noun detection
                let prev_char = if i > 0 {
                    words[i - 1].chars().last()
                } else {
                    None
                };

                let is_proper = self.is_likely_proper_noun(clean_word, i, prev_char);

                // Check for person indicators
                if i > 0
                    && self
                        .person_indicators
                        .contains(&words[i - 1].to_lowercase())
                {
                    entity_label = EntityLabel::Person;
                }

                // Check for multi-word capitalized sequences
                let mut j = i + 1;
                while j < words.len()
                    && words[j]
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                {
                    let next_word = words[j].trim_matches(|c: char| !c.is_alphanumeric());
                    // Skip stop words in multi-word sequences
                    if !self.stop_words.contains(&next_word.to_lowercase()) {
                        entity_name.push(' ');
                        entity_name.push_str(next_word);
                    }
                    j += 1;
                }

                // Set skip_until_index to avoid extracting sub-spans
                // e.g., if we extracted "John Smith", skip "Smith" on next iteration
                if j > i + 1 {
                    skip_until_index = j;
                }

                let entity_name_lower = entity_name.to_lowercase();

                // Check multi-word entity against known lists
                if self.org_keywords.contains(&entity_name_lower) {
                    entity_label = EntityLabel::Organization;
                } else if self.location_keywords.contains(&entity_name_lower) {
                    entity_label = EntityLabel::Location;
                }

                // Check for organization indicators (suffixes)
                if matches!(entity_label, EntityLabel::Other(_)) {
                    for word in entity_name.split_whitespace() {
                        if self.org_indicators.contains(&word.to_lowercase()) {
                            entity_label = EntityLabel::Organization;
                            break;
                        }
                    }
                }

                // Only extract entities we have evidence for
                // Don't guess on single unknown capitalized words - they're often noise
                if matches!(entity_label, EntityLabel::Other(_)) {
                    if entity_name.contains(' ') {
                        // Multi-word capitalized sequences (like "John Smith", "New York")
                        // are likely proper names - extract as Person
                        entity_label = EntityLabel::Person;
                    } else {
                        // Single capitalized word not in any keyword list
                        // Skip it - we don't have enough evidence it's a real entity
                        // The neural NER model handles these cases properly
                        continue;
                    }
                }

                let entity_key = entity_name_lower;
                if !seen.contains(&entity_key) {
                    let base_salience = Self::calculate_base_salience(&entity_label, is_proper);
                    let entity = ExtractedEntity {
                        name: entity_name,
                        label: entity_label,
                        base_salience,
                    };
                    entities.push(entity);
                    seen.insert(entity_key);
                }
            }
        }

        entities
    }
}

impl Default for EntityExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    /// Create a test relationship edge with specified strength and last_activated
    fn create_test_edge(strength: f32, days_since_activated: i64) -> RelationshipEdge {
        RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: Uuid::new_v4(),
            to_entity: Uuid::new_v4(),
            relation_type: RelationType::RelatedTo,
            strength,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: Utc::now() - Duration::days(days_since_activated),
            activation_count: 0,
            potentiated: false,
        }
    }

    #[test]
    fn test_hebbian_strengthen_increases_strength() {
        let mut edge = create_test_edge(0.5, 0);
        let initial_strength = edge.strength;

        edge.strengthen();

        assert!(
            edge.strength > initial_strength,
            "Strengthen should increase strength"
        );
        assert_eq!(edge.activation_count, 1);
    }

    #[test]
    fn test_hebbian_strengthen_asymptotic() {
        let mut edge = create_test_edge(0.95, 0);

        edge.strengthen();

        // High strength should still increase but slowly (asymptotic to 1.0)
        assert!(edge.strength > 0.95);
        assert!(edge.strength <= 1.0);
    }

    #[test]
    fn test_hebbian_strengthen_formula() {
        // Test: w_new = w_old + η × (1 - w_old) where η = 0.1
        let mut edge = create_test_edge(0.5, 0);

        edge.strengthen();

        // Expected: 0.5 + 0.1 * (1 - 0.5) = 0.5 + 0.05 = 0.55
        let expected = 0.5 + 0.1 * 0.5;
        assert!(
            (edge.strength - expected).abs() < 0.001,
            "Expected {}, got {}",
            expected,
            edge.strength
        );
    }

    #[test]
    fn test_ltp_threshold_potentiation() {
        let mut edge = create_test_edge(0.5, 0);
        assert!(!edge.potentiated);

        // Strengthen 10 times (LTP_THRESHOLD = 10)
        for _ in 0..10 {
            edge.strengthen();
        }

        assert!(
            edge.potentiated,
            "Should be potentiated after 10 activations"
        );
        assert!(
            edge.strength > 0.7,
            "Potentiated edge should have bonus strength"
        );
    }

    #[test]
    fn test_decay_reduces_strength() {
        let mut edge = create_test_edge(0.5, 7); // 7 days elapsed

        let initial_strength = edge.strength;
        edge.decay();

        assert!(
            edge.strength < initial_strength,
            "Decay should reduce strength"
        );
    }

    #[test]
    fn test_decay_hybrid_model() {
        // Test hybrid decay: exponential (< 3 days) → power-law (≥ 3 days)
        // At 14 days with β=0.5:
        // - Crossover value at 3 days: e^(-0.693 * 3) ≈ 0.125
        // - Power-law from crossover: 0.125 * (14/3)^(-0.5) ≈ 0.058
        let mut edge = create_test_edge(1.0, 14);

        edge.decay();

        // Hybrid decay at 14 days should be much less than old exponential 0.5
        // Expected ~0.058 for normal, allowing some tolerance
        assert!(
            edge.strength < 0.15,
            "After 14 days with hybrid decay, strength should be < 0.15, got {}",
            edge.strength
        );
        assert!(
            edge.strength > 0.01,
            "Strength should still be above floor, got {}",
            edge.strength
        );
    }

    #[test]
    fn test_decay_minimum_floor() {
        let mut edge = create_test_edge(0.02, 100); // Very old, very weak

        edge.decay();

        assert!(
            edge.strength >= LTP_MIN_STRENGTH,
            "Strength should not go below minimum floor"
        );
    }

    #[test]
    fn test_potentiated_decay_slower() {
        let mut edge1 = create_test_edge(0.8, 14);
        let mut edge2 = create_test_edge(0.8, 14);
        edge2.potentiated = true;

        edge1.decay();
        edge2.decay();

        assert!(
            edge2.strength > edge1.strength,
            "Potentiated edge should decay slower"
        );
    }

    #[test]
    fn test_effective_strength_read_only() {
        let edge = create_test_edge(0.5, 7);
        let initial_strength = edge.strength;

        let effective = edge.effective_strength();

        // effective_strength should not modify the edge
        assert_eq!(edge.strength, initial_strength);
        assert!(effective < initial_strength);
    }

    #[test]
    fn test_decay_prune_threshold() {
        let mut weak_edge = create_test_edge(LTP_MIN_STRENGTH, 30);
        weak_edge.potentiated = false;

        let should_prune = weak_edge.decay();

        // Non-potentiated edge at minimum strength after decay should be prunable
        assert!(
            should_prune,
            "Weak non-potentiated edge should be marked for pruning"
        );
    }

    #[test]
    fn test_potentiated_never_pruned() {
        let mut edge = create_test_edge(LTP_MIN_STRENGTH, 100);
        edge.potentiated = true;

        let should_prune = edge.decay();

        assert!(
            !should_prune,
            "Potentiated edges should never be pruned regardless of strength"
        );
    }

    #[test]
    fn test_salience_calculation() {
        let person_salience = EntityExtractor::calculate_base_salience(&EntityLabel::Person, false);
        let person_proper_salience =
            EntityExtractor::calculate_base_salience(&EntityLabel::Person, true);

        assert_eq!(person_salience, 0.8);
        assert!((person_proper_salience - 0.96).abs() < 0.01); // 0.8 * 1.2 = 0.96
    }

    #[test]
    fn test_salience_caps_at_one() {
        // Person (0.8) * 1.2 = 0.96, should not exceed 1.0
        let salience = EntityExtractor::calculate_base_salience(&EntityLabel::Person, true);
        assert!(salience <= 1.0);
    }

    #[test]
    fn test_hebbian_strength_no_episode() {
        // Create a temporary graph memory for testing
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path()).unwrap();

        // Random memory ID with no associated episode should return 0.5 (neutral)
        let fake_memory_id = crate::memory::MemoryId(Uuid::new_v4());
        let strength = graph.get_memory_hebbian_strength(&fake_memory_id);
        assert_eq!(strength, Some(0.5), "No episode should return neutral 0.5");
    }

    #[test]
    fn test_hebbian_strength_with_episode_no_edges() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path()).unwrap();

        // Create entities
        let entity1 = EntityNode {
            uuid: Uuid::new_v4(),
            name: "Entity1".to_string(),
            labels: vec![EntityLabel::Person],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
        };
        let entity2 = EntityNode {
            uuid: Uuid::new_v4(),
            name: "Entity2".to_string(),
            labels: vec![EntityLabel::Organization],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
        };

        let entity1_uuid = graph.add_entity(entity1.clone()).unwrap();
        let entity2_uuid = graph.add_entity(entity2.clone()).unwrap();

        // Create episode with entities but no edges
        let memory_id = crate::memory::MemoryId(Uuid::new_v4());
        let episode = EpisodicNode {
            uuid: memory_id.0,
            name: "Test Episode".to_string(),
            content: "Test content".to_string(),
            valid_at: Utc::now(),
            created_at: Utc::now(),
            entity_refs: vec![entity1_uuid, entity2_uuid],
            source: EpisodeSource::Message,
            metadata: std::collections::HashMap::new(),
        };
        graph.add_episode(episode).unwrap();

        // Episode with entities but no edges should return 0.5
        let strength = graph.get_memory_hebbian_strength(&memory_id);
        assert_eq!(
            strength,
            Some(0.5),
            "Episode without edges should return neutral 0.5"
        );
    }

    #[test]
    fn test_hebbian_strength_with_edges() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path()).unwrap();

        // Create entities
        let entity1_uuid = Uuid::new_v4();
        let entity2_uuid = Uuid::new_v4();

        let entity1 = EntityNode {
            uuid: entity1_uuid,
            name: "Entity1".to_string(),
            labels: vec![EntityLabel::Person],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
        };
        let entity2 = EntityNode {
            uuid: entity2_uuid,
            name: "Entity2".to_string(),
            labels: vec![EntityLabel::Organization],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
        };

        graph.add_entity(entity1).unwrap();
        graph.add_entity(entity2).unwrap();

        // Create episode
        let memory_id = crate::memory::MemoryId(Uuid::new_v4());
        let episode = EpisodicNode {
            uuid: memory_id.0,
            name: "Test Episode".to_string(),
            content: "Test content".to_string(),
            valid_at: Utc::now(),
            created_at: Utc::now(),
            entity_refs: vec![entity1_uuid, entity2_uuid],
            source: EpisodeSource::Message,
            metadata: std::collections::HashMap::new(),
        };
        graph.add_episode(episode).unwrap();

        // Create edge between entities with known strength
        let edge = RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: entity1_uuid,
            to_entity: entity2_uuid,
            relation_type: RelationType::RelatedTo,
            strength: 0.8,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: Some(memory_id.0),
            context: "Test context".to_string(),
            last_activated: Utc::now(), // Just activated - no decay
            activation_count: 5,
            potentiated: false,
        };
        graph.add_relationship(edge).unwrap();

        // Should return the edge strength (0.8, with minimal decay since just activated)
        let strength = graph.get_memory_hebbian_strength(&memory_id);
        assert!(strength.is_some());
        let s = strength.unwrap();
        assert!(s > 0.75 && s <= 0.8, "Strength should be ~0.8, got {}", s);
    }
}
