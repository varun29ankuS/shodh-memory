//! Graph Memory System - Inspired by Graphiti
//!
//! Temporal knowledge graph for tracking entities, relationships, and episodic memories.
//! Implements bi-temporal tracking and hybrid retrieval (semantic + graph traversal).

use anyhow::Result;
use chrono::{DateTime, Utc};
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, Options, WriteBatch, DB};
use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use uuid::Uuid;

use crate::constants::{
    ENTITY_CONCEPT_MERGE_THRESHOLD, ENTITY_EMBEDDING_CACHE_MAX, LTP_MIN_STRENGTH, LTP_PRUNE_FLOOR,
};

// Column family names for the unified graph database
const CF_ENTITIES: &str = "entities";
const CF_RELATIONSHIPS: &str = "relationships";
const CF_EPISODES: &str = "episodes";
const CF_ENTITY_EDGES: &str = "entity_edges";
const CF_ENTITY_PAIR_INDEX: &str = "entity_pair_index";
const CF_ENTITY_EPISODES: &str = "entity_episodes";
const CF_NAME_INDEX: &str = "name_index";
const CF_LOWERCASE_INDEX: &str = "lowercase_index";
const CF_STEMMED_INDEX: &str = "stemmed_index";
const CF_RELATION_STATS: &str = "relation_stats";
/// Alias index: a surface form (lowercased) -> the canonical entity UUID it
/// resolves to. Seeded by the entity resolver (ER Phase 1.2/1.3); consulted first
/// in `find_entity_by_name` so `container ship`/`cargo ship`/`the Dali` all land
/// on one canonical node instead of minting parallel mention nodes.
const CF_ALIAS: &str = "entity_alias";

const GRAPH_CF_NAMES: &[&str] = &[
    CF_ENTITIES,
    CF_RELATIONSHIPS,
    CF_EPISODES,
    CF_ENTITY_EDGES,
    CF_ENTITY_PAIR_INDEX,
    CF_ENTITY_EPISODES,
    CF_NAME_INDEX,
    CF_LOWERCASE_INDEX,
    CF_STEMMED_INDEX,
    CF_RELATION_STATS,
    CF_ALIAS,
];

/// Per-(label-pair, relation) evidence counter for the learned pair table
/// (Stanford-1, PMI² relation mapping). `src_is_a` counts observations where
/// the relation's SOURCE entity carried the canonically-first label.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PairRelationStat {
    relation: RelationType,
    count: u64,
    src_is_a: u64,
}

impl PairRelationStat {
    fn new(relation: RelationType) -> Self {
        Self {
            relation,
            count: 0,
            src_is_a: 0,
        }
    }
}

/// Stable key string for an entity label in relation-stats keys.
fn label_key(label: &EntityLabel) -> String {
    format!("{label:?}")
}

/// Labels too generic to carry a learned relation default. The first
/// measurement (batched guard run 27348362950) showed one label-pair
/// mass-applying CreatedBy to 344 cue-less co-mentions (open_domain -0.067):
/// catch-all labels make every co-mention look like the same "pair", so the
/// purity gate sees a clean signal that is actually label poverty. Learned
/// mappings require both endpoints to carry a SPECIFIC label.
fn label_is_generic(label: &EntityLabel) -> bool {
    matches!(
        label,
        EntityLabel::Concept | EntityLabel::Keyword | EntityLabel::Other(_)
    )
}

/// Canonicalize an unordered label pair for stats keys. Returns
/// (first, second, input_src_is_first): the two label keys in lexicographic
/// order plus whether the FIRST argument landed in the first slot.
fn canonical_label_pair(a: &EntityLabel, b: &EntityLabel) -> (String, String, bool) {
    let ka = label_key(a);
    let kb = label_key(b);
    if ka <= kb {
        (ka, kb, true)
    } else {
        (kb, ka, false)
    }
}

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

    /// Curvature selectivity: stdev(incident edge curvatures) / degree
    ///
    /// Measures whether this entity participates in specific communities
    /// (high selectivity) or connects to everything uniformly (low selectivity).
    ///
    /// High selectivity → concept (e.g., "Hebbian learning" connects within neuroscience)
    /// Low selectivity  → stop word (e.g., "impl" connects to everything equally)
    ///
    /// Used to gate LTP protection: low-selectivity entities cannot earn
    /// permanent LTP regardless of activation count, mimicking habituation
    /// in biological neural systems.
    ///
    /// Computed during Forman-Ricci curvature pass. None = not yet computed.
    #[serde(default)]
    pub selectivity: Option<f32>,

    /// Fine-grained schema leaf type (e.g. `"bridge"`, `"malware"`), one level
    /// more specific than the coarse [`EntityLabel`]. Populated by the GLiNER
    /// schema-driven typer (`crate::entity_type`); `None` for entities typed
    /// only at coarse granularity (regex/tag/heuristic extraction paths, or
    /// records written before this field existed).
    #[serde(default)]
    pub fine_type: Option<String>,

    /// Stable real-world identity: a Wikidata QID (e.g. `"Q37156"` for IBM), set
    /// by offline KB linking (`crate::kb`) when a mention resolves unambiguously.
    ///
    /// Two nodes carrying the same `kb_id` are the same real-world thing — that
    /// is the entire claim, and it is one no amount of string similarity can
    /// make: `IBM` and `International Business Machines` share almost no
    /// characters. `None` is the norm and is never an error; it means the
    /// surface was unknown to the KB, or known but too ambiguous to link, and
    /// abstaining is the designed behaviour rather than a gap.
    ///
    /// Write-once: set only when `None`, never overwritten. A node that already
    /// carries an identity keeps it even if a later mention resolves differently
    /// (see `add_entity`), because silently repointing an identity is exactly
    /// the corruption KB linking exists to avoid.
    #[serde(default)]
    pub kb_id: Option<String>,
}

fn default_salience() -> f32 {
    0.5 // Default middle salience
}

/// Entity labels for ontological classification of graph nodes.
///
/// Extends Graphiti's base categorization with DevOps, software engineering,
/// and robotics domain types. Used by spreading activation (Layer 2) and
/// post-RRF re-ranking (Layer 4.9) for type-aware retrieval.
///
/// New variants are additive only — never remove existing variants to
/// maintain backward compatibility with MessagePack-serialized data.
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
    /// YAKE-extracted discriminative keyword (not a named entity)
    /// Used for graph-based retrieval of rare/important terms like "sunrise"
    Keyword,
    /// Git repos, Jira projects, feature work, mission plans
    Project,
    /// Issues, tickets, work items, action items
    Task,
    /// READMEs, specs, design docs, RFCs, runbooks
    Document,
    /// Git repositories, container registries
    Repository,
    /// Microservices, APIs, daemons, ROS2 nodes
    Service,
    /// PostgreSQL, Redis, RocksDB, DynamoDB instances
    Database,
    /// Latency p99, error rate, SLOs, telemetry signals
    Metric,
    /// Env vars, feature flags, config files, parameters
    Configuration,
    /// staging, production, dev, CI, robot deployment zones
    Environment,
    /// CI/CD pipelines, data pipelines, ETL, mission workflows
    Pipeline,
    /// Engineering squads, SRE teams, robot fleet groups
    Team,
    /// SRE, tech lead, PM, architect, operator
    Role,
    /// Code modules, packages, crates, libraries, ROS2 packages
    Module,
    /// Nationalities, religious, or political groups (schema coarse: `norp`)
    Norp,
    /// Geopolitical entity — countries, cities, states as political actors (schema coarse: `gpe`)
    Gpe,
    /// Buildings, airports, highways, bridges (schema coarse: `facility`)
    Facility,
    /// Cars, trains, ships, aircraft (schema coarse: `vehicle`)
    Vehicle,
    /// Weapons and munitions (schema coarse: `weapon`)
    Weapon,
    /// Titled creative works — books, films, songs, artworks (schema coarse: `work`)
    Work,
    /// Named laws, treaties, regulations, court cases (schema coarse: `law`)
    Law,
    /// Honorifics and named positions — "President", "CEO" (schema coarse: `title`)
    Title,
    /// Malware, CVEs, threat actors, cyber campaigns (schema coarse: `cyber`)
    Cyber,
    /// Monetary amounts (schema coarse: `money`)
    Money,
    /// Non-monetary measurements — distance, weight, percentages (schema coarse: `quantity`)
    Quantity,
    /// Times of day / durations, distinct from calendar `Date` (schema coarse: `time`)
    Time,
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
            Self::Keyword => "Keyword",
            Self::Project => "Project",
            Self::Task => "Task",
            Self::Document => "Document",
            Self::Repository => "Repository",
            Self::Service => "Service",
            Self::Database => "Database",
            Self::Metric => "Metric",
            Self::Configuration => "Configuration",
            Self::Environment => "Environment",
            Self::Pipeline => "Pipeline",
            Self::Team => "Team",
            Self::Role => "Role",
            Self::Module => "Module",
            Self::Norp => "Norp",
            Self::Gpe => "Gpe",
            Self::Facility => "Facility",
            Self::Vehicle => "Vehicle",
            Self::Weapon => "Weapon",
            Self::Work => "Work",
            Self::Law => "Law",
            Self::Title => "Title",
            Self::Cyber => "Cyber",
            Self::Money => "Money",
            Self::Quantity => "Quantity",
            Self::Time => "Time",
            Self::Other(s) => s.as_str(),
        }
    }

    /// Map a schema coarse id (from `crate::entity_type::schema().coarse`) to the
    /// matching `EntityLabel` variant.
    ///
    /// All 18 coarse ids in the entity-type schema resolve to a real variant —
    /// GLiNER typing must never degrade a schema-recognized coarse class to
    /// `Other(String)`. An id outside the schema (e.g. a stale/foreign value)
    /// still falls back to `Other` rather than panicking.
    pub fn from_coarse_id(id: &str) -> EntityLabel {
        match id {
            "person" => EntityLabel::Person,
            "organization" => EntityLabel::Organization,
            "location" => EntityLabel::Location,
            "product" => EntityLabel::Product,
            "event" => EntityLabel::Event,
            "date" => EntityLabel::Date,
            "norp" => EntityLabel::Norp,
            "gpe" => EntityLabel::Gpe,
            "facility" => EntityLabel::Facility,
            "vehicle" => EntityLabel::Vehicle,
            "weapon" => EntityLabel::Weapon,
            "work" => EntityLabel::Work,
            "law" => EntityLabel::Law,
            "title" => EntityLabel::Title,
            "cyber" => EntityLabel::Cyber,
            "money" => EntityLabel::Money,
            "quantity" => EntityLabel::Quantity,
            "time" => EntityLabel::Time,
            other => EntityLabel::Other(other.to_string()),
        }
    }

    /// Static type hierarchy: returns parent labels for ontological matching.
    ///
    /// Enables hierarchical type matching so "who manages" can match Team
    /// (because Team is_a Organization-like group entity) and "what technology"
    /// can match Service (because Service is_a Technology subtype).
    ///
    /// Hierarchy is shallow (max depth 1) and fixed at compile time.
    /// No graph storage, no runtime cost beyond a match.
    ///
    /// Reference: Collins & Quillian (1969) "Retrieval time from semantic memory"
    /// — type hierarchies reduce retrieval time for plausible paths.
    pub fn parent_labels(&self) -> &'static [EntityLabel] {
        match self {
            // Organizational subtypes
            Self::Team => &[EntityLabel::Organization],
            Self::Role => &[EntityLabel::Concept],
            // Technical subtypes
            Self::Service | Self::Database | Self::Repository | Self::Module | Self::Pipeline => {
                &[EntityLabel::Technology]
            }
            // Work / knowledge subtypes
            Self::Task => &[EntityLabel::Event],
            Self::Document
            | Self::Configuration
            | Self::Metric
            | Self::Environment
            | Self::Project => &[EntityLabel::Concept],
            // GLiNER coarse subtypes (schema-driven typing)
            Self::Gpe | Self::Facility => &[EntityLabel::Location],
            Self::Vehicle | Self::Weapon => &[EntityLabel::Product],
            Self::Title => &[EntityLabel::Role],
            Self::Work | Self::Law | Self::Cyber => &[EntityLabel::Concept],
            Self::Norp => &[EntityLabel::Organization],
            // Base types and Other have no parents
            _ => &[],
        }
    }

    /// Check if this label matches the expected label, considering type hierarchy.
    ///
    /// Returns true if:
    /// - Direct match (self == expected)
    /// - Self's parent matches expected (Team matches Organization)
    ///
    /// Used by spreading activation (Layer 2 entity penalty) and
    /// Layer 4.9 (ontological re-ranking) for hierarchical type matching.
    pub fn matches_with_hierarchy(&self, expected: &EntityLabel) -> bool {
        if self == expected {
            return true;
        }
        self.parent_labels().contains(expected)
    }
}

/// Classify a tag string into a richer `EntityLabel`.
///
/// Tags are short, user-supplied descriptors (e.g. "production", "rocksdb",
/// "config.toml"). This maps them onto the ontology so they participate in
/// type-aware spreading activation and hierarchy matching during retrieval.
/// Shared by every ingest path so a tag receives the same label whether it
/// arrives via `remember`, an integration sync, or an `upsert`.
pub fn classify_tag_label(tag: &str) -> EntityLabel {
    let lower = tag.to_lowercase();

    // Deployment / environment indicators
    if matches!(
        lower.as_str(),
        "production"
            | "staging"
            | "dev"
            | "development"
            | "ci"
            | "cd"
            | "kubernetes"
            | "k8s"
            | "docker"
            | "container"
            | "aws"
            | "gcp"
            | "azure"
    ) {
        return EntityLabel::Environment;
    }

    // Pipeline / workflow indicators
    if lower.contains("pipeline")
        || lower.contains("workflow")
        || lower.contains("ci-cd")
        || lower.contains("cicd")
    {
        return EntityLabel::Pipeline;
    }

    // Database / storage indicators
    if matches!(
        lower.as_str(),
        "rocksdb"
            | "postgres"
            | "postgresql"
            | "redis"
            | "sqlite"
            | "mongodb"
            | "mysql"
            | "dynamodb"
            | "s3"
            | "cassandra"
            | "elasticsearch"
    ) || lower.ends_with("db")
        || lower.ends_with("-db")
        || lower.ends_with("_db")
    {
        return EntityLabel::Database;
    }

    // Service / API indicators (suffix-only to avoid "my-api-docs" false positives)
    if lower.ends_with("-service")
        || lower.ends_with("_service")
        || lower.ends_with("-api")
        || lower.ends_with("_api")
        || lower.ends_with("-server")
        || lower.ends_with("_server")
        || lower.ends_with("-daemon")
    {
        return EntityLabel::Service;
    }

    // Documentation indicators (check before module — README.md is a doc, not a module)
    if lower.ends_with(".md")
        || lower.contains("readme")
        || lower.contains("runbook")
        || lower.ends_with("-rfc")
        || lower.ends_with("-spec")
    {
        return EntityLabel::Document;
    }

    // Configuration indicators
    if lower.ends_with(".toml")
        || lower.ends_with(".yaml")
        || lower.ends_with(".yml")
        || lower.ends_with(".env")
        || lower.ends_with(".json")
        || lower.contains("config")
    {
        return EntityLabel::Configuration;
    }

    // Module / library indicators
    if lower.ends_with(".rs")
        || lower.ends_with(".ts")
        || lower.ends_with(".js")
        || lower.ends_with(".py")
        || lower.ends_with("-lib")
        || lower.ends_with("_lib")
    {
        return EntityLabel::Module;
    }

    // Default: Concept — an honest "we did not recognise this".
    //
    // This used to fall through to `Technology`, which is a CLAIM, not a
    // default: every unrecognised tag asserted a specific ontological class.
    // The rules above are a dev-ops keyword matcher (kubernetes, postgres,
    // ci-cd, …), so on any corpus that is not about infrastructure almost
    // everything reaches this line — and the graph renders a uniform wall of
    // "Technology" nodes under a legend that promises an ontology. A visibly
    // wrong type is worse than a visibly absent one: it is indistinguishable
    // from a confident correct answer.
    //
    // `Concept` is the same label the NER path already uses for entities whose
    // class it could not resolve (`NerEntityType::Misc`), so unrecognised
    // surfaces from both paths now agree instead of disagreeing.
    EntityLabel::Concept
}

/// Memory tier for edge consolidation
///
/// Based on hippocampal-cortical memory consolidation research:
/// - L1 (Working): Dense, fast encoding, aggressive pruning (Dentate Gyrus-like)
/// - L2 (Episodic): Moderate density, Hebbian selection (CA1/CA3-like)
/// - L3 (Semantic): Sparse, near-permanent (Neocortex-like)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EdgeTier {
    /// Working memory tier: new edges, dense, aggressive decay
    #[default]
    L1Working,
    /// Episodic memory tier: proven edges, moderate decay
    L2Episodic,
    /// Semantic memory tier: consolidated edges, near-permanent
    L3Semantic,
}

impl EdgeTier {
    /// Get the initial weight for edges in this tier
    pub fn initial_weight(&self) -> f32 {
        use crate::constants::*;
        match self {
            Self::L1Working => L1_INITIAL_WEIGHT,
            Self::L2Episodic => L2_PROMOTION_WEIGHT,
            Self::L3Semantic => L3_PROMOTION_WEIGHT,
        }
    }

    /// Get the prune threshold for this tier
    pub fn prune_threshold(&self) -> f32 {
        use crate::constants::*;
        match self {
            Self::L1Working => L1_PRUNE_THRESHOLD,
            Self::L2Episodic => L2_PRUNE_THRESHOLD,
            Self::L3Semantic => L3_PRUNE_THRESHOLD,
        }
    }

    /// Get the promotion threshold to move to next tier
    pub fn promotion_threshold(&self) -> Option<f32> {
        use crate::constants::*;
        match self {
            Self::L1Working => Some(L1_PROMOTION_THRESHOLD),
            Self::L2Episodic => Some(L2_PROMOTION_THRESHOLD),
            Self::L3Semantic => None, // Already at highest tier
        }
    }

    /// Get the next tier (for promotion)
    pub fn next_tier(&self) -> Option<Self> {
        match self {
            Self::L1Working => Some(Self::L2Episodic),
            Self::L2Episodic => Some(Self::L3Semantic),
            Self::L3Semantic => None,
        }
    }

    /// How fast an edge in this tier experiences time on the Wixted hybrid decay
    /// curve, relative to the L2/episodic reference.
    ///
    /// L1 does not use the hybrid curve at all (it has its own aggressive
    /// exponential via `tier_decay_factor`), so its value here is the inert
    /// reference 1.0 and is never read; L2 IS the reference; L3 is
    /// [`crate::decay::L3_TIME_SCALE_VS_L2`], the ratio that
    /// `L2_DECAY_PER_DAY` and `L3_DECAY_PER_MONTH` jointly assert.
    pub fn decay_time_scale(&self) -> f64 {
        match self {
            Self::L1Working | Self::L2Episodic => 1.0,
            Self::L3Semantic => crate::decay::L3_TIME_SCALE_VS_L2,
        }
    }

    /// Minimum elapsed time, in seconds, between the moment an edge entered this
    /// tier and the moment it may leave it for the next one. `None` for L3:
    /// there is no next tier.
    ///
    /// Strength alone is not evidence of consolidation. From the birth weight of
    /// 0.4 a single `strengthen` call clears `L1_PROMOTION_THRESHOLD` and three
    /// clear `L2_PROMOTION_THRESHOLD`, while three independent strengthen paths
    /// touch the same entity edges within one request — so without a gate, one
    /// conversational turn promoted an edge from "working" all the way to
    /// "near-permanent semantic".
    ///
    /// The separations deliberately **mirror the memory-tier clock** rather than
    /// inventing a second set of numbers: 30 minutes for the first step
    /// ([`TIER_PROMOTION_WORKING_AGE_SECS`](crate::constants::TIER_PROMOTION_WORKING_AGE_SECS))
    /// and 24 hours for the second
    /// ([`TIER_PROMOTION_SESSION_AGE_SECS`](crate::constants::TIER_PROMOTION_SESSION_AGE_SECS)).
    /// Those are the synaptic-consolidation window (McGaugh 2000) and the
    /// sleep-dependent hippocampal→cortical transfer window (Rasch & Born 2013)
    /// respectively — the same biology the edge tiers cite, so the two
    /// consolidation clocks at least tick in the same units.
    ///
    /// This is now **one of two** ways to satisfy the sustained-evidence gate;
    /// see [`promotion_min_episodes`](Self::promotion_min_episodes) for why the
    /// clock alone was not merely conservative but wrong.
    pub fn promotion_min_separation_secs(&self) -> Option<i64> {
        use crate::constants::*;
        match self {
            Self::L1Working => Some(TIER_PROMOTION_WORKING_AGE_SECS),
            Self::L2Episodic => Some(TIER_PROMOTION_SESSION_AGE_SECS),
            Self::L3Semantic => None,
        }
    }

    /// Minimum number of **distinct attesting episodes** an edge must carry to
    /// leave this tier. `None` for L3: there is no next tier.
    ///
    /// The clock was a *proxy* for independent evidence, and it was the wrong
    /// one. Elapsed minutes are a property of the ingest schedule, not of the
    /// evidence: a corpus imported in one pass — which is how every import,
    /// every eval run and every seeded deployment works — creates all of its
    /// edges within minutes, so under a clock-only gate not one of them could
    /// ever promote. They stayed at L1 with `EDGE_TIER_TRUST_L1` (0.20, a 4×
    /// retrieval-trust penalty against L3) and then aged out on the L1 prune
    /// schedule. That is not a consolidation policy; it is scheduled data loss
    /// that happens to spare interactively-built graphs.
    ///
    /// Distinct episodes measure the thing the clock was standing in for. The
    /// anti-burst intent is fully preserved, because the counter is
    /// deduplicated by `source_episode_id` in `merge_provenance`: one
    /// conversation mentioning two entities forty times is ONE episode however
    /// many times it is re-read, and however many of the three in-request
    /// strengthen paths touch the edge. Only a genuinely different source
    /// memory advances it.
    ///
    /// See [`TIER_PROMOTION_L2_MIN_EPISODES`](crate::constants::TIER_PROMOTION_L2_MIN_EPISODES)
    /// and [`TIER_PROMOTION_L3_MIN_EPISODES`](crate::constants::TIER_PROMOTION_L3_MIN_EPISODES)
    /// for the values and for why cumulative counts give "evidence acquired
    /// since entering this tier" without persisting a per-tier counter.
    pub fn promotion_min_episodes(&self) -> Option<usize> {
        use crate::constants::*;
        match self {
            Self::L1Working => Some(TIER_PROMOTION_L2_MIN_EPISODES),
            Self::L2Episodic => Some(TIER_PROMOTION_L3_MIN_EPISODES),
            Self::L3Semantic => None,
        }
    }
}

/// Long-Term Potentiation status for edges (PIPE-4)
///
/// Multi-scale LTP based on neuroscience research:
/// - Burst: Temporary protection from high-frequency activation (E-LTP)
/// - Weekly: Moderate protection from consistent routine use (L-LTP)
/// - Full: Maximum protection from sustained long-term use (systems consolidation)
///
/// Reference: Frey & Morris (1997) "Synaptic tagging and long-term potentiation"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LtpStatus {
    /// Not potentiated - normal decay applies
    #[default]
    None,

    /// Burst potentiated: 5+ activations in 24 hours
    /// Temporary protection (2x slower decay) that expires after 48h
    /// Represents early-phase LTP (protein synthesis independent)
    Burst {
        /// When burst was detected (for expiration check)
        #[serde(with = "chrono::serde::ts_seconds")]
        detected_at: DateTime<Utc>,
    },

    /// Weekly potentiated: 3+/week for 2+ weeks
    /// Moderate protection (3x slower decay)
    /// Represents late-phase LTP (habit formation)
    Weekly,

    /// Fully potentiated: 10+ activations OR 5+ over 30 days
    /// Maximum protection (10x slower decay)
    /// Represents systems consolidation (semantic memory)
    Full,
}

impl LtpStatus {
    /// Get the decay factor for this LTP status
    pub fn decay_factor(&self) -> f32 {
        use crate::constants::*;
        match self {
            Self::None => 1.0,
            Self::Burst { detected_at } => {
                // Check if burst has expired. Uses the frozen scoring clock
                // (SHODH_EVAL_NOW) because this feeds effective_strength() on
                // the read path: a live clock makes the burst-protection factor
                // flip between recall repeats minutes apart, wobbling edge
                // strength and graph activation. Production leaves the env
                // unset, so this is Utc::now() there.
                let hours_since = (crate::memory::scoring_now() - *detected_at).num_hours();
                if hours_since > LTP_BURST_DURATION_HOURS {
                    1.0 // Expired, normal decay
                } else {
                    LTP_BURST_DECAY_FACTOR
                }
            }
            Self::Weekly => LTP_WEEKLY_DECAY_FACTOR,
            Self::Full => LTP_DECAY_FACTOR,
        }
    }

    /// Check if this status provides any protection
    pub fn is_potentiated(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Check if burst protection has expired
    pub fn is_burst_expired(&self) -> bool {
        use crate::constants::LTP_BURST_DURATION_HOURS;
        match self {
            Self::Burst { detected_at } => {
                // Frozen scoring clock on the read path (see decay_factor).
                (crate::memory::scoring_now() - *detected_at).num_hours() > LTP_BURST_DURATION_HOURS
            }
            _ => false,
        }
    }

    /// Get priority for LTP upgrades (higher = stronger protection)
    pub fn priority(&self) -> u8 {
        match self {
            Self::None => 0,
            Self::Burst { .. } => 1,
            Self::Weekly => 2,
            Self::Full => 3,
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

    /// Long-Term Potentiation status (PIPE-4: multi-scale LTP)
    /// Replaces simple bool with tiered protection levels:
    /// - None: Normal decay
    /// - Burst: Temporary 2x protection (5+ activations in 24h)
    /// - Weekly: Moderate 3x protection (3+/week for 2 weeks)
    /// - Full: Maximum 10x protection (10+ activations or 5+ over 30 days)
    #[serde(default)]
    pub ltp_status: LtpStatus,

    /// Memory tier for consolidation (L1→L2→L3)
    /// Edges start in L1 (working memory) and promote based on Hebbian strength
    #[serde(default)]
    pub tier: EdgeTier,

    /// Activation timestamp history for temporal pattern detection (PIPE-4)
    /// Only populated for L2+ edges (L1 edges die too quickly to need history)
    /// Capacity: L2 = 20 timestamps, L3 = 50 timestamps
    /// Enables: burst detection, weekly patterns, temporal query relevance
    #[serde(default)]
    pub activation_timestamps: Option<VecDeque<DateTime<Utc>>>,

    /// Entity extraction confidence (PIPE-5: Unified LTP Readiness)
    /// Average confidence of the entities connected by this edge.
    /// Affects LTP threshold: high confidence → faster LTP (7 activations)
    /// Low confidence → slower LTP (13 activations).
    /// Based on synaptic tagging: behaviorally relevant synapses consolidate faster.
    #[serde(default)]
    pub entity_confidence: Option<f32>,

    /// Minimum curvature selectivity of the two endpoint entities.
    ///
    /// Cached from EntityNode.selectivity during curvature computation.
    /// Used by decay() to gate LTP protection: if the weakest endpoint
    /// is a stop word (low selectivity), LTP protection is reduced,
    /// allowing curvature-accelerated decay to clean up noise edges
    /// even if they're frequently co-activated.
    ///
    /// None = not yet computed (full LTP protection applies).
    #[serde(default)]
    pub endpoint_selectivity: Option<f32>,

    /// Forman-Ricci curvature of this edge (Leal, Restrepo, Stadler, Jost 2018)
    ///
    /// For directed graphs:
    ///   F(→e→) = 2 - in_deg(source) - out_deg(target)   [flow-through]
    ///   F(←e←) = 2 - out_deg(source) - in_deg(target)   [flow-loss]
    ///   F(e)   = F(→e→) + F(←e←) = 4 - deg(source) - deg(target)
    ///
    /// Interpretation:
    ///   Positive: edge connects low-degree nodes (tight community interior)
    ///   Near zero: neutral / transitional
    ///   Negative: edge bridges high-degree hubs (information bottleneck)
    ///
    /// Computed during heavy maintenance cycle. None = not yet computed.
    ///
    /// Reference: arXiv:1811.07825 — "Forman-Ricci curvature for hypergraphs"
    #[serde(default)]
    pub forman_curvature: Option<f32>,

    /// Provenance trail: every source episode that attested this edge.
    ///
    /// Increment 1 (robust edge provenance): an edge is rarely attested by a
    /// single observation. Each `ProvenanceRecord` records one source episode
    /// that contributed to this edge, with its mention count, observation
    /// window, optional confidence, an optional char-span REFERENCE into that
    /// episode's content, and the typing method that decided the relation type
    /// for that attestation. This is the capture foundation for
    /// provenance-driven corroboration and multi-hop recall.
    ///
    /// `#[serde(default)]` lets legacy edges (serialized before this field
    /// existed) deserialize to an empty trail. Bounded by
    /// `SHODH_PROVENANCE_MAX_SOURCES` (default 8) at write time.
    #[serde(default)]
    pub provenance: Vec<ProvenanceRecord>,

    /// When this edge last moved UP a tier (L1→L2 or L2→L3).
    ///
    /// The promotion clock. Tier promotion used to be a pure function of
    /// strength, so from the birth weight of 0.4 it took **one** `strengthen`
    /// call to reach L2 and **three** to reach L3 — and three independent
    /// strengthen paths hit the same entity edges within a SINGLE request
    /// (`graph_retrieval::batch_strengthen_synapses`,
    /// `MemorySystem::reinforce_recall_with_momentum`, and
    /// `recall::strengthen_episode_entity_edges`). One noisy conversation could
    /// therefore mint a "near-permanent semantic" edge carrying a 4× retrieval
    /// trust multiplier and a 90-day prune shield.
    ///
    /// Consolidation is defined by *sustained* evidence, not by a burst of
    /// co-activation inside one turn, so promotion now also requires elapsed
    /// time since the previous promotion — see
    /// [`EdgeTier::promotion_min_separation_secs`]. `None` means "never
    /// promoted", in which case the clock is anchored at `created_at`; that
    /// makes legacy edges and edges minted directly into L2 (lineage bridges,
    /// fact edges) behave identically to freshly born ones without a
    /// special case.
    ///
    /// Trailing field: `#[serde(default)]` plus the 4th byte of
    /// [`EDGE_PROVENANCE_DEFAULT_SUFFIX`] let records written before this field
    /// existed decode to `None`.
    #[serde(default)]
    pub promoted_at: Option<DateTime<Utc>>,
}

/// How an edge's relation type was decided for a given attestation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypingMethod {
    Cue,
    Semantic,
    LabelPair,
    Learned,
    CoOccurrence,
    Glirel,
    OpenIe,
    /// Event→event causal link from the CATENA event arm (signal + temporal order).
    Catena,
}

/// One source episode that attested an edge — the unit of provenance/corroboration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    pub source_episode_id: Uuid,
    pub mention_count: u32,
    pub first_observed: DateTime<Utc>,
    pub last_observed: DateTime<Utc>,
    #[serde(default)]
    pub confidence: Option<f32>,
    /// Char span (start,end) into the source episode's content — a REFERENCE, not raw text (keeps edges small).
    #[serde(default)]
    pub evidence_span: Option<(u32, u32)>,
    #[serde(default)]
    pub typed_by: Option<TypingMethod>,
}

/// Default cap on the number of provenance sources retained per edge.
const PROVENANCE_MAX_SOURCES_DEFAULT: usize = 8;

/// Resolve the provenance source cap from the environment.
///
/// `SHODH_PROVENANCE_MAX_SOURCES` overrides the default of 8. A value of 0 or
/// an unparseable value falls back to the default (a cap of 0 would discard all
/// provenance, defeating the feature).
fn provenance_max_sources() -> usize {
    std::env::var("SHODH_PROVENANCE_MAX_SOURCES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(PROVENANCE_MAX_SOURCES_DEFAULT)
}

/// Default minimum number of distinct attesting episodes for corroboration to
/// shield an edge from strength-based pruning.
///
/// **Deliberately equal to [`TIER_PROMOTION_L2_MIN_EPISODES`](crate::constants::TIER_PROMOTION_L2_MIN_EPISODES),
/// and derived from it so the two cannot drift.** The invariant is: *the
/// evidence that earns promotion also earns survival.*
///
/// It has to be, because promotion and pruning read the same evidence but push
/// in opposite directions. L2's prune threshold (0.2) is double L1's (0.1),
/// while the executed decay rate is near-identical across the two tiers over the
/// first three days (L1 λ = 0.029/hour ≈ 0.696/day; L2's hybrid consolidation
/// leg λ = `DECAY_LAMBDA_CONSOLIDATION` = 0.693/day). So promotion, taken alone,
/// makes an edge prunable EARLIER — for a batch-ingested edge at birth weight,
/// ~41.5 idle hours at L2 against ~58.8 at L1. Protecting at a HIGHER episode
/// count than promotion requires would leave exactly the newly-promoted band
/// (2 episodes here) promoted-but-unprotected, i.e. it would consolidate an
/// edge and shorten its life in the same step.
const PROVENANCE_PRUNE_CORROBORATION_MIN_DEFAULT: usize =
    crate::constants::TIER_PROMOTION_L2_MIN_EPISODES;

/// Resolve the provenance-aware-pruning corroboration threshold from the env.
///
/// `SHODH_PROVENANCE_AWARE_PRUNE` gates the feature and is cached (read once per
/// process; eval/production set it before start):
/// - unset → `Some(PROVENANCE_PRUNE_CORROBORATION_MIN_DEFAULT)` (feature ON),
/// - `0` / `false` / empty → `None` (kill switch: only LTP protects, the
///   pre-existing behavior),
/// - `1` / `true` → `Some(PROVENANCE_PRUNE_CORROBORATION_MIN_DEFAULT)`,
/// - an explicit integer `N ≥ 1` → `Some(N)` (lets an A/B sweep the threshold).
///
/// **The unset default flipped from OFF to ON**, because the edge-tier fix made
/// it load-bearing rather than experimental. Promotion is now driven by distinct
/// attesting episodes so that a batch-ingested corpus can consolidate at all;
/// but promotion moves an edge onto a tier whose prune threshold is twice as
/// high at essentially the same decay rate, so consolidating an edge without
/// also honouring its corroboration on the prune side would shorten the life of
/// the very edges the fix is meant to rescue. The two halves are one policy:
/// evidence that promotes must also protect.
///
/// When `Some(min)`, an edge attested by at least `min` distinct episodes is
/// protected from STRENGTH-based pruning only — age-based reaping
/// (`exceeded_max_age`) still applies, so there are no immortal edges, and
/// single-attestation edges are not protected at all and still die on their
/// tier's schedule.
fn provenance_prune_min() -> Option<usize> {
    static FLAG: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| match std::env::var("SHODH_PROVENANCE_AWARE_PRUNE") {
        Ok(v) => {
            let v = v.trim();
            if v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false") {
                None
            } else if v == "1" || v.eq_ignore_ascii_case("true") {
                Some(PROVENANCE_PRUNE_CORROBORATION_MIN_DEFAULT)
            } else {
                v.parse::<usize>().ok().filter(|&n| n > 0)
            }
        }
        Err(_) => Some(PROVENANCE_PRUNE_CORROBORATION_MIN_DEFAULT),
    })
}

/// Pure corroboration decision, separated from the cached env read so it is
/// deterministically unit-testable: an edge with `provenance_len` distinct
/// attesting episodes is protected iff the feature is enabled (`Some(min)`) and
/// the count meets `min`.
fn corroboration_meets(provenance_len: usize, min: Option<usize>) -> bool {
    matches!(min, Some(m) if provenance_len >= m)
}

/// #8 measurement: total CAUSAL edges that collapsed into an existing edge in the
/// REVERSE direction (incoming from/to is the swap of the stored edge) since
/// process start. The order-independent typed pair key merges these silently,
/// keeping the first-observed arrow. A high count means direction is being lost
/// and a direction-sensitive key is justified; near-zero means it is not worth
/// the index migration. Printed by the recall harness (grep
/// "DIRECTED_REVERSE_COLLAPSE=").
static DIRECTED_REVERSE_COLLAPSE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Count of causal reverse-direction collapses since process start (see above).
pub fn directed_reverse_collapse_count() -> u64 {
    DIRECTED_REVERSE_COLLAPSE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Merge a single observation into a provenance trail, deduplicating by episode.
///
/// - If `record.source_episode_id` is already present, its `mention_count` is
///   incremented (saturating), the observation window is widened, and
///   `confidence`/`evidence_span`/`typed_by` are filled in only when the
///   existing value is `None` and the incoming value is `Some` (never overwrite
///   an already-known attribute).
/// - Otherwise the record is appended.
///
/// After merging, the trail is capped to `provenance_max_sources()` entries,
/// retaining the records with the highest `mention_count` (tiebreak: most recent
/// `last_observed`). The just-merged episode is guaranteed to survive the cap.
fn merge_provenance(trail: &mut Vec<ProvenanceRecord>, record: ProvenanceRecord) {
    let merged_episode = record.source_episode_id;

    if let Some(existing) = trail
        .iter_mut()
        .find(|p| p.source_episode_id == record.source_episode_id)
    {
        existing.mention_count = existing.mention_count.saturating_add(record.mention_count);
        if record.last_observed > existing.last_observed {
            existing.last_observed = record.last_observed;
        }
        if record.first_observed < existing.first_observed {
            existing.first_observed = record.first_observed;
        }
        if existing.confidence.is_none() {
            existing.confidence = record.confidence;
        }
        if existing.evidence_span.is_none() {
            existing.evidence_span = record.evidence_span;
        }
        if existing.typed_by.is_none() {
            existing.typed_by = record.typed_by;
        }
    } else {
        trail.push(record);
    }

    let cap = provenance_max_sources();
    if trail.len() > cap {
        // Keep the strongest records (highest mention_count, then most recent),
        // but always keep the episode we just merged in.
        trail.sort_by(|a, b| {
            let a_keep = a.source_episode_id == merged_episode;
            let b_keep = b.source_episode_id == merged_episode;
            b_keep
                .cmp(&a_keep)
                .then_with(|| b.mention_count.cmp(&a.mention_count))
                .then_with(|| b.last_observed.cmp(&a.last_observed))
        });
        trail.truncate(cap);
    }
}

/// Postcard defaults for every trailing `RelationshipEdge` field added after the
/// postcard cutover (#192), in field order: `endpoint_selectivity: Option` (None
/// = `0x00`), `forman_curvature: Option` (None = `0x00`), `provenance: Vec` (empty
/// = varint `0x00`), `promoted_at: Option` (None = `0x00`). `try_decode_compat`
/// appends these one at a time, so a record missing any suffix of these fields
/// decodes (postcard has no `#[serde(default)]` EOF tolerance). Keep in sync with
/// any new trailing field.
const EDGE_PROVENANCE_DEFAULT_SUFFIX: &[u8] = &[0x00, 0x00, 0x00, 0x00];

/// Decode a stored `RelationshipEdge`, tolerating legacy records written before
/// the `provenance` field existed. See [`crate::serialization::try_decode_compat`].
///
/// Returns `(edge, needs_migration)` where `needs_migration = true` means the
/// record was in an older format and would benefit from being rewritten.
///
/// This is the single choke point every runtime read of a `RelationshipEdge`
/// funnels through, so it is also where legacy string-typed relations get
/// normalized to their first-class variant (`RelationType::normalize`) —
/// e.g. a pre-promotion `Custom("Precedes")` edge reads back as `Precedes`.
fn decode_relationship_edge(data: &[u8]) -> Result<(RelationshipEdge, bool)> {
    let (mut edge, needs_migration) = crate::serialization::try_decode_compat::<RelationshipEdge>(
        data,
        EDGE_PROVENANCE_DEFAULT_SUFFIX,
    )?;
    edge.relation_type = edge.relation_type.clone().normalize();
    Ok((edge, needs_migration))
}

/// Postcard defaults for trailing `EntityNode` fields added after the postcard
/// cutover (#192), in declaration order: `selectivity: Option` (None = `0x00`),
/// `fine_type: Option<String>` (None = `0x00`), `kb_id: Option<String>`
/// (None = `0x00`). Keep in sync with any new trailing field — append one
/// `0x00` (or the field's postcard-encoded default) per field, in the order the
/// fields appear in the struct.
const ENTITY_NODE_DEFAULT_SUFFIX: &[u8] = &[0x00, 0x00, 0x00];

/// Whether two graph nodes may be canonicalized into one, given the real-world
/// identities they carry.
///
/// Two *different* Wikidata QIDs are a hard veto: the KB has established that
/// these name different things, and no surface-similarity score should be able
/// to overrule that. Anything else permits the merge — an unlinked node is not
/// evidence of difference, only of silence, which is the common case.
///
/// Split out of `canonicalize_entities` because that function needs a live
/// dependency parser to run at all, and this rule is too important to be
/// reachable only through it.
fn kb_identities_permit_merge(canonical: Option<&str>, member: Option<&str>) -> bool {
    match (canonical, member) {
        (Some(a), Some(b)) => a == b,
        _ => true,
    }
}

/// Decode a stored `EntityNode`, tolerating legacy records written before trailing
/// fields (e.g. `selectivity`, `fine_type`) existed. See [`crate::serialization::try_decode_compat`].
fn decode_entity_node(data: &[u8]) -> Result<(EntityNode, bool)> {
    crate::serialization::try_decode_compat::<EntityNode>(data, ENTITY_NODE_DEFAULT_SUFFIX)
}

fn default_last_activated() -> DateTime<Utc> {
    Utc::now()
}

// Hebbian learning constants now imported from crate::constants:
// - LTP_LEARNING_RATE (0.1): η for strength increase per co-activation
// - LTP_DECAY_HALF_LIFE_DAYS (14.0): λ for time-based decay
// - LTP_THRESHOLD (10): Activations needed for Full LTP
// - LTP_DECAY_FACTOR (0.1): Fully potentiated synapses decay 10x slower
// - LTP_MIN_STRENGTH (0.01): Floor to prevent complete forgetting
// PIPE-4 additions:
// - LTP_BURST_THRESHOLD (5): Activations in 24h for burst LTP
// - LTP_BURST_WINDOW_HOURS (24): Window for burst detection
// - LTP_WEEKLY_THRESHOLD (3): Activations per week for weekly LTP
// - LTP_WEEKLY_MIN_WEEKS (2): Minimum weeks of consistent activation

impl RelationshipEdge {
    /// Strengthen this synapse (Hebbian learning)
    ///
    /// Called when both connected entities are accessed together.
    /// Formula: w_new = w_old + η × (1 - w_old) × co_activation_boost
    ///
    /// PIPE-4: Multi-scale LTP detection
    /// - Records activation timestamps for L2+ edges
    /// - Detects burst patterns (5+ in 24h) → temporary protection
    /// - Detects weekly patterns (3+/week for 2 weeks) → moderate protection
    /// - Detects sustained patterns (10+ total or 5+ over 30 days) → full protection
    ///
    /// Also handles tier promotion (L1→L2→L3) when strength exceeds tier threshold.
    ///
    /// Returns `Some((old_tier_name, new_tier_name))` if a tier promotion occurred,
    /// `None` otherwise. This enables the memory-edge coupling: edge promotions
    /// can signal the memory layer to boost the associated memory's importance.
    pub fn strengthen(&mut self) -> Option<(String, String)> {
        self.strengthen_at(Utc::now())
    }

    /// Strengthen as of an explicit `now`, returning any tier promotion.
    ///
    /// `strengthen()` is the production entry point (`now = Utc::now()`); this
    /// variant takes the clock as a parameter for the same reason
    /// [`decay_at`](Self::decay_at) does — promotion is now time-gated (see
    /// [`EdgeTier::promotion_min_separation_secs`]), so any test that wants to
    /// observe a second promotion would otherwise have to sleep for 30 minutes.
    pub fn strengthen_at(&mut self, now: DateTime<Utc>) -> Option<(String, String)> {
        self.strengthen_scaled_at(now, 1.0)
    }

    /// Importance-gated Hebbian strengthening.
    ///
    /// Identical to `strengthen()` but scales the Hebbian boost by source memory
    /// importance. The importance is mapped to [`STRENGTHEN_IMPORTANCE_FLOOR`, 1.0],
    /// so even low-importance memories still strengthen edges (preventing starvation),
    /// but high-importance memories get disproportionately stronger edges — making
    /// them win during spreading activation traversal.
    ///
    /// Use this instead of `strengthen()` when the source memory importance is known
    /// (e.g., in `reinforce_recall` where recalled memory IDs are available).
    ///
    /// [`STRENGTHEN_IMPORTANCE_FLOOR`]: crate::constants::STRENGTHEN_IMPORTANCE_FLOOR
    pub fn strengthen_with_importance(&mut self, importance: f32) -> Option<(String, String)> {
        self.strengthen_with_importance_at(importance, Utc::now())
    }

    /// [`strengthen_with_importance`](Self::strengthen_with_importance) as of an
    /// explicit `now`. See [`strengthen_at`](Self::strengthen_at) for why the
    /// clock is a parameter.
    pub fn strengthen_with_importance_at(
        &mut self,
        importance: f32,
        now: DateTime<Utc>,
    ) -> Option<(String, String)> {
        use crate::constants::*;
        let scale = STRENGTHEN_IMPORTANCE_FLOOR + importance * (1.0 - STRENGTHEN_IMPORTANCE_FLOOR);
        self.strengthen_scaled_at(now, scale)
    }

    /// The single implementation of Hebbian strengthening.
    ///
    /// `strengthen` and `strengthen_with_importance` were two hand-copied bodies
    /// differing only in this `boost_scale` factor (1.0 vs
    /// `STRENGTHEN_IMPORTANCE_FLOOR + importance·(1 − FLOOR)`, a 5× spread in
    /// arrival rate at identical criteria — deliberate, and preserved). Keeping
    /// two copies of the LTP-detection and tier-promotion blocks was a latent
    /// drift risk on the codebase's most safety-relevant transition; there is now
    /// one copy, which is also the only place the promotion clock has to be
    /// enforced. That matters because three independent call paths strengthen the
    /// same entity edges within one request — a gate at any single call site
    /// would be bypassed by the other two.
    fn strengthen_scaled_at(
        &mut self,
        now: DateTime<Utc>,
        boost_scale: f32,
    ) -> Option<(String, String)> {
        use crate::constants::*;

        self.activation_count += 1;
        self.last_activated = now;

        // PIPE-4: Record activation timestamp for L2+ edges
        self.record_activation_timestamp(now);

        // Hebbian strengthening with tier-specific boost
        let tier_boost = match self.tier {
            EdgeTier::L1Working => TIER_CO_ACCESS_BOOST,
            EdgeTier::L2Episodic => TIER_CO_ACCESS_BOOST * 0.8,
            EdgeTier::L3Semantic => TIER_CO_ACCESS_BOOST * 0.5,
        };
        let boost = (LTP_LEARNING_RATE + tier_boost) * (1.0 - self.strength) * boost_scale;
        self.strength = (self.strength + boost).min(1.0);

        // PIPE-4: Multi-scale LTP detection (only upgrade, never downgrade)
        let new_ltp_status = self.detect_ltp_status(now);
        if new_ltp_status.priority() > self.ltp_status.priority() {
            let old_status = self.ltp_status;
            self.ltp_status = new_ltp_status;

            // LTP bonus: immediate strength boost on upgrade
            let bonus = match new_ltp_status {
                LtpStatus::Burst { .. } => 0.05,
                LtpStatus::Weekly => 0.1,
                LtpStatus::Full => 0.2,
                LtpStatus::None => 0.0,
            };
            self.strength = (self.strength + bonus).min(1.0);

            tracing::debug!(
                "Edge {} LTP upgrade: {:?} → {:?} (activations: {}, age: {} days)",
                self.uuid,
                old_status,
                self.ltp_status,
                self.activation_count,
                (now - self.created_at).num_days()
            );
        }

        // Check for burst expiration and potential downgrade
        if self.ltp_status.is_burst_expired() {
            // Burst expired - check if weekly pattern has emerged
            let weekly_check = self.detect_weekly_pattern();
            if weekly_check {
                self.ltp_status = LtpStatus::Weekly;
            } else {
                self.ltp_status = LtpStatus::None;
            }
        }

        // PIPE-5: L3 auto-LTP removed - now handled by unified ltp_readiness()
        // The readiness formula combines strength + activation count + entity confidence,
        // ensuring both intensity and repetition evidence are required for Full LTP.

        self.try_promote_at(now)
    }

    /// Number of DISTINCT source episodes that have attested this edge.
    ///
    /// This is `provenance.len()` and is only a distinct count because
    /// [`merge_provenance`] is the sole writer of the trail and deduplicates by
    /// `source_episode_id` — a repeat observation from an episode already in the
    /// trail raises that record's `mention_count` instead of appending. Anything
    /// that appends to `provenance` without going through `merge_provenance`
    /// breaks the invariant this method's callers rely on.
    ///
    /// Saturates at `SHODH_PROVENANCE_MAX_SOURCES` (default 8), the trail cap,
    /// and is monotonically non-decreasing below it: `merge_provenance` only
    /// truncates when the length *exceeds* the cap, so an edge cannot lose
    /// corroboration it has already earned and tiers cannot flap.
    pub fn distinct_attesting_episodes(&self) -> usize {
        self.provenance.len()
    }

    /// Re-evaluate this edge's tier without strengthening it.
    ///
    /// [`try_promote_at`](Self::try_promote_at) used to be reachable from
    /// exactly one place — `strengthen_scaled_at` — which made promotion
    /// conditional on being *re-strengthened later*. That is precisely what a
    /// batch-ingested edge, or an edge that receives all of its evidence in one
    /// burst and is then left alone, never gets: its provenance trail can grow
    /// to full corroboration and its strength can sit far above the threshold,
    /// and nothing would ever look. Worse, on the ingest path the strengthen
    /// call happened BEFORE the incoming attestation was merged, so the gate was
    /// always evaluated one episode stale.
    ///
    /// The two callers are the two branches of
    /// [`GraphMemory::add_relationship`], each immediately after the provenance
    /// trail reaches its final state for that write. Promotion is deliberately
    /// NOT driven from the decay/maintenance sweep: that would restore
    /// "strengthen once, wait out the clock, promote" — burst promotion on a
    /// timer — for edges with no corroborating episode at all.
    pub fn reconsider_promotion_at(&mut self, now: DateTime<Utc>) -> Option<(String, String)> {
        self.try_promote_at(now)
    }

    /// The one place an edge changes tier upward.
    ///
    /// Two conditions, both required:
    ///
    /// 1. **Strength** ≥ the current tier's promotion threshold — the original
    ///    criterion, unchanged.
    /// 2. **Independent evidence**, satisfied by EITHER of:
    ///    - at least [`EdgeTier::promotion_min_episodes`] distinct source
    ///      episodes have attested the edge, or
    ///    - at least [`EdgeTier::promotion_min_separation_secs`] has elapsed
    ///      since this edge entered its current tier. The anchor is
    ///      `promoted_at`, falling back to `created_at` for an edge that has
    ///      never been promoted — which covers both legacy records (field
    ///      absent, decodes to `None`) and the two production paths that mint
    ///      directly into L2 (lineage bridges, fact edges) without ever passing
    ///      through here.
    ///
    /// The disjunction is the fix, and the two arms are not redundant. The
    /// episode arm is what a batch-ingested graph can actually reach: a clock is
    /// a property of the ingest schedule, not of the evidence, so a clock-only
    /// gate froze every imported corpus at L1 permanently (see
    /// [`EdgeTier::promotion_min_episodes`]). The clock arm is what edges with
    /// no provenance trail can reach — memory↔memory `CoRetrieved` edges carry
    /// `source_episode_id: None` and so never accumulate episodes; removing the
    /// clock would make those, and every legacy record already on disk,
    /// permanently unpromotable.
    ///
    /// The anti-burst property survives both arms. A burst of strengthen calls
    /// inside one request moves neither: the clock does not advance, and the
    /// episode count cannot move because all three in-request strengthen paths
    /// derive from the SAME observation and `merge_provenance` deduplicates by
    /// episode. Promotion remains monotonic and at most one step per call, so
    /// reaching L3 always takes at least two separate calls.
    ///
    /// Returns `Some((old_tier_name, new_tier_name))` on promotion. This enables
    /// the memory-edge coupling: edge promotions can signal the memory layer to
    /// boost the associated memory's importance.
    fn try_promote_at(&mut self, now: DateTime<Utc>) -> Option<(String, String)> {
        use crate::constants::*;

        let threshold = self.tier.promotion_threshold()?;
        if self.strength < threshold {
            return None;
        }
        let next_tier = self.tier.next_tier()?;

        // Independent-evidence gate: corroboration OR elapsed time.
        // `promoted_at.unwrap_or(created_at)` is the moment this edge entered
        // its current tier.
        let entered_tier_at = self.promoted_at.unwrap_or(self.created_at);
        let separated_by_clock = self
            .tier
            .promotion_min_separation_secs()
            .is_some_and(|min_secs| (now - entered_tier_at).num_seconds() >= min_secs);
        let corroborated = self
            .tier
            .promotion_min_episodes()
            .is_some_and(|min_episodes| self.distinct_attesting_episodes() >= min_episodes);
        if !separated_by_clock && !corroborated {
            return None;
        }

        let old_tier = self.tier;
        self.tier = next_tier;
        self.promoted_at = Some(now);
        // Preserve strength if already above next tier's initial weight
        self.strength = self.strength.max(next_tier.initial_weight());

        // PIPE-4: Initialize activation_timestamps on L1→L2 promotion
        if old_tier == EdgeTier::L1Working {
            self.activation_timestamps =
                Some(VecDeque::with_capacity(ACTIVATION_HISTORY_L2_CAPACITY));
            // Seed with current timestamp
            if let Some(ref mut ts) = self.activation_timestamps {
                ts.push_back(now);
            }
        }

        // Expand capacity on L2→L3 promotion
        if old_tier == EdgeTier::L2Episodic {
            if let Some(ref mut ts) = self.activation_timestamps {
                let current = ts.capacity();
                if current < ACTIVATION_HISTORY_L3_CAPACITY {
                    ts.reserve(ACTIVATION_HISTORY_L3_CAPACITY - current);
                }
            }
        }

        tracing::debug!(
            "Edge {} promoted: {:?} → {:?}",
            self.uuid,
            old_tier,
            self.tier
        );

        Some((format!("{:?}", old_tier), format!("{:?}", self.tier)))
    }

    /// Record an activation timestamp (PIPE-4)
    ///
    /// Only records for L2+ edges. Maintains capacity limits.
    fn record_activation_timestamp(&mut self, timestamp: DateTime<Utc>) {
        use crate::constants::*;

        // L1 edges don't track history (too transient)
        if matches!(self.tier, EdgeTier::L1Working) {
            return;
        }

        // Initialize if needed
        if self.activation_timestamps.is_none() {
            let capacity = match self.tier {
                EdgeTier::L1Working => return,
                EdgeTier::L2Episodic => ACTIVATION_HISTORY_L2_CAPACITY,
                EdgeTier::L3Semantic => ACTIVATION_HISTORY_L3_CAPACITY,
            };
            self.activation_timestamps = Some(VecDeque::with_capacity(capacity));
        }

        if let Some(ref mut timestamps) = self.activation_timestamps {
            let capacity = match self.tier {
                EdgeTier::L1Working => return,
                EdgeTier::L2Episodic => ACTIVATION_HISTORY_L2_CAPACITY,
                EdgeTier::L3Semantic => ACTIVATION_HISTORY_L3_CAPACITY,
            };

            // Maintain capacity limit (ring buffer behavior)
            while timestamps.len() >= capacity {
                timestamps.pop_front();
            }
            timestamps.push_back(timestamp);
        }
    }

    /// Detect LTP status based on unified readiness model (PIPE-4 + PIPE-5)
    ///
    /// PIPE-5 unifies LTP detection into a single readiness score that combines:
    /// - Activation count (repetition path)
    /// - Strength (intensity/durability path)
    /// - Entity confidence (synaptic tagging bonus)
    ///
    /// Multiple paths can lead to Full LTP:
    /// - High repetition alone (15+ activations)
    /// - High intensity alone (0.95+ strength at L3)
    /// - Balanced contribution from both
    /// - High-confidence edges reach threshold ~30% faster
    ///
    /// Temporal patterns (Burst, Weekly) remain separate as they represent
    /// different consolidation mechanisms (E-LTP vs habit formation).
    fn detect_ltp_status(&self, now: DateTime<Utc>) -> LtpStatus {
        use crate::constants::*;

        // PIPE-5: Unified LTP readiness for Full LTP
        // Combines activation count, strength, and entity confidence
        if self.ltp_readiness() >= LTP_READINESS_THRESHOLD {
            return LtpStatus::Full;
        }

        // Legacy time-aware path: 5+ activations over 30+ days
        // Kept for backward compatibility and edges that survived long decay
        let edge_age_days = (now - self.created_at).num_days();
        if edge_age_days >= LTP_TIME_AWARE_DAYS && self.activation_count >= LTP_TIME_AWARE_THRESHOLD
        {
            return LtpStatus::Full;
        }

        // Check for Weekly LTP (requires timestamp history)
        // Temporal pattern: 3+/week for 2+ weeks indicates habit
        if self.detect_weekly_pattern() {
            return LtpStatus::Weekly;
        }

        // Check for Burst LTP (requires timestamp history)
        // Temporal pattern: 5+ in 24h indicates high immediate interest
        if self.detect_burst_pattern(now) {
            return LtpStatus::Burst { detected_at: now };
        }

        LtpStatus::None
    }

    /// Detect burst pattern: 5+ activations in 24 hours (PIPE-4)
    fn detect_burst_pattern(&self, now: DateTime<Utc>) -> bool {
        use crate::constants::*;
        use chrono::Duration;

        let timestamps = match &self.activation_timestamps {
            Some(ts) => ts,
            None => return false,
        };

        let window_start = now - Duration::hours(LTP_BURST_WINDOW_HOURS);
        let count_in_window = timestamps.iter().filter(|&&ts| ts >= window_start).count();

        count_in_window >= LTP_BURST_THRESHOLD as usize
    }

    /// Detect weekly pattern: 3+/week for 2+ weeks (PIPE-4)
    fn detect_weekly_pattern(&self) -> bool {
        use crate::constants::*;
        use chrono::Duration;

        let timestamps = match &self.activation_timestamps {
            Some(ts) => ts,
            None => return false,
        };

        if timestamps.is_empty() {
            return false;
        }

        let now = Utc::now();
        let mut weeks_meeting_threshold = 0u32;

        // Check each of the last LTP_WEEKLY_MIN_WEEKS weeks
        for week_offset in 0..LTP_WEEKLY_MIN_WEEKS {
            let week_end = now - Duration::weeks(week_offset as i64);
            let week_start = week_end - Duration::weeks(1);

            let count_in_week = timestamps
                .iter()
                .filter(|&&ts| ts >= week_start && ts < week_end)
                .count();

            if count_in_week >= LTP_WEEKLY_THRESHOLD as usize {
                weeks_meeting_threshold += 1;
            }
        }

        weeks_meeting_threshold >= LTP_WEEKLY_MIN_WEEKS
    }

    /// Get activation count within a time window (for temporal retrieval scoring)
    pub fn activations_in_window(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> usize {
        match &self.activation_timestamps {
            Some(ts) => ts.iter().filter(|&&t| t >= start && t <= end).count(),
            None => 0,
        }
    }

    /// Apply time-based decay to this synapse
    ///
    /// Uses a tier-aware decay model (3-tier memory consolidation). The three
    /// tiers use TWO different curves, not three:
    /// - **L1 (Working)**: simple exponential, `L1_DECAY_PER_HOUR` λ=0.029/h
    ///   (~2.9%/hour), max 48 hours before prune. This is the only tier that
    ///   still routes through `decay::tier_decay_factor`.
    /// - **L2 (Episodic)**: the Wixted 2004 hybrid — exponential λ=0.693/day
    ///   below `DECAY_CROSSOVER_DAYS` (3), power-law β=0.5 above — pruned at 30
    ///   days. This is the reference curve.
    /// - **L3 (Semantic)**: the SAME hybrid on a time axis scaled by
    ///   `decay::L3_TIME_SCALE_VS_L2` (≈0.0215), i.e. ~46.5× slower, which is the
    ///   ratio `L2_DECAY_PER_DAY` and `L3_DECAY_PER_MONTH` jointly assert.
    ///   Pruned at 90 days.
    ///
    /// Note that L2's *shipped* exponential rate (0.693/day) is far faster than
    /// `L2_DECAY_PER_DAY` (0.031/day) describes; only the L2:L3 ratio is wired,
    /// deliberately, because re-rating L2 in absolute terms would move every
    /// episodic edge in the retrieval substrate and is a measured decision, not a
    /// correctness fix.
    ///
    /// PIPE-4: Multi-scale LTP protection
    /// - Burst: 2x slower decay (temporary, 48h)
    /// - Weekly: 3x slower decay (habit protection)
    /// - Full: 10x slower decay (permanent protection)
    ///
    /// **Important:** Updates `last_activated` to prevent double-decay on
    /// repeated calls.
    ///
    /// Returns true if synapse should be pruned (below tier's threshold)
    /// True when multi-source corroboration should shield this edge from
    /// STRENGTH-based pruning. Off by default; enabled via
    /// `SHODH_PROVENANCE_AWARE_PRUNE`. Age-based reaping is unaffected, so a
    /// corroborated edge is never immortal.
    fn corroboration_protected(&self) -> bool {
        corroboration_meets(self.provenance.len(), provenance_prune_min())
    }

    /// True when this edge is shielded from strength-based pruning by either LTP
    /// potentiation or (when enabled) multi-source corroboration. Used by the
    /// strength-only prune-queue paths, which carry no age dimension.
    pub fn is_prune_protected(&self) -> bool {
        self.ltp_status.is_potentiated() || self.corroboration_protected()
    }

    pub fn decay(&mut self) -> bool {
        self.decay_at(Utc::now())
    }

    /// Apply time-decay as of an explicit `now`, returning whether the edge
    /// should be pruned.
    ///
    /// `decay()` is the production entry point (`now = Utc::now()`); this
    /// variant takes the clock as a parameter so decay is deterministically
    /// testable and so the decay-simulation harness can drive an edge through
    /// many cycles at a controlled cadence. The cadence is load-bearing:
    /// because each call resets `last_activated = now`, the *per-cycle* elapsed
    /// time is what `hybrid_decay_factor` sees. Simulating one large jump would
    /// land directly in the power-law phase and hide the periodic dynamics that
    /// production actually exhibits (every ~6h), so faithful evaluation must
    /// step at the real cadence.
    pub fn decay_at(&mut self, now: DateTime<Utc>) -> bool {
        use crate::decay::tier_decay_factor;

        let elapsed = now.signed_duration_since(self.last_activated);
        let hours_elapsed = elapsed.num_seconds() as f64 / 3600.0;

        if hours_elapsed <= 0.0 {
            return false;
        }

        // Cap max decay to protect against clock jumps (max 1 year = 8760 hours)
        let hours_elapsed = hours_elapsed.min(8760.0);

        // Tier-aware decay with PIPE-4 multi-scale LTP
        let raw_ltp_factor = self.ltp_status.decay_factor();

        // Gate LTP protection by endpoint selectivity (habituation mechanism).
        // Low-selectivity edges (connecting stop-word entities) get reduced
        // LTP protection regardless of activation count, because high-frequency
        // low-information signals should habituate, not potentiate.
        //
        // effective_ltp = raw_ltp * (selectivity / (selectivity + half_sat))
        //
        // selectivity=0.0 → effective_ltp=raw_ltp*0.0 = 1.0 (no protection)
        // selectivity=0.5 → effective_ltp=raw_ltp*0.5 (half protection)
        // selectivity=5.0 → effective_ltp=raw_ltp*0.91 (nearly full protection)
        // selectivity=None → full protection (not yet computed, conservative)
        let ltp_factor = match self.endpoint_selectivity {
            Some(sel) if sel < crate::constants::SELECTIVITY_STOP_WORD_THRESHOLD => {
                // Blend toward 1.0 (no protection) as selectivity → 0
                let gate = sel / (sel + crate::constants::SELECTIVITY_HALF_SAT);
                // ltp_factor is in (0, 1]: 0.1 = Full LTP, 1.0 = no LTP
                // gate → 0 means override toward 1.0 (no protection)
                raw_ltp_factor + (1.0 - raw_ltp_factor) * (1.0 - gate)
            }
            _ => raw_ltp_factor, // Not computed yet or above threshold: full protection
        };

        let (decay_factor, exceeded_max_age) = match self.tier {
            EdgeTier::L1Working => {
                // L1: aggressive exponential — correct for working memory
                tier_decay_factor(hours_elapsed, 0, ltp_factor)
            }
            EdgeTier::L2Episodic | EdgeTier::L3Semantic => {
                // L2/L3: Wixted 2004 hybrid (exponential consolidation → power-law long-term)
                // on a TIER-SCALED time axis. Both tiers used to share one
                // unscaled call — identical decay — which made "L3 is
                // near-permanent" false in the executed path and left
                // L2_DECAY_PER_DAY / L3_DECAY_PER_MONTH with no production
                // reference. L2 is the κ=1 reference (unchanged); L3 ages at the
                // ratio those two constants assert. See `decay::L3_TIME_SCALE_VS_L2`.
                let days = hours_elapsed / 24.0;
                // Burst/Weekly/Full LTP all use the potentiated (slower) power-law.
                // decay_factor() returns exactly 0.5 for Burst, so `<` would exclude it
                // (off-by-one) and decay Burst edges at the non-potentiated rate.
                let is_potentiated = ltp_factor <= 0.5;
                let decay = crate::decay::hybrid_decay_factor_scaled(
                    days,
                    is_potentiated,
                    self.tier.decay_time_scale(),
                );
                let prune_threshold = self.tier.prune_threshold();
                // Min age before pruning: 30 days for L2, 90 days for L3
                let min_prune_hours = if matches!(self.tier, EdgeTier::L3Semantic) {
                    2160.0
                } else {
                    720.0
                };
                let should_prune = decay < prune_threshold && hours_elapsed > min_prune_hours;
                (decay, should_prune)
            }
        };
        self.strength *= decay_factor;

        // Update last_activated to prevent double-decay on repeated calls
        self.last_activated = now;

        // Apply floor to prevent complete forgetting
        let prune_threshold = self.tier.prune_threshold();
        if self.strength < LTP_MIN_STRENGTH {
            self.strength = LTP_MIN_STRENGTH;
        }

        // Downgrade expired burst LTP before prune decision
        // decay_factor() already returns 1.0 for expired bursts (correct rate),
        // but is_potentiated() still returns true — preventing pruning
        if self.ltp_status.is_burst_expired() {
            if self.detect_weekly_pattern() {
                self.ltp_status = LtpStatus::Weekly;
            } else {
                self.ltp_status = LtpStatus::None;
            }
        }

        // Strip LTP protection from near-zero edges (zombie edge cleanup)
        // Prevents immortal edges that retain LTP despite negligible strength
        if self.ltp_status.is_potentiated() && self.strength <= LTP_PRUNE_FLOOR {
            self.ltp_status = LtpStatus::None;
        }

        // Return whether this synapse should be pruned.
        // - LTP potentiation protects from everything (age AND strength).
        // - Age forces a prune next (no immortal edges).
        // - Multi-source corroboration (opt-in) protects from strength-only decay:
        //   a relationship independently attested by several episodes survives the
        //   decay of its activation strength, but not old age.
        // - Otherwise, prune once strength falls to the tier threshold.
        if self.ltp_status.is_potentiated() {
            false
        } else if exceeded_max_age {
            true
        } else if self.corroboration_protected() {
            false
        } else {
            self.strength <= prune_threshold
        }
    }

    /// Construct a synthetic, non-persisted edge for the decay-simulation
    /// harness. `created_at`/`valid_at`/`last_activated` are anchored at
    /// `origin` so a harness can then drive `decay_at` forward from a known
    /// point. Entity UUIDs are random; this is never written to storage.
    pub(crate) fn synthetic_for_sim(
        strength: f32,
        tier: EdgeTier,
        ltp_status: LtpStatus,
        origin: DateTime<Utc>,
    ) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            from_entity: Uuid::new_v4(),
            to_entity: Uuid::new_v4(),
            relation_type: RelationType::RelatedTo,
            strength,
            created_at: origin,
            valid_at: origin,
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: origin,
            activation_count: 0,
            ltp_status,
            activation_timestamps: None,
            tier,
            entity_confidence: None,
            forman_curvature: None,
            endpoint_selectivity: None,
            provenance: Vec::new(),
            promoted_at: None,
        }
    }

    /// Get the effective strength considering recency
    ///
    /// This is a read-only version that calculates what the strength
    /// would be after decay, without modifying the edge.
    /// Uses tier-aware decay (L1/L2/L3 have different decay rates).
    pub fn effective_strength(&self) -> f32 {
        use crate::decay::tier_decay_factor;

        // Use the frozen scoring clock (SHODH_EVAL_NOW) on the eval path: this
        // read-only decay feeds spreading-activation strength, and a live clock
        // makes edges decay measurably between recall repeats minutes apart
        // (L1Working edges decay over hours), wobbling episode activations and
        // flipping near-tie graph-leg ranks. Production leaves SHODH_EVAL_NOW
        // unset, so this is `Utc::now()` there — identical behaviour.
        let now = crate::memory::scoring_now();
        let elapsed = now.signed_duration_since(self.last_activated);
        let hours_elapsed = elapsed.num_seconds() as f64 / 3600.0;

        if hours_elapsed <= 0.0 {
            return self.strength;
        }

        let ltp_factor = self.ltp_status.decay_factor();
        let (decay_factor, _) = match self.tier {
            EdgeTier::L1Working => tier_decay_factor(hours_elapsed, 0, ltp_factor),
            EdgeTier::L2Episodic | EdgeTier::L3Semantic => {
                // Wixted 2004 hybrid: exponential consolidation → power-law long-term,
                // on the same tier-scaled time axis `decay_at` uses. This is the
                // read path (spreading activation), so it MUST agree with the
                // write path — an L3 edge that decays slowly in storage but fast
                // at scoring time would be the worst of both.
                let days = hours_elapsed / 24.0;
                // Inclusive: Burst decay_factor() == 0.5 must take the potentiated path.
                let is_potentiated = ltp_factor <= 0.5;
                (
                    crate::decay::hybrid_decay_factor_scaled(
                        days,
                        is_potentiated,
                        self.tier.decay_time_scale(),
                    ),
                    false,
                )
            }
        };
        (self.strength * decay_factor).max(LTP_MIN_STRENGTH)
    }

    /// Check if this edge has any LTP protection (for backward compatibility)
    pub fn is_potentiated(&self) -> bool {
        self.ltp_status.is_potentiated()
    }

    // =========================================================================
    // PIPE-5: Unified LTP Readiness Model
    // =========================================================================

    /// Get confidence-adjusted LTP threshold (PIPE-5)
    ///
    /// High-confidence edges (strong entity extraction) need fewer activations.
    /// Low-confidence edges need more activations to prove value.
    ///
    /// Returns: threshold in range [LTP_THRESHOLD_MIN, LTP_THRESHOLD_MAX]
    pub fn adjusted_threshold(&self) -> u32 {
        use crate::constants::*;

        let confidence = self.entity_confidence.unwrap_or(0.5);

        // Linear interpolation: high confidence → low threshold
        // confidence 0.0 → threshold_max (13)
        // confidence 1.0 → threshold_min (7)
        let range = LTP_THRESHOLD_MAX - LTP_THRESHOLD_MIN;
        let threshold = LTP_THRESHOLD_MAX as f32 - (confidence * range as f32);
        threshold.round() as u32
    }

    /// Get tier-specific strength floor for Full LTP (PIPE-5)
    ///
    /// L2 edges have lower floor (still proving themselves).
    /// L3 edges have higher floor (must demonstrate durability).
    /// L1 edges return 1.0 (effectively impossible to reach Full LTP).
    pub fn strength_floor(&self) -> f32 {
        use crate::constants::*;

        match self.tier {
            EdgeTier::L1Working => 1.0, // L1 can't reach Full LTP via readiness
            EdgeTier::L2Episodic => LTP_STRENGTH_FLOOR_L2,
            EdgeTier::L3Semantic => LTP_STRENGTH_FLOOR_L3,
        }
    }

    /// Calculate LTP readiness score (PIPE-5)
    ///
    /// Unified formula combining activation count, strength, and entity confidence:
    /// - count_score = activation_count / adjusted_threshold
    /// - strength_score = strength / strength_floor
    /// - tag_bonus = entity_confidence * TAG_WEIGHT
    ///
    /// readiness = count_score * COUNT_WEIGHT + strength_score * STRENGTH_WEIGHT + tag_bonus
    ///
    /// Full LTP when readiness >= 1.0
    ///
    /// This allows multiple paths to LTP:
    /// - Repetition-dominant: 15 activations can compensate for lower strength
    /// - Intensity-dominant: 0.95 strength can compensate for fewer activations
    /// - Balanced: 10 activations + 0.75 strength + moderate confidence
    /// - Tagged boost: high-confidence edges reach LTP ~30% faster
    pub fn ltp_readiness(&self) -> f32 {
        use crate::constants::*;

        // L1 edges can't reach Full LTP via readiness (too transient)
        if matches!(self.tier, EdgeTier::L1Working) {
            return 0.0;
        }

        let threshold = self.adjusted_threshold() as f32;
        let floor = self.strength_floor();

        // Count score: how close to activation threshold
        let count_score = self.activation_count as f32 / threshold;

        // Strength score: how close to strength floor
        let strength_score = self.strength / floor;

        // Tag bonus: entity confidence provides synaptic tagging advantage
        let confidence = self.entity_confidence.unwrap_or(0.5);
        let tag_bonus = confidence * LTP_READINESS_TAG_WEIGHT;

        // Weighted combination
        count_score * LTP_READINESS_COUNT_WEIGHT
            + strength_score * LTP_READINESS_STRENGTH_WEIGHT
            + tag_bonus
    }
}

/// Relationship types for ontological edge classification.
///
/// Extends Graphiti's semantic model with management, operational, and
/// evolution relationships for DevOps, software engineering, and robotics.
///
/// New variants are additive only, and MUST be appended at the end (never
/// inserted between existing variants): this type derives plain
/// `Serialize`/`Deserialize`, so `RelationshipEdge` — persisted via postcard
/// (see `crate::serialization`), not MessagePack — encodes each variant as
/// the varint of its DECLARATION INDEX. Reordering shifts every subsequent
/// variant's on-disk discriminant and mis-decodes existing stored edges.
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

    /// Generic relationships (never penalized by ontological filtering)
    RelatedTo,
    AssociatedWith,

    /// Memory co-retrieval (Hebbian association between memories)
    CoRetrieved,

    /// Sentence co-occurrence (entities appearing in same sentence)
    /// Key for multi-hop: "Melanie" <-> "sunrise" when "Melanie painted a sunrise"
    CoOccurs,

    // =========================================================================
    // Extended ontological relations (added 2026-03)
    // =========================================================================
    /// Management / Governance
    /// Person/Team manages Project/Service/Team
    Manages,
    /// Service/Module depends on Service/Module
    DependsOn,
    /// Task requires Skill/Technology
    Requires,
    /// Configuration configures Service/Environment
    Configures,

    /// Preference / Recommendation
    /// Person prefers Technology/Configuration/Approach
    Prefers,
    /// Person/Role recommends Technology/Approach
    Recommends,

    /// Document / Knowledge
    /// Document documents Service/API/Project
    Documents,
    /// Module/Service implements Concept/Spec
    Implements,

    /// Operational
    /// Pipeline deploys Service to Environment
    DeploysTo,
    /// Metric/Service monitors Service/Pipeline
    Monitors,
    /// Event/Pipeline triggers Pipeline/Task
    Triggers,

    /// Comparison / Evolution
    /// Technology/Config superseded by newer version
    SupersededBy,
    /// Technology is alternative to Technology
    AlternativeTo,

    /// Assignment
    /// Task assigned to Person/Team
    AssignedTo,
    /// Person/Role approves Task/Document
    Approves,

    /// Custom relationship
    Custom(String),

    /// Temporal order (CATENA): head event occurs before tail event. NOT
    /// causation. Declared LAST, after `Custom`, deliberately: postcard (and
    /// serde's default enum tagging in general) encodes a variant as the
    /// varint of its DECLARATION INDEX, not its name — inserting a variant
    /// anywhere but the end would shift every subsequent variant's on-disk
    /// discriminant and mis-decode every existing stored edge (see the
    /// "additive only" contract in this enum's doc comment above).
    Precedes,
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
            Self::CoOccurs => "CoOccurs",
            Self::Manages => "Manages",
            Self::DependsOn => "DependsOn",
            Self::Requires => "Requires",
            Self::Configures => "Configures",
            Self::Prefers => "Prefers",
            Self::Recommends => "Recommends",
            Self::Documents => "Documents",
            Self::Implements => "Implements",
            Self::DeploysTo => "DeploysTo",
            Self::Monitors => "Monitors",
            Self::Triggers => "Triggers",
            Self::SupersededBy => "SupersededBy",
            Self::AlternativeTo => "AlternativeTo",
            Self::AssignedTo => "AssignedTo",
            Self::Approves => "Approves",
            Self::Custom(s) => s.as_str(),
            Self::Precedes => "Precedes",
        }
    }

    /// Normalize legacy string-typed relations to first-class variants. The
    /// CATENA mint site (`mint_causal_spine_edges`) shipped straight to `main`
    /// in 622eb10d (2026-07-10) minting `Custom("Precedes")` — the temporal-
    /// order arm predates this promotion. The demo-server deploy gap that kept
    /// the dependency-parser stack from actually firing in production closed
    /// the next day (2026-07-11); since then, live stores accumulate real
    /// `Custom("Precedes")` edges from CATENA temporal signals. Those legacy
    /// edges are EXPECTED to exist and this normalization makes every runtime
    /// read robust to them regardless of when they were minted.
    ///
    /// Exact case match only (`"Precedes"`, not `eq_ignore_ascii_case`): the
    /// only known minter (the CATENA mint site) always wrote exact title
    /// case, and the lowercase round trip this could otherwise plausibly
    /// guard against (`mif/export.rs` lowercases `Custom(_)` on MIF export) is
    /// closed by this same commit's `mif/import.rs` `"precedes"` arm, which
    /// maps the lowercase string straight to `Self::Precedes` on import —
    /// never back to `Custom("precedes")`. Matching case-insensitively here
    /// would instead silently retype any unrelated user-authored
    /// `Custom("precedes")` (a relation some caller happened to spell that
    /// way), changing its `spreading_weight` from `Custom`'s 1.0 down to
    /// 0.6 — a semantic mutation with no evidence backing it, so it isn't
    /// made on speculation.
    pub fn normalize(self) -> Self {
        match self {
            Self::Custom(ref s) if s == "Precedes" => Self::Precedes,
            other => other,
        }
    }

    /// Intrinsic spreading-activation weight for this relation type — lever-1
    /// prototype, gated at the call site by `SHODH_GRAPH_PREDICATE_WEIGHTS`.
    ///
    /// A co-occurrence edge is weak evidence of a real relationship: the two
    /// entities were merely co-mentioned. A typed predicate (causal, employment,
    /// structural) encodes an actual relation. Spreading activation should flow
    /// preferentially along meaning rather than adjacency, otherwise traversal over
    /// a co-occurrence graph just rediscovers lexical co-occurrence (which BM25
    /// already has). Weights are centred near 1.0 so the flag RE-WEIGHTS the graph
    /// instead of globally scaling it; the load-bearing contrast is the 2.6× gap
    /// between `CoOccurs` (0.5) and the causal relations (1.3).
    pub fn spreading_weight(&self) -> f32 {
        use RelationType::*;
        match self {
            // Causal relations — the lineage / multi-hop backbone.
            Causes | ResultsIn | Triggers | SupersededBy => 1.3,
            // Strong typed relations between distinct entities.
            WorksAt | EmployedBy | Manages | AssignedTo | Approves | OwnedBy | CreatedBy
            | DevelopedBy | Teaches => 1.1,
            // Structural / functional relations.
            PartOf | Contains | LocatedIn | LocatedAt | DependsOn | Requires | Uses
            | Implements | Configures | DeploysTo | Monitors | Documents | WorksWith | Knows
            | Learned | Prefers | Recommends => 1.0,
            AlternativeTo => 0.9,
            // Generic associations — progressively weaker evidence of meaning.
            AssociatedWith | CoRetrieved => 0.7,
            // Temporal order (CATENA `Precedes`) is weaker evidence than causation
            // but stronger than bare co-occurrence — same weight as `RelatedTo`
            // (spec decision, not measured).
            RelatedTo | Precedes => 0.6,
            CoOccurs => 0.5,
            Custom(_) => 1.0,
        }
    }

    /// Whether this relation encodes forward causation (cause `from` → effect
    /// `to`). Used by backward causal-origin tracing: to find the origin of an
    /// effect, walk these edges from the effect (`to`) toward the cause (`from`).
    pub fn is_causal(&self) -> bool {
        matches!(
            self,
            RelationType::Causes | RelationType::Triggers | RelationType::ResultsIn
        )
    }
}

/// SHODH_PERSON_PERSON_KNOWS=1 — type Person↔Person co-mentions as `Knows`
/// instead of `CoOccurs`. Cached: read once per process (eval sets env before
/// start; production restarts to change it).
fn person_person_knows() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("SHODH_PERSON_PERSON_KNOWS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Match relational cue phrases in ALREADY-LOWERCASED text → a typed predicate.
/// Shared by the whole-text and span-scoped extractors below.
fn predicate_from_cues(t: &str) -> Option<(RelationType, &'static str)> {
    use RelationType::*;
    // Return the FIRST cue needle present in `t` (needle order = signal strength),
    // so the caller learns both the predicate and the exact lexeme that fired.
    let first = |needles: &[&'static str]| needles.iter().copied().find(|n| t.contains(n));

    // Ordered by signal strength; first match wins. Cue fragments are chosen to
    // survive an entity splitting the verb phrase ("X set Y in motion") — real text
    // rarely keeps a relational verb contiguous, so matching only "set in motion"
    // would miss the relation it is meant to capture.
    if let Some(n) = first(&[
        "in motion",
        "brought about",
        "gave rise",
        "triggered",
        "led directly to",
        "led to",
        "resulted in",
        "caused",
        "because of",
        "due to",
    ]) {
        return Some((Triggers, n));
    }
    if let Some(n) = first(&[
        "superseded",
        "replaced by",
        "deprecated",
        "obsoleted",
        "rolled back",
    ]) {
        return Some((SupersededBy, n));
    }
    if let Some(n) = first(&[
        "manages",
        "manager of",
        "oversees",
        "supervises",
        "in charge of",
    ]) {
        return Some((Manages, n));
    }
    if let Some(n) = first(&[
        "works at",
        "works for",
        "employed by",
        "employee of",
        "joined",
    ]) {
        return Some((WorksAt, n));
    }
    if let Some(n) = first(&[
        "created",
        "developed",
        "built",
        "founded",
        "designed",
        "authored",
    ]) {
        return Some((CreatedBy, n));
    }
    if let Some(n) = first(&["depends on", "relies on", "requires", "needs"]) {
        return Some((DependsOn, n));
    }
    if let Some(n) = first(&["located in", "based in", "headquartered", "situated in"]) {
        return Some((LocatedIn, n));
    }
    if let Some(n) = first(&["part of", "belongs to", "member of", "division of"]) {
        return Some((PartOf, n));
    }
    if let Some(n) = first(&["uses", "using", "powered by", "built on"]) {
        return Some((Uses, n));
    }
    None
}

/// Whole-text predicate recovery (lever-1, gated at the call site by
/// `SHODH_GRAPH_EXTRACTED_PREDICATES`). Undirected and clause-blind — kept for the
/// simple single-relation path and unit tests. Prefer `extract_directed_predicate`
/// at edge-creation time.
pub fn extract_predicate_from_text(text: &str) -> Option<RelationType> {
    predicate_from_cues(&text.to_ascii_lowercase()).map(|(rt, _)| rt)
}

/// Span-scoped, DIRECTION-aware predicate recovery. Locates both entity mentions,
/// scans only the sentence containing BOTH for a cue, and decides direction by
/// surface order (the earlier mention is the subject/cause → `from_entity`),
/// FLIPPED for effect-first constructions (see below).
/// Returns `(relation, a_is_source)`; `None` when the mentions don't co-occur in
/// one sentence or no cue is present, so the caller keeps the label-pair inference.
///
/// This fixes the two bugs that left lineage at P@1=0: a multi-relation sentence no
/// longer types every pair the same (sentence-scoped), and the causal arrow now
/// points cause→effect by surface order instead of trusting NER's entity ordering
/// (the likely cause of the failed first measurement — a reversed arrow makes the
/// backward origin-walk dead-end).
///
/// DIRECTION FIX (substrate audit 2026-06-10): surface order is only correct for
/// cause-first constructions ("B triggered A"). Effect-first constructions invert
/// it — "A happened BECAUSE OF B" / "A DUE TO B" / passive "A WAS CAUSED BY B" all
/// put the EFFECT first, so the earlier mention is NOT the cause. Three of the ten
/// causal cues systematically wrote reversed arrows, dead-ending the backward
/// origin-walk for exactly those sentences. When an effect-first cue is present in
/// the sentence, the arrow flips.
pub fn extract_directed_predicate(
    text: &str,
    name_a: &str,
    name_b: &str,
) -> Option<(RelationType, bool)> {
    let lc = text.to_ascii_lowercase();
    let a = name_a.to_ascii_lowercase();
    let b = name_b.to_ascii_lowercase();
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let pa = lc.find(&a)?;
    let pb = lc.find(&b)?;
    if pa == pb {
        return None;
    }
    // Window spanning both mentions, then clamp to the enclosing sentence so a cue
    // from a neighbouring clause cannot leak in.
    let (lo, hi) = if pa < pb {
        (pa, pb + b.len())
    } else {
        (pb, pa + a.len())
    };
    let sent_start = lc[..lo]
        .rfind(['.', '!', '?', ';', '\n'])
        .map(|i| i + 1)
        .unwrap_or(0);
    let sent_end = lc[hi..]
        .find(['.', '!', '?', ';', '\n'])
        .map(|i| hi + i)
        .unwrap_or(lc.len());
    let sentence = &lc[sent_start..sent_end];
    let (rt, cue) = predicate_from_cues(sentence)?;

    // PREDICATE-FRAGMENT GATE. A model-free fallback NER (no GLiNER assets) mints
    // the cue's OWN words as entities — "motion" from "in motion", "brought" from
    // "brought about". Such a mention sits INSIDE the predicate span, so it is the
    // relation lexeme, not a causal argument. Left unchecked it becomes a shared
    // causal endpoint across every sentence that uses the same cue, welding
    // unrelated chains into cross-document causal bridges (the fallback-path
    // lineage flood; the GLiNER typer never emits these predicate spans, so this
    // gate is a no-op on the model path). An endpoint whose mention overlaps the
    // fired cue's span is disqualified.
    if let Some(cue_off) = sentence.find(cue) {
        let cue_lo = sent_start + cue_off;
        let cue_hi = cue_lo + cue.len();
        let overlaps_cue = |lo: usize, len: usize| lo < cue_hi && cue_lo < lo + len;
        if overlaps_cue(pa, a.len()) || overlaps_cue(pb, b.len()) {
            return None;
        }
    }

    // Effect-first constructions: the earlier mention is the EFFECT, not the cause.
    const EFFECT_FIRST_CUES: [&str; 4] = ["because of", "due to", "caused by", "triggered by"];
    let effect_first = EFFECT_FIRST_CUES.iter().any(|c| sentence.contains(c));
    let a_first = pa < pb;
    Some((rt, if effect_first { !a_first } else { a_first }))
}

/// Infer a typed relation between two entities based on their labels.
///
/// Uses ontological rules to assign semantically meaningful edge types
/// instead of the default `RelatedTo`. Falls back to `CoOccurs` for
/// co-mentioned entity pairs with no specific ontological relationship.
///
/// `add_relationship()` deduplicates by (from, to, relation_type) — same
/// typed pair strengthens the existing edge rather than creating a duplicate.
pub fn infer_relation_type_for_pair(from: &EntityLabel, to: &EntityLabel) -> RelationType {
    use EntityLabel::*;
    use RelationType::*;

    match (from, to) {
        // Person↔Person — the dominant pair in conversational/personal memory —
        // previously had NO rule and fell through to CoOccurs (the mechanical
        // cause of the >80%-untyped graph on conversational corpora, substrate
        // audit 2026-06-10). Typed as Knows behind SHODH_PERSON_PERSON_KNOWS
        // because the change is NOT free: non-generic types become eligible for
        // ontological relation penalties and predicate weighting, so it must be
        // A/B'd, not assumed.
        (Person, Person) if person_person_knows() => Knows,

        // Person relationships
        (Person, Organization) | (Person, Team) => WorksAt,
        (Person, Technology) | (Person, Service) | (Person, Database) => Uses,
        (Person, Skill) => Learned,

        // Task relationships
        (Task, Person) | (Task, Team) => AssignedTo,
        (Task, Technology) | (Task, Service) => Requires,

        // Service/Module dependencies
        (Service, Database) => Uses,
        (Module, Module) | (Service, Service) | (Module, Service) | (Service, Module) => DependsOn,

        // Pipeline / deployment
        (Pipeline, Service) | (Pipeline, Environment) => DeploysTo,

        // Monitoring
        (Metric, Service) | (Metric, Pipeline) => Monitors,

        // Configuration
        (Configuration, Service) | (Configuration, Environment) => Configures,

        // Documentation
        (Document, Service) | (Document, Module) | (Document, Pipeline) | (Document, Project) => {
            Documents
        }

        // Part-of / containment
        (Task, Project)
        | (Document, Repository)
        | (Repository, Project)
        | (Module, Project)
        | (Service, Project) => PartOf,

        // Two technologies co-mentioned were previously typed AlternativeTo —
        // fabricated semantics from labels alone (co-mention says nothing about
        // substitutability), and as a "confident" type it suppressed the cue
        // extractor, starving the causal walk (the lineage-zero root cause,
        // 2026-06-10). Co-mention is co-occurrence; real relations come from
        // sentence evidence (semantic typer / cues).
        (Technology, Technology) => CoOccurs,

        // Location relationships (either direction)
        (_, Location) | (Location, _) => LocatedIn,

        // Default: co-occurrence (neutral bridge weight in spreading activation)
        _ => CoOccurs,
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

/// Minimum dormancy (days since last activation) for an edge re-attestation
/// to count as a [`TemporalAnomalyKind::DormantReactivation`] event.
const DORMANT_REACTIVATION_MIN_DAYS: f32 = 7.0;

/// Backstop cap on the pending temporal-anomaly queue (drained by the ingest
/// path; the cap only matters if nothing drains it).
const TEMPORAL_EVENT_QUEUE_CAP: usize = 1024;

/// What kind of temporal anomaly the LTP/strengthen machinery surfaced.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum TemporalAnomalyKind {
    /// An edge dormant for at least [`DORMANT_REACTIVATION_MIN_DAYS`] was
    /// re-attested — a stale pattern suddenly became live again.
    DormantReactivation,
}

/// A temporal anomaly detected at edge-strengthen time — SURFACED from the
/// dynamics the Hebbian/decay machinery already runs, not re-detected.
/// Queued on the graph (mirroring `pending_prune`) and drained by the ingest
/// path, which resolves entity names and emits the event on SSE.
#[derive(Debug, Clone, Serialize)]
pub struct TemporalAnomalyEvent {
    pub kind: TemporalAnomalyKind,
    pub edge_uuid: Uuid,
    pub from_entity: Uuid,
    pub to_entity: Uuid,
    pub relation_type: RelationType,
    /// Days the edge sat dormant before this reactivation.
    pub gap_days: f32,
    pub detected_at: DateTime<Utc>,
}

/// Graph memory storage and operations
///
/// Uses a single RocksDB instance with 9 column families for all graph data.
/// This reduces file descriptor usage from 9 separate DBs to 1 (sharing WAL, MANIFEST, LOCK).
pub struct GraphMemory {
    /// Unified RocksDB database with column families for entities, relationships,
    /// episodes, and all index tables
    db: Arc<DB>,

    /// In-memory entity name index for fast lookups (loaded from name_index CF)
    entity_name_index: Arc<parking_lot::RwLock<HashMap<String, Uuid>>>,

    /// In-memory lowercase name index for O(1) case-insensitive lookups
    entity_lowercase_index: Arc<parking_lot::RwLock<HashMap<String, Uuid>>>,

    /// In-memory stemmed name index for O(1) linguistic lookups
    /// Key: Porter-stemmed lowercase name, Value: Entity UUID
    entity_stemmed_index: Arc<parking_lot::RwLock<HashMap<String, Uuid>>>,

    /// In-memory alias index (loaded from the alias CF): a surface form
    /// (lowercased) -> the canonical entity UUID it resolves to. Curated by the
    /// entity resolver; the highest-priority tier in `find_entity_by_name`.
    entity_alias_index: Arc<parking_lot::RwLock<HashMap<String, Uuid>>>,

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

    /// In-memory cache of entity name embeddings for concept merging.
    /// Maps entity UUID → embedding vector. Loaded on startup, updated on add.
    /// Used when string-based dedup (exact/case/stemmed) fails — catches synonyms
    /// like "authentication" ↔ "auth" via cosine similarity.
    #[allow(clippy::type_complexity)]
    entity_embedding_cache: Arc<parking_lot::RwLock<Vec<(Uuid, Vec<f32>)>>>,

    /// Edges found below prune threshold during lazy-decay reads.
    /// Flushed as batch deletes on each maintenance cycle (no full scan needed).
    pending_prune: parking_lot::Mutex<Vec<Uuid>>,

    /// Temporal anomalies surfaced at strengthen time (dormant reactivation),
    /// drained by the ingest path and emitted on SSE. Capped backstop.
    pending_temporal_anomalies: parking_lot::Mutex<Vec<TemporalAnomalyEvent>>,

    /// Entities that may have become orphaned from pruned edges.
    /// Checked during flush_pending_maintenance().
    pending_orphan_checks: parking_lot::Mutex<Vec<Uuid>>,

    /// Topology-aware decay (W1-B): smoothed per-node structural protection in
    /// `[0, 1]`, carried across heavy cycles for hysteresis. Recomputed (not
    /// persisted) each heavy cycle from the current edge set when
    /// `SHODH_TOPOLOGY_AWARE_DECAY` is on — persisting it would go stale the
    /// instant any edge is added or pruned, and would cost a full entities-CF
    /// rewrite per cycle (like the curvature pass) for a value consumed only
    /// within the same cycle's prune decision. In-memory means a process restart
    /// re-derives protection from current structure (conservative: it never
    /// over-protects, it just loses the "was recently a bridge" tail). Empty and
    /// untouched when the flag is off.
    topology_protection: parking_lot::RwLock<HashMap<Uuid, f32>>,
}

impl GraphMemory {
    /// Get a reference to the underlying RocksDB instance (for backup/checkpoint).
    pub fn get_db(&self) -> &DB {
        &self.db
    }

    // Column family accessors — cheap HashMap lookups on DB internals
    fn entities_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_ENTITIES)
            .expect("entities CF must exist")
    }
    fn relationships_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_RELATIONSHIPS)
            .expect("relationships CF must exist")
    }
    fn episodes_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_EPISODES)
            .expect("episodes CF must exist")
    }
    fn entity_edges_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_ENTITY_EDGES)
            .expect("entity_edges CF must exist")
    }
    fn entity_pair_index_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_ENTITY_PAIR_INDEX)
            .expect("entity_pair_index CF must exist")
    }
    fn entity_episodes_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_ENTITY_EPISODES)
            .expect("entity_episodes CF must exist")
    }
    fn name_index_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_NAME_INDEX)
            .expect("name_index CF must exist")
    }
    fn lowercase_index_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_LOWERCASE_INDEX)
            .expect("lowercase_index CF must exist")
    }
    fn stemmed_index_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_STEMMED_INDEX)
            .expect("stemmed_index CF must exist")
    }
    fn alias_cf(&self) -> &ColumnFamily {
        self.db.cf_handle(CF_ALIAS).expect("alias CF must exist")
    }

    /// Create a new graph memory system.
    ///
    /// If `shared_cache` is provided, block-cache reads are charged against the
    /// shared LRU cache (recommended for multi-tenant server mode). When `None`,
    /// a small per-instance cache is created (standalone / test use).
    pub fn new(path: &Path, shared_cache: Option<&rocksdb::Cache>) -> Result<Self> {
        use crate::constants::ROCKSDB_GRAPH_WRITE_BUFFER_BYTES;

        let graph_path = path.join("graph");
        std::fs::create_dir_all(&graph_path)?;

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(ROCKSDB_GRAPH_WRITE_BUFFER_BYTES);
        opts.set_max_write_buffer_number(2);

        // Shared block cache for multi-tenant, small local for standalone/tests.
        use rocksdb::{BlockBasedOptions, Cache};
        let mut block_opts = BlockBasedOptions::default();
        let local_cache;
        let cache = match shared_cache {
            Some(c) => c,
            None => {
                local_cache = Cache::new_lru_cache(8 * 1024 * 1024); // 8MB standalone
                &local_cache
            }
        };
        block_opts.set_block_cache(cache);
        block_opts.set_cache_index_and_filter_blocks(true);
        opts.set_block_based_table_factory(&block_opts);

        // Build column family descriptors — all CFs share the same options
        let cf_descriptors: Vec<ColumnFamilyDescriptor> = GRAPH_CF_NAMES
            .iter()
            .map(|name| ColumnFamilyDescriptor::new(*name, opts.clone()))
            .collect();

        let db = Arc::new(DB::open_cf_descriptors(&opts, &graph_path, cf_descriptors)?);

        // Migrate data from old separate-DB layout if needed
        let migrated = Self::migrate_from_separate_dbs(path, &db)?;
        if migrated > 0 {
            tracing::info!(
                "Migrated {} entries from separate graph DBs to column families",
                migrated
            );
        }

        // Load entity name index from name_index CF (O(n) but faster than deserializing entities)
        // If empty, migrate from entities CF (one-time migration for existing data)
        let entity_name_index = Self::load_or_migrate_name_index(&db)?;

        // Load/migrate lowercase index for O(1) case-insensitive lookup
        let entity_lowercase_index =
            Self::load_or_migrate_lowercase_index(&db, &entity_name_index)?;

        // Load/migrate stemmed index for O(1) linguistic lookup
        let entity_stemmed_index = Self::load_or_migrate_stemmed_index(&db, &entity_name_index)?;

        // Load the alias index (surface -> canonical UUID). New CF: empty for
        // pre-existing DBs, no migration — it is populated only by the resolver.
        let entity_alias_index = Self::load_alias_index(&db)?;

        let entity_count = entity_name_index.len();

        // Count relationships and episodes during startup (one-time cost)
        // This is O(n) at startup, but get_stats() will be O(1) at runtime
        let relationships_cf = db
            .cf_handle(CF_RELATIONSHIPS)
            .ok_or_else(|| anyhow::anyhow!("CF '{}' not found after DB open", CF_RELATIONSHIPS))?;
        let episodes_cf = db
            .cf_handle(CF_EPISODES)
            .ok_or_else(|| anyhow::anyhow!("CF '{}' not found after DB open", CF_EPISODES))?;
        let relationship_count = Self::count_relationship_edges(&db, relationships_cf);
        let episode_count = Self::count_cf_entries(&db, episodes_cf);

        // Load entity embedding cache for concept merging
        // Only entities with pre-computed name_embeddings are cached
        let entities_cf = db
            .cf_handle(CF_ENTITIES)
            .ok_or_else(|| anyhow::anyhow!("CF '{}' not found after DB open", CF_ENTITIES))?;
        let entity_embedding_cache =
            Self::load_entity_embedding_cache(&db, entities_cf, &entity_name_index);
        let embedding_cache_size = entity_embedding_cache.len();

        let graph = Self {
            db,
            entity_name_index: Arc::new(parking_lot::RwLock::new(entity_name_index)),
            entity_lowercase_index: Arc::new(parking_lot::RwLock::new(entity_lowercase_index)),
            entity_stemmed_index: Arc::new(parking_lot::RwLock::new(entity_stemmed_index)),
            entity_alias_index: Arc::new(parking_lot::RwLock::new(entity_alias_index)),
            entity_count: Arc::new(AtomicUsize::new(entity_count)),
            relationship_count: Arc::new(AtomicUsize::new(relationship_count)),
            episode_count: Arc::new(AtomicUsize::new(episode_count)),
            synapse_update_lock: Arc::new(parking_lot::Mutex::new(())),
            entity_embedding_cache: Arc::new(parking_lot::RwLock::new(entity_embedding_cache)),
            pending_prune: parking_lot::Mutex::new(Vec::new()),
            pending_temporal_anomalies: parking_lot::Mutex::new(Vec::new()),
            pending_orphan_checks: parking_lot::Mutex::new(Vec::new()),
            topology_protection: parking_lot::RwLock::new(HashMap::new()),
        };

        if entity_count > 0 || relationship_count > 0 || episode_count > 0 {
            tracing::info!(
                "Loaded graph with {} entities ({} with embeddings), {} relationships, {} episodes",
                entity_count,
                embedding_cache_size,
                relationship_count,
                episode_count
            );
        }

        Ok(graph)
    }

    /// Migrate data from the old separate-DB layout (pre-CF) into column families.
    ///
    /// Detects old `graph_*` subdirectories, opens them read-only, copies all KV
    /// pairs into the corresponding CF, then renames the old directory for rollback safety.
    fn migrate_from_separate_dbs(base_path: &Path, db: &DB) -> Result<usize> {
        let old_dirs: &[(&str, &str)] = &[
            ("graph_entities", CF_ENTITIES),
            ("graph_relationships", CF_RELATIONSHIPS),
            ("graph_episodes", CF_EPISODES),
            ("graph_entity_edges", CF_ENTITY_EDGES),
            ("graph_entity_pair_index", CF_ENTITY_PAIR_INDEX),
            ("graph_entity_episodes", CF_ENTITY_EPISODES),
            ("graph_entity_name_index", CF_NAME_INDEX),
            ("graph_entity_lowercase_index", CF_LOWERCASE_INDEX),
            ("graph_entity_stemmed_index", CF_STEMMED_INDEX),
        ];

        let mut total_migrated = 0usize;

        for (old_name, cf_name) in old_dirs {
            let old_path = base_path.join(old_name);
            if !old_path.exists() {
                continue;
            }

            let cf = db
                .cf_handle(cf_name)
                .ok_or_else(|| anyhow::anyhow!("CF '{}' not found during migration", cf_name))?;

            // Only migrate if the CF is empty (avoid double migration)
            if db
                .iterator_cf(cf, rocksdb::IteratorMode::Start)
                .next()
                .is_some()
            {
                // CF already has data — just rename the old dir
                let renamed = base_path.join(format!("{}.pre_cf_migration", old_name));
                if !renamed.exists() {
                    let _ = std::fs::rename(&old_path, &renamed);
                }
                continue;
            }

            // Open old DB read-only and copy all entries
            let old_opts = Options::default();
            match DB::open_for_read_only(&old_opts, &old_path, false) {
                Ok(old_db) => {
                    let mut batch = WriteBatch::default();
                    let mut count = 0usize;

                    for item in old_db.iterator(rocksdb::IteratorMode::Start) {
                        match item {
                            Ok((key, value)) => {
                                batch.put_cf(cf, &key, &value);
                                count += 1;
                                // Flush in chunks to limit memory usage
                                if count.is_multiple_of(10_000) {
                                    db.write(std::mem::take(&mut batch))?;
                                    batch = WriteBatch::default();
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Error reading from old {}: {}", old_name, e);
                                break;
                            }
                        }
                    }

                    if count > 0 {
                        db.write(batch)?;
                    }

                    drop(old_db);

                    // Rename old directory for rollback safety
                    let renamed = base_path.join(format!("{}.pre_cf_migration", old_name));
                    if let Err(e) = std::fs::rename(&old_path, &renamed) {
                        tracing::warn!(
                            "Migrated {} entries from {} but failed to rename: {}",
                            count,
                            old_name,
                            e
                        );
                    } else {
                        tracing::info!(
                            "Migrated {} entries from {} to CF '{}'",
                            count,
                            old_name,
                            cf_name
                        );
                    }

                    total_migrated += count;
                }
                Err(e) => {
                    tracing::warn!("Failed to open old DB {} for migration: {}", old_name, e);
                }
            }
        }

        Ok(total_migrated)
    }

    /// Load entity name->UUID index from name_index CF, or migrate from entities CF if empty
    fn load_or_migrate_name_index(db: &DB) -> Result<HashMap<String, Uuid>> {
        let name_index_cf = db
            .cf_handle(CF_NAME_INDEX)
            .ok_or_else(|| anyhow::anyhow!("CF '{}' not found", CF_NAME_INDEX))?;
        let entities_cf = db
            .cf_handle(CF_ENTITIES)
            .ok_or_else(|| anyhow::anyhow!("CF '{}' not found", CF_ENTITIES))?;
        let mut index = HashMap::new();

        // Try to load from name_index CF first
        let iter = db.iterator_cf(name_index_cf, rocksdb::IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if let (Ok(name), Ok(uuid_bytes)) = (
                std::str::from_utf8(&key),
                <[u8; 16]>::try_from(value.as_ref()),
            ) {
                index.insert(name.to_string(), Uuid::from_bytes(uuid_bytes));
            }
        }

        // If name_index CF is empty but entities exist, migrate (one-time operation)
        if index.is_empty() {
            let entity_iter = db.iterator_cf(entities_cf, rocksdb::IteratorMode::Start);
            let mut migrated_count = 0;
            for (_, value) in entity_iter.flatten() {
                if let Ok((entity, _)) = decode_entity_node(&value) {
                    // Store in name_index CF: name -> UUID bytes
                    db.put_cf(
                        name_index_cf,
                        entity.name.as_bytes(),
                        entity.uuid.as_bytes(),
                    )?;
                    index.insert(entity.name.clone(), entity.uuid);
                    migrated_count += 1;
                }
            }
            if migrated_count > 0 {
                tracing::info!("Migrated {} entities to name index CF", migrated_count);
            }
        }

        Ok(index)
    }

    /// Load the alias index (surface -> canonical UUID) from the alias CF. Unlike
    /// the name indexes there is no migration: the CF is empty until the resolver
    /// seeds it, and a surface with no alias simply falls through to name lookup.
    fn load_alias_index(db: &DB) -> Result<HashMap<String, Uuid>> {
        let alias_cf = db
            .cf_handle(CF_ALIAS)
            .ok_or_else(|| anyhow::anyhow!("CF '{}' not found", CF_ALIAS))?;
        let mut index = HashMap::new();
        let iter = db.iterator_cf(alias_cf, rocksdb::IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if let (Ok(surface), Ok(uuid_bytes)) = (
                std::str::from_utf8(&key),
                <[u8; 16]>::try_from(value.as_ref()),
            ) {
                index.insert(surface.to_string(), Uuid::from_bytes(uuid_bytes));
            }
        }
        Ok(index)
    }

    /// Register that `surface` is an alias of the canonical entity `canonical`.
    /// Idempotent; the surface is trimmed and lowercased so resolution is
    /// case-insensitive. Persists to the alias CF and the in-memory index.
    pub fn put_alias(&self, surface: &str, canonical: Uuid) -> Result<()> {
        let key = surface.trim().to_lowercase();
        if key.is_empty() {
            return Ok(());
        }
        self.db
            .put_cf(self.alias_cf(), key.as_bytes(), canonical.as_bytes())?;
        self.entity_alias_index.write().insert(key, canonical);
        Ok(())
    }

    /// Resolve a surface form to its canonical entity UUID, if an alias is
    /// registered. O(1) in-memory lookup.
    pub fn resolve_alias(&self, surface: &str) -> Option<Uuid> {
        let key = surface.trim().to_lowercase();
        self.entity_alias_index.read().get(&key).copied()
    }

    /// Number of registered aliases.
    pub fn alias_count(&self) -> usize {
        self.entity_alias_index.read().len()
    }

    /// Seed the alias table from a batch of `(surface, canonical_uuid)` pairs —
    /// the output of an entity-resolution pass. Written atomically; the in-memory
    /// index is updated to match. Returns the number of aliases written.
    pub fn seed_aliases<I>(&self, pairs: I) -> Result<usize>
    where
        I: IntoIterator<Item = (String, Uuid)>,
    {
        let mut batch = rocksdb::WriteBatch::default();
        let mut staged: Vec<(String, Uuid)> = Vec::new();
        for (surface, canonical) in pairs {
            let key = surface.trim().to_lowercase();
            if key.is_empty() {
                continue;
            }
            batch.put_cf(self.alias_cf(), key.as_bytes(), canonical.as_bytes());
            staged.push((key, canonical));
        }
        self.db.write(batch)?;
        let written = staged.len();
        let mut index = self.entity_alias_index.write();
        index.extend(staged);
        Ok(written)
    }

    /// Extract appositive / definite-description aliases from the episode text and
    /// seed them (ER Plan Task 3.1 — the LLM-free "free-label engine"). "Apple, the
    /// iPhone maker" → alias `iphone maker` → Apple's canonical node, so a later
    /// bare "iPhone maker" mention resolves to Apple (Tier-0 `resolve_alias`)
    /// instead of forming a duplicate. Model-free (spaCy-rusty appositive parse);
    /// no-ops without the parser. Returns the number of aliases seeded.
    pub fn mint_appositive_aliases(&self, content: &str) -> usize {
        let Some(pairs) = crate::appositive::extract_from_text(content) else {
            return 0;
        };
        if pairs.is_empty() {
            return 0;
        }
        let mut to_seed: Vec<(String, Uuid)> = Vec::new();
        for p in pairs {
            // Idempotent: skip if this surface already resolves. The anchor must be
            // a real entity in this graph — the appositive phrase becomes its alias.
            if self.resolve_alias(&p.alias).is_some() {
                continue;
            }
            if let Ok(Some(anchor)) = self.find_entity_by_name(&p.canonical) {
                to_seed.push((p.alias, anchor.uuid));
            }
        }
        if to_seed.is_empty() {
            return 0;
        }
        self.seed_aliases(to_seed).unwrap_or(0)
    }

    /// Collapse corpus mentions onto their KB-canonical node — GATED behind
    /// `SHODH_KB_LINKING=1`.
    ///
    /// Distinct from the always-on `kb_id` stamp in `add_entity`. Stamping only
    /// *records* an identity and cannot move retrieval; this merges graph
    /// structure by seeding a surface→canonical alias, which changes what
    /// traversal and dedup see. That is a real behavioural change with a real
    /// blast radius, so it stays opt-in until it has been measured against the
    /// recall gate.
    ///
    /// Only unambiguous links act (`crate::kb::Resolution::Linked`); an abstain
    /// seeds nothing. Returns the number of aliases seeded.
    pub fn kb_link_entities(&self, entity_uuids: &[(String, Uuid, EntityLabel)]) -> usize {
        let on = std::env::var("SHODH_KB_LINKING")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !on {
            return 0;
        }
        let kb = crate::kb::global();
        let mut seeded = 0usize;
        for (name, uuid, label) in entity_uuids {
            if self.resolve_alias(name).is_some() {
                continue;
            }
            let Some(kb_type) = crate::kb::kb_type_for_label(label) else {
                continue;
            };
            let crate::kb::Resolution::Linked { entity: kbe, .. } = kb.resolve(name, kb_type)
            else {
                continue;
            };
            if name.eq_ignore_ascii_case(kbe.label) {
                continue;
            }
            if let Ok(Some(canon)) = self.find_entity_by_name(kbe.label) {
                if canon.uuid != *uuid && self.seed_aliases([(name.clone(), canon.uuid)]).is_ok() {
                    seeded += 1;
                }
            }
        }
        seeded
    }

    /// Map a causal-spine canonical relation label to a `RelationType`, preferring
    /// the named variants where they exist and falling back to `Custom`.
    fn relation_type_from_label(label: &str) -> RelationType {
        match label {
            "Causes" => RelationType::Causes,
            "Triggers" => RelationType::Triggers,
            "ResultsIn" => RelationType::ResultsIn,
            "RelatedTo" => RelationType::RelatedTo,
            other => RelationType::Custom(other.to_string()),
        }
    }

    /// Map a CATENA `LinkRelation` to its graph `RelationType` — factored out
    /// of the mint-site match (`mint_causal_spine_edges`) so it is unit-
    /// testable without a live dependency parser, mirroring
    /// `relation_type_from_label` above.
    fn relation_type_from_link(rel: crate::causal_vocab::LinkRelation) -> RelationType {
        match rel {
            crate::causal_vocab::LinkRelation::Causes => RelationType::Causes,
            crate::causal_vocab::LinkRelation::Precedes => RelationType::Precedes,
        }
    }

    /// Find or create an EVENT node for a CATENA event lemma. Exact-match reuse so
    /// repeated events (collapse, blackout) are one node; new events are added
    /// with `EntityLabel::Event`.
    fn get_or_create_event(&self, lemma: &str, now: DateTime<Utc>) -> Option<Uuid> {
        let name = lemma.trim();
        if name.len() < 3 {
            return None;
        }
        if let Ok(Some(existing)) = self.find_entity_by_name_strict(name) {
            return Some(existing.uuid);
        }
        let node = EntityNode {
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            labels: vec![EntityLabel::Event],
            created_at: now,
            last_seen_at: now,
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: EntityExtractor::calculate_base_salience(&EntityLabel::Event, false),
            is_proper_noun: false,
            selectivity: None,
            fine_type: None,
            kb_id: None,
        };
        self.add_entity(node).ok()
    }

    /// Build a causal-spine `RelationshipEdge` with provenance from the given
    /// typing method. Born strong (typed edges are the spine, not co-occurrence).
    #[allow(clippy::too_many_arguments)]
    fn build_spine_edge(
        &self,
        from_entity: Uuid,
        to_entity: Uuid,
        relation_type: RelationType,
        source_episode: Uuid,
        context: &str,
        now: DateTime<Utc>,
        typed_by: TypingMethod,
    ) -> RelationshipEdge {
        let ctx: String = context.chars().take(150).collect();
        let span_len = ctx.chars().count() as u32;
        RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity,
            to_entity,
            relation_type,
            strength: 0.7,
            created_at: now,
            valid_at: now,
            invalidated_at: None,
            source_episode_id: Some(source_episode),
            context: ctx,
            last_activated: now,
            activation_count: 1,
            ltp_status: LtpStatus::None,
            tier: EdgeTier::L1Working,
            activation_timestamps: None,
            entity_confidence: None,
            forman_curvature: None,
            endpoint_selectivity: None,
            provenance: vec![ProvenanceRecord {
                source_episode_id: source_episode,
                mention_count: 1,
                first_observed: now,
                last_observed: now,
                confidence: None,
                evidence_span: Some((0, span_len)),
                typed_by: Some(typed_by),
            }],
            promoted_at: None,
        }
    }

    /// Mint the causal-spine edges for an ingested passage: OpenIE clause-level
    /// entity→entity causal edges + CATENA event→event edges (the sparse narrative
    /// spine). Parser-based clause/event extraction reaches causation the
    /// entity-pair cue typer structurally cannot. Runs only when the dependency
    /// parser is available (`SHODH_SPACY_MODEL_PATH`) and not disabled via
    /// `SHODH_CAUSAL_SPINE=0`. Precision-gated: causal families only, abstract-
    /// social predicates (`supported`/`affected`) skipped. Returns edges minted.
    pub fn mint_causal_spine_edges(
        &self,
        content: &str,
        entity_uuids: &[(String, Uuid, EntityLabel)],
        source_episode: Uuid,
        now: DateTime<Utc>,
    ) -> usize {
        if !crate::dep_parser::is_available() {
            return 0;
        }
        if std::env::var("SHODH_CAUSAL_SPINE")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
        {
            return 0;
        }
        let mut minted = 0usize;

        // Entity name → UUID, longest name first so `container ship` beats `ship`.
        let mut names: Vec<(String, Uuid)> = entity_uuids
            .iter()
            .filter(|(n, _, _)| n.trim().len() >= 3)
            .map(|(n, u, _)| (n.to_lowercase(), *u))
            .collect();
        names.sort_by_key(|(n, _)| std::cmp::Reverse(n.len()));
        // Word-boundary match (space-padded) so `port` matches "the Port of
        // Baltimore" but NOT "support"; longest entity name wins (sort above).
        let match_entity = |span: &str| -> Option<Uuid> {
            let s = format!(" {} ", span.to_lowercase());
            names
                .iter()
                .find(|(n, _)| s.contains(&format!(" {} ", n)))
                .map(|(_, u)| *u)
        };

        // OpenIE — entity→entity typed causal edges (grammar supplies the predicate).
        if let Some(triples) = crate::openie::extract_triples(content) {
            for tr in triples {
                if !tr.causal || tr.low_precision {
                    continue;
                }
                let (Some(from), Some(to)) = (match_entity(&tr.subject), match_entity(&tr.object))
                else {
                    continue;
                };
                if from == to {
                    continue;
                }
                let rt = Self::relation_type_from_label(tr.relation);
                let edge = self.build_spine_edge(
                    from,
                    to,
                    rt,
                    source_episode,
                    content,
                    now,
                    TypingMethod::OpenIe,
                );
                if self.add_relationship(edge).is_ok() {
                    minted += 1;
                }
            }
        }

        // CATENA — event→event edges (the inchoative pivots no entity arm sees):
        // causal signals mint `Causes`, temporal signals mint `Precedes` — sequence
        // is never reported as causation.
        if let Some(links) = crate::catena::extract_event_links(content) {
            for link in links {
                let (Some(from), Some(to)) = (
                    self.get_or_create_event(&link.source, now),
                    self.get_or_create_event(&link.target, now),
                ) else {
                    continue;
                };
                if from == to {
                    continue;
                }
                let rt = Self::relation_type_from_link(link.relation);
                let edge = self.build_spine_edge(
                    from,
                    to,
                    rt,
                    source_episode,
                    content,
                    now,
                    TypingMethod::Catena,
                );
                if self.add_relationship(edge).is_ok() {
                    minted += 1;
                }
            }
        }

        if minted > 0 {
            tracing::info!(edges = minted, "causal spine: minted OpenIE + CATENA edges");
        }
        minted
    }

    /// Canonicalize the entity graph: run the resolver over the live mentions and
    /// MERGE each cluster's duplicate mention nodes into its canonical entity —
    /// re-pointing every edge onto the canonical (add_relationship dedups by type,
    /// so duplicate edges collapse) and deleting the duplicate node. This is what
    /// folds `Dali` / `the Dali` / `container ship` into one node and prunes the
    /// mention-duplication that makes the graph read as a hairball. The resolver
    /// uses the dependency parser for head detection; no-op returning `(0, 0)` when
    /// the parser is unavailable. Returns `(nodes_merged, edges_repointed)`:
    /// nodes actually DELETED and edges whose canonical copy was written. A member
    /// whose migration failed partway survives untouched-or-duplicated (never
    /// holed) and is retried by the next pass — failure never loses an edge.
    pub fn canonicalize_entities(&self) -> Result<(usize, usize)> {
        use crate::entity_resolution::parse_mention_tokens;
        use crate::fs_matcher::{cluster, MatchRecord};

        if !crate::dep_parser::is_available() {
            return Ok((0, 0));
        }
        let entities = self.get_all_entities()?;
        if entities.len() < 2 {
            return Ok((0, 0));
        }

        // Parse each mention (spaCy-rusty) for its syntactic head, drop verb-
        // fragment junk (`is_entity`), and build a Fellegi-Sunter record with the
        // full comparison evidence the matcher scores over. This is the
        // Galárraga (CIKM'14) + CESI (WWW'18) feature union realized as FS
        // comparisons: name (Jaro-Winkler + IDF), head, type, ATTRIBUTE OVERLAP
        // (the typed relations the entity participates in — Galárraga's key
        // signal), and NAME EMBEDDING (CESI's learned-embedding signal). Name
        // + head + type alone only catch exact duplicates; the relation and
        // embedding evidence let it merge abbreviations / paraphrases
        // ("Key Bridge" ≡ "Francis Scott Key Bridge") that surface strings miss.
        let mut records: Vec<MatchRecord> = Vec::new();
        // (uuid, is_proper, mentions, name, kb_id)
        let mut meta: Vec<(Uuid, bool, usize, String, Option<String>)> = Vec::new();
        for e in &entities {
            let Some(parsed) =
                crate::dep_parser::parse(&e.name).and_then(|t| parse_mention_tokens(&t))
            else {
                continue;
            };
            if !parsed.is_entity() {
                continue;
            }
            let primary_type = e
                .labels
                .first()
                .map(|l| l.as_str().to_string())
                .unwrap_or_default();
            // Attribute overlap: the TYPED relations this entity participates in.
            // Generic co-occurrence (CoOccurs/RelatedTo/CoRetrieved) is excluded —
            // it is undiscriminative (everything co-occurs) and would wash out the
            // signal. Two mentions of one real entity share these typed relations.
            let agent_roles: HashSet<String> = self
                .get_entity_relationships(&e.uuid)
                .unwrap_or_default()
                .iter()
                .filter(|edge| {
                    !matches!(
                        edge.relation_type,
                        RelationType::CoOccurs
                            | RelationType::RelatedTo
                            | RelationType::CoRetrieved
                    )
                })
                .map(|edge| edge.relation_type.as_str().to_string())
                .collect();
            records.push(MatchRecord {
                name: parsed.clean.to_lowercase(),
                head: parsed.head.clone(),
                entity_type: primary_type,
                agent_roles,
                name_embedding: e.name_embedding.clone(),
                ..Default::default()
            });
            meta.push((
                e.uuid,
                e.is_proper_noun,
                e.mention_count,
                e.name.clone(),
                e.kb_id.clone(),
            ));
        }
        if records.len() < 2 {
            return Ok((0, 0));
        }

        // Splink: fit + type-blocked clustering at a precision-first threshold.
        let mut clusters = cluster(&records, 0.9, 20);

        // Contrastive projection adapter (ER Task 4.2, Sudowoodo-lite) — GATED.
        // Self-supervise a tiny projection over the frozen name embeddings from the
        // confident base merges (within-cluster pairs = positives; cross-type pairs =
        // negatives, which prevent collapse), then re-cluster in the learned space to
        // catch coreferent surfaces the frozen embedding missed. Opt in with
        // SHODH_CONTRASTIVE_ADAPTER=1 so it can be measured before earning the default.
        let adapter_on = std::env::var("SHODH_CONTRASTIVE_ADAPTER")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if adapter_on {
            if let Some(dim) = records
                .iter()
                .find_map(|r| r.name_embedding.as_ref().map(|e| e.len()))
            {
                let mut adapter = crate::contrastive::ContrastiveAdapter::identity(dim);
                for c in clusters.iter().filter(|c| c.len() >= 2) {
                    for a in 0..c.len() {
                        for b in (a + 1)..c.len() {
                            let (ri, rj) = (&records[c[a]], &records[c[b]]);
                            if let (Some(ea), Some(eb)) = (&ri.name_embedding, &rj.name_embedding) {
                                // A negative: the nearest-indexed record of a DIFFERENT
                                // entity type (never coreferent), guarding over-merge.
                                let neg = records
                                    .iter()
                                    .find(|r| {
                                        r.entity_type != ri.entity_type
                                            && r.name_embedding.is_some()
                                    })
                                    .and_then(|r| r.name_embedding.clone());
                                adapter.learn(ea, eb, neg.as_deref());
                                adapter.learn(eb, ea, neg.as_deref());
                            }
                        }
                    }
                }
                for r in records.iter_mut() {
                    if let Some(e) = r.name_embedding.take() {
                        r.name_embedding = Some(adapter.project(&e));
                    }
                }
                clusters = cluster(&records, 0.9, 20);
            }
        }

        let mut merged_nodes = 0usize;
        // Edges whose canonical copy was written (member original deleted, or at
        // worst left as a recoverable duplicate — see `edges_unmigrated` for the
        // edges that never got a canonical copy at all).
        let mut repointed = 0usize;
        // Merges refused because the two nodes carry different KB identities.
        let mut kb_vetoed = 0usize;
        // Edges whose migration failed in a way that kept them on the member
        // (add of the canonical copy failed, or a self-loop delete failed). Each
        // one also blocks its member's deletion — see `members_kept`.
        let mut edges_unmigrated = 0usize;
        // Members that survived their merge: either an edge could not be
        // migrated (the node must stay so the edge keeps a real endpoint) or
        // `delete_entity` itself failed. Not counted in `merged_nodes` — the
        // return value reports nodes actually removed, and the next
        // canonicalize pass re-forms the cluster and retries.
        let mut members_kept = 0usize;
        // Alias pairs to seed after merging: every merged surface → its canonical
        // UUID. `resolve_alias` is checked at ingest (Tier 0), so once seeded, a
        // future mention of that surface redirects to the canonical node instead of
        // re-creating the duplicate — this closes the ingest loop (canonicalize →
        // aliases → cleaner ingest → less to canonicalize).
        let mut alias_pairs: Vec<(String, Uuid)> = Vec::new();
        for cluster_idxs in clusters {
            if cluster_idxs.len() < 2 {
                continue;
            }
            // Canonical = the most-proper / most-mentioned member.
            //
            // SURVIVAL COUPLING: `meta[i].1` is the stored `is_proper_noun` flag,
            // and as the PRIMARY sort key it decides which node SURVIVES a merge
            // and which is deleted — it is not a mere salience input here.
            // Changing how that flag is derived at construction is therefore a
            // change to entity resolution, not just to scoring. And because
            // stored nodes keep the flag they were written with (postcard is
            // positional, and the add_entity OR-merge only ever promotes), a
            // live graph holds a MIXED population across derivation eras, so
            // which member wins can depend on when each node was written.
            let canon_idx = *cluster_idxs
                .iter()
                .max_by_key(|&&i| (meta[i].1, meta[i].2))
                .unwrap();
            let canonical = meta[canon_idx].0;
            // The canonical node's real-world identity. It can be *acquired*
            // below: the canonical is picked for prominence, not for being
            // linked, so the node carrying the QID is often the one about to be
            // deleted.
            let mut canon_kb_id = meta[canon_idx].4.clone();
            for &i in &cluster_idxs {
                if i == canon_idx {
                    continue;
                }
                let member = meta[i].0;
                // KB identity is a merge VETO. Two nodes carrying different QIDs
                // are known to be different real-world things no matter how well
                // their surfaces score, and fusing them is precisely the
                // permanent corruption entity linking exists to prevent. The
                // string matcher cannot see this; the KB can.
                if !kb_identities_permit_merge(canon_kb_id.as_deref(), meta[i].4.as_deref()) {
                    tracing::debug!(
                        canonical = %meta[canon_idx].3,
                        member = %meta[i].3,
                        kept_kb_id = canon_kb_id.as_deref().unwrap_or(""),
                        member_kb_id = meta[i].4.as_deref().unwrap_or(""),
                        "canonicalize: refused merge — distinct KB identities"
                    );
                    kb_vetoed += 1;
                    continue;
                }
                // Re-point every edge touching the member onto the canonical node.
                //
                // ORDER IS LOAD-BEARING: the canonical copy is written BEFORE the
                // member's original edge is deleted. `add_relationship` can fail,
                // and deleting first turned that failure into silent, permanent
                // edge loss mid-merge (the edge was gone from the member and never
                // reached the canonical). Failing in this order instead leaves at
                // worst a DUPLICATE — and add_relationship dedups by
                // (from,to,type), so the duplicate collapses on retry anyway. A
                // duplicate is recoverable; a hole is not.
                let member_edges = self.get_entity_relationships(&member)?;
                // Set when an edge could not be fully migrated off the member.
                // The member must then SURVIVE this pass: deleting it would leave
                // the un-migrated edge dangling from a nonexistent node.
                let mut member_dirty = false;
                for edge in member_edges {
                    let mut ne = edge.clone();
                    ne.uuid = Uuid::new_v4();
                    if ne.from_entity == member {
                        ne.from_entity = canonical;
                    }
                    if ne.to_entity == member {
                        ne.to_entity = canonical;
                    }
                    if ne.from_entity == ne.to_entity {
                        // Merge-created self-loop (member↔canonical, or an existing
                        // member↔member loop): two mentions of one entity relating
                        // to each other is not a relation — drop it. There is no
                        // canonical copy to add first, so here the DELETE is the
                        // migration and is the step whose failure must keep the
                        // member alive.
                        if let Err(e) = self.delete_relationship(&edge.uuid) {
                            tracing::warn!(
                                edge = %edge.uuid,
                                member = %meta[i].3,
                                error = %e,
                                "canonicalize: failed to drop merge self-loop; keeping member"
                            );
                            edges_unmigrated += 1;
                            member_dirty = true;
                        }
                        continue;
                    }
                    match self.add_relationship(ne) {
                        Ok(_) => {
                            // The canonical copy exists; the member's original may
                            // now go. If THIS delete fails the edge is duplicated,
                            // never lost — but the member still cannot be deleted
                            // under a live edge.
                            match self.delete_relationship(&edge.uuid) {
                                Ok(_) => repointed += 1,
                                Err(e) => {
                                    tracing::warn!(
                                        edge = %edge.uuid,
                                        member = %meta[i].3,
                                        error = %e,
                                        "canonicalize: canonical copy written but member edge delete failed; keeping member"
                                    );
                                    // The canonical side is complete, so this still
                                    // counts as re-pointed; the survivor is a dup.
                                    repointed += 1;
                                    member_dirty = true;
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                edge = %edge.uuid,
                                member = %meta[i].3,
                                error = %e,
                                "canonicalize: failed to re-point edge onto canonical; keeping original edge and member"
                            );
                            edges_unmigrated += 1;
                            member_dirty = true;
                        }
                    }
                }
                if member_dirty {
                    // Partial migration: leave the member (and its surviving
                    // edges) in place, seed no alias, inherit no identity. The
                    // next canonicalize pass re-clusters this pair and retries;
                    // already-migrated edges dedup into the canonical copies.
                    members_kept += 1;
                    continue;
                }
                if let Err(e) = self.delete_entity(&member) {
                    // Edges are fully migrated (the member is edgeless), but the
                    // node itself would not die. It is NOT a completed merge —
                    // the return value reports nodes actually removed — so leave
                    // alias/identity alone and let the next pass retry.
                    tracing::warn!(
                        member = %meta[i].3,
                        error = %e,
                        "canonicalize: edges re-pointed but member node delete failed; merge not counted"
                    );
                    members_kept += 1;
                    continue;
                }
                merged_nodes += 1;
                // Remember this surface → canonical mapping (raw name + parsed clean
                // form) so future ingests of the merged surface resolve directly.
                // Only for COMPLETED merges: an alias for a still-live member would
                // shadow its node at ingest while it keeps accreting nothing.
                alias_pairs.push((meta[i].3.clone(), canonical));
                if records[i].name != meta[i].3.trim().to_lowercase() {
                    alias_pairs.push((records[i].name.clone(), canonical));
                }
                // Inherit an identity rather than deleting it with the node that
                // held it.
                if canon_kb_id.is_none() {
                    canon_kb_id.clone_from(&meta[i].4);
                }
            }
            // Persist an identity acquired from a merged member.
            if canon_kb_id != meta[canon_idx].4 {
                if let Ok(Some(mut node)) = self.get_entity(&canonical) {
                    node.kb_id.clone_from(&canon_kb_id);
                    if let Ok(encoded) = crate::serialization::encode(&node) {
                        let _ = self
                            .db
                            .put_cf(self.entities_cf(), canonical.as_bytes(), encoded);
                    }
                }
            }
        }
        // Close the ingest loop: seed the merged surfaces as aliases of their
        // canonical node so re-ingesting them never re-creates the duplicate.
        let aliases_seeded = if alias_pairs.is_empty() {
            0
        } else {
            self.seed_aliases(alias_pairs).unwrap_or(0)
        };
        tracing::info!(
            merged = merged_nodes,
            repointed,
            aliases_seeded,
            kb_vetoed,
            edges_unmigrated,
            members_kept,
            "canonicalize (Splink): merged duplicate mention nodes into canonical entities"
        );
        Ok((merged_nodes, repointed))
    }

    /// Load lowercase name->UUID index, or migrate from name_index if empty
    ///
    /// This enables O(1) case-insensitive entity lookup instead of O(n) linear search.
    fn load_or_migrate_lowercase_index(
        db: &DB,
        name_index: &HashMap<String, Uuid>,
    ) -> Result<HashMap<String, Uuid>> {
        let lowercase_cf = db
            .cf_handle(CF_LOWERCASE_INDEX)
            .ok_or_else(|| anyhow::anyhow!("CF '{}' not found", CF_LOWERCASE_INDEX))?;
        let mut index = HashMap::new();

        // Try to load from lowercase_index CF
        let iter = db.iterator_cf(lowercase_cf, rocksdb::IteratorMode::Start);
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
                db.put_cf(lowercase_cf, lowercase_name.as_bytes(), uuid.as_bytes())?;
                index.insert(lowercase_name, *uuid);
            }
            tracing::info!(
                "Migrated {} entities to lowercase index CF",
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
        db: &DB,
        name_index: &HashMap<String, Uuid>,
    ) -> Result<HashMap<String, Uuid>> {
        let stemmed_cf = db
            .cf_handle(CF_STEMMED_INDEX)
            .ok_or_else(|| anyhow::anyhow!("CF '{}' not found", CF_STEMMED_INDEX))?;
        let mut index = HashMap::new();

        // Try to load from stemmed_index CF
        let iter = db.iterator_cf(stemmed_cf, rocksdb::IteratorMode::Start);
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
                db.put_cf(stemmed_cf, stemmed_name.as_bytes(), uuid.as_bytes())?;
                index.insert(stemmed_name, *uuid);
            }
            tracing::info!("Migrated {} entities to stemmed index CF", name_index.len());
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

    /// Count entries in a column family (one-time startup cost)
    fn count_cf_entries(db: &DB, cf: &ColumnFamily) -> usize {
        db.iterator_cf(cf, rocksdb::IteratorMode::Start).count()
    }

    /// Count only true relationship-edge records in the relationships CF.
    ///
    /// The relationships CF holds two kinds of keys: real edge records (16-byte
    /// UUID key → encoded `RelationshipEdge`) and `mem_edge:<a>:<b>`
    /// forward/reverse index keys (ASCII key → 16-byte edge UUID). The generic
    /// `count_cf_entries` counts both, which over-states the edge count (~3x for
    /// memory-pair edges). The startup seed for `relationship_count` must count
    /// only the 16-byte-keyed edge records so it stays consistent with the
    /// increments/decrements applied at runtime.
    fn count_relationship_edges(db: &DB, cf: &ColumnFamily) -> usize {
        db.iterator_cf(cf, rocksdb::IteratorMode::Start)
            .filter_map(|r| r.ok())
            .filter(|(key, _)| key.len() == 16)
            .count()
    }

    /// Load entity embedding cache from persisted entities.
    ///
    /// Scans entities referenced by the name index and collects those with
    /// pre-computed name_embeddings into an in-memory cache for O(n) concept
    /// merging during `add_entity()`. Entities without embeddings (pre-upgrade
    /// data) are skipped and will gain embeddings on their next mention.
    fn load_entity_embedding_cache(
        db: &DB,
        entities_cf: &ColumnFamily,
        name_index: &HashMap<String, Uuid>,
    ) -> Vec<(Uuid, Vec<f32>)> {
        let mut cache = Vec::with_capacity(ENTITY_EMBEDDING_CACHE_MAX.min(name_index.len()));
        for uuid in name_index.values() {
            let key = uuid.as_bytes();
            if let Ok(Some(value)) = db.get_cf(entities_cf, key) {
                if let Ok((entity, _)) = decode_entity_node(&value) {
                    if let Some(emb) = entity.name_embedding {
                        cache.push((*uuid, emb));
                        if cache.len() >= ENTITY_EMBEDDING_CACHE_MAX {
                            break;
                        }
                    }
                }
            }
        }
        cache
    }

    /// Add or update an entity node
    /// Salience is updated using the formula: salience = base_salience * (1 + 0.1 * ln(mention_count))
    /// This means frequently mentioned entities grow in salience (gravitational wells get heavier)
    ///
    /// BUG-002 FIX: Handles crash recovery for orphaned entities/stale indices
    pub fn add_entity(&self, mut entity: EntityNode) -> Result<Uuid> {
        // Multi-tier dedup pipeline: exact → case-insensitive → stemmed → embedding
        // Each tier is faster than the next; short-circuits on first match.

        // Tier 1: Exact name match (O(1))
        let mut existing_uuid = {
            let index = self.entity_name_index.read();
            index.get(&entity.name).cloned()
        };

        // Tier 2: Case-insensitive match (O(1))
        if existing_uuid.is_none() {
            let lowercase_name = entity.name.to_lowercase();
            let index = self.entity_lowercase_index.read();
            existing_uuid = index.get(&lowercase_name).cloned();
        }

        // Tier 3: Stemmed match (O(1)) — "running" matches "run"
        // Skip for proper nouns to prevent "Paris" → "pari" merging with "Parison"
        if existing_uuid.is_none() && !entity.is_proper_noun {
            let stemmer = Stemmer::create(Algorithm::English);
            let stemmed_name = Self::stem_entity_name(&stemmer, &entity.name);
            let index = self.entity_stemmed_index.read();
            existing_uuid = index.get(&stemmed_name).cloned();
        }

        // Tier 4: Embedding-based concept merge (O(n) over cache)
        // Catches synonyms like "authentication" ↔ "auth" that string matching misses.
        // Only runs when the entity carries a name_embedding (populated by caller).
        if existing_uuid.is_none() {
            if let Some(ref new_emb) = entity.name_embedding {
                let cache = self.entity_embedding_cache.read();
                let mut best_match: Option<(Uuid, f32)> = None;
                for (uuid, existing_emb) in cache.iter() {
                    let sim = crate::similarity::cosine_similarity(new_emb, existing_emb);
                    if sim >= ENTITY_CONCEPT_MERGE_THRESHOLD
                        && best_match.is_none_or(|(_, best_sim)| sim > best_sim)
                    {
                        best_match = Some((*uuid, sim));
                    }
                }
                if let Some((matched_uuid, sim)) = best_match {
                    tracing::debug!(
                        "Concept merge: '{}' matched existing entity {} (cosine={:.3})",
                        entity.name,
                        matched_uuid,
                        sim
                    );
                    existing_uuid = Some(matched_uuid);
                }
            }
        }

        let is_new_entity;
        if let Some(uuid) = existing_uuid {
            // BUG-002 FIX: Verify entity actually exists in DB (handles stale index)
            if let Some(existing) = self.get_entity(&uuid)? {
                // Update existing entity — merge into canonical node
                entity.uuid = uuid;
                entity.mention_count = existing.mention_count + 1;
                entity.last_seen_at = Utc::now();
                entity.created_at = existing.created_at;
                entity.is_proper_noun = existing.is_proper_noun || entity.is_proper_noun;

                // Preserve the canonical name (first-seen name wins)
                entity.name = existing.name.clone();

                // Merge labels: preserve all observed entity types
                for label in &existing.labels {
                    if !entity.labels.contains(label) {
                        entity.labels.push(label.clone());
                    }
                }

                // Preserve existing embedding if the incoming one is None
                if entity.name_embedding.is_none() {
                    entity.name_embedding = existing.name_embedding;
                }

                // Preserve an existing fine type when the re-mention carries none.
                // Re-mentions are the common case (pre-extracted names, tags, fallback
                // entities), and they must not wipe a fine type GLiNER already set.
                if entity.fine_type.is_none() {
                    entity.fine_type = existing.fine_type.clone();
                }

                // KB identity is write-once. An id already on the node always
                // wins: re-mentions arrive with a freshly-resolved id, and
                // letting the newest one overwrite would let a single ambiguous
                // mention silently repoint an established entity at a different
                // real-world thing. A disagreement is worth knowing about, so
                // log it rather than resolving it silently.
                if existing.kb_id.is_some() {
                    if let (Some(old), Some(new)) = (&existing.kb_id, &entity.kb_id) {
                        if old != new {
                            tracing::warn!(
                                entity = %entity.name,
                                kept = %old,
                                rejected = %new,
                                "KB id conflict on re-mention; keeping the established identity"
                            );
                        }
                    }
                    entity.kb_id = existing.kb_id.clone();
                }

                // Merge summary: first non-empty wins, preserve existing
                if !existing.summary.is_empty() {
                    entity.summary = existing.summary.clone();
                }

                // Merge attributes: add new keys without overwriting existing
                for (k, v) in &existing.attributes {
                    entity
                        .attributes
                        .entry(k.clone())
                        .or_insert_with(|| v.clone());
                }

                // Update salience with frequency boost
                // Formula: salience = base_salience * (1 + 0.1 * ln(mention_count))
                // This caps at about 1.3x boost at 20 mentions
                let frequency_boost = 1.0 + 0.1 * (entity.mention_count as f32).ln();
                entity.salience = (existing.salience * frequency_boost).min(1.0);
                is_new_entity = false;
            } else {
                // BUG-002 FIX: Stale index entry - entity in index but not in DB
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
            // Genuinely new entity — no match at any tier
            entity.uuid = Uuid::new_v4();
            entity.created_at = Utc::now();
            entity.last_seen_at = entity.created_at;
            entity.mention_count = 1;
            is_new_entity = true;
        }

        // Stamp the real-world identity. Free of extra I/O — the node is about to
        // be written anyway — and a pure function of (name, labels), so a
        // re-ingest or a re-enrichment of the same content recomputes the same
        // id instead of accumulating state. Abstains by default: most entities
        // legitimately end up with `None`.
        if entity.kb_id.is_none() {
            entity.kb_id = crate::kb::stamp(&entity.name, &entity.labels);
        }

        // BUG-002 FIX: Write index FIRST, then entity
        let lowercase_name = entity.name.to_lowercase();
        let stemmer = Stemmer::create(Algorithm::English);
        let stemmed_name = Self::stem_entity_name(&stemmer, &entity.name);

        // Update in-memory indices
        {
            let mut index = self.entity_name_index.write();
            index.insert(entity.name.clone(), entity.uuid);
        }
        {
            let mut lowercase_index = self.entity_lowercase_index.write();
            lowercase_index.insert(lowercase_name.clone(), entity.uuid);
        }
        // Skip stemmed index for proper nouns to prevent "Paris" → "pari" collisions
        if !entity.is_proper_noun {
            let mut stemmed_index = self.entity_stemmed_index.write();
            stemmed_index.insert(stemmed_name.clone(), entity.uuid);
        }

        // Update entity embedding cache for future concept merges.
        // Recency-of-mention ordering: the front is the least-recently-mentioned
        // entry (the eviction victim), the back is the most recent. A re-mention
        // counts as an access and moves the entry to the back, so the drain below
        // removes genuinely cold entities rather than merely the earliest-added
        // (which may still be hot). Only matters once the graph exceeds
        // ENTITY_EMBEDDING_CACHE_MAX (10k) distinct embedded entities.
        if let Some(ref emb) = entity.name_embedding {
            let mut cache = self.entity_embedding_cache.write();
            if is_new_entity {
                cache.push((entity.uuid, emb.clone()));
                if cache.len() > ENTITY_EMBEDDING_CACHE_MAX {
                    let excess = cache.len() - ENTITY_EMBEDDING_CACHE_MAX;
                    cache.drain(..excess);
                }
            } else if let Some(pos) = cache.iter().position(|(uuid, _)| *uuid == entity.uuid) {
                // Re-mention: refresh the (possibly changed) embedding and promote
                // to the back as the most-recently-accessed entry.
                let mut entry = cache.remove(pos);
                entry.1 = emb.clone();
                cache.push(entry);
            }
        }

        // Persist name->UUID mappings
        self.db.put_cf(
            self.name_index_cf(),
            entity.name.as_bytes(),
            entity.uuid.as_bytes(),
        )?;
        self.db.put_cf(
            self.lowercase_index_cf(),
            lowercase_name.as_bytes(),
            entity.uuid.as_bytes(),
        )?;
        if !entity.is_proper_noun {
            self.db.put_cf(
                self.stemmed_index_cf(),
                stemmed_name.as_bytes(),
                entity.uuid.as_bytes(),
            )?;
        }

        // Store entity in database
        let key = entity.uuid.as_bytes();
        let value = crate::serialization::encode(&entity)?;
        self.db.put_cf(self.entities_cf(), key, value)?;

        // Increment counter only for truly new entities
        if is_new_entity {
            self.entity_count.fetch_add(1, Ordering::Relaxed);
        }

        Ok(entity.uuid)
    }

    /// Get entity by UUID
    pub fn get_entity(&self, uuid: &Uuid) -> Result<Option<EntityNode>> {
        let key = uuid.as_bytes();
        match self.db.get_cf(self.entities_cf(), key)? {
            Some(value) => {
                let (entity, _) = decode_entity_node(&value)?;
                Ok(Some(entity))
            }
            None => Ok(None),
        }
    }

    /// Delete an entity and all its index entries.
    ///
    /// Removes the entity from:
    /// 1. `entities` CF (primary storage)
    /// 2. `entity_name_index` (exact name → UUID)
    /// 3. `entity_lowercase_index` (lowercase name → UUID)
    /// 4. `entity_stemmed_index` (stemmed name → UUID)
    /// 5. `entity_embedding_cache` (in-memory embedding vector)
    /// 6. `entity_pair_index` CF (co-occurrence pair entries)
    /// 7. Decrements `entity_count`
    ///
    /// Returns true if the entity existed and was deleted.
    pub fn delete_entity(&self, uuid: &Uuid) -> Result<bool> {
        let entity = match self.get_entity(uuid)? {
            Some(e) => e,
            None => return Ok(false),
        };

        // 1. Remove from entities CF
        self.db.delete_cf(self.entities_cf(), uuid.as_bytes())?;

        // 2-3-4. Remove from name indices (in-memory + persisted)
        let lowercase_name = entity.name.to_lowercase();
        let stemmer = Stemmer::create(Algorithm::English);
        let stemmed_name = Self::stem_entity_name(&stemmer, &entity.name);

        {
            let mut index = self.entity_name_index.write();
            index.remove(&entity.name);
        }
        self.db
            .delete_cf(self.name_index_cf(), entity.name.as_bytes())?;

        {
            let mut index = self.entity_lowercase_index.write();
            index.remove(&lowercase_name);
        }
        self.db
            .delete_cf(self.lowercase_index_cf(), lowercase_name.as_bytes())?;

        {
            let mut index = self.entity_stemmed_index.write();
            index.remove(&stemmed_name);
        }
        self.db
            .delete_cf(self.stemmed_index_cf(), stemmed_name.as_bytes())?;

        // 5. Remove from embedding cache
        {
            let mut cache = self.entity_embedding_cache.write();
            cache.retain(|(id, _)| id != uuid);
        }

        // 6. Remove entity_pair_index entries (prefix scan)
        let prefix = format!("{}:", uuid);
        let mut pairs_to_delete = Vec::new();
        let iter = self
            .db
            .prefix_iterator_cf(self.entity_pair_index_cf(), prefix.as_bytes());
        for item in iter {
            match item {
                Ok((key, _)) => {
                    let key_str = String::from_utf8_lossy(&key);
                    if key_str.starts_with(&prefix) {
                        pairs_to_delete.push(key.to_vec());
                    } else {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        // Also scan for reverse direction (other_uuid:this_uuid)
        let suffix = format!(":{}", uuid);
        let iter = self
            .db
            .iterator_cf(self.entity_pair_index_cf(), rocksdb::IteratorMode::Start);
        for item in iter {
            match item {
                Ok((key, _)) => {
                    let key_str = String::from_utf8_lossy(&key);
                    if key_str.ends_with(&suffix) {
                        pairs_to_delete.push(key.to_vec());
                    }
                }
                Err(_) => break,
            }
        }
        for key in &pairs_to_delete {
            self.db.delete_cf(self.entity_pair_index_cf(), key)?;
        }

        // 6b. Remove entity_episodes inverted-index entries (entity_uuid:episode_uuid).
        // These are otherwise only cleaned per-episode (delete_episode), so deleting
        // an entity while episodes still reference it in entity_refs leaves stale
        // entries that accumulate over orphan-cleanup churn. (entity_edges entries
        // need no handling here: delete_relationship already removes both endpoints
        // when each edge dies, and this entity is edgeless by the time it is deleted.)
        let ep_prefix = format!("{}:", uuid);
        let mut episode_keys_to_delete = Vec::new();
        let iter = self
            .db
            .prefix_iterator_cf(self.entity_episodes_cf(), ep_prefix.as_bytes());
        for item in iter {
            match item {
                Ok((key, _)) => {
                    if key.starts_with(ep_prefix.as_bytes()) {
                        episode_keys_to_delete.push(key.to_vec());
                    } else {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        for key in &episode_keys_to_delete {
            if let Err(e) = self.db.delete_cf(self.entity_episodes_cf(), key) {
                tracing::warn!(entity = %uuid, error = %e, "Failed to delete from entity_episodes index");
            }
        }

        // 7. Decrement counter
        self.entity_count.fetch_sub(1, Ordering::Relaxed);

        tracing::debug!("Deleted orphaned entity '{}' (uuid={})", entity.name, uuid);
        Ok(true)
    }

    /// PHRASE-LEVEL precision resolution: entities whose FULL (lowercased) name
    /// occurs verbatim inside `text_lower`. The inverse of fuzzy lookup — instead
    /// of asking "which entity does this fragment match?" (where "incident"
    /// binds to an arbitrary hub), it asks "which entity names does the query
    /// actually CONTAIN?" ("the selvic incident" ⊂ query ✓; "the selvic1
    /// incident" ⊄ query ✗). Built for causal-walk seeding, where a wrong seed
    /// injects a wrong chain's origins into the candidate pool. Names shorter
    /// than `min_len` are skipped (stop-word-like entity names would match
    /// everything); capped at `max` results, longest names first (most specific).
    pub fn find_entities_contained_in_text(
        &self,
        text_lower: &str,
        min_len: usize,
        max: usize,
    ) -> Result<Vec<EntityNode>> {
        let mut hits: Vec<(usize, Uuid)> = {
            let lowercase_index = self.entity_lowercase_index.read();
            lowercase_index
                .iter()
                .filter(|(name, _)| name.len() >= min_len && text_lower.contains(name.as_str()))
                .map(|(name, uuid)| (name.len(), *uuid))
                .collect()
        };
        // Longest (most specific) first; deterministic tie-break by uuid.
        hits.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let mut out = Vec::new();
        for (_, uuid) in hits.into_iter().take(max) {
            if let Some(ent) = self.get_entity(&uuid)? {
                out.push(ent);
            }
        }
        Ok(out)
    }

    /// STRICT entity resolution: tiers 1-3 only (exact / case-insensitive /
    /// stemmed) — no substring or word-level fuzzing.
    ///
    /// Use this when the caller needs PRECISION over recall — e.g. seeds for
    /// the causal-origin walk. The lineage diagnosis (2026-06-10) showed the
    /// fuzzy tiers binding a fragmented query token ("incident") to an
    /// arbitrary hub node, making the backward walk inject OTHER chains' roots
    /// into the candidate pool and crowding the true root out of the top-10
    /// (harness root-cause P@1 pinned at 0.0). Spreading-activation seeding
    /// keeps the recall-oriented fuzzy resolver; precision consumers use this.
    pub fn find_entity_by_name_strict(&self, name: &str) -> Result<Option<EntityNode>> {
        let uuid = {
            let index = self.entity_name_index.read();
            index.get(name).copied()
        };
        if let Some(uuid) = uuid {
            return self.get_entity(&uuid);
        }
        let name_lower = name.to_lowercase();
        let uuid = {
            let lowercase_index = self.entity_lowercase_index.read();
            lowercase_index.get(&name_lower).copied()
        };
        if let Some(uuid) = uuid {
            return self.get_entity(&uuid);
        }
        let stemmer = Stemmer::create(Algorithm::English);
        let stemmed_name = Self::stem_entity_name(&stemmer, name);
        let uuid = {
            let stemmed_index = self.entity_stemmed_index.read();
            stemmed_index.get(&stemmed_name).copied()
        };
        if let Some(uuid) = uuid {
            return self.get_entity(&uuid);
        }
        Ok(None)
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
        // Tier 0: Alias resolution (curated canonical mapping — highest priority).
        // A surface the resolver merged (e.g. "cargo ship" -> the Dali) resolves
        // straight to its canonical node rather than matching its own stale
        // mention. If the canonical UUID is dangling, fall through to name lookup.
        if let Some(canonical) = self.resolve_alias(name) {
            if let Some(entity) = self.get_entity(&canonical)? {
                return Ok(Some(entity));
            }
        }

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
        // Deterministic: collect ALL matches, pick highest salience (break ties by shortest name)
        if name.len() >= 3 {
            let lowercase_index = self.entity_lowercase_index.read();
            let mut candidates: Vec<(Uuid, String)> = Vec::new();

            // Tier 4: Substring match - query is substring of entity
            // e.g., "York" matches "New York City"
            for (entity_name, uuid) in lowercase_index.iter() {
                if entity_name.contains(&name_lower) {
                    candidates.push((*uuid, entity_name.clone()));
                }
            }

            // Tier 5: Word-level match (only if Tier 4 found nothing)
            if candidates.is_empty() {
                let query_words: Vec<&str> = name_lower.split_whitespace().collect();
                for (entity_name, uuid) in lowercase_index.iter() {
                    let entity_words: Vec<&str> = entity_name.split_whitespace().collect();
                    for qw in &query_words {
                        if entity_words.iter().any(|ew| ew == qw || ew.starts_with(qw)) {
                            candidates.push((*uuid, entity_name.clone()));
                            break;
                        }
                    }
                }
            }

            // Pick best candidate: highest salience, then shortest name for ties
            if !candidates.is_empty() {
                let mut best: Option<(Uuid, f32, usize)> = None; // (uuid, salience, name_len)
                for (uuid, name) in &candidates {
                    let salience = self.get_entity(uuid)?.map(|e| e.salience).unwrap_or(0.0);
                    match &best {
                        Some((_, best_sal, best_len))
                            if salience > *best_sal
                                || (salience == *best_sal && name.len() < *best_len) =>
                        {
                            best = Some((*uuid, salience, name.len()));
                        }
                        None => {
                            best = Some((*uuid, salience, name.len()));
                        }
                        _ => {}
                    }
                }
                if let Some((uuid, _, _)) = best {
                    return self.get_entity(&uuid);
                }
            }
        }

        Ok(None)
    }

    /// Total number of episodes stored in the graph (the `N` in PMI / IDF statistics).
    /// O(1) — read from the maintained atomic counter.
    pub fn total_episode_count(&self) -> usize {
        self.episode_count.load(Ordering::Relaxed)
    }

    /// Lightweight quality metrics for an existing entity in the graph.
    /// Used by NER filtering to suppress known stop-word entities.
    pub fn get_entity_reputation(&self, name: &str) -> Option<EntityReputation> {
        // O(1) lookup via lowercase index
        let uuid = {
            let index = self.entity_lowercase_index.read();
            index.get(&name.to_lowercase()).copied()
        }?;

        let entity = self.get_entity(&uuid).ok()??;
        let degree = self.entity_edge_count(&uuid).unwrap_or(0);

        Some(EntityReputation {
            selectivity: entity.selectivity.unwrap_or(1.0), // Conservative default
            mention_count: entity.mention_count,
            degree,
            salience: entity.salience,
        })
    }

    /// Canonical pair key for the entity-pair index.
    /// Uses min/max UUID ordering so A→B and B→A produce the same key.
    fn pair_key(entity_a: &Uuid, entity_b: &Uuid) -> String {
        if entity_a < entity_b {
            format!("{entity_a}:{entity_b}")
        } else {
            format!("{entity_b}:{entity_a}")
        }
    }

    /// Typed pair key: includes relation type to support multiple edges per pair
    fn typed_pair_key(entity_a: &Uuid, entity_b: &Uuid, relation_type: &RelationType) -> String {
        format!(
            "{}:{}",
            Self::pair_key(entity_a, entity_b),
            relation_type.as_str()
        )
    }

    /// Index an entity pair + relation type → edge UUID for O(1) dedup lookups
    fn index_entity_pair(
        &self,
        entity_a: &Uuid,
        entity_b: &Uuid,
        edge_uuid: &Uuid,
        relation_type: &RelationType,
    ) -> Result<()> {
        let key = Self::typed_pair_key(entity_a, entity_b, relation_type);
        self.db.put_cf(
            self.entity_pair_index_cf(),
            key.as_bytes(),
            edge_uuid.as_bytes(),
        )?;
        Ok(())
    }

    /// Remove entity pair + relation type from the pair index
    fn remove_entity_pair_index(
        &self,
        entity_a: &Uuid,
        entity_b: &Uuid,
        relation_type: &RelationType,
    ) -> Result<()> {
        let key = Self::typed_pair_key(entity_a, entity_b, relation_type);
        self.db
            .delete_cf(self.entity_pair_index_cf(), key.as_bytes())?;
        Ok(())
    }

    /// Find existing relationship between two entities with a specific relation type.
    ///
    /// O(1) lookup via typed pair index, with fallback to linear scan for pre-index edges.
    /// Matches only edges with the same `RelationType`, allowing multiple semantically
    /// distinct edges (e.g. WorksWith + PartOf) between the same entity pair.
    pub fn find_relationship_between_typed(
        &self,
        entity_a: &Uuid,
        entity_b: &Uuid,
        relation_type: &RelationType,
    ) -> Result<Option<RelationshipEdge>> {
        // Fast path: direct typed key lookup
        let key = Self::typed_pair_key(entity_a, entity_b, relation_type);
        if let Some(edge_uuid_bytes) = self
            .db
            .get_cf(self.entity_pair_index_cf(), key.as_bytes())?
        {
            if edge_uuid_bytes.len() == 16 {
                let edge_uuid = Uuid::from_slice(&edge_uuid_bytes)?;
                if let Some(edge) = self.get_relationship(&edge_uuid)? {
                    return Ok(Some(edge));
                }
                // Stale index entry — clean up
                let _ = self
                    .db
                    .delete_cf(self.entity_pair_index_cf(), key.as_bytes());
            }
        }

        // Slow path: linear scan for pre-index edges
        let edges = self.get_entity_relationships(entity_a)?;
        for edge in edges {
            if edge.relation_type == *relation_type
                && ((edge.from_entity == *entity_a && edge.to_entity == *entity_b)
                    || (edge.from_entity == *entity_b && edge.to_entity == *entity_a))
            {
                // Backfill typed pair index
                let _ = self.index_entity_pair(entity_a, entity_b, &edge.uuid, relation_type);
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
        // Check for existing relationship between these entities WITH SAME TYPE
        // Different relation types (e.g. WorksWith vs PartOf) are distinct edges
        if let Some(mut existing) = self.find_relationship_between_typed(
            &edge.from_entity,
            &edge.to_entity,
            &edge.relation_type,
        )? {
            // #8 DIAGNOSTIC (default-safe, log-only): the typed pair key is
            // order-independent (pair_key sorts min/max UUID), so an incoming
            // CAUSAL attestation whose direction is the REVERSE of the stored
            // edge collapses into it silently — the stored cause→effect arrow
            // is never flipped. Count how often that actually happens for causal
            // relations: if it's frequent, the order-independent key is losing
            // real direction and a direction-sensitive key + reindex is justified;
            // if near-zero (expected, given the effect-first extractor already
            // fixes most mis-direction), the migration isn't worth its cost.
            // grep eval logs for "directed reverse-collapse".
            if edge.relation_type.is_causal()
                && existing.from_entity == edge.to_entity
                && existing.to_entity == edge.from_entity
            {
                DIRECTED_REVERSE_COLLAPSE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tracing::info!(
                    target: "shodh::provenance",
                    relation = ?edge.relation_type,
                    "directed reverse-collapse: incoming {}->{} merged into stored {}->{}",
                    edge.from_entity,
                    edge.to_entity,
                    existing.from_entity,
                    existing.to_entity
                );
            }
            // Temporal anomaly (B4): a long-dormant edge being re-attested is a
            // pattern reactivation. Measured BEFORE strengthen()/last_activated
            // overwrite the gap; queued for the ingest path to drain + emit.
            let now = Utc::now();
            let gap_days = now
                .signed_duration_since(existing.last_activated)
                .num_seconds() as f32
                / 86_400.0;
            if gap_days >= DORMANT_REACTIVATION_MIN_DAYS {
                let mut queue = self.pending_temporal_anomalies.lock();
                if queue.len() < TEMPORAL_EVENT_QUEUE_CAP {
                    queue.push(TemporalAnomalyEvent {
                        kind: TemporalAnomalyKind::DormantReactivation,
                        edge_uuid: existing.uuid,
                        from_entity: existing.from_entity,
                        to_entity: existing.to_entity,
                        relation_type: existing.relation_type.clone(),
                        gap_days,
                        detected_at: now,
                    });
                }
            }

            // Provenance capture (Increment 1, bug fix): the prior implementation
            // strengthened the synapse but DISCARDED the new attestation's source
            // episode. Merge the incoming attestation into the trail so every
            // source episode that wires this edge is recorded — not just the last.
            //
            // ORDERING IS LOAD-BEARING: this merge runs BEFORE `strengthen()`.
            // Promotion is gated on the number of distinct attesting episodes
            // (`EdgeTier::promotion_min_episodes`), and `strengthen()` ends by
            // calling `try_promote_at`. Merging afterwards evaluated the gate
            // against a trail that was always one attestation stale, so the
            // episode that *earned* a promotion could never be the one that
            // triggered it — on a batch ingest, where an edge's last attestation
            // is usually its last write ever, that lost the promotion outright.
            if edge.provenance.is_empty() {
                // Synthesize one attestation from the incoming edge's source
                // episode. With no source episode there is nothing to attest, so
                // we skip rather than record a nil-UUID record.
                if let Some(source_episode_id) = edge.source_episode_id {
                    merge_provenance(
                        &mut existing.provenance,
                        ProvenanceRecord {
                            source_episode_id,
                            mention_count: 1,
                            first_observed: now,
                            last_observed: now,
                            confidence: edge.entity_confidence,
                            evidence_span: None,
                            typed_by: None,
                        },
                    );
                }
            } else {
                for record in edge.provenance.drain(..) {
                    merge_provenance(&mut existing.provenance, record);
                }
            }

            // Strengthen existing edge instead of creating duplicate. This also
            // runs the promotion gate, now against the up-to-date trail.
            let _ = existing.strengthen_at(now);
            existing.last_activated = now;

            // Update context if new context is more informative
            if edge.context.len() > existing.context.len() {
                existing.context = edge.context;
            }

            // Persist the strengthened edge
            let key = existing.uuid.as_bytes();
            let value = crate::serialization::encode(&existing)?;
            self.db.put_cf(self.relationships_cf(), key, value)?;

            return Ok(existing.uuid);
        }

        // No existing edge - create new one
        edge.uuid = Uuid::new_v4();
        edge.created_at = Utc::now();

        // Provenance capture (Increment 1): seed the trail from this edge's
        // source episode if the caller did not already provide a richer seed.
        // Enforce the cap regardless so a caller-seeded trail can't exceed it.
        if edge.provenance.is_empty() {
            if let Some(source_episode_id) = edge.source_episode_id {
                let now = edge.created_at;
                merge_provenance(
                    &mut edge.provenance,
                    ProvenanceRecord {
                        source_episode_id,
                        mention_count: 1,
                        first_observed: now,
                        last_observed: now,
                        confidence: edge.entity_confidence,
                        evidence_span: None,
                        typed_by: None,
                    },
                );
            }
        } else {
            // A caller-seeded trail goes through `merge_provenance` too, rather
            // than being capped in place. `merge_provenance` is the ONLY writer
            // that maintains the trail's defining invariant — one record per
            // distinct `source_episode_id` — and promotion now reads
            // `provenance.len()` as a count of DISTINCT attesting episodes
            // (see `RelationshipEdge::distinct_attesting_episodes`). The in-place
            // sort+truncate that used to live here deduplicated nothing, so a
            // caller that seeded the same episode twice (e.g. a `SemanticFact`
            // whose `source_memories` repeats a memory id) would have inflated
            // its own corroboration count. It also applies the same
            // keep-the-strongest cap, so nothing is lost by routing through it.
            let seeded = std::mem::take(&mut edge.provenance);
            for record in seeded {
                merge_provenance(&mut edge.provenance, record);
            }
        }

        // Reachability: an edge can be BORN corroborated. The SemanticFact path
        // (`MemorySystem::connect_facts_to_graph`) seeds the trail from
        // `fact.source_memories` — every memory that attested the fact — and
        // mints straight into L2 without ever calling `strengthen`. Under the
        // old code `try_promote_at` was reachable only from `strengthen_scaled_at`,
        // so such an edge could carry full corroboration from birth and still
        // never be considered for its tier unless something happened to
        // re-strengthen it later.
        //
        // This cannot manufacture a tier on its own: birth strength is
        // `EdgeTier::{L1,L2}::initial_weight()` scaled by semantic similarity
        // (≤ 0.4 and ≤ 0.5 respectively), both strictly below the corresponding
        // promotion thresholds (0.5 and 0.7), so the strength condition in
        // `try_promote_at` rejects every currently-minted edge. It is here so
        // that the invariant "promotion is evaluated wherever the evidence
        // changes" holds at the choke point rather than by coincidence.
        let born_at = edge.created_at;
        let _ = edge.reconsider_promotion_at(born_at);

        // Store relationship
        let key = edge.uuid.as_bytes();
        let value = crate::serialization::encode(&edge)?;
        self.db.put_cf(self.relationships_cf(), key, value)?;

        // Increment relationship counter
        self.relationship_count.fetch_add(1, Ordering::Relaxed);

        // Update entity->edges index for both entities
        self.index_entity_edge(&edge.from_entity, &edge.uuid)?;
        self.index_entity_edge(&edge.to_entity, &edge.uuid)?;

        // Update entity-pair index for O(1) dedup lookups (typed key supports multi-edge)
        self.index_entity_pair(
            &edge.from_entity,
            &edge.to_entity,
            &edge.uuid,
            &edge.relation_type,
        )?;

        // Insert-time degree pruning: cap edges per entity to prevent O(n²) explosion.
        // If either entity exceeds MAX_ENTITY_DEGREE, prune the weakest edges.
        // This is the primary defense against graph bloat (132MB for 600KB of content).
        self.prune_entity_if_over_degree(&edge.from_entity)?;
        self.prune_entity_if_over_degree(&edge.to_entity)?;

        Ok(edge.uuid)
    }

    /// Drain the temporal anomalies queued by the strengthen path (dormant
    /// reactivations). Called by the ingest path after each graph write; the
    /// caller resolves entity names and emits SSE events.
    pub fn drain_temporal_anomalies(&self) -> Vec<TemporalAnomalyEvent> {
        std::mem::take(&mut *self.pending_temporal_anomalies.lock())
    }

    /// RocksDB in-process memory for this graph DB: (memtable bytes,
    /// table-reader bytes), summed over all graph column families. The shared
    /// block cache is deliberately excluded — it is one pool shared by every
    /// DB instance and is reported once at the manager level.
    pub fn rocksdb_memory_breakdown(&self) -> (u64, u64) {
        let mut memtables = 0u64;
        let mut readers = 0u64;
        for cf_name in GRAPH_CF_NAMES {
            if let Some(cf) = self.db.cf_handle(cf_name) {
                if let Ok(Some(v)) = self
                    .db
                    .property_int_value_cf(cf, "rocksdb.cur-size-all-mem-tables")
                {
                    memtables += v;
                }
                if let Ok(Some(v)) = self
                    .db
                    .property_int_value_cf(cf, "rocksdb.estimate-table-readers-mem")
                {
                    readers += v;
                }
            }
        }
        (memtables, readers)
    }

    /// Index an edge for an entity
    fn index_entity_edge(&self, entity_uuid: &Uuid, edge_uuid: &Uuid) -> Result<()> {
        let key = format!("{entity_uuid}:{edge_uuid}");
        self.db
            .put_cf(self.entity_edges_cf(), key.as_bytes(), b"1")?;
        Ok(())
    }

    /// Prune an entity's edges if degree exceeds MAX_ENTITY_DEGREE
    ///
    /// Loads all edges for the entity, sorts by effective strength, and deletes
    /// the weakest edges that exceed the cap. LTP-protected edges are preserved
    /// preferentially (sorted last, so they survive pruning).
    ///
    /// This is called at insert time to prevent unbounded edge growth.
    /// Amortized cost: O(1) for most insertions (only triggers when over cap),
    /// O(d log d) when pruning is needed (d = entity degree).
    fn prune_entity_if_over_degree(&self, entity_uuid: &Uuid) -> Result<()> {
        use crate::constants::MAX_ENTITY_DEGREE;

        // Fast path: count edges without loading them
        let prefix = format!("{entity_uuid}:");
        let iter = self
            .db
            .prefix_iterator_cf(self.entity_edges_cf(), prefix.as_bytes());

        let mut edge_count = 0usize;
        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }
                edge_count += 1;
            }
        }

        if edge_count <= MAX_ENTITY_DEGREE {
            return Ok(());
        }

        // Over cap — load all edges, sort, prune weakest
        let all_edges = self.get_entity_relationships(entity_uuid)?;
        if all_edges.len() <= MAX_ENTITY_DEGREE {
            return Ok(()); // Race condition guard
        }

        // Sort by pruning priority: LTP-protected edges last (survive pruning),
        // then by effective strength descending (strongest survive)
        let mut scored: Vec<(Uuid, f32, bool)> = all_edges
            .iter()
            .map(|e| {
                let is_protected = e.is_potentiated();
                (e.uuid, e.effective_strength(), is_protected)
            })
            .collect();

        // Sort: unprotected+weak first (pruning candidates), protected+strong last (survivors)
        scored.sort_by(|a, b| {
            // Protected edges sort after unprotected
            match a.2.cmp(&b.2) {
                CmpOrdering::Equal => {
                    // Within same protection class, weaker edges first (prune candidates)
                    a.1.total_cmp(&b.1)
                }
                other => other,
            }
        });

        // Prune excess: first N edges in sorted order are weakest/unprotected
        let prune_count = scored.len() - MAX_ENTITY_DEGREE;
        let to_prune: Vec<Uuid> = scored.iter().take(prune_count).map(|s| s.0).collect();

        for edge_uuid in &to_prune {
            if let Err(e) = self.delete_relationship(edge_uuid) {
                tracing::warn!(
                    edge = %edge_uuid,
                    entity = %entity_uuid,
                    "Failed to prune edge during degree cap: {}",
                    e
                );
            }
        }

        if !to_prune.is_empty() {
            tracing::debug!(
                entity = %entity_uuid,
                pruned = to_prune.len(),
                remaining = MAX_ENTITY_DEGREE,
                "Pruned edges exceeding degree cap"
            );
        }

        Ok(())
    }

    /// Get relationships for an entity with optional limit
    ///
    /// Uses batch reading (multi_get) to eliminate N+1 query pattern.
    /// If limit is None, returns all edges (use sparingly for large graphs).
    pub fn get_entity_relationships(&self, entity_uuid: &Uuid) -> Result<Vec<RelationshipEdge>> {
        self.get_entity_relationships_limited(entity_uuid, None)
    }

    /// Get relationships for an entity with limit, ordered by effective strength
    ///
    /// Collects ALL edge UUIDs first, batch-reads them, sorts by effective_strength
    /// descending, then returns the top `limit` strongest edges. This ensures
    /// traversal and queries always use the most valuable connections.
    ///
    /// When no limit is specified, returns all edges sorted by strength.
    pub fn get_entity_relationships_limited(
        &self,
        entity_uuid: &Uuid,
        limit: Option<usize>,
    ) -> Result<Vec<RelationshipEdge>> {
        let prefix = format!("{entity_uuid}:");

        // Phase 1: Collect ALL edge UUIDs from index (fast prefix scan)
        // We must read all to sort by strength — storage order is arbitrary
        let mut edge_uuids: Vec<Uuid> = Vec::with_capacity(256);
        let iter = self
            .db
            .prefix_iterator_cf(self.entity_edges_cf(), prefix.as_bytes());

        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }

                if let Some(edge_uuid_str) = key_str.split(':').nth(1) {
                    if let Ok(edge_uuid) = Uuid::parse_str(edge_uuid_str) {
                        edge_uuids.push(edge_uuid);
                    }
                }
            }
        }

        if edge_uuids.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 2: Batch read all edges using multi_get (single RocksDB call)
        let keys: Vec<[u8; 16]> = edge_uuids.iter().map(|u| *u.as_bytes()).collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();

        let results = self
            .db
            .batched_multi_get_cf(self.relationships_cf(), &key_refs, false);

        let mut edges = Vec::with_capacity(edge_uuids.len());
        for value in results.into_iter().flatten().flatten() {
            if let Ok((edge, _)) = decode_relationship_edge(&value) {
                edges.push(edge);
            }
        }

        // Phase 3: Sort by effective strength descending (strongest first)
        // Snapshot strengths BEFORE sorting — effective_strength() calls Utc::now()
        // internally, so repeated calls during sort can return different values,
        // violating total ordering (Rust 1.81+ panics on this).
        let mut strength_cache: HashMap<Uuid, f32> = HashMap::with_capacity(edges.len());
        for edge in &edges {
            strength_cache.insert(edge.uuid, edge.effective_strength());
        }
        edges.sort_by(|a, b| {
            let sa = strength_cache.get(&a.uuid).copied().unwrap_or(0.0);
            let sb = strength_cache.get(&b.uuid).copied().unwrap_or(0.0);
            sb.total_cmp(&sa)
        });

        // Phase 3.1: CROSS-INGEST DETERMINISM. The stable sort above leaves
        // equal-strength edges in Phase-1 prefix-scan order, i.e. edge-UUID
        // lexicographic order — random per ingest, since edge UUIDs are
        // `Uuid::new_v4()`. Freshly built graphs have large exact-strength
        // plateaus (every co-occurrence edge starts at its tier's initial
        // weight), and hub entities carry more edges than callers' caps
        // (locomo-gate: 3 hubs at 185-290 edges vs the PPR per-node cap of
        // 100), so WHICH edges survive Phase 4's `truncate(limit)` — and the
        // summation order of everything downstream — was decided by that
        // per-ingest random order. Two ingests of the same corpus then walk
        // different subgraphs, activations jitter at the ~1% level, and the
        // recall harness's repeat-determinism guard (RH-12) fires on near-tie
        // rank flips (PR #462: conv-42_q1/q51, a Δ≈9e-5 fused pair at the
        // top-10 cutoff).
        //
        // Order equal-strength runs by a key that is a pure function of graph
        // CONTENT, not of ingest randomness: (peer entity name, relation
        // type, edge uuid). Names are repeat-stable (deterministic entity
        // resolution — verified by cross-repeat graph checksums) and
        // `add_relationship` dedups by (from, to, type), so the uuid fallback
        // only orders edges that are content-identical — where either order
        // yields bit-identical downstream sums (equal strengths commute).
        // Peer names are resolved lazily and only inside runs of 2+, so the
        // common no-tie case costs nothing extra.
        {
            let mut peer_names: HashMap<Uuid, String> = HashMap::new();
            let strength_bits = |e: &RelationshipEdge| -> u32 {
                strength_cache
                    .get(&e.uuid)
                    .copied()
                    .unwrap_or(0.0)
                    .to_bits()
            };
            let mut i = 0;
            while i < edges.len() {
                let si = strength_bits(&edges[i]);
                let mut j = i + 1;
                while j < edges.len() && strength_bits(&edges[j]) == si {
                    j += 1;
                }
                if j - i > 1 {
                    for e in &edges[i..j] {
                        let peer = if e.from_entity == *entity_uuid {
                            e.to_entity
                        } else {
                            e.from_entity
                        };
                        peer_names.entry(peer).or_insert_with(|| {
                            self.get_entity(&peer)
                                .ok()
                                .flatten()
                                .map(|ent| ent.name)
                                .unwrap_or_default()
                        });
                    }
                    edges[i..j].sort_by(|a, b| {
                        let pa = if a.from_entity == *entity_uuid {
                            a.to_entity
                        } else {
                            a.from_entity
                        };
                        let pb = if b.from_entity == *entity_uuid {
                            b.to_entity
                        } else {
                            b.from_entity
                        };
                        peer_names[&pa]
                            .cmp(&peer_names[&pb])
                            .then_with(|| {
                                format!("{:?}", a.relation_type)
                                    .cmp(&format!("{:?}", b.relation_type))
                            })
                            .then_with(|| a.uuid.cmp(&b.uuid))
                    });
                }
                i = j;
            }
        }

        // Phase 3.5: Opportunistic pruning — queue edges that have decayed below
        // their tier's threshold for batch deletion on next maintenance cycle.
        // This replaces the eager full-scan apply_decay() with lazy on-read pruning.
        let mut has_prunable = false;
        for edge in &edges {
            if edge.effective_strength() < edge.tier.prune_threshold() && !edge.is_prune_protected()
            {
                has_prunable = true;
                break;
            }
        }
        if has_prunable {
            let mut prune_queue = self.pending_prune.lock();
            let mut orphan_queue = self.pending_orphan_checks.lock();
            // Prevent unbounded growth — these queues are drained by maintenance,
            // but if maintenance hasn't run, cap to avoid increasing lock contention.
            if prune_queue.len() > 1000 {
                tracing::debug!(
                    "Prune queue overflow ({}) — clearing to prevent lock contention",
                    prune_queue.len()
                );
                prune_queue.clear();
            }
            if orphan_queue.len() > 2000 {
                orphan_queue.clear();
            }
            edges.retain(|edge| {
                if edge.effective_strength() < edge.tier.prune_threshold()
                    && !edge.is_prune_protected()
                {
                    prune_queue.push(edge.uuid);
                    orphan_queue.push(edge.from_entity);
                    orphan_queue.push(edge.to_entity);
                    false // remove from results
                } else {
                    true
                }
            });
        }

        // Phase 4: Truncate to limit if specified
        if let Some(max) = limit {
            edges.truncate(max);
        }

        Ok(edges)
    }

    /// Trace the causal origin(s) of a set of effect entities — the "root cause".
    ///
    /// Spreading activation cannot answer "what was the origin of C?": it favours
    /// the PROXIMAL cause (1 hop) over the distal root (2+ hops), and its per-edge
    /// step hardcodes the `to_entity` as the target so it only flows forward
    /// (cause → effect), never backward. This walks the OTHER way: from each effect
    /// it follows incoming causal edges (where the current node is the `to_entity`
    /// of a `Causes`/`Triggers`/`ResultsIn` edge) toward the `from_entity` cause,
    /// repeatedly, and returns the terminal sources — the nodes with no further
    /// causal antecedent. Those are the roots whose episodes answer the query.
    ///
    /// Requires causally-typed edges (see `extract_predicate_from_text` /
    /// `SHODH_GRAPH_EXTRACTED_PREDICATES`); on a pure co-occurrence graph there are
    /// no causal edges to walk and this correctly returns nothing.
    ///
    /// Returns origins SCORED by backward-path strength — the max over paths of the
    /// product of edge `effective_strength` with per-hop decay — sorted strongest
    /// first. The unscored version of this walk returned EVERY terminal source in
    /// the backward cone; on a graph with cross-chain causal bleed that was 21–57
    /// origins per query when exactly 1 was correct (the measured lineage flood),
    /// and the downstream injection amplified all of them. Scores let the caller
    /// take a bounded top-k — the reach_inject lesson (bounded ranked set, never
    /// everything reachable), causal edition. Relaxation re-expands improved nodes
    /// (exact best-path); products of factors < 1 cannot improve around a cycle, so
    /// it terminates.
    pub fn trace_causal_origins(
        &self,
        seeds: &[Uuid],
        max_depth: usize,
    ) -> Result<Vec<(Uuid, f32)>> {
        use std::collections::HashSet;
        const HOP_DECAY: f32 = 0.7;
        const MAX_NODES: usize = 4000;

        let seed_set: HashSet<Uuid> = seeds.iter().copied().collect();
        // Best backward-path strength per node (max-product relaxation).
        let mut best: HashMap<Uuid, f32> = seeds.iter().map(|s| (*s, 1.0_f32)).collect();
        let mut origins: HashMap<Uuid, f32> = HashMap::new();
        let mut frontier: Vec<Uuid> = seeds.to_vec();

        for _ in 0..max_depth.max(1) {
            if frontier.is_empty() || best.len() >= MAX_NODES {
                break;
            }
            let mut next: Vec<Uuid> = Vec::new();
            for node in std::mem::take(&mut frontier) {
                let node_score = best.get(&node).copied().unwrap_or(0.0);
                if node_score <= 0.0 {
                    continue;
                }
                let edges = self.get_entity_relationships_limited(&node, Some(64))?;
                // Incoming causal edges: this node is the EFFECT (to_entity); the
                // cause is the from_entity. Self-loops are skipped.
                let parents: Vec<(Uuid, f32)> = edges
                    .iter()
                    .filter(|e| {
                        e.to_entity == node && e.from_entity != node && e.relation_type.is_causal()
                    })
                    .map(|e| (e.from_entity, e.effective_strength().clamp(0.0, 1.0)))
                    .collect();
                if parents.is_empty() {
                    // No causal antecedent → this is a source. A seed with no causal
                    // parent is the query subject itself, not an origin, so skip it.
                    if !seed_set.contains(&node) {
                        let entry = origins.entry(node).or_insert(0.0);
                        if node_score > *entry {
                            *entry = node_score;
                        }
                    }
                } else {
                    for (parent, strength) in parents {
                        let candidate = node_score * strength * HOP_DECAY;
                        let entry = best.entry(parent).or_insert(0.0);
                        if candidate > *entry {
                            *entry = candidate;
                            next.push(parent);
                        }
                    }
                }
            }
            frontier = next;
        }

        let mut out: Vec<(Uuid, f32)> = origins.into_iter().collect();
        // Strongest first; uuid tiebreak for determinism.
        out.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        Ok(out)
    }

    fn relation_stats_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_RELATION_STATS)
            .expect("relation_stats CF must exist")
    }

    /// Record one piece of high-precision relation evidence — the cue extractor
    /// acting as distant supervisor (Angeli 2015 §5, adapted): every explicit
    /// lexical cue hit teaches the per-user (label-pair → relation, direction)
    /// statistics that `lookup_learned_pair_relation` later applies to cue-less
    /// pairs. `src_label`/`dst_label` are the labels of the relation's source
    /// and target entities respectively.
    pub fn record_relation_evidence(
        &self,
        src_label: &EntityLabel,
        dst_label: &EntityLabel,
        relation: &RelationType,
    ) -> Result<()> {
        // Generic labels never participate (see label_is_generic) — refusing
        // at RECORD time keeps the stats CF small and the signal honest.
        if label_is_generic(src_label) || label_is_generic(dst_label) {
            return Ok(());
        }
        let (la, lb, src_is_a) = canonical_label_pair(src_label, dst_label);
        let key = format!("lp:{la}:{lb}:{}", relation.as_str());
        let mut stat: PairRelationStat = match self.db.get_cf(self.relation_stats_cf(), &key)? {
            Some(bytes) => crate::serialization::try_decode(&bytes)
                .map(|(s, _)| s)
                .unwrap_or_else(|_| PairRelationStat::new(relation.clone())),
            None => PairRelationStat::new(relation.clone()),
        };
        stat.count += 1;
        if src_is_a {
            stat.src_is_a += 1;
        }
        self.db.put_cf(
            self.relation_stats_cf(),
            &key,
            crate::serialization::encode(&stat)?,
        )?;
        self.bump_stat_counter(&format!("lp_total:{la}:{lb}"))?;
        self.bump_stat_counter(&format!("rel_total:{}", relation.as_str()))?;
        self.bump_stat_counter("lp_grand_total")?;
        Ok(())
    }

    fn bump_stat_counter(&self, key: &str) -> Result<()> {
        let n: u64 = match self.db.get_cf(self.relation_stats_cf(), key)? {
            Some(bytes) => crate::serialization::try_decode(&bytes)
                .map(|(v, _)| v)
                .unwrap_or(0),
            None => 0,
        };
        self.db.put_cf(
            self.relation_stats_cf(),
            key,
            crate::serialization::encode(&(n + 1))?,
        )?;
        Ok(())
    }

    fn read_stat_counter(&self, key: &str) -> Result<u64> {
        Ok(match self.db.get_cf(self.relation_stats_cf(), key)? {
            Some(bytes) => crate::serialization::try_decode(&bytes)
                .map(|(v, _)| v)
                .unwrap_or(0),
            None => 0,
        })
    }

    /// Learned label-pair relation for a cue-less pair: the typed default this
    /// user's own cue evidence has EARNED, replacing the hardcoded label-pair
    /// table. Returns (relation, label_a_entity_is_source, support) for the
    /// caller's (label_a, label_b) mention order, or None.
    ///
    /// Gates (each against a measured failure mode; tightened after the first
    /// measurement, batched guard run 27348362950, rejected v1 for
    /// per-application over-generalization):
    /// - NO generic labels (Concept/Keyword/Other) on either endpoint —
    ///   catch-all labels made every co-mention one "pair" (CreatedBy 3→344);
    /// - support ≥ 10 (was 3) — a mapping must be earned by real evidence mass;
    /// - purity ≥ 0.6 — a near-tie between relations stays generic (the
    ///   embedding-argmax platform-divergence lesson: never let a coin flip
    ///   pick semantics);
    /// - PMI² > 0 — the pair must co-occur with the relation MORE than chance
    ///   (Angeli 2015 §5);
    /// - NEVER causal — causal edges demand explicit sentence-level evidence;
    ///   a statistically-defaulted causal edge is lineage poison (the fragment
    ///   bridge class);
    /// - direction by ≥ 0.7 majority for cross-label pairs, mention order
    ///   otherwise.
    /// The caller adds the per-APPLICATION cap (max learned edges per memory).
    pub fn lookup_learned_pair_relation(
        &self,
        label_a: &EntityLabel,
        label_b: &EntityLabel,
    ) -> Result<Option<(RelationType, bool, u64)>> {
        if label_is_generic(label_a) || label_is_generic(label_b) {
            return Ok(None);
        }
        let (la, lb, a_is_canonical_a) = canonical_label_pair(label_a, label_b);
        let pair_total = self.read_stat_counter(&format!("lp_total:{la}:{lb}"))?;
        if pair_total < 10 {
            return Ok(None);
        }
        let prefix = format!("lp:{la}:{lb}:");
        let iter = self
            .db
            .prefix_iterator_cf(self.relation_stats_cf(), prefix.as_bytes());
        let mut best: Option<PairRelationStat> = None;
        for (key, value) in iter.flatten() {
            let Ok(key_str) = std::str::from_utf8(&key) else {
                break;
            };
            if !key_str.starts_with(&prefix) {
                break;
            }
            if let Ok((stat, _)) = crate::serialization::try_decode::<PairRelationStat>(&value) {
                if best.as_ref().map(|b| stat.count > b.count).unwrap_or(true) {
                    best = Some(stat);
                }
            }
        }
        let Some(best) = best else {
            return Ok(None);
        };
        if best.relation.is_causal() {
            return Ok(None);
        }
        let purity = best.count as f64 / pair_total as f64;
        if purity < 0.6 {
            return Ok(None);
        }
        let grand = self.read_stat_counter("lp_grand_total")?.max(1);
        let rel_total = self
            .read_stat_counter(&format!("rel_total:{}", best.relation.as_str()))?
            .max(1);
        // PMI² = log2( c(pair,rel)² · N / (c(pair) · c(rel)) ).
        let pmi2 = ((best.count as f64).powi(2) * grand as f64
            / (pair_total as f64 * rel_total as f64))
            .log2();
        if pmi2 <= 0.0 {
            return Ok(None);
        }
        // Direction: majority vote across the evidence, expressed for the
        // canonical pair, then mapped back to the caller's mention order.
        let a_is_source = if la == lb {
            true // same-label pairs carry no label-level direction signal
        } else {
            let ratio = best.src_is_a as f64 / best.count as f64;
            if ratio >= 0.7 {
                a_is_canonical_a
            } else if ratio <= 0.3 {
                !a_is_canonical_a
            } else {
                return Ok(None); // direction unsettled → stay generic
            }
        };
        Ok(Some((best.relation.clone(), a_is_source, best.count)))
    }

    /// Typed-relation neighbor lookup — the lineage walk's machinery as a
    /// general retrieval primitive (typed-walk retrieval, #67). For each seed,
    /// collect the entities connected by an edge whose relation type is in
    /// `relations`, respecting direction: `incoming=false` follows
    /// seed --R--> neighbor (e.g. Caroline --LocatedIn--> Denver for "where
    /// does Caroline live"); `incoming=true` follows neighbor --R--> seed
    /// (e.g. creator --CreatedBy--> artifact for "who made X"). Scored by
    /// edge `effective_strength`, deduped by max, strongest first — the same
    /// bounded-ranked-set discipline as the causal walk.
    pub fn typed_neighbors(
        &self,
        seeds: &[Uuid],
        relations: &[RelationType],
        incoming: bool,
        max_edges_per_seed: usize,
    ) -> Result<Vec<(Uuid, f32)>> {
        let mut best: HashMap<Uuid, f32> = HashMap::new();
        for seed in seeds {
            let edges = self.get_entity_relationships_limited(seed, Some(max_edges_per_seed))?;
            for edge in &edges {
                if !relations.contains(&edge.relation_type) {
                    continue;
                }
                let neighbor = if incoming {
                    if edge.to_entity != *seed || edge.from_entity == *seed {
                        continue;
                    }
                    edge.from_entity
                } else {
                    if edge.from_entity != *seed || edge.to_entity == *seed {
                        continue;
                    }
                    edge.to_entity
                };
                let score = edge.effective_strength().clamp(0.0, 1.0);
                let entry = best.entry(neighbor).or_insert(0.0);
                if score > *entry {
                    *entry = score;
                }
            }
        }
        let mut out: Vec<(Uuid, f32)> = best.into_iter().collect();
        out.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        Ok(out)
    }

    /// Count edges by relation type — the substrate's typed-fraction scoreboard
    /// (audit 2026-06-10: >80% of the graph was CoOccurs; the typed fraction is
    /// the progress metric for the relation substrate). Full scan; intended for
    /// post-ingest diagnostics, not the query path. Index entries in the CF fail
    /// the format-tag decode and are skipped.
    pub fn relation_type_distribution(&self) -> Result<Vec<(String, usize)>> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let iter = self
            .db
            .iterator_cf(self.relationships_cf(), rocksdb::IteratorMode::Start);
        for (_, value) in iter.flatten() {
            if let Ok((edge, _)) = decode_relationship_edge(&value) {
                *counts
                    .entry(edge.relation_type.as_str().to_string())
                    .or_default() += 1;
            }
        }
        let mut out: Vec<(String, usize)> = counts.into_iter().collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(out)
    }

    /// Calculate edge density for a specific entity (SHO-D5)
    ///
    /// Returns the number of edges connected to this entity.
    /// Used for per-entity density calculation: dense entities use vector search,
    /// sparse entities use graph search.
    ///
    /// This is an O(1) prefix count operation.
    pub fn entity_edge_count(&self, entity_uuid: &Uuid) -> Result<usize> {
        let prefix = format!("{entity_uuid}:");
        let iter = self
            .db
            .prefix_iterator_cf(self.entity_edges_cf(), prefix.as_bytes());

        let mut count = 0;
        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }
                count += 1;
            }
        }

        Ok(count)
    }

    /// Calculate average edge density for a set of entities (SHO-D5)
    ///
    /// Returns the mean number of edges per entity for the given UUIDs.
    /// Used to determine optimal retrieval strategy:
    /// - Low density (<5 edges): Trust graph search (sparse, high-signal)
    /// - High density (>20 edges): Trust vector search (dense, noisy)
    ///
    /// Returns None if no entities provided.
    pub fn entities_average_density(&self, entity_uuids: &[Uuid]) -> Result<Option<f32>> {
        if entity_uuids.is_empty() {
            return Ok(None);
        }

        let mut total_edges = 0usize;
        for uuid in entity_uuids {
            total_edges += self.entity_edge_count(uuid)?;
        }

        Ok(Some(total_edges as f32 / entity_uuids.len() as f32))
    }

    /// Get relationship by UUID (raw, without decay applied)
    pub fn get_relationship(&self, uuid: &Uuid) -> Result<Option<RelationshipEdge>> {
        let key = uuid.as_bytes();
        match self.db.get_cf(self.relationships_cf(), key)? {
            Some(value) => {
                let (edge, _) = decode_relationship_edge(&value)?;
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
        match self.db.get_cf(self.relationships_cf(), key)? {
            Some(value) => {
                let (mut edge, _) = decode_relationship_edge(&value)?;
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

        // Get the edge first to find both entities for index cleanup
        let edge = match self.get_relationship(uuid)? {
            Some(e) => e,
            None => return Ok(false),
        };

        // Delete from main storage
        self.db.delete_cf(self.relationships_cf(), key)?;
        // Saturating decrement: never wrap below 0. relationship_count is an
        // accounting estimate that can legitimately drift (some creation paths
        // historically omitted the increment), and a plain fetch_sub would wrap
        // usize to ~1.8e19 once it hits 0.
        let _ = self
            .relationship_count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |c| {
                Some(c.saturating_sub(1))
            });

        // Remove from entity_edges index for BOTH entities
        // (add_relationship indexes both from_entity and to_entity)
        let from_key = format!("{}:{}", edge.from_entity, uuid);
        if let Err(e) = self
            .db
            .delete_cf(self.entity_edges_cf(), from_key.as_bytes())
        {
            tracing::warn!(edge = %uuid, key = %from_key, error = %e, "Failed to delete from entity_edges index");
        }
        let to_key = format!("{}:{}", edge.to_entity, uuid);
        if let Err(e) = self.db.delete_cf(self.entity_edges_cf(), to_key.as_bytes()) {
            tracing::warn!(edge = %uuid, key = %to_key, error = %e, "Failed to delete from entity_edges index");
        }

        // Remove from entity-pair index (typed key)
        if let Err(e) =
            self.remove_entity_pair_index(&edge.from_entity, &edge.to_entity, &edge.relation_type)
        {
            tracing::warn!(edge = %uuid, "Failed to delete from entity_pair index: {}", e);
        }

        Ok(true)
    }

    /// Delete an episode and clean up associated indices and orphan edges
    ///
    /// When a memory is deleted, its corresponding episode should also be removed.
    /// This method:
    /// 1. Removes the episode from the episodes DB
    /// 2. Removes entity_episodes index entries
    /// 3. Deletes relationship edges that were sourced from this episode
    pub fn delete_episode(&self, episode_uuid: &Uuid) -> Result<bool> {
        // Fetch episode to get entity_refs for index cleanup
        let episode = match self.get_episode(episode_uuid)? {
            Some(ep) => ep,
            None => return Ok(false),
        };

        // Delete episode from main storage
        self.db
            .delete_cf(self.episodes_cf(), episode_uuid.as_bytes())?;
        self.episode_count.fetch_sub(1, Ordering::Relaxed);

        // Remove from entity_episodes inverted index
        for entity_uuid in &episode.entity_refs {
            let key = format!("{entity_uuid}:{episode_uuid}");
            if let Err(e) = self.db.delete_cf(self.entity_episodes_cf(), key.as_bytes()) {
                tracing::warn!(episode = %episode_uuid, key = %key, error = %e, "Failed to delete from entity_episodes index");
            }
        }

        // Scrub this episode from every edge's attestation trail (multi-source aware).
        //
        // An edge survives iff at least one attesting episode remains after scrubbing.
        // The deleted episode is removed from the provenance trail; if it was the
        // edge's primary `source_episode_id` but other sources still attest the edge,
        // primacy is promoted to a surviving source. Only edges left with NO remaining
        // attestation are deleted. This avoids both over-deletion (nuking a
        // well-corroborated edge because its first source was removed) and dangling
        // trails (provenance still pointing at a now-deleted episode).
        //
        // Edges never sourced from this episode (source_episode_id is a different
        // episode or None, and the episode is absent from the trail) are untouched —
        // including memory↔memory co-retrieval edges, which carry no episode
        // attestation by design.
        let iter = self
            .db
            .iterator_cf(self.relationships_cf(), rocksdb::IteratorMode::Start);
        let mut edges_to_delete = Vec::new();
        let mut edges_to_update: Vec<RelationshipEdge> = Vec::new();
        for (_, value) in iter.flatten() {
            let mut edge = match decode_relationship_edge(&value) {
                Ok((edge, _)) => edge,
                Err(_) => continue,
            };
            let was_primary_source = edge.source_episode_id == Some(*episode_uuid);
            let in_trail = edge
                .provenance
                .iter()
                .any(|p| p.source_episode_id == *episode_uuid);
            if !was_primary_source && !in_trail {
                continue;
            }
            edge.provenance
                .retain(|p| p.source_episode_id != *episode_uuid);
            if was_primary_source {
                // Promote a surviving attesting episode to primary source, if any.
                edge.source_episode_id = edge.provenance.first().map(|p| p.source_episode_id);
            }
            let still_attested = edge.source_episode_id.is_some() || !edge.provenance.is_empty();
            if still_attested {
                edges_to_update.push(edge);
            } else {
                edges_to_delete.push(edge.uuid);
            }
        }

        // Persist scrubbed survivors. Endpoints and relation_type are unchanged, so
        // no edge index (pair-key / entity-edge) needs updating — only the payload.
        for edge in &edges_to_update {
            match crate::serialization::encode(edge) {
                Ok(encoded) => {
                    if let Err(e) =
                        self.db
                            .put_cf(self.relationships_cf(), edge.uuid.as_bytes(), encoded)
                    {
                        tracing::debug!("Failed to persist scrubbed edge {}: {}", edge.uuid, e);
                    }
                }
                Err(e) => tracing::debug!("Failed to encode scrubbed edge {}: {}", edge.uuid, e),
            }
        }

        for edge_uuid in &edges_to_delete {
            if let Err(e) = self.delete_relationship(edge_uuid) {
                tracing::debug!("Failed to delete orphan edge {}: {}", edge_uuid, e);
            }
        }

        tracing::debug!(
            "Deleted episode {} with {} entity_refs; {} edges scrubbed, {} orphan edges removed",
            &episode_uuid.to_string()[..8],
            episode.entity_refs.len(),
            edges_to_update.len(),
            edges_to_delete.len()
        );

        Ok(true)
    }

    /// Clear all graph data (GDPR full erasure)
    ///
    /// Wipes all entities, relationships, episodes, and all indices.
    /// Resets all counters to zero.
    /// Returns (entity_count, relationship_count, episode_count) that were cleared.
    pub fn clear_all(&self) -> Result<(usize, usize, usize)> {
        let entity_count = self.entity_count.load(Ordering::Relaxed);
        let relationship_count = self.relationship_count.load(Ordering::Relaxed);
        let episode_count = self.episode_count.load(Ordering::Relaxed);

        // Clear each column family by iterating and batch-deleting
        for cf_name in GRAPH_CF_NAMES {
            let cf = self
                .db
                .cf_handle(cf_name)
                .ok_or_else(|| anyhow::anyhow!("CF '{}' not found during clear_all", cf_name))?;
            let mut batch = rocksdb::WriteBatch::default();
            let iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);
            for (key, _) in iter.flatten() {
                batch.delete_cf(cf, &key);
            }
            self.db.write(batch)?;
        }

        // Clear in-memory indices
        self.entity_name_index.write().clear();
        self.entity_lowercase_index.write().clear();
        self.entity_stemmed_index.write().clear();
        self.entity_alias_index.write().clear();

        // Reset counters
        self.entity_count.store(0, Ordering::Relaxed);
        self.relationship_count.store(0, Ordering::Relaxed);
        self.episode_count.store(0, Ordering::Relaxed);

        // Drain pending maintenance queues — they reference now-deleted entities/edges
        let _ = std::mem::take(&mut *self.pending_prune.lock());
        let _ = std::mem::take(&mut *self.pending_orphan_checks.lock());
        let _ = std::mem::take(&mut *self.pending_temporal_anomalies.lock());

        tracing::info!(
            "Graph data cleared (GDPR erasure): {} entities, {} relationships, {} episodes",
            entity_count,
            relationship_count,
            episode_count
        );
        Ok((entity_count, relationship_count, episode_count))
    }

    /// Add an episodic node
    pub fn add_episode(&self, episode: EpisodicNode) -> Result<Uuid> {
        let key = episode.uuid.as_bytes();
        let entity_count = episode.entity_refs.len();
        tracing::debug!(
            "add_episode: {} with {} entity_refs",
            &episode.uuid.to_string()[..8],
            entity_count
        );

        let value = crate::serialization::encode(&episode)?;
        let already_existed = self.db.get_cf(self.episodes_cf(), key)?.is_some();
        self.db.put_cf(self.episodes_cf(), key, value)?;

        // Only increment counter for genuinely new episodes (not overwrites from retries)
        if !already_existed {
            let prev = self.episode_count.fetch_add(1, Ordering::Relaxed);
            tracing::debug!("add_episode: count {} -> {}", prev, prev + 1);
        }

        // Update inverted index: entity_uuid -> episode_uuid
        for entity_uuid in &episode.entity_refs {
            self.index_entity_episode(entity_uuid, &episode.uuid)?;
        }

        Ok(episode.uuid)
    }

    /// Index an episode for an entity (inverted index)
    fn index_entity_episode(&self, entity_uuid: &Uuid, episode_uuid: &Uuid) -> Result<()> {
        let key = format!("{entity_uuid}:{episode_uuid}");
        self.db
            .put_cf(self.entity_episodes_cf(), key.as_bytes(), b"1")?;
        Ok(())
    }

    /// Get episode by UUID
    pub fn get_episode(&self, uuid: &Uuid) -> Result<Option<EpisodicNode>> {
        let key = uuid.as_bytes();
        match self.db.get_cf(self.episodes_cf(), key)? {
            Some(value) => {
                let (episode, _) = crate::serialization::try_decode::<EpisodicNode>(&value)?;
                Ok(Some(episode))
            }
            None => Ok(None),
        }
    }

    /// Get all episodes that contain a specific entity
    ///
    /// Uses inverted index for O(k) lookup instead of O(n) full scan.
    /// Collects episode UUIDs first, then batch-reads them using multi_get.
    /// Crucial for spreading activation algorithm.
    pub fn get_episodes_by_entity(&self, entity_uuid: &Uuid) -> Result<Vec<EpisodicNode>> {
        let prefix = format!("{entity_uuid}:");
        tracing::debug!("get_episodes_by_entity: prefix {}", &prefix[..12]);

        // Phase 1: Collect episode UUIDs from index (fast prefix scan, no data transfer)
        let mut episode_uuids: Vec<Uuid> = Vec::new();
        let iter = self
            .db
            .prefix_iterator_cf(self.entity_episodes_cf(), prefix.as_bytes());
        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }
                if let Some(episode_uuid_str) = key_str.split(':').nth(1) {
                    if let Ok(episode_uuid) = Uuid::parse_str(episode_uuid_str) {
                        episode_uuids.push(episode_uuid);
                    }
                }
            }
        }

        if episode_uuids.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 2: Batch read all episodes using multi_get (single RocksDB call)
        let keys: Vec<[u8; 16]> = episode_uuids.iter().map(|u| *u.as_bytes()).collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();

        let results = self
            .db
            .batched_multi_get_cf(self.episodes_cf(), &key_refs, false);

        let mut episodes = Vec::with_capacity(episode_uuids.len());
        for value in results.into_iter().flatten().flatten() {
            if let Ok((episode, _)) = crate::serialization::try_decode::<EpisodicNode>(&value) {
                episodes.push(episode);
            }
        }

        tracing::debug!("get_episodes_by_entity: found {} episodes", episodes.len());
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
    ///
    /// Performance: Uses batch edge reading and limits to handle large graphs.
    pub fn traverse_from_entity(
        &self,
        start_uuid: &Uuid,
        max_depth: usize,
    ) -> Result<GraphTraversal> {
        self.traverse_from_entity_filtered(start_uuid, max_depth, None)
    }

    /// BFS graph traversal with optional minimum edge strength filter.
    ///
    /// When `min_strength` is Some, edges below the threshold are skipped
    /// during traversal and NOT Hebbianly strengthened (prevents ghost edge revival).
    pub fn traverse_from_entity_filtered(
        &self,
        start_uuid: &Uuid,
        max_depth: usize,
        min_strength: Option<f32>,
    ) -> Result<GraphTraversal> {
        // Performance limits
        const MAX_ENTITIES: usize = 200;
        const MAX_EDGES_PER_NODE: usize = 100;

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
            // Early termination if we have enough entities
            if all_entities.len() >= MAX_ENTITIES {
                break;
            }

            let mut next_level = Vec::new();

            for (entity_uuid, _hop) in &current_level {
                // Use limited edge reading
                let edges =
                    self.get_entity_relationships_limited(entity_uuid, Some(MAX_EDGES_PER_NODE))?;

                for edge in edges {
                    if visited_edges.contains(&edge.uuid) {
                        continue;
                    }

                    visited_edges.insert(edge.uuid);

                    // Only traverse non-invalidated edges
                    if edge.invalidated_at.is_some() {
                        continue;
                    }

                    // Compute effective strength (lazy decay calculation)
                    let effective = edge.effective_strength();

                    // Skip weak edges if min_strength filter is set
                    if let Some(threshold) = min_strength {
                        if effective < threshold {
                            continue;
                        }
                    }

                    // Collect edge UUID for Hebbian strengthening (only for traversed edges)
                    edges_to_strengthen.push(edge.uuid);

                    // Return edge with effective strength
                    let mut edge_with_decay = edge.clone();
                    edge_with_decay.strength = effective;
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

    /// Subgraph pattern matching (Cypher-like MATCH patterns)
    ///
    /// Pattern format: Vec of (relation_type, direction) tuples
    /// Direction: true = outgoing (a->b), false = incoming (a<-b)
    ///
    /// Example: MATCH (a)-[:WORKS_AT]->(b)-[:LOCATED_IN]->(c)
    /// Pattern: vec![(WorksAt, true), (LocatedIn, true)]
    ///
    /// Returns all entities that match the pattern starting from start_uuid.
    #[cfg(test)]
    pub fn match_pattern(
        &self,
        start_uuid: &Uuid,
        pattern: &[(RelationType, bool)], // (relation_type, is_outgoing)
        min_strength: f32,
    ) -> Result<Vec<Vec<TraversedEntity>>> {
        let mut matches: Vec<Vec<TraversedEntity>> = Vec::new();

        // Start entity
        let start_entity = match self.get_entity(start_uuid)? {
            Some(e) => e,
            None => return Ok(matches),
        };

        // DFS backtracking search
        let mut path: Vec<TraversedEntity> = vec![TraversedEntity {
            entity: start_entity,
            hop_distance: 0,
            decay_factor: 1.0,
        }];

        self.match_pattern_recursive(
            *start_uuid,
            pattern,
            0,
            min_strength,
            1.0,
            &mut path,
            &mut matches,
        )?;

        tracing::debug!(
            "match_pattern: found {} matches for {}-step pattern",
            matches.len(),
            pattern.len()
        );

        Ok(matches)
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn match_pattern_recursive(
        &self,
        current_uuid: Uuid,
        pattern: &[(RelationType, bool)],
        pattern_idx: usize,
        min_strength: f32,
        path_score: f32,
        path: &mut Vec<TraversedEntity>,
        matches: &mut Vec<Vec<TraversedEntity>>,
    ) -> Result<()> {
        // Base case: completed the pattern
        if pattern_idx >= pattern.len() {
            matches.push(path.clone());
            return Ok(());
        }

        const MAX_EDGES_PER_NODE: usize = 100;
        let (required_type, is_outgoing) = &pattern[pattern_idx];
        let edges =
            self.get_entity_relationships_limited(&current_uuid, Some(MAX_EDGES_PER_NODE))?;

        for edge in edges {
            if edge.invalidated_at.is_some() {
                continue;
            }

            // Check relationship type
            if edge.relation_type != *required_type {
                continue;
            }

            // Check direction
            let (next_uuid, direction_matches) = if *is_outgoing {
                // Looking for current -> next
                if edge.from_entity == current_uuid {
                    (edge.to_entity, true)
                } else {
                    (edge.from_entity, false) // Wrong direction
                }
            } else {
                // Looking for current <- next (incoming)
                if edge.to_entity == current_uuid {
                    (edge.from_entity, true)
                } else {
                    (edge.to_entity, false) // Wrong direction
                }
            };

            if !direction_matches {
                continue;
            }

            // Check strength
            let effective = edge.effective_strength();
            if effective < min_strength {
                continue;
            }

            // Avoid cycles in pattern
            if path.iter().any(|te| te.entity.uuid == next_uuid) {
                continue;
            }

            // Add to path and recurse
            if let Some(entity) = self.get_entity(&next_uuid)? {
                let new_score = path_score * effective;
                path.push(TraversedEntity {
                    entity,
                    hop_distance: pattern_idx + 1,
                    decay_factor: new_score,
                });

                self.match_pattern_recursive(
                    next_uuid,
                    pattern,
                    pattern_idx + 1,
                    min_strength,
                    new_score,
                    path,
                    matches,
                )?;

                path.pop();
            }
        }

        Ok(())
    }

    /// Find entities matching a pattern from any starting point
    ///
    /// Scans all entities and finds those that match the given pattern.
    /// More expensive than match_pattern but doesn't require a known start.
    ///
    /// Pattern: Vec of (relation_type, is_outgoing) tuples
    /// Returns: All complete pattern matches with their paths.
    #[cfg(test)]
    pub fn find_pattern_matches(
        &self,
        pattern: &[(RelationType, bool)],
        min_strength: f32,
        limit: usize,
    ) -> Result<Vec<Vec<TraversedEntity>>> {
        let mut all_matches: Vec<Vec<TraversedEntity>> = Vec::new();

        // Iterate through all entities as potential starting points
        let iter = self
            .db
            .iterator_cf(self.entities_cf(), rocksdb::IteratorMode::Start);
        for result in iter {
            if all_matches.len() >= limit {
                break;
            }

            let (_, value) = result?;
            let (entity, _) = decode_entity_node(&value)?;

            let entity_matches = self.match_pattern(&entity.uuid, pattern, min_strength)?;
            for m in entity_matches {
                if all_matches.len() >= limit {
                    break;
                }
                all_matches.push(m);
            }
        }

        tracing::debug!(
            "find_pattern_matches: {} total matches (limit={})",
            all_matches.len(),
            limit
        );

        Ok(all_matches)
    }

    /// Invalidate a relationship (temporal edge invalidation)
    ///
    /// Guarded by synapse_update_lock to prevent race with strengthen/decay.
    pub fn invalidate_relationship(&self, edge_uuid: &Uuid) -> Result<()> {
        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!("synapse_update_lock timeout in invalidate_relationship")
            })?;

        if let Some(mut edge) = self.get_relationship(edge_uuid)? {
            edge.invalidated_at = Some(Utc::now());

            let key = edge.uuid.as_bytes();
            let value = crate::serialization::encode(&edge)?;
            self.db.put_cf(self.relationships_cf(), key, value)?;
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
        // Lock with timeout to prevent deadlock on panic
        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| anyhow::anyhow!("synapse_update_lock timeout in strengthen_synapse"))?;

        if let Some(mut edge) = self.get_relationship(edge_uuid)? {
            let _ = edge.strengthen();

            let key = edge.uuid.as_bytes();
            let value = crate::serialization::encode(&edge)?;
            self.db.put_cf(self.relationships_cf(), key, value)?;
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

        // Single lock acquisition for entire batch, with timeout
        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!("synapse_update_lock timeout in batch_strengthen_synapses")
            })?;

        // Batch read all edges in a single RocksDB call (same pattern as get_entity_relationships_limited)
        let keys: Vec<[u8; 16]> = edge_uuids.iter().map(|u| *u.as_bytes()).collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
        let results = self
            .db
            .batched_multi_get_cf(self.relationships_cf(), &key_refs, false);

        let mut batch = WriteBatch::default();
        let mut strengthened = 0;

        for (i, result) in results.into_iter().enumerate() {
            if let Ok(Some(value)) = result {
                if let Ok((mut edge, _)) = decode_relationship_edge(&value) {
                    let _ = edge.strengthen();
                    match crate::serialization::encode(&edge) {
                        Ok(encoded) => {
                            batch.put_cf(self.relationships_cf(), keys[i], encoded);
                            strengthened += 1;
                        }
                        Err(e) => {
                            tracing::debug!("Failed to serialize edge {}: {}", edge_uuids[i], e);
                        }
                    }
                }
            }
        }

        // Atomic write of all updates
        if strengthened > 0 {
            self.db.write(batch)?;
        }

        Ok(strengthened)
    }

    /// Importance-gated batch strengthening.
    ///
    /// Same as `batch_strengthen_synapses` but calls `strengthen_with_importance(importance)`
    /// instead of `strengthen()`. The importance value scales the Hebbian boost
    /// from [`STRENGTHEN_IMPORTANCE_FLOOR`, 1.0].
    pub fn batch_strengthen_synapses_with_importance(
        &self,
        edge_uuids: &[Uuid],
        importance: f32,
    ) -> Result<usize> {
        if edge_uuids.is_empty() {
            return Ok(0);
        }

        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "synapse_update_lock timeout in batch_strengthen_synapses_with_importance"
                )
            })?;

        let keys: Vec<[u8; 16]> = edge_uuids.iter().map(|u| *u.as_bytes()).collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
        let results = self
            .db
            .batched_multi_get_cf(self.relationships_cf(), &key_refs, false);

        let mut batch = WriteBatch::default();
        let mut strengthened = 0;

        for (i, result) in results.into_iter().enumerate() {
            if let Ok(Some(value)) = result {
                if let Ok((mut edge, _)) = decode_relationship_edge(&value) {
                    let _ = edge.strengthen_with_importance(importance);
                    match crate::serialization::encode(&edge) {
                        Ok(encoded) => {
                            batch.put_cf(self.relationships_cf(), keys[i], encoded);
                            strengthened += 1;
                        }
                        Err(e) => {
                            tracing::debug!("Failed to serialize edge {}: {}", edge_uuids[i], e);
                        }
                    }
                }
            }
        }

        if strengthened > 0 {
            self.db.write(batch)?;
        }

        Ok(strengthened)
    }

    /// Weaken edges in batch (anti-Hebbian: misleading retrieval feedback).
    ///
    /// Applies multiplicative decay (`1 - decay_factor`) to each edge's strength.
    /// Edges below their tier's prune threshold are queued for lazy removal.
    /// Uses a single RocksDB `WriteBatch` and one lock acquisition.
    ///
    /// Returns the number of edges successfully weakened.
    pub fn batch_weaken_synapses(&self, edge_uuids: &[Uuid], decay_factor: f32) -> Result<usize> {
        if edge_uuids.is_empty() {
            return Ok(0);
        }

        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!("synapse_update_lock timeout in batch_weaken_synapses")
            })?;

        let keys: Vec<[u8; 16]> = edge_uuids.iter().map(|u| *u.as_bytes()).collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
        let results = self
            .db
            .batched_multi_get_cf(self.relationships_cf(), &key_refs, false);

        let mut batch = WriteBatch::default();
        let mut weakened = 0;
        let clamped_decay = decay_factor.clamp(0.0, 1.0);

        for (i, result) in results.into_iter().enumerate() {
            if let Ok(Some(value)) = result {
                if let Ok((mut edge, _)) = decode_relationship_edge(&value) {
                    // Fully potentiated edges are protected from batch weakening
                    if matches!(edge.ltp_status, LtpStatus::Full) {
                        continue;
                    }
                    edge.strength *= 1.0 - clamped_decay;
                    edge.strength = edge.strength.max(crate::constants::LTP_MIN_STRENGTH);
                    // Queue for pruning if below tier threshold. When enabled,
                    // multi-source corroboration shields the edge from this
                    // strength-only decay path (LTP::Full was already skipped above).
                    if edge.strength < edge.tier.prune_threshold()
                        && !edge.corroboration_protected()
                    {
                        self.pending_prune.lock().push(edge.uuid);
                    }
                    match crate::serialization::encode(&edge) {
                        Ok(encoded) => {
                            batch.put_cf(self.relationships_cf(), keys[i], encoded);
                            weakened += 1;
                        }
                        Err(e) => {
                            tracing::debug!("Failed to serialize edge {}: {}", edge_uuids[i], e);
                        }
                    }
                }
            }
        }

        if weakened > 0 {
            self.db.write(batch)?;
        }

        Ok(weakened)
    }

    /// Collect edge UUIDs for a set of memory UUIDs via their episodic entity refs.
    ///
    /// For each memory, looks up its EpisodicNode to find entity_refs, then collects
    /// all edges incident to those entities (capped at `max_edges`).
    /// Used by feedback-driven Hebbian reinforcement.
    pub fn collect_entity_edges_for_memories(
        &self,
        memory_uuids: &[Uuid],
        max_edges: usize,
    ) -> Result<Vec<Uuid>> {
        use std::collections::HashSet;

        let mut entity_set: HashSet<Uuid> = HashSet::new();

        // Phase 1: gather all entities referenced by these memories' episodes
        for mem_uuid in memory_uuids.iter().take(20) {
            if let Some(episode) = self.get_episode(mem_uuid)? {
                for entity_ref in &episode.entity_refs {
                    entity_set.insert(*entity_ref);
                }
            }
        }

        if entity_set.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 2: collect edge UUIDs from entity_edges index (deduplicated)
        let mut edge_set: HashSet<Uuid> = HashSet::new();
        for entity_uuid in &entity_set {
            let prefix = format!("{entity_uuid}:");
            let iter = self
                .db
                .prefix_iterator_cf(self.entity_edges_cf(), prefix.as_bytes());

            for (key, _) in iter.flatten() {
                if edge_set.len() >= max_edges {
                    break;
                }
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if !key_str.starts_with(&prefix) {
                        break;
                    }
                    if let Some(edge_uuid_str) = key_str.split(':').nth(1) {
                        if let Ok(edge_uuid) = Uuid::parse_str(edge_uuid_str) {
                            edge_set.insert(edge_uuid);
                        }
                    }
                }
            }

            if edge_set.len() >= max_edges {
                break;
            }
        }

        Ok(edge_set.into_iter().collect())
    }

    /// Record co-retrieval of memories (Hebbian learning between memories)
    ///
    /// When memories are retrieved together, they form associations.
    /// This creates or strengthens CoRetrieved edges between all pairs of memories.
    ///
    /// Note: Limits to top N memories to avoid O(n²) explosion on large retrievals.
    /// Returns the number of edges created/strengthened.
    pub fn record_memory_coactivation(&self, memory_ids: &[Uuid]) -> Result<usize> {
        // STRENGTHEN-ONLY gate. Read here (not in the hot loop) so the impl is
        // env-free and unit-testable by parameter. Default ON: co-retrieval
        // reinforces existing edges and never mints all-pairs `CoRetrieved` (which
        // was ~80% of the graph and the OOM driver). Disable with
        // SHODH_COACT_STRENGTHEN_ONLY=0 to restore the legacy flood.
        let strengthen_only = std::env::var("SHODH_COACT_STRENGTHEN_ONLY")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(true);
        self.record_memory_coactivation_impl(memory_ids, strengthen_only)
    }

    /// Co-retrieval Hebbian update. `strengthen_only = true`: reinforce edges that
    /// ALREADY exist between co-active memories; do NOT mint a new CoRetrieved edge
    /// for every co-retrieved pair. Un-gated all-pairs creation is the recall-time
    /// flood (measured ~80% of graph edges, bypassing the ingest gates, unbounded
    /// with query volume — the OOM driver). Mirrors schema-gated consolidation:
    /// usage strengthens existing structure, it does not wire every co-activation.
    fn record_memory_coactivation_impl(
        &self,
        memory_ids: &[Uuid],
        strengthen_only: bool,
    ) -> Result<usize> {
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

        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!("synapse_update_lock timeout in record_memory_coactivation")
            })?;
        let mut batch = WriteBatch::default();
        let mut edges_updated = 0;
        let mut new_edges = 0;

        // Process all pairs
        for i in 0..memories_to_process.len() {
            for j in (i + 1)..memories_to_process.len() {
                let mem_a = memories_to_process[i];
                let mem_b = memories_to_process[j];

                // Try to find existing edge between these memories
                let existing_edge = self.find_edge_between_entities(&mem_a, &mem_b)?;

                if let Some(mut edge) = existing_edge {
                    // Strengthen existing edge
                    let _ = edge.strengthen();
                    let key = edge.uuid.as_bytes();
                    if let Ok(value) = crate::serialization::encode(&edge) {
                        batch.put_cf(self.relationships_cf(), key, value);
                        edges_updated += 1;
                    }
                } else if !strengthen_only {
                    // Create new CoRetrieved edge (bidirectional represented as single edge)
                    // Starts in L1 (working memory) with tier-specific initial weight
                    let edge = RelationshipEdge {
                        uuid: Uuid::new_v4(),
                        from_entity: mem_a,
                        to_entity: mem_b,
                        relation_type: RelationType::CoRetrieved,
                        strength: EdgeTier::L1Working.initial_weight(),
                        created_at: Utc::now(),
                        valid_at: Utc::now(),
                        invalidated_at: None,
                        source_episode_id: None,
                        context: String::new(),
                        last_activated: Utc::now(),
                        activation_count: 1,
                        ltp_status: LtpStatus::None,
                        activation_timestamps: None,
                        tier: EdgeTier::L1Working,
                        // PIPE-5: Memory-to-memory edges use default confidence
                        entity_confidence: None,
                        forman_curvature: None,
                        endpoint_selectivity: None,
                        provenance: Vec::new(),
                        promoted_at: None,
                    };

                    let key = edge.uuid.as_bytes();
                    if let Ok(value) = crate::serialization::encode(&edge) {
                        batch.put_cf(self.relationships_cf(), key, value);

                        // Index in mem_edge: for fast pair lookup
                        let idx_key_fwd = format!("mem_edge:{mem_a}:{mem_b}");
                        let idx_key_rev = format!("mem_edge:{mem_b}:{mem_a}");
                        batch.put_cf(
                            self.relationships_cf(),
                            idx_key_fwd.as_bytes(),
                            edge.uuid.as_bytes(),
                        );
                        batch.put_cf(
                            self.relationships_cf(),
                            idx_key_rev.as_bytes(),
                            edge.uuid.as_bytes(),
                        );

                        // Index in entity_edges_cf for graph traversal visibility
                        let ee_key_a = format!("{mem_a}:{}", edge.uuid);
                        let ee_key_b = format!("{mem_b}:{}", edge.uuid);
                        batch.put_cf(self.entity_edges_cf(), ee_key_a.as_bytes(), b"1");
                        batch.put_cf(self.entity_edges_cf(), ee_key_b.as_bytes(), b"1");

                        edges_updated += 1;
                        new_edges += 1;
                    }
                }
            }
        }

        if edges_updated > 0 {
            self.db.write(batch)?;
            // Update relationship counter for newly created edges
            if new_edges > 0 {
                self.relationship_count
                    .fetch_add(new_edges, Ordering::Relaxed);
            }
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
        let idx_key = format!("mem_edge:{entity_a}:{entity_b}");
        if let Some(edge_uuid_bytes) = self
            .db
            .get_cf(self.relationships_cf(), idx_key.as_bytes())?
        {
            if edge_uuid_bytes.len() == 16 {
                let edge_uuid = Uuid::from_slice(&edge_uuid_bytes)?;
                return self.get_relationship(&edge_uuid);
            }
        }

        // Check reverse index
        let idx_key_rev = format!("mem_edge:{entity_b}:{entity_a}");
        if let Some(edge_uuid_bytes) = self
            .db
            .get_cf(self.relationships_cf(), idx_key_rev.as_bytes())?
        {
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
    ///
    /// Returns (count_strengthened, promotion_boosts) where promotion_boosts contains
    /// signals for any edge tier promotions that occurred (Direction 1 coupling).
    pub fn strengthen_memory_edges(
        &self,
        edge_boosts: &[(String, String, f32)],
    ) -> Result<(usize, Vec<crate::memory::types::EdgePromotionBoost>)> {
        use crate::constants::{EDGE_PROMOTION_MEMORY_BOOST_L2, EDGE_PROMOTION_MEMORY_BOOST_L3};

        if edge_boosts.is_empty() {
            return Ok((0, Vec::new()));
        }

        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!("synapse_update_lock timeout in strengthen_edges_from_boosts")
            })?;
        let mut batch = WriteBatch::default();
        let mut strengthened = 0;
        let mut promotion_boosts = Vec::new();

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
                // Strengthen existing edge — capture tier promotion if it occurs
                let promotion = edge.strengthen();
                let key = edge.uuid.as_bytes();
                if let Ok(value) = crate::serialization::encode(&edge) {
                    batch.put_cf(self.relationships_cf(), key, value);
                    strengthened += 1;

                    // If a tier promotion occurred, emit boost signals for both memories
                    if let Some((old_tier, new_tier)) = promotion {
                        let boost = if new_tier.contains("L2") {
                            EDGE_PROMOTION_MEMORY_BOOST_L2
                        } else {
                            EDGE_PROMOTION_MEMORY_BOOST_L3
                        };
                        let entity_name = format!(
                            "{}↔{}",
                            &from_id_str[..8.min(from_id_str.len())],
                            &to_id_str[..8.min(to_id_str.len())]
                        );
                        // Boost both memories involved in the promoted edge
                        promotion_boosts.push(crate::memory::types::EdgePromotionBoost {
                            memory_id: from_id_str.clone(),
                            entity_name: entity_name.clone(),
                            old_tier: old_tier.clone(),
                            new_tier: new_tier.clone(),
                            boost,
                        });
                        promotion_boosts.push(crate::memory::types::EdgePromotionBoost {
                            memory_id: to_id_str.clone(),
                            entity_name,
                            old_tier,
                            new_tier,
                            boost,
                        });
                    }
                }
            }
            // No `else`: the replay path reinforces EXISTING edges only. It used to
            // mint a new `CoRetrieved` edge for every boosted memory pair without an
            // edge — the second all-pairs flood site — REMOVED entirely. Co-retrieval
            // strengthens structure; it does not create it.
        }

        if strengthened > 0 {
            self.db.write(batch)?;

            // Index new replay edges in entity_edges CF so they're visible to
            // traversal and degree-cap enforcement (GQ-11 fix)
            let mut entities_to_prune = Vec::new();
            for (from_id_str, to_id_str, _boost) in edge_boosts {
                let from_uuid = match Uuid::parse_str(from_id_str) {
                    Ok(u) => u,
                    Err(_) => continue,
                };
                let to_uuid = match Uuid::parse_str(to_id_str) {
                    Ok(u) => u,
                    Err(_) => continue,
                };
                // Only index edges that we actually wrote (find_edge_between_entities returns
                // the edge if it existed before, so new edges are the ones that didn't exist)
                if let Ok(Some(edge)) = self.find_edge_between_entities(&from_uuid, &to_uuid) {
                    if edge.context == "replay_strengthened" && edge.activation_count <= 1 {
                        if let Err(e) = self.index_entity_edge(&from_uuid, &edge.uuid) {
                            tracing::debug!("Failed to index replay edge for entity: {}", e);
                        }
                        if let Err(e) = self.index_entity_edge(&to_uuid, &edge.uuid) {
                            tracing::debug!("Failed to index replay edge for entity: {}", e);
                        }
                        entities_to_prune.push(from_uuid);
                        entities_to_prune.push(to_uuid);
                    }
                }
            }

            // Enforce degree cap on affected entities
            for entity_uuid in &entities_to_prune {
                let _ = self.prune_entity_if_over_degree(entity_uuid);
            }

            tracing::debug!(
                "Applied {} edge boosts from replay consolidation ({} tier promotions)",
                strengthened,
                promotion_boosts.len()
            );
        }

        Ok((strengthened, promotion_boosts))
    }

    /// Strengthen graph edges between two causally-linked memories.
    ///
    /// Resolves memory UUIDs to their entity_refs via EpisodicNodes, then strengthens
    /// the cross-product of entity pairs. This couples the lineage system (explicit
    /// causal chains) with the knowledge graph (spreading activation), so causally-linked
    /// memories naturally co-activate during retrieval.
    ///
    /// Returns the number of entity-pair edges strengthened.
    pub fn strengthen_lineage_connection(
        &self,
        from_memory_uuid: &Uuid,
        to_memory_uuid: &Uuid,
        boost: f32,
    ) -> Result<usize> {
        // Resolve entity_refs for both memories
        let from_episode = self.get_episode(from_memory_uuid)?;
        let to_episode = self.get_episode(to_memory_uuid)?;

        let (from_entities, to_entities) = match (from_episode, to_episode) {
            (Some(fe), Some(te)) if !fe.entity_refs.is_empty() && !te.entity_refs.is_empty() => {
                (fe.entity_refs, te.entity_refs)
            }
            _ => return Ok(0),
        };

        // Cap entity lists to prevent O(N^2) explosion on heavily-tagged memories.
        // 8 × 8 = 64 pairs max, which is reasonable for a single lineage edge.
        const MAX_ENTITIES_PER_SIDE: usize = 8;
        let from_capped = &from_entities[..from_entities.len().min(MAX_ENTITIES_PER_SIDE)];
        let to_capped = &to_entities[..to_entities.len().min(MAX_ENTITIES_PER_SIDE)];

        // Build cross-product of entity pairs with lineage boost
        let mut edge_boosts: Vec<(String, String, f32)> =
            Vec::with_capacity(from_capped.len() * to_capped.len());
        for from_entity in from_capped {
            for to_entity in to_capped {
                if from_entity != to_entity {
                    edge_boosts.push((from_entity.to_string(), to_entity.to_string(), boost));
                }
            }
        }

        if edge_boosts.is_empty() {
            return Ok(0);
        }

        let (strengthened, _promotions) = self.strengthen_memory_edges(&edge_boosts)?;

        tracing::debug!(
            from_memory = %&from_memory_uuid.to_string()[..8],
            to_memory = %&to_memory_uuid.to_string()[..8],
            entity_pairs = edge_boosts.len(),
            strengthened,
            boost,
            "Lineage→graph edge strengthening"
        );

        Ok(strengthened)
    }

    /// Create typed graph edges between entity pairs of two causally-linked episodes.
    ///
    /// This bridges the lineage namespace into the knowledge graph: lineage edges
    /// (stored in plain RocksDB keys) become typed RelationshipEdge entries visible
    /// to spreading activation. The `CausalRelation::to_graph_relation_type()`
    /// mapping provides the edge type (Causes, Triggers, SupersededBy, etc.).
    ///
    /// `add_relationship()` deduplicates by (from, to, type) — repeat calls
    /// strengthen existing edges rather than creating duplicates.
    pub fn create_lineage_graph_edges(
        &self,
        from_memory_uuid: &Uuid,
        to_memory_uuid: &Uuid,
        relation_type: RelationType,
        confidence: f32,
    ) -> Result<usize> {
        if confidence < crate::constants::LINEAGE_GRAPH_BRIDGE_MIN_CONFIDENCE {
            return Ok(0);
        }

        let from_episode = self.get_episode(from_memory_uuid)?;
        let to_episode = self.get_episode(to_memory_uuid)?;

        let (from_entities, to_entities) = match (from_episode, to_episode) {
            (Some(fe), Some(te)) if !fe.entity_refs.is_empty() && !te.entity_refs.is_empty() => {
                (fe.entity_refs, te.entity_refs)
            }
            _ => return Ok(0),
        };

        const MAX_PER_SIDE: usize = 8;
        let from_capped = &from_entities[..from_entities.len().min(MAX_PER_SIDE)];
        let to_capped = &to_entities[..to_entities.len().min(MAX_PER_SIDE)];

        let now = chrono::Utc::now();
        let base_strength = EdgeTier::L2Episodic.initial_weight()
            * confidence
            * crate::constants::LINEAGE_GRAPH_BRIDGE_BOOST;
        let mut created = 0usize;

        for &from_entity in from_capped {
            for &to_entity in to_capped {
                if from_entity == to_entity {
                    continue;
                }
                let edge = RelationshipEdge {
                    uuid: Uuid::new_v4(),
                    from_entity,
                    to_entity,
                    relation_type: relation_type.clone(),
                    strength: base_strength,
                    created_at: now,
                    valid_at: now,
                    invalidated_at: None,
                    source_episode_id: Some(*from_memory_uuid),
                    context: String::new(),
                    last_activated: now,
                    activation_count: 1,
                    ltp_status: LtpStatus::None,
                    tier: EdgeTier::L2Episodic,
                    activation_timestamps: None,
                    entity_confidence: Some(confidence),
                    forman_curvature: None,
                    endpoint_selectivity: None,
                    // Lineage bridge edge: the source episode is known and the
                    // bridge confidence is meaningful, so seed the attestation
                    // trail here (typed_by left None — no dedicated lineage method).
                    provenance: vec![ProvenanceRecord {
                        source_episode_id: *from_memory_uuid,
                        mention_count: 1,
                        first_observed: now,
                        last_observed: now,
                        confidence: Some(confidence),
                        evidence_span: None,
                        typed_by: None,
                    }],
                    promoted_at: None,
                };
                if self.add_relationship(edge).is_ok() {
                    created += 1;
                }
            }
        }

        if created > 0 {
            tracing::debug!(
                from_memory = %&from_memory_uuid.to_string()[..8],
                to_memory = %&to_memory_uuid.to_string()[..8],
                relation = ?relation_type,
                created,
                "Lineage→graph typed edge creation"
            );
        }

        Ok(created)
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
        let prefix_fwd = format!("mem_edge:{memory_id}:");

        let iter = self
            .db
            .prefix_iterator_cf(self.relationships_cf(), prefix_fwd.as_bytes());
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
        associations.sort_by(|a, b| b.1.total_cmp(&a.1));
        associations.truncate(max_results);

        Ok(associations)
    }

    /// Strengthen entity-entity edges for a replayed memory's episode.
    ///
    /// During consolidation replay, this reinforces the entity relationships that
    /// were involved in the replayed memory. This is "Direction 3" of the Hebbian
    /// maintenance system — entity-entity edges get strengthened alongside
    /// memory-to-memory edges (Direction 1) and lazy pruning (Direction 2).
    ///
    /// Algorithm:
    /// 1. Look up EpisodicNode for episode_id → get entity_refs
    /// 2. For each pair of entities, find their RelationshipEdge
    /// 3. Call strengthen() on each edge (Hebbian boost + LTP detection + tier promotion)
    /// 4. Batch write all updates
    pub fn strengthen_episode_entity_edges(&self, episode_id: &Uuid) -> Result<usize> {
        let episode = match self.get_episode(episode_id) {
            Ok(Some(ep)) => ep,
            Ok(None) => return Ok(0),
            Err(_) => return Ok(0),
        };

        if episode.entity_refs.len() < 2 {
            return Ok(0);
        }

        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!("synapse_update_lock timeout in strengthen_episode_entity_edges")
            })?;
        let mut batch = WriteBatch::default();
        let mut strengthened = 0;

        // Iterate over unique entity pairs
        let refs = &episode.entity_refs;
        let max_pairs = refs.len().min(20); // Cap to avoid O(n²) on large episodes
        for i in 0..max_pairs {
            for j in (i + 1)..max_pairs {
                let entity_a = &refs[i];
                let entity_b = &refs[j];

                // Find existing entity-entity edge via entity_edges_cf index
                // (find_edge_between_entities uses mem_edge: prefix which is memory-to-memory only)
                let edges = match self.get_entity_relationships(entity_a) {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                for mut edge in edges {
                    if (edge.from_entity == *entity_a && edge.to_entity == *entity_b)
                        || (edge.from_entity == *entity_b && edge.to_entity == *entity_a)
                    {
                        if edge.invalidated_at.is_some() {
                            continue;
                        }
                        let _ = edge.strengthen();
                        let key = edge.uuid.as_bytes();
                        if let Ok(value) = crate::serialization::encode(&edge) {
                            batch.put_cf(self.relationships_cf(), key, value);
                            strengthened += 1;
                        }
                        break; // Only strengthen one edge per pair
                    }
                }
            }
        }

        if strengthened > 0 {
            self.db.write(batch)?;
            tracing::debug!(
                "Strengthened {} entity-entity edges for episode {}",
                strengthened,
                &episode_id.to_string()[..8]
            );
        }

        Ok(strengthened)
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

        const MAX_EDGES_PER_ENTITY: usize = 50; // Limit per entity for Hebbian lookup
        for entity_uuid in &episode.entity_refs {
            if let Ok(edges) =
                self.get_entity_relationships_limited(entity_uuid, Some(MAX_EDGES_PER_ENTITY))
            {
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
        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| anyhow::anyhow!("synapse_update_lock timeout in decay_synapse"))?;

        if let Some(mut edge) = self.get_relationship(edge_uuid)? {
            let should_prune = edge.decay();

            let key = edge.uuid.as_bytes();
            let value = crate::serialization::encode(&edge)?;
            self.db.put_cf(self.relationships_cf(), key, value)?;

            return Ok(should_prune);
        }

        Ok(false)
    }

    /// Apply decay to already-loaded edges in-place as of an explicit `now`,
    /// avoiding double deserialization.
    ///
    /// Mutates edges directly, serializes results into a WriteBatch, and returns
    /// the UUIDs of edges that should be pruned. Used by
    /// [`apply_decay_at`](Self::apply_decay_at) (which already has the full edge
    /// list from `get_all_relationships()`); production reaches it via
    /// [`apply_decay`](Self::apply_decay) with `now = Utc::now()`. The injectable
    /// clock lets the decay-evaluation harness age a real graph at the
    /// production cadence without waiting wall-clock time.
    fn batch_decay_edges_in_place_at(
        &self,
        edges: &mut [RelationshipEdge],
        now: DateTime<Utc>,
        protection: Option<&crate::decay::TopologyProtection>,
    ) -> Result<Vec<Uuid>> {
        if edges.is_empty() {
            return Ok(Vec::new());
        }

        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!("synapse_update_lock timeout in batch_decay_edges_in_place_at")
            })?;
        let mut batch = WriteBatch::default();
        // Indices (into `edges`) the BASE (time+usage) gate flagged for pruning.
        // Decisions are deferred so the topology rescue budget can be computed
        // against the full candidate count for this cycle.
        let mut flagged: Vec<usize> = Vec::new();

        for (i, edge) in edges.iter_mut().enumerate() {
            let strength_before = edge.strength;
            let should_prune = edge.decay_at(now);

            // Only write back edges whose strength actually changed (or need pruning).
            // With 300s maintenance intervals, most edges won't have meaningful decay,
            // so this reduces the WriteBatch from ~12MB (all 34k edges) to ~150KB.
            // Rescued edges keep this decayed strength written; the rescue defends
            // EXISTENCE, not strength — a bridge re-evaluated next cycle.
            if should_prune || (edge.strength - strength_before).abs() > f32::EPSILON {
                let key = edge.uuid.as_bytes();
                match crate::serialization::encode(&*edge) {
                    Ok(value) => {
                        batch.put_cf(self.relationships_cf(), key, value);
                        // Flag for prune only on a successful write-back, exactly
                        // as the pre-W1-B code did — an edge that fails to
                        // serialize is neither written nor pruned (byte-identical
                        // to today when protection is None).
                        if should_prune {
                            flagged.push(i);
                        }
                    }
                    Err(e) => {
                        tracing::debug!("Failed to serialize edge {}: {}", edge.uuid, e);
                    }
                }
            }
        }

        // Base prune set = every flagged edge. Topology-aware decay (W1-B) rescues
        // a BUDGETED top slice of the flagged edges whose loss would fragment the
        // graph. With `protection == None` (flag off) this is byte-identical to
        // the pre-W1-B behaviour: every flagged edge is pruned.
        let to_prune = self.select_prune_set(edges, &flagged, protection);

        self.db.write(batch)?;
        Ok(to_prune)
    }

    /// Apply the topology-aware rescue budget to the base prune set.
    ///
    /// `flagged` are the indices the base (time+usage) gate wants to prune this
    /// cycle. When protection is present, edges touching genuine structure
    /// (`edge_protection > TOPOLOGY_RESCUE_MIN_PROTECTION`) are rescue-eligible;
    /// they are ranked by the keep score `strength + α · protection` and the top
    /// `ceil(BUDGET_FRAC · |flagged|)` (≥1 when any qualify) are spared. Everything
    /// else is pruned. Ranking (not an absolute threshold) is the measured choice:
    /// step-1 showed bridge scores are corpus-relative and compressed, so the
    /// budget bounds forgetting-loss while rank picks the genuinely critical edges.
    fn select_prune_set(
        &self,
        edges: &[RelationshipEdge],
        flagged: &[usize],
        protection: Option<&crate::decay::TopologyProtection>,
    ) -> Vec<Uuid> {
        let Some(prot) = protection else {
            return flagged.iter().map(|&i| edges[i].uuid).collect();
        };
        if flagged.is_empty() {
            return Vec::new();
        }

        // Rescue-eligible flagged edges, keyed by keep score (descending rank).
        let mut eligible: Vec<(usize, f32)> = Vec::new();
        for &i in flagged {
            let e = &edges[i];
            let p = prot.edge_protection(&e.from_entity, &e.to_entity);
            if p > crate::constants::TOPOLOGY_RESCUE_MIN_PROTECTION {
                let keep = crate::decay::topology_keep_score(
                    e.strength,
                    p,
                    crate::constants::TOPOLOGY_RESCUE_ALPHA,
                );
                eligible.push((i, keep));
            }
        }

        let budget = ((crate::constants::TOPOLOGY_RESCUE_BUDGET_FRAC * flagged.len() as f32).ceil()
            as usize)
            .max(1)
            .min(eligible.len());

        eligible.sort_by(|a, b| b.1.total_cmp(&a.1));
        let rescued: std::collections::HashSet<usize> =
            eligible.iter().take(budget).map(|(i, _)| *i).collect();

        if !rescued.is_empty() {
            tracing::debug!(
                rescued = rescued.len(),
                flagged = flagged.len(),
                budget,
                "Topology-aware decay: rescued bridge edges from prune"
            );
        }

        flagged
            .iter()
            .filter(|&&i| !rescued.contains(&i))
            .map(|&i| edges[i].uuid)
            .collect()
    }

    /// Compute this cycle's raw topology protection over the active edge set and
    /// smooth it against the stored (previous-cycle) map via hysteresis, updating
    /// the stored map in place. Returns the protection (smoothed node scores +
    /// this cycle's bridge-pair set) used by the prune gate.
    fn compute_and_smooth_topology(
        &self,
        edges: &[RelationshipEdge],
    ) -> crate::decay::TopologyProtection {
        let pairs: Vec<(Uuid, Uuid)> = edges.iter().map(|e| (e.from_entity, e.to_entity)).collect();
        let raw = crate::decay::compute_topology_protection(&pairs);

        let mut guard = self.topology_protection.write();
        let smoothed = crate::decay::smooth_protection(
            &guard,
            &raw.node_protection,
            crate::constants::TOPOLOGY_HYSTERESIS_DECAY,
        );
        *guard = smoothed.clone();

        crate::decay::TopologyProtection {
            node_protection: smoothed,
            bridge_pairs: raw.bridge_pairs,
        }
    }

    /// Apply synaptic homeostasis: global downscaling of all edge strengths.
    ///
    /// After each maintenance cycle, scales ALL edge weights by `factor` (typically 0.95).
    /// Edges with LtpStatus::Full are protected — fully consolidated synapses resist
    /// homeostatic downscaling, matching biological systems consolidation.
    ///
    /// This prevents runaway strengthening and keeps total network energy bounded.
    /// Strong edges survive; weak edges fall below prune thresholds and are cleared
    /// in the next decay cycle.
    ///
    /// Reference: Tononi & Cirelli (2003) "Sleep and synaptic homeostasis: a hypothesis"
    pub fn apply_synaptic_homeostasis(&self, factor: f32) -> Result<usize> {
        let mut all_edges = self.get_all_relationships()?;
        if all_edges.is_empty() {
            return Ok(0);
        }

        let _guard = self
            .synapse_update_lock
            .try_lock_for(std::time::Duration::from_secs(5))
            .ok_or_else(|| {
                anyhow::anyhow!("synapse_update_lock timeout in apply_synaptic_homeostasis")
            })?;

        let mut batch = WriteBatch::default();
        let mut scaled_count = 0;

        for edge in all_edges.iter_mut() {
            // Fully potentiated edges are protected from homeostatic downscaling
            if matches!(edge.ltp_status, LtpStatus::Full) {
                continue;
            }

            let old_strength = edge.strength;
            edge.strength *= factor;
            // Never scale below absolute floor
            edge.strength = edge.strength.max(crate::constants::LTP_MIN_STRENGTH);

            if (edge.strength - old_strength).abs() > f32::EPSILON {
                let key = edge.uuid.as_bytes();
                if let Ok(value) = crate::serialization::encode(&*edge) {
                    batch.put_cf(self.relationships_cf(), key, value);
                    scaled_count += 1;
                }
            }
        }

        if scaled_count > 0 {
            self.db.write(batch)?;
            tracing::debug!(
                "Synaptic homeostasis: scaled {} of {} edges by factor {}",
                scaled_count,
                all_edges.len(),
                factor
            );
        }

        Ok(scaled_count)
    }

    /// Apply decay to all synapses and prune weak edges (AUD-2)
    ///
    /// Called during maintenance cycle to:
    /// 1. Apply time-based decay to all edge strengths
    /// 2. Remove edges that have decayed below threshold
    /// 3. Detect orphaned entities (entities that lost all their edges)
    /// 4. Apply synaptic homeostasis (global edge downscaling)
    ///
    /// Returns a `GraphDecayResult` with pruned count and orphaned entity/memory IDs
    /// for Direction 2 coupling (edge pruning → orphan detection).
    pub fn apply_decay(&self) -> Result<crate::memory::types::GraphDecayResult> {
        self.apply_decay_at(Utc::now())
    }

    /// As [`apply_decay`](Self::apply_decay), but ages edges as of an explicit
    /// `now`. Production calls `apply_decay()` (wall clock). The decay-evaluation
    /// harness calls this repeatedly at the ~6h production cadence to age a real
    /// graph through simulated time — the only faithful way to reproduce the
    /// periodic-decay dynamics, since a single large jump would land directly in
    /// the power-law phase and hide the per-cycle behaviour.
    pub fn apply_decay_at(
        &self,
        now: DateTime<Utc>,
    ) -> Result<crate::memory::types::GraphDecayResult> {
        // Get all edges (need full data for orphan tracking)
        let mut all_edges = self.get_all_relationships()?;

        if all_edges.is_empty() {
            return Ok(crate::memory::types::GraphDecayResult::default());
        }

        // W1-B topology-aware decay (default OFF). When on, compute per-node
        // structural protection ONCE per heavy cycle over the active edge set —
        // this is the "sleep" pass, colocated with the existing full decay, never
        // per-write or at recall time. `None` ⇒ the prune gate is byte-identical
        // to today's time+usage-only decision.
        let topo_on = std::env::var("SHODH_TOPOLOGY_AWARE_DECAY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let protection = if topo_on {
            Some(self.compute_and_smooth_topology(&all_edges))
        } else {
            None
        };

        // Apply decay in-place on already-deserialized edges (avoids double deserialization)
        let to_prune =
            self.batch_decay_edges_in_place_at(&mut all_edges, now, protection.as_ref())?;

        if to_prune.is_empty() {
            return Ok(crate::memory::types::GraphDecayResult::default());
        }

        // Collect entity UUIDs from edges being pruned (candidates for orphan status)
        let pruned_set: std::collections::HashSet<Uuid> = to_prune.iter().copied().collect();
        let mut orphan_candidates: std::collections::HashSet<Uuid> =
            std::collections::HashSet::new();
        for edge in &all_edges {
            if pruned_set.contains(&edge.uuid) {
                orphan_candidates.insert(edge.from_entity);
                orphan_candidates.insert(edge.to_entity);
            }
        }

        // Delete pruned edges
        let mut pruned_count = 0;
        for edge_uuid in &to_prune {
            if self.delete_relationship(edge_uuid)? {
                pruned_count += 1;
            }
        }

        // Check which candidate entities became orphaned (lost ALL edges)
        // Delete orphaned entities to prevent stale index pollution
        let mut orphaned_entity_ids = Vec::new();
        for entity_uuid in &orphan_candidates {
            let remaining = self.get_entity_relationships(entity_uuid)?;
            if remaining.is_empty() {
                orphaned_entity_ids.push(entity_uuid.to_string());
                if let Err(e) = self.delete_entity(entity_uuid) {
                    tracing::warn!("Failed to delete orphaned entity {}: {}", entity_uuid, e);
                }
            }
        }

        if pruned_count > 0 {
            tracing::debug!(
                "Graph decay: {} edges pruned (of {} total), {} entities orphaned",
                pruned_count,
                all_edges.len(),
                orphaned_entity_ids.len()
            );
        }

        // 4. Apply synaptic homeostasis (global edge downscaling)
        // Runs after pruning so that homeostatic pressure doesn't interfere with
        // the prune decision (edges are pruned at their true decayed strength first).
        if let Err(e) =
            self.apply_synaptic_homeostasis(crate::constants::HOMEOSTASIS_SCALING_FACTOR)
        {
            tracing::warn!("Synaptic homeostasis failed (non-fatal): {}", e);
        }

        Ok(crate::memory::types::GraphDecayResult {
            pruned_count,
            orphaned_entity_ids,
            orphaned_memory_ids: Vec::new(), // Populated by memory layer via entity→memory lookup
        })
    }

    /// Compute Forman-Ricci curvature for all active edges in the graph.
    ///
    /// For a directed edge e = (u → v) in the knowledge graph:
    ///   F(→e→) = 2 - in_deg(u) - out_deg(v)    [flow-through curvature]
    ///   F(←e←) = 2 - out_deg(u) - in_deg(v)    [flow-loss curvature]
    ///   F(e)   = F(→e→) + F(←e←) = 4 - deg(u) - deg(v)
    ///
    /// The directed components reveal information flow structure:
    /// - F(→e→) captures how much flow converges through this edge
    /// - F(←e←) captures how much flow is lost/dispersed at this edge
    ///
    /// Stores computed curvature on each edge and writes back to RocksDB.
    ///
    /// Reference: Leal, Restrepo, Stadler, Jost (2018) arXiv:1811.07825
    ///            "Forman-Ricci curvature for hypergraphs"
    pub fn compute_forman_ricci_curvature(&self) -> Result<CurvatureStats> {
        let all_edges = self.get_all_relationships()?;

        if all_edges.is_empty() {
            return Ok(CurvatureStats {
                edges_computed: 0,
                mean_curvature: 0.0,
                min_curvature: 0.0,
                max_curvature: 0.0,
                positive_count: 0,
                zero_count: 0,
                negative_count: 0,
            });
        }

        // Phase 1: Build degree maps (in-degree and out-degree per entity)
        // Single pass over all edges — O(|E|)
        let mut in_degree: HashMap<Uuid, i32> = HashMap::new();
        let mut out_degree: HashMap<Uuid, i32> = HashMap::new();

        for edge in &all_edges {
            *out_degree.entry(edge.from_entity).or_insert(0) += 1;
            *in_degree.entry(edge.to_entity).or_insert(0) += 1;
        }

        // Phase 2: Compute curvature for each edge, collect per-entity curvatures
        let mut entity_curvatures: HashMap<Uuid, Vec<f32>> = HashMap::new();

        let mut sum_curvature: f64 = 0.0;
        let mut min_curvature = f32::MAX;
        let mut max_curvature = f32::MIN;
        let mut positive_count: usize = 0;
        let mut zero_count: usize = 0;
        let mut negative_count: usize = 0;

        // First pass: compute curvature values and collect per-entity
        let mut edge_curvatures: Vec<f32> = Vec::with_capacity(all_edges.len());
        for edge in &all_edges {
            let src_in = in_degree.get(&edge.from_entity).copied().unwrap_or(0);
            let src_out = out_degree.get(&edge.from_entity).copied().unwrap_or(0);
            let tgt_in = in_degree.get(&edge.to_entity).copied().unwrap_or(0);
            let tgt_out = out_degree.get(&edge.to_entity).copied().unwrap_or(0);

            // Directed Forman-Ricci (Equations 5 + 7 from the paper)
            let flow_through = 2 - src_in - tgt_out; // F(→e→)
            let flow_loss = 2 - src_out - tgt_in; // F(←e←)
            let curvature = (flow_through + flow_loss) as f32; // F(e) = 4 - deg(u) - deg(v)

            edge_curvatures.push(curvature);

            // Collect per-entity curvature for selectivity computation
            entity_curvatures
                .entry(edge.from_entity)
                .or_default()
                .push(curvature);
            entity_curvatures
                .entry(edge.to_entity)
                .or_default()
                .push(curvature);

            // Update stats
            sum_curvature += curvature as f64;
            if curvature < min_curvature {
                min_curvature = curvature;
            }
            if curvature > max_curvature {
                max_curvature = curvature;
            }
            #[allow(clippy::float_cmp)]
            if curvature > 0.0 {
                positive_count += 1;
            } else if curvature == 0.0 {
                zero_count += 1;
            } else {
                negative_count += 1;
            }
        }

        // Phase 3: Compute selectivity per entity
        // selectivity = stdev(incident curvatures) / degree
        // High selectivity → concept (mixed community + bridge edges)
        // Low selectivity → stop word (uniform curvature across all edges)
        let mut entity_selectivity: HashMap<Uuid, f32> = HashMap::new();
        for (entity_id, curvs) in &entity_curvatures {
            let n = curvs.len() as f32;
            if n < 2.0 {
                // Single edge: can't compute variance, assign neutral selectivity
                entity_selectivity.insert(*entity_id, 1.0);
                continue;
            }
            let mean = curvs.iter().sum::<f32>() / n;
            let variance = curvs.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / (n - 1.0);
            let stdev = variance.sqrt();
            let selectivity = stdev / n; // Normalize by degree
            entity_selectivity.insert(*entity_id, selectivity);
        }

        // Phase 4: Write entity selectivity back to RocksDB
        let mut entity_batch = WriteBatch::default();
        for (entity_id, selectivity) in &entity_selectivity {
            let key = entity_id.as_bytes();
            if let Ok(Some(value)) = self.db.get_cf(self.entities_cf(), key) {
                if let Ok((mut entity, _)) = decode_entity_node(&value) {
                    entity.selectivity = Some(*selectivity);
                    if let Ok(encoded) = crate::serialization::encode(&entity) {
                        entity_batch.put_cf(self.entities_cf(), key, encoded);
                    }
                }
            }
        }
        self.db
            .write(entity_batch)
            .map_err(|e| anyhow::anyhow!("Failed to write entity selectivity batch: {}", e))?;

        // Phase 5: Write edges with curvature + endpoint_selectivity
        let mut edge_batch = WriteBatch::default();
        for (mut edge, curvature) in all_edges.into_iter().zip(edge_curvatures.iter()) {
            edge.forman_curvature = Some(*curvature);

            // endpoint_selectivity = min of both endpoints
            // The weakest link determines if this edge connects stop words
            let src_sel = entity_selectivity
                .get(&edge.from_entity)
                .copied()
                .unwrap_or(1.0);
            let tgt_sel = entity_selectivity
                .get(&edge.to_entity)
                .copied()
                .unwrap_or(1.0);
            edge.endpoint_selectivity = Some(src_sel.min(tgt_sel));

            let key = edge.uuid.as_bytes();
            match crate::serialization::encode(&edge) {
                Ok(value) => {
                    edge_batch.put_cf(self.relationships_cf(), key, value);
                }
                Err(e) => {
                    tracing::debug!("Failed to serialize edge {}: {}", edge.uuid, e);
                }
            }
        }

        let edges_computed = positive_count + zero_count + negative_count;

        self.db
            .write(edge_batch)
            .map_err(|e| anyhow::anyhow!("Failed to write curvature batch: {}", e))?;

        let stats = CurvatureStats {
            edges_computed,
            mean_curvature: if edges_computed > 0 {
                (sum_curvature / edges_computed as f64) as f32
            } else {
                0.0
            },
            min_curvature: if edges_computed > 0 {
                min_curvature
            } else {
                0.0
            },
            max_curvature: if edges_computed > 0 {
                max_curvature
            } else {
                0.0
            },
            positive_count,
            zero_count,
            negative_count,
        };

        tracing::info!(
            edges = stats.edges_computed,
            mean = format!("{:.2}", stats.mean_curvature),
            min = stats.min_curvature,
            max = stats.max_curvature,
            positive = stats.positive_count,
            negative = stats.negative_count,
            entities_with_selectivity = entity_selectivity.len(),
            "Forman-Ricci curvature and selectivity computed"
        );

        Ok(stats)
    }

    /// Flush pending maintenance from opportunistic pruning queues.
    ///
    /// Called every maintenance cycle (5 min). Instead of scanning all 34k+ edges,
    /// this only processes edges that were found below prune threshold during normal
    /// reads (via `get_entity_relationships_limited`). Typical cost: 0-50 targeted
    /// deletes per cycle vs a full CF iterator scan.
    pub fn flush_pending_maintenance(&self) -> Result<crate::memory::types::GraphDecayResult> {
        // 1. Drain queues (fast — just swaps empty Vecs)
        let to_prune: Vec<Uuid> = std::mem::take(&mut *self.pending_prune.lock());
        let orphan_candidates: Vec<Uuid> = std::mem::take(&mut *self.pending_orphan_checks.lock());

        if to_prune.is_empty() {
            return Ok(crate::memory::types::GraphDecayResult::default());
        }

        // 2. Dedup UUIDs
        let to_prune: std::collections::HashSet<Uuid> = to_prune.into_iter().collect();
        let orphan_candidates: std::collections::HashSet<Uuid> =
            orphan_candidates.into_iter().collect();

        // 3. Batch delete pruned edges
        let mut pruned_count = 0;
        for edge_uuid in &to_prune {
            if self.delete_relationship(edge_uuid)? {
                pruned_count += 1;
            }
        }

        // 4. Check which candidate entities became orphaned (lost ALL edges)
        let mut orphaned_entity_ids = Vec::new();
        for entity_uuid in &orphan_candidates {
            let remaining = self.get_entity_relationships(entity_uuid)?;
            if remaining.is_empty() {
                orphaned_entity_ids.push(entity_uuid.to_string());
                if let Err(e) = self.delete_entity(entity_uuid) {
                    tracing::warn!("Failed to delete orphaned entity {}: {}", entity_uuid, e);
                }
            }
        }

        if pruned_count > 0 {
            tracing::debug!(
                "Lazy pruning: {} edges deleted, {} entities orphaned",
                pruned_count,
                orphaned_entity_ids.len()
            );
        }

        Ok(crate::memory::types::GraphDecayResult {
            pruned_count,
            orphaned_entity_ids,
            orphaned_memory_ids: Vec::new(),
        })
    }

    /// Adjust entity salience for all entities connected to the given memories.
    ///
    /// Used by the reward loop to propagate recall feedback to entity reputation:
    /// - Helpful recall → positive boost → entities become more trusted
    /// - Misleading recall → negative penalty → entities lose reputation
    /// - Habituation (ignored surfacings) → tiny penalty → noise entities decay
    ///
    /// Looks up each memory's EpisodicNode, reads `entity_refs`, deduplicates,
    /// then batch-updates salience clamped to [0.05, 1.0].
    ///
    /// Returns the number of entities adjusted.
    pub fn reinforce_entity_salience(&self, memory_uuids: &[Uuid], boost: f32) -> Result<usize> {
        if memory_uuids.is_empty() || boost == 0.0 {
            return Ok(0);
        }

        // Collect all unique entity UUIDs across all memories' episodes
        let mut entity_uuids: std::collections::HashSet<Uuid> =
            std::collections::HashSet::with_capacity(memory_uuids.len() * 4);

        for mem_uuid in memory_uuids {
            if let Some(episode) = self.get_episode(mem_uuid)? {
                for entity_uuid in &episode.entity_refs {
                    entity_uuids.insert(*entity_uuid);
                }
            }
        }

        if entity_uuids.is_empty() {
            return Ok(0);
        }

        // Batch-read all entities
        let keys: Vec<[u8; 16]> = entity_uuids.iter().map(|u| *u.as_bytes()).collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
        let results = self
            .db
            .batched_multi_get_cf(self.entities_cf(), &key_refs, false);

        let mut batch = rocksdb::WriteBatch::default();
        let mut adjusted = 0;

        for (i, result) in results.into_iter().enumerate() {
            if let Ok(Some(value)) = result {
                if let Ok((mut entity, _)) = decode_entity_node(&value) {
                    let old_salience = entity.salience;
                    entity.salience = (entity.salience + boost).clamp(0.05, 1.0);

                    // Skip write if salience didn't actually change (already at boundary)
                    if (entity.salience - old_salience).abs() < f32::EPSILON {
                        continue;
                    }

                    if let Ok(encoded) = crate::serialization::encode(&entity) {
                        batch.put_cf(self.entities_cf(), keys[i], encoded);
                        adjusted += 1;
                    }
                }
            }
        }

        if adjusted > 0 {
            self.db.write(batch)?;
            tracing::debug!(
                adjusted,
                boost,
                "Entity salience reinforcement batch committed"
            );
        }

        Ok(adjusted)
    }

    /// Get graph statistics - O(1) using atomic counters
    pub fn get_stats(&self) -> Result<GraphStats> {
        Ok(GraphStats {
            entity_count: self.entity_count.load(Ordering::Relaxed),
            relationship_count: self.relationship_count.load(Ordering::Relaxed),
            episode_count: self.episode_count.load(Ordering::Relaxed),
        })
    }

    /// Count edges per tier, with mean decay-aware strength for each.
    ///
    /// Deliberately NOT folded into [`get_stats`](Self::get_stats): that method
    /// reads three atomics and is called from the periodic maintenance loop, so
    /// it must stay O(1). This is a full O(E) scan of the relationships column
    /// family, for observability surfaces that ask for it explicitly.
    ///
    /// Uses an uncached iterator so a census cannot evict the working set of a
    /// live server, matching [`get_all_entities`](Self::get_all_entities).
    pub fn edge_tier_census(&self) -> Result<EdgeTierCensus> {
        let mut census = EdgeTierCensus::default();
        let (mut l1_sum, mut l2_sum, mut l3_sum) = (0.0f64, 0.0f64, 0.0f64);

        let mut read_opts = rocksdb::ReadOptions::default();
        read_opts.fill_cache(false);
        let iter = self.db.iterator_cf_opt(
            self.relationships_cf(),
            read_opts,
            rocksdb::IteratorMode::Start,
        );

        for (_, value) in iter.flatten() {
            let Ok((edge, _)) = decode_relationship_edge(&value) else {
                continue;
            };
            census.total_scanned += 1;

            let strength = edge.effective_strength();
            match edge.tier {
                EdgeTier::L1Working => {
                    census.l1_working += 1;
                    l1_sum += strength as f64;
                }
                EdgeTier::L2Episodic => {
                    census.l2_episodic += 1;
                    l2_sum += strength as f64;
                }
                EdgeTier::L3Semantic => {
                    census.l3_semantic += 1;
                    l3_sum += strength as f64;
                }
            }

            if strength < edge.tier.prune_threshold() {
                census.below_prune_threshold += 1;
            }
        }

        let mean = |sum: f64, n: usize| {
            if n == 0 {
                0.0
            } else {
                (sum / n as f64) as f32
            }
        };
        census.l1_mean_strength = mean(l1_sum, census.l1_working);
        census.l2_mean_strength = mean(l2_sum, census.l2_episodic);
        census.l3_mean_strength = mean(l3_sum, census.l3_semantic);

        Ok(census)
    }

    /// Get all entities in the graph
    pub fn get_all_entities(&self) -> Result<Vec<EntityNode>> {
        let mut entities = Vec::new();

        let mut read_opts = rocksdb::ReadOptions::default();
        read_opts.fill_cache(false);
        let iter =
            self.db
                .iterator_cf_opt(self.entities_cf(), read_opts, rocksdb::IteratorMode::Start);
        for (_, value) in iter.flatten() {
            if let Ok((entity, _)) = decode_entity_node(&value) {
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

        // fill_cache(false) prevents this full scan from evicting hot data from
        // the block cache. Decompressed blocks are used transiently and freed
        // after the iterator advances, reducing peak C++ heap usage.
        let mut read_opts = rocksdb::ReadOptions::default();
        read_opts.fill_cache(false);
        let iter = self.db.iterator_cf_opt(
            self.relationships_cf(),
            read_opts,
            rocksdb::IteratorMode::Start,
        );
        for (_, value) in iter.flatten() {
            if let Ok((edge, _)) = decode_relationship_edge(&value) {
                // Only include non-invalidated relationships
                if edge.invalidated_at.is_none() {
                    relationships.push(edge);
                }
            }
        }

        // Sort by strength (strongest first)
        relationships.sort_by(|a, b| b.strength.total_cmp(&a.strength));

        Ok(relationships)
    }

    /// Get all episodes in the graph
    pub fn get_all_episodes(&self) -> Result<Vec<EpisodicNode>> {
        let mut episodes = Vec::new();

        // fill_cache(false) prevents this full scan from evicting hot data from
        // the block cache. Decompressed blocks are used transiently and freed
        // after the iterator advances, reducing peak C++ heap usage.
        let mut read_opts = rocksdb::ReadOptions::default();
        read_opts.fill_cache(false);
        let iter =
            self.db
                .iterator_cf_opt(self.episodes_cf(), read_opts, rocksdb::IteratorMode::Start);
        for (_, value) in iter.flatten() {
            if let Ok((episode, _)) = crate::serialization::try_decode::<EpisodicNode>(&value) {
                episodes.push(episode);
            }
        }

        // Sort by created_at descending (newest first)
        episodes.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(episodes)
    }

    /// Get the Memory Universe visualization data
    /// Returns entities as "stars" with positions based on their relationships,
    /// sized by salience, and colored by entity type.
    /// Whether a relation type carries no extracted meaning — the co-occurrence
    /// substrate rather than a typed relation.
    fn is_generic_relation(relation_type: &RelationType) -> bool {
        matches!(
            relation_type,
            RelationType::CoOccurs | RelationType::RelatedTo
        )
    }

    /// Direction-independent key for an entity pair.
    fn unordered_pair(a: Uuid, b: Uuid) -> (Uuid, Uuid) {
        if a <= b {
            (a, b)
        } else {
            (b, a)
        }
    }

    /// Project the graph for visualization with the default read filter.
    pub fn get_universe(&self) -> Result<MemoryUniverse> {
        self.get_universe_filtered(UniverseFilter::default())
    }

    /// Project the graph for visualization, applying `filter` and reporting
    /// exactly what it removed on [`MemoryUniverse::filter`].
    pub fn get_universe_filtered(&self, filter: UniverseFilter) -> Result<MemoryUniverse> {
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
                // Apply small gravitational pull based on effective (decay-aware) strength
                let pull_factor = rel.effective_strength() * 0.05;

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

        // Pairs already joined by a TYPED edge. A bare generic edge over the same
        // pair adds nothing a viewer can act on, and is the main source of
        // doubled lines between two nodes. Unordered key: direction does not make
        // a co-occurrence edge informative.
        let typed_pairs: HashSet<(Uuid, Uuid)> = relationships
            .iter()
            .filter(|rel| !Self::is_generic_relation(&rel.relation_type))
            .map(|rel| Self::unordered_pair(rel.from_entity, rel.to_entity))
            .collect();

        let mut report = UniverseFilterReport {
            min_generic_strength: filter.min_generic_strength,
            hide_redundant_generic: filter.hide_redundant_generic,
            ..Default::default()
        };

        // Create gravitational connections AFTER star positions are finalized
        // This ensures from_position/to_position match current star positions
        let connections: Vec<GravitationalConnection> = relationships
            .iter()
            .filter(|rel| {
                // Typed edges are never hidden: a typed relation is an extraction
                // result, and suppressing it would misrepresent what is known.
                if !Self::is_generic_relation(&rel.relation_type) {
                    return true;
                }
                if rel.effective_strength() < filter.min_generic_strength {
                    report.hidden_weak_generic += 1;
                    return false;
                }
                if filter.hide_redundant_generic
                    && typed_pairs.contains(&Self::unordered_pair(rel.from_entity, rel.to_entity))
                {
                    report.hidden_redundant_generic += 1;
                    return false;
                }
                true
            })
            .filter_map(|rel| {
                let (Some(from_idx), Some(to_idx)) = (
                    entity_indices.get(&rel.from_entity),
                    entity_indices.get(&rel.to_entity),
                ) else {
                    // Dangling endpoint — undrawable. Counted so referential
                    // damage in the store surfaces instead of vanishing.
                    report.dropped_dangling += 1;
                    return None;
                };

                Some(GravitationalConnection {
                    id: rel.uuid.to_string(),
                    from_id: rel.from_entity.to_string(),
                    to_id: rel.to_entity.to_string(),
                    relation_type: rel.relation_type.as_str().to_string(),
                    strength: rel.effective_strength(),
                    tier: rel.tier,
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
            // The TRUE total, not the rendered count — see the field docs.
            total_connections: relationships.len(),
            filter: report,
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
        Some(EntityLabel::Keyword) => "#FF9F43".to_string(), // Orange for YAKE keywords
        Some(EntityLabel::Project) => "#E74C3C".to_string(), // Red — project anchors
        Some(EntityLabel::Task) => "#F39C12".to_string(),   // Amber — work items
        Some(EntityLabel::Document) => "#1ABC9C".to_string(), // Turquoise — knowledge
        Some(EntityLabel::Repository) => "#2ECC71".to_string(), // Emerald — code entities
        Some(EntityLabel::Service) => "#3498DB".to_string(), // Blue — architectural
        Some(EntityLabel::Database) => "#9B59B6".to_string(), // Purple — data stores
        Some(EntityLabel::Metric) => "#E67E22".to_string(), // Dark orange — telemetry
        Some(EntityLabel::Configuration) => "#95A5A6".to_string(), // Silver — config
        Some(EntityLabel::Environment) => "#16A085".to_string(), // Dark teal — infra
        Some(EntityLabel::Pipeline) => "#2980B9".to_string(), // Dark blue — CI/CD
        Some(EntityLabel::Team) => "#27AE60".to_string(),   // Green — organizational
        Some(EntityLabel::Role) => "#8E44AD".to_string(),   // Dark purple — roles
        Some(EntityLabel::Module) => "#D35400".to_string(), // Pumpkin — code modules
        Some(EntityLabel::Norp) => "#C0392B".to_string(),   // Brick red — groups/affiliations
        Some(EntityLabel::Gpe) => "#5DADE2".to_string(),    // Lighter blue — political geography
        Some(EntityLabel::Facility) => "#7F8C8D".to_string(), // Slate — built structures
        Some(EntityLabel::Vehicle) => "#34495E".to_string(), // Dark slate — vehicles
        Some(EntityLabel::Weapon) => "#922B21".to_string(), // Dark red — weapons
        Some(EntityLabel::Work) => "#AF7AC5".to_string(),   // Orchid — creative works
        Some(EntityLabel::Law) => "#6C3483".to_string(),    // Deep violet — legal instruments
        Some(EntityLabel::Title) => "#CA6F1E".to_string(),  // Burnt orange — honorifics/positions
        Some(EntityLabel::Cyber) => "#17202A".to_string(),  // Near-black — threats/malware
        Some(EntityLabel::Money) => "#28B463".to_string(),  // Money green
        Some(EntityLabel::Quantity) => "#F5B041".to_string(), // Amber — measurements
        Some(EntityLabel::Time) => "#A569BD".to_string(), // Violet — distinct from Date's light purple
        Some(EntityLabel::Other(_)) => "#AEB6BF".to_string(), // Gray
        None => "#AEB6BF".to_string(),                    // Gray default
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
    /// Consolidation tier of the underlying edge: L1 working (new, dense,
    /// aggressive decay) → L2 episodic (proven) → L3 semantic (consolidated,
    /// near-permanent).
    ///
    /// `RelationshipEdge` has carried this since the tier system landed, but
    /// the universe payload dropped it, so the only consumer that could see
    /// tiers was `/api/graph/data/{user_id}` — which hard-truncates at 200
    /// relationships PER TIER and reports the truncated counts as totals
    /// (src/handlers/visualization.rs:378-392, :458-459). A client wanting both
    /// the whole graph and its tier structure could not have both. It is a
    /// `Copy` enum, so echoing it costs one byte-range on the wire and no
    /// allocation.
    pub tier: EdgeTier,
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
    /// TRUE number of relationships in the graph, before any read filter.
    ///
    /// Deliberately not reduced by filtering: a viewer must be able to tell that
    /// it is looking at a subset, and how big a subset. `connections.len()` is
    /// what was rendered; this is what exists.
    pub total_connections: usize,
    pub bounds: UniverseBounds,
    /// What the read filter removed, declared rather than silently applied.
    pub filter: UniverseFilterReport,
}

/// The read-side filter applied to a universe projection.
///
/// `get_universe` renders the graph the ingest pipeline actually built, and on a
/// dense corpus that is a near-clique of untyped co-occurrence edges. Filtering
/// at READ is the honest place to fix that: the edges stay in the substrate
/// where retrieval still uses them, and the viewer is told what was hidden
/// instead of being shown a prettier graph than the one that exists.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UniverseFilter {
    /// Generic (`CoOccurs`/`RelatedTo`) edges below this effective strength are
    /// not rendered. Typed edges are NEVER hidden by this — a typed relation is
    /// an extraction result, and hiding it would misrepresent what the system
    /// knows.
    pub min_generic_strength: f32,
    /// Hide a bare generic edge when a TYPED edge already joins the same pair.
    /// The generic one adds no information a viewer can act on, and it is the
    /// main source of doubled lines between the same two nodes.
    pub hide_redundant_generic: bool,
}

impl Default for UniverseFilter {
    /// The default hides only generic edges the system ALREADY considers dead:
    /// `L1_PRUNE_THRESHOLD` is the strength at which L1 edges become eligible
    /// for pruning, so anything below it is scheduled for deletion and is
    /// nothing but noise on screen.
    ///
    /// This threshold is chosen because it is one the engine already acts on —
    /// not tuned until the picture looked good. Raising it further is a product
    /// decision that wants the tier census (`edge_tier_census`) as evidence
    /// first; the handler exposes it as a query parameter so that decision can
    /// be made from data rather than from a guess baked into a default.
    fn default() -> Self {
        Self {
            min_generic_strength: crate::constants::L1_PRUNE_THRESHOLD,
            hide_redundant_generic: true,
        }
    }
}

/// What a [`UniverseFilter`] actually removed, returned alongside the payload.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct UniverseFilterReport {
    /// The strength floor that was applied to generic edges.
    pub min_generic_strength: f32,
    /// Whether redundant generic edges were hidden.
    pub hide_redundant_generic: bool,
    /// Generic edges hidden for falling below the strength floor.
    pub hidden_weak_generic: usize,
    /// Generic edges hidden because a typed edge already joined the pair.
    pub hidden_redundant_generic: usize,
    /// Edges dropped because an endpoint is missing from the entity set. Not a
    /// filter decision — a dangling edge cannot be drawn — but reported so
    /// referential damage in the store is visible rather than invisible.
    pub dropped_dangling: usize,
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

/// Per-tier edge population, from [`GraphMemory::edge_tier_census`].
///
/// The edge tiers carry real consequences — L3 gets a 4x retrieval trust
/// multiplier over L1 (`EDGE_TIER_TRUST_*`) and a 2160-hour prune shield versus
/// L1's 168 — but nothing reported how many edges were in each. "Is L3 empty, or
/// is L1 overwhelmed?" was unanswerable, so every tier tuning decision was a
/// guess. This is the measurement that makes those decisions evidence-based.
///
/// Mean effective (decay-aware) strength accompanies each count because a count
/// alone cannot distinguish a healthy L3 from one full of edges that decayed to
/// the floor but kept the tier — tier is a ratchet that decay never lowers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EdgeTierCensus {
    /// Edges currently tagged `L1Working`.
    pub l1_working: usize,
    /// Edges currently tagged `L2Episodic`.
    pub l2_episodic: usize,
    /// Edges currently tagged `L3Semantic`.
    pub l3_semantic: usize,
    /// Mean `effective_strength()` of L1 edges; 0.0 when the tier is empty.
    pub l1_mean_strength: f32,
    /// Mean `effective_strength()` of L2 edges; 0.0 when the tier is empty.
    pub l2_mean_strength: f32,
    /// Mean `effective_strength()` of L3 edges; 0.0 when the tier is empty.
    pub l3_mean_strength: f32,
    /// Edges whose effective strength has fallen below their tier's prune
    /// threshold but which have not yet been pruned. A large number here means
    /// the tier labels the UI colours by are stale relative to real strength.
    pub below_prune_threshold: usize,
    /// Total edges scanned. Equals the sum of the three tier counts, and is
    /// reported separately so a decode failure during the scan is visible rather
    /// than silently shrinking the census.
    pub total_scanned: usize,
}

/// Summary statistics from Forman-Ricci curvature computation
///
/// Lightweight reputation signal for an entity, derived from graph topology.
/// Used by NER quality gating to penalize/reject known stop-word entities.
#[derive(Debug, Clone)]
pub struct EntityReputation {
    /// Curvature selectivity: high = concept, low = stop-word hub
    pub selectivity: f32,
    /// How many times this entity has been mentioned
    pub mention_count: usize,
    /// Number of edges incident to this entity
    pub degree: usize,
    /// Feedback-driven salience (reward loop output, 0.05–1.0)
    pub salience: f32,
}

/// Captures the distribution of curvature across the knowledge graph.
/// Positive curvature = tightly-connected community interior edges.
/// Negative curvature = bridge/bottleneck edges between hubs.
///
/// Reference: Leal, Restrepo, Stadler, Jost (2018) arXiv:1811.07825
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureStats {
    /// Number of edges with curvature computed
    pub edges_computed: usize,
    /// Mean curvature across all edges
    pub mean_curvature: f32,
    /// Minimum curvature (most negative = strongest bottleneck)
    pub min_curvature: f32,
    /// Maximum curvature (most positive = tightest community)
    pub max_curvature: f32,
    /// Number of edges with positive curvature (community-interior)
    pub positive_count: usize,
    /// Number of edges with zero curvature
    pub zero_count: usize,
    /// Number of edges with negative curvature (bridges/bottlenecks)
    pub negative_count: usize,
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

    // Extended ontological keyword dictionaries (2026-03)
    /// Service/infrastructure keywords (microservices, daemons, proxies)
    service_keywords: HashSet<String>,
    /// Database system keywords
    database_keywords: HashSet<String>,
    /// CI/CD and data pipeline keywords
    pipeline_keywords: HashSet<String>,
    /// Deployment environment keywords
    environment_keywords: HashSet<String>,
    /// Observability and metric keywords
    metric_keywords: HashSet<String>,
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

        // Service/infrastructure keywords
        let service_keywords: HashSet<String> = vec![
            "microservice",
            "daemon",
            "worker",
            "cron",
            "lambda",
            "serverless",
            "gateway",
            "proxy",
            "nginx",
            "envoy",
            "istio",
            "grpc",
            "graphql",
            "webhook",
            "middleware",
            "sidecar",
            "ingress",
            "loadbalancer",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Database system keywords
        let database_keywords: HashSet<String> = vec![
            "postgresql",
            "postgres",
            "mysql",
            "mariadb",
            "sqlite",
            "dynamodb",
            "cassandra",
            "elasticsearch",
            "opensearch",
            "clickhouse",
            "timescaledb",
            "cockroachdb",
            "neo4j",
            "arangodb",
            "rocksdb",
            "leveldb",
            "badger",
            "redb",
            "sled",
            "foundationdb",
            "vitess",
            "couchbase",
            "couchdb",
            "influxdb",
            "questdb",
            "duckdb",
            "dragonfly",
            "valkey",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // CI/CD and pipeline keywords
        let pipeline_keywords: HashSet<String> = vec![
            "jenkins",
            "circleci",
            "travis",
            "argo",
            "tekton",
            "spinnaker",
            "buildkite",
            "drone",
            "concourse",
            "woodpecker",
            "flux",
            "argocd",
            "terraform",
            "pulumi",
            "ansible",
            "airflow",
            "dagster",
            "prefect",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Deployment environment keywords
        let environment_keywords: HashSet<String> = vec![
            "staging",
            "production",
            "prod",
            "dev",
            "development",
            "sandbox",
            "canary",
            "preview",
            "qa",
            "uat",
            "integration",
            "preprod",
            "hotfix",
            "nightly",
            "edge",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Observability and metric keywords
        let metric_keywords: HashSet<String> = vec![
            "latency",
            "throughput",
            "p99",
            "p95",
            "p50",
            "slo",
            "sla",
            "sli",
            "uptime",
            "availability",
            "mttr",
            "mttf",
            "error rate",
            "saturation",
            "prometheus",
            "grafana",
            "datadog",
            "newrelic",
            "jaeger",
            "tempo",
            "loki",
            "opentelemetry",
            "otel",
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
            service_keywords,
            database_keywords,
            pipeline_keywords,
            environment_keywords,
            metric_keywords,
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
            EntityLabel::Person => 0.8,         // People are highly salient
            EntityLabel::Organization => 0.7,   // Organizations are important
            EntityLabel::Location => 0.6,       // Locations matter for context
            EntityLabel::Technology => 0.6,     // Tech keywords matter for dev context
            EntityLabel::Product => 0.7,        // Products are specific entities
            EntityLabel::Event => 0.6,          // Events are temporal anchors
            EntityLabel::Skill => 0.5,          // Skills are somewhat important
            EntityLabel::Keyword => 0.55,       // YAKE keywords - discriminative terms
            EntityLabel::Concept => 0.4,        // Concepts are more generic
            EntityLabel::Date => 0.3,           // Dates are structural, not salient
            EntityLabel::Project => 0.7,        // Projects are specific, high-salience anchors
            EntityLabel::Task => 0.55,          // Tasks are common but specific work items
            EntityLabel::Document => 0.5,       // Documents provide contextual reference
            EntityLabel::Repository => 0.65,    // Repos are identifiable code entities
            EntityLabel::Service => 0.65,       // Services are architectural anchors
            EntityLabel::Database => 0.6,       // Databases are infrastructure anchors
            EntityLabel::Metric => 0.5,         // Metrics are precise but numerous
            EntityLabel::Configuration => 0.45, // Config is low-salience, high-frequency
            EntityLabel::Environment => 0.5,    // Environments are contextual
            EntityLabel::Pipeline => 0.6,       // Pipelines are operational entities
            EntityLabel::Team => 0.7,           // Teams are organizational anchors
            EntityLabel::Role => 0.55,          // Roles are semi-generic
            EntityLabel::Module => 0.55,        // Modules are code-level entities
            EntityLabel::Norp => 0.55,          // Nationalities/groups are moderately salient
            EntityLabel::Gpe => 0.6,            // Geopolitical entities, on par with Location
            EntityLabel::Facility => 0.5,       // Facilities are concrete but generic
            EntityLabel::Vehicle => 0.5,        // Vehicles are concrete but generic
            EntityLabel::Weapon => 0.55,        // Weapons are specific, notable entities
            EntityLabel::Work => 0.6,           // Named works are specific, on par with Product
            EntityLabel::Law => 0.55,           // Named laws/regulations are specific anchors
            EntityLabel::Title => 0.5,          // Titles are semi-generic, like Role
            EntityLabel::Cyber => 0.55,         // Cyber entities (CVEs, malware) are specific
            EntityLabel::Money => 0.4,          // Monetary amounts are structural, like Date
            EntityLabel::Quantity => 0.35,      // Measurements are structural, low salience
            EntityLabel::Time => 0.3,           // Time-of-day is structural, like Date
            EntityLabel::Other(_) => 0.3,       // Unknown types get low salience
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

            // Check for known organization keywords (direct match, min 2 chars to filter "x" noise)
            if lower.len() >= 2 && self.org_keywords.contains(&lower) && !seen.contains(&lower) {
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
            // Check extended ontological keyword dictionaries (more specific before generic)
            if self.database_keywords.contains(&lower) && !seen.contains(&lower) {
                entities.push(ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Database,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Database, true),
                });
                seen.insert(lower.clone());
                continue;
            }

            if self.pipeline_keywords.contains(&lower) && !seen.contains(&lower) {
                entities.push(ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Pipeline,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Pipeline, true),
                });
                seen.insert(lower.clone());
                continue;
            }

            if self.service_keywords.contains(&lower) && !seen.contains(&lower) {
                entities.push(ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Service,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Service, true),
                });
                seen.insert(lower.clone());
                continue;
            }

            if self.environment_keywords.contains(&lower) && !seen.contains(&lower) {
                entities.push(ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Environment,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Environment, false),
                });
                seen.insert(lower.clone());
                continue;
            }

            if self.metric_keywords.contains(&lower) && !seen.contains(&lower) {
                entities.push(ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Metric,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Metric, false),
                });
                seen.insert(lower.clone());
                continue;
            }

            // Generic technology keywords (catch-all for tech not matched above)
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

                // Check for multi-word capitalized sequences.
                // Include capitalized stop words (Of, The, And) in entity names
                // to preserve proper nouns like "Bank Of America", "University Of Delhi".
                let mut j = i + 1;
                while j < words.len()
                    && words[j]
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                {
                    let next_word = words[j].trim_matches(|c: char| !c.is_alphanumeric());
                    entity_name.push(' ');
                    entity_name.push_str(next_word);
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
                        // are likely proper names — use Concept as safe default
                        // Concept(0.4) + proper noun boost(1.2x) = 0.48 salience
                        // Hebbian strengthening will promote genuinely important entities
                        entity_label = EntityLabel::Concept;
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

        // HYBRID APPROACH: POS-based extraction + YAKE importance scoring
        //
        // 1. POS extraction ensures ALL content words are captured (no frequency bias)
        // 2. YAKE provides discriminativeness scores for boosting rare/important terms
        //
        // This solves the "sunrise problem": YAKE alone buries rare words at position 41,
        // but POS ensures "sunrise" is extracted, and YAKE boosts its salience.
        use crate::embeddings::keywords::{KeywordConfig, KeywordExtractor};
        use crate::memory::query_parser::{extract_chunks, PosTag};

        // Get YAKE importance scores for discriminative weighting
        let kw_config = KeywordConfig {
            max_keywords: 100, // Get many keywords for lookup
            ngrams: 1,
            min_length: 3,
            ..Default::default()
        };
        let kw_extractor = KeywordExtractor::with_config(kw_config);
        let keywords = kw_extractor.extract(text);

        // Build a lookup map: term -> importance (0.0-1.0)
        let yake_importance: std::collections::HashMap<String, f32> = keywords
            .into_iter()
            .map(|kw| (kw.text.to_lowercase(), kw.importance))
            .collect();

        // POS-based extraction for comprehensive coverage
        let chunk_extraction = extract_chunks(text);

        // Add all proper nouns (these are likely named entities we might have missed)
        for proper_noun in &chunk_extraction.proper_nouns {
            let term_lower = proper_noun.to_lowercase();
            if !seen.contains(&term_lower) && term_lower.len() >= 3 {
                // Boost salience if YAKE identified this as discriminative
                let yake_boost = yake_importance.get(&term_lower).copied().unwrap_or(0.0);
                let entity = ExtractedEntity {
                    name: proper_noun.clone(),
                    label: EntityLabel::Person,
                    base_salience: 0.7 + (yake_boost * 0.2), // 0.7-0.9
                };
                entities.push(entity);
                seen.insert(term_lower);
            }
        }

        // Add all content words as Keyword entities
        // POS ensures comprehensive extraction, YAKE boosts discriminative terms
        for chunk in &chunk_extraction.chunks {
            for word in &chunk.words {
                let term_lower = word.text.to_lowercase();

                // Skip if already extracted or too short
                if seen.contains(&term_lower) || term_lower.len() < 4 {
                    continue;
                }

                // Skip stop words
                if self.stop_words.contains(&term_lower) {
                    continue;
                }

                // Base salience by POS, boosted by YAKE importance
                let yake_boost = yake_importance.get(&term_lower).copied().unwrap_or(0.0);

                let (label, base_salience) = match word.pos {
                    PosTag::Noun | PosTag::ProperNoun => {
                        // Nouns are most important, start at 0.5
                        (EntityLabel::Keyword, 0.5)
                    }
                    PosTag::Verb => {
                        // Verbs connect entities, start at 0.4
                        (EntityLabel::Keyword, 0.4)
                    }
                    PosTag::Adjective => {
                        // Adjectives are modifiers, start at 0.35
                        (EntityLabel::Keyword, 0.35)
                    }
                    _ => continue,
                };

                // Boost by YAKE importance (0.0-0.3 boost based on discriminativeness)
                let final_salience = base_salience + (yake_boost * 0.3);

                let entity = ExtractedEntity {
                    name: word.text.clone(),
                    label,
                    base_salience: final_salience,
                };
                entities.push(entity);
                seen.insert(term_lower);
            }
        }

        entities
    }

    /// Extract co-occurrence pairs from text for graph edge creation
    ///
    /// Returns pairs of (entity1, entity2) that appear in the same sentence.
    /// This enables creating edges between words that co-occur, which is critical
    /// for multi-hop retrieval (e.g., connecting "Melanie" to "sunrise" when
    /// they appear in the same sentence about painting).
    pub fn extract_cooccurrence_pairs(&self, text: &str) -> Vec<(String, String)> {
        use crate::memory::query_parser::extract_chunks;

        let chunk_extraction = extract_chunks(text);
        let mut pairs = Vec::new();

        // Get all co-occurrence pairs from chunks (same sentence)
        for chunk in &chunk_extraction.chunks {
            let content_words = chunk.content_words();

            // Create pairs between all content words in the same sentence
            for i in 0..content_words.len() {
                for j in (i + 1)..content_words.len() {
                    let w1 = content_words[i].text.to_lowercase();
                    let w2 = content_words[j].text.to_lowercase();

                    // Skip very short words and stop words
                    if w1.len() >= 3
                        && w2.len() >= 3
                        && !self.stop_words.contains(&w1)
                        && !self.stop_words.contains(&w2)
                    {
                        pairs.push((w1, w2));
                    }
                }
            }
        }

        pairs
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
    use chrono::{Duration, TimeZone};

    /// Every one of the 18 schema coarse ids must resolve to a real
    /// `EntityLabel` variant, never the `Other(String)` fallback. This couples
    /// the entity-type schema asset (`src/entity_type/entity-type-schema.json`)
    /// to the type system so future schema drift (a coarse id added to the
    /// JSON without a matching enum variant) fails CI instead of silently
    /// degrading GLiNER-typed entities to `Other`.
    #[test]
    fn every_schema_coarse_maps_to_a_variant() {
        for coarse in &crate::entity_type::schema().coarse {
            let label = EntityLabel::from_coarse_id(&coarse.id);
            assert!(
                !matches!(label, EntityLabel::Other(_)),
                "coarse id `{}` rolled up to Other(_) — EntityLabel::from_coarse_id is missing a variant for it",
                coarse.id
            );
        }
    }

    /// Every one of the 141 schema fine leaves must roll up (via
    /// `coarse_of`) to a coarse id that `from_coarse_id` resolves to a real
    /// `EntityLabel` variant. This is the end-to-end check: a fine label
    /// GLiNER actually predicts must never dead-end in `Other(_)`.
    #[test]
    fn every_fine_rolls_up_to_a_real_variant() {
        for fine in &crate::entity_type::schema().fine {
            let coarse_id = crate::entity_type::coarse_of(&fine.label).unwrap_or_else(|| {
                panic!(
                    "fine label `{}` has no coarse rollup in the schema",
                    fine.label
                )
            });
            let label = EntityLabel::from_coarse_id(coarse_id);
            assert!(
                !matches!(label, EntityLabel::Other(_)),
                "fine label `{}` rolls up to coarse `{}` which resolves to Other(_) — EntityLabel::from_coarse_id is missing a variant for it",
                fine.label,
                coarse_id
            );
        }
    }

    /// The GLiNER coarse subtypes added alongside the schema (Gpe, Facility,
    /// Vehicle, Weapon, Title, Work, Law, Cyber, Norp) must roll up through
    /// `parent_labels()` so type-gated retrieval still matches them — a
    /// "where"-query expecting `[Location]` must not miss `Gpe`/`Facility`
    /// entities just because GLiNER typed them at the finer variant.
    #[test]
    fn new_coarse_variants_match_their_hierarchy_parent() {
        assert!(EntityLabel::Gpe.matches_with_hierarchy(&EntityLabel::Location));
        assert!(EntityLabel::Facility.matches_with_hierarchy(&EntityLabel::Location));
        assert!(EntityLabel::Vehicle.matches_with_hierarchy(&EntityLabel::Product));
        assert!(EntityLabel::Weapon.matches_with_hierarchy(&EntityLabel::Product));
        assert!(EntityLabel::Title.matches_with_hierarchy(&EntityLabel::Role));
        assert!(EntityLabel::Work.matches_with_hierarchy(&EntityLabel::Concept));
        assert!(EntityLabel::Law.matches_with_hierarchy(&EntityLabel::Concept));
        assert!(EntityLabel::Cyber.matches_with_hierarchy(&EntityLabel::Concept));
        assert!(EntityLabel::Norp.matches_with_hierarchy(&EntityLabel::Organization));
    }

    #[test]
    fn spreading_weight_prefers_predicates_over_cooccurrence() {
        // The load-bearing contrast: a real causal predicate must out-weight bare
        // co-occurrence so activation flows along meaning, not adjacency.
        assert!(
            RelationType::Causes.spreading_weight() > RelationType::CoOccurs.spreading_weight()
        );
        assert!(
            RelationType::Triggers.spreading_weight() > RelationType::RelatedTo.spreading_weight()
        );
        assert!(
            RelationType::WorksAt.spreading_weight() > RelationType::CoOccurs.spreading_weight()
        );
        // Co-occurrence is the weakest evidence of meaning.
        assert_eq!(RelationType::CoOccurs.spreading_weight(), 0.5);
        assert!((RelationType::Causes.spreading_weight() - 1.3).abs() < f32::EPSILON);
    }

    #[test]
    fn extract_predicate_recovers_causal_relations_from_text() {
        // The lineage harness phrasing must recover a causal predicate that the
        // (Event, Event) label heuristic would have flattened to CoOccurs.
        assert_eq!(
            extract_predicate_from_text("Vornak set Meslin in motion."),
            Some(RelationType::Triggers)
        );
        assert_eq!(
            extract_predicate_from_text("the outage caused the rollback"),
            Some(RelationType::Triggers)
        );
        assert_eq!(
            extract_predicate_from_text("Alice manages the platform team"),
            Some(RelationType::Manages)
        );
        assert_eq!(
            extract_predicate_from_text("Service A depends on Service B"),
            Some(RelationType::DependsOn)
        );
        // No relational cue → None, so the caller keeps the label-pair inference.
        assert_eq!(
            extract_predicate_from_text("the sky is a colour today"),
            None
        );
    }

    #[test]
    fn extract_directed_predicate_sets_cause_effect_arrow() {
        // "A set B in motion": A is the cause (appears first) → a_is_source = true.
        assert_eq!(
            extract_directed_predicate("Vornak set Meslin in motion.", "Vornak", "Meslin"),
            Some((RelationType::Triggers, true))
        );
        // Same fact, arguments swapped: B is queried first but A is still the cause,
        // so a_is_source = false (the caller swaps from/to). This is the arrow-
        // direction fix that the NER-order default got wrong.
        assert_eq!(
            extract_directed_predicate("Vornak set Meslin in motion.", "Meslin", "Vornak"),
            Some((RelationType::Triggers, false))
        );
        // Sentence scoping: a cue in a NEIGHBOURING sentence must not be applied to
        // a pair that merely co-occurs across the boundary.
        assert_eq!(
            extract_directed_predicate("Alpha and Beta met. Gamma caused Delta.", "Alpha", "Beta"),
            None
        );
        // Both mentions in the cued sentence → recovered.
        assert_eq!(
            extract_directed_predicate("Gamma caused Delta.", "Gamma", "Delta"),
            Some((RelationType::Triggers, true))
        );
        // Missing mention → None (caller keeps label-pair inference).
        assert_eq!(
            extract_directed_predicate("Gamma caused Delta.", "Gamma", "Epsilon"),
            None
        );
    }

    #[test]
    fn extract_directed_predicate_flips_effect_first_constructions() {
        // "A happened because of B": A appears FIRST but B is the CAUSE.
        // Surface order alone would mark A the source — the audit's inversion bug.
        assert_eq!(
            extract_directed_predicate(
                "The flood happened because of the dam failure.",
                "dam failure",
                "flood"
            ),
            Some((RelationType::Triggers, true)),
            "because-of: the later-mentioned cause must be the source"
        );
        // Passive voice: "A was caused by B" — B is the cause despite appearing second.
        assert_eq!(
            extract_directed_predicate("The outage was caused by Redis.", "Redis", "outage"),
            Some((RelationType::Triggers, true)),
            "passive caused-by: the later-mentioned cause must be the source"
        );
        // "due to": same effect-first shape.
        assert_eq!(
            extract_directed_predicate("The delay was due to the storm.", "delay", "storm"),
            Some((RelationType::Triggers, false)),
            "due-to: the earlier-mentioned effect must NOT be the source"
        );
        // Control: cause-first active voice is unchanged by the fix.
        assert_eq!(
            extract_directed_predicate("Redis caused the outage.", "Redis", "outage"),
            Some((RelationType::Triggers, true))
        );
    }

    #[test]
    fn extract_directed_predicate_rejects_cue_fragment_endpoints() {
        // The fallback NER (no GLiNER assets) mints the cue's OWN words as
        // entities: "motion" from "in motion", "brought" from "brought about".
        // Those mentions sit inside the predicate span, so they are the relation
        // lexeme — never a causal argument. Pairing a real event with such a
        // fragment must recover NO causal edge, or the shared fragment welds every
        // chain into a cross-document causal bridge (fallback-path lineage flood).
        assert_eq!(
            extract_directed_predicate("Vornak set Meslin in motion.", "Vornak", "motion"),
            None,
            "'motion' is part of the 'in motion' cue, not a causal endpoint"
        );
        assert_eq!(
            extract_directed_predicate("Vornak set Meslin in motion.", "Meslin", "motion"),
            None,
            "'motion' is part of the 'in motion' cue, not a causal endpoint"
        );
        assert_eq!(
            extract_directed_predicate(
                "the Meslin incident then brought about the Caldor incident.",
                "the Meslin incident",
                "brought"
            ),
            None,
            "'brought' is part of the 'brought about' cue, not a causal endpoint"
        );
        assert_eq!(
            extract_directed_predicate(
                "the Meslin incident then brought about the Caldor incident.",
                "brought",
                "the Caldor incident"
            ),
            None,
            "'brought' is part of the 'brought about' cue, not a causal endpoint"
        );
        // The genuine argument pair in the SAME sentences still types causal — the
        // gate removes only the predicate-fragment endpoints, not the relation.
        assert_eq!(
            extract_directed_predicate("Vornak set Meslin in motion.", "Vornak", "Meslin"),
            Some((RelationType::Triggers, true))
        );
        assert_eq!(
            extract_directed_predicate(
                "the Meslin incident then brought about the Caldor incident.",
                "the Meslin incident",
                "the Caldor incident"
            ),
            Some((RelationType::Triggers, true))
        );
    }

    #[test]
    fn trace_causal_origins_walks_back_to_root() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        let mk = |from: Uuid, to: Uuid, rt: RelationType| RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: from,
            to_entity: to,
            relation_type: rt,
            strength: 0.8,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: Utc::now(),
            activation_count: 1,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic,
            entity_confidence: None,
            forman_curvature: None,
            endpoint_selectivity: None,
            provenance: Vec::new(),
            promoted_at: None,
        };
        // Causal chain a → b → c (cause = from_entity).
        graph
            .add_relationship(mk(a, b, RelationType::Triggers))
            .unwrap();
        graph
            .add_relationship(mk(b, c, RelationType::Triggers))
            .unwrap();

        // Origin of the effect c is the root a (walk c ← b ← a), NOT the proximal
        // cause b. This is the exact failure mode spreading activation cannot solve.
        let origins = graph.trace_causal_origins(&[c], 8).unwrap();
        assert_eq!(origins.len(), 1);
        assert_eq!(origins[0].0, a);
        // Two hops at strength 0.8 each with HOP_DECAY 0.7: the score is the path
        // product, strictly positive and below a single-hop score.
        assert!(
            origins[0].1 > 0.0 && origins[0].1 < 0.7,
            "two-hop origin score should be decayed: {}",
            origins[0].1
        );

        // A non-causal edge into c must NOT be followed (co-occurrence is not cause).
        let d = Uuid::new_v4();
        graph
            .add_relationship(mk(d, c, RelationType::CoOccurs))
            .unwrap();
        let origins = graph.trace_causal_origins(&[c], 8).unwrap();
        assert_eq!(origins.len(), 1);
        assert_eq!(origins[0].0, a);
    }

    // ------------------------------------------------------------------
    // record_memory_coactivation_impl: strengthen-only contract (2026-07-10,
    // f6b730ee). The both-modes contract (mint on strengthen_only=false;
    // strengthen-not-mint on strengthen_only=true given a prior edge; mint
    // nothing on strengthen_only=true given no prior edge) is already
    // pinned by the pre-existing `coactivation_strengthen_only_creates_no_new_edges`
    // and `coactivation_strengthen_only_still_strengthens_existing` tests
    // above/below in this module. Neither of those, however, verifies that
    // "strengthen" actually mutates the edge — only that the impl's return
    // count matches. That gap is what the test below fills. Together, these
    // three are the mode-contract holder that
    // tests/brutal_stress_tests.rs::test_brutal_dense_graph now points back
    // to instead of asserting a specific pair count itself.
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // Tier census + declared read filter
    // ------------------------------------------------------------------

    /// A relationship edge at an explicit strength, in the L1 birth tier.
    fn universe_edge(
        from: Uuid,
        to: Uuid,
        relation_type: RelationType,
        strength: f32,
    ) -> RelationshipEdge {
        RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: from,
            to_entity: to,
            relation_type,
            strength,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: Utc::now(),
            activation_count: 1,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L1Working,
            entity_confidence: None,
            forman_curvature: None,
            endpoint_selectivity: None,
            provenance: Vec::new(),
            promoted_at: None,
        }
    }

    /// A bare entity node for the universe fixtures.
    fn universe_entity(name: &str) -> EntityNode {
        EntityNode {
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            labels: vec![EntityLabel::Concept],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            selectivity: None,
            fine_type: None,
            kb_id: None,
        }
    }

    /// Build a graph with two entities joined by a generic edge of the given
    /// strength, plus a typed edge over the same pair when `also_typed`.
    fn universe_fixture(
        generic_strength: f32,
        also_typed: bool,
    ) -> (GraphMemory, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(dir.path(), None).unwrap();

        let a = graph.add_entity(universe_entity("alpha")).unwrap();
        let b = graph.add_entity(universe_entity("beta")).unwrap();

        graph
            .add_relationship(universe_edge(
                a,
                b,
                RelationType::CoOccurs,
                generic_strength,
            ))
            .unwrap();

        if also_typed {
            graph
                .add_relationship(universe_edge(a, b, RelationType::Causes, 0.8))
                .unwrap();
        }

        (graph, dir)
    }

    #[test]
    fn tier_census_counts_edges_and_totals_agree() {
        let (graph, _dir) = universe_fixture(0.9, true);
        let census = graph.edge_tier_census().unwrap();

        assert_eq!(census.total_scanned, 2, "both edges scanned");
        assert_eq!(
            census.l1_working + census.l2_episodic + census.l3_semantic,
            census.total_scanned,
            "tier counts must partition the scanned set — a mismatch means a \
             decode failure was swallowed"
        );
        assert!(
            census.l1_mean_strength > 0.0,
            "a populated tier must report a mean strength"
        );
    }

    #[test]
    fn tier_census_reports_zero_means_for_empty_tiers() {
        // Guards against dividing by an empty count.
        let (graph, _dir) = universe_fixture(0.5, false);
        let census = graph.edge_tier_census().unwrap();
        assert_eq!(census.l3_semantic, 0);
        assert_eq!(census.l3_mean_strength, 0.0);
    }

    #[test]
    fn read_filter_hides_weak_generic_edges_and_declares_it() {
        // Below L1_PRUNE_THRESHOLD: the engine already considers this edge dead.
        let (graph, _dir) = universe_fixture(crate::constants::L1_PRUNE_THRESHOLD - 0.01, false);
        let universe = graph.get_universe().unwrap();

        assert!(
            universe.connections.is_empty(),
            "a generic edge below the prune threshold must not render"
        );
        assert_eq!(universe.filter.hidden_weak_generic, 1);
        assert_eq!(
            universe.total_connections, 1,
            "total_connections reports what EXISTS, not what was drawn — a viewer \
             must be able to tell it is seeing a subset"
        );
    }

    #[test]
    fn read_filter_never_hides_typed_edges() {
        // A typed relation is an extraction result. Even at a strength far below
        // the generic floor it must render, or the graph misrepresents what the
        // system knows.
        let dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(dir.path(), None).unwrap();
        let a = graph.add_entity(universe_entity("cause")).unwrap();
        let b = graph.add_entity(universe_entity("effect")).unwrap();
        graph
            .add_relationship(universe_edge(a, b, RelationType::Causes, 0.001))
            .unwrap();

        let universe = graph.get_universe().unwrap();
        assert_eq!(universe.connections.len(), 1, "typed edge must survive");
        assert_eq!(universe.filter.hidden_weak_generic, 0);
    }

    #[test]
    fn read_filter_hides_generic_edge_redundant_with_a_typed_one() {
        let (graph, _dir) = universe_fixture(0.9, true);
        let universe = graph.get_universe().unwrap();

        assert_eq!(
            universe.connections.len(),
            1,
            "only the typed edge should be drawn for a pair that has both"
        );
        assert_eq!(universe.filter.hidden_redundant_generic, 1);
        assert_eq!(universe.total_connections, 2);
    }

    #[test]
    fn read_filter_can_be_disabled_to_show_the_raw_substrate() {
        // The filter is a view, not a deletion: the edges are still there and a
        // caller can ask for all of them.
        let (graph, _dir) = universe_fixture(0.9, true);
        let raw = graph
            .get_universe_filtered(UniverseFilter {
                min_generic_strength: 0.0,
                hide_redundant_generic: false,
            })
            .unwrap();

        assert_eq!(raw.connections.len(), 2, "both edges render unfiltered");
        assert_eq!(raw.filter.hidden_redundant_generic, 0);
        assert_eq!(raw.filter.hidden_weak_generic, 0);
    }

    #[test]
    fn coactivation_strengthen_only_actually_increments_activation_and_strength() {
        // Complements `coactivation_strengthen_only_still_strengthens_existing`,
        // which only checks the impl's returned count (3). This test checks
        // the actual edge mutation: activation_count increments and strength
        // increases (Hebbian `RelationshipEdge::strengthen()`), and that
        // relationship_count does NOT grow across the strengthen-only pass
        // (no edge minted on top of the seeded ones).
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let ids: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect(); // 3 pairs

        // Seed pre-existing edges via strengthen_only=false (the only way to
        // populate the `mem_edge:` index `find_edge_between_entities` reads —
        // see the diagnosis's "Flagged finding": nothing else writes it).
        let minted = graph.record_memory_coactivation_impl(&ids, false).unwrap();
        assert_eq!(minted, 3, "seed step should mint all 3 pairs");
        let seeded_count = graph.get_stats().unwrap().relationship_count;
        assert_eq!(seeded_count, 3);

        let (a, b) = (ids[0], ids[1]);
        let before = graph
            .find_edge_between_entities(&a, &b)
            .unwrap()
            .expect("edge must exist after seeding");
        assert_eq!(before.activation_count, 1);

        // Re-run co-activation for the same pairs under the DEFAULT gate.
        let strengthened = graph.record_memory_coactivation_impl(&ids, true).unwrap();

        assert_eq!(
            strengthened, 3,
            "all 3 pre-existing edges should be strengthened, none skipped"
        );
        assert_eq!(
            graph.get_stats().unwrap().relationship_count,
            seeded_count,
            "strengthen_only=true must NOT mint any new edge on top of the seeded ones"
        );

        let after = graph
            .find_edge_between_entities(&a, &b)
            .unwrap()
            .expect("edge must still exist");
        assert_eq!(
            after.activation_count, 2,
            "the pre-existing edge must be strengthened (activation_count incremented)"
        );
        assert!(
            after.strength > before.strength,
            "Hebbian strengthening must increase edge strength: before={}, after={}",
            before.strength,
            after.strength
        );
    }

    // ------------------------------------------------------------------
    // Memory-to-memory coactivation DURABILITY contract.
    //
    // These three tests hold the intent that
    // tests/hebbian_learning_tests.rs::test_hebbian_graph_persists_across_restart,
    // ::test_hebbian_edge_strength_persists and ::test_ltp_persists_across_restart
    // used to hold. Those integration tests reached the durability question
    // only THROUGH minting: they called `reinforce_recall` to create
    // CoRetrieved edges, then restarted the `MemorySystem` and asserted the
    // edges were still there. Since `f6b730ee` (2026-07-10) flipped
    // `SHODH_COACT_STRENGTHEN_ONLY` to default-ON, `record_memory_coactivation`
    // no longer mints memory-to-memory edges, so those tests' setup produces
    // an empty graph and the restart assertions became unreachable — pinning
    // them at zero on both sides would have silently deleted three
    // persistence tests. The durability question is real and independent of
    // the gate, so it is tested here instead, where
    // `record_memory_coactivation_impl(&ids, false)` is reachable by
    // parameter (no env var, no process-global state, hermetic under
    // parallel test execution — nothing outside the mint branch writes the
    // `mem_edge:` pair index these edges are found through, so `tests/` has
    // no public seeding API).
    //
    // A remove-vs-revive decision on the memory-to-memory coactivation layer
    // is PENDING. These tests deliberately pin durability only; they say
    // nothing about whether the layer should mint by default. If the layer is
    // revived, the integration tests can point back at minting; if it is
    // removed, these go with it.
    // ------------------------------------------------------------------

    #[test]
    fn coactivation_edges_survive_graph_reopen() {
        // Durability half of `test_hebbian_graph_persists_across_restart`:
        // memory-to-memory CoRetrieved edges written by coactivation are
        // durable across a close/reopen of the same RocksDB path, and remain
        // reachable through the `mem_edge:` pair index (not merely present in
        // the relationships CF).
        let temp_dir = tempfile::tempdir().unwrap();
        let ids: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
        let (a, b) = (ids[0], ids[1]);

        let edge_uuid;
        let relationship_count_before;
        {
            let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
            let minted = graph.record_memory_coactivation_impl(&ids, false).unwrap();
            assert_eq!(minted, 3, "C(3,2) = 3 pairs should be minted");
            relationship_count_before = graph.get_stats().unwrap().relationship_count;
            assert_eq!(relationship_count_before, 3);
            edge_uuid = graph
                .find_edge_between_entities(&a, &b)
                .unwrap()
                .expect("edge must exist after minting")
                .uuid;
        }
        // Graph dropped — simulates a process restart on the same store.

        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        assert_eq!(
            graph.get_stats().unwrap().relationship_count,
            relationship_count_before,
            "coactivation edges must survive a reopen of the same store"
        );
        let reopened = graph
            .find_edge_between_entities(&a, &b)
            .unwrap()
            .expect("the mem_edge: pair index must survive the reopen too");
        assert_eq!(
            reopened.uuid, edge_uuid,
            "the reopened edge must be the same edge, not a re-mint"
        );
        assert_eq!(reopened.relation_type, RelationType::CoRetrieved);
    }

    #[test]
    fn coactivation_edge_strength_survives_graph_reopen() {
        // Durability half of `test_hebbian_edge_strength_persists`: repeated
        // co-activation raises Hebbian edge strength, and the RAISED value —
        // not the initial tier weight — is what comes back after a reopen.
        let temp_dir = tempfile::tempdir().unwrap();
        let ids: Vec<Uuid> = (0..2).map(|_| Uuid::new_v4()).collect();
        let (a, b) = (ids[0], ids[1]);

        let strength_before;
        let activation_count_before;
        {
            let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
            // Mint once, then strengthen 9 more times through the shipped
            // default path (strengthen_only=true) — the edge now exists, so
            // the default gate reinforces it rather than skipping it.
            graph.record_memory_coactivation_impl(&ids, false).unwrap();
            let initial = graph
                .find_edge_between_entities(&a, &b)
                .unwrap()
                .expect("edge must exist after minting")
                .strength;
            for _ in 0..9 {
                graph.record_memory_coactivation_impl(&ids, true).unwrap();
            }
            let edge = graph
                .find_edge_between_entities(&a, &b)
                .unwrap()
                .expect("edge must still exist");
            strength_before = edge.strength;
            activation_count_before = edge.activation_count;
            assert!(
                strength_before > initial,
                "10 co-activations must raise strength above the initial tier weight: \
                 initial={initial}, after={strength_before}"
            );
            assert_eq!(activation_count_before, 10);
        }

        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let edge = graph
            .find_edge_between_entities(&a, &b)
            .unwrap()
            .expect("edge must exist after reopen");
        assert!(
            (edge.strength - strength_before).abs() < 1e-6,
            "edge strength must persist exactly across reopen: before={}, after={}",
            strength_before,
            edge.strength
        );
        assert_eq!(
            edge.activation_count, activation_count_before,
            "activation_count must persist across reopen"
        );
    }

    #[test]
    fn coactivation_ltp_status_survives_graph_reopen() {
        // Durability half of `test_ltp_persists_across_restart`: whatever LTP
        // state many co-activations produced is the state that comes back.
        // Asserted as an equality against the pre-restart value rather than
        // against a hard-coded LtpStatus, because promotion depends on
        // `detect_ltp_status` thresholds (activation timestamps are only
        // recorded for L2+ edges; a freshly minted CoRetrieved edge is
        // L1Working) — this test pins DURABILITY, not the promotion rule.
        let temp_dir = tempfile::tempdir().unwrap();
        let ids: Vec<Uuid> = (0..2).map(|_| Uuid::new_v4()).collect();
        let (a, b) = (ids[0], ids[1]);

        let ltp_before;
        let strength_before;
        {
            let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
            graph.record_memory_coactivation_impl(&ids, false).unwrap();
            for _ in 0..14 {
                graph.record_memory_coactivation_impl(&ids, true).unwrap();
            }
            let edge = graph
                .find_edge_between_entities(&a, &b)
                .unwrap()
                .expect("edge must exist");
            ltp_before = edge.ltp_status;
            strength_before = edge.strength;
            assert_eq!(edge.activation_count, 15);
            assert!(
                strength_before > 0.8,
                "15 co-activations must drive strength high: {strength_before}"
            );
        }

        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let edge = graph
            .find_edge_between_entities(&a, &b)
            .unwrap()
            .expect("edge must exist after reopen");
        assert_eq!(
            edge.ltp_status.priority(),
            ltp_before.priority(),
            "LTP status must persist across reopen: before={:?}, after={:?}",
            ltp_before,
            edge.ltp_status
        );
        assert!(
            (edge.strength - strength_before).abs() < 1e-6,
            "potentiated strength must persist across reopen: before={}, after={}",
            strength_before,
            edge.strength
        );
    }

    #[test]
    fn typed_neighbors_respects_relation_and_direction() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let caroline = Uuid::new_v4();
        let denver = Uuid::new_v4();
        let painting = Uuid::new_v4();
        let melanie = Uuid::new_v4();
        let mk = |from: Uuid, to: Uuid, rt: RelationType, strength: f32| RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: from,
            to_entity: to,
            relation_type: rt,
            strength,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: Utc::now(),
            activation_count: 1,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic,
            entity_confidence: None,
            forman_curvature: None,
            endpoint_selectivity: None,
            provenance: Vec::new(),
            promoted_at: None,
        };
        // Caroline --LocatedIn--> Denver; Melanie --CreatedBy--> painting;
        // plus a CoOccurs distractor edge Caroline--painting.
        graph
            .add_relationship(mk(caroline, denver, RelationType::LocatedIn, 0.8))
            .unwrap();
        graph
            .add_relationship(mk(melanie, painting, RelationType::CreatedBy, 0.9))
            .unwrap();
        graph
            .add_relationship(mk(caroline, painting, RelationType::CoOccurs, 0.9))
            .unwrap();

        // Outgoing LocatedIn from Caroline → Denver only (CoOccurs filtered).
        let out = graph
            .typed_neighbors(&[caroline], &[RelationType::LocatedIn], false, 64)
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, denver);

        // Incoming CreatedBy into the painting → Melanie (the creator).
        let creators = graph
            .typed_neighbors(&[painting], &[RelationType::CreatedBy], true, 64)
            .unwrap();
        assert_eq!(creators.len(), 1);
        assert_eq!(creators[0].0, melanie);

        // Wrong direction yields nothing: nobody is LocatedIn Caroline.
        let incoming_loc = graph
            .typed_neighbors(&[caroline], &[RelationType::LocatedIn], true, 64)
            .unwrap();
        assert!(incoming_loc.is_empty());
    }

    #[test]
    fn learned_pair_relation_gates() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let person = EntityLabel::Person;
        let org = EntityLabel::Organization;

        // Below support: 9 observations → None (threshold 10 after the v1
        // rejection, run 27348362950).
        for _ in 0..9 {
            graph
                .record_relation_evidence(&person, &org, &RelationType::WorksAt)
                .unwrap();
        }
        assert!(graph
            .lookup_learned_pair_relation(&person, &org)
            .unwrap()
            .is_none());

        // Tenth observation clears support: mapping earned, direction settled
        // (source = Person in all evidence), and it maps to the caller's
        // mention order in both argument orders.
        graph
            .record_relation_evidence(&person, &org, &RelationType::WorksAt)
            .unwrap();
        let (rt, a_is_source, support) = graph
            .lookup_learned_pair_relation(&person, &org)
            .unwrap()
            .expect("mapping earned at support 10");
        assert_eq!(rt, RelationType::WorksAt);
        assert!(a_is_source, "Person mentioned first is the source");
        assert_eq!(support, 10);
        let (_, a_is_source, _) = graph
            .lookup_learned_pair_relation(&org, &person)
            .unwrap()
            .expect("swapped order still maps");
        assert!(!a_is_source, "Org mentioned first is the target");

        // Purity gate: a near-tie between relations stays generic.
        for _ in 0..10 {
            graph
                .record_relation_evidence(&person, &org, &RelationType::Manages)
                .unwrap();
        }
        assert!(
            graph
                .lookup_learned_pair_relation(&person, &org)
                .unwrap()
                .is_none(),
            "10 WorksAt vs 10 Manages = purity 0.5 → no mapping"
        );

        // Causal exclusion: statistics may NEVER assign causal relations —
        // a defaulted causal edge is lineage poison (the fragment-bridge class).
        let event = EntityLabel::Event;
        let tech = EntityLabel::Technology;
        for _ in 0..12 {
            graph
                .record_relation_evidence(&event, &tech, &RelationType::Triggers)
                .unwrap();
        }
        assert!(
            graph
                .lookup_learned_pair_relation(&event, &tech)
                .unwrap()
                .is_none(),
            "causal relations must never be statistically defaulted"
        );

        // Generic-label exclusion: catch-all labels (Concept/Keyword/Other)
        // never record and never map — the v1 mass-application vector.
        let concept = EntityLabel::Concept;
        for _ in 0..12 {
            graph
                .record_relation_evidence(&concept, &org, &RelationType::WorksAt)
                .unwrap();
        }
        assert!(
            graph
                .lookup_learned_pair_relation(&concept, &org)
                .unwrap()
                .is_none(),
            "generic labels must never carry learned mappings"
        );
    }

    #[test]
    fn classify_tag_label_maps_known_categories() {
        assert_eq!(classify_tag_label("production"), EntityLabel::Environment);
        assert_eq!(classify_tag_label("rocksdb"), EntityLabel::Database);
        assert_eq!(classify_tag_label("metrics-service"), EntityLabel::Service);
        assert_eq!(classify_tag_label("README"), EntityLabel::Document);
        assert_eq!(
            classify_tag_label("config.toml"),
            EntityLabel::Configuration
        );
        assert_eq!(classify_tag_label("router.rs"), EntityLabel::Module);
        assert_eq!(classify_tag_label("ci-cd"), EntityLabel::Pipeline);
        // Unknown → Technology fallback
        // CHANGED: the fallthrough was `Technology`; it is now `Concept`.
        // "widgetron" matches none of the rules above, and labelling an
        // unrecognised surface `Technology` is an assertion the matcher cannot
        // support — it made every unrecognised tag indistinguishable from a
        // confidently-classified one, and rendered the graph as a flat wall of
        // "Technology". `Concept` is what the NER path already emits for
        // unresolved classes, so the two agree.
        assert_eq!(classify_tag_label("widgetron"), EntityLabel::Concept);
    }

    #[test]
    fn entity_node_fine_type_roundtrips() {
        // Same persistence path production code uses: GraphMemory::add_entity
        // (crate::serialization::encode over postcard) and get_entity
        // (decode_entity_node / try_decode_compat).
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let now = Utc::now();

        let entity = EntityNode {
            uuid: Uuid::new_v4(),
            name: "Francis Scott Key Bridge".to_string(),
            labels: vec![EntityLabel::Facility],
            created_at: now,
            last_seen_at: now,
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: true,
            selectivity: None,
            fine_type: Some("bridge".to_string()),
            kb_id: None,
        };
        let entity_uuid = graph.add_entity(entity).unwrap();

        let roundtripped = graph.get_entity(&entity_uuid).unwrap().unwrap();
        assert_eq!(roundtripped.fine_type, Some("bridge".to_string()));
    }

    #[test]
    fn add_entity_stamps_kb_id_from_the_offline_kb() {
        // The always-on half of KB linking: an unambiguous organisation mention
        // acquires its Wikidata identity as a side effect of being stored, with
        // no extra I/O and no configuration.
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let now = Utc::now();

        let entity = EntityNode {
            uuid: Uuid::new_v4(),
            name: "International Business Machines".to_string(),
            labels: vec![EntityLabel::Organization],
            created_at: now,
            last_seen_at: now,
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: true,
            selectivity: None,
            fine_type: None,
            kb_id: None,
        };
        let uuid = graph.add_entity(entity).unwrap();
        let stored = graph.get_entity(&uuid).unwrap().unwrap();
        assert_eq!(
            stored.kb_id,
            Some("Q37156".to_string()),
            "an unambiguous org mention must acquire its KB identity on write"
        );
    }

    #[test]
    fn add_entity_remention_never_overwrites_an_established_kb_id() {
        // Write-once identity. A node that already carries a QID keeps it even if
        // a later mention arrives claiming a different one — silently repointing
        // an identity is precisely the corruption KB linking exists to prevent.
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let now = Utc::now();

        let make = |kb: Option<String>| EntityNode {
            uuid: Uuid::new_v4(),
            // Not in the KB, so the automatic stamp cannot interfere and the test
            // exercises only the merge rule.
            name: "Zzyzx Internal Platform".to_string(),
            labels: vec![EntityLabel::Organization],
            created_at: now,
            last_seen_at: now,
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: true,
            selectivity: None,
            fine_type: None,
            kb_id: kb,
        };

        let uuid = graph.add_entity(make(Some("Q1".to_string()))).unwrap();

        // A re-mention carrying a DIFFERENT id must not win...
        let uuid2 = graph.add_entity(make(Some("Q999".to_string()))).unwrap();
        assert_eq!(uuid, uuid2, "re-mention should merge into the same node");
        assert_eq!(
            graph.get_entity(&uuid).unwrap().unwrap().kb_id,
            Some("Q1".to_string()),
            "a conflicting re-mention must not repoint an established identity"
        );

        // ...and a re-mention carrying none must not erase it.
        graph.add_entity(make(None)).unwrap();
        assert_eq!(
            graph.get_entity(&uuid).unwrap().unwrap().kb_id,
            Some("Q1".to_string()),
            "an unlinked re-mention must not wipe an existing identity"
        );
    }

    #[test]
    fn add_entity_leaves_ambiguous_and_generic_mentions_unlinked() {
        // The abstain path, end to end through storage: an ambiguous acronym and
        // a non-linkable label both come back with no identity rather than a
        // guessed one.
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let now = Utc::now();

        let make = |name: &str, label: EntityLabel| EntityNode {
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            labels: vec![label],
            created_at: now,
            last_seen_at: now,
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: true,
            selectivity: None,
            fine_type: None,
            kb_id: None,
        };

        for (name, label) in [
            ("ACM", EntityLabel::Organization), // ambiguous: two real orgs
            ("IBM", EntityLabel::Person),       // right surface, wrong type
            ("IBM", EntityLabel::Concept),      // generic label never links
            ("Zzyzx Holdings", EntityLabel::Organization), // unknown
        ] {
            let uuid = graph.add_entity(make(name, label.clone())).unwrap();
            assert_eq!(
                graph.get_entity(&uuid).unwrap().unwrap().kb_id,
                None,
                "{name} as {label:?} must abstain, not guess"
            );
        }
    }

    #[test]
    fn add_entity_remention_preserves_existing_fine_type() {
        // Deferred fix: a re-mention that carries no fine type must NOT wipe the
        // fine type an earlier mention (e.g. GLiNER) already set. Re-mentions are
        // the common case (pre-extracted names, tags, fallback entities).
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let now = Utc::now();

        let make = |fine: Option<String>| EntityNode {
            uuid: Uuid::new_v4(),
            name: "Baltimore".to_string(),
            labels: vec![EntityLabel::Gpe],
            created_at: now,
            last_seen_at: now,
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: true,
            selectivity: None,
            fine_type: fine,
            kb_id: None,
        };

        // First mention: GLiNER typed it "city".
        let uuid = graph.add_entity(make(Some("city".to_string()))).unwrap();
        // Re-mention with no fine type (e.g. a tag or fallback entity).
        let uuid2 = graph.add_entity(make(None)).unwrap();
        assert_eq!(uuid, uuid2, "re-mention should merge into the same node");

        let merged = graph.get_entity(&uuid).unwrap().unwrap();
        assert_eq!(
            merged.fine_type,
            Some("city".to_string()),
            "re-mention without a fine type must preserve the existing one"
        );
        assert_eq!(merged.mention_count, 2);
    }

    #[test]
    fn legacy_entity_node_defaults_fine_type_to_none() {
        // Mirrors the pre-fine_type EntityNode schema (all fields through
        // `selectivity`, no trailing `fine_type`) — the exact shape of every
        // EntityNode already written to the live RocksDB store before this
        // field existed. `decode_entity_node` must backfill fine_type=None
        // via ENTITY_NODE_DEFAULT_SUFFIX rather than failing to decode.
        #[derive(serde::Serialize)]
        struct LegacyEntityNode {
            uuid: Uuid,
            name: String,
            labels: Vec<EntityLabel>,
            created_at: DateTime<Utc>,
            last_seen_at: DateTime<Utc>,
            mention_count: usize,
            summary: String,
            attributes: HashMap<String, String>,
            name_embedding: Option<Vec<f32>>,
            salience: f32,
            is_proper_noun: bool,
            selectivity: Option<f32>,
        }

        let now = Utc::now();
        let uuid = Uuid::new_v4();
        let legacy = LegacyEntityNode {
            uuid,
            name: "Legacy Bridge".to_string(),
            labels: vec![EntityLabel::Facility],
            created_at: now,
            last_seen_at: now,
            mention_count: 3,
            summary: "a pre-fine_type record".to_string(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.7,
            is_proper_noun: true,
            selectivity: Some(0.42),
        };

        let bytes = crate::serialization::encode(&legacy).unwrap();
        let (decoded, needs_migration) = decode_entity_node(&bytes).unwrap();

        assert_eq!(decoded.uuid, uuid);
        assert_eq!(decoded.name, "Legacy Bridge");
        assert_eq!(decoded.selectivity, Some(0.42));
        assert_eq!(
            decoded.fine_type, None,
            "legacy record without fine_type must default to None"
        );
        assert!(
            needs_migration,
            "legacy-shaped record should be flagged for rewrite-on-read"
        );
    }

    #[test]
    fn legacy_entity_node_defaults_kb_id_to_none() {
        // The shape of every EntityNode already in the live store before `kb_id`
        // existed: all fields through `fine_type`, nothing after. postcard is
        // positional with no EOF tolerance, so without the extra `0x00` in
        // ENTITY_NODE_DEFAULT_SUFFIX this record fails to decode at all — which
        // would make an upgraded store unreadable, not merely un-linked.
        #[derive(serde::Serialize)]
        struct PreKbIdEntityNode {
            uuid: Uuid,
            name: String,
            labels: Vec<EntityLabel>,
            created_at: DateTime<Utc>,
            last_seen_at: DateTime<Utc>,
            mention_count: usize,
            summary: String,
            attributes: HashMap<String, String>,
            name_embedding: Option<Vec<f32>>,
            salience: f32,
            is_proper_noun: bool,
            selectivity: Option<f32>,
            fine_type: Option<String>,
        }

        let now = Utc::now();
        let uuid = Uuid::new_v4();
        let legacy = PreKbIdEntityNode {
            uuid,
            name: "IBM".to_string(),
            labels: vec![EntityLabel::Organization],
            created_at: now,
            last_seen_at: now,
            mention_count: 7,
            summary: "written before kb_id existed".to_string(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.8,
            is_proper_noun: true,
            selectivity: Some(0.31),
            fine_type: Some("company".to_string()),
        };

        let bytes = crate::serialization::encode(&legacy).unwrap();
        let (decoded, needs_migration) = decode_entity_node(&bytes).unwrap();

        assert_eq!(decoded.uuid, uuid);
        assert_eq!(decoded.name, "IBM");
        assert_eq!(decoded.selectivity, Some(0.31));
        assert_eq!(decoded.fine_type, Some("company".to_string()));
        assert_eq!(
            decoded.kb_id, None,
            "a pre-kb_id record must decode with kb_id defaulted, not fail"
        );
        assert!(needs_migration);
    }

    #[test]
    fn distinct_kb_identities_veto_a_canonicalizer_merge() {
        // The canonicalizer clusters on surface similarity, so "Apollo Global
        // Management" and "Apollo Theatre" can score as one entity. When both
        // carry a KB identity and the identities differ, the merge must be
        // refused — this is the one signal the string matcher structurally
        // cannot have.
        assert!(!kb_identities_permit_merge(Some("Q1"), Some("Q2")));

        // Agreement is a positive signal, not merely permission.
        assert!(kb_identities_permit_merge(Some("Q1"), Some("Q1")));

        // Silence is not evidence of difference: an unlinked node (the common
        // case, since linking abstains by default) must not block a merge the
        // matcher is confident about, or KB linking would make canonicalization
        // strictly worse.
        assert!(kb_identities_permit_merge(None, Some("Q1")));
        assert!(kb_identities_permit_merge(Some("Q1"), None));
        assert!(kb_identities_permit_merge(None, None));
    }

    #[test]
    fn entity_node_kb_id_roundtrips() {
        let now = Utc::now();
        let node = EntityNode {
            uuid: Uuid::new_v4(),
            name: "IBM".to_string(),
            labels: vec![EntityLabel::Organization],
            created_at: now,
            last_seen_at: now,
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: true,
            selectivity: None,
            fine_type: None,
            kb_id: Some("Q37156".to_string()),
        };
        let bytes = crate::serialization::encode(&node).unwrap();
        let (decoded, needs_migration) = decode_entity_node(&bytes).unwrap();
        assert_eq!(decoded.kb_id, Some("Q37156".to_string()));
        assert!(
            !needs_migration,
            "a current-shape record needs no migration"
        );
    }

    // =====================================================================
    // RelationshipEdge postcard schema evolution.
    //
    // There was NO old-bytes round-trip test for `RelationshipEdge` before
    // this — the provenance increment took EDGE_PROVENANCE_DEFAULT_SUFFIX
    // from 2 bytes to 3 with the compat path unexercised. Postcard is
    // positional and carries no field presence, so a wrong suffix does not
    // fail loudly: it silently mis-decodes every edge in a live store. These
    // two tests exercise the two legacy generations that can exist on disk.
    // =====================================================================

    /// A `RelationshipEdge` exactly as serialized before `promoted_at` existed:
    /// every field through `provenance`, nothing after it. This is the shape of
    /// every edge already in the live RocksDB store.
    #[derive(serde::Serialize)]
    struct LegacyEdgeThroughProvenance {
        uuid: Uuid,
        from_entity: Uuid,
        to_entity: Uuid,
        relation_type: RelationType,
        strength: f32,
        created_at: DateTime<Utc>,
        valid_at: DateTime<Utc>,
        invalidated_at: Option<DateTime<Utc>>,
        source_episode_id: Option<Uuid>,
        context: String,
        last_activated: DateTime<Utc>,
        activation_count: u32,
        ltp_status: LtpStatus,
        tier: EdgeTier,
        activation_timestamps: Option<VecDeque<DateTime<Utc>>>,
        entity_confidence: Option<f32>,
        endpoint_selectivity: Option<f32>,
        forman_curvature: Option<f32>,
        provenance: Vec<ProvenanceRecord>,
    }

    /// The older generation still: stops at `entity_confidence`, so it is short
    /// by FOUR fields. `try_decode_compat` must append the defaults one at a
    /// time to reach it.
    #[derive(serde::Serialize)]
    struct LegacyEdgeThroughEntityConfidence {
        uuid: Uuid,
        from_entity: Uuid,
        to_entity: Uuid,
        relation_type: RelationType,
        strength: f32,
        created_at: DateTime<Utc>,
        valid_at: DateTime<Utc>,
        invalidated_at: Option<DateTime<Utc>>,
        source_episode_id: Option<Uuid>,
        context: String,
        last_activated: DateTime<Utc>,
        activation_count: u32,
        ltp_status: LtpStatus,
        tier: EdgeTier,
        activation_timestamps: Option<VecDeque<DateTime<Utc>>>,
        entity_confidence: Option<f32>,
    }

    #[test]
    fn legacy_edge_without_promoted_at_decodes_to_none() {
        let now = Utc::now();
        let uuid = Uuid::new_v4();
        let from = Uuid::new_v4();
        let to = Uuid::new_v4();
        let episode = Uuid::new_v4();

        let legacy = LegacyEdgeThroughProvenance {
            uuid,
            from_entity: from,
            to_entity: to,
            relation_type: RelationType::Causes,
            strength: 0.63,
            created_at: now,
            valid_at: now,
            invalidated_at: None,
            source_episode_id: Some(episode),
            context: "a pre-promoted_at record".to_string(),
            last_activated: now,
            activation_count: 7,
            ltp_status: LtpStatus::Weekly,
            tier: EdgeTier::L2Episodic,
            activation_timestamps: None,
            entity_confidence: Some(0.8),
            endpoint_selectivity: Some(0.31),
            forman_curvature: Some(-2.0),
            provenance: vec![ProvenanceRecord {
                source_episode_id: episode,
                mention_count: 2,
                first_observed: now,
                last_observed: now,
                confidence: Some(0.9),
                evidence_span: Some((3, 11)),
                typed_by: Some(TypingMethod::Cue),
            }],
        };

        let bytes = crate::serialization::encode(&legacy).unwrap();
        let (decoded, needs_migration) = decode_relationship_edge(&bytes).unwrap();

        // Everything that WAS on disk must survive untouched — a wrong suffix
        // length would shift the tail and corrupt these silently.
        assert_eq!(decoded.uuid, uuid);
        assert_eq!(decoded.from_entity, from);
        assert_eq!(decoded.to_entity, to);
        assert_eq!(decoded.relation_type, RelationType::Causes);
        assert!((decoded.strength - 0.63).abs() < 1e-6);
        assert_eq!(decoded.activation_count, 7);
        assert_eq!(decoded.ltp_status, LtpStatus::Weekly);
        assert_eq!(decoded.tier, EdgeTier::L2Episodic);
        assert_eq!(decoded.entity_confidence, Some(0.8));
        assert_eq!(decoded.endpoint_selectivity, Some(0.31));
        assert_eq!(decoded.forman_curvature, Some(-2.0));
        assert_eq!(decoded.provenance.len(), 1);
        assert_eq!(decoded.provenance[0].mention_count, 2);
        assert_eq!(decoded.provenance[0].evidence_span, Some((3, 11)));

        // ...and the new field backfills to None.
        assert_eq!(
            decoded.promoted_at, None,
            "a record written before promoted_at existed must default to None"
        );
        assert!(
            needs_migration,
            "legacy-shaped record should be flagged for rewrite-on-read"
        );

        // A legacy L2 edge with promoted_at=None anchors its promotion clock at
        // created_at — it is NOT instantly eligible for L3 just because the
        // field is absent. This is the property that makes the None fallback
        // safe on a live store.
        let mut edge = decoded;
        edge.strength = 0.95;
        assert!(
            edge.try_promote_at(now + Duration::hours(1)).is_none(),
            "1h after birth is short of the 24h L2→L3 separation"
        );
        assert!(
            edge.try_promote_at(now + Duration::hours(25)).is_some(),
            "25h after birth clears it"
        );
    }

    #[test]
    fn legacy_edge_short_by_four_fields_decodes() {
        // The compat path appends defaults ONE AT A TIME; a record missing
        // endpoint_selectivity, forman_curvature, provenance AND promoted_at
        // exercises all four bytes of EDGE_PROVENANCE_DEFAULT_SUFFIX.
        assert_eq!(
            EDGE_PROVENANCE_DEFAULT_SUFFIX.len(),
            4,
            "suffix must carry one default per trailing field added since the postcard cutover"
        );

        let now = Utc::now();
        let uuid = Uuid::new_v4();
        let legacy = LegacyEdgeThroughEntityConfidence {
            uuid,
            from_entity: Uuid::new_v4(),
            to_entity: Uuid::new_v4(),
            relation_type: RelationType::Knows,
            strength: 0.42,
            created_at: now,
            valid_at: now,
            invalidated_at: None,
            source_episode_id: None,
            context: "an even older record".to_string(),
            last_activated: now,
            activation_count: 1,
            ltp_status: LtpStatus::None,
            tier: EdgeTier::L1Working,
            activation_timestamps: None,
            entity_confidence: None,
        };

        let bytes = crate::serialization::encode(&legacy).unwrap();
        let (decoded, needs_migration) = decode_relationship_edge(&bytes).unwrap();

        assert_eq!(decoded.uuid, uuid);
        assert!((decoded.strength - 0.42).abs() < 1e-6);
        assert_eq!(decoded.tier, EdgeTier::L1Working);
        assert_eq!(decoded.endpoint_selectivity, None);
        assert_eq!(decoded.forman_curvature, None);
        assert!(decoded.provenance.is_empty());
        assert_eq!(decoded.promoted_at, None);
        assert!(needs_migration);
    }

    #[test]
    fn current_edge_round_trips_promoted_at() {
        // Forward direction: a promoted_at that IS set must survive encode/decode
        // with no migration flag. Postcard truncates chrono to its serde repr, so
        // compare at second granularity.
        let now = Utc::now();
        let mut edge = create_test_edge_with_tier(0.75, 0, EdgeTier::L2Episodic);
        edge.promoted_at = Some(now);

        let bytes = crate::serialization::encode(&edge).unwrap();
        let (decoded, needs_migration) = decode_relationship_edge(&bytes).unwrap();

        assert!(
            !needs_migration,
            "a current-schema record needs no migration"
        );
        let got = decoded.promoted_at.expect("promoted_at must round-trip");
        assert_eq!(
            got.timestamp(),
            now.timestamp(),
            "promoted_at must survive the postcard round trip"
        );
    }

    // =====================================================================
    // Edge-tier invariant (c): promotion requires SUSTAINED evidence.
    // =====================================================================

    #[test]
    fn a_single_requests_burst_cannot_promote_past_one_step() {
        // The defect this pins: from birth weight 0.4 it took ONE strengthen
        // call to clear L1_PROMOTION_THRESHOLD (0.5) and three to clear
        // L2_PROMOTION_THRESHOLD (0.7) — and three independent strengthen paths
        // touch the same entity edges inside a single request
        // (batch_strengthen_synapses, reinforce_recall_with_momentum,
        // strengthen_episode_entity_edges). One conversational turn could mint an
        // L3 edge carrying EDGE_TIER_TRUST_L3 = 0.80 and a 90-day prune shield.
        //
        // Setup is the WORST case for the gate: an edge already old enough for
        // its first promotion, then hammered. It must advance exactly one tier.
        use crate::constants::*;
        let mut edge = create_test_edge(L1_INITIAL_WEIGHT, 0);
        edge.created_at = Utc::now() - Duration::seconds(TIER_PROMOTION_WORKING_AGE_SECS + 1);
        assert_eq!(edge.tier, EdgeTier::L1Working);

        // 25 strengthen calls, all at the same instant — a single request.
        let request_instant = Utc::now();
        let mut promotions = Vec::new();
        for _ in 0..25 {
            if let Some(p) = edge.strengthen_at(request_instant) {
                promotions.push(p);
            }
        }

        assert_eq!(
            promotions.len(),
            1,
            "a burst inside one instant may advance at most one tier, got {promotions:?}"
        );
        assert_eq!(
            edge.tier,
            EdgeTier::L2Episodic,
            "the edge must stop at L2 no matter how hard it is strengthened"
        );
        // Strength is well past L2_PROMOTION_THRESHOLD — it is the CLOCK, not a
        // strength shortfall, that is holding the edge back.
        assert!(
            edge.strength >= L2_PROMOTION_THRESHOLD,
            "strength {} should already clear the L3 bar",
            edge.strength
        );

        // 23h 59m after the L2 promotion: still not enough.
        let almost = request_instant + Duration::seconds(TIER_PROMOTION_SESSION_AGE_SECS - 60);
        assert!(edge.strengthen_at(almost).is_none());
        assert_eq!(edge.tier, EdgeTier::L2Episodic);

        // Past 24h: the second step is finally allowed, and only one.
        let next_day = request_instant + Duration::seconds(TIER_PROMOTION_SESSION_AGE_SECS + 1);
        let mut later_promotions = Vec::new();
        for _ in 0..25 {
            if let Some(p) = edge.strengthen_at(next_day) {
                later_promotions.push(p);
            }
        }
        assert_eq!(later_promotions.len(), 1, "one step per window");
        assert_eq!(edge.tier, EdgeTier::L3Semantic);
        assert_eq!(edge.promoted_at, Some(next_day));

        // L3 is terminal — no further promotion, ever.
        let much_later = next_day + Duration::days(365);
        assert!(edge.strengthen_at(much_later).is_none());
        assert_eq!(edge.tier, EdgeTier::L3Semantic);
    }

    #[test]
    fn importance_weighted_strengthen_obeys_the_same_promotion_clock() {
        // `strengthen_with_importance` was a hand-copied twin of `strengthen`
        // with its own promotion block. Both now route through
        // `strengthen_scaled_at` → `try_promote_at`, so the gate cannot be
        // bypassed by choosing the other entry point — which matters because the
        // three strengthen paths in one request do not all use the same one.
        use crate::constants::*;
        let mut edge = create_test_edge(L1_INITIAL_WEIGHT, 0);
        edge.created_at = Utc::now() - Duration::seconds(TIER_PROMOTION_WORKING_AGE_SECS + 1);

        let instant = Utc::now();
        let mut promotions = 0;
        for _ in 0..25 {
            if edge.strengthen_with_importance_at(1.0, instant).is_some() {
                promotions += 1;
            }
        }
        assert_eq!(promotions, 1, "the importance path shares the same clock");
        assert_eq!(edge.tier, EdgeTier::L2Episodic);
    }

    // =====================================================================
    // Episode-based promotion: the evidence arm of the gate.
    //
    // The wall-clock arm alone made elapsed minutes a PRECONDITION for
    // consolidation, which a batch ingest can never satisfy — every import,
    // eval run and seeded deployment creates all of its edges within one
    // pass, so the entire graph froze at L1 (trust 0.20) and then aged out
    // on the L1 prune schedule. These tests pin both directions: a burst
    // inside ONE episode must not promote, and genuine evidence across
    // DISTINCT episodes must promote with no waiting.
    // =====================================================================

    /// Build an attestation for `episode`, observed at `at`.
    fn attestation(episode: Uuid, at: DateTime<Utc>) -> ProvenanceRecord {
        ProvenanceRecord {
            source_episode_id: episode,
            mention_count: 1,
            first_observed: at,
            last_observed: at,
            confidence: None,
            evidence_span: None,
            typed_by: None,
        }
    }

    #[test]
    fn a_burst_inside_one_episode_never_promotes() {
        // The anti-burst intent, restated in the units that actually carry it.
        // One conversation mentioning two entities forty times is ONE source
        // episode. It may hammer the edge through every strengthen path there
        // is; with no elapsed time and no second episode, neither arm of the
        // gate opens.
        use crate::constants::*;
        let now = Utc::now();
        let episode = Uuid::new_v4();

        let mut edge = create_test_edge(L1_INITIAL_WEIGHT, 0);
        edge.created_at = now;
        edge.last_activated = now;

        // Forty mentions, all from the same episode, merged the way the ingest
        // path merges them.
        for _ in 0..40 {
            merge_provenance(&mut edge.provenance, attestation(episode, now));
            let promotion = edge.strengthen_at(now);
            assert!(
                promotion.is_none(),
                "a burst inside one episode must not promote, got {promotion:?}"
            );
        }

        assert_eq!(
            edge.distinct_attesting_episodes(),
            1,
            "forty mentions of one episode are one episode"
        );
        assert_eq!(edge.provenance[0].mention_count, 40);
        assert_eq!(
            edge.tier,
            EdgeTier::L1Working,
            "the edge must still be L1 after a forty-mention burst"
        );
        // It is the EVIDENCE, not the strength, that is holding it back — the
        // same property the clock-only gate asserted.
        assert!(
            edge.strength >= L2_PROMOTION_THRESHOLD,
            "strength {} should already clear even the L3 bar",
            edge.strength
        );
    }

    #[test]
    fn distinct_episodes_promote_with_no_wall_clock_wait() {
        // The batch-ingest case: every attestation arrives within the same
        // millisecond, from DIFFERENT source memories. This is real
        // corroboration and must consolidate, which under a clock-only gate it
        // never could.
        use crate::constants::*;
        let now = Utc::now();
        let mut edge = create_test_edge(L1_INITIAL_WEIGHT, 0);
        edge.created_at = now;
        edge.last_activated = now;

        // Episode 1: single observation. Not corroborated yet.
        merge_provenance(&mut edge.provenance, attestation(Uuid::new_v4(), now));
        assert!(edge.strengthen_at(now).is_none());
        assert_eq!(edge.tier, EdgeTier::L1Working);

        // Episode 2, same instant: L1 → L2, with zero elapsed time.
        merge_provenance(&mut edge.provenance, attestation(Uuid::new_v4(), now));
        let promotion = edge.strengthen_at(now);
        assert_eq!(
            promotion,
            Some(("L1Working".to_string(), "L2Episodic".to_string())),
            "two distinct attesting episodes must promote regardless of the clock"
        );
        assert_eq!(edge.tier, EdgeTier::L2Episodic);
        assert_eq!(edge.promoted_at, Some(now));

        // Episode 3, still the same instant: short of TIER_PROMOTION_L3_MIN_EPISODES.
        merge_provenance(&mut edge.provenance, attestation(Uuid::new_v4(), now));
        assert!(
            edge.strengthen_at(now).is_none(),
            "3 episodes is short of the {TIER_PROMOTION_L3_MIN_EPISODES} required for L3"
        );
        assert_eq!(edge.tier, EdgeTier::L2Episodic);

        // Episode 4: L2 → L3. Two attestations BEYOND the two that earned L2 —
        // the cumulative thresholds encode "evidence since entering this tier"
        // without persisting a per-tier counter.
        merge_provenance(&mut edge.provenance, attestation(Uuid::new_v4(), now));
        assert_eq!(
            edge.strengthen_at(now),
            Some(("L2Episodic".to_string(), "L3Semantic".to_string()))
        );
        assert_eq!(edge.tier, EdgeTier::L3Semantic);

        // L3 is still terminal.
        assert!(edge.strengthen_at(now).is_none());
    }

    #[test]
    fn corroboration_still_advances_at_most_one_tier_per_call() {
        // The episode arm must not become a bypass for the one-step invariant:
        // an edge that arrives fully corroborated may not skip L2.
        use crate::constants::*;
        let now = Utc::now();
        let mut edge = create_test_edge(L1_INITIAL_WEIGHT, 0);
        edge.created_at = now;
        for _ in 0..TIER_PROMOTION_L3_MIN_EPISODES + 4 {
            merge_provenance(&mut edge.provenance, attestation(Uuid::new_v4(), now));
        }

        let first = edge.strengthen_at(now);
        assert_eq!(
            first,
            Some(("L1Working".to_string(), "L2Episodic".to_string())),
            "however corroborated, the first step is L1 → L2"
        );
        assert_eq!(edge.tier, EdgeTier::L2Episodic);
    }

    #[test]
    fn a_batch_ingested_edge_is_neither_frozen_at_l1_nor_pruned_out() {
        // The two halves of the defect, end to end on one edge: under a
        // clock-only gate a corroborated batch-ingested edge stayed at L1 with
        // EDGE_TIER_TRUST_L1 = 0.20 and then fell below L1_PRUNE_THRESHOLD on
        // the L1 decay schedule — roughly 79 idle hours — deleting an imported
        // corpus on a timer. Corroboration now lifts it to L2, whose schedule it
        // survives.
        use crate::constants::*;
        let ingest = Utc::now();
        let mut edge = create_test_edge(L1_INITIAL_WEIGHT, 0);
        edge.created_at = ingest;
        edge.last_activated = ingest;

        // Two source memories in the same import pass attest the same pair.
        for _ in 0..2 {
            merge_provenance(&mut edge.provenance, attestation(Uuid::new_v4(), ingest));
            edge.strengthen_at(ingest);
        }
        assert_eq!(
            edge.tier,
            EdgeTier::L2Episodic,
            "a corroborated batch-ingested edge must not be frozen at L1"
        );
        assert!(
            EDGE_TIER_TRUST_L2 > EDGE_TIER_TRUST_L1,
            "and must therefore no longer carry the L1 retrieval-trust penalty"
        );

        // Leaving L1 also unlocks the LTP machinery, which is an L2+ mechanism:
        // `record_activation_timestamp` returns early for L1, so an edge frozen
        // at L1 can never accumulate the activation history that Burst/Weekly
        // LTP are detected from, and can therefore never earn LTP's decay
        // protection or its prune shield, no matter how well attested it is.
        assert!(
            edge.activation_timestamps.is_some(),
            "promotion must seed the activation history that LTP is detected from"
        );

        // Survives the L1 death horizon: an L1 edge from this ingest is prunable
        // by ~59 idle hours (0.55 · e^{-0.029·59} < L1_PRUNE_THRESHOLD).
        let mut frozen = create_test_edge(L1_INITIAL_WEIGHT, 0);
        frozen.created_at = ingest;
        frozen.last_activated = ingest;
        frozen.strengthen_at(ingest);
        assert_eq!(frozen.tier, EdgeTier::L1Working);
        assert!(
            frozen.decay_at(ingest + Duration::hours(59)),
            "a single-attestation L1 edge at strength {} must still be prunable \
             after 59 idle hours — singletons keep dying on the L1 schedule, \
             which is the intended L1 semantics and is NOT what the fix changes",
            frozen.strength
        );

        // The corroborated edge survives the same horizon — but NOT by
        // arithmetic: its effective strength at 59h is ~0.12 against an L2
        // prune threshold of 0.2, so on strength alone it would die EARLIER
        // than the L1 singleton (~41.5h vs ~58.8h), because L2's threshold is
        // double L1's at essentially the same executed decay rate. What saves
        // it is `corroboration_protected()`: the same distinct-episode evidence
        // that earned the promotion also shields it from strength-based
        // pruning. Promotion and protection are one policy; splitting them
        // would consolidate an edge and shorten its life in the same step.
        let idle = ingest + Duration::hours(59);
        assert!(
            !edge.decay_at(idle),
            "the corroborated edge must survive that horizon"
        );
        assert!(
            edge.effective_strength() < edge.tier.prune_threshold(),
            "and it must be surviving on corroboration, not on strength: \
             effective {} vs threshold {}",
            edge.effective_strength(),
            edge.tier.prune_threshold()
        );
        // `decay_at`'s return value is only one of three prune doors. This is
        // the one the lazy on-read path in `get_edges_for_entity` consults —
        // the path that actually deletes edges out from under a live reader.
        assert!(
            edge.is_prune_protected(),
            "the lazy on-read prune path must see the corroboration too"
        );
    }

    #[test]
    fn the_evidence_that_promotes_also_protects() {
        // The load-bearing coupling, pinned so the two thresholds cannot drift
        // apart. If the prune-protection minimum ever exceeds the L1→L2
        // promotion minimum, edges in between are promoted onto a tier with
        // DOUBLE the prune threshold while still unprotected — the fix would
        // then shorten the life of exactly the batch-ingested edges it exists
        // to rescue (~41.5 idle hours at L2 vs ~58.8 at L1, at birth weight).
        use crate::constants::*;
        assert!(
            PROVENANCE_PRUNE_CORROBORATION_MIN_DEFAULT <= TIER_PROMOTION_L2_MIN_EPISODES,
            "prune protection ({PROVENANCE_PRUNE_CORROBORATION_MIN_DEFAULT}) must not \
             require more evidence than promotion ({TIER_PROMOTION_L2_MIN_EPISODES})"
        );

        // And the feature must be ON by default, or the coupling is inert.
        assert_eq!(
            provenance_prune_min(),
            Some(PROVENANCE_PRUNE_CORROBORATION_MIN_DEFAULT),
            "provenance-aware pruning must default to ON; if this fails because \
             SHODH_PROVENANCE_AWARE_PRUNE is set in the environment, that is the \
             kill switch working as designed"
        );

        // A single attestation is still unprotected: the fix rescues corroborated
        // knowledge, not everything.
        assert!(!corroboration_meets(
            1,
            Some(PROVENANCE_PRUNE_CORROBORATION_MIN_DEFAULT)
        ));
        assert!(corroboration_meets(
            TIER_PROMOTION_L2_MIN_EPISODES,
            Some(PROVENANCE_PRUNE_CORROBORATION_MIN_DEFAULT)
        ));
    }

    #[test]
    fn an_edge_with_no_provenance_still_obeys_the_clock() {
        // Every RelationshipEdge already on disk predates the provenance trail,
        // and memory↔memory CoRetrieved edges carry source_episode_id: None and
        // so never accumulate episodes at all. Dropping the clock arm would make
        // both permanently unpromotable. The disjunction is not belt-and-braces.
        use crate::constants::*;
        let birth = Utc::now();
        let mut edge = create_test_edge(L1_INITIAL_WEIGHT, 0);
        edge.created_at = birth;
        assert_eq!(edge.distinct_attesting_episodes(), 0);

        assert!(
            edge.strengthen_at(birth).is_none(),
            "no episodes and no elapsed time: neither arm is satisfied"
        );
        assert_eq!(edge.tier, EdgeTier::L1Working);

        let past_window = birth + Duration::seconds(TIER_PROMOTION_WORKING_AGE_SECS + 1);
        assert_eq!(
            edge.strengthen_at(past_window),
            Some(("L1Working".to_string(), "L2Episodic".to_string())),
            "the clock arm must still work on its own"
        );
    }

    #[test]
    fn a_caller_seeded_trail_is_deduplicated_by_episode() {
        // `provenance.len()` is only a count of DISTINCT attesting episodes
        // because `merge_provenance` is the sole writer. The `SemanticFact`
        // ingest path seeds a whole trail at once from `fact.source_memories`,
        // and nothing guarantees that list has no repeats — an edge could
        // otherwise claim four attestations from one memory cited four times
        // and promote straight past L2 on evidence it does not have.
        let now = Utc::now();
        let repeated = Uuid::new_v4();
        let other = Uuid::new_v4();

        let mut trail: Vec<ProvenanceRecord> = Vec::new();
        for record in [
            attestation(repeated, now),
            attestation(repeated, now),
            attestation(repeated, now),
            attestation(other, now),
        ] {
            merge_provenance(&mut trail, record);
        }

        assert_eq!(
            trail.len(),
            2,
            "one memory cited three times is one attesting episode"
        );
        let repeated_record = trail
            .iter()
            .find(|p| p.source_episode_id == repeated)
            .expect("repeated episode present");
        assert_eq!(
            repeated_record.mention_count, 3,
            "the repeats must land on mention_count, which promotion ignores"
        );
    }

    #[test]
    fn promotion_min_episodes_encodes_evidence_since_entering_the_tier() {
        // The cumulative thresholds are what let the gate mean "independent
        // evidence acquired since this edge entered its tier" without a
        // persisted per-tier counter: L3's requirement must exceed L2's, or an
        // edge could reach L3 on exactly the evidence that earned it L2.
        use crate::constants::*;
        let l1 = EdgeTier::L1Working.promotion_min_episodes().unwrap();
        let l2 = EdgeTier::L2Episodic.promotion_min_episodes().unwrap();
        assert!(
            l2 > l1,
            "L3 must require strictly more distinct episodes than L2: {l2} vs {l1}"
        );
        assert!(l1 >= 2, "corroboration means more than one observation");
        assert_eq!(l1, TIER_PROMOTION_L2_MIN_EPISODES);
        assert_eq!(l2, TIER_PROMOTION_L3_MIN_EPISODES);
        assert_eq!(EdgeTier::L3Semantic.promotion_min_episodes(), None);

        // The episode route to L3 is only reachable if the provenance trail can
        // hold that many distinct sources. If SHODH_PROVENANCE_MAX_SOURCES is
        // set below it the route silently closes (the clock route remains), so
        // the DEFAULT cap must not close it.
        assert!(
            PROVENANCE_MAX_SOURCES_DEFAULT >= TIER_PROMOTION_L3_MIN_EPISODES,
            "the default provenance cap ({PROVENANCE_MAX_SOURCES_DEFAULT}) must be able \
             to hold {TIER_PROMOTION_L3_MIN_EPISODES} distinct episodes"
        );
    }

    #[test]
    fn importance_scaling_is_preserved_by_the_unification() {
        // The 5x arrival-rate spread between the two entry points is deliberate
        // (STRENGTHEN_IMPORTANCE_FLOOR = 0.2 ⇒ scale ∈ [0.2, 1.0]) and must
        // survive collapsing them onto one implementation.
        //
        // L2 edge at 0.3, boost coefficient = LTP_LEARNING_RATE + 0.15*0.8 = 0.22.
        //   full importance (scale 1.0): 0.3 + 0.22 * 0.7 * 1.0 = 0.454
        //   zero importance (scale 0.2): 0.3 + 0.22 * 0.7 * 0.2 = 0.3308
        let now = Utc::now();
        let mut high = create_test_edge_with_tier(0.3, 0, EdgeTier::L2Episodic);
        let mut low = create_test_edge_with_tier(0.3, 0, EdgeTier::L2Episodic);
        let mut plain = create_test_edge_with_tier(0.3, 0, EdgeTier::L2Episodic);

        high.strengthen_with_importance_at(1.0, now);
        low.strengthen_with_importance_at(0.0, now);
        plain.strengthen_at(now);

        assert!(
            (high.strength - 0.454).abs() < 1e-5,
            "got {}",
            high.strength
        );
        assert!((low.strength - 0.3308).abs() < 1e-5, "got {}", low.strength);
        assert!(
            (plain.strength - high.strength).abs() < 1e-6,
            "importance 1.0 must equal the unweighted path exactly"
        );
    }

    // =====================================================================
    // Edge-tier invariant (d), L2-vs-L3 half: the two tiers must decay at
    // measurably different rates. Before this change they dispatched to the
    // same call and differed only in prune gates.
    // =====================================================================

    #[test]
    fn l2_and_l3_edges_decay_at_measurably_different_rates() {
        // Same starting strength, same elapsed time, same LTP status — the ONLY
        // difference is the tier. Derivations (non-potentiated, so λ=0.693,
        // β=0.5, crossover 3 days):
        //   L2, 30 days: power-law leg
        //     f = exp(-0.693×3) × (30/3)^-0.5 = 0.12505521 × 0.31622777 = 0.03954593
        //     strength = 0.8 × 0.03954593 = 0.03163674
        //   L3, 30 days: scaled age 30 × 0.021505376 = 0.64516129 days ⇒ exponential leg
        //     f = exp(-0.693 × 0.64516129) = exp(-0.44709677) = 0.63948202
        //     strength = 0.8 × 0.63948202 = 0.51158562
        let mut l2 = create_test_edge_with_tier(0.8, 30, EdgeTier::L2Episodic);
        let mut l3 = create_test_edge_with_tier(0.8, 30, EdgeTier::L3Semantic);

        l2.decay();
        l3.decay();

        assert!(
            (l2.strength - 0.031_636_74).abs() < 1e-5,
            "L2 at 30 days should be ~0.0316, got {}",
            l2.strength
        );
        assert!(
            (l3.strength - 0.511_585_62).abs() < 1e-5,
            "L3 at 30 days should be ~0.5116, got {}",
            l3.strength
        );
        assert!(
            l3.strength > l2.strength * 15.0,
            "L3 must retain far more strength than L2: l3={} l2={}",
            l3.strength,
            l2.strength
        );
    }

    #[test]
    fn effective_strength_agrees_with_decay_on_the_tier_split() {
        // The read path (spreading activation) and the write path must apply the
        // same curve. If `effective_strength` kept the unscaled hybrid, an L3
        // edge would decay slowly in storage but fast at scoring time — the fix
        // would be invisible exactly where it is supposed to matter.
        let mut l3_written = create_test_edge_with_tier(0.8, 30, EdgeTier::L3Semantic);
        let l3_read = create_test_edge_with_tier(0.8, 30, EdgeTier::L3Semantic);

        let read_view = l3_read.effective_strength();
        l3_written.decay();

        assert!(
            (read_view - l3_written.strength).abs() < 1e-5,
            "read view {read_view} and decayed strength {} must agree",
            l3_written.strength
        );
    }

    #[test]
    fn l1_decay_is_untouched_by_the_tier_split() {
        // L1 has its own aggressive exponential via `tier_decay_factor` and is
        // deliberately NOT part of this change — invariant (d) is only half
        // fixed. Pinning L1 makes that explicit rather than accidental.
        //   f(48h) = exp(-0.029 × 48) = exp(-1.392) = 0.2485
        //   strength = 0.9 × 0.2485 = 0.22365
        let mut l1 = create_test_edge_with_tier(0.9, 2, EdgeTier::L1Working);
        l1.decay();
        assert!(
            (l1.strength - 0.223_65).abs() < 1e-3,
            "L1 48h decay unchanged, got {}",
            l1.strength
        );
    }

    #[test]
    fn delete_entity_scrubs_episode_index() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let now = Utc::now();

        let entity = EntityNode {
            uuid: Uuid::new_v4(),
            name: "Zephyrine".to_string(),
            labels: vec![EntityLabel::Concept],
            created_at: now,
            last_seen_at: now,
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            selectivity: None,
            fine_type: None,
            kb_id: None,
        };
        let entity_uuid = graph.add_entity(entity).unwrap();

        let episode = EpisodicNode {
            uuid: Uuid::new_v4(),
            name: "ep".to_string(),
            content: "content".to_string(),
            valid_at: now,
            created_at: now,
            entity_refs: vec![entity_uuid],
            source: EpisodeSource::Message,
            metadata: HashMap::new(),
        };
        graph.add_episode(episode).unwrap();

        assert!(
            !graph
                .get_episodes_by_entity(&entity_uuid)
                .unwrap()
                .is_empty(),
            "episode should be indexed under the entity before deletion"
        );

        assert!(graph.delete_entity(&entity_uuid).unwrap());

        assert!(
            graph
                .get_episodes_by_entity(&entity_uuid)
                .unwrap()
                .is_empty(),
            "entity_episodes index entries must be scrubbed when the entity is deleted"
        );
    }

    #[test]
    fn corroboration_meets_gates_on_threshold() {
        // Feature disabled (None) → never protected, regardless of attestation count.
        assert!(!corroboration_meets(0, None));
        assert!(!corroboration_meets(100, None));
        // Enabled → protected iff distinct attesting episodes >= min.
        assert!(!corroboration_meets(2, Some(3)));
        assert!(corroboration_meets(3, Some(3)));
        assert!(corroboration_meets(8, Some(3)));
        assert!(!corroboration_meets(0, Some(1)));
        assert!(corroboration_meets(1, Some(1)));
    }

    #[test]
    fn delete_episode_is_multi_source_aware() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let e1 = Uuid::new_v4();
        let e2 = Uuid::new_v4();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let now = Utc::now();

        let make_episode = |uuid: Uuid| EpisodicNode {
            uuid,
            name: "ep".to_string(),
            content: "content".to_string(),
            valid_at: now,
            created_at: now,
            entity_refs: vec![a, b],
            source: EpisodeSource::Message,
            metadata: HashMap::new(),
        };
        graph.add_episode(make_episode(e1)).unwrap();
        graph.add_episode(make_episode(e2)).unwrap();

        // Edge A—B attested by BOTH episodes; e1 is the primary source.
        let make_record = |episode: Uuid| ProvenanceRecord {
            source_episode_id: episode,
            mention_count: 1,
            first_observed: now,
            last_observed: now,
            confidence: None,
            evidence_span: None,
            typed_by: None,
        };
        let edge = RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: a,
            to_entity: b,
            relation_type: RelationType::CoOccurs,
            strength: 0.5,
            created_at: now,
            valid_at: now,
            invalidated_at: None,
            source_episode_id: Some(e1),
            context: String::new(),
            last_activated: now,
            activation_count: 1,
            ltp_status: LtpStatus::None,
            tier: EdgeTier::L1Working,
            activation_timestamps: None,
            entity_confidence: None,
            forman_curvature: None,
            endpoint_selectivity: None,
            provenance: vec![make_record(e1), make_record(e2)],
            promoted_at: None,
        };
        let edge_uuid = graph.add_relationship(edge).unwrap();

        // Deleting the primary source must NOT delete a corroborated edge: the
        // trail is scrubbed of e1 and primacy is promoted to the survivor e2.
        assert!(graph.delete_episode(&e1).unwrap());
        let survived = graph
            .get_relationship(&edge_uuid)
            .unwrap()
            .expect("edge corroborated by e2 must survive deletion of its primary source e1");
        assert_eq!(
            survived.provenance.len(),
            1,
            "e1 must be scrubbed from the trail"
        );
        assert_eq!(
            survived.provenance[0].source_episode_id, e2,
            "only e2's attestation should remain"
        );
        assert_eq!(
            survived.source_episode_id,
            Some(e2),
            "primary source must be promoted to the surviving attester"
        );

        // Deleting the last attesting episode leaves no attestation → edge removed.
        assert!(graph.delete_episode(&e2).unwrap());
        assert!(
            graph.get_relationship(&edge_uuid).unwrap().is_none(),
            "an edge with no remaining attesting episode must be deleted"
        );
    }

    #[test]
    fn dormant_reactivation_is_queued_and_drained() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let now = Utc::now();

        let make_edge = || RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: a,
            to_entity: b,
            relation_type: RelationType::CoOccurs,
            strength: 0.5,
            created_at: now,
            valid_at: now,
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: now,
            activation_count: 1,
            ltp_status: LtpStatus::None,
            tier: EdgeTier::L2Episodic,
            activation_timestamps: None,
            entity_confidence: None,
            forman_curvature: None,
            endpoint_selectivity: None,
            provenance: Vec::new(),
            promoted_at: None,
        };

        let edge_uuid = graph.add_relationship(make_edge()).unwrap();

        // Fresh re-attestation (gap ~0): strengthen fires, but no event.
        graph.add_relationship(make_edge()).unwrap();
        assert!(
            graph.drain_temporal_anomalies().is_empty(),
            "a fresh re-attestation must not be a dormant reactivation"
        );

        // Backdate the stored edge 8 days, then re-attest: event queued.
        let mut stored = graph.get_relationship(&edge_uuid).unwrap().unwrap();
        stored.last_activated = now - Duration::days(8);
        let encoded = crate::serialization::encode(&stored).unwrap();
        graph
            .db
            .put_cf(graph.relationships_cf(), stored.uuid.as_bytes(), encoded)
            .unwrap();

        graph.add_relationship(make_edge()).unwrap();
        let events = graph.drain_temporal_anomalies();
        assert_eq!(
            events.len(),
            1,
            "8-day dormancy must queue exactly one event"
        );
        let ev = &events[0];
        assert_eq!(ev.kind, TemporalAnomalyKind::DormantReactivation);
        assert_eq!(ev.edge_uuid, edge_uuid);
        assert!(
            ev.gap_days >= 7.9 && ev.gap_days <= 8.1,
            "gap_days should be ~8, got {}",
            ev.gap_days
        );

        // Drained means drained.
        assert!(graph.drain_temporal_anomalies().is_empty());
    }

    #[test]
    fn rocksdb_memory_breakdown_reflects_writes() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        // A fresh unflushed write lands in a memtable — the breakdown must see it.
        let now = Utc::now();
        graph
            .add_entity(EntityNode {
                uuid: Uuid::new_v4(),
                name: "Breakdown Probe".to_string(),
                labels: vec![EntityLabel::Concept],
                created_at: now,
                last_seen_at: now,
                mention_count: 1,
                summary: String::new(),
                attributes: HashMap::new(),
                name_embedding: None,
                salience: 0.5,
                is_proper_noun: false,
                selectivity: None,
                fine_type: None,
                kb_id: None,
            })
            .unwrap();
        let (memtables, _readers) = graph.rocksdb_memory_breakdown();
        assert!(
            memtables > 0,
            "memtable bytes must be visible after an unflushed write, got {memtables}"
        );
    }

    /// Create a test relationship edge with specified strength and last_activated (L1 tier)
    fn create_test_edge(strength: f32, days_since_activated: i64) -> RelationshipEdge {
        create_test_edge_with_tier(strength, days_since_activated, EdgeTier::L1Working)
    }

    /// Create a test relationship edge with specified strength, last_activated, and tier
    fn create_test_edge_with_tier(
        strength: f32,
        days_since_activated: i64,
        tier: EdgeTier,
    ) -> RelationshipEdge {
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
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier,
            entity_confidence: None, // PIPE-5: Default for tests
            forman_curvature: None,
            endpoint_selectivity: None,
            provenance: Vec::new(),
            promoted_at: None,
        }
    }

    #[test]
    fn test_hebbian_strengthen_increases_strength() {
        use crate::constants::*;
        // Use L2 tier to avoid L1 promotion resetting strength
        let mut edge = create_test_edge_with_tier(0.3, 0, EdgeTier::L2Episodic);
        let initial_strength = edge.strength;

        let _ = edge.strengthen();

        // With tier boost (L2 gets 80% of TIER_CO_ACCESS_BOOST), strength should increase
        let tier_boost = TIER_CO_ACCESS_BOOST * 0.8;
        let expected_boost = (LTP_LEARNING_RATE + tier_boost) * (1.0 - initial_strength);
        assert!(
            edge.strength > initial_strength,
            "Strengthen should increase strength (expected boost {expected_boost})"
        );
        assert_eq!(edge.activation_count, 1);
    }

    #[test]
    fn test_hebbian_strengthen_asymptotic() {
        use crate::constants::*;
        // Use L3 tier (no promotion) with high initial strength
        let mut edge = create_test_edge_with_tier(0.95, 0, EdgeTier::L3Semantic);

        let _ = edge.strengthen();

        // High strength should still increase but slowly (asymptotic to 1.0)
        // L3 tier boost = TIER_CO_ACCESS_BOOST * 0.5 = 0.075
        let tier_boost = TIER_CO_ACCESS_BOOST * 0.5;
        let expected_min = 0.95 + (LTP_LEARNING_RATE + tier_boost) * 0.05 - 0.01;
        assert!(
            edge.strength > expected_min,
            "Expected > {expected_min}, got {}",
            edge.strength
        );
        assert!(edge.strength <= 1.0);
    }

    #[test]
    fn test_hebbian_strengthen_formula() {
        use crate::constants::*;
        // Test: w_new = w_old + (η + tier_boost) × (1 - w_old)
        // Use L2 tier (tier_boost = TIER_CO_ACCESS_BOOST * 0.8) at 0.3 to avoid promotion
        let mut edge = create_test_edge_with_tier(0.3, 0, EdgeTier::L2Episodic);

        let _ = edge.strengthen();

        // L2 tier boost = 0.15 * 0.8 = 0.12
        // Expected: 0.3 + (0.1 + 0.12) * (1 - 0.3) = 0.3 + 0.22 * 0.7 = 0.454
        let tier_boost = TIER_CO_ACCESS_BOOST * 0.8;
        let expected = 0.3 + (LTP_LEARNING_RATE + tier_boost) * 0.7;
        assert!(
            (edge.strength - expected).abs() < 0.001,
            "Expected {expected}, got {}",
            edge.strength
        );
    }

    #[test]
    fn test_ltp_threshold_potentiation() {
        use crate::constants::*;

        // LTP is an L2+ mechanism: `record_activation_timestamp` returns early
        // for L1 (working edges "die too quickly to need history"), and
        // `ltp_readiness()` returns 0 at L1. So an edge can only potentiate once
        // it has legitimately reached L2.
        //
        // Before the promotion clock landed this was invisible, because a single
        // strengthen call promoted L1→L2 instantly — a fresh edge could reach
        // Full LTP (10x decay protection AND prune immunity) inside one instant.
        // That was the same "burst buys permanence" pathology the clock exists to
        // stop, so the coupling is intended: potentiation, like promotion, now
        // requires the edge to survive the 30-minute consolidation window first.
        let mut edge = create_test_edge(0.5, 0);
        assert!(!edge.is_potentiated());

        // Ten activations while still inside the L1 window: no promotion, and
        // therefore no potentiation either.
        let birth = edge.created_at;
        for _ in 0..10 {
            let _ = edge.strengthen_at(birth);
        }
        assert_eq!(edge.tier, EdgeTier::L1Working);
        assert!(
            !edge.is_potentiated(),
            "an edge that has not consolidated to L2 must not be potentiated, \
             however hard it is hit inside one instant"
        );

        // Past the window: the first call promotes to L2 and seeds the activation
        // history; the rest accumulate it. LTP_THRESHOLD is 10 activations.
        let after_window = birth + Duration::seconds(TIER_PROMOTION_WORKING_AGE_SECS + 1);
        for _ in 0..10 {
            let _ = edge.strengthen_at(after_window);
        }

        assert_eq!(edge.tier, EdgeTier::L2Episodic);
        assert!(
            edge.is_potentiated(),
            "Should be potentiated after 10 activations at L2"
        );
        assert!(
            matches!(edge.ltp_status, LtpStatus::Full),
            "Should have Full LTP status after 10 activations at L2"
        );
        assert!(
            edge.strength > 0.7,
            "Potentiated edge should have bonus strength"
        );
    }

    #[test]
    fn test_pipe4_burst_ltp_detection() {
        // Create an L2 edge with low strength to avoid early tier promotion
        let mut edge = create_test_edge_with_tier(0.22, 0, EdgeTier::L2Episodic);

        // Strengthen 5 times (LTP_BURST_THRESHOLD = 5) within 24 hours
        for _ in 0..5 {
            let _ = edge.strengthen();
        }

        // Should have burst LTP (5+ activations in 24h)
        // Edge may promote to L3 during strengthening, but should keep Burst status
        assert!(
            matches!(edge.ltp_status, LtpStatus::Burst { .. }),
            "Should have Burst LTP after 5 rapid activations, got {:?}",
            edge.ltp_status
        );
    }

    #[test]
    fn test_pipe4_activation_timestamps_recorded() {
        // L2 edges should record activation timestamps
        let mut edge = create_test_edge_with_tier(0.22, 0, EdgeTier::L2Episodic);

        // Strengthen a few times
        for _ in 0..3 {
            let _ = edge.strengthen();
        }

        // Should have recorded timestamps (edge may have promoted to L3, but still tracks)
        assert!(
            edge.activation_timestamps.is_some(),
            "L2+ edge should have activation timestamps"
        );
        assert_eq!(
            edge.activation_timestamps.as_ref().unwrap().len(),
            3,
            "Should have 3 recorded timestamps"
        );
    }

    #[test]
    fn test_pipe4_fresh_l1_no_timestamps() {
        // Fresh L1 edges should NOT have activation timestamps
        let edge = create_test_edge(0.3, 0);
        assert!(matches!(edge.tier, EdgeTier::L1Working));
        assert!(
            edge.activation_timestamps.is_none(),
            "Fresh L1 edges should not have timestamps"
        );
    }

    #[test]
    fn test_pipe4_l1_promotes_and_tracks() {
        // L1 edges that promote to L2 should start tracking timestamps.
        //
        // The loop is bounded and clocked: promotion now needs BOTH strength
        // >= L1_PROMOTION_THRESHOLD (0.5) and 30 minutes since the edge entered
        // L1. An unclocked `while` loop over `strengthen()` would spin forever,
        // because a test performs all its activations inside the same instant.
        use crate::constants::*;
        let mut edge = create_test_edge(0.3, 0);
        assert!(matches!(edge.tier, EdgeTier::L1Working));

        let born = edge.created_at;
        let after_window = born + Duration::seconds(TIER_PROMOTION_WORKING_AGE_SECS + 1);
        for _ in 0..10 {
            if !matches!(edge.tier, EdgeTier::L1Working) {
                break;
            }
            let _ = edge.strengthen_at(after_window);
        }

        // After promotion to L2, should start tracking
        assert!(
            matches!(edge.tier, EdgeTier::L2Episodic),
            "Should have promoted to L2"
        );
        // Timestamps are initialized on promotion
        assert!(
            edge.activation_timestamps.is_some(),
            "L2 edges should track timestamps after promotion"
        );
        // And the promotion clock is stamped, so the next step is gated on it
        // rather than on birth.
        assert_eq!(
            edge.promoted_at,
            Some(after_window),
            "promotion must stamp promoted_at — it is the anchor for the next step"
        );
    }

    #[test]
    fn test_pipe4_ltp_status_decay_factors() {
        // Test that each LTP status has correct decay factor
        use crate::constants::*;

        assert_eq!(LtpStatus::None.decay_factor(), 1.0);
        assert_eq!(LtpStatus::Weekly.decay_factor(), LTP_WEEKLY_DECAY_FACTOR);
        assert_eq!(LtpStatus::Full.decay_factor(), LTP_DECAY_FACTOR);

        // Burst factor depends on expiration
        let burst = LtpStatus::Burst {
            detected_at: Utc::now(),
        };
        assert_eq!(burst.decay_factor(), LTP_BURST_DECAY_FACTOR);
    }

    #[test]
    fn test_pipe4_burst_to_full_upgrade() {
        // LTP should upgrade from Burst to Full after 10 activations
        let mut edge = create_test_edge_with_tier(0.22, 0, EdgeTier::L2Episodic);

        // Get to burst LTP (5 activations)
        for _ in 0..5 {
            let _ = edge.strengthen();
        }
        assert!(
            matches!(edge.ltp_status, LtpStatus::Burst { .. }),
            "Should have Burst after 5 activations, got {:?}",
            edge.ltp_status
        );

        // Continue strengthening to Full LTP (10 total)
        for _ in 0..5 {
            let _ = edge.strengthen();
        }

        // Should now be Full (upgraded from Burst via 10 activations)
        assert!(
            matches!(edge.ltp_status, LtpStatus::Full),
            "Should have upgraded to Full LTP after 10 activations"
        );
    }

    #[test]
    fn test_pipe4_activations_in_window() {
        let mut edge = create_test_edge_with_tier(0.22, 0, EdgeTier::L2Episodic);

        // Record some activations
        for _ in 0..5 {
            let _ = edge.strengthen();
        }

        let now = Utc::now();
        let hour_ago = now - chrono::Duration::hours(1);
        let day_ago = now - chrono::Duration::days(1);

        // All activations are recent (within last second really)
        let in_hour = edge.activations_in_window(hour_ago, now);
        let in_day = edge.activations_in_window(day_ago, now);
        assert!(in_hour >= 5, "Expected 5+ in hour window, got {in_hour}");
        assert!(in_day >= 5, "Expected 5+ in day window, got {in_day}");
    }

    // =========================================================================
    // PIPE-5: Unified LTP Readiness Model Tests
    // =========================================================================

    #[test]
    fn test_pipe5_adjusted_threshold_default() {
        // Default confidence (None → 0.5) should give default threshold (10)
        let edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L2Episodic);
        assert!(edge.entity_confidence.is_none());

        let threshold = edge.adjusted_threshold();
        // confidence 0.5 → threshold = 13 - (0.5 * 6) = 10
        assert_eq!(threshold, 10, "Default confidence should give threshold 10");
    }

    #[test]
    fn test_pipe5_adjusted_threshold_high_confidence() {
        // High confidence (0.9) should give lower threshold (7-8)
        let mut edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L2Episodic);
        edge.entity_confidence = Some(0.9);

        let threshold = edge.adjusted_threshold();
        // confidence 0.9 → threshold = 13 - (0.9 * 6) = 7.6 → 8
        assert!(
            threshold <= 8,
            "High confidence should give threshold <= 8, got {threshold}"
        );
    }

    #[test]
    fn test_pipe5_adjusted_threshold_low_confidence() {
        // Low confidence (0.2) should give higher threshold (12-13)
        let mut edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L2Episodic);
        edge.entity_confidence = Some(0.2);

        let threshold = edge.adjusted_threshold();
        // confidence 0.2 → threshold = 13 - (0.2 * 6) = 11.8 → 12
        assert!(
            threshold >= 11,
            "Low confidence should give threshold >= 11, got {threshold}"
        );
    }

    #[test]
    fn test_pipe5_strength_floor_by_tier() {
        use crate::constants::*;

        let l1_edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L1Working);
        let l2_edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L2Episodic);
        let l3_edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L3Semantic);

        assert_eq!(
            l1_edge.strength_floor(),
            1.0,
            "L1 should have floor 1.0 (impossible)"
        );
        assert_eq!(
            l2_edge.strength_floor(),
            LTP_STRENGTH_FLOOR_L2,
            "L2 floor mismatch"
        );
        assert_eq!(
            l3_edge.strength_floor(),
            LTP_STRENGTH_FLOOR_L3,
            "L3 floor mismatch"
        );
    }

    #[test]
    fn test_pipe5_ltp_readiness_l1_returns_zero() {
        // L1 edges should always return 0 readiness (can't reach Full LTP)
        let mut edge = create_test_edge_with_tier(0.99, 0, EdgeTier::L1Working);
        edge.activation_count = 100;
        edge.entity_confidence = Some(1.0);

        assert_eq!(
            edge.ltp_readiness(),
            0.0,
            "L1 edges should always return 0 readiness"
        );
    }

    #[test]
    fn test_pipe5_ltp_readiness_balanced_path() {
        use crate::constants::*;

        // Balanced: 10 activations + 0.75 strength + 0.5 confidence
        // count_score = 10 / 10 = 1.0
        // strength_score = 0.75 / 0.65 = 1.15
        // tag_bonus = 0.5 * 0.3 = 0.15
        // readiness = 1.0 * 0.5 + 1.15 * 0.5 + 0.15 = 0.5 + 0.575 + 0.15 = 1.225
        let mut edge = create_test_edge_with_tier(0.75, 0, EdgeTier::L2Episodic);
        edge.activation_count = 10;
        edge.entity_confidence = Some(0.5);

        let readiness = edge.ltp_readiness();
        assert!(
            readiness >= LTP_READINESS_THRESHOLD,
            "Balanced path should reach LTP, readiness = {}",
            readiness
        );
    }

    #[test]
    fn test_pipe5_ltp_readiness_repetition_dominant() {
        use crate::constants::*;

        // Repetition dominant: 15 activations + 0.50 strength (below floor)
        // count_score = 15 / 10 = 1.5
        // strength_score = 0.50 / 0.65 = 0.77
        // tag_bonus = 0.5 * 0.3 = 0.15
        // readiness = 1.5 * 0.5 + 0.77 * 0.5 + 0.15 = 0.75 + 0.385 + 0.15 = 1.285
        let mut edge = create_test_edge_with_tier(0.50, 0, EdgeTier::L2Episodic);
        edge.activation_count = 15;
        edge.entity_confidence = Some(0.5);

        let readiness = edge.ltp_readiness();
        assert!(
            readiness >= LTP_READINESS_THRESHOLD,
            "Repetition-dominant path should reach LTP, readiness = {}",
            readiness
        );
    }

    #[test]
    fn test_pipe5_ltp_readiness_intensity_dominant() {
        use crate::constants::*;

        // Intensity dominant: 5 activations + 0.95 strength (L3)
        // count_score = 5 / 10 = 0.5
        // strength_score = 0.95 / 0.80 = 1.1875
        // tag_bonus = 0.5 * 0.3 = 0.15
        // readiness = 0.5 * 0.5 + 1.1875 * 0.5 + 0.15 = 0.25 + 0.59 + 0.15 = 0.99
        // Need more strength or count for intensity-only path on L3
        let mut edge = create_test_edge_with_tier(0.99, 0, EdgeTier::L3Semantic);
        edge.activation_count = 6;
        edge.entity_confidence = Some(0.5);

        let readiness = edge.ltp_readiness();
        // count_score = 6/10 = 0.6, strength_score = 0.99/0.80 = 1.24
        // readiness = 0.6*0.5 + 1.24*0.5 + 0.15 = 0.3 + 0.62 + 0.15 = 1.07
        assert!(
            readiness >= LTP_READINESS_THRESHOLD,
            "Intensity-dominant path should reach LTP, readiness = {}",
            readiness
        );
    }

    #[test]
    fn test_pipe5_ltp_readiness_high_confidence_boost() {
        use crate::constants::*;

        // High confidence edge reaches LTP faster
        // 7 activations + 0.65 strength + 0.9 confidence
        // threshold = 13 - 0.9*6 = 7.6 → 8
        // count_score = 7 / 8 = 0.875
        // strength_score = 0.65 / 0.65 = 1.0
        // tag_bonus = 0.9 * 0.3 = 0.27
        // readiness = 0.875 * 0.5 + 1.0 * 0.5 + 0.27 = 0.44 + 0.5 + 0.27 = 1.21
        let mut edge = create_test_edge_with_tier(0.65, 0, EdgeTier::L2Episodic);
        edge.activation_count = 7;
        edge.entity_confidence = Some(0.9);

        let readiness = edge.ltp_readiness();
        assert!(
            readiness >= LTP_READINESS_THRESHOLD,
            "High-confidence should boost to LTP, readiness = {}",
            readiness
        );
    }

    #[test]
    fn test_pipe5_weak_edge_no_ltp() {
        use crate::constants::*;

        // Weak edge: 4 activations + 0.40 strength + 0.3 confidence
        // threshold = 13 - 0.3*6 = 11.2 → 11
        // count_score = 4 / 11 = 0.36
        // strength_score = 0.40 / 0.65 = 0.62
        // tag_bonus = 0.3 * 0.3 = 0.09
        // readiness = 0.36 * 0.5 + 0.62 * 0.5 + 0.09 = 0.18 + 0.31 + 0.09 = 0.58
        let mut edge = create_test_edge_with_tier(0.40, 0, EdgeTier::L2Episodic);
        edge.activation_count = 4;
        edge.entity_confidence = Some(0.3);

        let readiness = edge.ltp_readiness();
        assert!(
            readiness < LTP_READINESS_THRESHOLD,
            "Weak edge should NOT reach LTP, readiness = {}",
            readiness
        );
    }

    #[test]
    fn test_pipe5_unified_detect_ltp_status() {
        // Test that detect_ltp_status uses the unified readiness formula
        let mut edge = create_test_edge_with_tier(0.75, 0, EdgeTier::L2Episodic);
        edge.activation_count = 10;
        edge.entity_confidence = Some(0.5);
        edge.activation_timestamps = Some(std::collections::VecDeque::new());

        let status = edge.detect_ltp_status(Utc::now());
        assert_eq!(
            status,
            LtpStatus::Full,
            "Balanced path should grant Full LTP via readiness"
        );
    }

    #[test]
    fn test_pipe5_l3_no_auto_ltp_without_activations() {
        // L3 with high strength but low activation count should NOT auto-LTP
        // This tests that the old auto-LTP behavior is removed
        let mut edge = create_test_edge_with_tier(0.85, 0, EdgeTier::L3Semantic);
        edge.activation_count = 2; // Low count
        edge.entity_confidence = Some(0.5);
        edge.activation_timestamps = Some(std::collections::VecDeque::new());

        // count_score = 2/10 = 0.2, strength_score = 0.85/0.80 = 1.06
        // readiness = 0.2*0.5 + 1.06*0.5 + 0.15 = 0.1 + 0.53 + 0.15 = 0.78
        let status = edge.detect_ltp_status(Utc::now());
        assert_eq!(
            status,
            LtpStatus::None,
            "L3 high strength alone should NOT grant Full LTP, needs activations too"
        );
    }

    #[test]
    fn test_decay_reduces_strength() {
        // Use L2 tier for multi-day decay testing (L1 max age is only 4 hours)
        let mut edge = create_test_edge_with_tier(0.5, 7, EdgeTier::L2Episodic);

        let initial_strength = edge.strength;
        edge.decay();

        assert!(
            edge.strength < initial_strength,
            "Decay should reduce strength (initial: {}, after: {})",
            initial_strength,
            edge.strength
        );
    }

    #[test]
    fn test_decay_tier_aware() {
        // Test tier-aware decay: L2 episodic with Wixted 2004 hybrid decay over 7 days
        // 7 days > crossover (3 days), so power-law phase applies:
        //   value_at_crossover = exp(-0.693 * 3) ≈ 0.125
        //   power_law = (7/3)^(-0.5) ≈ 0.655
        //   decay ≈ 0.125 * 0.655 ≈ 0.082
        let mut edge = create_test_edge_with_tier(1.0, 7, EdgeTier::L2Episodic);

        edge.decay();

        // Hybrid decay is aggressive for non-potentiated edges: ~8% retained at 7 days
        assert!(
            edge.strength < 0.15,
            "After 7 days with hybrid decay, strength should be below 0.15, got {}",
            edge.strength
        );
        assert!(
            edge.strength > 0.05,
            "After 7 days with hybrid decay, strength should be above 0.05, got {}",
            edge.strength
        );
        assert!(
            edge.strength > LTP_MIN_STRENGTH,
            "Strength should still be above floor, got {}",
            edge.strength
        );
    }

    #[test]
    fn test_decay_minimum_floor() {
        // Use L3 tier for very old edge testing (L3 has 10 year max age)
        let mut edge = create_test_edge_with_tier(0.02, 100, EdgeTier::L3Semantic);

        edge.decay();

        assert!(
            edge.strength >= LTP_MIN_STRENGTH,
            "Strength should not go below minimum floor"
        );
    }

    #[test]
    fn test_potentiated_decay_slower() {
        // Use L2 tier for multi-day decay comparison
        let mut edge1 = create_test_edge_with_tier(0.8, 7, EdgeTier::L2Episodic);
        let mut edge2 = create_test_edge_with_tier(0.8, 7, EdgeTier::L2Episodic);
        edge2.ltp_status = LtpStatus::Full; // Full LTP = 10x slower decay

        edge1.decay();
        edge2.decay();

        assert!(
            edge2.strength > edge1.strength,
            "Potentiated edge should decay slower (normal: {}, potentiated: {})",
            edge1.strength,
            edge2.strength
        );
    }

    #[test]
    fn test_effective_strength_read_only() {
        // Use L2 tier for multi-day testing
        let edge = create_test_edge_with_tier(0.5, 7, EdgeTier::L2Episodic);
        let initial_strength = edge.strength;

        let effective = edge.effective_strength();

        // effective_strength should not modify the edge
        assert_eq!(edge.strength, initial_strength);
        assert!(effective < initial_strength);
    }

    #[test]
    fn test_decay_prune_threshold() {
        // Use L2 tier for decay testing beyond its max age (14 days)
        let mut weak_edge = create_test_edge_with_tier(LTP_MIN_STRENGTH, 30, EdgeTier::L2Episodic);
        // No LTP status = normal decay
        assert!(matches!(weak_edge.ltp_status, LtpStatus::None));

        let should_prune = weak_edge.decay();

        // Non-potentiated edge at minimum strength past max age should be prunable
        assert!(
            should_prune,
            "Weak non-potentiated edge past max age should be marked for pruning"
        );
    }

    #[test]
    fn test_potentiated_above_floor_never_pruned() {
        // Potentiated edge above LTP_PRUNE_FLOOR should be protected
        // With hybrid decay at 30 days (potentiated): decay ≈ 0.177
        // Need initial strength high enough that final > LTP_PRUNE_FLOOR (0.05)
        // 0.5 * 0.177 ≈ 0.089 > 0.05 ✓
        let mut edge = create_test_edge_with_tier(0.5, 30, EdgeTier::L2Episodic);
        edge.ltp_status = LtpStatus::Full;

        let should_prune = edge.decay();

        assert!(
            !should_prune,
            "Potentiated edges above LTP_PRUNE_FLOOR should not be pruned"
        );
    }

    #[test]
    fn test_potentiated_at_floor_stripped_and_prunable() {
        // Potentiated edge at/below LTP_PRUNE_FLOOR should have LTP stripped
        let mut edge = create_test_edge_with_tier(LTP_MIN_STRENGTH, 30, EdgeTier::L2Episodic);
        edge.ltp_status = LtpStatus::Full;

        let should_prune = edge.decay();

        // LTP gets stripped because strength <= LTP_PRUNE_FLOOR,
        // then normal prune logic applies (strength at floor, past max age)
        assert!(
            should_prune,
            "Zombie potentiated edges at floor strength should be prunable"
        );
        assert!(
            matches!(edge.ltp_status, LtpStatus::None),
            "LTP status should be stripped when strength at floor"
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
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        // Random memory ID with no associated episode should return 0.5 (neutral)
        let fake_memory_id = crate::memory::MemoryId(Uuid::new_v4());
        let strength = graph.get_memory_hebbian_strength(&fake_memory_id);
        assert_eq!(strength, Some(0.5), "No episode should return neutral 0.5");
    }

    #[test]
    fn test_hebbian_strength_with_episode_no_edges() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

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
            selectivity: None,
            fine_type: None,
            kb_id: None,
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
            selectivity: None,
            fine_type: None,
            kb_id: None,
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
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

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
            selectivity: None,
            fine_type: None,
            kb_id: None,
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
            selectivity: None,
            fine_type: None,
            kb_id: None,
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
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic, // Use L2 since it has activation count
            entity_confidence: None,    // PIPE-5: Default for tests
            forman_curvature: None,
            endpoint_selectivity: None,
            provenance: Vec::new(),
            promoted_at: None,
        };
        graph.add_relationship(edge).unwrap();

        // Should return the edge strength (0.8, with minimal decay since just activated)
        let strength = graph.get_memory_hebbian_strength(&memory_id);
        assert!(strength.is_some());
        let s = strength.unwrap();
        assert!(s > 0.75 && s <= 0.8, "Strength should be ~0.8, got {}", s);
    }

    #[test]
    fn apply_decay_at_ages_a_real_graph_at_cadence() {
        // End-to-end check of the virtual-clock decay plumbing: drive
        // apply_decay_at over simulated time at the production ~6h cadence and
        // confirm a real on-disk edge actually decays. This is the mechanism the
        // decay-evaluation harness uses to age a graph for recall@k-vs-age
        // measurement without waiting wall-clock days.
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let edge = RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: Uuid::new_v4(),
            to_entity: Uuid::new_v4(),
            relation_type: RelationType::RelatedTo,
            strength: 0.8,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: Utc::now(),
            activation_count: 1,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic,
            entity_confidence: None,
            forman_curvature: None,
            endpoint_selectivity: None,
            provenance: Vec::new(),
            promoted_at: None,
        };
        let edge_id = graph.add_relationship(edge).unwrap();
        let start = graph.get_relationship(&edge_id).unwrap().unwrap().strength;

        // Age ~10 days at the 6h cadence (40 cycles), advancing a virtual clock.
        let t0 = Utc::now();
        for step in 1..=40 {
            let now = t0 + Duration::hours(6 * step);
            graph.apply_decay_at(now).unwrap();
        }

        // Under the real cadence an L2 edge floors below its 0.2 prune threshold
        // within ~2 days (chained ~daily-half-life exponential), and decay()'s
        // return (`exceeded_max_age || strength <= prune_threshold`) then prunes
        // it — the min-prune-age gate is OR'd, not a hard floor. So after 10
        // simulated days the edge is either pruned or sitting at the decay floor.
        // Either outcome proves the virtual clock drove real decay end-to-end.
        assert!(start > 0.5, "sanity: edge started strong, got {start}");
        match graph.get_relationship(&edge_id).unwrap() {
            None => { /* pruned after flooring — the expected outcome */ }
            Some(aged) => assert!(
                aged.strength < 0.3,
                "edge should have decayed near the floor under cadenced aging: \
                 start={start}, aged={}",
                aged.strength
            ),
        }
    }

    // =========================================================================
    // BEHAVIORAL LOOP TESTS
    // Verify cognitive loops are closed (not just structurally present).
    // =========================================================================

    #[test]
    fn test_loop_episode_idempotency() {
        // Verify: calling add_episode() twice with same UUID doesn't inflate episode_count
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let memory_uuid = Uuid::new_v4();
        let entity_uuid = Uuid::new_v4();

        let entity = EntityNode {
            uuid: entity_uuid,
            name: "IdempotentEntity".to_string(),
            labels: vec![EntityLabel::Concept],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            selectivity: None,
            fine_type: None,
            kb_id: None,
        };
        graph.add_entity(entity).unwrap();

        let episode = EpisodicNode {
            uuid: memory_uuid,
            name: "Test Episode".to_string(),
            content: "Test content for idempotency".to_string(),
            valid_at: Utc::now(),
            created_at: Utc::now(),
            entity_refs: vec![entity_uuid],
            source: EpisodeSource::Message,
            metadata: std::collections::HashMap::new(),
        };

        // First insert
        graph.add_episode(episode.clone()).unwrap();
        let count_after_first = graph.episode_count.load(Ordering::Relaxed);

        // Second insert (same UUID — simulates retry)
        graph.add_episode(episode).unwrap();
        let count_after_second = graph.episode_count.load(Ordering::Relaxed);

        assert_eq!(
            count_after_first, count_after_second,
            "episode_count should not inflate on overwrite (got {} then {})",
            count_after_first, count_after_second
        );
    }

    #[test]
    fn test_loop_decay_reduces_edge_strength() {
        // Verify: apply_decay() actually reduces edge strength (decay loop is closed)
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let from_uuid = Uuid::new_v4();
        let to_uuid = Uuid::new_v4();

        // Create entities
        for (uuid, name) in [(from_uuid, "DecayFrom"), (to_uuid, "DecayTo")] {
            let entity = EntityNode {
                uuid,
                name: name.to_string(),
                labels: vec![EntityLabel::Concept],
                created_at: Utc::now(),
                last_seen_at: Utc::now(),
                mention_count: 1,
                summary: String::new(),
                attributes: std::collections::HashMap::new(),
                name_embedding: None,
                salience: 0.5,
                is_proper_noun: false,
                selectivity: None,
                fine_type: None,
                kb_id: None,
            };
            graph.add_entity(entity).unwrap();
        }

        // Create edge with strength 0.5, last activated 30 days ago (well past decay threshold)
        let edge_uuid = Uuid::new_v4();
        let edge = RelationshipEdge {
            uuid: edge_uuid,
            from_entity: from_uuid,
            to_entity: to_uuid,
            relation_type: RelationType::RelatedTo,
            strength: 0.5,
            created_at: Utc::now() - Duration::days(60),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: None,
            context: "decay test".to_string(),
            last_activated: Utc::now() - Duration::days(30),
            activation_count: 1,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L1Working,
            entity_confidence: None,
            forman_curvature: None,
            endpoint_selectivity: None,
            provenance: Vec::new(),
            promoted_at: None,
        };
        graph.add_relationship(edge).unwrap();

        // Run decay
        let result = graph.apply_decay().unwrap();

        // Edge should have been affected (either decayed or pruned)
        if result.pruned_count == 0 {
            // Not pruned — check it was weakened
            let updated = graph.get_relationship(&edge_uuid).unwrap();
            if let Some(updated_edge) = updated {
                assert!(
                    updated_edge.strength < 0.5,
                    "Edge strength should have decayed from 0.5, got {}",
                    updated_edge.strength
                );
            }
        }
        // If pruned, the loop is working (edge was weak enough to remove)
    }

    #[test]
    fn test_loop_feedback_strengthens_edges() {
        // Verify: batch_strengthen_synapses() actually increases edge strength (feedback loop)
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let from_uuid = Uuid::new_v4();
        let to_uuid = Uuid::new_v4();

        for (uuid, name) in [(from_uuid, "FeedbackFrom"), (to_uuid, "FeedbackTo")] {
            let entity = EntityNode {
                uuid,
                name: name.to_string(),
                labels: vec![EntityLabel::Concept],
                created_at: Utc::now(),
                last_seen_at: Utc::now(),
                mention_count: 1,
                summary: String::new(),
                attributes: std::collections::HashMap::new(),
                name_embedding: None,
                salience: 0.5,
                is_proper_noun: false,
                selectivity: None,
                fine_type: None,
                kb_id: None,
            };
            graph.add_entity(entity).unwrap();
        }

        let initial_strength = 0.3;
        let edge = RelationshipEdge {
            uuid: Uuid::new_v4(), // add_relationship() will assign a new UUID
            from_entity: from_uuid,
            to_entity: to_uuid,
            relation_type: RelationType::RelatedTo,
            strength: initial_strength,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: None,
            context: "feedback test".to_string(),
            last_activated: Utc::now(),
            activation_count: 1,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic,
            entity_confidence: None,
            forman_curvature: None,
            endpoint_selectivity: None,
            provenance: Vec::new(),
            promoted_at: None,
        };
        let edge_uuid = graph.add_relationship(edge).unwrap();

        // Simulate "Helpful" feedback → strengthen
        let count = graph.batch_strengthen_synapses(&[edge_uuid]).unwrap();
        assert_eq!(count, 1, "Should have strengthened 1 edge");

        let updated = graph.get_relationship(&edge_uuid).unwrap().unwrap();
        assert!(
            updated.strength > initial_strength,
            "Edge strength should increase after Helpful feedback: {} -> {}",
            initial_strength,
            updated.strength
        );
    }

    #[test]
    fn test_strengthen_with_importance_scales_boost() {
        // Verify: high-importance strengthening produces stronger edges than low-importance,
        // and low-importance still strengthens (floor = STRENGTHEN_IMPORTANCE_FLOOR = 0.2).
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        // Use separate entity pairs so add_relationship doesn't dedup
        // (same from+to+relation_type gets merged into existing edge)
        let high_from = Uuid::new_v4();
        let high_to = Uuid::new_v4();
        let low_from = Uuid::new_v4();
        let low_to = Uuid::new_v4();

        for (uuid, name) in [
            (high_from, "HighFrom"),
            (high_to, "HighTo"),
            (low_from, "LowFrom"),
            (low_to, "LowTo"),
        ] {
            let entity = EntityNode {
                uuid,
                name: name.to_string(),
                labels: vec![EntityLabel::Concept],
                created_at: Utc::now(),
                last_seen_at: Utc::now(),
                mention_count: 1,
                summary: String::new(),
                attributes: std::collections::HashMap::new(),
                name_embedding: None,
                salience: 0.5,
                is_proper_noun: false,
                selectivity: None,
                fine_type: None,
                kb_id: None,
            };
            graph.add_entity(entity).unwrap();
        }

        let initial_strength = 0.3;

        let make_edge = |from: Uuid, to: Uuid| RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: from,
            to_entity: to,
            relation_type: RelationType::RelatedTo,
            strength: initial_strength,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: None,
            context: "importance test".to_string(),
            last_activated: Utc::now(),
            activation_count: 1,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic,
            entity_confidence: None,
            forman_curvature: None,
            endpoint_selectivity: None,
            provenance: Vec::new(),
            promoted_at: None,
        };

        let high_edge_uuid = graph
            .add_relationship(make_edge(high_from, high_to))
            .unwrap();
        let low_edge_uuid = graph.add_relationship(make_edge(low_from, low_to)).unwrap();

        // Strengthen with high importance (1.0) — should get full boost
        graph
            .batch_strengthen_synapses_with_importance(&[high_edge_uuid], 1.0)
            .unwrap();

        // Strengthen with low importance (0.0) — should get floor boost (0.2x)
        graph
            .batch_strengthen_synapses_with_importance(&[low_edge_uuid], 0.0)
            .unwrap();

        let high_edge = graph.get_relationship(&high_edge_uuid).unwrap().unwrap();
        let low_edge = graph.get_relationship(&low_edge_uuid).unwrap().unwrap();

        // Both should be stronger than initial
        assert!(
            high_edge.strength > initial_strength,
            "High-importance edge should strengthen: {} > {}",
            high_edge.strength,
            initial_strength
        );
        assert!(
            low_edge.strength > initial_strength,
            "Low-importance edge should still strengthen (floor=0.2): {} > {}",
            low_edge.strength,
            initial_strength
        );

        // High-importance should be stronger than low-importance
        assert!(
            high_edge.strength > low_edge.strength,
            "High-importance edge should be stronger: {} > {}",
            high_edge.strength,
            low_edge.strength
        );

        // Verify the ratio roughly matches expected scaling:
        // high boost ≈ base * 1.0, low boost ≈ base * 0.2
        let high_delta = high_edge.strength - initial_strength;
        let low_delta = low_edge.strength - initial_strength;
        let ratio = high_delta / low_delta;
        // Expected ratio ≈ 1.0 / 0.2 = 5.0 (not exact due to (1-strength) term)
        assert!(
            ratio > 3.0 && ratio < 7.0,
            "Boost ratio should be ~5x (floor=0.2), got {:.2}x (high_delta={:.4}, low_delta={:.4})",
            ratio,
            high_delta,
            low_delta
        );
    }

    // =========================================================================
    // Forman-Ricci Curvature Tests
    // Reference: Leal, Restrepo, Stadler, Jost (2018) arXiv:1811.07825
    // =========================================================================

    /// Helper: create an entity node with a given name
    fn make_entity(graph: &GraphMemory, name: &str) -> Uuid {
        let uuid = Uuid::new_v4();
        let entity = EntityNode {
            uuid,
            name: name.to_string(),
            labels: vec![EntityLabel::Concept],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            selectivity: None,
            fine_type: None,
            kb_id: None,
        };
        // add_entity may dedup and return a different UUID
        graph.add_entity(entity).unwrap()
    }

    #[test]
    fn alias_resolves_surface_to_canonical() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let dali = make_entity(&graph, "Dali");
        graph.put_alias("cargo ship", dali).unwrap();

        // Direct resolution, case-insensitive; unknown surfaces miss.
        assert_eq!(graph.resolve_alias("cargo ship"), Some(dali));
        assert_eq!(graph.resolve_alias("Cargo Ship"), Some(dali));
        assert_eq!(graph.resolve_alias("unknown surface"), None);

        // find_entity_by_name redirects the alias to the canonical node (Tier 0).
        let found = graph.find_entity_by_name("cargo ship").unwrap().unwrap();
        assert_eq!(found.uuid, dali);
        assert_eq!(found.name, "Dali");
    }

    #[test]
    fn aliases_persist_across_reopen() {
        // "Canonical set stable across re-runs": the alias table is CF-backed and
        // reloads into the in-memory index on open.
        let temp_dir = tempfile::tempdir().unwrap();
        let dali = {
            let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
            let dali = make_entity(&graph, "Dali");
            graph.put_alias("the vessel", dali).unwrap();
            assert_eq!(graph.resolve_alias("the vessel"), Some(dali));
            dali
        };
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        assert_eq!(graph.resolve_alias("the vessel"), Some(dali));
        assert_eq!(
            graph
                .find_entity_by_name("the vessel")
                .unwrap()
                .unwrap()
                .name,
            "Dali"
        );
    }

    #[test]
    fn seed_aliases_writes_batch() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let bridge = make_entity(&graph, "Francis Scott Key Bridge");

        let written = graph
            .seed_aliases([
                ("Key Bridge".to_string(), bridge),
                ("the bridge".to_string(), bridge),
                ("   ".to_string(), bridge), // blank surface is skipped
            ])
            .unwrap();

        assert_eq!(written, 2, "blank surface must be skipped");
        assert_eq!(graph.resolve_alias("key bridge"), Some(bridge));
        assert_eq!(graph.resolve_alias("the bridge"), Some(bridge));
        assert_eq!(graph.alias_count(), 2);
    }

    #[test]
    fn dangling_alias_falls_through_to_name_lookup() {
        // An alias pointing at a non-existent canonical UUID must not shadow the
        // real entity of that surface: Tier 0 returns None, the tiers continue.
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let real = make_entity(&graph, "Baltimore");
        graph.put_alias("baltimore", Uuid::new_v4()).unwrap();

        let found = graph.find_entity_by_name("Baltimore").unwrap().unwrap();
        assert_eq!(
            found.uuid, real,
            "dangling alias must not shadow the real node"
        );
    }

    #[test]
    fn clear_all_wipes_aliases() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let dali = make_entity(&graph, "Dali");
        graph.put_alias("cargo ship", dali).unwrap();
        assert_eq!(graph.alias_count(), 1);

        graph.clear_all().unwrap();
        assert_eq!(graph.alias_count(), 0, "GDPR erasure must wipe aliases");
        assert_eq!(graph.resolve_alias("cargo ship"), None);
    }

    #[test]
    fn causal_spine_noops_without_parser() {
        // No SHODH_SPACY_MODEL_PATH in unit tests → the causal-spine pass is a
        // graceful no-op (additive, never load-bearing for ingest).
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let ship = make_entity(&graph, "ship");
        let bridge = make_entity(&graph, "bridge");
        let entities = vec![
            ("ship".to_string(), ship, EntityLabel::Concept),
            ("bridge".to_string(), bridge, EntityLabel::Concept),
        ];
        if crate::dep_parser::is_available() {
            return;
        }
        let minted = graph.mint_causal_spine_edges(
            "the ship rammed the bridge, causing the collapse",
            &entities,
            Uuid::new_v4(),
            Utc::now(),
        );
        assert_eq!(minted, 0, "no parser → no causal-spine edges");
    }

    #[test]
    fn precedes_is_first_class() {
        assert_eq!(RelationType::Precedes.as_str(), "Precedes");
        assert!(
            !RelationType::Precedes.is_causal(),
            "temporal order is not causation"
        );

        // Legacy edges (pre-promotion CATENA builds, or any store that predates
        // this change) normalize on read; anything else passes through untouched.
        // Exact case only — see `normalize()`'s doc comment for why.
        assert_eq!(
            RelationType::Custom("Precedes".into()).normalize(),
            RelationType::Precedes
        );
        assert_eq!(
            RelationType::Custom("precedes".into()).normalize(),
            RelationType::Custom("precedes".into()),
            "normalize is exact-case only; lowercase is a distinct Custom value"
        );
        assert_eq!(
            RelationType::Custom("Other".into()).normalize(),
            RelationType::Custom("Other".into())
        );

        // A legacy Custom("Precedes") and the first-class variant must key
        // identically in `typed_pair_key` (:3783, keyed on `as_str()`), or
        // add_relationship's type-dedup silently doubles the edge instead of
        // collapsing it.
        assert_eq!(
            RelationType::Custom("Precedes".into()).as_str(),
            RelationType::Precedes.as_str()
        );

        // Serde/postcard round trip: Precedes is a plain unit variant.
        let bytes = crate::serialization::encode(&RelationType::Precedes).unwrap();
        let decoded: RelationType = crate::serialization::decode(&bytes).unwrap();
        assert_eq!(decoded, RelationType::Precedes);

        // A legacy store wrote Custom("Precedes") on disk; decode it back as-is
        // (no normalize applied by serde itself), then normalize explicitly, the
        // way `decode_relationship_edge` does for every runtime read.
        let legacy = RelationType::Custom("Precedes".to_string());
        let legacy_bytes = crate::serialization::encode(&legacy).unwrap();
        let legacy_decoded: RelationType = crate::serialization::decode(&legacy_bytes).unwrap();
        assert_eq!(
            legacy_decoded, legacy,
            "raw decode must not silently mutate"
        );
        assert_eq!(legacy_decoded.normalize(), RelationType::Precedes);

        // Pin backward compatibility with bytes actually written by a
        // PRE-CHANGE build (base commit f113ad58), not merely round-tripped
        // through today's encoder — `legacy_bytes` above would pass even if a
        // future reorder shifted both the encoder and decoder together, since
        // it encodes and decodes with the same (new) code. These bytes were
        // derived independently (a standalone scratch binary against postcard
        // 1.1.3, cross-checked against an in-tree scratch test using the exact
        // pre-change enum shape) and are NOT regenerated by this test. They are
        // RAW postcard (no `crate::serialization` 2-byte format tag), matching
        // how `relation_type` is actually stored — embedded inside a larger
        // `RelationshipEdge` postcard payload via `try_decode_compat`, not
        // tagged on its own — so decode with `decode_raw`, not `decode`:
        //   - byte 0 = 0x23 (35 decimal): postcard's single-byte varint tag for
        //     `Custom`'s DECLARATION INDEX in the pre-change enum. Counting the
        //     pre-change variants in order (WorksWith=0, WorksAt=1, ...,
        //     Approves=34) puts `Custom(String)` at index 35 — recount this if
        //     the pre-change variant list above is ever edited.
        //   - byte 1 = 0x08: postcard's varint length prefix for the following
        //     UTF-8 string (8 bytes).
        //   - bytes 2..=9 = the UTF-8 bytes of "Precedes".
        let pre_change_custom_precedes_bytes: [u8; 10] =
            [0x23, 0x08, 0x50, 0x72, 0x65, 0x63, 0x65, 0x64, 0x65, 0x73];
        let from_pre_change: RelationType =
            crate::serialization::decode_raw(&pre_change_custom_precedes_bytes).unwrap();
        assert_eq!(
            from_pre_change,
            RelationType::Custom("Precedes".to_string()),
            "Custom's own discriminant must never move — a pre-change edge must \
             still decode as Custom(\"Precedes\") before normalization"
        );
        assert_eq!(
            from_pre_change.normalize(),
            RelationType::Precedes,
            "a byte-for-byte pre-change edge must normalize to the first-class variant"
        );
    }

    #[test]
    fn catena_temporal_signal_mints_first_class_precedes() {
        use crate::causal_vocab::LinkRelation;

        // The CATENA mint-site mapping (factored into `relation_type_from_link`
        // so it's testable without a live dependency parser) must produce the
        // first-class variant for a temporal signal, not Custom("Precedes").
        assert_eq!(
            GraphMemory::relation_type_from_link(LinkRelation::Precedes),
            RelationType::Precedes
        );
        assert_eq!(
            GraphMemory::relation_type_from_link(LinkRelation::Causes),
            RelationType::Causes
        );

        // End-to-end: build + persist + read back an edge exactly as
        // `mint_causal_spine_edges` would for a CATENA temporal link, and
        // confirm it survives the storage round trip (through
        // `decode_relationship_edge`) as `Precedes`.
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let collapse = make_entity(&graph, "collapse");
        let closure = make_entity(&graph, "closure");
        let rt = GraphMemory::relation_type_from_link(LinkRelation::Precedes);
        let edge = graph.build_spine_edge(
            collapse,
            closure,
            rt,
            Uuid::new_v4(),
            "the collapse happened, then the closure followed",
            Utc::now(),
            TypingMethod::Catena,
        );
        let edge_id = graph.add_relationship(edge).unwrap();
        let stored = graph.get_relationship(&edge_id).unwrap().unwrap();
        assert_eq!(
            stored.relation_type,
            RelationType::Precedes,
            "CATENA temporal edges must persist as first-class Precedes, not Custom"
        );
    }

    /// Helper: create a directed edge from → to
    fn make_edge(graph: &GraphMemory, from: Uuid, to: Uuid) -> Uuid {
        let edge = RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: from,
            to_entity: to,
            relation_type: RelationType::RelatedTo,
            strength: 0.5,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: Utc::now(),
            activation_count: 1,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic,
            entity_confidence: None,
            forman_curvature: None,
            endpoint_selectivity: None,
            provenance: Vec::new(),
            promoted_at: None,
        };
        graph.add_relationship(edge).unwrap()
    }

    #[test]
    fn test_forman_curvature_isolated_edge() {
        // Single edge A → B: deg(A) = 1, deg(B) = 1
        // F(e) = 4 - 1 - 1 = 2 (positive = tight community)
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let a = make_entity(&graph, "NodeA");
        let b = make_entity(&graph, "NodeB");
        let edge_id = make_edge(&graph, a, b);

        let stats = graph.compute_forman_ricci_curvature().unwrap();
        assert_eq!(stats.edges_computed, 1);
        assert_eq!(stats.positive_count, 1);
        assert_eq!(stats.negative_count, 0);

        let edge = graph.get_relationship(&edge_id).unwrap().unwrap();
        let curv = edge.forman_curvature.expect("curvature should be computed");
        // F(e) = 4 - deg(A) - deg(B) = 4 - 1 - 1 = 2
        assert!(
            (curv - 2.0).abs() < f32::EPSILON,
            "Isolated edge curvature should be 2.0, got {}",
            curv
        );
    }

    #[test]
    fn test_forman_curvature_hub_topology() {
        // Star topology: Hub → {A, B, C, D}
        // For edge Hub→A: deg(Hub) = 4 (out), deg(A) = 1 (in)
        //   in_deg(Hub) = 0, out_deg(Hub) = 4
        //   in_deg(A) = 1, out_deg(A) = 0
        //   F(→e→) = 2 - in(Hub) - out(A) = 2 - 0 - 0 = 2
        //   F(←e←) = 2 - out(Hub) - in(A) = 2 - 4 - 1 = -3
        //   F(e) = 2 + (-3) = -1
        // Equivalently: F(e) = 4 - deg(Hub) - deg(A) = 4 - 4 - 1 = -1
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let hub = make_entity(&graph, "Hub");
        let a = make_entity(&graph, "SpokeA");
        let b = make_entity(&graph, "SpokeB");
        let c = make_entity(&graph, "SpokeC");
        let d = make_entity(&graph, "SpokeD");

        let edge_a = make_edge(&graph, hub, a);
        make_edge(&graph, hub, b);
        make_edge(&graph, hub, c);
        make_edge(&graph, hub, d);

        let stats = graph.compute_forman_ricci_curvature().unwrap();
        assert_eq!(stats.edges_computed, 4);
        // All edges should have negative curvature (hub bridges)
        assert_eq!(
            stats.negative_count, 4,
            "All hub edges should be negative, got {} negative",
            stats.negative_count
        );

        let edge = graph.get_relationship(&edge_a).unwrap().unwrap();
        let curv = edge.forman_curvature.unwrap();
        // F(e) = 4 - 4 - 1 = -1
        assert!(
            (curv - (-1.0)).abs() < f32::EPSILON,
            "Hub→Spoke curvature should be -1.0, got {}",
            curv
        );
    }

    #[test]
    fn test_forman_curvature_triangle() {
        // Triangle: A→B, B→C, C→A
        // Each node: in_deg=1, out_deg=1, total deg=2
        // F(e) = 4 - 2 - 2 = 0 (neutral / transitional)
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let a = make_entity(&graph, "TriA");
        let b = make_entity(&graph, "TriB");
        let c = make_entity(&graph, "TriC");

        make_edge(&graph, a, b);
        make_edge(&graph, b, c);
        make_edge(&graph, c, a);

        let stats = graph.compute_forman_ricci_curvature().unwrap();
        assert_eq!(stats.edges_computed, 3);
        assert_eq!(
            stats.zero_count, 3,
            "Triangle edges should all be zero curvature"
        );
        assert!(
            stats.mean_curvature.abs() < f32::EPSILON,
            "Mean curvature of triangle should be 0.0, got {}",
            stats.mean_curvature
        );
    }

    #[test]
    fn test_forman_curvature_empty_graph() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let stats = graph.compute_forman_ricci_curvature().unwrap();
        assert_eq!(stats.edges_computed, 0);
        assert_eq!(stats.mean_curvature, 0.0);
    }

    #[test]
    fn test_forman_curvature_bridge_detection() {
        // Two triangles connected by a bridge:
        // Triangle 1: A→B, B→C, C→A
        // Bridge: C→D
        // Triangle 2: D→E, E→F, F→D
        //
        // Bridge edge C→D: deg(C) = 3 (in from B, out to A, out to D)
        //                  deg(D) = 3 (in from C, in from F, out to E)
        // F(C→D) = 4 - 3 - 3 = -2 (most negative = bottleneck)
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let a = make_entity(&graph, "BridgeA");
        let b = make_entity(&graph, "BridgeB");
        let c = make_entity(&graph, "BridgeC");
        let d = make_entity(&graph, "BridgeD");
        let e = make_entity(&graph, "BridgeE");
        let f = make_entity(&graph, "BridgeF");

        // Triangle 1
        make_edge(&graph, a, b);
        make_edge(&graph, b, c);
        make_edge(&graph, c, a);
        // Bridge
        let bridge_id = make_edge(&graph, c, d);
        // Triangle 2
        make_edge(&graph, d, e);
        make_edge(&graph, e, f);
        make_edge(&graph, f, d);

        let stats = graph.compute_forman_ricci_curvature().unwrap();
        assert_eq!(stats.edges_computed, 7);

        // Bridge should be the most negative edge
        let bridge = graph.get_relationship(&bridge_id).unwrap().unwrap();
        let bridge_curv = bridge.forman_curvature.unwrap();
        assert!(
            bridge_curv <= stats.min_curvature + f32::EPSILON,
            "Bridge should be among most negative edges: bridge={}, min={}",
            bridge_curv,
            stats.min_curvature
        );
        assert!(
            bridge_curv < 0.0,
            "Bridge curvature should be negative, got {}",
            bridge_curv
        );
    }

    #[test]
    fn test_selectivity_hub_vs_concept() {
        // Hub connected to many unrelated spokes → uniform curvature → LOW selectivity
        // Concept node in a tight cluster → varied curvature → HIGH selectivity
        //
        // Topology:
        //   Hub → {S1, S2, S3, S4, S5, S6}  (star — uniform edges)
        //   A → B → C → A  (triangle — tight community)
        //   A → Hub  (bridge connecting triangle to star)
        //
        // Hub: all incident edges have similar curvature (uniform negative)
        //   → low stdev/degree → low selectivity
        // A: participates in triangle (curvature=0-ish) AND bridge to hub (curvature<<0)
        //   → high stdev across incident edges → high selectivity
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let hub = make_entity(&graph, "SelectHub");
        let s1 = make_entity(&graph, "Spoke1");
        let s2 = make_entity(&graph, "Spoke2");
        let s3 = make_entity(&graph, "Spoke3");
        let s4 = make_entity(&graph, "Spoke4");
        let s5 = make_entity(&graph, "Spoke5");
        let s6 = make_entity(&graph, "Spoke6");
        let a = make_entity(&graph, "ConceptA");
        let b = make_entity(&graph, "ConceptB");
        let c = make_entity(&graph, "ConceptC");

        // Star
        make_edge(&graph, hub, s1);
        make_edge(&graph, hub, s2);
        make_edge(&graph, hub, s3);
        make_edge(&graph, hub, s4);
        make_edge(&graph, hub, s5);
        make_edge(&graph, hub, s6);

        // Triangle
        make_edge(&graph, a, b);
        make_edge(&graph, b, c);
        make_edge(&graph, c, a);

        // Bridge
        make_edge(&graph, a, hub);

        graph.compute_forman_ricci_curvature().unwrap();

        let hub_entity = graph.get_entity(&hub).unwrap().unwrap();
        let a_entity = graph.get_entity(&a).unwrap().unwrap();

        let hub_sel = hub_entity.selectivity.expect("hub should have selectivity");
        let a_sel = a_entity
            .selectivity
            .expect("concept A should have selectivity");

        // Concept node A should have higher selectivity than the hub
        // because A's incident edges span different curvature regimes
        assert!(
            a_sel > hub_sel,
            "Concept node selectivity ({}) should exceed hub selectivity ({})",
            a_sel,
            hub_sel
        );
    }

    #[test]
    fn test_selectivity_gated_ltp_decay() {
        // Two edges with identical LTP (Full) but different endpoint_selectivity.
        // Low-selectivity edge should decay MORE (less LTP protection).
        // High-selectivity edge should decay LESS (full LTP protection).
        let now = Utc::now();
        let one_hour_ago = now - chrono::Duration::hours(1);

        let mut low_sel_edge = RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: Uuid::new_v4(),
            to_entity: Uuid::new_v4(),
            relation_type: RelationType::RelatedTo,
            strength: 0.8,
            created_at: one_hour_ago,
            valid_at: one_hour_ago,
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: one_hour_ago,
            activation_count: 20,
            ltp_status: LtpStatus::Full,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic,
            entity_confidence: None,
            forman_curvature: Some(-5.0),
            endpoint_selectivity: Some(0.05), // very low = stop-word
            provenance: Vec::new(),
            promoted_at: None,
        };

        let mut high_sel_edge = RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: Uuid::new_v4(),
            to_entity: Uuid::new_v4(),
            relation_type: RelationType::RelatedTo,
            strength: 0.8,
            created_at: one_hour_ago,
            valid_at: one_hour_ago,
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: one_hour_ago,
            activation_count: 20,
            ltp_status: LtpStatus::Full,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic,
            entity_confidence: None,
            forman_curvature: Some(-5.0),
            endpoint_selectivity: Some(2.0), // high = concept, above threshold
            provenance: Vec::new(),
            promoted_at: None,
        };

        low_sel_edge.decay();
        high_sel_edge.decay();

        // Both started at 0.8. Low-selectivity should have decayed more.
        assert!(
            low_sel_edge.strength < high_sel_edge.strength,
            "Low-selectivity edge ({}) should decay more than high-selectivity edge ({})",
            low_sel_edge.strength,
            high_sel_edge.strength
        );

        // High-selectivity edge with Full LTP should retain most of its strength
        // LTP_DECAY_FACTOR = 0.1 → 10x slower decay → after 1 hour should be ~0.79+
        assert!(
            high_sel_edge.strength > 0.7,
            "High-selectivity Full LTP edge should retain most strength, got {}",
            high_sel_edge.strength
        );
    }

    #[test]
    fn test_selectivity_none_preserves_ltp() {
        // Edge with endpoint_selectivity=None (not yet computed) should get
        // full LTP protection — conservative default.
        let now = Utc::now();
        let one_hour_ago = now - chrono::Duration::hours(1);

        let mut edge_none = RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: Uuid::new_v4(),
            to_entity: Uuid::new_v4(),
            relation_type: RelationType::RelatedTo,
            strength: 0.8,
            created_at: one_hour_ago,
            valid_at: one_hour_ago,
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: one_hour_ago,
            activation_count: 20,
            ltp_status: LtpStatus::Full,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic,
            entity_confidence: None,
            forman_curvature: None,
            endpoint_selectivity: None, // not computed yet
            provenance: Vec::new(),
            promoted_at: None,
        };

        let mut edge_high = RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: Uuid::new_v4(),
            to_entity: Uuid::new_v4(),
            relation_type: RelationType::RelatedTo,
            strength: 0.8,
            created_at: one_hour_ago,
            valid_at: one_hour_ago,
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: one_hour_ago,
            activation_count: 20,
            ltp_status: LtpStatus::Full,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic,
            entity_confidence: None,
            forman_curvature: Some(-2.0),
            endpoint_selectivity: Some(5.0), // high selectivity
            provenance: Vec::new(),
            promoted_at: None,
        };

        edge_none.decay();
        edge_high.decay();

        // Both should get full LTP protection, so strengths should be equal
        let diff = (edge_none.strength - edge_high.strength).abs();
        assert!(
            diff < 0.001,
            "None selectivity should match high selectivity: none={}, high={}, diff={}",
            edge_none.strength,
            edge_high.strength,
            diff
        );
    }

    #[test]
    fn test_endpoint_selectivity_written_to_edges() {
        // After compute_forman_ricci_curvature(), edges should have
        // endpoint_selectivity set based on their source entity's selectivity.
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let a = make_entity(&graph, "EndpointA");
        let b = make_entity(&graph, "EndpointB");
        let c = make_entity(&graph, "EndpointC");

        let e1 = make_edge(&graph, a, b);
        let e2 = make_edge(&graph, b, c);

        graph.compute_forman_ricci_curvature().unwrap();

        let edge1 = graph.get_relationship(&e1).unwrap().unwrap();
        let edge2 = graph.get_relationship(&e2).unwrap().unwrap();

        // All edges should have endpoint_selectivity set after curvature pass
        assert!(
            edge1.endpoint_selectivity.is_some(),
            "Edge 1 should have endpoint_selectivity after curvature computation"
        );
        assert!(
            edge2.endpoint_selectivity.is_some(),
            "Edge 2 should have endpoint_selectivity after curvature computation"
        );

        // A has degree 1 (only outgoing to B) — selectivity = stdev/degree = 0/1 = 0
        // B has degree 2 (in from A, out to C) — selectivity = stdev/degree
        let entity_a = graph.get_entity(&a).unwrap().unwrap();
        let entity_b = graph.get_entity(&b).unwrap().unwrap();
        assert!(
            entity_a.selectivity.is_some(),
            "Entity A should have selectivity"
        );
        assert!(
            entity_b.selectivity.is_some(),
            "Entity B should have selectivity"
        );
    }

    #[test]
    fn test_entity_reputation_returns_none_for_unknown() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        assert!(graph.get_entity_reputation("nonexistent").is_none());
    }

    #[test]
    fn test_entity_reputation_reflects_graph_state() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let hub = make_entity(&graph, "HubEntity");
        let a = make_entity(&graph, "SpokeA");
        let b = make_entity(&graph, "SpokeB");
        let c = make_entity(&graph, "SpokeC");

        make_edge(&graph, hub, a);
        make_edge(&graph, hub, b);
        make_edge(&graph, hub, c);
        make_edge(&graph, a, b); // give a,b some edges too

        graph.compute_forman_ricci_curvature().unwrap();

        let rep = graph.get_entity_reputation("HubEntity").unwrap();
        assert_eq!(rep.degree, 3);
        assert_eq!(rep.mention_count, 1);
        // Hub has lower selectivity than spokes
        let rep_a = graph.get_entity_reputation("SpokeA").unwrap();
        assert!(
            rep.selectivity <= rep_a.selectivity || rep.degree > rep_a.degree,
            "Hub should have lower selectivity or higher degree than spoke"
        );
    }

    #[test]
    fn test_entity_reputation_case_insensitive() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        make_entity(&graph, "TestEntity");
        assert!(graph.get_entity_reputation("testentity").is_some());
        assert!(graph.get_entity_reputation("TESTENTITY").is_some());
    }

    #[test]
    fn test_get_all_episodes() {
        let dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(dir.path(), None).unwrap();

        // Empty graph returns empty vec
        let episodes = graph.get_all_episodes().unwrap();
        assert!(episodes.is_empty(), "Expected empty vec for fresh graph");

        // Add an episode
        let now = Utc::now();
        let episode = EpisodicNode {
            uuid: Uuid::new_v4(),
            name: "Test Episode".to_string(),
            content: "Test content for get_all_episodes".to_string(),
            valid_at: now,
            created_at: now,
            entity_refs: Vec::new(),
            source: EpisodeSource::Message,
            metadata: std::collections::HashMap::new(),
        };
        let episode_uuid = episode.uuid;
        let episode_name = episode.name.clone();
        let episode_content = episode.content.clone();
        graph.add_episode(episode).unwrap();

        // Verify it's returned
        let episodes = graph.get_all_episodes().unwrap();
        assert_eq!(episodes.len(), 1, "Expected one episode");

        // Verify fields match
        let returned = &episodes[0];
        assert_eq!(returned.uuid, episode_uuid);
        assert_eq!(returned.name, episode_name);
        assert_eq!(returned.content, episode_content);
        assert_eq!(returned.source, EpisodeSource::Message);
    }

    #[test]
    fn test_infer_relation_person_org() {
        assert_eq!(
            infer_relation_type_for_pair(&EntityLabel::Person, &EntityLabel::Organization),
            RelationType::WorksAt
        );
    }

    #[test]
    fn test_infer_relation_person_technology() {
        assert_eq!(
            infer_relation_type_for_pair(&EntityLabel::Person, &EntityLabel::Technology),
            RelationType::Uses
        );
    }

    #[test]
    fn test_infer_relation_service_database() {
        assert_eq!(
            infer_relation_type_for_pair(&EntityLabel::Service, &EntityLabel::Database),
            RelationType::Uses
        );
    }

    #[test]
    fn test_infer_relation_module_module() {
        assert_eq!(
            infer_relation_type_for_pair(&EntityLabel::Module, &EntityLabel::Module),
            RelationType::DependsOn
        );
    }

    #[test]
    fn test_infer_relation_pipeline_env() {
        assert_eq!(
            infer_relation_type_for_pair(&EntityLabel::Pipeline, &EntityLabel::Environment),
            RelationType::DeploysTo
        );
    }

    #[test]
    fn test_infer_relation_config_service() {
        assert_eq!(
            infer_relation_type_for_pair(&EntityLabel::Configuration, &EntityLabel::Service),
            RelationType::Configures
        );
    }

    #[test]
    fn test_infer_relation_task_project() {
        assert_eq!(
            infer_relation_type_for_pair(&EntityLabel::Task, &EntityLabel::Project),
            RelationType::PartOf
        );
    }

    #[test]
    fn test_infer_relation_default_cooccurs() {
        assert_eq!(
            infer_relation_type_for_pair(&EntityLabel::Concept, &EntityLabel::Event),
            RelationType::CoOccurs
        );
    }

    // ===================================================================
    // Increment 1: robust edge provenance
    // ===================================================================

    /// Build a typed edge between two entities attributed to a source episode.
    fn provenance_edge(from: Uuid, to: Uuid, source_episode_id: Uuid) -> RelationshipEdge {
        let now = Utc::now();
        RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: from,
            to_entity: to,
            // A non-generic type so find_relationship_between_typed matches a
            // single edge across re-attestations.
            relation_type: RelationType::WorksWith,
            strength: 0.5,
            created_at: now,
            valid_at: now,
            invalidated_at: None,
            source_episode_id: Some(source_episode_id),
            context: String::new(),
            last_activated: now,
            activation_count: 1,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic,
            entity_confidence: Some(0.9),
            forman_curvature: None,
            endpoint_selectivity: None,
            provenance: Vec::new(),
            promoted_at: None,
        }
    }

    #[test]
    fn test_provenance_create_seeds_from_source_episode() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let a = make_entity(&graph, "ProvCreateA");
        let b = make_entity(&graph, "ProvCreateB");
        let episode = Uuid::new_v4();

        let edge_id = graph
            .add_relationship(provenance_edge(a, b, episode))
            .unwrap();

        let edge = graph.get_relationship(&edge_id).unwrap().unwrap();
        assert_eq!(
            edge.provenance.len(),
            1,
            "create path must seed exactly one provenance record"
        );
        assert_eq!(edge.provenance[0].source_episode_id, episode);
        assert_eq!(edge.provenance[0].mention_count, 1);
        assert_eq!(
            edge.provenance[0].confidence,
            Some(0.9),
            "confidence should be seeded from entity_confidence"
        );
    }

    #[test]
    fn test_provenance_strengthen_accumulates_sources() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let a = make_entity(&graph, "ProvAccA");
        let b = make_entity(&graph, "ProvAccB");
        let episode_a = Uuid::new_v4();
        let episode_b = Uuid::new_v4();

        // First attestation from episode A creates the edge.
        let edge_id = graph
            .add_relationship(provenance_edge(a, b, episode_a))
            .unwrap();

        // Second attestation from episode B must strengthen the SAME edge and
        // ADD a provenance record — the bug was that this episode was discarded.
        let edge_id_2 = graph
            .add_relationship(provenance_edge(a, b, episode_b))
            .unwrap();
        assert_eq!(edge_id, edge_id_2, "same entity pair + type = same edge");

        let edge = graph.get_relationship(&edge_id).unwrap().unwrap();
        assert_eq!(
            edge.provenance.len(),
            2,
            "strengthen must accumulate the second source episode (bug fix)"
        );

        // Re-attest from episode A: no new record, A's mention_count bumps to 2.
        graph
            .add_relationship(provenance_edge(a, b, episode_a))
            .unwrap();
        let edge = graph.get_relationship(&edge_id).unwrap().unwrap();
        assert_eq!(
            edge.provenance.len(),
            2,
            "re-attesting an existing episode must not add a record"
        );
        let rec_a = edge
            .provenance
            .iter()
            .find(|p| p.source_episode_id == episode_a)
            .expect("episode A record present");
        assert_eq!(
            rec_a.mention_count, 2,
            "re-attested episode A must have mention_count == 2"
        );
    }

    #[test]
    fn test_provenance_cap_enforced() {
        // Pin the cap low and deterministically so the test is hermetic.
        std::env::set_var("SHODH_PROVENANCE_MAX_SOURCES", "3");

        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();

        let a = make_entity(&graph, "ProvCapA");
        let b = make_entity(&graph, "ProvCapB");

        // Attest from 5 distinct episodes; the trail must be capped at 3.
        let mut episodes = Vec::new();
        let mut edge_id = None;
        for _ in 0..5 {
            let ep = Uuid::new_v4();
            episodes.push(ep);
            edge_id = Some(graph.add_relationship(provenance_edge(a, b, ep)).unwrap());
        }
        let edge_id = edge_id.unwrap();

        let edge = graph.get_relationship(&edge_id).unwrap().unwrap();
        assert_eq!(
            edge.provenance.len(),
            3,
            "provenance must be capped at SHODH_PROVENANCE_MAX_SOURCES"
        );
        // The most recently added episode must survive the cap.
        let last_episode = *episodes.last().unwrap();
        assert!(
            edge.provenance
                .iter()
                .any(|p| p.source_episode_id == last_episode),
            "the just-added episode must never be dropped by the cap"
        );

        std::env::remove_var("SHODH_PROVENANCE_MAX_SOURCES");
    }

    #[test]
    fn test_provenance_serde_backward_compat() {
        // Simulate a LEGACY edge serialized before the `provenance` field
        // existed: a struct identical to RelationshipEdge minus that trailing
        // field. Encoding it and decoding as the current RelationshipEdge must
        // succeed and yield an EMPTY provenance trail (the #[serde(default)]).
        #[derive(Serialize)]
        struct LegacyEdge {
            uuid: Uuid,
            from_entity: Uuid,
            to_entity: Uuid,
            relation_type: RelationType,
            strength: f32,
            created_at: DateTime<Utc>,
            valid_at: DateTime<Utc>,
            invalidated_at: Option<DateTime<Utc>>,
            source_episode_id: Option<Uuid>,
            context: String,
            last_activated: DateTime<Utc>,
            activation_count: u32,
            ltp_status: LtpStatus,
            tier: EdgeTier,
            activation_timestamps: Option<VecDeque<DateTime<Utc>>>,
            entity_confidence: Option<f32>,
            endpoint_selectivity: Option<f32>,
            forman_curvature: Option<f32>,
        }

        let legacy = LegacyEdge {
            uuid: Uuid::new_v4(),
            from_entity: Uuid::new_v4(),
            to_entity: Uuid::new_v4(),
            relation_type: RelationType::WorksWith,
            strength: 0.42,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: Some(Uuid::new_v4()),
            context: "legacy".to_string(),
            last_activated: Utc::now(),
            activation_count: 7,
            ltp_status: LtpStatus::None,
            tier: EdgeTier::L2Episodic,
            activation_timestamps: None,
            entity_confidence: Some(0.5),
            endpoint_selectivity: None,
            forman_curvature: None,
        };

        let bytes = crate::serialization::encode(&legacy).unwrap();

        // Postcard is NOT self-describing: a record missing the trailing
        // `provenance` field cannot be decoded by the plain `decode` path (it
        // runs off the end of the buffer), so `#[serde(default)]` alone does not
        // grant backward-compat. The edge read path goes through
        // `decode_relationship_edge`, which supplies the field's postcard default
        // on EOF — that is the path production uses, so that is what we assert.
        assert!(
            crate::serialization::decode::<RelationshipEdge>(&bytes).is_err(),
            "plain postcard decode of a pre-provenance edge must fail on EOF"
        );

        let (decoded, needs_migration) = decode_relationship_edge(&bytes).unwrap();
        assert!(
            needs_migration,
            "a pre-provenance edge should be flagged for rewrite"
        );
        assert_eq!(decoded.strength, 0.42);
        assert_eq!(decoded.activation_count, 7);
        assert_eq!(decoded.context, "legacy");
        assert_eq!(decoded.entity_confidence, Some(0.5));
        assert!(
            decoded.provenance.is_empty(),
            "legacy edge without provenance must decode to an empty trail"
        );

        // Forward round-trip: a current edge WITH provenance survives both the
        // plain decode and the compat decode unchanged (no migration needed).
        let mut edge = provenance_edge(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        edge.provenance = vec![ProvenanceRecord {
            source_episode_id: Uuid::new_v4(),
            mention_count: 3,
            first_observed: Utc::now(),
            last_observed: Utc::now(),
            confidence: Some(0.8),
            evidence_span: Some((0, 42)),
            typed_by: Some(TypingMethod::Cue),
        }];
        let bytes = crate::serialization::encode(&edge).unwrap();
        let decoded: RelationshipEdge = crate::serialization::decode(&bytes).unwrap();
        assert_eq!(decoded.provenance.len(), 1);
        assert_eq!(decoded.provenance[0].mention_count, 3);
        assert_eq!(decoded.provenance[0].evidence_span, Some((0, 42)));
        assert_eq!(decoded.provenance[0].typed_by, Some(TypingMethod::Cue));

        let (decoded, needs_migration) = decode_relationship_edge(&bytes).unwrap();
        assert!(
            !needs_migration,
            "a current-schema edge must not be flagged for migration"
        );
        assert_eq!(decoded.provenance.len(), 1);
    }

    #[test]
    fn coactivation_strengthen_only_creates_no_new_edges() {
        // CoRetrieved recall-time flood fix: strengthen-only must never mint a new
        // CoRetrieved edge for a co-retrieved pair that has no prior edge.
        let temp = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp.path(), None).unwrap();
        let mems: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();

        // Default (flag off) floods all C(5,2)=10 CoRetrieved edges.
        let created = graph.record_memory_coactivation_impl(&mems, false).unwrap();
        assert_eq!(
            created, 10,
            "default coactivation floods all-pairs CoRetrieved edges"
        );

        // Strengthen-only on a FRESH graph: no existing edges => nothing created.
        let temp2 = tempfile::tempdir().unwrap();
        let graph2 = GraphMemory::new(temp2.path(), None).unwrap();
        let created2 = graph2.record_memory_coactivation_impl(&mems, true).unwrap();
        assert_eq!(
            created2, 0,
            "strengthen-only must create no new edges on a fresh graph"
        );
    }

    #[test]
    fn coactivation_strengthen_only_still_strengthens_existing() {
        // Strengthen-only keeps the good half: it reinforces edges that already exist.
        let temp = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp.path(), None).unwrap();
        let mems: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
        // First pass creates the C(3,2)=3 edges.
        graph.record_memory_coactivation_impl(&mems, false).unwrap();
        // Strengthen-only pass: all 3 now exist => all strengthened, none created.
        let strengthened = graph.record_memory_coactivation_impl(&mems, true).unwrap();
        assert_eq!(
            strengthened, 3,
            "strengthen-only reinforces the existing edges"
        );
    }

    // =========================================================================
    // W1-B measurement — articulation/bridge signal distribution on a REAL graph.
    //
    // The surviving demo RocksDB stores are unloadable (`demo-data/bridge/graph`
    // has only its WAL `000004.log`; the MANIFEST/CURRENT were lost to OneDrive
    // sync — the documented file-watcher failure mode — and `defence-live` is
    // empty), so this uses the task's sanctioned fallback: ingest the W1-C bridge
    // corpus through the real production pipeline (neural NER mints the entity
    // nodes) and measure the resulting graph. Ignored (pays one full pipeline
    // ingest; not run in CI). Run explicitly:
    //   cargo test --lib measure_topology_signal_on_real_graph -- --ignored --nocapture
    // Override unit count with SHODH_TOPO_MEASURE_UNITS (default 24, the e2e size).
    // =========================================================================
    #[test]
    #[ignore = "pays one full pipeline ingest; run explicitly for the W1-B numbers table."]
    fn measure_topology_signal_on_real_graph() {
        use crate::recall_harness::bridge_harness::generate_bridge_fixtures;
        use crate::recall_harness::runner::{build_manager, ingest_corpus, EVAL_USER};

        // Deterministic single-threaded NER so the ingested topology is reproducible.
        unsafe {
            for (k, v) in [("SHODH_ONNX_THREADS", "1"), ("RAYON_NUM_THREADS", "1")] {
                if std::env::var_os(k).is_none() {
                    std::env::set_var(k, v);
                }
            }
        }

        let units: usize = std::env::var("SHODH_TOPO_MEASURE_UNITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(24);
        let fx = generate_bridge_fixtures(units, 6, 1);

        let storage =
            std::env::temp_dir().join(format!("shodh-topo-measure-{}", Uuid::new_v4().simple()));
        let _ = std::fs::remove_dir_all(&storage);
        let manager = build_manager(&storage).expect("build manager");
        ingest_corpus(&manager, &fx.corpus).expect("ingest bridge corpus");
        let graph = manager.get_user_graph(EVAL_USER).expect("user graph");
        let edges = graph.read().get_all_relationships().expect("relationships");
        let pairs: Vec<(Uuid, Uuid)> = edges.iter().map(|e| (e.from_entity, e.to_entity)).collect();

        // Distinct entities that appear in an active edge (the structural graph).
        let mut distinct: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
        for (a, b) in &pairs {
            distinct.insert(*a);
            distinct.insert(*b);
        }
        let node_count = distinct.len();

        let prot = crate::decay::compute_topology_protection(&pairs);
        let artic_nodes = prot.node_protection.len(); // only critical nodes are stored
        let bridge_edges = prot.bridge_pairs.len();

        // Continuous score distribution over the critical (nonzero) nodes.
        let mut scores: Vec<f32> = prot.node_protection.values().copied().collect();
        scores.sort_by(|a, b| a.total_cmp(b));
        let pct = |p: f64| -> f32 {
            if scores.is_empty() {
                return 0.0;
            }
            let idx = ((p / 100.0) * (scores.len() as f64 - 1.0)).round() as usize;
            scores[idx.min(scores.len() - 1)]
        };
        let artic_frac = if node_count > 0 {
            artic_nodes as f64 / node_count as f64
        } else {
            0.0
        };
        // Fraction of nodes above a candidate rescue floor (0.15) — the true
        // "would be rescued" population, which the budget cap then bounds.
        let above_floor = scores.iter().filter(|&&s| s >= 0.15).count();

        eprintln!("W1B_MEASURE corpus=bridge-fixtures units={units}");
        eprintln!(
            "W1B_MEASURE nodes={node_count} active_edges={} distinct_undirected_edges={}",
            pairs.len(),
            {
                let mut u: std::collections::HashSet<(Uuid, Uuid)> =
                    std::collections::HashSet::new();
                for (a, b) in &pairs {
                    if a != b {
                        u.insert(if a <= b { (*a, *b) } else { (*b, *a) });
                    }
                }
                u.len()
            }
        );
        eprintln!(
            "W1B_MEASURE critical_nodes={artic_nodes} articulation_fraction={artic_frac:.4} bridge_edges={bridge_edges}"
        );
        eprintln!(
            "W1B_MEASURE score_pctile p50={:.3} p75={:.3} p90={:.3} p95={:.3} p99={:.3} max={:.3}",
            pct(50.0),
            pct(75.0),
            pct(90.0),
            pct(95.0),
            pct(99.0),
            scores.last().copied().unwrap_or(0.0)
        );
        eprintln!(
            "W1B_MEASURE nodes_at_or_above_0.15={above_floor} ({:.4} of all nodes)",
            if node_count > 0 {
                above_floor as f64 / node_count as f64
            } else {
                0.0
            }
        );

        drop(graph);
        drop(manager);
        let _ = std::fs::remove_dir_all(&storage);
    }

    // =========================================================================
    // W1-B two-cluster decay simulation.
    //
    // Two dense clusters (triangles, so intra-cluster edges are redundant — NOT
    // bridges) joined by a SINGLE bridge edge. Age every edge equally past the
    // L2 prune threshold; the base gate flags all of them. With topology-aware
    // decay the bridge is rescued while the equally-old redundant cluster edges
    // are pruned; with the feature off, every flagged edge (bridge included) is
    // pruned — today's byte-identical behaviour.
    // =========================================================================

    fn l2_edge(from: Uuid, to: Uuid, last_activated: DateTime<Utc>) -> RelationshipEdge {
        RelationshipEdge {
            uuid: Uuid::new_v4(), // overwritten by add_relationship
            from_entity: from,
            to_entity: to,
            relation_type: RelationType::RelatedTo,
            strength: 0.5,
            created_at: last_activated,
            valid_at: last_activated,
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated,
            activation_count: 1,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic,
            entity_confidence: None,
            forman_curvature: None,
            endpoint_selectivity: None,
            provenance: Vec::new(),
            promoted_at: None,
        }
    }

    /// Build two triangle clusters joined by one bridge edge. Returns the graph,
    /// its tempdir (kept alive), and the bridge edge's uuid. All edges share
    /// `origin` as `last_activated`, so a single aged `decay_at(now)` flags them
    /// together.
    fn build_two_cluster_bridge_graph(
        origin: DateTime<Utc>,
    ) -> (GraphMemory, tempfile::TempDir, Uuid) {
        let dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(dir.path(), None).unwrap();

        // Cluster A = {a0,a1,a2}, Cluster B = {b0,b1,b2}. a0<->b0 is the bridge.
        let a: [Uuid; 3] = [
            Uuid::from_u128(0xA0),
            Uuid::from_u128(0xA1),
            Uuid::from_u128(0xA2),
        ];
        let b: [Uuid; 3] = [
            Uuid::from_u128(0xB0),
            Uuid::from_u128(0xB1),
            Uuid::from_u128(0xB2),
        ];
        for &id in a.iter().chain(b.iter()) {
            let entity = EntityNode {
                uuid: id,
                name: format!("e{id}"),
                labels: vec![EntityLabel::Concept],
                created_at: origin,
                last_seen_at: origin,
                mention_count: 1,
                summary: String::new(),
                attributes: std::collections::HashMap::new(),
                name_embedding: None,
                salience: 0.5,
                is_proper_noun: false,
                selectivity: None,
                fine_type: None,
                kb_id: None,
            };
            graph.add_entity(entity).unwrap();
        }

        // Dense (triangle) intra-cluster edges — redundant, never bridges.
        for &(x, y) in &[(0usize, 1usize), (1, 2), (2, 0)] {
            graph.add_relationship(l2_edge(a[x], a[y], origin)).unwrap();
            graph.add_relationship(l2_edge(b[x], b[y], origin)).unwrap();
        }
        // The single bridge edge.
        let bridge_uuid = graph.add_relationship(l2_edge(a[0], b[0], origin)).unwrap();

        (graph, dir, bridge_uuid)
    }

    #[test]
    fn topology_decay_rescues_bridge_prunes_redundant_cluster_edges() {
        let origin = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        let (graph, _dir, bridge_uuid) = build_two_cluster_bridge_graph(origin);

        // Snapshot the active edges, then age them ~60 days in one jump so the L2
        // min-prune-age (30d) is cleared and strength floors below threshold.
        let mut edges = graph.get_all_relationships().unwrap();
        assert_eq!(edges.len(), 7, "6 cluster edges + 1 bridge");
        let now = origin + Duration::days(60);

        let mut flagged: Vec<usize> = Vec::new();
        for (i, e) in edges.iter_mut().enumerate() {
            if e.decay_at(now) {
                flagged.push(i);
            }
        }
        assert_eq!(
            flagged.len(),
            7,
            "all equally-aged edges flagged by base gate"
        );

        // Topology protection over the (connectivity-unchanged) edge set.
        let prot = graph.compute_and_smooth_topology(&edges);
        assert!(
            prot.bridge_pairs.len() == 1,
            "exactly one bridge pair, got {}",
            prot.bridge_pairs.len()
        );

        // FLAG OFF (None): byte-identical to today — every flagged edge pruned.
        let off = graph.select_prune_set(&edges, &flagged, None);
        assert_eq!(off.len(), 7, "flag off prunes all flagged edges");
        assert!(off.contains(&bridge_uuid), "flag off prunes the bridge too");

        // FLAG ON: the bridge is rescued; the 6 redundant edges are pruned.
        let on = graph.select_prune_set(&edges, &flagged, Some(&prot));
        assert_eq!(on.len(), 6, "flag on rescues exactly the bridge");
        assert!(
            !on.contains(&bridge_uuid),
            "the single bridge edge must survive the prune pass"
        );
        // The rescue removes exactly the bridge from the prune set.
        let off_set: std::collections::HashSet<Uuid> = off.into_iter().collect();
        let on_set: std::collections::HashSet<Uuid> = on.into_iter().collect();
        let diff: Vec<&Uuid> = off_set.difference(&on_set).collect();
        assert_eq!(
            diff,
            vec![&bridge_uuid],
            "ON differs from OFF by only the bridge"
        );
    }

    #[test]
    fn topology_decay_flag_off_is_end_to_end_byte_identical() {
        // Full apply_decay_at path with the env flag UNSET (default): the prune
        // outcome must equal a graph with no topology consideration at all. We
        // assert the bridge is NOT spared when the feature is off.
        // Guard: this test's premise is "feature off". If the flag is set in the
        // environment the byte-identical claim can't be exercised — skip rather
        // than fail spuriously. No in-suite test sets it, so this is normally live.
        if std::env::var_os("SHODH_TOPOLOGY_AWARE_DECAY").is_some() {
            return;
        }
        let origin = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        let (graph, _dir, bridge_uuid) = build_two_cluster_bridge_graph(origin);

        let now = origin + Duration::days(60);
        let result = graph.apply_decay_at(now).unwrap();
        assert!(
            result.pruned_count >= 7,
            "all aged edges pruned when flag off"
        );
        assert!(
            graph.get_relationship(&bridge_uuid).unwrap().is_none(),
            "with the feature off the bridge is pruned like any other edge"
        );
    }

    // ------------------------------------------------------------------
    // canonicalize_entities — first tests for the merge/delete path. This
    // function deletes nodes and edges; until now it had zero coverage.
    // ------------------------------------------------------------------

    /// An entity as the canonicalizer meets one. `name_embedding` is
    /// deliberately `None`: with an embedding, add_entity's Tier-4 concept
    /// merge (cosine ≥ 0.85) would fold the duplicates at INSERT time and the
    /// FS-matcher merge path — the path under test — would never run.
    fn canon_entity(
        name: &str,
        label: EntityLabel,
        proper: bool,
        mentions: usize,
        kb_id: Option<&str>,
    ) -> EntityNode {
        EntityNode {
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            labels: vec![label],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: mentions,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: proper,
            selectivity: None,
            fine_type: None,
            kb_id: kb_id.map(str::to_string),
        }
    }

    /// The Baltimore-bridge micro-corpus: "Dali" and "the Dali" are one ship
    /// (determiner-stripped clean form and head coincide), the rest are
    /// distractors that give the unsupervised FS fit enough non-match pairs to
    /// estimate u from. `kb` optionally pins distinct QIDs on the two Dali
    /// nodes to exercise the merge veto.
    #[allow(clippy::type_complexity)]
    fn canonicalize_fixture(
        kb: Option<(&str, &str)>,
    ) -> (GraphMemory, tempfile::TempDir, Uuid, Uuid, Uuid, Uuid) {
        // Guard against vacuous passes: canonicalize_entities early-returns
        // (0, 0) without the dependency parser, which would turn every
        // assertion below into dead code. The embedded en_core_web_sm bundle
        // makes the parser unconditionally available; if this fires, the
        // bundle regressed and these tests are not testing anything.
        assert!(
            crate::dep_parser::is_available(),
            "dependency parser unavailable — canonicalize_entities tests would pass vacuously"
        );
        let dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(dir.path(), None).unwrap();

        let (dali_kb, the_dali_kb) = match kb {
            Some((a, b)) => (Some(a), Some(b)),
            None => (None, None),
        };
        let dali = graph
            .add_entity(canon_entity("Dali", EntityLabel::Vehicle, true, 5, dali_kb))
            .unwrap();
        let the_dali = graph
            .add_entity(canon_entity(
                "the Dali",
                EntityLabel::Vehicle,
                true,
                2,
                the_dali_kb,
            ))
            .unwrap();
        let baltimore = graph
            .add_entity(canon_entity(
                "Baltimore",
                EntityLabel::Location,
                true,
                3,
                None,
            ))
            .unwrap();
        let ntsb = graph
            .add_entity(canon_entity(
                "NTSB",
                EntityLabel::Organization,
                true,
                2,
                None,
            ))
            .unwrap();
        graph
            .add_entity(canon_entity(
                "Patapsco River",
                EntityLabel::Location,
                true,
                1,
                None,
            ))
            .unwrap();
        assert_ne!(dali, the_dali, "fixture must start with two Dali nodes");

        // Canonical already holds Dali→Baltimore LocatedIn, so the member's
        // copy of the same triple must DEDUP into it, not duplicate it.
        graph
            .add_relationship(universe_edge(dali, baltimore, RelationType::LocatedIn, 0.8))
            .unwrap();
        graph
            .add_relationship(universe_edge(
                the_dali,
                baltimore,
                RelationType::LocatedIn,
                0.8,
            ))
            .unwrap();
        // An edge whose OTHER endpoint moves: NTSB→member must become NTSB→canonical.
        graph
            .add_relationship(universe_edge(ntsb, the_dali, RelationType::Manages, 0.8))
            .unwrap();
        // Member↔canonical edge: re-pointing would make it canonical↔canonical,
        // so the merge must DROP it (from the member too), never re-point it.
        graph
            .add_relationship(universe_edge(the_dali, dali, RelationType::CoOccurs, 0.8))
            .unwrap();

        (graph, dir, dali, the_dali, baltimore, ntsb)
    }

    #[test]
    fn canonicalize_merges_duplicate_mention_and_repoints_edges() {
        let (graph, _dir, dali, the_dali, baltimore, ntsb) = canonicalize_fixture(None);

        let (merged, repointed) = graph.canonicalize_entities().unwrap();
        assert_eq!(merged, 1, "exactly the Dali pair must merge");
        // LocatedIn (dedup into the canonical's copy) + Manages; the self-loop
        // CoOccurs edge is dropped, not re-pointed.
        assert_eq!(repointed, 2, "both non-loop member edges must re-point");

        // Survival: the proper, most-mentioned node wins; the member is gone.
        // This pins canonical selection to (is_proper_noun, mention_count) — if
        // the is_proper_noun derivation changes upstream, this fails loudly,
        // because that IS a change to which node survives entity resolution.
        let survivor = graph.get_entity(&dali).unwrap();
        assert!(survivor.is_some(), "canonical node must survive the merge");
        assert_eq!(survivor.unwrap().name, "Dali");
        assert!(
            graph.get_entity(&the_dali).unwrap().is_none(),
            "merged member node must be deleted"
        );

        // NO EDGE LOST (bug-1 regression): every pre-merge triple, with the
        // member mapped onto the canonical, exists after — LocatedIn deduped,
        // Manages re-pointed — and nothing dangles from the deleted member.
        let edges = graph.get_entity_relationships(&dali).unwrap();
        assert!(
            edges.iter().any(|e| e.from_entity == dali
                && e.to_entity == baltimore
                && e.relation_type == RelationType::LocatedIn),
            "Dali→Baltimore LocatedIn must survive the merge"
        );
        assert!(
            edges.iter().any(|e| e.from_entity == ntsb
                && e.to_entity == dali
                && e.relation_type == RelationType::Manages),
            "NTSB→member edge must be re-pointed to NTSB→canonical"
        );
        assert!(
            edges.iter().all(|e| e.from_entity != e.to_entity),
            "merge-created self-loops must be dropped, not re-pointed"
        );
        assert!(
            edges
                .iter()
                .all(|e| e.from_entity != the_dali && e.to_entity != the_dali),
            "no surviving edge may reference the deleted member"
        );
        assert_eq!(
            edges.len(),
            2,
            "canonical must hold exactly the deduped LocatedIn + the re-pointed Manages"
        );
        assert!(
            graph
                .get_entity_relationships(&the_dali)
                .unwrap()
                .is_empty(),
            "the deleted member must hold no edges"
        );

        // The merged surface is seeded as an alias of the canonical (raw and
        // determiner-stripped forms), closing the ingest loop.
        assert_eq!(graph.resolve_alias("the Dali"), Some(dali));
        assert_eq!(graph.resolve_alias("dali"), Some(dali));
    }

    #[test]
    fn canonicalize_refuses_merge_across_distinct_kb_identities() {
        // Same surfaces, same scores — but the KB says these are two different
        // real-world things. The veto must hold regardless of match strength.
        let (graph, _dir, dali, the_dali, _baltimore, _ntsb) =
            canonicalize_fixture(Some(("Q107647811", "Q4655133")));

        let (merged, _repointed) = graph.canonicalize_entities().unwrap();
        assert_eq!(merged, 0, "distinct QIDs must veto the merge");
        assert!(
            graph.get_entity(&dali).unwrap().is_some()
                && graph.get_entity(&the_dali).unwrap().is_some(),
            "both KB-distinct nodes must survive"
        );
        // The member keeps its full edge set — nothing was re-pointed or dropped.
        assert_eq!(
            graph.get_entity_relationships(&the_dali).unwrap().len(),
            3,
            "a vetoed member's edges must be untouched"
        );
        assert_eq!(
            graph.resolve_alias("the Dali"),
            None,
            "no alias may be seeded for a vetoed merge"
        );
    }
}
