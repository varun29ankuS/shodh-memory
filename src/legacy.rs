//! Legacy v0.2.0 schema mirrors for schema-drift recovery.
//!
//! The npm-bundled v0.2.0 server (Jul 2 - Aug 19, 2026) wrote graph records
//! with struct layouts that differ from the current build:
//!   - `EntityLabel` gained 12 variants (Norp, Gpe, Facility, Vehicle,
//!     Weapon, Work, Law, Title, Cyber, Money, Quantity, Time) mid-enum,
//!     shifting every index after Module; an old `Other(String)` (index 23)
//!     now decodes as `Module` and the trailing String corrupts the next
//!     field.
//!   - `EntityNode` gained trailing fields `fine_type`/`kb_id`.
//!   - `RelationshipEdge` gained trailing fields `provenance`/`promoted_at`.
//!
//! These mirrors decode the old bytes (via `try_decode`, which handles both
//! the postcard tag and the bincode fallbacks); the `From` impls convert to
//! the current types with defaults for the added fields. Used by the
//! migration command to repair records in place.
//! DO NOT rename/reorder these fields — they must stay byte-compatible with
//! v0.2.0 payloads.

use std::collections::{HashMap, VecDeque};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::graph_memory::{
    EdgeTier, EntityLabel, EntityNode, LtpStatus, ProvenanceRecord, RelationType, RelationshipEdge,
};
use crate::serialization;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LegacyEntityLabel {
    Person,
    Organization,
    Location,
    Technology,
    Concept,
    Event,
    Date,
    Product,
    Skill,
    Keyword,
    Project,
    Task,
    Document,
    Repository,
    Service,
    Database,
    Metric,
    Configuration,
    Environment,
    Pipeline,
    Team,
    Role,
    Module,
    Other(String),
}

impl From<LegacyEntityLabel> for EntityLabel {
    fn from(l: LegacyEntityLabel) -> Self {
        match l {
            LegacyEntityLabel::Person => EntityLabel::Person,
            LegacyEntityLabel::Organization => EntityLabel::Organization,
            LegacyEntityLabel::Location => EntityLabel::Location,
            LegacyEntityLabel::Technology => EntityLabel::Technology,
            LegacyEntityLabel::Concept => EntityLabel::Concept,
            LegacyEntityLabel::Event => EntityLabel::Event,
            LegacyEntityLabel::Date => EntityLabel::Date,
            LegacyEntityLabel::Product => EntityLabel::Product,
            LegacyEntityLabel::Skill => EntityLabel::Skill,
            LegacyEntityLabel::Keyword => EntityLabel::Keyword,
            LegacyEntityLabel::Project => EntityLabel::Project,
            LegacyEntityLabel::Task => EntityLabel::Task,
            LegacyEntityLabel::Document => EntityLabel::Document,
            LegacyEntityLabel::Repository => EntityLabel::Repository,
            LegacyEntityLabel::Service => EntityLabel::Service,
            LegacyEntityLabel::Database => EntityLabel::Database,
            LegacyEntityLabel::Metric => EntityLabel::Metric,
            LegacyEntityLabel::Configuration => EntityLabel::Configuration,
            LegacyEntityLabel::Environment => EntityLabel::Environment,
            LegacyEntityLabel::Pipeline => EntityLabel::Pipeline,
            LegacyEntityLabel::Team => EntityLabel::Team,
            LegacyEntityLabel::Role => EntityLabel::Role,
            LegacyEntityLabel::Module => EntityLabel::Module,
            LegacyEntityLabel::Other(s) => EntityLabel::Other(s),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyEntityNode {
    pub uuid: Uuid,
    pub name: String,
    pub labels: Vec<LegacyEntityLabel>,
    pub created_at: DateTime<Utc>,
    pub last_seen_at: DateTime<Utc>,
    pub mention_count: usize,
    pub summary: String,
    pub attributes: HashMap<String, String>,
    pub name_embedding: Option<Vec<f32>>,
    pub salience: f32,
    pub is_proper_noun: bool,
    pub selectivity: Option<f32>,
}

impl From<LegacyEntityNode> for EntityNode {
    fn from(n: LegacyEntityNode) -> Self {
        EntityNode {
            uuid: n.uuid,
            name: n.name,
            labels: n.labels.into_iter().map(Into::into).collect(),
            created_at: n.created_at,
            last_seen_at: n.last_seen_at,
            mention_count: n.mention_count,
            summary: n.summary,
            attributes: n.attributes,
            name_embedding: n.name_embedding,
            salience: n.salience,
            is_proper_noun: n.is_proper_noun,
            selectivity: n.selectivity,
            fine_type: None,
            kb_id: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyRelationshipEdge {
    pub uuid: Uuid,
    pub from_entity: Uuid,
    pub to_entity: Uuid,
    pub relation_type: RelationType,
    pub strength: f32,
    pub created_at: DateTime<Utc>,
    pub valid_at: DateTime<Utc>,
    pub invalidated_at: Option<DateTime<Utc>>,
    pub source_episode_id: Option<Uuid>,
    pub context: String,
    pub last_activated: DateTime<Utc>,
    pub activation_count: u32,
    pub ltp_status: LtpStatus,
    pub tier: EdgeTier,
    pub activation_timestamps: Option<VecDeque<DateTime<Utc>>>,
    pub entity_confidence: Option<f32>,
    pub endpoint_selectivity: Option<f32>,
    pub forman_curvature: Option<f32>,
}

impl From<LegacyRelationshipEdge> for RelationshipEdge {
    fn from(e: LegacyRelationshipEdge) -> Self {
        RelationshipEdge {
            uuid: e.uuid,
            from_entity: e.from_entity,
            to_entity: e.to_entity,
            relation_type: e.relation_type,
            strength: e.strength,
            created_at: e.created_at,
            valid_at: e.valid_at,
            invalidated_at: e.invalidated_at,
            source_episode_id: e.source_episode_id,
            context: e.context,
            last_activated: e.last_activated,
            activation_count: e.activation_count,
            ltp_status: e.ltp_status,
            tier: e.tier,
            activation_timestamps: e.activation_timestamps,
            entity_confidence: e.entity_confidence,
            endpoint_selectivity: e.endpoint_selectivity,
            forman_curvature: e.forman_curvature,
            provenance: Vec::<ProvenanceRecord>::new(),
            promoted_at: None,
        }
    }
}

/// Decode a v0.2.0-era entity record (tagged postcard or legacy bincode) into
/// a current-schema `EntityNode`.
pub fn decode_legacy_entity(value: &[u8]) -> Result<EntityNode, String> {
    let (legacy, _): (LegacyEntityNode, bool) =
        serialization::try_decode(value).map_err(|e| format!("legacy EntityNode: {e}"))?;
    Ok(legacy.into())
}

/// Decode a v0.2.0-era relationship record (tagged postcard or legacy bincode)
/// into a current-schema `RelationshipEdge`.
pub fn decode_legacy_relationship(value: &[u8]) -> Result<RelationshipEdge, String> {
    let (legacy, _): (LegacyRelationshipEdge, bool) =
        serialization::try_decode(value).map_err(|e| format!("legacy RelationshipEdge: {e}"))?;
    Ok(legacy.into())
}
