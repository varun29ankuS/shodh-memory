//! Graph export types and node/edge builder functions.
//!
//! Defines the serializable types for the graph snapshot export API and
//! provides conversion functions from internal types to the export format.

use chrono::{DateTime, Utc};
use serde::Serialize;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Export types
// ---------------------------------------------------------------------------

/// Top-level response for a full graph export.
#[derive(Debug, Serialize)]
pub struct GraphExportResponse {
    pub metadata: ExportMetadata,
    pub nodes: Vec<ExportNode>,
    pub edges: Vec<ExportEdge>,
}

/// Summary statistics for the export.
#[derive(Debug, Serialize)]
pub struct ExportMetadata {
    pub exported_at: DateTime<Utc>,
    pub user_id: String,
    pub node_count: usize,
    pub edge_count: usize,
    pub node_counts_by_type: HashMap<String, usize>,
    pub edge_counts_by_type: HashMap<String, usize>,
}

/// A single graph node in the export format.
#[derive(Debug, Clone, Serialize)]
pub struct ExportNode {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
    pub label: String,
    pub attributes: serde_json::Value,
}

/// A single directed graph edge in the export format.
#[derive(Debug, Clone, Serialize)]
pub struct ExportEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    #[serde(rename = "type")]
    pub edge_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub attributes: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Node builder functions
// ---------------------------------------------------------------------------

/// Convert an [`EntityNode`] to an [`ExportNode`].
///
/// `include_embeddings` controls whether the raw `name_embedding` vector is
/// serialised into `attributes.embedding`.  Embeddings are large (384–1536
/// floats) so callers should omit them unless downstream consumers need them.
pub fn entity_to_node(
    entity: &crate::graph_memory::EntityNode,
    include_embeddings: bool,
) -> ExportNode {
    let labels_vec: Vec<String> = entity
        .labels
        .iter()
        .map(|l| l.as_str().to_owned())
        .collect();

    let mut attrs = serde_json::json!({
        "salience": entity.salience,
        "mention_count": entity.mention_count,
        "is_proper_noun": entity.is_proper_noun,
        "labels": labels_vec,
        "created_at": entity.created_at,
        "last_seen_at": entity.last_seen_at,
        "summary": entity.summary,
    });

    if !entity.attributes.is_empty() {
        attrs["entity_attributes"] =
            serde_json::to_value(&entity.attributes).unwrap_or(serde_json::Value::Null);
    }

    if include_embeddings {
        if let Some(ref emb) = entity.name_embedding {
            attrs["embedding"] =
                serde_json::to_value(emb).unwrap_or(serde_json::Value::Null);
        }
    }

    ExportNode {
        id: entity.uuid.to_string(),
        node_type: "entity".to_owned(),
        label: entity.name.clone(),
        attributes: attrs,
    }
}

/// Convert a [`Memory`] to an [`ExportNode`].
///
/// `include_embeddings` controls whether the experience embedding vector is
/// included in `attributes.embedding`.
pub fn memory_to_node(
    memory: &crate::memory::Memory,
    include_embeddings: bool,
) -> ExportNode {
    // Truncate content for the label (display-friendly summary).
    let label = if memory.experience.content.len() > 100 {
        let boundary = memory.experience.content.floor_char_boundary(97);
        format!("{}...", &memory.experience.content[..boundary])
    } else {
        memory.experience.content.clone()
    };

    let mut attrs = serde_json::json!({
        "content": memory.experience.content,
        "importance": memory.importance(),
        "tier": format!("{:?}", memory.tier),
        "access_count": memory.access_count(),
        "last_accessed": memory.last_accessed(),
        "temporal_relevance": memory.temporal_relevance(),
        "activation": memory.activation(),
        "experience_type": format!("{:?}", memory.experience.experience_type),
        "created_at": memory.created_at,
    });

    if let Some(ref agent_id) = memory.agent_id {
        attrs["agent_id"] = serde_json::Value::String(agent_id.clone());
    }
    if let Some(ref run_id) = memory.run_id {
        attrs["run_id"] = serde_json::Value::String(run_id.clone());
    }

    if include_embeddings {
        if let Some(ref emb) = memory.experience.embeddings {
            attrs["embedding"] =
                serde_json::to_value(emb).unwrap_or(serde_json::Value::Null);
        }
    }

    ExportNode {
        id: memory.id.0.to_string(),
        node_type: "memory".to_owned(),
        label,
        attributes: attrs,
    }
}

/// Convert an [`EpisodicNode`] to an [`ExportNode`].
pub fn episode_to_node(episode: &crate::graph_memory::EpisodicNode) -> ExportNode {
    let attrs = serde_json::json!({
        "content": episode.content,
        "source": format!("{:?}", episode.source),
        "valid_at": episode.valid_at,
        "created_at": episode.created_at,
    });

    ExportNode {
        id: episode.uuid.to_string(),
        node_type: "episode".to_owned(),
        label: episode.name.clone(),
        attributes: attrs,
    }
}

// ---------------------------------------------------------------------------
// Edge builder functions
// ---------------------------------------------------------------------------

/// Convert a [`RelationshipEdge`] to an [`ExportEdge`].
pub fn relationship_to_edge(edge: &crate::graph_memory::RelationshipEdge) -> ExportEdge {
    let attrs = serde_json::json!({
        "strength": edge.strength,
        "relation_type": format!("{:?}", edge.relation_type),
        "ltp_status": format!("{:?}", edge.ltp_status),
        "tier": format!("{:?}", edge.tier),
        "activation_count": edge.activation_count,
        "last_activated": edge.last_activated,
        "created_at": edge.created_at,
        "valid_at": edge.valid_at,
        "entity_confidence": edge.entity_confidence,
    });

    ExportEdge {
        id: edge.uuid.to_string(),
        source: edge.from_entity.to_string(),
        target: edge.to_entity.to_string(),
        edge_type: "relationship".to_owned(),
        label: Some(format!("{:?}", edge.relation_type)),
        attributes: attrs,
    }
}

/// Synthesize `entity_ref` edges from a [`Memory`]'s entity references.
///
/// Each [`EntityRef`] becomes a directed edge from the memory node to the
/// referenced entity node, with the `relation` field preserved as an attribute.
pub fn entity_refs_to_edges(
    source_id: &uuid::Uuid,
    refs: &[crate::memory::EntityRef],
) -> Vec<ExportEdge> {
    refs.iter()
        .map(|r| ExportEdge {
            id: format!("{}-{}", source_id, r.entity_id),
            source: source_id.to_string(),
            target: r.entity_id.to_string(),
            edge_type: "entity_ref".to_owned(),
            label: None,
            attributes: serde_json::json!({ "relation": r.relation }),
        })
        .collect()
}

/// Synthesize `entity_ref` edges from an [`EpisodicNode`]'s entity UUID list.
///
/// Each entity UUID becomes a directed edge from the episode node to the entity
/// node, with a static `"referenced"` relation attribute.
pub fn episode_refs_to_edges(
    episode_id: &uuid::Uuid,
    entity_ids: &[uuid::Uuid],
) -> Vec<ExportEdge> {
    entity_ids
        .iter()
        .map(|entity_id| ExportEdge {
            id: format!("{}-{}", episode_id, entity_id),
            source: episode_id.to_string(),
            target: entity_id.to_string(),
            edge_type: "entity_ref".to_owned(),
            label: None,
            attributes: serde_json::json!({ "relation": "referenced" }),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_memory::{
        EdgeTier, EntityLabel, EntityNode, EpisodicNode, EpisodeSource, LtpStatus,
        RelationshipEdge, RelationType,
    };
    use crate::memory::EntityRef;
    use chrono::Utc;
    use std::collections::{HashMap, VecDeque};
    use uuid::Uuid;

    fn make_entity() -> EntityNode {
        EntityNode {
            uuid: Uuid::new_v4(),
            name: "Ferris".to_owned(),
            labels: vec![EntityLabel::Person],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 3,
            summary: "The Rust mascot".to_owned(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.8,
            is_proper_noun: true,
        }
    }

    #[test]
    fn test_entity_to_node() {
        let entity = make_entity();
        let node = entity_to_node(&entity, false);

        assert_eq!(node.node_type, "entity");
        assert_eq!(node.label, "Ferris");
        assert_eq!(node.id, entity.uuid.to_string());

        let attrs = &node.attributes;
        // f32 → JSON → f64 has a small representation difference; check approx.
        let salience = attrs["salience"].as_f64().unwrap();
        assert!((salience - 0.8_f64).abs() < 1e-5, "salience {salience} not ~0.8");
        assert_eq!(attrs["mention_count"], 3_u64);
        assert_eq!(attrs["is_proper_noun"], true);
        assert!(attrs["labels"].is_array());
        assert_eq!(attrs["labels"][0], "Person");
        assert_eq!(attrs["summary"], "The Rust mascot");
        // No entity_attributes key when HashMap is empty
        assert!(attrs.get("entity_attributes").is_none());
        // No embedding when flag is false
        assert!(attrs.get("embedding").is_none());
    }

    #[test]
    fn test_entity_to_node_with_embeddings() {
        let mut entity = make_entity();
        entity.name_embedding = Some(vec![0.1, 0.2, 0.3]);

        let node_with = entity_to_node(&entity, true);
        assert!(node_with.attributes.get("embedding").is_some());
        let emb = &node_with.attributes["embedding"];
        assert!(emb.is_array());
        assert_eq!(emb.as_array().unwrap().len(), 3);

        let node_without = entity_to_node(&entity, false);
        assert!(node_without.attributes.get("embedding").is_none());
    }

    #[test]
    fn test_entity_to_node_with_custom_attributes() {
        let mut entity = make_entity();
        entity.attributes.insert("role".to_owned(), "mascot".to_owned());

        let node = entity_to_node(&entity, false);
        let ea = &node.attributes["entity_attributes"];
        assert!(ea.is_object());
        assert_eq!(ea["role"], "mascot");
    }

    #[test]
    fn test_episode_to_node() {
        let episode = EpisodicNode {
            uuid: Uuid::new_v4(),
            name: "First meeting".to_owned(),
            content: "We discussed the project roadmap".to_owned(),
            valid_at: Utc::now(),
            created_at: Utc::now(),
            entity_refs: vec![],
            source: EpisodeSource::Message,
            metadata: HashMap::new(),
        };

        let node = episode_to_node(&episode);

        assert_eq!(node.node_type, "episode");
        assert_eq!(node.label, "First meeting");
        assert_eq!(node.id, episode.uuid.to_string());

        let attrs = &node.attributes;
        assert_eq!(attrs["content"], "We discussed the project roadmap");
        assert_eq!(attrs["source"], "Message");
        assert!(attrs["valid_at"].is_string());
        assert!(attrs["created_at"].is_string());
    }

    fn make_relationship_edge() -> RelationshipEdge {
        RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: Uuid::new_v4(),
            to_entity: Uuid::new_v4(),
            relation_type: RelationType::WorksWith,
            strength: 0.75,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: None,
            context: "test context".to_owned(),
            last_activated: Utc::now(),
            activation_count: 5,
            ltp_status: LtpStatus::None,
            tier: EdgeTier::L1Working,
            activation_timestamps: Some(VecDeque::new()),
            entity_confidence: Some(0.9),
        }
    }

    #[test]
    fn test_relationship_to_edge() {
        let rel = make_relationship_edge();
        let edge = relationship_to_edge(&rel);

        assert_eq!(edge.id, rel.uuid.to_string());
        assert_eq!(edge.source, rel.from_entity.to_string());
        assert_eq!(edge.target, rel.to_entity.to_string());
        assert_eq!(edge.edge_type, "relationship");
        assert_eq!(edge.label.as_deref(), Some("WorksWith"));

        let attrs = &edge.attributes;
        let strength = attrs["strength"].as_f64().unwrap();
        assert!((strength - 0.75_f64).abs() < 1e-5, "strength {strength} not ~0.75");
        assert_eq!(attrs["relation_type"], "WorksWith");
        assert_eq!(attrs["ltp_status"], "None");
        assert_eq!(attrs["tier"], "L1Working");
        assert_eq!(attrs["activation_count"], 5_u64);
        assert!(attrs["last_activated"].is_string());
        assert!(attrs["created_at"].is_string());
        assert!(attrs["valid_at"].is_string());
        let confidence = attrs["entity_confidence"].as_f64().unwrap();
        assert!((confidence - 0.9_f64).abs() < 1e-5);
    }

    #[test]
    fn test_entity_ref_edges() {
        let source_id = Uuid::new_v4();
        let entity_a = Uuid::new_v4();
        let entity_b = Uuid::new_v4();
        let refs = vec![
            EntityRef { entity_id: entity_a, name: "Alice".to_owned(), relation: "subject".to_owned() },
            EntityRef { entity_id: entity_b, name: "Rust".to_owned(), relation: "mentioned".to_owned() },
        ];

        let edges = entity_refs_to_edges(&source_id, &refs);

        assert_eq!(edges.len(), 2);
        for (edge, r) in edges.iter().zip(refs.iter()) {
            assert_eq!(edge.id, format!("{}-{}", source_id, r.entity_id));
            assert_eq!(edge.source, source_id.to_string());
            assert_eq!(edge.target, r.entity_id.to_string());
            assert_eq!(edge.edge_type, "entity_ref");
            assert!(edge.label.is_none());
            assert_eq!(edge.attributes["relation"], r.relation.as_str());
        }
    }

    #[test]
    fn test_episode_entity_ref_edges() {
        let episode_id = Uuid::new_v4();
        let entity_a = Uuid::new_v4();
        let entity_b = Uuid::new_v4();
        let entity_ids = vec![entity_a, entity_b];

        let edges = episode_refs_to_edges(&episode_id, &entity_ids);

        assert_eq!(edges.len(), 2);
        for (edge, entity_id) in edges.iter().zip(entity_ids.iter()) {
            assert_eq!(edge.id, format!("{}-{}", episode_id, entity_id));
            assert_eq!(edge.source, episode_id.to_string());
            assert_eq!(edge.target, entity_id.to_string());
            assert_eq!(edge.edge_type, "entity_ref");
            assert!(edge.label.is_none());
            assert_eq!(edge.attributes["relation"], "referenced");
        }
    }
}
