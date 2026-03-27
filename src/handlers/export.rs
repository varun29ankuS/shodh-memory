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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_memory::{EntityLabel, EntityNode, EpisodicNode, EpisodeSource};
    use chrono::Utc;
    use std::collections::HashMap;
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
}
