//! Graph export types, node/edge builder functions, and the export handler.
//!
//! Defines the serializable types for the graph snapshot export API,
//! provides conversion functions from internal types to the export format,
//! and implements the GET /api/graph/{user_id}/export endpoint.

use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
use chrono::{DateTime, Utc};
use serde::Serialize;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::sync::Arc;

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::validation;

type AppState = Arc<MultiUserMemoryManager>;

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
            attrs["embedding"] = serde_json::to_value(emb).unwrap_or(serde_json::Value::Null);
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
pub fn memory_to_node(memory: &crate::memory::Memory, include_embeddings: bool) -> ExportNode {
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
            attrs["embedding"] = serde_json::to_value(emb).unwrap_or(serde_json::Value::Null);
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
// GEXF serialization
// ---------------------------------------------------------------------------

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            '\t' | '\n' | '\r' => out.push(c),
            c if (c as u32) < 0x20 => {}
            c => out.push(c),
        }
    }
    out
}

/// Serialize a [`GraphExportResponse`] to a GEXF 1.3 XML string.
///
/// Uses manual string building (no external XML library) since the schema is fixed.
pub fn to_gexf(export: &GraphExportResponse) -> String {
    let date = export.metadata.exported_at.format("%Y-%m-%d").to_string();
    let mut out = String::new();

    writeln!(out, r#"<?xml version="1.0" encoding="UTF-8"?>"#).unwrap();
    writeln!(out, r#"<gexf xmlns="http://gexf.net/1.3" version="1.3">"#).unwrap();
    writeln!(out, r#"  <meta lastmodifieddate="{date}">"#).unwrap();
    writeln!(out, r#"    <creator>shodh-memory</creator>"#).unwrap();
    writeln!(out, r#"  </meta>"#).unwrap();
    writeln!(out, r#"  <graph defaultedgetype="directed" mode="static">"#).unwrap();

    // Node attribute declarations
    writeln!(out, r#"    <attributes class="node">"#).unwrap();
    writeln!(
        out,
        r#"      <attribute id="0" title="type" type="string"/>"#
    )
    .unwrap();
    writeln!(
        out,
        r#"      <attribute id="1" title="importance" type="float"/>"#
    )
    .unwrap();
    writeln!(
        out,
        r#"      <attribute id="2" title="salience" type="float"/>"#
    )
    .unwrap();
    writeln!(
        out,
        r#"      <attribute id="3" title="tier" type="string"/>"#
    )
    .unwrap();
    writeln!(
        out,
        r#"      <attribute id="4" title="access_count" type="integer"/>"#
    )
    .unwrap();
    writeln!(
        out,
        r#"      <attribute id="5" title="activation" type="float"/>"#
    )
    .unwrap();
    writeln!(
        out,
        r#"      <attribute id="6" title="mention_count" type="integer"/>"#
    )
    .unwrap();
    writeln!(
        out,
        r#"      <attribute id="7" title="experience_type" type="string"/>"#
    )
    .unwrap();
    writeln!(out, r#"    </attributes>"#).unwrap();

    // Edge attribute declarations
    writeln!(out, r#"    <attributes class="edge">"#).unwrap();
    writeln!(
        out,
        r#"      <attribute id="0" title="type" type="string"/>"#
    )
    .unwrap();
    writeln!(
        out,
        r#"      <attribute id="1" title="ltp_status" type="string"/>"#
    )
    .unwrap();
    writeln!(
        out,
        r#"      <attribute id="2" title="tier" type="string"/>"#
    )
    .unwrap();
    writeln!(
        out,
        r#"      <attribute id="3" title="activation_count" type="integer"/>"#
    )
    .unwrap();
    writeln!(out, r#"    </attributes>"#).unwrap();

    // Nodes
    writeln!(out, r#"    <nodes>"#).unwrap();
    for node in &export.nodes {
        let id = xml_escape(&node.id);
        let label = xml_escape(&node.label);
        writeln!(out, r#"      <node id="{id}" label="{label}">"#).unwrap();
        writeln!(out, r#"        <attvalues>"#).unwrap();

        // for="0" type — always emitted
        let node_type = xml_escape(&node.node_type);
        writeln!(out, r#"          <attvalue for="0" value="{node_type}"/>"#).unwrap();

        // for="1" importance
        if let Some(v) = node.attributes.get("importance").and_then(|v| v.as_f64()) {
            writeln!(out, r#"          <attvalue for="1" value="{v}"/>"#).unwrap();
        }
        // for="2" salience
        if let Some(v) = node.attributes.get("salience").and_then(|v| v.as_f64()) {
            writeln!(out, r#"          <attvalue for="2" value="{v}"/>"#).unwrap();
        }
        // for="3" tier
        if let Some(v) = node.attributes.get("tier").and_then(|v| v.as_str()) {
            let v = xml_escape(v);
            writeln!(out, r#"          <attvalue for="3" value="{v}"/>"#).unwrap();
        }
        // for="4" access_count
        if let Some(v) = node.attributes.get("access_count").and_then(|v| v.as_u64()) {
            writeln!(out, r#"          <attvalue for="4" value="{v}"/>"#).unwrap();
        }
        // for="5" activation
        if let Some(v) = node.attributes.get("activation").and_then(|v| v.as_f64()) {
            writeln!(out, r#"          <attvalue for="5" value="{v}"/>"#).unwrap();
        }
        // for="6" mention_count
        if let Some(v) = node
            .attributes
            .get("mention_count")
            .and_then(|v| v.as_u64())
        {
            writeln!(out, r#"          <attvalue for="6" value="{v}"/>"#).unwrap();
        }
        // for="7" experience_type
        if let Some(v) = node
            .attributes
            .get("experience_type")
            .and_then(|v| v.as_str())
        {
            let v = xml_escape(v);
            writeln!(out, r#"          <attvalue for="7" value="{v}"/>"#).unwrap();
        }

        writeln!(out, r#"        </attvalues>"#).unwrap();
        writeln!(out, r#"      </node>"#).unwrap();
    }
    writeln!(out, r#"    </nodes>"#).unwrap();

    // Edges
    writeln!(out, r#"    <edges>"#).unwrap();
    for edge in &export.edges {
        let id = xml_escape(&edge.id);
        let source = xml_escape(&edge.source);
        let target = xml_escape(&edge.target);
        let weight = edge
            .attributes
            .get("strength")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let label_attr = match &edge.label {
            Some(l) => format!(r#" label="{}""#, xml_escape(l)),
            None => String::new(),
        };
        writeln!(
            out,
            r#"      <edge id="{id}" source="{source}" target="{target}" weight="{weight}"{label_attr}>"#
        )
        .unwrap();
        writeln!(out, r#"        <attvalues>"#).unwrap();

        // for="0" type — always emitted
        let edge_type = xml_escape(&edge.edge_type);
        writeln!(out, r#"          <attvalue for="0" value="{edge_type}"/>"#).unwrap();

        // for="1" ltp_status
        if let Some(v) = edge.attributes.get("ltp_status").and_then(|v| v.as_str()) {
            let v = xml_escape(v);
            writeln!(out, r#"          <attvalue for="1" value="{v}"/>"#).unwrap();
        }
        // for="2" tier
        if let Some(v) = edge.attributes.get("tier").and_then(|v| v.as_str()) {
            let v = xml_escape(v);
            writeln!(out, r#"          <attvalue for="2" value="{v}"/>"#).unwrap();
        }
        // for="3" activation_count
        if let Some(v) = edge
            .attributes
            .get("activation_count")
            .and_then(|v| v.as_u64())
        {
            writeln!(out, r#"          <attvalue for="3" value="{v}"/>"#).unwrap();
        }

        writeln!(out, r#"        </attvalues>"#).unwrap();
        writeln!(out, r#"      </edge>"#).unwrap();
    }
    writeln!(out, r#"    </edges>"#).unwrap();

    writeln!(out, r#"  </graph>"#).unwrap();
    write!(out, r#"</gexf>"#).unwrap();

    out
}

// ---------------------------------------------------------------------------
// Query params and handler
// ---------------------------------------------------------------------------

/// Query parameters for the graph export endpoint.
#[derive(Debug, serde::Deserialize)]
pub struct ExportParams {
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default = "default_include")]
    pub include: String,
    #[serde(default)]
    pub min_importance: f32,
    #[serde(default)]
    pub include_embeddings: bool,
}

fn default_format() -> String {
    "json".to_string()
}
fn default_include() -> String {
    "entities,memories,episodes".to_string()
}

/// Parse the comma-separated `include` param into (entities, memories, episodes) flags.
fn parse_include(include: &str) -> (bool, bool, bool) {
    let parts: Vec<&str> = include.split(',').map(|s| s.trim()).collect();
    (
        parts.contains(&"entities"),
        parts.contains(&"memories"),
        parts.contains(&"episodes"),
    )
}

/// GET /api/graph/{user_id}/export - Export the full knowledge graph as JSON or GEXF
pub async fn export_graph(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(params): Query<ExportParams>,
) -> Result<axum::response::Response, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let (inc_entities, inc_memories, inc_episodes) = parse_include(&params.include);

    let mut nodes: Vec<ExportNode> = Vec::new();
    let mut edges: Vec<ExportEdge> = Vec::new();

    // --- Graph data: entities, relationships, episodes ---
    if inc_entities || inc_episodes {
        if let Ok(graph) = state.get_user_graph(&user_id) {
            let graph_guard = graph.read();

            if inc_entities {
                if let Ok(entities) = graph_guard.get_all_entities() {
                    for entity in &entities {
                        nodes.push(entity_to_node(entity, params.include_embeddings));
                    }
                }
                if let Ok(relationships) = graph_guard.get_all_relationships() {
                    for rel in &relationships {
                        edges.push(relationship_to_edge(rel));
                    }
                }
            }

            if inc_episodes {
                if let Ok(episodes) = graph_guard.get_all_episodes() {
                    for episode in &episodes {
                        nodes.push(episode_to_node(episode));
                        if inc_entities {
                            edges
                                .extend(episode_refs_to_edges(&episode.uuid, &episode.entity_refs));
                        }
                    }
                }
            }
        }
    }

    // --- Memory data ---
    if inc_memories {
        if let Ok(mem_sys) = state.get_user_memory(&user_id) {
            let mem_guard = mem_sys.read();
            if let Ok(memories) = mem_guard.get_all_memories() {
                for memory in &memories {
                    if memory.importance() < params.min_importance {
                        continue;
                    }
                    if inc_entities {
                        edges.extend(entity_refs_to_edges(&memory.id.0, &memory.entity_refs));
                    }
                    nodes.push(memory_to_node(memory, params.include_embeddings));
                }
            }
        }
    }

    // --- Build metadata ---
    let mut node_counts_by_type: HashMap<String, usize> = HashMap::new();
    for node in &nodes {
        *node_counts_by_type
            .entry(node.node_type.clone())
            .or_insert(0) += 1;
    }
    let mut edge_counts_by_type: HashMap<String, usize> = HashMap::new();
    for edge in &edges {
        *edge_counts_by_type
            .entry(edge.edge_type.clone())
            .or_insert(0) += 1;
    }

    let metadata = ExportMetadata {
        exported_at: chrono::Utc::now(),
        user_id,
        node_count: nodes.len(),
        edge_count: edges.len(),
        node_counts_by_type,
        edge_counts_by_type,
    };

    let response = GraphExportResponse {
        metadata,
        nodes,
        edges,
    };

    match params.format.as_str() {
        "gexf" => {
            let gexf = to_gexf(&response);
            Ok((
                [(axum::http::header::CONTENT_TYPE, "application/gexf+xml")],
                gexf,
            )
                .into_response())
        }
        _ => Ok(axum::response::Json(response).into_response()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_memory::{
        EdgeTier, EntityLabel, EntityNode, EpisodeSource, EpisodicNode, LtpStatus, RelationType,
        RelationshipEdge,
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
            selectivity: None,
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
        assert!(
            (salience - 0.8_f64).abs() < 1e-5,
            "salience {salience} not ~0.8"
        );
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
        entity
            .attributes
            .insert("role".to_owned(), "mascot".to_owned());

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
            endpoint_selectivity: None,
            forman_curvature: None,
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
        assert!(
            (strength - 0.75_f64).abs() < 1e-5,
            "strength {strength} not ~0.75"
        );
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
            EntityRef {
                entity_id: entity_a,
                name: "Alice".to_owned(),
                relation: "subject".to_owned(),
            },
            EntityRef {
                entity_id: entity_b,
                name: "Rust".to_owned(),
                relation: "mentioned".to_owned(),
            },
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

    #[test]
    fn test_xml_escape_strips_control_chars_and_escapes_specials() {
        assert_eq!(xml_escape("hello\u{0001}world\u{0008}end"), "helloworldend");
        assert_eq!(xml_escape("a\tb\nc\rd"), "a\tb\nc\rd");
        assert_eq!(
            xml_escape("<tag attr=\"v&w\">it's</tag>"),
            "&lt;tag attr=&quot;v&amp;w&quot;&gt;it&apos;s&lt;/tag&gt;"
        );
    }
}

#[cfg(test)]
mod integration_tests {
    use crate::handlers::test_helpers::{self, TestHarness};
    use axum::http::StatusCode;
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_export_empty_graph() {
        let harness = TestHarness::new();
        let app = harness.router();

        let req = test_helpers::get("/api/graph/test-user/export");
        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 10_000_000)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["metadata"]["node_count"], 0);
        assert_eq!(json["metadata"]["edge_count"], 0);
        assert_eq!(json["metadata"]["user_id"], "test-user");
        assert!(json["nodes"].as_array().unwrap().is_empty());
        assert!(json["edges"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_export_with_data() {
        let harness = TestHarness::new();

        // Store a memory
        {
            let mem_sys = harness.manager.get_user_memory("test-user").unwrap();
            let mem_guard = mem_sys.read();
            let experience = crate::memory::Experience {
                content: "Rust is a systems programming language".to_string(),
                ..Default::default()
            };
            mem_guard.remember(experience, None).unwrap();
        }

        let app = harness.router();
        let req = test_helpers::get("/api/graph/test-user/export");
        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 10_000_000)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // At least one memory node should exist
        let nodes = json["nodes"].as_array().unwrap();
        let memory_nodes: Vec<_> = nodes.iter().filter(|n| n["type"] == "memory").collect();
        assert!(
            !memory_nodes.is_empty(),
            "expected at least one memory node, got none"
        );
        assert_eq!(
            memory_nodes[0]["attributes"]["content"],
            "Rust is a systems programming language"
        );
    }

    #[tokio::test]
    async fn test_export_min_importance_filter() {
        let harness = TestHarness::new();

        // Store a memory (default importance is typically low)
        {
            let mem_sys = harness.manager.get_user_memory("test-user").unwrap();
            let mem_guard = mem_sys.read();
            let experience = crate::memory::Experience {
                content: "Ephemeral test content".to_string(),
                ..Default::default()
            };
            mem_guard.remember(experience, None).unwrap();
        }

        let app = harness.router();
        // Request with very high min_importance filter
        let req = test_helpers::get("/api/graph/test-user/export?min_importance=0.99");
        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 10_000_000)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Memory should be filtered out by the high importance threshold
        let nodes = json["nodes"].as_array().unwrap();
        let memory_nodes: Vec<_> = nodes.iter().filter(|n| n["type"] == "memory").collect();
        assert!(
            memory_nodes.is_empty(),
            "expected no memory nodes with min_importance=0.99, got {}",
            memory_nodes.len()
        );
    }

    #[tokio::test]
    async fn test_export_gexf_format() {
        let harness = TestHarness::new();

        // Store a memory
        {
            let mem_sys = harness.manager.get_user_memory("test-user").unwrap();
            let mem_guard = mem_sys.read();
            let experience = crate::memory::Experience {
                content: "GEXF format test memory".to_string(),
                ..Default::default()
            };
            mem_guard.remember(experience, None).unwrap();
        }

        let app = harness.router();
        let req = test_helpers::get("/api/graph/test-user/export?format=gexf");
        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        // Check Content-Type header
        let content_type = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .expect("missing content-type header")
            .to_str()
            .unwrap();
        assert_eq!(content_type, "application/gexf+xml");

        let body = axum::body::to_bytes(resp.into_body(), 10_000_000)
            .await
            .unwrap();
        let text = std::str::from_utf8(&body).expect("body is not valid UTF-8");

        assert!(text.contains("<?xml"), "missing XML declaration");
        assert!(text.contains("<gexf"), "missing <gexf> element");
        assert!(text.contains("<nodes>"), "missing <nodes> element");
        assert!(text.contains("<edges>"), "missing <edges> element");
        assert!(
            text.contains("GEXF format test memory"),
            "memory content not present in GEXF output"
        );
    }

    #[tokio::test]
    async fn test_export_include_filter() {
        let harness = TestHarness::new();

        // Store a memory
        {
            let mem_sys = harness.manager.get_user_memory("test-user").unwrap();
            let mem_guard = mem_sys.read();
            let experience = crate::memory::Experience {
                content: "Include filter test".to_string(),
                ..Default::default()
            };
            mem_guard.remember(experience, None).unwrap();
        }

        let app = harness.router();
        // Request only entities (no memories, no episodes)
        let req = test_helpers::get("/api/graph/test-user/export?include=entities");
        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 10_000_000)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // No memory nodes should appear since we only requested entities
        let nodes = json["nodes"].as_array().unwrap();
        let memory_nodes: Vec<_> = nodes.iter().filter(|n| n["type"] == "memory").collect();
        assert!(
            memory_nodes.is_empty(),
            "expected no memory nodes when include=entities, got {}",
            memory_nodes.len()
        );
    }
}
