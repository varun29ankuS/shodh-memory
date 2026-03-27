# Graph Snapshot Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `GET /api/graph/{user_id}/export` endpoint that dumps the full knowledge graph (entities, memories, episodes, all edges) as graph-native JSON or GEXF, with a CLI wrapper.

**Architecture:** New handler module `src/handlers/export.rs` with export-specific types, node/edge builders, and GEXF serializer. New `get_all_episodes()` method on GraphMemory. CLI subcommand calls the API via reqwest.

**Tech Stack:** Rust, axum (existing), serde_json (existing), reqwest (existing, for CLI), clap (existing). No new dependencies — GEXF is generated via `std::fmt::Write` with XML escaping (schema is fixed, no need for a full XML library).

**Spec:** `docs/specs/2026-03-26-graph-snapshot-export-design.md`

---

### Task 1: Add `get_all_episodes()` to GraphMemory

`get_all_entities()` and `get_all_relationships()` exist but there's no `get_all_episodes()`. We need it for the bulk export.

**Files:**
- Modify: `src/graph_memory.rs:4460` (after `get_all_relationships`)
- Test: `src/graph_memory.rs` (inline test module)

- [ ] **Step 1: Write the failing test**

Add to the existing test module in `src/graph_memory.rs`:

```rust
#[test]
fn test_get_all_episodes() {
    let temp_dir = tempfile::tempdir().unwrap();
    let graph = GraphMemory::new(temp_dir.path()).unwrap();

    // Empty graph should return empty vec
    let episodes = graph.get_all_episodes().unwrap();
    assert!(episodes.is_empty());

    // Add an episode and verify it's returned
    let entity = graph
        .find_or_create_entity("TestEntity", &[EntityLabel::Concept])
        .unwrap();
    let episode = graph
        .add_episode(
            "Test Episode",
            "Test content",
            Utc::now(),
            vec![entity.uuid],
            EpisodeSource::Event,
            HashMap::new(),
        )
        .unwrap();

    let episodes = graph.get_all_episodes().unwrap();
    assert_eq!(episodes.len(), 1);
    assert_eq!(episodes[0].uuid, episode.uuid);
    assert_eq!(episodes[0].name, "Test Episode");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test test_get_all_episodes -- --nocapture 2>&1 | tail -20`
Expected: FAIL — `get_all_episodes` method not found

- [ ] **Step 3: Implement `get_all_episodes`**

Add after `get_all_relationships()` (around line 4461) in `src/graph_memory.rs`:

```rust
/// Get all episodes in the graph
pub fn get_all_episodes(&self) -> Result<Vec<EpisodicNode>> {
    let mut episodes = Vec::new();

    let mut read_opts = rocksdb::ReadOptions::default();
    read_opts.fill_cache(false);
    let iter =
        self.db
            .iterator_cf_opt(self.episodes_cf(), read_opts, rocksdb::IteratorMode::Start);
    for (_, value) in iter.flatten() {
        if let Ok(episode) = bincode::serde::decode_from_slice::<EpisodicNode, _>(
            &value,
            bincode::config::standard(),
        )
        .map(|(v, _)| v)
        {
            episodes.push(episode);
        }
    }

    // Sort by creation time (newest first)
    episodes.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    Ok(episodes)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test test_get_all_episodes -- --nocapture 2>&1 | tail -20`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/graph_memory.rs
git commit -m "feat: add get_all_episodes() for bulk graph export"
```

---

### Task 2: Export types and node builders

Define the export response types and functions that convert internal types (EntityNode, Memory, EpisodicNode) to the graph-native export format.

**Files:**
- Create: `src/handlers/export.rs`
- Modify: `src/handlers/mod.rs` (add `pub mod export;`)

- [ ] **Step 1: Write the failing test**

Create `src/handlers/export.rs` with types and test:

```rust
//! Graph Export Handler
//!
//! Exports the full knowledge graph as graph-native JSON or GEXF.
//! All node types (entities, memories, episodes) become nodes with a `type` field.
//! All connections become edges with a `type` field.

use chrono::{DateTime, Utc};
use serde::Serialize;
use std::collections::HashMap;

// =============================================================================
// EXPORT TYPES
// =============================================================================

#[derive(Debug, Serialize)]
pub struct GraphExportResponse {
    pub metadata: ExportMetadata,
    pub nodes: Vec<ExportNode>,
    pub edges: Vec<ExportEdge>,
}

#[derive(Debug, Serialize)]
pub struct ExportMetadata {
    pub exported_at: DateTime<Utc>,
    pub user_id: String,
    pub node_count: usize,
    pub edge_count: usize,
    pub node_counts_by_type: HashMap<String, usize>,
    pub edge_counts_by_type: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExportNode {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
    pub label: String,
    pub attributes: serde_json::Value,
}

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

// =============================================================================
// NODE BUILDERS
// =============================================================================

/// Convert an EntityNode to an ExportNode
pub fn entity_to_node(
    entity: &crate::graph_memory::EntityNode,
    include_embeddings: bool,
) -> ExportNode {
    let mut attrs = serde_json::json!({
        "salience": entity.salience,
        "mention_count": entity.mention_count,
        "is_proper_noun": entity.is_proper_noun,
        "labels": entity.labels,
        "created_at": entity.created_at,
        "last_seen_at": entity.last_seen_at,
        "summary": entity.summary,
    });

    if !entity.attributes.is_empty() {
        attrs["entity_attributes"] = serde_json::to_value(&entity.attributes).unwrap_or_default();
    }

    if include_embeddings {
        if let Some(ref emb) = entity.name_embedding {
            attrs["name_embedding"] = serde_json::to_value(emb).unwrap_or_default();
        }
    }

    ExportNode {
        id: entity.uuid.to_string(),
        node_type: "entity".to_string(),
        label: entity.name.clone(),
        attributes: attrs,
    }
}

/// Convert a Memory to an ExportNode
pub fn memory_to_node(
    memory: &crate::memory::Memory,
    include_embeddings: bool,
) -> ExportNode {
    let content = &memory.experience.content;
    let label = if content.len() > 100 {
        format!("{}...", &content[..content.floor_char_boundary(97)])
    } else {
        content.clone()
    };

    let mut attrs = serde_json::json!({
        "content": content,
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
            attrs["embedding"] = serde_json::to_value(emb).unwrap_or_default();
        }
    }

    ExportNode {
        id: memory.id.to_string(),
        node_type: "memory".to_string(),
        label,
        attributes: attrs,
    }
}

/// Convert an EpisodicNode to an ExportNode
pub fn episode_to_node(episode: &crate::graph_memory::EpisodicNode) -> ExportNode {
    let attrs = serde_json::json!({
        "content": episode.content,
        "source": format!("{:?}", episode.source),
        "valid_at": episode.valid_at,
        "created_at": episode.created_at,
    });

    ExportNode {
        id: episode.uuid.to_string(),
        node_type: "episode".to_string(),
        label: episode.name.clone(),
        attributes: attrs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_memory::{EntityLabel, EntityNode, EpisodicNode, EpisodeSource};
    use chrono::Utc;
    use std::collections::HashMap;
    use uuid::Uuid;

    fn sample_entity() -> EntityNode {
        EntityNode {
            uuid: Uuid::new_v4(),
            name: "TestEntity".to_string(),
            labels: vec![EntityLabel::Technology],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 5,
            summary: "A test entity".to_string(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.7,
            is_proper_noun: false,
        }
    }

    #[test]
    fn test_entity_to_node() {
        let entity = sample_entity();
        let node = entity_to_node(&entity, false);

        assert_eq!(node.node_type, "entity");
        assert_eq!(node.label, "TestEntity");
        assert_eq!(node.id, entity.uuid.to_string());
        assert_eq!(node.attributes["salience"], 0.7);
        assert_eq!(node.attributes["mention_count"], 5);
        // Embeddings excluded by default
        assert!(node.attributes.get("name_embedding").is_none());
    }

    #[test]
    fn test_entity_to_node_with_embeddings() {
        let mut entity = sample_entity();
        entity.name_embedding = Some(vec![0.1, 0.2, 0.3]);
        let node = entity_to_node(&entity, true);

        assert!(node.attributes.get("name_embedding").is_some());
    }

    #[test]
    fn test_episode_to_node() {
        let episode = EpisodicNode {
            uuid: Uuid::new_v4(),
            name: "Run 32".to_string(),
            content: "Discovered cloaking upgrade".to_string(),
            valid_at: Utc::now(),
            created_at: Utc::now(),
            entity_refs: vec![],
            source: EpisodeSource::Event,
            metadata: HashMap::new(),
        };
        let node = episode_to_node(&episode);

        assert_eq!(node.node_type, "episode");
        assert_eq!(node.label, "Run 32");
        assert_eq!(node.attributes["source"], "Event");
    }
}
```

- [ ] **Step 2: Register the module**

Add to `src/handlers/mod.rs`, in the "Knowledge graph" section:

```rust
// Knowledge graph
pub mod graph;
pub mod export;
pub mod visualization;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test handlers::export::tests -- --nocapture 2>&1 | tail -20`
Expected: All 3 tests PASS

Note: `memory_to_node` is not tested here because constructing a `Memory` requires internal machinery. It will be covered by the integration test in Task 5.

- [ ] **Step 4: Commit**

```bash
git add src/handlers/export.rs src/handlers/mod.rs
git commit -m "feat: add graph export types and node builders"
```

---

### Task 3: Edge builders

Convert RelationshipEdge to ExportEdge and synthesize entity_ref edges from Memory/Episode entity references.

**Files:**
- Modify: `src/handlers/export.rs` (add edge builder functions and tests)

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/handlers/export.rs`:

```rust
#[test]
fn test_relationship_to_edge() {
    use crate::graph_memory::{RelationType, LtpStatus, EdgeTier};
    use std::collections::VecDeque;

    let edge = crate::graph_memory::RelationshipEdge {
        uuid: Uuid::new_v4(),
        from_entity: Uuid::new_v4(),
        to_entity: Uuid::new_v4(),
        relation_type: RelationType::RelatedTo,
        strength: 0.85,
        created_at: Utc::now(),
        valid_at: Utc::now(),
        invalidated_at: None,
        source_episode_id: None,
        context: "test context".to_string(),
        last_activated: Utc::now(),
        activation_count: 14,
        ltp_status: LtpStatus::Full,
        tier: EdgeTier::L3Semantic,
        activation_timestamps: Some(VecDeque::new()),
        entity_confidence: Some(0.9),
    };
    let export_edge = relationship_to_edge(&edge);

    assert_eq!(export_edge.edge_type, "relationship");
    assert_eq!(export_edge.label.as_deref(), Some("RelatedTo"));
    assert_eq!(export_edge.source, edge.from_entity.to_string());
    assert_eq!(export_edge.target, edge.to_entity.to_string());
    assert_eq!(export_edge.attributes["strength"], 0.85);
    assert_eq!(export_edge.attributes["ltp_status"], "Full");
    assert_eq!(export_edge.attributes["tier"], "L3Semantic");
    assert_eq!(export_edge.attributes["activation_count"], 14);
}

#[test]
fn test_entity_ref_edges() {
    use crate::memory::EntityRef;

    let memory_id = Uuid::new_v4();
    let entity_id = Uuid::new_v4();
    let refs = vec![EntityRef {
        entity_id,
        name: "SomeEntity".to_string(),
        relation: "mentioned".to_string(),
    }];

    let edges = entity_refs_to_edges(&memory_id, &refs);
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].edge_type, "entity_ref");
    assert_eq!(edges[0].source, memory_id.to_string());
    assert_eq!(edges[0].target, entity_id.to_string());
    assert_eq!(edges[0].attributes["relation"], "mentioned");
}

#[test]
fn test_episode_entity_ref_edges() {
    let episode_id = Uuid::new_v4();
    let entity_ids = vec![Uuid::new_v4(), Uuid::new_v4()];

    let edges = episode_refs_to_edges(&episode_id, &entity_ids);
    assert_eq!(edges.len(), 2);
    assert_eq!(edges[0].edge_type, "entity_ref");
    assert_eq!(edges[0].source, episode_id.to_string());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test handlers::export::tests -- --nocapture 2>&1 | tail -20`
Expected: FAIL — `relationship_to_edge`, `entity_refs_to_edges`, `episode_refs_to_edges` not found

- [ ] **Step 3: Implement edge builders**

Add to `src/handlers/export.rs`, after the node builders:

```rust
// =============================================================================
// EDGE BUILDERS
// =============================================================================

/// Convert a RelationshipEdge to an ExportEdge
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
        edge_type: "relationship".to_string(),
        label: Some(format!("{:?}", edge.relation_type)),
        attributes: attrs,
    }
}

/// Synthesize entity_ref edges from a Memory's entity_refs
pub fn entity_refs_to_edges(
    source_id: &uuid::Uuid,
    refs: &[crate::memory::EntityRef],
) -> Vec<ExportEdge> {
    refs.iter()
        .map(|r| ExportEdge {
            id: format!("{}-{}", source_id, r.entity_id),
            source: source_id.to_string(),
            target: r.entity_id.to_string(),
            edge_type: "entity_ref".to_string(),
            label: None,
            attributes: serde_json::json!({
                "relation": r.relation,
            }),
        })
        .collect()
}

/// Synthesize entity_ref edges from an EpisodicNode's entity_refs (UUID-only)
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
            edge_type: "entity_ref".to_string(),
            label: None,
            attributes: serde_json::json!({
                "relation": "referenced",
            }),
        })
        .collect()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test handlers::export::tests -- --nocapture 2>&1 | tail -20`
Expected: All 6 tests PASS (3 node + 3 edge)

- [ ] **Step 5: Commit**

```bash
git add src/handlers/export.rs
git commit -m "feat: add graph export edge builders"
```

---

### Task 4: JSON export handler and route registration

The core handler: reads all RocksDB column families, builds nodes and edges, applies filters, returns JSON.

**Files:**
- Modify: `src/handlers/export.rs` (add handler function)
- Modify: `src/handlers/router.rs:200` (add route)

- [ ] **Step 1: Write the integration test**

Add to `src/handlers/export.rs` at the bottom:

```rust
#[cfg(test)]
mod integration_tests {
    use crate::handlers::test_helpers::{self, TestHarness};
    use axum::body::Body;
    use http::StatusCode;
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_export_empty_graph() {
        let harness = TestHarness::new();
        let app = harness.router();

        let resp = app
            .oneshot(test_helpers::get("/api/graph/test-user/export"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 10_000_000).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["metadata"]["node_count"], 0);
        assert_eq!(json["metadata"]["edge_count"], 0);
        assert!(json["nodes"].as_array().unwrap().is_empty());
        assert!(json["edges"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_export_with_data() {
        let harness = TestHarness::new();

        // Store a memory so there's data to export
        {
            let mem_sys = harness.manager.get_user_memory("test-user").unwrap();
            let mut mem_guard = mem_sys.write();
            let experience = crate::memory::Experience {
                content: "Test memory content for export".to_string(),
                ..Default::default()
            };
            mem_guard.record(experience, None).unwrap();
        }

        let app = harness.router();
        let resp = app
            .oneshot(test_helpers::get("/api/graph/test-user/export"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 10_000_000).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Should have at least the memory node
        assert!(json["metadata"]["node_count"].as_u64().unwrap() >= 1);

        // Verify node structure
        let nodes = json["nodes"].as_array().unwrap();
        let memory_nodes: Vec<_> = nodes.iter().filter(|n| n["type"] == "memory").collect();
        assert!(!memory_nodes.is_empty());
        assert!(memory_nodes[0]["attributes"]["content"]
            .as_str()
            .unwrap()
            .contains("Test memory content"));
    }

    #[tokio::test]
    async fn test_export_min_importance_filter() {
        let harness = TestHarness::new();

        // Store a memory
        {
            let mem_sys = harness.manager.get_user_memory("test-user").unwrap();
            let mut mem_guard = mem_sys.write();
            let experience = crate::memory::Experience {
                content: "Low importance test memory".to_string(),
                ..Default::default()
            };
            mem_guard.record(experience, None).unwrap();
        }

        let app = harness.router();

        // With very high min_importance, should filter out the memory
        let resp = app
            .oneshot(test_helpers::get(
                "/api/graph/test-user/export?min_importance=0.99",
            ))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 10_000_000).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let nodes = json["nodes"].as_array().unwrap();
        let memory_nodes: Vec<_> = nodes.iter().filter(|n| n["type"] == "memory").collect();
        assert!(memory_nodes.is_empty(), "High min_importance should filter out memories");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test handlers::export::integration_tests -- --nocapture 2>&1 | tail -20`
Expected: FAIL — handler function doesn't exist, route not registered

- [ ] **Step 3: Add query params type and handler**

Add to `src/handlers/export.rs`, after the edge builders:

```rust
use axum::{
    extract::{Path, Query, State},
    response::Json,
};
use std::sync::Arc;

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::validation;

type AppState = Arc<MultiUserMemoryManager>;

/// Query parameters for graph export
#[derive(Debug, serde::Deserialize)]
pub struct ExportParams {
    /// Output format: "json" or "gexf"
    #[serde(default = "default_format")]
    pub format: String,
    /// Comma-separated node types to include: "entities,memories,episodes"
    #[serde(default = "default_include")]
    pub include: String,
    /// Minimum importance for memory nodes (0.0 = all)
    #[serde(default)]
    pub min_importance: f32,
    /// Include embedding vectors in output
    #[serde(default)]
    pub include_embeddings: bool,
}

fn default_format() -> String {
    "json".to_string()
}
fn default_include() -> String {
    "entities,memories,episodes".to_string()
}

/// Parse the `include` param into a set of included types
fn parse_include(include: &str) -> (bool, bool, bool) {
    let parts: Vec<&str> = include.split(',').map(|s| s.trim()).collect();
    (
        parts.contains(&"entities"),
        parts.contains(&"memories"),
        parts.contains(&"episodes"),
    )
}

/// GET /api/graph/{user_id}/export
pub async fn export_graph(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(params): Query<ExportParams>,
) -> Result<Json<GraphExportResponse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let (inc_entities, inc_memories, inc_episodes) = parse_include(&params.include);

    let mut nodes: Vec<ExportNode> = Vec::new();
    let mut edges: Vec<ExportEdge> = Vec::new();

    // --- Entities and Relationships from GraphMemory ---
    if inc_entities || inc_episodes {
        if let Ok(graph) = state.get_user_graph(&user_id) {
            let graph_guard = graph.read();

            if inc_entities {
                let entities = graph_guard
                    .get_all_entities()
                    .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;
                nodes.extend(
                    entities.iter().map(|e| entity_to_node(e, params.include_embeddings)),
                );

                // Relationship edges (entity↔entity)
                let relationships = graph_guard
                    .get_all_relationships()
                    .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;
                edges.extend(relationships.iter().map(relationship_to_edge));
            }

            if inc_episodes {
                let episodes = graph_guard
                    .get_all_episodes()
                    .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

                // Episode → entity ref edges
                if inc_entities {
                    for ep in &episodes {
                        edges.extend(episode_refs_to_edges(&ep.uuid, &ep.entity_refs));
                    }
                }

                nodes.extend(episodes.iter().map(episode_to_node));
            }
        }
    }

    // --- Memories from MemorySystem ---
    if inc_memories {
        if let Ok(mem_sys) = state.get_user_memory(&user_id) {
            let mem_guard = mem_sys.read();
            let memories = mem_guard
                .get_all_memories()
                .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

            for mem in &memories {
                // Apply importance filter
                if mem.importance() < params.min_importance {
                    continue;
                }

                // Memory → entity ref edges
                if inc_entities {
                    edges.extend(entity_refs_to_edges(&mem.id.0, &mem.entity_refs));
                }

                nodes.push(memory_to_node(mem, params.include_embeddings));
            }
        }
    }

    // --- Build metadata ---
    let mut node_counts: HashMap<String, usize> = HashMap::new();
    for node in &nodes {
        *node_counts.entry(node.node_type.clone()).or_default() += 1;
    }
    let mut edge_counts: HashMap<String, usize> = HashMap::new();
    for edge in &edges {
        *edge_counts.entry(edge.edge_type.clone()).or_default() += 1;
    }

    let response = GraphExportResponse {
        metadata: ExportMetadata {
            exported_at: Utc::now(),
            user_id,
            node_count: nodes.len(),
            edge_count: edges.len(),
            node_counts_by_type: node_counts,
            edge_counts_by_type: edge_counts,
        },
        nodes,
        edges,
    };

    Ok(Json(response))
}
```

Note: The `memory.id.0` access assumes `MemoryId` is a newtype around `Uuid`. Check the actual type — if it's `MemoryId(Uuid)` use `.0`, if it has a method use that. Adjust based on what the compiler says.

- [ ] **Step 4: Register the route**

In `src/handlers/router.rs`, add after the existing graph routes (around line 220):

```rust
        .route("/api/graph/{user_id}/export", get(export::export_graph))
```

And add `export` to the module imports at the top of `router.rs` if not already accessible via `super::`.

- [ ] **Step 5: Run integration tests**

Run: `cargo test handlers::export::integration_tests -- --nocapture 2>&1 | tail -30`
Expected: All 3 integration tests PASS

- [ ] **Step 6: Compile check**

Run: `cargo check 2>&1 | tail -20`
Expected: No errors. Fix any type mismatches (especially around MemoryId access).

- [ ] **Step 7: Commit**

```bash
git add src/handlers/export.rs src/handlers/router.rs
git commit -m "feat: add GET /api/graph/{user_id}/export JSON endpoint"
```

---

### Task 5: GEXF output format

Add `quick-xml` dependency and GEXF serialization when `?format=gexf` is requested.

**Files:**
- Modify: `Cargo.toml` (add quick-xml)
- Modify: `src/handlers/export.rs` (add GEXF serializer, modify handler for format dispatch)

- [ ] **Step 1: Write the GEXF test**

Add to `integration_tests` module in `src/handlers/export.rs`:

```rust
#[tokio::test]
async fn test_export_gexf_format() {
    let harness = TestHarness::new();

    // Store a memory
    {
        let mem_sys = harness.manager.get_user_memory("test-user").unwrap();
        let mut mem_guard = mem_sys.write();
        let experience = crate::memory::Experience {
            content: "GEXF test memory".to_string(),
            ..Default::default()
        };
        mem_guard.record(experience, None).unwrap();
    }

    let app = harness.router();
    let resp = app
        .oneshot(test_helpers::get("/api/graph/test-user/export?format=gexf"))
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "application/gexf+xml"
    );

    let body = axum::body::to_bytes(resp.into_body(), 10_000_000).await.unwrap();
    let xml = String::from_utf8(body.to_vec()).unwrap();

    assert!(xml.contains("<?xml"));
    assert!(xml.contains("<gexf"));
    assert!(xml.contains("<nodes>"));
    assert!(xml.contains("<edges>"));
    assert!(xml.contains("GEXF test memory"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test test_export_gexf_format -- --nocapture 2>&1 | tail -20`
Expected: FAIL — GEXF format not handled

- [ ] **Step 3: Implement GEXF serializer**

Add to `src/handlers/export.rs`:

```rust
use axum::response::{IntoResponse, Response};

// =============================================================================
// GEXF SERIALIZATION
// =============================================================================

/// Serialize a GraphExportResponse as GEXF XML
pub fn to_gexf(export: &GraphExportResponse) -> String {
    use std::fmt::Write;

    let mut xml = String::with_capacity(4096);

    // XML declaration and GEXF root
    writeln!(xml, r#"<?xml version="1.0" encoding="UTF-8"?>"#).unwrap();
    writeln!(
        xml,
        r#"<gexf xmlns="http://gexf.net/1.3" version="1.3">"#
    )
    .unwrap();
    writeln!(
        xml,
        r#"  <meta lastmodifieddate="{}">"#,
        export.metadata.exported_at.format("%Y-%m-%d")
    )
    .unwrap();
    writeln!(xml, r#"    <creator>shodh-memory</creator>"#).unwrap();
    writeln!(xml, r#"  </meta>"#).unwrap();
    writeln!(
        xml,
        r#"  <graph defaultedgetype="directed" mode="static">"#
    )
    .unwrap();

    // Node attribute declarations
    writeln!(xml, r#"    <attributes class="node">"#).unwrap();
    writeln!(xml, r#"      <attribute id="0" title="type" type="string"/>"#).unwrap();
    writeln!(xml, r#"      <attribute id="1" title="importance" type="float"/>"#).unwrap();
    writeln!(xml, r#"      <attribute id="2" title="salience" type="float"/>"#).unwrap();
    writeln!(xml, r#"      <attribute id="3" title="tier" type="string"/>"#).unwrap();
    writeln!(xml, r#"      <attribute id="4" title="access_count" type="integer"/>"#).unwrap();
    writeln!(xml, r#"      <attribute id="5" title="activation" type="float"/>"#).unwrap();
    writeln!(xml, r#"      <attribute id="6" title="mention_count" type="integer"/>"#).unwrap();
    writeln!(xml, r#"      <attribute id="7" title="experience_type" type="string"/>"#).unwrap();
    writeln!(xml, r#"    </attributes>"#).unwrap();

    // Edge attribute declarations
    writeln!(xml, r#"    <attributes class="edge">"#).unwrap();
    writeln!(xml, r#"      <attribute id="0" title="type" type="string"/>"#).unwrap();
    writeln!(xml, r#"      <attribute id="1" title="ltp_status" type="string"/>"#).unwrap();
    writeln!(xml, r#"      <attribute id="2" title="tier" type="string"/>"#).unwrap();
    writeln!(xml, r#"      <attribute id="3" title="activation_count" type="integer"/>"#).unwrap();
    writeln!(xml, r#"    </attributes>"#).unwrap();

    // Nodes
    writeln!(xml, r#"    <nodes>"#).unwrap();
    for node in &export.nodes {
        writeln!(
            xml,
            r#"      <node id="{}" label="{}">"#,
            xml_escape(&node.id),
            xml_escape(&node.label)
        )
        .unwrap();
        writeln!(xml, r#"        <attvalues>"#).unwrap();
        writeln!(
            xml,
            r#"          <attvalue for="0" value="{}"/>"#,
            xml_escape(&node.node_type)
        )
        .unwrap();

        // Type-specific attributes
        if let Some(v) = node.attributes.get("importance") {
            if let Some(f) = v.as_f64() {
                writeln!(xml, r#"          <attvalue for="1" value="{f:.4}"/>"#).unwrap();
            }
        }
        if let Some(v) = node.attributes.get("salience") {
            if let Some(f) = v.as_f64() {
                writeln!(xml, r#"          <attvalue for="2" value="{f:.4}"/>"#).unwrap();
            }
        }
        if let Some(v) = node.attributes.get("tier") {
            if let Some(s) = v.as_str() {
                writeln!(xml, r#"          <attvalue for="3" value="{}"/>"#, xml_escape(s))
                    .unwrap();
            }
        }
        if let Some(v) = node.attributes.get("access_count") {
            if let Some(n) = v.as_u64() {
                writeln!(xml, r#"          <attvalue for="4" value="{n}"/>"#).unwrap();
            }
        }
        if let Some(v) = node.attributes.get("activation") {
            if let Some(f) = v.as_f64() {
                writeln!(xml, r#"          <attvalue for="5" value="{f:.4}"/>"#).unwrap();
            }
        }
        if let Some(v) = node.attributes.get("mention_count") {
            if let Some(n) = v.as_u64() {
                writeln!(xml, r#"          <attvalue for="6" value="{n}"/>"#).unwrap();
            }
        }
        if let Some(v) = node.attributes.get("experience_type") {
            if let Some(s) = v.as_str() {
                writeln!(xml, r#"          <attvalue for="7" value="{}"/>"#, xml_escape(s))
                    .unwrap();
            }
        }

        writeln!(xml, r#"        </attvalues>"#).unwrap();
        writeln!(xml, r#"      </node>"#).unwrap();
    }
    writeln!(xml, r#"    </nodes>"#).unwrap();

    // Edges
    writeln!(xml, r#"    <edges>"#).unwrap();
    for edge in &export.edges {
        let weight = edge
            .attributes
            .get("strength")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        writeln!(
            xml,
            r#"      <edge id="{}" source="{}" target="{}" weight="{weight:.4}"{}>"#,
            xml_escape(&edge.id),
            xml_escape(&edge.source),
            xml_escape(&edge.target),
            edge.label
                .as_ref()
                .map(|l| format!(r#" label="{}""#, xml_escape(l)))
                .unwrap_or_default(),
        )
        .unwrap();
        writeln!(xml, r#"        <attvalues>"#).unwrap();
        writeln!(
            xml,
            r#"          <attvalue for="0" value="{}"/>"#,
            xml_escape(&edge.edge_type)
        )
        .unwrap();
        if let Some(v) = edge.attributes.get("ltp_status") {
            if let Some(s) = v.as_str() {
                writeln!(xml, r#"          <attvalue for="1" value="{}"/>"#, xml_escape(s))
                    .unwrap();
            }
        }
        if let Some(v) = edge.attributes.get("tier") {
            if let Some(s) = v.as_str() {
                writeln!(xml, r#"          <attvalue for="2" value="{}"/>"#, xml_escape(s))
                    .unwrap();
            }
        }
        if let Some(v) = edge.attributes.get("activation_count") {
            if let Some(n) = v.as_u64() {
                writeln!(xml, r#"          <attvalue for="3" value="{n}"/>"#).unwrap();
            }
        }
        writeln!(xml, r#"        </attvalues>"#).unwrap();
        writeln!(xml, r#"      </edge>"#).unwrap();
    }
    writeln!(xml, r#"    </edges>"#).unwrap();

    writeln!(xml, r#"  </graph>"#).unwrap();
    writeln!(xml, r#"</gexf>"#).unwrap();

    xml
}

/// Escape XML special characters
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
```

Note: This uses `std::fmt::Write` for string building rather than `quick-xml`, which is simpler for a fixed schema like GEXF. If the team prefers using quick-xml's writer API for stricter correctness, that's a reasonable alternative — but `xml_escape` + `writeln!` is sufficient for GEXF since we control the schema.

- [ ] **Step 4: Modify handler for format dispatch**

Update the `export_graph` handler signature to return either JSON or GEXF. Replace the return type and the end of the function:

Change the handler signature from:
```rust
pub async fn export_graph(...) -> Result<Json<GraphExportResponse>, AppError> {
```
to:
```rust
pub async fn export_graph(...) -> Result<Response, AppError> {
```

And replace the final `Ok(Json(response))` with:

```rust
    match params.format.as_str() {
        "gexf" => {
            let gexf = to_gexf(&response);
            Ok((
                [(axum::http::header::CONTENT_TYPE, "application/gexf+xml")],
                gexf,
            )
                .into_response())
        }
        _ => Ok(Json(response).into_response()),
    }
```

- [ ] **Step 5: Run all export tests**

Run: `cargo test handlers::export -- --nocapture 2>&1 | tail -30`
Expected: All tests PASS (unit + integration)

- [ ] **Step 6: Commit**

```bash
git add src/handlers/export.rs
git commit -m "feat: add GEXF output format for Gephi import"
```

---

### Task 6: CLI `export-graph` subcommand

Thin HTTP client that calls the export API endpoint and writes to stdout or file.

**Files:**
- Modify: `src/cli.rs` (add ExportGraph command variant and handler)

- [ ] **Step 1: Add the command variant**

In `src/cli.rs`, add to the `Commands` enum (after `Version`):

```rust
    /// Export knowledge graph as JSON or GEXF
    ExportGraph {
        /// User ID whose graph to export
        user_id: String,

        /// Output format: json or gexf
        #[arg(long, default_value = "json")]
        format: String,

        /// Node types to include (comma-separated: entities,memories,episodes)
        #[arg(long, default_value = "entities,memories,episodes")]
        include: String,

        /// Minimum importance threshold for memory nodes
        #[arg(long, default_value_t = 0.0)]
        min_importance: f32,

        /// Include embedding vectors
        #[arg(long)]
        include_embeddings: bool,

        /// Output file (stdout if omitted)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// API URL for the memory server
        #[arg(long, env = "SHODH_API_URL", default_value = "http://127.0.0.1:3030")]
        api_url: String,

        /// API key for authentication
        #[arg(
            long,
            env = "SHODH_API_KEY",
            default_value = "sk-shodh-dev-local-testing-key"
        )]
        api_key: String,
    },
```

- [ ] **Step 2: Add the command handler**

In the `match cli.command { ... }` block in `main()`, add:

```rust
        Commands::ExportGraph {
            user_id,
            format,
            include,
            min_importance,
            include_embeddings,
            output,
            api_url,
            api_key,
        } => {
            let url = format!(
                "{}/api/graph/{}/export?format={}&include={}&min_importance={}&include_embeddings={}",
                api_url, user_id, format, include, min_importance, include_embeddings
            );

            let client = reqwest::blocking::Client::new();
            let resp = client
                .get(&url)
                .header("x-api-key", &api_key)
                .send()
                .map_err(|e| anyhow::anyhow!("Failed to connect to shodh server: {e}"))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().unwrap_or_default();
                anyhow::bail!("Export failed (HTTP {status}): {body}");
            }

            let body = resp.text().map_err(|e| anyhow::anyhow!("Failed to read response: {e}"))?;

            match output {
                Some(path) => {
                    std::fs::write(&path, &body)
                        .map_err(|e| anyhow::anyhow!("Failed to write to {}: {e}", path.display()))?;
                    eprintln!("Exported to {}", path.display());
                }
                None => {
                    println!("{body}");
                }
            }
        }
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check 2>&1 | tail -20`
Expected: No errors

- [ ] **Step 4: Verify CLI help shows the new command**

Run: `cargo run --bin shodh -- export-graph --help 2>&1`
Expected: Shows usage for export-graph with all options

- [ ] **Step 5: Commit**

```bash
git add src/cli.rs
git commit -m "feat: add shodh export-graph CLI command"
```

---

### Task 7: Run full test suite and verify

Final verification that nothing is broken.

**Files:** None (verification only)

- [ ] **Step 1: Run all export tests**

Run: `cargo test handlers::export -- --nocapture 2>&1 | tail -40`
Expected: All tests PASS

- [ ] **Step 2: Run full test suite**

Run: `cargo test 2>&1 | tail -20`
Expected: No regressions

- [ ] **Step 3: Run clippy**

Run: `cargo clippy 2>&1 | tail -20`
Expected: No new warnings in export.rs

- [ ] **Step 4: Final commit if any clippy fixes were needed**

```bash
git add -A
git commit -m "fix: address clippy warnings in export module"
```
