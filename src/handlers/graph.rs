//! Knowledge Graph Handlers
//!
//! Handlers for advanced knowledge graph operations including traversal,
//! entity management, and memory universe visualization.

use axum::{
    extract::{Path, Query, State},
    response::Json,
};
use serde::Deserialize;
use tracing::info;

use super::state::MultiUserMemoryManager;
use super::types::MemoryEvent;
use crate::errors::{AppError, ValidationErrorExt};
use crate::graph_memory::{
    CurvatureStats, EdgeTierCensus, EntityNode, EpisodicNode, GraphStats, GraphTraversal,
    MemoryUniverse, UniverseFilter,
};
use crate::memory::{Experience, MemoryId};
use crate::validation;
use std::collections::HashMap;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

/// GET /api/graph/{user_id}/stats - Get graph statistics for a user
pub async fn get_graph_stats(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<GraphStats>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let stats = state
        .get_user_graph_stats(&user_id)
        .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

/// GET /api/graph/{user_id}/tier-census - Edge population per consolidation tier
///
/// Separate from `/stats` on purpose: `/stats` reads three atomic counters and
/// is polled by the maintenance loop, while this performs a full O(E) scan of
/// the relationships column family.
///
/// The edge tiers carry real weight — L3 gets a 4x retrieval trust multiplier
/// over L1 and a 2160-hour prune shield versus 168 — but nothing reported how
/// many edges sat in each, so "is L3 empty or is L1 overwhelmed?" could not be
/// answered and every tier decision was a guess. Mean strength accompanies each
/// count because tier is a ratchet that decay never lowers: a full L3 of edges
/// that decayed to the floor looks identical to a healthy one by count alone.
pub async fn get_edge_tier_census(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<EdgeTierCensus>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let graph = state.get_user_graph(&user_id).map_err(AppError::Internal)?;

    let census = tokio::task::spawn_blocking(move || {
        let graph_guard = graph.read();
        graph_guard.edge_tier_census()
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Task join error: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(census))
}

/// POST /api/graph/{user_id}/curvature - Compute Forman-Ricci curvature
///
/// Triggers on-demand Forman-Ricci curvature computation for all edges
/// in the user's knowledge graph. Returns distribution statistics.
///
/// This is also computed automatically during heavy maintenance cycles.
pub async fn compute_curvature(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<CurvatureStats>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let graph = state.get_user_graph(&user_id).map_err(AppError::Internal)?;

    let stats = tokio::task::spawn_blocking(move || {
        let graph_guard = graph.read();
        graph_guard.compute_forman_ricci_curvature()
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Task join error: {}", e)))?
    .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

/// Request to find an entity
#[derive(Debug, Deserialize)]
pub struct FindEntityRequest {
    pub user_id: String,
    pub entity_name: String,
}

/// POST /api/graph/entity/find - Find an entity by name
pub async fn find_entity(
    State(state): State<AppState>,
    Json(req): Json<FindEntityRequest>,
) -> Result<Json<Option<EntityNode>>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let entity_name = req.entity_name;
    let entity = tokio::task::spawn_blocking(move || {
        let graph_guard = graph.read();
        graph_guard.find_entity_by_name(&entity_name)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Task join error: {e}")))?
    .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    Ok(Json(entity))
}

/// Request to traverse graph
#[derive(Debug, Deserialize)]
pub struct TraverseGraphRequest {
    pub user_id: String,
    pub entity_name: String,
    pub max_depth: Option<usize>,
}

/// POST /api/graph/traverse - Traverse graph from an entity
pub async fn traverse_graph(
    State(state): State<AppState>,
    Json(req): Json<TraverseGraphRequest>,
) -> Result<Json<GraphTraversal>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let entity_name = req.entity_name.clone();
    let missing_entity_name = req.entity_name;
    let max_depth = req.max_depth.unwrap_or(2);
    let traversal = tokio::task::spawn_blocking(move || {
        let graph_guard = graph.read();

        let Some(entity) = graph_guard
            .find_entity_by_name(&entity_name)
            .map_err(|e| anyhow::anyhow!(e))?
        else {
            return Ok(None);
        };

        let traversal = graph_guard
            .traverse_from_entity(&entity.uuid, max_depth)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(Some(traversal))
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Task join error: {e}")))?
    .map_err(AppError::Internal)?
    .ok_or_else(|| AppError::EntityNotFound(missing_entity_name))?;

    Ok(Json(traversal))
}

/// Request to get an episode
#[derive(Debug, Deserialize)]
pub struct GetEpisodeRequest {
    pub user_id: String,
    pub episode_uuid: String,
}

/// POST /api/graph/episode/get - Get an episodic node by UUID
pub async fn get_episode(
    State(state): State<AppState>,
    Json(req): Json<GetEpisodeRequest>,
) -> Result<Json<Option<EpisodicNode>>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let episode_uuid =
        uuid::Uuid::parse_str(&req.episode_uuid).map_err(|_| AppError::InvalidInput {
            field: "episode_uuid".to_string(),
            reason: "Invalid UUID format".to_string(),
        })?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let episode = tokio::task::spawn_blocking(move || {
        let graph_guard = graph.read();
        graph_guard.get_episode(&episode_uuid)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Task join error: {e}")))?
    .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    Ok(Json(episode))
}

/// Request to get all entities
#[derive(Debug, Deserialize)]
pub struct GetAllEntitiesRequest {
    pub user_id: String,
    pub limit: Option<usize>,
    /// Include each entity's 384-float `name_embedding` in the response.
    ///
    /// Defaults to FALSE. The embedding is roughly 3KB of JSON per entity, so a
    /// 500-entity listing shipped about 3MB that every known caller discarded
    /// on arrival. It stays available for a caller that genuinely wants to
    /// compute similarity itself; it is no longer the price of asking which
    /// entities exist.
    #[serde(default)]
    pub include_embeddings: bool,
}

/// POST /api/graph/entities/all - Get all entities, most salient first
pub async fn get_all_entities(
    State(state): State<AppState>,
    Json(req): Json<GetAllEntitiesRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let limit = req.limit.unwrap_or(100);
    let include_embeddings = req.include_embeddings;
    let mut entities = tokio::task::spawn_blocking(move || {
        let graph_guard = graph.read();
        graph_guard.get_all_entities()
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Task join error: {e}")))?
    .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    // Rank BEFORE truncating. Taking `limit` straight off the iterator returned
    // whatever storage order produced, so "the 100 entities" meant 100
    // arbitrary ones — a caller asking for the graph's cast of characters got
    // an unordered sample, and the most important entity in the graph could sit
    // outside it for no reason it could observe.
    //
    // Salience is the ranking the graph already computes for exactly this
    // question ("how much does this entity matter here"); mention count breaks
    // ties, and name breaks those, so the order is total and the same request
    // twice returns the same page.
    entities.sort_by(|a, b| {
        b.salience
            .partial_cmp(&a.salience)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.mention_count.cmp(&a.mention_count))
            .then_with(|| a.name.cmp(&b.name))
    });
    entities.truncate(limit);

    if !include_embeddings {
        for entity in &mut entities {
            entity.name_embedding = None;
        }
    }

    let count = entities.len();

    Ok(Json(serde_json::json!({
        "entities": entities,
        "count": count
    })))
}

/// GET /api/graph/{user_id}/universe - Get Memory Universe visualization
pub async fn get_memory_universe(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<MemoryUniverse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    // Read filter, overridable per request. The default hides only generic edges
    // already below the prune threshold; a caller exploring the raw substrate
    // passes `?min_generic_strength=0&hide_redundant_generic=false` to see
    // everything the ingest pipeline built. Whatever is applied is echoed back on
    // `filter`, so a viewer is never shown a subset without being told.
    let default_filter = UniverseFilter::default();
    let filter = UniverseFilter {
        min_generic_strength: params
            .get("min_generic_strength")
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(default_filter.min_generic_strength),
        hide_redundant_generic: params
            .get("hide_redundant_generic")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(default_filter.hide_redundant_generic),
    };

    let graph = state.get_user_graph(&user_id).map_err(AppError::Internal)?;

    let universe = tokio::task::spawn_blocking(move || {
        let graph_guard = graph.read();
        graph_guard.get_universe_filtered(filter)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Task join error: {e}")))?
    .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    Ok(Json(universe))
}

/// DELETE /api/graph/{user_id}/clear - Clear all graph data for a user
pub async fn clear_user_graph(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let graph = state.get_user_graph(&user_id).map_err(AppError::Internal)?;

    let (entities, relationships, episodes) = tokio::task::spawn_blocking(move || {
        let graph_guard = graph.write();
        graph_guard.clear_all()
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Task join error: {e}")))?
    .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    info!(
        "Cleared graph for user {}: {} entities, {} relationships, {} episodes",
        user_id, entities, relationships, episodes
    );

    state.emit_event(MemoryEvent {
        event_type: "GRAPH_CLEAR".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: user_id.clone(),
        memory_id: Some(format!("{}/{}/{}", entities, relationships, episodes)),
        content_preview: Some(format!(
            "Cleared {} entities, {} relationships, {} episodes",
            entities, relationships, episodes
        )),
        memory_type: Some("graph".to_string()),
        importance: None,
        count: Some(entities + relationships + episodes),
        entities: None,
        results: None,
    });

    Ok(Json(serde_json::json!({
        "cleared": {
            "entities": entities,
            "relationships": relationships,
            "episodes": episodes
        }
    })))
}

/// POST /api/graph/{user_id}/canonicalize - Collapse duplicate mention-nodes
///
/// Runs entity-linking over the live graph: the spaCy-rusty parser detects each
/// mention's syntactic head and routes out verb-fragment junk, then the
/// Fellegi-Sunter (Splink) matcher clusters the surviving mentions type-blocked
/// at a precision-first threshold. Each cluster's members are merged into the
/// most-proper / most-mentioned node, re-pointing edges and deleting the
/// duplicates. Returns how many nodes were merged and how many edges re-pointed.
pub async fn canonicalize_user_graph(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let graph = state.get_user_graph(&user_id).map_err(AppError::Internal)?;

    let (merged, repointed) = tokio::task::spawn_blocking(move || {
        let graph_guard = graph.write();
        graph_guard.canonicalize_entities()
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Task join error: {e}")))?
    .map_err(AppError::Internal)?;

    info!(
        "Canonicalized graph for user {}: merged {} mention-nodes, re-pointed {} edges",
        user_id, merged, repointed
    );

    state.emit_event(MemoryEvent {
        event_type: "GRAPH_CANONICALIZE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: user_id.clone(),
        memory_id: Some(format!("{merged}/{repointed}")),
        content_preview: Some(format!(
            "Merged {merged} duplicate mention-nodes, re-pointed {repointed} edges"
        )),
        memory_type: Some("graph".to_string()),
        importance: None,
        count: Some(merged),
        entities: None,
        results: None,
    });

    Ok(Json(serde_json::json!({
        "canonicalized": {
            "merged_nodes": merged,
            "repointed_edges": repointed
        }
    })))
}

/// Query parameters for the graph rebuild.
#[derive(Debug, Default, Deserialize)]
pub struct RebuildGraphParams {
    /// Discard each memory's cached entity extraction and re-run NER over its
    /// content (`?fresh_ner=true`). Defaults to false.
    ///
    /// A rebuild replays every stored memory through the graph pipeline, which
    /// reads `experience.entities` / `experience.ner_entities` first and only
    /// falls through to the neural typer when both are empty. That ordering is
    /// right for a normal rebuild — it is what makes the operation cheap and
    /// reproducible — but it means a rebuild reconstructs the typing decisions
    /// that were cached at ingest, not the ones the current typer would make.
    ///
    /// So a corpus ingested before a typer improvement cannot be re-typed at
    /// all. It replays its own history forever, and every entity keeps the class
    /// (and the absent `fine_type`) it was given by whatever ran the day it was
    /// written. On a corpus whose cached `entities` are really keyphrase tags,
    /// that also means the graph is rebuilt out of tags rather than out of the
    /// text — filenames and fragments included.
    ///
    /// Setting this drops the cache for the duration of the rebuild so entities
    /// are re-derived from `content`. It costs a full NER pass over the corpus,
    /// which is why it is opt-in.
    #[serde(default)]
    pub fresh_ner: bool,
}

/// POST /api/graph/{user_id}/rebuild - Rebuild graph from all existing memories
pub async fn rebuild_user_graph(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(params): Query<RebuildGraphParams>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    // First, clear existing graph data
    let graph = state.get_user_graph(&user_id).map_err(AppError::Internal)?;
    {
        let graph_clone = graph.clone();
        let _ = tokio::task::spawn_blocking(move || {
            let graph_guard = graph_clone.write();
            graph_guard.clear_all()
        })
        .await;
    }

    // Get all memories for this user
    let memory_sys = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;
    let memories: Vec<(MemoryId, Experience)> = {
        let memory_guard = memory_sys.read();
        memory_guard
            .get_all_memories()
            .map_err(AppError::Internal)?
            .into_iter()
            .map(|m| (m.id.clone(), m.experience.clone()))
            .collect()
    };

    let total_memories = memories.len();
    let mut processed = 0;
    let fresh_ner = params.fresh_ner;

    // Re-process each memory through entity extraction
    for (memory_id, mut experience) in memories {
        if fresh_ner {
            // Drop the cached extraction so the pipeline re-derives entities from
            // `content`. See `RebuildGraphParams::fresh_ner` — without this the
            // rebuild cannot change any entity's type, however much the typer has
            // improved since the memory was written.
            experience.entities.clear();
            experience.ner_entities.clear();
        }
        if let Err(e) = state.process_experience_into_graph(&user_id, &experience, &memory_id, None)
        {
            tracing::debug!("Failed to process memory {}: {}", memory_id.0, e);
        } else {
            processed += 1;
        }
    }

    // Get final stats
    let stats = state
        .get_user_graph_stats(&user_id)
        .map_err(AppError::Internal)?;
    let entities_created = stats.entity_count;
    let relationships_created = stats.relationship_count;

    info!(
        "Rebuilt graph for user {}: processed {}/{} memories, created {} entities, {} relationships",
        user_id, processed, total_memories, entities_created, relationships_created
    );

    state.emit_event(MemoryEvent {
        event_type: "GRAPH_REBUILD".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: user_id.clone(),
        memory_id: None,
        content_preview: Some(format!(
            "Rebuilt: {} memories -> {} entities, {} relationships",
            processed, entities_created, relationships_created
        )),
        memory_type: Some("graph".to_string()),
        importance: None,
        count: Some(entities_created + relationships_created),
        entities: None,
        results: None,
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "processed_memories": processed,
        "total_memories": total_memories,
        "entities_created": entities_created,
        "relationships_created": relationships_created
    })))
}

/// Request to invalidate a relationship
#[derive(Debug, Deserialize)]
pub struct InvalidateRelationshipRequest {
    pub user_id: String,
    pub relationship_uuid: String,
}

/// POST /api/graph/relationship/invalidate - Invalidate a relationship edge
pub async fn invalidate_relationship(
    State(state): State<AppState>,
    Json(req): Json<InvalidateRelationshipRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_guard = graph.write();

    let rel_uuid =
        uuid::Uuid::parse_str(&req.relationship_uuid).map_err(|_| AppError::InvalidInput {
            field: "relationship_uuid".to_string(),
            reason: "Invalid UUID format".to_string(),
        })?;

    graph_guard
        .invalidate_relationship(&rel_uuid)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    state.emit_event(MemoryEvent {
        event_type: "EDGE_INVALIDATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(req.relationship_uuid.clone()),
        content_preview: Some("Relationship invalidated".to_string()),
        memory_type: Some("graph".to_string()),
        importance: None,
        count: None,
        entities: None,
        results: None,
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Relationship invalidated"
    })))
}
