//! Knowledge Graph Handlers
//!
//! Handlers for advanced knowledge graph operations including traversal,
//! entity management, and memory universe visualization.

use axum::{
    extract::{Path, State},
    response::Json,
};
use serde::Deserialize;
use tracing::info;

use super::state::MultiUserMemoryManager;
use super::types::MemoryEvent;
use crate::errors::{AppError, ValidationErrorExt};
use crate::graph_memory::{EntityNode, EpisodicNode, GraphStats, GraphTraversal, MemoryUniverse};
use crate::memory::{Experience, MemoryId};
use crate::validation;
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

    let graph_guard = graph.read();
    let entity = graph_guard
        .find_entity_by_name(&req.entity_name)
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

    let graph_guard = graph.read();

    let entity = graph_guard
        .find_entity_by_name(&req.entity_name)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?
        .ok_or_else(|| {
            AppError::MemoryNotFound(format!("Entity not found: {}", req.entity_name))
        })?;

    let max_depth = req.max_depth.unwrap_or(2);
    let traversal = graph_guard
        .traverse_from_entity(&entity.uuid, max_depth)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

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

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_guard = graph.read();

    let episode_uuid =
        uuid::Uuid::parse_str(&req.episode_uuid).map_err(|_| AppError::InvalidInput {
            field: "episode_uuid".to_string(),
            reason: "Invalid UUID format".to_string(),
        })?;

    let episode = graph_guard
        .get_episode(&episode_uuid)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    Ok(Json(episode))
}

/// Request to get all entities
#[derive(Debug, Deserialize)]
pub struct GetAllEntitiesRequest {
    pub user_id: String,
    pub limit: Option<usize>,
}

/// POST /api/graph/entities/all - Get all entities
pub async fn get_all_entities(
    State(state): State<AppState>,
    Json(req): Json<GetAllEntitiesRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;
    let graph_guard = graph.read();

    let entities = graph_guard
        .get_all_entities()
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    let limit = req.limit.unwrap_or(100);
    let entities: Vec<_> = entities.into_iter().take(limit).collect();
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
) -> Result<Json<MemoryUniverse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let graph = state.get_user_graph(&user_id).map_err(AppError::Internal)?;

    let graph_guard = graph.read();
    let universe = graph_guard
        .get_universe()
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
    let graph_guard = graph.write();

    let (entities, relationships, episodes) = graph_guard
        .clear_all()
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
    });

    Ok(Json(serde_json::json!({
        "cleared": {
            "entities": entities,
            "relationships": relationships,
            "episodes": episodes
        }
    })))
}

/// POST /api/graph/{user_id}/rebuild - Rebuild graph from all existing memories
pub async fn rebuild_user_graph(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    // First, clear existing graph data
    let graph = state.get_user_graph(&user_id).map_err(AppError::Internal)?;
    {
        let graph_guard = graph.write();
        let _ = graph_guard.clear_all();
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

    // Re-process each memory through entity extraction
    for (memory_id, experience) in memories {
        if let Err(e) = state.process_experience_into_graph(&user_id, &experience, &memory_id) {
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
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Relationship invalidated"
    })))
}
