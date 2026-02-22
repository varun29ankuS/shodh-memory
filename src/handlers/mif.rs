//! MIF v2 Handlers — Export, Import, Adapters, Graph Entity/Relationship, Storage
//!
//! Thin HTTP layer over `crate::mif` for vendor-neutral memory interchange.

use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::state::MultiUserMemoryManager;
use super::types::RetrieveResponse;
use crate::errors::{AppError, ValidationErrorExt};
use crate::graph_memory;
use crate::mif::adapters::AdapterRegistry;
use crate::mif::{export, import};
use crate::validation;

pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

// =============================================================================
// MIF EXPORT
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct MifExportRequest {
    pub user_id: String,
    #[serde(default)]
    pub include_embeddings: bool,
    #[serde(default)]
    pub include_graph: bool,
    #[serde(default)]
    pub since: Option<String>,
    #[serde(default)]
    pub redact_pii: bool,
    #[serde(default = "default_format")]
    pub format: String,
}

fn default_format() -> String {
    "shodh".to_string()
}

#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn export_mif(
    State(state): State<AppState>,
    Json(req): Json<MifExportRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let user_id = req.user_id.clone();

    // Gather data
    let memory_sys = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;
    let memories = {
        let guard = memory_sys.read();
        guard.get_all_memories().unwrap_or_default()
    };

    let graph = if req.include_graph {
        Some(state.get_user_graph(&user_id).map_err(AppError::Internal)?)
    } else {
        None
    };

    let todos = state
        .todo_store
        .list_todos_for_user(&user_id, None)
        .unwrap_or_default();

    let projects = state.todo_store.list_projects(&user_id).unwrap_or_default();

    let reminders = state
        .prospective_store
        .list_for_user(&user_id, None)
        .unwrap_or_default();

    let since = req
        .since
        .as_ref()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc));

    let options = export::ExportOptions {
        user_id: user_id.clone(),
        include_embeddings: req.include_embeddings,
        include_graph: req.include_graph,
        redact_pii: req.redact_pii,
        since,
    };

    let graph_ref = graph.as_ref().map(|g| g.read());
    let graph_guard = graph_ref.as_deref();

    let doc = export::build_document(
        &memories,
        graph_guard,
        &todos,
        &projects,
        &reminders,
        &options,
    )
    .map_err(AppError::Internal)?;

    // Convert to requested output format
    let output = if req.format == "shodh" || req.format == "json" {
        serde_json::to_value(&doc).map_err(|e| AppError::Internal(e.into()))?
    } else {
        let registry = AdapterRegistry::new();
        let bytes = registry
            .export_with(&req.format, &doc)
            .map_err(AppError::Internal)?;
        let output_str = String::from_utf8(bytes).map_err(|e| AppError::Internal(e.into()))?;
        // For non-JSON formats, wrap in an object
        serde_json::json!({
            "format": req.format,
            "data": output_str,
            "memory_count": doc.memories.len(),
            "todo_count": doc.todos.len(),
        })
    };

    state.log_event(
        &user_id,
        "MIF_EXPORT",
        &doc.export_meta.id,
        &format!(
            "Exported {} memories, {} todos (format: {})",
            doc.memories.len(),
            doc.todos.len(),
            req.format,
        ),
    );

    Ok(Json(output))
}

// =============================================================================
// MIF IMPORT
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct MifImportRequest {
    pub user_id: String,
    #[serde(default = "default_skip_duplicates")]
    pub skip_duplicates: bool,
    #[serde(default)]
    pub format: Option<String>,
    pub data: serde_json::Value,
}

fn default_skip_duplicates() -> bool {
    true
}

#[derive(Debug, Serialize)]
pub struct MifImportResponse {
    pub success: bool,
    pub result: import::ImportResult,
}

#[tracing::instrument(skip(state, req), fields(user_id = %req.user_id))]
pub async fn import_mif(
    State(state): State<AppState>,
    Json(req): Json<MifImportRequest>,
) -> Result<Json<MifImportResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Convert data to bytes for adapter detection
    let data_bytes = serde_json::to_vec(&req.data).map_err(|e| AppError::Internal(e.into()))?;

    // Parse through adapter registry (auto-detect or explicit format)
    let registry = AdapterRegistry::new();
    let doc = if let Some(ref fmt) = req.format {
        registry
            .import_with(fmt, &data_bytes)
            .map_err(AppError::Internal)?
    } else {
        registry
            .auto_import(&data_bytes)
            .map_err(AppError::Internal)?
    };

    // Validate version
    import::validate_version(&doc).map_err(AppError::Internal)?;

    let options = import::ImportOptions {
        user_id: req.user_id.clone(),
        skip_duplicates: req.skip_duplicates,
    };

    let mut result = import::ImportResult::default();

    // Build dedup set from existing memories
    let dedup_set = if req.skip_duplicates {
        let memory_sys = state
            .get_user_memory(&req.user_id)
            .map_err(AppError::Internal)?;
        let guard = memory_sys.read();
        let existing: Vec<String> = guard
            .get_all_memories()
            .unwrap_or_default()
            .iter()
            .map(|m| m.experience.content.clone())
            .collect();
        import::build_dedup_set(&existing)
    } else {
        std::collections::HashSet::new()
    };

    // Import memories
    let (prepared_memories, skipped) = import::prepare_memories(&doc, &dedup_set, &options);
    result.duplicates_skipped = skipped;

    {
        let memory_sys = state
            .get_user_memory(&req.user_id)
            .map_err(AppError::Internal)?;
        let guard = memory_sys.read();

        for (memory_id, experience, created_at) in prepared_memories {
            match guard.remember_with_id(memory_id, experience, created_at) {
                Ok(_) => result.memories_imported += 1,
                Err(e) => result.errors.push(format!("Memory: {e}")),
            }
        }
    }

    // Import projects first (todos may reference them)
    let prepared_projects = import::prepare_projects(&doc, &req.user_id);
    for project in prepared_projects {
        match state.todo_store.store_project(&project) {
            Ok(_) => result.projects_imported += 1,
            Err(e) => result
                .errors
                .push(format!("Project '{}': {e}", project.name)),
        }
    }

    // Import todos
    let prepared_todos = import::prepare_todos(&doc, &req.user_id);
    for todo in prepared_todos {
        match state.todo_store.store_todo(&todo) {
            Ok(_) => result.todos_imported += 1,
            Err(e) => result.errors.push(format!("Todo: {e}")),
        }
    }

    // Import reminders
    let prepared_reminders = import::prepare_reminders(&doc, &req.user_id);
    for reminder in prepared_reminders {
        match state.prospective_store.store(&reminder) {
            Ok(_) => result.reminders_imported += 1,
            Err(e) => result.errors.push(format!("Reminder: {e}")),
        }
    }

    // Import knowledge graph
    if let Some(ref kg) = doc.knowledge_graph {
        let graph_arc = state
            .get_user_graph(&req.user_id)
            .map_err(AppError::Internal)?;
        let graph_guard = graph_arc.read();

        let (entities_imported, entity_errors) = import::import_graph_entities(kg, &graph_guard);
        result.entities_imported = entities_imported;
        result.errors.extend(entity_errors);

        let (edges_imported, edge_errors) =
            import::import_graph_relationships(kg, &graph_guard, &doc.vendor_extensions);
        result.edges_imported = edges_imported;
        result.errors.extend(edge_errors);
    }

    state.log_event(
        &req.user_id,
        "MIF_IMPORT",
        "import",
        &format!(
            "Imported {} memories, {} todos, {} projects, {} reminders, {} entities, {} edges. Skipped {} dupes, {} errors",
            result.memories_imported,
            result.todos_imported,
            result.projects_imported,
            result.reminders_imported,
            result.entities_imported,
            result.edges_imported,
            result.duplicates_skipped,
            result.errors.len(),
        ),
    );

    Ok(Json(MifImportResponse {
        success: result.errors.is_empty(),
        result,
    }))
}

// =============================================================================
// ADAPTER LISTING
// =============================================================================

pub async fn list_adapters() -> Json<serde_json::Value> {
    let registry = AdapterRegistry::new();
    let adapters: Vec<serde_json::Value> = registry
        .list_adapters()
        .into_iter()
        .map(|(name, format_id)| {
            serde_json::json!({
                "name": name,
                "format": format_id,
            })
        })
        .collect();

    Json(serde_json::json!({
        "adapters": adapters,
        "default_export": "shodh",
        "default_import": "auto-detect",
    }))
}

// =============================================================================
// GRAPH ENTITY HANDLERS (kept here for backward compat)
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct AddEntityRequest {
    pub user_id: String,
    pub name: String,
    pub label: String,
    pub attributes: Option<HashMap<String, String>>,
}

pub async fn add_entity(
    State(state): State<AppState>,
    Json(req): Json<AddEntityRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_entity(&req.name).map_validation_err("name")?;
    validation::validate_entity(&req.label).map_validation_err("label")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;
    let graph_guard = graph.write();

    let entity_label = match req.label.as_str() {
        "Person" => graph_memory::EntityLabel::Person,
        "Organization" => graph_memory::EntityLabel::Organization,
        "Location" => graph_memory::EntityLabel::Location,
        "Event" => graph_memory::EntityLabel::Event,
        "Concept" => graph_memory::EntityLabel::Concept,
        "Technology" => graph_memory::EntityLabel::Technology,
        "Product" => graph_memory::EntityLabel::Product,
        "Skill" => graph_memory::EntityLabel::Skill,
        "Date" => graph_memory::EntityLabel::Date,
        _ => graph_memory::EntityLabel::Other(req.label.clone()),
    };

    let mut attributes = HashMap::new();
    if let Some(attrs) = &req.attributes {
        for (key, value) in attrs {
            attributes.insert(
                key.clone(),
                serde_json::to_string(value).unwrap_or_default(),
            );
        }
    }

    let entity = graph_memory::EntityNode {
        uuid: uuid::Uuid::new_v4(),
        name: req.name.clone(),
        labels: vec![entity_label.clone()],
        created_at: chrono::Utc::now(),
        last_seen_at: chrono::Utc::now(),
        mention_count: 1,
        summary: String::new(),
        attributes,
        name_embedding: None,
        salience: 0.5,
        is_proper_noun: true,
    };

    graph_guard.add_entity(entity).map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "ADD_ENTITY",
        &req.name,
        &format!("Added entity '{}' with label {:?}", req.name, entity_label),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "entity": req.name,
        "label": format!("{:?}", entity_label)
    })))
}

#[derive(Debug, Deserialize)]
pub struct AddRelationshipRequest {
    pub user_id: String,
    pub from_entity: String,
    pub to_entity: String,
    pub relation_type: String,
    pub strength: Option<f32>,
}

pub async fn add_relationship(
    State(state): State<AppState>,
    Json(req): Json<AddRelationshipRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_entity(&req.from_entity).map_validation_err("from_entity")?;
    validation::validate_entity(&req.to_entity).map_validation_err("to_entity")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;
    let graph_guard = graph.write();

    let relation_type = match req.relation_type.as_str() {
        "RelatedTo" => graph_memory::RelationType::RelatedTo,
        "AssociatedWith" => graph_memory::RelationType::AssociatedWith,
        "PartOf" => graph_memory::RelationType::PartOf,
        "Contains" => graph_memory::RelationType::Contains,
        "OwnedBy" => graph_memory::RelationType::OwnedBy,
        "CreatedBy" => graph_memory::RelationType::CreatedBy,
        "DevelopedBy" => graph_memory::RelationType::DevelopedBy,
        "Uses" => graph_memory::RelationType::Uses,
        "WorksWith" => graph_memory::RelationType::WorksWith,
        "WorksAt" => graph_memory::RelationType::WorksAt,
        "EmployedBy" => graph_memory::RelationType::EmployedBy,
        "LocatedIn" => graph_memory::RelationType::LocatedIn,
        "LocatedAt" => graph_memory::RelationType::LocatedAt,
        "Causes" => graph_memory::RelationType::Causes,
        "ResultsIn" => graph_memory::RelationType::ResultsIn,
        "Learned" => graph_memory::RelationType::Learned,
        "Knows" => graph_memory::RelationType::Knows,
        "Teaches" => graph_memory::RelationType::Teaches,
        "CoRetrieved" => graph_memory::RelationType::CoRetrieved,
        _ => graph_memory::RelationType::Custom(req.relation_type.clone()),
    };

    let strength = req.strength.unwrap_or(0.5);

    let from_entity = graph_guard
        .find_entity_by_name(&req.from_entity)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::InvalidInput {
            field: "from_entity".to_string(),
            reason: format!("Entity '{}' not found", req.from_entity),
        })?;

    let to_entity = graph_guard
        .find_entity_by_name(&req.to_entity)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::InvalidInput {
            field: "to_entity".to_string(),
            reason: format!("Entity '{}' not found", req.to_entity),
        })?;

    let entity_confidence = Some((from_entity.salience + to_entity.salience) / 2.0);

    let edge = graph_memory::RelationshipEdge {
        uuid: uuid::Uuid::new_v4(),
        from_entity: from_entity.uuid,
        to_entity: to_entity.uuid,
        relation_type: relation_type.clone(),
        strength,
        created_at: chrono::Utc::now(),
        valid_at: chrono::Utc::now(),
        invalidated_at: None,
        source_episode_id: None,
        context: String::new(),
        last_activated: chrono::Utc::now(),
        activation_count: 1,
        ltp_status: graph_memory::LtpStatus::None,
        tier: graph_memory::EdgeTier::L1Working,
        activation_timestamps: None,
        entity_confidence,
    };

    graph_guard
        .add_relationship(edge)
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "ADD_RELATIONSHIP",
        &format!("{}->{}", req.from_entity, req.to_entity),
        &format!(
            "Added {:?} relationship from '{}' to '{}'",
            relation_type, req.from_entity, req.to_entity
        ),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "from": req.from_entity,
        "to": req.to_entity,
        "type": format!("{:?}", relation_type),
        "strength": strength
    })))
}

// =============================================================================
// STORAGE HANDLER
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct GetUncompressedRequest {
    pub user_id: String,
    pub days_old: u32,
}

pub async fn get_uncompressed_old(
    State(state): State<AppState>,
    Json(req): Json<GetUncompressedRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    let cutoff = chrono::Utc::now() - chrono::Duration::days(req.days_old as i64);
    let raw_memories = memory_guard
        .get_uncompressed_older_than(cutoff)
        .map_err(AppError::Internal)?;

    let count = raw_memories.len();
    let memories: Vec<serde_json::Value> = raw_memories
        .into_iter()
        .filter_map(|m| serde_json::to_value(&m).ok())
        .collect();

    Ok(Json(RetrieveResponse { memories, count }))
}
