//! Streaming export from shodh internals to MifDocument.
//!
//! Converts internal Memory, Todo, Project, Reminder, and Graph data
//! into the vendor-neutral MIF v2 schema. Entity types are preserved
//! by looking up EntityNode labels from the knowledge graph.

use std::collections::HashMap;

use anyhow::Result;
use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::graph_memory::{EntityLabel, GraphMemory, RelationshipEdge};
use crate::memory::types::{
    ExperienceType, Memory, ProspectiveTask, ProspectiveTrigger, SourceType, Todo, TodoPriority,
    TodoStatus,
};
use crate::memory::Project;

use super::pii::PiiPatterns;
use super::schema::*;

/// Options controlling what gets exported.
#[derive(Debug, Clone)]
pub struct ExportOptions {
    pub user_id: String,
    pub include_embeddings: bool,
    pub include_graph: bool,
    pub redact_pii: bool,
    pub since: Option<DateTime<Utc>>,
}

/// Build a complete MIF v2 document from shodh internals.
///
/// This function takes pre-fetched data (memories, graph, todos, etc.) so the
/// caller can control locking and streaming. Entity types are resolved from the
/// graph when available; otherwise fall back to "unknown".
pub fn build_document(
    memories: &[std::sync::Arc<Memory>],
    graph: Option<&GraphMemory>,
    todos: &[Todo],
    projects: &[Project],
    reminders: &[ProspectiveTask],
    options: &ExportOptions,
) -> Result<MifDocument> {
    let pii = if options.redact_pii {
        Some(PiiPatterns::new())
    } else {
        None
    };

    let mut privacy = MifPrivacy {
        pii_detected: false,
        secrets_detected: false,
        redacted_fields: Vec::new(),
    };

    // Build entity lookup maps from graph for type resolution.
    // Two maps: UUID-based (primary) and name-based (fallback), because
    // entity_refs on memories may store UUIDs that don't match graph entity UUIDs
    // when the NER pipeline created refs before the graph entity was consolidated.
    let (entity_map_by_id, entity_map_by_name): (
        HashMap<Uuid, Vec<String>>,
        HashMap<String, Vec<String>>,
    ) = if let Some(g) = graph {
        let entities = g.get_all_entities().unwrap_or_default();
        let by_id: HashMap<Uuid, Vec<String>> = entities
            .iter()
            .map(|e| {
                let types: Vec<String> = e.labels.iter().map(label_to_string).collect();
                (e.uuid, types)
            })
            .collect();
        let by_name: HashMap<String, Vec<String>> = entities
            .iter()
            .map(|e| {
                let types: Vec<String> = e.labels.iter().map(label_to_string).collect();
                (e.name.to_lowercase(), types)
            })
            .collect();
        (by_id, by_name)
    } else {
        (HashMap::new(), HashMap::new())
    };

    // Convert memories
    let mut mif_memories = Vec::with_capacity(memories.len());
    let mut vendor_memory_meta: HashMap<String, serde_json::Value> = HashMap::new();

    for m in memories {
        if let Some(ref since) = options.since {
            if m.created_at < *since {
                continue;
            }
        }

        let (content, _redactions) = if let Some(ref patterns) = pii {
            let (redacted, records, found) = patterns.redact(&m.experience.content);
            if found {
                privacy.pii_detected = true;
                if patterns.has_secrets(&m.experience.content) {
                    privacy.secrets_detected = true;
                }
                for r in &records {
                    if !privacy.redacted_fields.contains(&r.redaction_type) {
                        privacy.redacted_fields.push(r.redaction_type.clone());
                    }
                }
            }
            (
                redacted,
                if records.is_empty() {
                    None
                } else {
                    Some(records)
                },
            )
        } else {
            (m.experience.content.clone(), None)
        };

        // Resolve entity types from graph (UUID lookup, then name fallback)
        let entities: Vec<MifEntityRef> = m
            .entity_refs
            .iter()
            .map(|eref| {
                let entity_type = entity_map_by_id
                    .get(&eref.entity_id)
                    .and_then(|types| types.first().cloned())
                    .or_else(|| {
                        entity_map_by_name
                            .get(&eref.name.to_lowercase())
                            .and_then(|types| types.first().cloned())
                    })
                    .unwrap_or_else(|| "unknown".to_string());
                MifEntityRef {
                    name: eref.name.clone(),
                    entity_type,
                    confidence: 1.0,
                }
            })
            .collect();

        // Also include experience.entities that didn't make it to entity_refs,
        // resolving types from the graph name map when possible.
        let ref_names: std::collections::HashSet<&str> =
            m.entity_refs.iter().map(|r| r.name.as_str()).collect();
        let mut extra_entities: Vec<MifEntityRef> = m
            .experience
            .entities
            .iter()
            .filter(|e| !ref_names.contains(e.as_str()))
            .map(|e| {
                let entity_type = entity_map_by_name
                    .get(&e.to_lowercase())
                    .and_then(|types| types.first().cloned())
                    .unwrap_or_else(|| "unknown".to_string());
                MifEntityRef {
                    name: e.clone(),
                    entity_type,
                    confidence: 0.8,
                }
            })
            .collect();

        let mut all_entities = entities;
        all_entities.append(&mut extra_entities);

        let embeddings = if options.include_embeddings {
            m.experience.embeddings.as_ref().map(|v| MifEmbedding {
                model: "minilm-l6-v2".to_string(),
                dimensions: v.len(),
                vector: v.clone(),
                normalized: true,
            })
        } else {
            None
        };

        let (source_type, session_id) = m
            .experience
            .context
            .as_ref()
            .map(|ctx| {
                let src = source_type_to_string(&ctx.source.source_type);
                let sess = ctx.episode.episode_id.clone();
                (src, sess)
            })
            .unwrap_or_else(|| ("unknown".to_string(), None));

        let agent_name = m
            .experience
            .context
            .as_ref()
            .and_then(|ctx| ctx.source.source_id.clone());

        let tags: Vec<String> = m.experience.tags.clone();

        let memory_type = experience_type_to_string(&m.experience.experience_type);

        let related_memory_ids: Vec<Uuid> = m
            .experience
            .related_memories
            .iter()
            .map(|id| id.0)
            .collect();
        let related_todo_ids: Vec<Uuid> = m.related_todo_ids.iter().map(|id| id.0).collect();

        mif_memories.push(MifMemory {
            id: m.id.0,
            content,
            memory_type,
            created_at: m.created_at,
            tags,
            entities: all_entities,
            metadata: m.experience.metadata.clone(),
            embeddings,
            source: Some(MifSource {
                source_type,
                session_id,
                agent: agent_name,
            }),
            parent_id: m.parent_id.as_ref().map(|p| p.0),
            related_memory_ids,
            related_todo_ids,
            agent_id: m.agent_id.clone(),
            external_id: m.external_id.clone(),
            version: m.version,
        });

        // Vendor extension: shodh-specific metadata per memory
        vendor_memory_meta.insert(
            m.id.0.to_string(),
            serde_json::json!({
                "importance": m.importance(),
                "access_count": m.access_count(),
                "tier": format!("{:?}", m.tier).to_lowercase(),
                "activation": m.importance(), // activation approximated by importance
                "last_accessed": m.last_accessed().to_rfc3339(),
            }),
        );
    }

    // Convert knowledge graph
    let knowledge_graph = if options.include_graph {
        if let Some(g) = graph {
            Some(build_knowledge_graph(g)?)
        } else {
            None
        }
    } else {
        None
    };

    // Convert todos
    let mif_todos: Vec<MifTodo> = todos.iter().map(convert_todo).collect();

    // Convert projects
    let mif_projects: Vec<MifProject> = projects.iter().map(convert_project).collect();

    // Convert reminders
    let mif_reminders: Vec<MifReminder> = reminders.iter().map(convert_reminder).collect();

    // Build vendor extensions
    let mut vendor_extensions: HashMap<String, serde_json::Value> = HashMap::new();
    let mut edge_metadata: HashMap<String, serde_json::Value> = HashMap::new();

    if let Some(g) = graph {
        for edge in g.get_all_relationships().unwrap_or_default() {
            edge_metadata.insert(
                edge.uuid.to_string(),
                serde_json::json!({
                    "strength": edge.strength,
                    "ltp_status": format!("{:?}", edge.ltp_status),
                    "tier": format!("{:?}", edge.tier),
                    "activation_count": edge.activation_count,
                    "last_activated": edge.last_activated.to_rfc3339(),
                }),
            );
        }
    }

    vendor_extensions.insert(
        "shodh-memory".to_string(),
        serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "memory_metadata": vendor_memory_meta,
            "edge_metadata": edge_metadata,
        }),
    );

    // Build checksum
    let mut hasher = Sha256::new();
    hasher.update(format!(
        "{}:{}:{}:{}",
        mif_memories.len(),
        mif_todos.len(),
        mif_projects.len(),
        mif_reminders.len()
    ));
    let checksum = format!("sha256:{}", hex::encode(hasher.finalize()));

    let now = Utc::now();
    let export_id = Uuid::new_v4().to_string();

    Ok(MifDocument {
        mif_version: "2.0".to_string(),
        generator: MifGenerator {
            name: "shodh-memory".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        },
        export_meta: MifExportMeta {
            id: export_id,
            created_at: now,
            user_id: options.user_id.clone(),
            checksum,
            privacy: Some(privacy),
        },
        memories: mif_memories,
        knowledge_graph,
        todos: mif_todos,
        projects: mif_projects,
        reminders: mif_reminders,
        vendor_extensions,
    })
}

fn build_knowledge_graph(graph: &GraphMemory) -> Result<MifKnowledgeGraph> {
    let entities = graph
        .get_all_entities()
        .unwrap_or_default()
        .into_iter()
        .map(|e| MifGraphEntity {
            id: e.uuid,
            name: e.name,
            types: e.labels.iter().map(label_to_string).collect(),
            attributes: e.attributes,
            summary: e.summary,
            created_at: e.created_at,
            last_seen_at: e.last_seen_at,
        })
        .collect();

    let relationships = graph
        .get_all_relationships()
        .unwrap_or_default()
        .into_iter()
        .map(|e| convert_relationship(&e))
        .collect();

    Ok(MifKnowledgeGraph {
        entities,
        relationships,
        episodes: Vec::new(), // Episodes exported via entity_ids references
    })
}

fn convert_relationship(edge: &RelationshipEdge) -> MifGraphRelationship {
    let confidence = Some(edge.strength.clamp(0.0, 1.0));
    MifGraphRelationship {
        id: edge.uuid,
        source_entity_id: edge.from_entity,
        target_entity_id: edge.to_entity,
        relation_type: relation_type_to_string(&edge.relation_type),
        context: edge.context.clone(),
        confidence,
        created_at: edge.created_at,
        valid_at: edge.valid_at,
        invalidated_at: edge.invalidated_at,
    }
}

fn convert_todo(t: &Todo) -> MifTodo {
    let comments: Vec<MifTodoComment> = t
        .comments
        .iter()
        .map(|c| MifTodoComment {
            id: c.id.0,
            content: c.content.clone(),
            comment_type: format!("{:?}", c.comment_type).to_lowercase(),
            created_at: c.created_at,
            author: Some(c.author.clone()),
        })
        .collect();

    MifTodo {
        id: t.id.0,
        content: t.content.clone(),
        status: todo_status_to_string(&t.status),
        priority: todo_priority_to_string(&t.priority),
        created_at: t.created_at,
        updated_at: t.updated_at,
        due_date: t.due_date,
        completed_at: t.completed_at,
        project_id: t.project_id.as_ref().map(|p| p.0),
        parent_id: t.parent_id.as_ref().map(|p| p.0),
        tags: t.tags.clone(),
        contexts: t.contexts.clone(),
        notes: t.notes.clone(),
        blocked_on: t.blocked_on.clone(),
        recurrence: t
            .recurrence
            .as_ref()
            .map(|r| format!("{r:?}").to_lowercase()),
        comments,
        related_memory_ids: t.related_memory_ids.iter().map(|id| id.0).collect(),
        external_id: t.external_id.clone(),
    }
}

fn convert_project(p: &Project) -> MifProject {
    MifProject {
        id: p.id.0,
        name: p.name.clone(),
        prefix: p.prefix.clone().unwrap_or_default(),
        description: p.description.clone(),
        status: format!("{:?}", p.status).to_lowercase(),
        created_at: p.created_at,
        color: p.color.clone(),
        icon: None,
    }
}

fn convert_reminder(r: &ProspectiveTask) -> MifReminder {
    let trigger = match &r.trigger {
        ProspectiveTrigger::AtTime { at } => MifTrigger::Time { at: *at },
        ProspectiveTrigger::AfterDuration { seconds, from } => MifTrigger::Duration {
            seconds: *seconds,
            from: *from,
        },
        ProspectiveTrigger::OnContext {
            keywords,
            threshold,
        } => MifTrigger::Context {
            keywords: keywords.clone(),
            threshold: *threshold,
        },
    };

    MifReminder {
        id: r.id.0,
        content: r.content.clone(),
        trigger,
        status: format!("{:?}", r.status).to_lowercase(),
        priority: r.priority,
        tags: r.tags.clone(),
        created_at: r.created_at,
        triggered_at: r.triggered_at,
        dismissed_at: r.dismissed_at,
    }
}

// =============================================================================
// STRING CONVERSION HELPERS
// =============================================================================

fn label_to_string(label: &EntityLabel) -> String {
    match label {
        EntityLabel::Person => "person".to_string(),
        EntityLabel::Organization => "organization".to_string(),
        EntityLabel::Location => "location".to_string(),
        EntityLabel::Technology => "technology".to_string(),
        EntityLabel::Concept => "concept".to_string(),
        EntityLabel::Event => "event".to_string(),
        EntityLabel::Date => "date".to_string(),
        EntityLabel::Product => "product".to_string(),
        EntityLabel::Skill => "skill".to_string(),
        EntityLabel::Keyword => "keyword".to_string(),
        EntityLabel::Other(s) => s.to_lowercase(),
    }
}

pub(crate) fn experience_type_to_string(t: &ExperienceType) -> String {
    match t {
        ExperienceType::Observation => "observation",
        ExperienceType::Decision => "decision",
        ExperienceType::Learning => "learning",
        ExperienceType::Error => "error",
        ExperienceType::Discovery => "discovery",
        ExperienceType::Pattern => "pattern",
        ExperienceType::Context => "context",
        ExperienceType::Task => "task",
        ExperienceType::CodeEdit => "code_edit",
        ExperienceType::FileAccess => "file_access",
        ExperienceType::Search => "search",
        ExperienceType::Command => "command",
        ExperienceType::Conversation => "conversation",
        ExperienceType::Intention => "intention",
    }
    .to_string()
}

fn source_type_to_string(s: &SourceType) -> String {
    match s {
        SourceType::User => "user",
        SourceType::System => "system",
        SourceType::ExternalApi => "api",
        SourceType::File => "file",
        SourceType::Web => "web",
        SourceType::AiGenerated => "ai_generated",
        SourceType::Inferred => "inferred",
        SourceType::Unknown => "unknown",
    }
    .to_string()
}

fn relation_type_to_string(r: &crate::graph_memory::RelationType) -> String {
    use crate::graph_memory::RelationType;
    match r {
        RelationType::WorksWith => "works_with",
        RelationType::WorksAt => "works_at",
        RelationType::EmployedBy => "employed_by",
        RelationType::PartOf => "part_of",
        RelationType::Contains => "contains",
        RelationType::OwnedBy => "owned_by",
        RelationType::LocatedIn => "located_in",
        RelationType::LocatedAt => "located_at",
        RelationType::Uses => "uses",
        RelationType::CreatedBy => "created_by",
        RelationType::DevelopedBy => "developed_by",
        RelationType::Causes => "causes",
        RelationType::ResultsIn => "results_in",
        RelationType::Learned => "learned",
        RelationType::Knows => "knows",
        RelationType::Teaches => "teaches",
        RelationType::RelatedTo => "related_to",
        RelationType::AssociatedWith => "associated_with",
        RelationType::CoRetrieved => "co_retrieved",
        RelationType::CoOccurs => "co_occurs",
        RelationType::Custom(s) => return s.to_lowercase(),
    }
    .to_string()
}

fn todo_status_to_string(s: &TodoStatus) -> String {
    match s {
        TodoStatus::Backlog => "backlog",
        TodoStatus::Todo => "todo",
        TodoStatus::InProgress => "in_progress",
        TodoStatus::Blocked => "blocked",
        TodoStatus::Done => "done",
        TodoStatus::Cancelled => "cancelled",
    }
    .to_string()
}

fn todo_priority_to_string(p: &TodoPriority) -> String {
    match p {
        TodoPriority::Urgent => "urgent",
        TodoPriority::High => "high",
        TodoPriority::Medium => "medium",
        TodoPriority::Low => "low",
        TodoPriority::None => "none",
    }
    .to_string()
}
