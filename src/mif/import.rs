//! Reference-preserving import from MifDocument to shodh internals.
//!
//! Key improvements over v1:
//! - UUID preservation: memories keep their original IDs via `remember_with_id()`
//! - Content-hash dedup: O(1) duplicate check via SHA256 HashSet (replaces O(n*k) recall)
//! - Reference mapping: remaps parent_id, related_ids when collisions occur
//! - Graph reconstruction: entities and edges restored with proper types

use std::collections::{HashMap, HashSet};

use crate::graph_memory::{
    EdgeTier, EntityLabel, EntityNode, GraphMemory, LtpStatus, RelationType, RelationshipEdge,
};
use crate::memory::types::{
    Experience, ExperienceType, MemoryId, ProspectiveTask, ProspectiveTaskId,
    ProspectiveTaskStatus, ProspectiveTrigger, Todo, TodoId, TodoPriority, TodoStatus,
};
use crate::memory::{Project, ProjectId, ProjectStatus};
use anyhow::{bail, Result};
use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};

use super::schema::*;

/// Options controlling import behavior.
#[derive(Debug, Clone)]
pub struct ImportOptions {
    pub user_id: String,
    pub skip_duplicates: bool,
}

/// Result of an import operation.
#[derive(Debug, Default, serde::Serialize)]
pub struct ImportResult {
    pub memories_imported: usize,
    pub todos_imported: usize,
    pub projects_imported: usize,
    pub reminders_imported: usize,
    pub edges_imported: usize,
    pub entities_imported: usize,
    pub duplicates_skipped: usize,
    pub errors: Vec<String>,
}

/// Build a content-hash set from existing memories for O(1) dedup.
pub fn build_dedup_set(existing_contents: &[String]) -> HashSet<[u8; 32]> {
    existing_contents.iter().map(|c| content_hash(c)).collect()
}

fn content_hash(content: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hasher.finalize().into()
}

/// Convert MIF memories into Experience structs ready for `remember_with_id()`.
///
/// A prepared memory ready for import: (id, experience, optional creation timestamp).
pub type PreparedMemory = (MemoryId, Experience, Option<DateTime<Utc>>);

/// Returns a vec of prepared memories plus the count of skipped duplicates.
/// The caller is responsible for actually storing them.
pub fn prepare_memories(
    doc: &MifDocument,
    dedup_set: &HashSet<[u8; 32]>,
    options: &ImportOptions,
) -> (Vec<PreparedMemory>, usize) {
    let mut prepared = Vec::new();
    let mut skipped = 0;

    for mem in &doc.memories {
        // Dedup check
        if options.skip_duplicates && dedup_set.contains(&content_hash(&mem.content)) {
            skipped += 1;
            continue;
        }

        let exp_type = parse_experience_type(&mem.memory_type);

        let mut metadata = mem.metadata.clone();
        if !mem.tags.is_empty() && !metadata.contains_key("tags") {
            metadata.insert("tags".to_string(), mem.tags.join(","));
        }

        let entities: Vec<String> = mem.entities.iter().map(|e| e.name.clone()).collect();

        let embeddings = mem.embeddings.as_ref().map(|e| e.vector.clone());

        let experience = Experience {
            experience_type: exp_type,
            content: mem.content.clone(),
            entities,
            metadata,
            embeddings,
            tags: mem.tags.clone(),
            ..Default::default()
        };

        let memory_id = MemoryId(mem.id);
        let created_at = Some(mem.created_at);

        prepared.push((memory_id, experience, created_at));
    }

    (prepared, skipped)
}

/// Convert MIF todos into internal Todo structs.
///
/// Returns todos ready for `store_todo()`. Project IDs are preserved if present.
pub fn prepare_todos(doc: &MifDocument, user_id: &str) -> Vec<Todo> {
    doc.todos
        .iter()
        .map(|t| {
            let status = parse_todo_status(&t.status);
            let priority = parse_todo_priority(&t.priority);

            let comments = t
                .comments
                .iter()
                .map(|c| crate::memory::types::TodoComment {
                    id: crate::memory::types::TodoCommentId(c.id),
                    todo_id: TodoId(t.id),
                    author: c.author.clone().unwrap_or_else(|| "import".to_string()),
                    content: c.content.clone(),
                    comment_type: parse_comment_type(&c.comment_type),
                    created_at: c.created_at,
                    updated_at: None,
                })
                .collect();

            let related_memory_ids: Vec<MemoryId> = t
                .related_memory_ids
                .iter()
                .map(|id| MemoryId(*id))
                .collect();

            Todo {
                id: TodoId(t.id),
                seq_num: 0,
                project_prefix: None,
                project: None,
                user_id: user_id.to_string(),
                content: t.content.clone(),
                status,
                priority,
                project_id: t.project_id.map(ProjectId),
                parent_id: t.parent_id.map(TodoId),
                contexts: t.contexts.clone(),
                tags: t.tags.clone(),
                notes: t.notes.clone(),
                blocked_on: t.blocked_on.clone(),
                recurrence: None,
                created_at: t.created_at,
                updated_at: t.updated_at,
                due_date: t.due_date,
                completed_at: t.completed_at,
                sort_order: 0,
                comments,
                embedding: None,
                related_memory_ids,
                external_id: t.external_id.clone(),
            }
        })
        .collect()
}

/// Convert MIF projects into internal Project structs.
pub fn prepare_projects(doc: &MifDocument, user_id: &str) -> Vec<Project> {
    doc.projects
        .iter()
        .map(|p| Project {
            id: ProjectId(p.id),
            user_id: user_id.to_string(),
            name: p.name.clone(),
            prefix: if p.prefix.is_empty() {
                None
            } else {
                Some(p.prefix.clone())
            },
            description: p.description.clone(),
            status: parse_project_status(&p.status),
            color: p.color.clone(),
            parent_id: None,
            created_at: p.created_at,
            completed_at: None,
            codebase_path: None,
            codebase_indexed: false,
            codebase_indexed_at: None,
            codebase_file_count: 0,
            embedding: None,
            related_memory_ids: Vec::new(),
            todo_counts: Default::default(),
        })
        .collect()
}

/// Convert MIF reminders into internal ProspectiveTask structs.
pub fn prepare_reminders(doc: &MifDocument, user_id: &str) -> Vec<ProspectiveTask> {
    doc.reminders
        .iter()
        .map(|r| {
            let trigger = match &r.trigger {
                MifTrigger::Time { at } => ProspectiveTrigger::AtTime { at: *at },
                MifTrigger::Duration { seconds, from } => ProspectiveTrigger::AfterDuration {
                    seconds: *seconds,
                    from: *from,
                },
                MifTrigger::Context {
                    keywords,
                    threshold,
                } => ProspectiveTrigger::OnContext {
                    keywords: keywords.clone(),
                    threshold: *threshold,
                },
            };

            let status = parse_reminder_status(&r.status);

            ProspectiveTask {
                id: ProspectiveTaskId(r.id),
                user_id: user_id.to_string(),
                content: r.content.clone(),
                trigger,
                status,
                created_at: r.created_at,
                triggered_at: r.triggered_at,
                dismissed_at: r.dismissed_at,
                tags: r.tags.clone(),
                priority: r.priority,
                embedding: None,
                related_memory_ids: Vec::new(),
            }
        })
        .collect()
}

/// Import graph entities from MIF document.
///
/// Returns the count of entities imported and any errors.
pub fn import_graph_entities(kg: &MifKnowledgeGraph, graph: &GraphMemory) -> (usize, Vec<String>) {
    let mut imported = 0;
    let mut errors = Vec::new();

    for entity in &kg.entities {
        let labels: Vec<EntityLabel> = entity.types.iter().map(|t| parse_entity_label(t)).collect();

        let node = EntityNode {
            uuid: entity.id,
            name: entity.name.clone(),
            labels: if labels.is_empty() {
                vec![EntityLabel::Concept]
            } else {
                labels
            },
            created_at: entity.created_at,
            last_seen_at: entity.last_seen_at,
            mention_count: 1,
            summary: entity.summary.clone(),
            attributes: entity.attributes.clone(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: true,
        };

        match graph.add_entity(node) {
            Ok(_) => imported += 1,
            Err(e) => errors.push(format!("Entity '{}': {}", entity.name, e)),
        }
    }

    (imported, errors)
}

/// Import graph relationships from MIF document.
///
/// Returns the count of edges imported and any errors.
pub fn import_graph_relationships(
    kg: &MifKnowledgeGraph,
    graph: &GraphMemory,
    vendor_extensions: &HashMap<String, serde_json::Value>,
) -> (usize, Vec<String>) {
    let mut imported = 0;
    let mut errors = Vec::new();

    // Extract shodh edge metadata if available for restoring strength/LTP
    let edge_meta = vendor_extensions
        .get("shodh-memory")
        .and_then(|v| v.get("edge_metadata"))
        .and_then(|v| v.as_object());

    for rel in &kg.relationships {
        let relation_type = parse_relation_type(&rel.relation_type);
        let strength = rel.confidence.unwrap_or(0.5);

        // Restore shodh-specific metadata from vendor extensions
        let (ltp_status, tier, activation_count) = if let Some(meta) = edge_meta {
            if let Some(em) = meta.get(&rel.id.to_string()) {
                let ltp = em
                    .get("ltp_status")
                    .and_then(|v| v.as_str())
                    .map(parse_ltp_status)
                    .unwrap_or_default();
                let tier = em
                    .get("tier")
                    .and_then(|v| v.as_str())
                    .map(parse_edge_tier)
                    .unwrap_or_default();
                let count = em
                    .get("activation_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as u32;
                (ltp, tier, count)
            } else {
                (LtpStatus::None, EdgeTier::L1Working, 1)
            }
        } else {
            (LtpStatus::None, EdgeTier::L1Working, 1)
        };

        let edge = RelationshipEdge {
            uuid: rel.id,
            from_entity: rel.source_entity_id,
            to_entity: rel.target_entity_id,
            relation_type,
            strength,
            created_at: rel.created_at,
            valid_at: rel.valid_at,
            invalidated_at: rel.invalidated_at,
            source_episode_id: None,
            context: rel.context.clone(),
            last_activated: rel.created_at,
            activation_count,
            ltp_status,
            tier,
            activation_timestamps: None,
            entity_confidence: rel.confidence,
        };

        match graph.add_relationship(edge) {
            Ok(_) => imported += 1,
            Err(e) => errors.push(format!("Edge {}: {}", rel.id, e)),
        }
    }

    (imported, errors)
}

/// Validate MIF document version.
pub fn validate_version(doc: &MifDocument) -> Result<()> {
    if !doc.mif_version.starts_with("2.") && !doc.mif_version.starts_with("1.") {
        bail!(
            "Unsupported MIF version: {}. Supported: 1.x, 2.x",
            doc.mif_version
        );
    }
    Ok(())
}

// =============================================================================
// STRING → ENUM PARSERS
// =============================================================================

pub(crate) fn parse_experience_type(s: &str) -> ExperienceType {
    match s.to_lowercase().as_str() {
        "observation" => ExperienceType::Observation,
        "decision" => ExperienceType::Decision,
        "learning" => ExperienceType::Learning,
        "error" => ExperienceType::Error,
        "discovery" => ExperienceType::Discovery,
        "pattern" => ExperienceType::Pattern,
        "context" => ExperienceType::Context,
        "task" => ExperienceType::Task,
        "code_edit" | "codeedit" => ExperienceType::CodeEdit,
        "file_access" | "fileaccess" => ExperienceType::FileAccess,
        "search" => ExperienceType::Search,
        "command" => ExperienceType::Command,
        "conversation" => ExperienceType::Conversation,
        "intention" => ExperienceType::Intention,
        _ => ExperienceType::Observation,
    }
}

fn parse_todo_status(s: &str) -> TodoStatus {
    match s {
        "backlog" => TodoStatus::Backlog,
        "todo" => TodoStatus::Todo,
        "in_progress" => TodoStatus::InProgress,
        "blocked" => TodoStatus::Blocked,
        "done" => TodoStatus::Done,
        "cancelled" => TodoStatus::Cancelled,
        _ => TodoStatus::Todo,
    }
}

fn parse_todo_priority(s: &str) -> TodoPriority {
    match s {
        "urgent" | "!!!" => TodoPriority::Urgent,
        "high" | "!!" => TodoPriority::High,
        "medium" | "!" => TodoPriority::Medium,
        "low" => TodoPriority::Low,
        "none" | "" => TodoPriority::None,
        _ => TodoPriority::Medium,
    }
}

fn parse_comment_type(s: &str) -> crate::memory::types::TodoCommentType {
    use crate::memory::types::TodoCommentType;
    match s {
        "comment" => TodoCommentType::Comment,
        "progress" => TodoCommentType::Progress,
        "resolution" => TodoCommentType::Resolution,
        "activity" => TodoCommentType::Activity,
        _ => TodoCommentType::Comment,
    }
}

fn parse_project_status(s: &str) -> ProjectStatus {
    match s {
        "active" => ProjectStatus::Active,
        "onhold" | "on_hold" => ProjectStatus::OnHold,
        "completed" => ProjectStatus::Completed,
        "archived" => ProjectStatus::Archived,
        _ => ProjectStatus::Active,
    }
}

fn parse_reminder_status(s: &str) -> ProspectiveTaskStatus {
    match s {
        "pending" => ProspectiveTaskStatus::Pending,
        "triggered" => ProspectiveTaskStatus::Triggered,
        "dismissed" => ProspectiveTaskStatus::Dismissed,
        "expired" => ProspectiveTaskStatus::Expired,
        _ => ProspectiveTaskStatus::Pending,
    }
}

fn parse_entity_label(s: &str) -> EntityLabel {
    match s.to_lowercase().as_str() {
        "person" => EntityLabel::Person,
        "organization" => EntityLabel::Organization,
        "location" => EntityLabel::Location,
        "technology" => EntityLabel::Technology,
        "concept" => EntityLabel::Concept,
        "event" => EntityLabel::Event,
        "date" => EntityLabel::Date,
        "product" => EntityLabel::Product,
        "skill" => EntityLabel::Skill,
        "keyword" => EntityLabel::Keyword,
        other => EntityLabel::Other(other.to_string()),
    }
}

pub(crate) fn parse_relation_type(s: &str) -> RelationType {
    match s {
        "works_with" | "workswith" => RelationType::WorksWith,
        "works_at" | "worksat" => RelationType::WorksAt,
        "employed_by" | "employedby" => RelationType::EmployedBy,
        "part_of" | "partof" => RelationType::PartOf,
        "contains" => RelationType::Contains,
        "owned_by" | "ownedby" => RelationType::OwnedBy,
        "located_in" | "locatedin" => RelationType::LocatedIn,
        "located_at" | "locatedat" => RelationType::LocatedAt,
        "uses" => RelationType::Uses,
        "created_by" | "createdby" => RelationType::CreatedBy,
        "developed_by" | "developedby" => RelationType::DevelopedBy,
        "causes" => RelationType::Causes,
        "results_in" | "resultsin" => RelationType::ResultsIn,
        "learned" => RelationType::Learned,
        "knows" => RelationType::Knows,
        "teaches" => RelationType::Teaches,
        "related_to" | "relatedto" => RelationType::RelatedTo,
        "associated_with" | "associatedwith" => RelationType::AssociatedWith,
        "co_retrieved" | "coretrieved" => RelationType::CoRetrieved,
        "co_occurs" | "cooccurs" => RelationType::CoOccurs,
        other => RelationType::Custom(other.to_string()),
    }
}

fn parse_ltp_status(s: &str) -> LtpStatus {
    match s {
        "None" => LtpStatus::None,
        "Weekly" => LtpStatus::Weekly,
        "Full" => LtpStatus::Full,
        s if s.starts_with("Burst") => LtpStatus::Burst {
            detected_at: Utc::now(),
        },
        _ => LtpStatus::None,
    }
}

fn parse_edge_tier(s: &str) -> EdgeTier {
    match s {
        "L1Working" => EdgeTier::L1Working,
        "L2Episodic" => EdgeTier::L2Episodic,
        "L3Semantic" => EdgeTier::L3Semantic,
        _ => EdgeTier::L1Working,
    }
}
