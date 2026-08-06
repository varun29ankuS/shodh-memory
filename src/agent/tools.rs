//! First-class shodh-memory tools exposed to the agent turn loop.
//!
//! The whole point of wiring these in as `rig_core::completion::ToolDefinition`s (registered
//! on every `CompletionRequest` the turn loop sends — see [`crate::agent::turn_loop`]) rather
//! than, say, describing them in the system prompt as free text, is that memory operations then
//! render as *structured* tool calls in the AG-UI event stream (`TOOL_CALL_START` /
//! `TOOL_CALL_ARGS` / `TOOL_CALL_END` / `TOOL_CALL_RESULT`) instead of opaque prose the model
//! might paraphrase, hallucinate the shape of, or a UI can't do anything with.
//!
//! Two tools are registered:
//!
//! - [`TOOL_MEMORY_RECALL`] runs the exact same [`crate::memory::MemorySystem::recall`]
//!   pipeline `src/handlers/recall.rs` calls for `POST /api/recall` (vector + graph + BM25 +
//!   rerank fusion) — this module does not reimplement or alter retrieval in any way, it is a
//!   thin argument-parsing/JSON-shaping wrapper.
//! - [`TOOL_MEMORY_REMEMBER`] calls [`crate::memory::MemorySystem::remember`] directly (the
//!   same light path `src/agent/conversation_memory.rs` uses for turn persistence — not the
//!   full HTTP `remember` handler's NER/knowledge-graph ingestion pipeline; see that module's
//!   docs for why that split is deliberate).
//!
//! Tool dispatch is synchronous (both underlying calls are blocking RocksDB/CPU work); callers
//! run [`execute`] inside `tokio::task::spawn_blocking`, matching how every other handler in
//! this crate touches `MemorySystem`.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use parking_lot::RwLock;
use serde::Deserialize;
use serde_json::json;

use rig_core::completion::ToolDefinition;

use crate::memory::types::{Experience, ExperienceType, Query};
use crate::memory::MemorySystem;

pub const TOOL_MEMORY_RECALL: &str = "memory_recall";
pub const TOOL_MEMORY_REMEMBER: &str = "memory_remember";

const MAX_RECALL_RESULTS: usize = 20;
const DEFAULT_RECALL_RESULTS: usize = 5;

/// The tool set registered on every turn-loop completion request.
pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: TOOL_MEMORY_RECALL.to_string(),
            description: "Search the user's durable memory for relevant prior facts, \
                           conversations, and context using shodh-memory's vector + graph + \
                           BM25 retrieval pipeline. Call this before answering questions that \
                           may depend on something learned in a previous session."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of memories to return (1-20).",
                        "minimum": 1,
                        "maximum": MAX_RECALL_RESULTS
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        },
        ToolDefinition {
            name: TOOL_MEMORY_REMEMBER.to_string(),
            description: "Persist a durable memory (a fact, preference, decision, or outcome) \
                           worth recalling in future sessions. Use sparingly, only for \
                           information actually worth remembering long-term — not for routine \
                           conversational filler."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact or note to remember, in plain language."
                    },
                    "importance": {
                        "type": "number",
                        "description": "Optional importance override, 0.0-1.0.",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["content"],
                "additionalProperties": false
            }),
        },
    ]
}

#[derive(Debug, Deserialize)]
struct RecallArgs {
    query: String,
    #[serde(default)]
    max_results: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RememberArgs {
    content: String,
    #[serde(default)]
    importance: Option<f32>,
}

/// Execute a registered tool call by name and return its result as JSON.
///
/// Never panics on malformed model-supplied arguments: a bad schema or an empty required field
/// becomes an `{"error": "..."}` JSON result the model can see and recover from in the next
/// turn, rather than aborting the whole run over one bad tool call.
pub fn execute(
    memory: &Arc<RwLock<MemorySystem>>,
    user_id: &str,
    tool_name: &str,
    arguments: &serde_json::Value,
) -> serde_json::Value {
    let result = match tool_name {
        TOOL_MEMORY_RECALL => execute_recall(memory, user_id, arguments),
        TOOL_MEMORY_REMEMBER => execute_remember(memory, user_id, arguments),
        other => Err(anyhow::anyhow!("unknown tool '{other}'")),
    };

    match result {
        Ok(value) => value,
        Err(e) => json!({ "error": e.to_string() }),
    }
}

fn execute_recall(
    memory: &Arc<RwLock<MemorySystem>>,
    user_id: &str,
    arguments: &serde_json::Value,
) -> Result<serde_json::Value> {
    let args: RecallArgs = serde_json::from_value(arguments.clone())
        .context("memory_recall arguments did not match the expected schema")?;
    if args.query.trim().is_empty() {
        bail!("memory_recall 'query' must not be empty");
    }
    let max_results = args
        .max_results
        .unwrap_or(DEFAULT_RECALL_RESULTS)
        .clamp(1, MAX_RECALL_RESULTS);

    let query = Query {
        user_id: Some(user_id.to_string()),
        query_text: Some(args.query),
        max_results,
        ..Default::default()
    };

    let memory_guard = memory.read();
    let results = memory_guard
        .recall(&query)
        .context("memory_recall pipeline failed")?;

    let results_json: Vec<serde_json::Value> = results
        .iter()
        .map(|m| {
            json!({
                "id": m.id.0.to_string(),
                "content": m.experience.content,
                "score": m.score,
                "experience_type": format!("{:?}", m.experience.experience_type),
                "created_at": m.created_at.to_rfc3339(),
            })
        })
        .collect();

    Ok(json!({
        "count": results_json.len(),
        "results": results_json,
    }))
}

fn execute_remember(
    memory: &Arc<RwLock<MemorySystem>>,
    user_id: &str,
    arguments: &serde_json::Value,
) -> Result<serde_json::Value> {
    let args: RememberArgs = serde_json::from_value(arguments.clone())
        .context("memory_remember arguments did not match the expected schema")?;
    if args.content.trim().is_empty() {
        bail!("memory_remember 'content' must not be empty");
    }

    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), "agent-tool-call".to_string());

    let experience = Experience {
        experience_type: ExperienceType::Learning,
        content: args.content,
        metadata,
        importance_override: args.importance,
        tags: vec!["agent-tool-remember".to_string()],
        ..Default::default()
    };

    let memory_guard = memory.read();
    let memory_id = memory_guard
        .remember(experience, None)
        .context("memory_remember failed to store the experience")?;

    Ok(json!({
        "memory_id": memory_id.0.to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryConfig;

    fn setup() -> (Arc<RwLock<MemorySystem>>, tempfile::TempDir) {
        let temp_dir = tempfile::TempDir::new().expect("temp dir");
        let config = MemoryConfig {
            storage_path: temp_dir.path().to_path_buf(),
            working_memory_size: 50,
            session_memory_size_mb: 50,
            max_heap_per_user_mb: 200,
            auto_compress: false,
            compression_age_days: 1,
            importance_threshold: 0.0,
        };
        let system = MemorySystem::new(config, None).expect("memory system");
        (Arc::new(RwLock::new(system)), temp_dir)
    }

    #[test]
    fn tool_definitions_cover_recall_and_remember_by_name() {
        let defs = tool_definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&TOOL_MEMORY_RECALL));
        assert!(names.contains(&TOOL_MEMORY_REMEMBER));
    }

    #[test]
    fn remember_then_recall_round_trips_through_the_tool_layer() {
        let (memory, _temp_dir) = setup();

        let remember_result = execute(
            &memory,
            "test-user",
            TOOL_MEMORY_REMEMBER,
            &json!({ "content": "The deployment key rotates every 90 days." }),
        );
        assert!(
            remember_result.get("memory_id").is_some(),
            "expected a memory_id in the result, got {remember_result}"
        );
        assert!(remember_result.get("error").is_none());

        let recall_result = execute(
            &memory,
            "test-user",
            TOOL_MEMORY_RECALL,
            &json!({ "query": "deployment key rotation", "max_results": 5 }),
        );
        let results = recall_result
            .get("results")
            .and_then(|v| v.as_array())
            .expect("recall result must have a results array");
        assert!(
            results
                .iter()
                .any(|r| r["content"].as_str() == Some("The deployment key rotates every 90 days.")),
            "expected the remembered content back from recall, got {recall_result}"
        );
    }

    #[test]
    fn recall_with_empty_query_returns_a_structured_error_not_a_panic() {
        let (memory, _temp_dir) = setup();
        let result = execute(&memory, "test-user", TOOL_MEMORY_RECALL, &json!({ "query": "" }));
        assert!(result.get("error").is_some(), "expected error field, got {result}");
    }

    #[test]
    fn unknown_tool_name_returns_a_structured_error_not_a_panic() {
        let (memory, _temp_dir) = setup();
        let result = execute(&memory, "test-user", "not_a_real_tool", &json!({}));
        assert!(result.get("error").is_some(), "expected error field, got {result}");
    }

    #[test]
    fn malformed_arguments_return_a_structured_error_not_a_panic() {
        let (memory, _temp_dir) = setup();
        let result = execute(&memory, "test-user", TOOL_MEMORY_RECALL, &json!({ "query": 12345 }));
        assert!(result.get("error").is_some(), "expected error field, got {result}");
    }
}
