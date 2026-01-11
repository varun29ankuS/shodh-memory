//! Router Configuration - Centralized route definitions
//!
//! This module builds the Axum router using handlers from the submodules.
//! Routes are organized by domain and split into public (no auth) and protected (auth required).

use axum::{
    routing::{delete, get, post, put},
    Router,
};
use std::sync::Arc;

use super::state::MultiUserMemoryManager;
use super::{
    ab_testing, compression, consolidation, crud, facts, files, graph, health, integrations,
    lineage, mif, recall, remember, search, sessions, todos, users, visualization, webhooks,
};

/// Application state type alias
pub type AppState = Arc<MultiUserMemoryManager>;

/// Build the public routes (no authentication required)
///
/// These routes must always be accessible for:
/// - Health checks (Kubernetes probes)
/// - Metrics (Prometheus scraping)
/// - Context status (local Claude Code status line script)
/// - External webhooks (have their own signature verification)
pub fn build_public_routes(state: AppState) -> Router {
    Router::new()
        // =================================================================
        // HEALTH & KUBERNETES PROBES
        // =================================================================
        .route("/health", get(health::health))
        .route("/health/live", get(health::health_live))
        .route("/health/ready", get(health::health_ready))
        .route("/health/index", get(health::health_index))
        // =================================================================
        // METRICS (PROMETHEUS)
        // =================================================================
        .route("/metrics", get(health::metrics_endpoint))
        // =================================================================
        // CONTEXT STATUS (LOCAL SCRIPT - NO AUTH)
        // =================================================================
        .route("/api/context/status", post(health::update_context_status))
        .route("/api/context/status", get(health::get_context_status))
        .route("/api/context_status", get(health::get_context_status)) // TUI alias
        .route("/api/context/sse", get(webhooks::context_status_sse))
        // =================================================================
        // EXTERNAL WEBHOOKS (SIGNATURE VERIFIED INTERNALLY)
        // =================================================================
        .route("/webhook/linear", post(integrations::linear_webhook))
        .route("/webhook/github", post(integrations::github_webhook))
        // =================================================================
        // GRAPH VISUALIZATION (PUBLIC FOR LOCAL DEV)
        // =================================================================
        .route("/graph/view", get(visualization::graph_view))
        .route(
            "/api/graph/data/{user_id}",
            get(visualization::get_graph_data),
        )
        // =================================================================
        // STATE
        // =================================================================
        .with_state(state)
}

/// Build the protected API routes (authentication required)
///
/// These routes require API key authentication and are rate-limited.
/// The auth middleware and rate limiter should be applied by the caller.
pub fn build_protected_routes(state: AppState) -> Router {
    Router::new()
        // =================================================================
        // REMEMBER/RECORD ENDPOINTS
        // =================================================================
        .route("/api/remember", post(remember::remember))
        .route("/api/remember/batch", post(remember::batch_remember))
        .route("/api/batch_remember", post(remember::batch_remember))
        .route("/api/upsert", post(remember::upsert_memory))
        // =================================================================
        // RECALL ENDPOINTS
        // =================================================================
        .route("/api/recall", post(recall::recall))
        .route("/api/recall/tracked", post(recall::recall_tracked))
        .route("/api/recall/tags", post(recall::recall_by_tags))
        .route("/api/recall/by-tags", post(recall::recall_by_tags)) // OpenAPI alias
        .route("/api/recall/date", post(recall::recall_by_date))
        // =================================================================
        // PROACTIVE CONTEXT & RELEVANCE
        // =================================================================
        .route("/api/context_summary", post(recall::context_summary))
        .route("/api/proactive_context", post(recall::proactive_context))
        .route("/api/context", post(recall::proactive_context)) // OpenAPI alias
        .route("/api/relevant", post(recall::surface_relevant))
        .route("/api/reinforce", post(recall::reinforce_feedback))
        // =================================================================
        // MEMORY CRUD OPERATIONS
        // =================================================================
        .route("/api/memory/{memory_id}", get(crud::get_memory))
        .route("/api/memory/{memory_id}", put(crud::update_memory))
        .route("/api/memory/{memory_id}", delete(crud::delete_memory))
        .route("/api/forget/{memory_id}", delete(crud::delete_memory)) // OpenAPI alias
        .route("/api/list/{user_id}", get(crud::list_memories)) // TUI uses this
        .route("/api/memories", post(crud::list_memories_post)) // POST version
        .route("/api/memories/bulk", post(crud::bulk_delete_memories))
        .route("/api/memories/clear", post(crud::clear_all_memories))
        // =================================================================
        // FORGET OPERATIONS
        // =================================================================
        .route("/api/forget/age", post(crud::forget_by_age))
        .route("/api/forget/importance", post(crud::forget_by_importance))
        .route("/api/forget/pattern", post(crud::forget_by_pattern))
        .route("/api/forget/tags", post(crud::forget_by_tags))
        .route("/api/forget/date", post(crud::forget_by_date))
        // =================================================================
        // USER MANAGEMENT
        // =================================================================
        .route("/api/users", get(users::list_users))
        .route("/api/users/{user_id}/stats", get(users::get_user_stats))
        .route("/api/users/{user_id}", delete(users::delete_user))
        .route("/api/stats", get(users::get_stats_query))
        // =================================================================
        // COMPRESSION
        // =================================================================
        .route("/api/memory/compress", post(compression::compress_memory))
        .route(
            "/api/memory/decompress",
            post(compression::decompress_memory),
        )
        .route("/api/storage/stats", get(compression::get_storage_stats))
        // =================================================================
        // ADVANCED SEARCH
        // =================================================================
        .route("/api/search/advanced", post(search::advanced_search))
        // =================================================================
        // STORAGE & INDEX MANAGEMENT
        // =================================================================
        .route("/api/storage/uncompressed", post(mif::get_uncompressed_old))
        .route(
            "/api/index/verify",
            post(consolidation::verify_index_integrity),
        )
        .route(
            "/api/index/repair",
            post(consolidation::repair_vector_index),
        )
        .route("/api/index/rebuild", post(consolidation::rebuild_index))
        .route(
            "/api/storage/cleanup",
            post(consolidation::cleanup_corrupted),
        )
        // =================================================================
        // CONSOLIDATION & BACKUPS
        // =================================================================
        .route(
            "/api/consolidate",
            post(consolidation::consolidate_memories),
        )
        .route(
            "/api/consolidation/report",
            post(consolidation::get_consolidation_report),
        )
        .route(
            "/api/consolidation/events",
            get(consolidation::get_consolidation_events),
        )
        .route("/api/backup/create", post(consolidation::create_backup))
        .route("/api/backup/list", get(consolidation::list_backups))
        .route("/api/backup/verify", post(consolidation::verify_backup))
        .route("/api/backup/purge", post(consolidation::purge_backups))
        // =================================================================
        // FACTS
        // =================================================================
        .route("/api/facts/list", post(facts::list_facts))
        .route("/api/facts/search", post(facts::search_facts))
        .route("/api/facts/by-entity", post(facts::facts_by_entity))
        .route("/api/facts/stats", post(facts::get_facts_stats))
        // =================================================================
        // LINEAGE
        // =================================================================
        .route("/api/lineage/trace", post(lineage::lineage_trace))
        .route("/api/lineage/edges", post(lineage::lineage_list_edges))
        .route("/api/lineage/confirm", post(lineage::lineage_confirm_edge))
        .route("/api/lineage/reject", post(lineage::lineage_reject_edge))
        .route("/api/lineage/link", post(lineage::lineage_add_edge))
        .route("/api/lineage/stats", post(lineage::lineage_stats))
        .route(
            "/api/lineage/branches",
            post(lineage::lineage_list_branches),
        )
        .route("/api/lineage/branch", post(lineage::lineage_create_branch))
        // =================================================================
        // KNOWLEDGE GRAPH (ADVANCED)
        // =================================================================
        .route("/api/graph/{user_id}/stats", get(graph::get_graph_stats))
        .route(
            "/api/graph/{user_id}/universe",
            get(graph::get_memory_universe),
        )
        .route(
            "/api/graph/{user_id}/clear",
            delete(graph::clear_user_graph),
        )
        .route(
            "/api/graph/{user_id}/rebuild",
            post(graph::rebuild_user_graph),
        )
        .route("/api/graph/entity/find", post(graph::find_entity))
        .route("/api/graph/entities/all", post(graph::get_all_entities))
        .route(
            "/api/graph/relationship/invalidate",
            post(graph::invalidate_relationship),
        )
        .route("/api/graph/traverse", post(graph::traverse_graph))
        .route("/api/graph/episode/get", post(graph::get_episode))
        // =================================================================
        // KNOWLEDGE GRAPH (BASIC)
        // =================================================================
        .route("/api/graph/entity/add", post(mif::add_entity))
        .route("/api/graph/relationship/add", post(mif::add_relationship))
        // =================================================================
        // VISUALIZATION
        // =================================================================
        .route("/api/brain/{user_id}", get(visualization::get_brain_state))
        .route(
            "/api/visualization/{user_id}/stats",
            get(visualization::get_visualization_stats),
        )
        .route(
            "/api/visualization/{user_id}/dot",
            get(visualization::get_visualization_dot),
        )
        .route(
            "/api/visualization/build",
            post(visualization::build_visualization),
        )
        // =================================================================
        // TODOS
        // =================================================================
        .route("/api/todos", post(todos::list_todos))
        .route("/api/todos/list", post(todos::list_todos)) // TUI compatibility
        .route("/api/todos/add", post(todos::create_todo))
        .route("/api/todos/update", post(todos::update_todo))
        .route("/api/todos/complete", post(todos::complete_todo))
        .route("/api/todos/delete", post(todos::delete_todo))
        .route("/api/todos/reorder", post(todos::reorder_todo))
        .route("/api/todos/due", post(todos::list_due_todos))
        .route("/api/todos/{todo_id}", get(todos::get_todo))
        .route("/api/todos/{todo_id}", delete(todos::delete_todo)) // TUI uses DELETE
        .route("/api/todos/{todo_id}/update", post(todos::update_todo)) // TUI path style
        .route("/api/todos/{todo_id}/complete", post(todos::complete_todo)) // TUI path style
        .route("/api/todos/{todo_id}/reorder", post(todos::reorder_todo)) // TUI path style
        .route("/api/todos/{todo_id}/subtasks", get(todos::list_subtasks))
        .route(
            "/api/todos/{todo_id}/comments",
            get(todos::list_todo_comments),
        )
        .route(
            "/api/todos/{todo_id}/comments",
            post(todos::add_todo_comment),
        )
        .route(
            "/api/todos/{todo_id}/comments/{comment_id}",
            put(todos::update_todo_comment),
        )
        .route(
            "/api/todos/{todo_id}/comments/{comment_id}",
            delete(todos::delete_todo_comment),
        )
        .route("/api/todos/stats", post(todos::get_todo_stats)) // TUI uses POST
        // =================================================================
        // PROJECTS
        // =================================================================
        .route("/api/projects", get(todos::list_projects))
        .route("/api/projects/add", post(todos::create_project))
        .route("/api/projects/{project_id}", get(todos::get_project))
        .route("/api/projects/update", post(todos::update_project))
        .route("/api/projects/delete", post(todos::delete_project))
        // =================================================================
        // FILE MEMORY / CODEBASE INTEGRATION
        // =================================================================
        .route(
            "/api/projects/{project_id}/files",
            post(files::list_project_files),
        )
        .route(
            "/api/projects/{project_id}/scan",
            post(files::scan_project_codebase),
        )
        .route(
            "/api/projects/{project_id}/index",
            post(files::index_project_codebase),
        )
        .route(
            "/api/projects/{project_id}/files/search",
            post(files::search_project_files),
        )
        .route("/api/files/stats", get(files::get_file_stats))
        // =================================================================
        // REMINDERS
        // =================================================================
        .route("/api/reminders", get(todos::list_reminders))
        .route("/api/reminders/set", post(todos::create_reminder))
        .route("/api/reminders/due", get(todos::get_due_reminders))
        .route("/api/reminders/check", post(todos::check_context_reminders))
        .route("/api/reminders/dismiss", post(todos::dismiss_reminder))
        .route("/api/reminders/delete", post(todos::delete_reminder))
        // =================================================================
        // SESSIONS
        // =================================================================
        .route("/api/sessions", post(sessions::list_sessions))
        .route("/api/sessions/stats", get(sessions::get_session_stats))
        .route("/api/sessions/end", post(sessions::end_session))
        .route("/api/sessions/{session_id}", get(sessions::get_session))
        // =================================================================
        // A/B TESTING
        // =================================================================
        .route("/api/ab/tests", get(ab_testing::list_ab_tests))
        .route("/api/ab/tests", post(ab_testing::create_ab_test))
        .route("/api/ab/tests/{test_id}", get(ab_testing::get_ab_test))
        .route(
            "/api/ab/tests/{test_id}",
            delete(ab_testing::delete_ab_test),
        )
        .route(
            "/api/ab/tests/{test_id}/start",
            post(ab_testing::start_ab_test),
        )
        .route(
            "/api/ab/tests/{test_id}/pause",
            post(ab_testing::pause_ab_test),
        )
        .route(
            "/api/ab/tests/{test_id}/resume",
            post(ab_testing::resume_ab_test),
        )
        .route(
            "/api/ab/tests/{test_id}/complete",
            post(ab_testing::complete_ab_test),
        )
        .route(
            "/api/ab/tests/{test_id}/analyze",
            get(ab_testing::analyze_ab_test),
        )
        .route(
            "/api/ab/tests/{test_id}/impression",
            post(ab_testing::record_ab_impression),
        )
        .route(
            "/api/ab/tests/{test_id}/click",
            post(ab_testing::record_ab_click),
        )
        .route(
            "/api/ab/tests/{test_id}/feedback",
            post(ab_testing::record_ab_feedback),
        )
        .route("/api/ab/summary", get(ab_testing::get_ab_summary))
        // =================================================================
        // EXTERNAL INTEGRATIONS (BULK SYNC)
        // =================================================================
        .route("/api/sync/linear", post(integrations::linear_sync))
        .route("/api/sync/github", post(integrations::github_sync))
        // =================================================================
        // WEBHOOKS & SSE (STREAMING)
        // =================================================================
        .route("/api/context/monitor", get(webhooks::context_monitor_ws))
        .route("/api/events/sse", get(webhooks::memory_events_sse))
        .route("/api/events", get(webhooks::memory_events_sse)) // TUI alias
        .route("/api/stream", get(webhooks::streaming_memory_ws))
        // =================================================================
        // SEARCH & MIF (Memory Interchange Format)
        // =================================================================
        .route("/api/search/multimodal", post(mif::multimodal_search))
        .route("/api/search/robotics", post(mif::robotics_search))
        .route("/api/export/mif", post(mif::export_mif))
        .route("/api/import/mif", post(mif::import_mif))
        // =================================================================
        // STATE
        // =================================================================
        .with_state(state)
}

/// Build the complete router with both public and protected routes
///
/// Note: This function does NOT apply auth middleware or rate limiting.
/// The caller (main.rs) should apply those layers as needed.
pub fn build_router(state: AppState) -> Router {
    let public = build_public_routes(state.clone());
    let protected = build_protected_routes(state);

    Router::new().merge(public).merge(protected)
}
