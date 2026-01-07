//! External Integration Handlers
//!
//! Handlers for Linear and GitHub webhook integrations and bulk sync.

use axum::{
    extract::State,
    http::HeaderMap,
    response::Json,
};

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::integrations;
use crate::memory::{self, Experience, ExperienceType};
use crate::validation;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

/// POST /webhook/linear - Linear webhook receiver
#[tracing::instrument(skip(state, body, headers))]
pub async fn linear_webhook(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<serde_json::Value>, AppError> {
    use integrations::linear::LinearWebhook;

    let signing_secret = std::env::var("LINEAR_WEBHOOK_SECRET").ok();
    let webhook = LinearWebhook::new(signing_secret);

    // Verify signature if present
    if let Some(signature) = headers
        .get("linear-signature")
        .and_then(|h| h.to_str().ok())
    {
        if !webhook
            .verify_signature(&body, signature)
            .map_err(AppError::Internal)?
        {
            return Err(AppError::InvalidInput {
                field: "signature".to_string(),
                reason: "Invalid webhook signature".to_string(),
            });
        }
    }

    let payload = webhook
        .parse_payload(&body)
        .map_err(AppError::Internal)?;

    if payload.entity_type != "Issue" {
        return Ok(Json(serde_json::json!({
            "status": "ignored",
            "reason": "Only Issue events are processed"
        })));
    }

    if payload.action == "remove" {
        return Ok(Json(serde_json::json!({
            "status": "acknowledged",
            "action": "remove"
        })));
    }

    let external_id = match &payload.data.identifier {
        Some(id) => format!("linear:{}", id),
        None => format!("linear:{}", payload.data.id),
    };

    let content = LinearWebhook::issue_to_content(&payload.data);
    let tags = LinearWebhook::issue_to_tags(&payload.data);
    let change_type = LinearWebhook::determine_change_type(&payload.action, &payload.data);

    let user_id =
        std::env::var("LINEAR_SYNC_USER_ID").unwrap_or_else(|_| "linear-sync".to_string());

    let experience = Experience {
        content: content.clone(),
        experience_type: ExperienceType::Task,
        entities: tags.clone(),
        ..Default::default()
    };

    let change_type_enum = match change_type.as_str() {
        "created" => memory::types::ChangeType::Created,
        "status_changed" => memory::types::ChangeType::StatusChanged,
        "tags_updated" => memory::types::ChangeType::TagsUpdated,
        _ => memory::types::ChangeType::ContentUpdated,
    };

    let memory_system = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let (memory_id, was_update) = {
        let memory = memory_system.clone();
        let ext_id = external_id.clone();
        let exp = experience.clone();
        let ct = change_type_enum;
        let actor_name = payload
            .actor
            .as_ref()
            .and_then(|a| a.name.clone())
            .unwrap_or_else(|| "linear-webhook".to_string());

        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.upsert(ext_id, exp, ct, Some(actor_name), None)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    Ok(Json(serde_json::json!({
        "status": "success",
        "id": memory_id.0.to_string(),
        "external_id": external_id,
        "was_update": was_update,
        "action": payload.action
    })))
}

/// POST /api/sync/linear - Bulk sync Linear issues
#[tracing::instrument(skip(state, req), fields(user_id = %req.user_id))]
pub async fn linear_sync(
    State(state): State<AppState>,
    Json(req): Json<integrations::linear::LinearSyncRequest>,
) -> Result<Json<integrations::linear::LinearSyncResponse>, AppError> {
    use integrations::linear::{LinearClient, LinearSyncResponse, LinearWebhook};

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.api_key.is_empty() {
        return Err(AppError::InvalidInput {
            field: "api_key".to_string(),
            reason: "Linear API key is required".to_string(),
        });
    }

    let client = LinearClient::new(req.api_key.clone());

    let issues = client
        .fetch_issues(
            req.team_id.as_deref(),
            req.updated_after.as_deref(),
            req.limit,
        )
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to fetch Linear issues: {}", e)))?;

    let total = issues.len();
    let mut created_count = 0;
    let mut updated_count = 0;
    let mut error_count = 0;
    let mut errors = Vec::new();

    let memory_system = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    for issue in issues {
        let external_id = match &issue.identifier {
            Some(id) => format!("linear:{}", id),
            None => format!("linear:{}", issue.id),
        };

        let content = LinearWebhook::issue_to_content(&issue);
        let tags = LinearWebhook::issue_to_tags(&issue);

        let experience = Experience {
            content,
            experience_type: ExperienceType::Task,
            entities: tags,
            ..Default::default()
        };

        let result = {
            let memory = memory_system.clone();
            let ext_id = external_id.clone();
            let exp = experience;

            tokio::task::spawn_blocking(move || {
                let memory_guard = memory.read();
                memory_guard.upsert(
                    ext_id,
                    exp,
                    memory::types::ChangeType::ContentUpdated,
                    Some("linear-bulk-sync".to_string()),
                    None,
                )
            })
            .await
        };

        match result {
            Ok(Ok((_, was_update))) => {
                if was_update {
                    updated_count += 1;
                } else {
                    created_count += 1;
                }
            }
            Ok(Err(e)) => {
                error_count += 1;
                errors.push(format!("{}: {}", external_id, e));
            }
            Err(e) => {
                error_count += 1;
                errors.push(format!("{}: Task panicked: {}", external_id, e));
            }
        }
    }

    Ok(Json(LinearSyncResponse {
        synced_count: total,
        created_count,
        updated_count,
        error_count,
        errors,
    }))
}

/// POST /webhook/github - GitHub webhook receiver
#[tracing::instrument(skip(state, body, headers))]
pub async fn github_webhook(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<serde_json::Value>, AppError> {
    use integrations::github::GitHubWebhook;

    let webhook_secret = std::env::var("GITHUB_WEBHOOK_SECRET").ok();
    let webhook = GitHubWebhook::new(webhook_secret);

    if let Some(signature) = headers
        .get("x-hub-signature-256")
        .and_then(|h| h.to_str().ok())
    {
        if !webhook
            .verify_signature(&body, signature)
            .map_err(AppError::Internal)?
        {
            return Err(AppError::InvalidInput {
                field: "signature".to_string(),
                reason: "Invalid webhook signature".to_string(),
            });
        }
    }

    let event_type = headers
        .get("x-github-event")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown");

    if event_type != "issues" && event_type != "pull_request" {
        return Ok(Json(serde_json::json!({
            "status": "ignored",
            "reason": format!("Only issues and pull_request events are processed, got: {}", event_type)
        })));
    }

    let payload = webhook
        .parse_payload(&body)
        .map_err(AppError::Internal)?;

    let user_id =
        std::env::var("GITHUB_SYNC_USER_ID").unwrap_or_else(|_| "github-sync".to_string());

    let (external_id, content, tags, change_type) = if let Some(issue) = &payload.issue {
        let ext_id = GitHubWebhook::issue_external_id(&payload.repository, issue.number);
        let content = GitHubWebhook::issue_to_content(issue, &payload.repository);
        let tags = GitHubWebhook::issue_to_tags(issue, &payload.repository);
        let ct = GitHubWebhook::determine_change_type(&payload.action, false);
        (ext_id, content, tags, ct)
    } else if let Some(pr) = &payload.pull_request {
        let ext_id = GitHubWebhook::pr_external_id(&payload.repository, pr.number);
        let content = GitHubWebhook::pr_to_content(pr, &payload.repository);
        let tags = GitHubWebhook::pr_to_tags(pr, &payload.repository);
        let ct = GitHubWebhook::determine_change_type(&payload.action, true);
        (ext_id, content, tags, ct)
    } else {
        return Ok(Json(serde_json::json!({
            "status": "ignored",
            "reason": "No issue or pull_request data in payload"
        })));
    };

    let experience = Experience {
        content: content.clone(),
        experience_type: ExperienceType::Task,
        entities: tags.clone(),
        ..Default::default()
    };

    let change_type_enum = match change_type.as_str() {
        "created" => memory::types::ChangeType::Created,
        "status_changed" => memory::types::ChangeType::StatusChanged,
        "tags_updated" => memory::types::ChangeType::TagsUpdated,
        _ => memory::types::ChangeType::ContentUpdated,
    };

    let memory_system = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let (memory_id, was_update) = {
        let memory = memory_system.clone();
        let ext_id = external_id.clone();
        let exp = experience.clone();
        let ct = change_type_enum;
        let actor_name = payload
            .sender
            .as_ref()
            .map(|s| s.login.clone())
            .unwrap_or_else(|| "github-webhook".to_string());

        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.upsert(ext_id, exp, ct, Some(actor_name), None)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    Ok(Json(serde_json::json!({
        "status": "success",
        "id": memory_id.0.to_string(),
        "external_id": external_id,
        "was_update": was_update,
        "action": payload.action,
        "event_type": event_type
    })))
}

/// POST /api/sync/github - Bulk sync GitHub issues and PRs
#[tracing::instrument(skip(state, req), fields(user_id = %req.user_id))]
pub async fn github_sync(
    State(state): State<AppState>,
    Json(req): Json<integrations::github::GitHubSyncRequest>,
) -> Result<Json<integrations::github::GitHubSyncResponse>, AppError> {
    use integrations::github::{GitHubClient, GitHubSyncResponse, GitHubWebhook};

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.token.is_empty() {
        return Err(AppError::InvalidInput {
            field: "token".to_string(),
            reason: "GitHub token is required".to_string(),
        });
    }

    let client = GitHubClient::new(req.token.clone());

    let repo_info = client
        .get_repository(&req.owner, &req.repo)
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to get repository: {}", e)))?;

    let mut issues_synced = 0;
    let mut prs_synced = 0;
    let mut created_count = 0;
    let mut updated_count = 0;
    let mut error_count = 0;
    let mut errors = Vec::new();

    let memory_system = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // Sync issues
    if req.sync_issues {
        let issues = client
            .fetch_issues(
                &req.owner,
                &req.repo,
                &req.state,
                req.limit,
            )
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to fetch issues: {}", e)))?;

        for issue in issues {
            let external_id = GitHubWebhook::issue_external_id(&repo_info, issue.number);
            let content = GitHubWebhook::issue_to_content(&issue, &repo_info);
            let tags = GitHubWebhook::issue_to_tags(&issue, &repo_info);

            let experience = Experience {
                content,
                experience_type: ExperienceType::Task,
                entities: tags,
                ..Default::default()
            };

            let result = {
                let memory = memory_system.clone();
                let ext_id = external_id.clone();
                let exp = experience;

                tokio::task::spawn_blocking(move || {
                    let memory_guard = memory.read();
                    memory_guard.upsert(
                        ext_id,
                        exp,
                        memory::types::ChangeType::ContentUpdated,
                        Some("github-bulk-sync".to_string()),
                        None,
                    )
                })
                .await
            };

            match result {
                Ok(Ok((_, was_update))) => {
                    issues_synced += 1;
                    if was_update {
                        updated_count += 1;
                    } else {
                        created_count += 1;
                    }
                }
                Ok(Err(e)) => {
                    error_count += 1;
                    errors.push(format!("{}: {}", external_id, e));
                }
                Err(e) => {
                    error_count += 1;
                    errors.push(format!("{}: {}", external_id, e));
                }
            }
        }
    }

    // Sync PRs
    if req.sync_prs {
        let prs = client
            .fetch_pull_requests(
                &req.owner,
                &req.repo,
                &req.state,
                req.limit,
            )
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to fetch PRs: {}", e)))?;

        for pr in prs {
            let external_id = GitHubWebhook::pr_external_id(&repo_info, pr.number);
            let content = GitHubWebhook::pr_to_content(&pr, &repo_info);
            let tags = GitHubWebhook::pr_to_tags(&pr, &repo_info);

            let experience = Experience {
                content,
                experience_type: ExperienceType::Task,
                entities: tags,
                ..Default::default()
            };

            let result = {
                let memory = memory_system.clone();
                let ext_id = external_id.clone();
                let exp = experience;

                tokio::task::spawn_blocking(move || {
                    let memory_guard = memory.read();
                    memory_guard.upsert(
                        ext_id,
                        exp,
                        memory::types::ChangeType::ContentUpdated,
                        Some("github-bulk-sync".to_string()),
                        None,
                    )
                })
                .await
            };

            match result {
                Ok(Ok((_, was_update))) => {
                    prs_synced += 1;
                    if was_update {
                        updated_count += 1;
                    } else {
                        created_count += 1;
                    }
                }
                Ok(Err(e)) => {
                    error_count += 1;
                    errors.push(format!("{}: {}", external_id, e));
                }
                Err(e) => {
                    error_count += 1;
                    errors.push(format!("{}: {}", external_id, e));
                }
            }
        }
    }

    Ok(Json(GitHubSyncResponse {
        synced_count: issues_synced + prs_synced,
        issues_synced,
        prs_synced,
        commits_synced: 0,
        created_count,
        updated_count,
        error_count,
        errors,
    }))
}
