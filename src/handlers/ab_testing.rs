//! A/B Testing Handlers
//!
//! Handlers for A/B test management, metrics recording, and analysis.

use axum::{
    extract::{Path, State},
    response::Json,
};
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use crate::ab_testing;
use crate::errors::{AppError, ValidationErrorExt};
use crate::relevance;
use crate::validation;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

fn default_traffic_split() -> f32 {
    0.5
}

fn default_min_impressions() -> u64 {
    100
}

/// Request to create a new A/B test
#[derive(Debug, Deserialize)]
pub struct CreateABTestRequest {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub control_weights: Option<relevance::LearnedWeights>,
    #[serde(default)]
    pub treatment_weights: Option<relevance::LearnedWeights>,
    #[serde(default = "default_traffic_split")]
    pub traffic_split: f32,
    #[serde(default = "default_min_impressions")]
    pub min_impressions: u64,
    #[serde(default)]
    pub max_duration_hours: Option<u64>,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Response for A/B test operations
#[derive(Debug, Serialize)]
pub struct ABTestResponse {
    pub success: bool,
    pub test_id: Option<String>,
    pub message: String,
}

/// Request to record an impression
#[derive(Debug, Deserialize)]
pub struct RecordImpressionRequest {
    pub user_id: String,
    #[serde(default)]
    pub relevance_score: Option<f64>,
    #[serde(default)]
    pub latency_us: Option<u64>,
}

/// Request to record a click
#[derive(Debug, Deserialize)]
pub struct RecordClickRequest {
    pub user_id: String,
    pub memory_id: uuid::Uuid,
}

/// Request to record feedback
#[derive(Debug, Deserialize)]
pub struct RecordFeedbackRequest {
    pub user_id: String,
    pub positive: bool,
}

/// GET /api/ab/tests - List all A/B tests
pub async fn list_ab_tests(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let tests = state.ab_test_manager.list_tests();
    let summary = state.ab_test_manager.summary();

    Ok(Json(serde_json::json!({
        "success": true,
        "tests": tests.iter().map(|t| serde_json::json!({
            "id": t.id,
            "name": t.config.name,
            "description": t.config.description,
            "status": format!("{:?}", t.status),
            "traffic_split": t.config.traffic_split,
            "control_impressions": t.control_metrics.impressions,
            "treatment_impressions": t.treatment_metrics.impressions,
            "created_at": t.created_at.to_rfc3339(),
        })).collect::<Vec<_>>(),
        "summary": {
            "total_active": summary.total_active,
            "draft": summary.draft,
            "running": summary.running,
            "paused": summary.paused,
            "completed": summary.completed,
            "archived": summary.archived,
        }
    })))
}

/// POST /api/ab/tests - Create a new A/B test
pub async fn create_ab_test(
    State(state): State<AppState>,
    Json(req): Json<CreateABTestRequest>,
) -> Result<Json<ABTestResponse>, AppError> {
    let mut builder = ab_testing::ABTest::builder(&req.name)
        .with_traffic_split(req.traffic_split)
        .with_min_impressions(req.min_impressions);

    if let Some(desc) = req.description {
        builder = builder.with_description(&desc);
    }

    if let Some(control) = req.control_weights {
        builder = builder.with_control(control);
    }

    if let Some(treatment) = req.treatment_weights {
        builder = builder.with_treatment(treatment);
    }

    if let Some(hours) = req.max_duration_hours {
        builder = builder.with_max_duration_hours(hours);
    }

    if !req.tags.is_empty() {
        builder = builder.with_tags(req.tags);
    }

    let test = builder.build();

    match state.ab_test_manager.create_test(test) {
        Ok(test_id) => Ok(Json(ABTestResponse {
            success: true,
            test_id: Some(test_id),
            message: "A/B test created successfully".to_string(),
        })),
        Err(e) => Ok(Json(ABTestResponse {
            success: false,
            test_id: None,
            message: format!("Failed to create test: {}", e),
        })),
    }
}

/// GET /api/ab/tests/{test_id} - Get a specific A/B test
pub async fn get_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    match state.ab_test_manager.get_test(&test_id) {
        Some(test) => Ok(Json(serde_json::json!({
            "success": true,
            "test": {
                "id": test.id,
                "name": test.config.name,
                "description": test.config.description,
                "status": format!("{:?}", test.status),
                "traffic_split": test.config.traffic_split,
                "min_impressions": test.config.min_impressions,
                "max_duration_hours": test.config.max_duration_hours,
                "control_weights": test.config.control_weights,
                "treatment_weights": test.config.treatment_weights,
                "control_metrics": {
                    "impressions": test.control_metrics.impressions,
                    "clicks": test.control_metrics.clicks,
                    "ctr": if test.control_metrics.impressions > 0 {
                        test.control_metrics.clicks as f64 / test.control_metrics.impressions as f64
                    } else { 0.0 },
                    "positive_feedback": test.control_metrics.positive_feedback,
                    "negative_feedback": test.control_metrics.negative_feedback,
                },
                "treatment_metrics": {
                    "impressions": test.treatment_metrics.impressions,
                    "clicks": test.treatment_metrics.clicks,
                    "ctr": if test.treatment_metrics.impressions > 0 {
                        test.treatment_metrics.clicks as f64 / test.treatment_metrics.impressions as f64
                    } else { 0.0 },
                    "positive_feedback": test.treatment_metrics.positive_feedback,
                    "negative_feedback": test.treatment_metrics.negative_feedback,
                },
                "created_at": test.created_at.to_rfc3339(),
                "started_at": test.started_at.map(|t| t.to_rfc3339()),
                "completed_at": test.completed_at.map(|t| t.to_rfc3339()),
            }
        }))),
        None => Ok(Json(serde_json::json!({
            "success": false,
            "message": format!("Test not found: {}", test_id)
        }))),
    }
}

/// DELETE /api/ab/tests/{test_id} - Delete an A/B test
pub async fn delete_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<ABTestResponse>, AppError> {
    match state.ab_test_manager.delete_test(&test_id) {
        Ok(()) => Ok(Json(ABTestResponse {
            success: true,
            test_id: Some(test_id),
            message: "Test deleted successfully".to_string(),
        })),
        Err(e) => Ok(Json(ABTestResponse {
            success: false,
            test_id: None,
            message: format!("Failed to delete test: {}", e),
        })),
    }
}

/// POST /api/ab/tests/{test_id}/start - Start an A/B test
pub async fn start_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<ABTestResponse>, AppError> {
    match state.ab_test_manager.start_test(&test_id) {
        Ok(()) => Ok(Json(ABTestResponse {
            success: true,
            test_id: Some(test_id),
            message: "Test started successfully".to_string(),
        })),
        Err(e) => Ok(Json(ABTestResponse {
            success: false,
            test_id: None,
            message: format!("Failed to start test: {}", e),
        })),
    }
}

/// POST /api/ab/tests/{test_id}/pause - Pause an A/B test
pub async fn pause_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<ABTestResponse>, AppError> {
    match state.ab_test_manager.pause_test(&test_id) {
        Ok(()) => Ok(Json(ABTestResponse {
            success: true,
            test_id: Some(test_id),
            message: "Test paused successfully".to_string(),
        })),
        Err(e) => Ok(Json(ABTestResponse {
            success: false,
            test_id: None,
            message: format!("Failed to pause test: {}", e),
        })),
    }
}

/// POST /api/ab/tests/{test_id}/resume - Resume a paused A/B test
pub async fn resume_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<ABTestResponse>, AppError> {
    match state.ab_test_manager.resume_test(&test_id) {
        Ok(()) => Ok(Json(ABTestResponse {
            success: true,
            test_id: Some(test_id),
            message: "Test resumed successfully".to_string(),
        })),
        Err(e) => Ok(Json(ABTestResponse {
            success: false,
            test_id: None,
            message: format!("Failed to resume test: {}", e),
        })),
    }
}

/// POST /api/ab/tests/{test_id}/complete - Complete an A/B test and get results
pub async fn complete_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    match state.ab_test_manager.complete_test(&test_id) {
        Ok(results) => Ok(Json(serde_json::json!({
            "success": true,
            "test_id": test_id,
            "results": {
                "is_significant": results.is_significant,
                "confidence_level": results.confidence_level,
                "chi_squared": results.chi_squared,
                "p_value": results.p_value,
                "winner": results.winner.map(|w| format!("{:?}", w)),
                "relative_improvement": results.relative_improvement,
                "control_ctr": results.control_ctr,
                "treatment_ctr": results.treatment_ctr,
                "confidence_interval": results.confidence_interval,
                "recommendations": results.recommendations,
            }
        }))),
        Err(e) => Ok(Json(serde_json::json!({
            "success": false,
            "message": format!("Failed to complete test: {}", e)
        }))),
    }
}

/// GET /api/ab/tests/{test_id}/analyze - Analyze an A/B test without completing it
pub async fn analyze_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    match state.ab_test_manager.analyze_test(&test_id) {
        Ok(results) => Ok(Json(serde_json::json!({
            "success": true,
            "test_id": test_id,
            "analysis": {
                "is_significant": results.is_significant,
                "confidence_level": results.confidence_level,
                "chi_squared": results.chi_squared,
                "p_value": results.p_value,
                "winner": results.winner.map(|w| format!("{:?}", w)),
                "relative_improvement": results.relative_improvement,
                "control_ctr": results.control_ctr,
                "treatment_ctr": results.treatment_ctr,
                "confidence_interval": results.confidence_interval,
                "recommendations": results.recommendations,
            }
        }))),
        Err(e) => Ok(Json(serde_json::json!({
            "success": false,
            "message": format!("Failed to analyze test: {}", e)
        }))),
    }
}

/// POST /api/ab/tests/{test_id}/impression - Record an impression for an A/B test
pub async fn record_ab_impression(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
    Json(req): Json<RecordImpressionRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let relevance_score = req.relevance_score.unwrap_or(0.0);
    let latency_us = req.latency_us.unwrap_or(0);

    match state.ab_test_manager.record_impression(
        &test_id,
        &req.user_id,
        relevance_score,
        latency_us,
    ) {
        Ok(()) => {
            let variant = state
                .ab_test_manager
                .get_variant(&test_id, &req.user_id)
                .ok();
            Ok(Json(serde_json::json!({
                "success": true,
                "variant": variant.map(|v| format!("{:?}", v)),
            })))
        }
        Err(e) => Ok(Json(serde_json::json!({
            "success": false,
            "message": format!("Failed to record impression: {}", e)
        }))),
    }
}

/// POST /api/ab/tests/{test_id}/click - Record a click for an A/B test
pub async fn record_ab_click(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
    Json(req): Json<RecordClickRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    match state
        .ab_test_manager
        .record_click(&test_id, &req.user_id, req.memory_id)
    {
        Ok(()) => Ok(Json(serde_json::json!({
            "success": true,
        }))),
        Err(e) => Ok(Json(serde_json::json!({
            "success": false,
            "message": format!("Failed to record click: {}", e)
        }))),
    }
}

/// POST /api/ab/tests/{test_id}/feedback - Record feedback for an A/B test
pub async fn record_ab_feedback(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
    Json(req): Json<RecordFeedbackRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    match state
        .ab_test_manager
        .record_feedback(&test_id, &req.user_id, req.positive)
    {
        Ok(()) => Ok(Json(serde_json::json!({
            "success": true,
        }))),
        Err(e) => Ok(Json(serde_json::json!({
            "success": false,
            "message": format!("Failed to record feedback: {}", e)
        }))),
    }
}

/// GET /api/ab/summary - Get summary of all A/B tests
pub async fn get_ab_summary(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let summary = state.ab_test_manager.summary();
    let expired = state.ab_test_manager.check_expired_tests();

    Ok(Json(serde_json::json!({
        "success": true,
        "summary": {
            "total_active": summary.total_active,
            "draft": summary.draft,
            "running": summary.running,
            "paused": summary.paused,
            "completed": summary.completed,
            "archived": summary.archived,
        },
        "expired_tests": expired,
    })))
}
