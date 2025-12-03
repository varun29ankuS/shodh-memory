//! Enterprise-grade error handling with structured error types and codes
//! Provides detailed error information for debugging and client error handling

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Serialize, Deserialize};
use std::fmt;

/// Structured error response for API clients
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Machine-readable error code
    pub code: String,

    /// Human-readable error message
    pub message: String,

    /// Additional error context
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,

    /// Request ID for tracing (enterprise feature)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

/// Application error types with proper categorization
#[derive(Debug)]
pub enum AppError {
    // Validation Errors (400)
    InvalidInput { field: String, reason: String },
    InvalidUserId(String),
    InvalidMemoryId(String),
    InvalidEmbeddings(String),
    ContentTooLarge { size: usize, max: usize },

    // Resource Limit Errors (429)
    ResourceLimit { resource: String, current: usize, limit: usize },

    // Not Found Errors (404)
    MemoryNotFound(String),
    UserNotFound(String),

    // Conflict Errors (409)
    MemoryAlreadyExists(String),

    // Internal Errors (500)
    StorageError(String),
    DatabaseError(String),
    SerializationError(String),
    ConcurrencyError(String),

    // Lock failures (500) - non-panicking lock handling
    LockPoisoned { resource: String, details: String },
    LockAcquisitionFailed { resource: String, reason: String },

    // Service Errors (503)
    ServiceUnavailable(String),

    // Generic wrapper for external errors
    Internal(anyhow::Error),
}

impl AppError {
    /// Create a lock poisoned error from a PoisonError
    pub fn from_lock_poison<T>(resource: &str, _err: std::sync::PoisonError<T>) -> Self {
        Self::LockPoisoned {
            resource: resource.to_string(),
            details: "Thread panicked while holding lock".to_string(),
        }
    }

    /// Create a lock acquisition failure
    pub fn lock_failed(resource: &str, reason: &str) -> Self {
        Self::LockAcquisitionFailed {
            resource: resource.to_string(),
            reason: reason.to_string(),
        }
    }

    /// Get error code for client identification
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidInput { .. } => "INVALID_INPUT",
            Self::InvalidUserId(_) => "INVALID_USER_ID",
            Self::InvalidMemoryId(_) => "INVALID_MEMORY_ID",
            Self::InvalidEmbeddings(_) => "INVALID_EMBEDDINGS",
            Self::ContentTooLarge { .. } => "CONTENT_TOO_LARGE",
            Self::ResourceLimit { .. } => "RESOURCE_LIMIT",
            Self::MemoryNotFound(_) => "MEMORY_NOT_FOUND",
            Self::UserNotFound(_) => "USER_NOT_FOUND",
            Self::MemoryAlreadyExists(_) => "MEMORY_ALREADY_EXISTS",
            Self::StorageError(_) => "STORAGE_ERROR",
            Self::DatabaseError(_) => "DATABASE_ERROR",
            Self::SerializationError(_) => "SERIALIZATION_ERROR",
            Self::ConcurrencyError(_) => "CONCURRENCY_ERROR",
            Self::LockPoisoned { .. } => "LOCK_POISONED",
            Self::LockAcquisitionFailed { .. } => "LOCK_ACQUISITION_FAILED",
            Self::ServiceUnavailable(_) => "SERVICE_UNAVAILABLE",
            Self::Internal(_) => "INTERNAL_ERROR",
        }
    }

    /// Get HTTP status code
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::InvalidInput { .. }
            | Self::InvalidUserId(_)
            | Self::InvalidMemoryId(_)
            | Self::InvalidEmbeddings(_)
            | Self::ContentTooLarge { .. } => StatusCode::BAD_REQUEST,

            Self::ResourceLimit { .. } => StatusCode::TOO_MANY_REQUESTS,

            Self::MemoryNotFound(_)
            | Self::UserNotFound(_) => StatusCode::NOT_FOUND,

            Self::MemoryAlreadyExists(_) => StatusCode::CONFLICT,

            Self::ServiceUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,

            Self::StorageError(_)
            | Self::DatabaseError(_)
            | Self::SerializationError(_)
            | Self::ConcurrencyError(_)
            | Self::LockPoisoned { .. }
            | Self::LockAcquisitionFailed { .. }
            | Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Get detailed error message
    pub fn message(&self) -> String {
        match self {
            Self::InvalidInput { field, reason } => {
                format!("Invalid input for field '{field}': {reason}")
            }
            Self::InvalidUserId(msg) => format!("Invalid user ID: {msg}"),
            Self::InvalidMemoryId(msg) => format!("Invalid memory ID: {msg}"),
            Self::InvalidEmbeddings(msg) => format!("Invalid embeddings: {msg}"),
            Self::ContentTooLarge { size, max } => {
                format!("Content too large: {size} bytes (max: {max} bytes)")
            }
            Self::ResourceLimit { resource, current, limit } => {
                format!("Resource limit exceeded for {resource}: current={current} MB, limit={limit} MB")
            }
            Self::MemoryNotFound(id) => format!("Memory not found: {id}"),
            Self::UserNotFound(id) => format!("User not found: {id}"),
            Self::MemoryAlreadyExists(id) => format!("Memory already exists: {id}"),
            Self::StorageError(msg) => format!("Storage error: {msg}"),
            Self::DatabaseError(msg) => format!("Database error: {msg}"),
            Self::SerializationError(msg) => format!("Serialization error: {msg}"),
            Self::ConcurrencyError(msg) => format!("Concurrency error: {msg}"),
            Self::LockPoisoned { resource, details } => {
                format!("Lock poisoned on resource '{resource}': {details}")
            }
            Self::LockAcquisitionFailed { resource, reason } => {
                format!("Failed to acquire lock on '{resource}': {reason}")
            }
            Self::ServiceUnavailable(msg) => format!("Service unavailable: {msg}"),
            Self::Internal(err) => format!("Internal error: {err}"),
        }
    }

    /// Convert to structured error response
    pub fn to_response(&self) -> ErrorResponse {
        ErrorResponse {
            code: self.code().to_string(),
            message: self.message(),
            details: None,
            request_id: None,  // Can be set by middleware
        }
    }
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for AppError {}

/// Convert from anyhow::Error to AppError
impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        Self::Internal(err)
    }
}

/// Axum IntoResponse implementation for proper HTTP responses
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let body = self.to_response();

        (status, Json(body)).into_response()
    }
}

/// Helper trait to convert validation errors
pub trait ValidationErrorExt<T> {
    fn map_validation_err(self, field: &str) -> Result<T>;
}

impl<T> ValidationErrorExt<T> for anyhow::Result<T> {
    fn map_validation_err(self, field: &str) -> Result<T> {
        self.map_err(|e| AppError::InvalidInput {
            field: field.to_string(),
            reason: e.to_string(),
        })
    }
}

/// Type alias for Results using AppError
pub type Result<T> = std::result::Result<T, AppError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(AppError::InvalidUserId("test".to_string()).code(), "INVALID_USER_ID");
        assert_eq!(AppError::MemoryNotFound("123".to_string()).code(), "MEMORY_NOT_FOUND");
    }

    #[test]
    fn test_status_codes() {
        assert_eq!(
            AppError::InvalidUserId("test".to_string()).status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            AppError::MemoryNotFound("123".to_string()).status_code(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            AppError::StorageError("failed".to_string()).status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn test_error_response_serialization() {
        let err = AppError::InvalidUserId("test123".to_string());
        let response = err.to_response();

        assert_eq!(response.code, "INVALID_USER_ID");
        assert!(response.message.contains("test123"));
    }
}
