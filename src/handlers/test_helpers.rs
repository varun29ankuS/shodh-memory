//! Shared test utilities for handler unit tests.
//!
//! Provides a [`TestHarness`] that sets up a temporary `MultiUserMemoryManager`
//! backed by a fresh RocksDB in a temp directory, plus convenience helpers for
//! building authenticated HTTP requests and reading JSON response bodies.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
    Router,
};
use http_body_util::BodyExt;
use serde::de::DeserializeOwned;
use tempfile::TempDir;
use tower::ServiceExt; // for oneshot()

use super::router::build_router;
use super::state::MultiUserMemoryManager;
use crate::config::ServerConfig;

/// Test API key used in all handler tests.
pub const TEST_API_KEY: &str = "test-handler-key-2025";

/// A self-contained test environment with its own temp storage.
///
/// Holds `TempDir` so the directory isn't cleaned up until the harness drops.
pub struct TestHarness {
    pub manager: Arc<MultiUserMemoryManager>,
    _temp_dir: TempDir,
}

impl TestHarness {
    /// Create a new harness with a fresh temp directory and default config.
    ///
    /// Sets `SHODH_API_KEYS` so the auth middleware accepts [`TEST_API_KEY`].
    pub fn new() -> Self {
        // Ensure the test API key is set before constructing the manager.
        // Safety: tests run single-threaded by default in each test binary,
        // and this env var is only read (not race-critical) by the auth layer.
        unsafe {
            std::env::set_var("SHODH_API_KEYS", TEST_API_KEY);
        }

        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let config = ServerConfig {
            storage_path: temp_dir.path().to_path_buf(),
            backup_enabled: false,
            ..ServerConfig::default()
        };

        let manager = MultiUserMemoryManager::new(temp_dir.path().to_path_buf(), config)
            .expect("failed to create test MultiUserMemoryManager");

        Self {
            manager: Arc::new(manager),
            _temp_dir: temp_dir,
        }
    }

    /// Get a clone of the shared state (what handlers receive via `State(..)`).
    pub fn state(&self) -> Arc<MultiUserMemoryManager> {
        self.manager.clone()
    }

    /// Build the full application router (public + protected routes).
    pub fn router(&self) -> Router {
        build_router(self.manager.clone())
    }
}

// ---------- Request builders ----------

/// Build a GET request to `uri` with the test API key header.
pub fn get(uri: &str) -> Request<Body> {
    Request::builder()
        .method(Method::GET)
        .uri(uri)
        .header("x-api-key", TEST_API_KEY)
        .body(Body::empty())
        .unwrap()
}

/// Build a POST request to `uri` with a JSON body and the test API key.
pub fn post_json<T: serde::Serialize>(uri: &str, body: &T) -> Request<Body> {
    let json = serde_json::to_string(body).unwrap();
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-api-key", TEST_API_KEY)
        .body(Body::from(json))
        .unwrap()
}

/// Build a PUT request to `uri` with a JSON body and the test API key.
pub fn put_json<T: serde::Serialize>(uri: &str, body: &T) -> Request<Body> {
    let json = serde_json::to_string(body).unwrap();
    Request::builder()
        .method(Method::PUT)
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-api-key", TEST_API_KEY)
        .body(Body::from(json))
        .unwrap()
}

/// Build a DELETE request to `uri` with the test API key.
pub fn delete(uri: &str) -> Request<Body> {
    Request::builder()
        .method(Method::DELETE)
        .uri(uri)
        .header("x-api-key", TEST_API_KEY)
        .body(Body::empty())
        .unwrap()
}

/// Build a DELETE request to `uri` with a JSON body and the test API key.
pub fn delete_json<T: serde::Serialize>(uri: &str, body: &T) -> Request<Body> {
    let json = serde_json::to_string(body).unwrap();
    Request::builder()
        .method(Method::DELETE)
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-api-key", TEST_API_KEY)
        .body(Body::from(json))
        .unwrap()
}

/// Build a GET request **without** an API key (for testing auth rejection).
pub fn get_unauthenticated(uri: &str) -> Request<Body> {
    Request::builder()
        .method(Method::GET)
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

/// Build a POST request **without** an API key (for testing auth rejection).
pub fn post_json_unauthenticated<T: serde::Serialize>(uri: &str, body: &T) -> Request<Body> {
    let json = serde_json::to_string(body).unwrap();
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

// ---------- Response helpers ----------

/// Send a request through the router and return (status, JSON body).
pub async fn send(app: Router, req: Request<Body>) -> (StatusCode, serde_json::Value) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = if body_bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&body_bytes).unwrap_or_else(|_| {
            serde_json::Value::String(String::from_utf8_lossy(&body_bytes).to_string())
        })
    };
    (status, json)
}

/// Send a request and deserialize the body into `T`.
pub async fn send_typed<T: DeserializeOwned>(
    app: Router,
    req: Request<Body>,
) -> (StatusCode, T) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let value: T = serde_json::from_slice(&body_bytes)
        .unwrap_or_else(|e| panic!("failed to deserialize response: {e}\nbody: {}", String::from_utf8_lossy(&body_bytes)));
    (status, value)
}
