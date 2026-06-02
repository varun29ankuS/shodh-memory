//! HTTP-backed embedding client for OpenAI-compatible embedding APIs.
//!
//! Supports LM Studio, Ollama, vLLM, or any server implementing the
//! `/v1/embeddings` endpoint. Used as an alternative backend for Nomic
//! when local ONNX model isn't available but a server is running.
//!
//! Configuration via environment variables:
//! - SHODH_EMBEDDING_API_URL: Base URL (default: http://127.0.0.1:1234)
//! - SHODH_EMBEDDING_API_MODEL: Model name (default: text-embedding-nomic-embed-text-v1.5)
//! - SHODH_EMBEDDING_API_KEY: Optional API key (default: none)

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::Embedder;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:1234";
const AVAILABILITY_CACHE_TTL: Duration = Duration::from_secs(30);

/// Configuration for HTTP embedding API
#[derive(Debug, Clone)]
pub struct HttpEmbedderConfig {
    /// Base URL of the embedding API
    pub base_url: String,
    /// Model name to pass in the request
    pub model: String,
    /// Optional API key
    pub api_key: Option<String>,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
}

impl Default for HttpEmbedderConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl HttpEmbedderConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        let mut base_url = crate::integrations::resolve_api_url_override(
            "SHODH_EMBEDDING_API_URL",
            DEFAULT_BASE_URL,
        );
        if crate::integrations::is_insecure_remote_url(&base_url)
            && !std::env::var("SHODH_ALLOW_INSECURE_REMOTE_EMBEDDER")
                .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
                .unwrap_or(false)
        {
            tracing::warn!(
                "SHODH_EMBEDDING_API_URL points to insecure remote HTTP; using localhost default. \
                 Set SHODH_ALLOW_INSECURE_REMOTE_EMBEDDER=true to override."
            );
            base_url = DEFAULT_BASE_URL.to_string();
        }
        Self {
            base_url,
            model: std::env::var("SHODH_EMBEDDING_API_MODEL")
                .unwrap_or_else(|_| "text-embedding-nomic-embed-text-v1.5".to_string()),
            api_key: std::env::var("SHODH_EMBEDDING_API_KEY").ok(),
            timeout_ms: std::env::var("SHODH_EMBEDDING_API_TIMEOUT_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5000),
        }
    }
}

/// OpenAI-compatible embedding request
#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a str,
}

/// OpenAI-compatible embedding response
#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// Batch request
#[derive(Serialize)]
struct BatchEmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

/// HTTP-backed embedder using OpenAI-compatible /v1/embeddings endpoint.
///
/// Connects to LM Studio, Ollama, vLLM, or any compatible server.
/// Falls back gracefully if server is unreachable.
pub struct HttpEmbedder {
    config: HttpEmbedderConfig,
    client: ureq::Agent,
    /// Cached dimension from first successful call
    cached_dimension: std::sync::OnceLock<usize>,
    availability_cache: Mutex<Option<(Instant, bool)>>,
}

impl HttpEmbedder {
    /// Create a new HTTP embedder.
    pub fn new(config: HttpEmbedderConfig) -> Self {
        let client: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_millis(config.timeout_ms)))
            .build()
            .into();

        Self {
            config,
            client,
            cached_dimension: std::sync::OnceLock::new(),
            availability_cache: Mutex::new(None),
        }
    }

    /// Check if the embedding server is reachable without sending billable text.
    pub fn is_available(&self) -> bool {
        if let Some((checked_at, available)) = *self.availability_cache.lock().unwrap() {
            if checked_at.elapsed() < AVAILABILITY_CACHE_TTL {
                return available;
            }
        }

        let mut builder = self.client.head(&self.config.base_url);
        if let Some(ref key) = self.config.api_key {
            builder = builder.header("Authorization", &format!("Bearer {key}"));
        }
        let available = builder.call().is_ok();
        *self.availability_cache.lock().unwrap() = Some((Instant::now(), available));
        available
    }

    /// Encode a single text via the HTTP API.
    fn encode_http(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/v1/embeddings", self.config.base_url);
        let req = EmbeddingRequest {
            model: &self.config.model,
            input: text,
        };

        let mut builder = self.client.post(&url);
        if let Some(ref key) = self.config.api_key {
            builder = builder.header("Authorization", &format!("Bearer {key}"));
        }

        let resp = builder
            .send_json(&req)
            .context("HTTP embedding request failed")?;

        let parsed: EmbeddingResponse = resp
            .into_body()
            .read_json()
            .context("Failed to parse embedding response")?;

        parsed
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .context("Empty embedding response")
    }
}

impl Embedder for HttpEmbedder {
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let embedding = self.encode_http(text)?;
        // Cache the dimension on first successful call
        let _ = self.cached_dimension.set(embedding.len());
        Ok(embedding)
    }

    fn dimension(&self) -> usize {
        *self.cached_dimension.get().unwrap_or(&768)
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Use batch endpoint if available
        let url = format!("{}/v1/embeddings", self.config.base_url);
        let req = BatchEmbeddingRequest {
            model: &self.config.model,
            input: texts.to_vec(),
        };

        let mut builder = self.client.post(&url);
        if let Some(ref key) = self.config.api_key {
            builder = builder.header("Authorization", &format!("Bearer {key}"));
        }

        let resp = match builder.send_json(&req) {
            Ok(resp) => resp,
            Err(_) => return texts.iter().map(|t| self.encode(t)).collect(),
        };

        let parsed: EmbeddingResponse = resp
            .into_body()
            .read_json()
            .context("Failed to parse batch embedding response")?;

        if parsed.data.len() != texts.len() {
            // Batch size mismatch — fall back to sequential
            return texts.iter().map(|t| self.encode(t)).collect();
        }

        let embeddings: Vec<Vec<f32>> = parsed.data.into_iter().map(|d| d.embedding).collect();
        if let Some(first) = embeddings.first() {
            let _ = self.cached_dimension.set(first.len());
        }
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = HttpEmbedderConfig::from_env();
        assert!(config.base_url.contains("127.0.0.1"));
        assert!(config.model.contains("nomic"));
        assert_eq!(config.timeout_ms, 5000);
    }

    #[test]
    fn test_dimension_default() {
        let embedder = HttpEmbedder::new(HttpEmbedderConfig {
            base_url: "http://localhost:99999".into(), // unreachable
            model: "test".into(),
            api_key: None,
            timeout_ms: 100,
        });
        assert_eq!(embedder.dimension(), 768);
    }
}
