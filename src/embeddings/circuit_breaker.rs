//! Circuit breaker pattern for embedding service resilience
//!
//! Implements a production-grade circuit breaker to prevent cascading failures
//! when the ONNX embedding service is degraded or unavailable.
//!
//! # States
//! - **Closed**: Normal operation, requests pass through
//! - **Open**: Service is failing, requests are rejected immediately
//! - **HalfOpen**: Testing if service has recovered
//!
//! # Configuration
//! - `failure_threshold`: Number of failures before opening (default: 5)
//! - `success_threshold`: Successes needed to close from half-open (default: 2)
//! - `open_duration`: Time circuit stays open before testing (default: 30s)
//!
//! # Metrics Integration
//! All state transitions and rejections are tracked via Prometheus metrics.

use anyhow::Result;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::{minilm::MiniLMEmbedder, Embedder};

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation - requests pass through
    Closed,
    /// Service is failing - requests rejected immediately
    Open,
    /// Testing recovery - limited requests allowed
    HalfOpen,
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitState::Closed => write!(f, "closed"),
            CircuitState::Open => write!(f, "open"),
            CircuitState::HalfOpen => write!(f, "half_open"),
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening the circuit
    pub failure_threshold: u32,
    /// Number of consecutive successes needed to close from half-open
    pub success_threshold: u32,
    /// Duration the circuit stays open before transitioning to half-open
    pub open_duration: Duration,
    /// Maximum time to wait for a single embedding operation
    pub call_timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            open_duration: Duration::from_secs(30),
            call_timeout: Duration::from_secs(10),
        }
    }
}

/// Internal state tracking
struct CircuitBreakerState {
    state: CircuitState,
    consecutive_failures: u32,
    consecutive_successes: u32,
    last_failure_time: Option<Instant>,
    last_state_change: Instant,
}

impl CircuitBreakerState {
    fn new() -> Self {
        Self {
            state: CircuitState::Closed,
            consecutive_failures: 0,
            consecutive_successes: 0,
            last_failure_time: None,
            last_state_change: Instant::now(),
        }
    }
}

/// Circuit breaker wrapper for embedding service
///
/// Provides resilience by:
/// 1. Tracking failure rates
/// 2. Opening circuit when failures exceed threshold
/// 3. Automatically testing recovery after cooldown
/// 4. Falling back to simplified embeddings when circuit is open
pub struct ResilientEmbedder {
    inner: Arc<MiniLMEmbedder>,
    config: CircuitBreakerConfig,
    state: Mutex<CircuitBreakerState>,
    // Atomic counters for metrics (lock-free)
    total_calls: AtomicU64,
    total_rejections: AtomicU64,
    total_fallbacks: AtomicU64,
}

impl ResilientEmbedder {
    /// Create a new resilient embedder wrapping the given MiniLM embedder
    pub fn new(embedder: Arc<MiniLMEmbedder>, config: CircuitBreakerConfig) -> Self {
        Self {
            inner: embedder,
            config,
            state: Mutex::new(CircuitBreakerState::new()),
            total_calls: AtomicU64::new(0),
            total_rejections: AtomicU64::new(0),
            total_fallbacks: AtomicU64::new(0),
        }
    }

    /// Create with default configuration
    pub fn with_defaults(embedder: Arc<MiniLMEmbedder>) -> Self {
        Self::new(embedder, CircuitBreakerConfig::default())
    }

    /// Get current circuit state
    pub fn state(&self) -> CircuitState {
        self.state.lock().state
    }

    /// Get metrics for monitoring
    pub fn metrics(&self) -> CircuitBreakerMetrics {
        let state = self.state.lock();
        CircuitBreakerMetrics {
            state: state.state,
            consecutive_failures: state.consecutive_failures,
            consecutive_successes: state.consecutive_successes,
            total_calls: self.total_calls.load(Ordering::Relaxed),
            total_rejections: self.total_rejections.load(Ordering::Relaxed),
            total_fallbacks: self.total_fallbacks.load(Ordering::Relaxed),
            time_in_current_state: state.last_state_change.elapsed(),
        }
    }

    /// Check if circuit allows requests and update state if needed
    fn should_allow_request(&self) -> bool {
        let mut state = self.state.lock();

        match state.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if enough time has passed to try recovery
                if state.last_state_change.elapsed() >= self.config.open_duration {
                    tracing::info!(
                        "Circuit breaker transitioning from Open to HalfOpen after {:?}",
                        self.config.open_duration
                    );
                    state.state = CircuitState::HalfOpen;
                    state.consecutive_successes = 0;
                    state.last_state_change = Instant::now();
                    self.record_state_change(CircuitState::HalfOpen);
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful operation
    fn record_success(&self) {
        let mut state = self.state.lock();
        state.consecutive_failures = 0;
        state.consecutive_successes += 1;

        if state.state == CircuitState::HalfOpen
            && state.consecutive_successes >= self.config.success_threshold
        {
            tracing::info!(
                "Circuit breaker closing after {} consecutive successes",
                state.consecutive_successes
            );
            state.state = CircuitState::Closed;
            state.last_state_change = Instant::now();
            self.record_state_change(CircuitState::Closed);
        }
    }

    /// Record a failed operation
    fn record_failure(&self) {
        let mut state = self.state.lock();
        state.consecutive_successes = 0;
        state.consecutive_failures += 1;
        state.last_failure_time = Some(Instant::now());

        match state.state {
            CircuitState::Closed => {
                if state.consecutive_failures >= self.config.failure_threshold {
                    tracing::warn!(
                        "Circuit breaker opening after {} consecutive failures",
                        state.consecutive_failures
                    );
                    state.state = CircuitState::Open;
                    state.last_state_change = Instant::now();
                    self.record_state_change(CircuitState::Open);
                }
            }
            CircuitState::HalfOpen => {
                // Single failure in half-open returns to open
                tracing::warn!("Circuit breaker returning to Open after failure in HalfOpen state");
                state.state = CircuitState::Open;
                state.last_state_change = Instant::now();
                self.record_state_change(CircuitState::Open);
            }
            CircuitState::Open => {
                // Already open, nothing to do
            }
        }
    }

    /// Record state change to metrics
    fn record_state_change(&self, new_state: CircuitState) {
        let label1 = format!("circuit_breaker_{new_state}");
        let label2 = String::from("embedding");
        crate::metrics::ERRORS_TOTAL
            .with_label_values(&[&label1, &label2])
            .inc();
    }

    /// Generate fallback embedding using simplified hash-based approach
    fn generate_fallback(&self, text: &str) -> Vec<f32> {
        self.total_fallbacks.fetch_add(1, Ordering::Relaxed);

        // Use the same hash-based approach as simplified mode
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let dimension = self.inner.dimension();
        let mut embedding = vec![0.0; dimension];
        let mut hasher = DefaultHasher::new();

        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            word.hash(&mut hasher);
            let hash = hasher.finish();

            for j in 0..dimension {
                let index = (i * dimension + j) % dimension;
                if j < 64 {
                    embedding[index] += ((hash >> j) & 1) as f32 * 0.1;
                } else {
                    embedding[index] += ((hash >> (j % 64)) & 1) as f32 * 0.1;
                }
            }
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        embedding
    }
}

impl Embedder for ResilientEmbedder {
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        self.total_calls.fetch_add(1, Ordering::Relaxed);

        if text.is_empty() {
            return Ok(vec![0.0; self.inner.dimension()]);
        }

        // Check circuit state
        if !self.should_allow_request() {
            self.total_rejections.fetch_add(1, Ordering::Relaxed);
            tracing::debug!("Circuit breaker open, using fallback embedding");
            return Ok(self.generate_fallback(text));
        }

        // Try the actual embedding
        match self.inner.encode(text) {
            Ok(embedding) => {
                self.record_success();
                Ok(embedding)
            }
            Err(e) => {
                self.record_failure();
                tracing::warn!("Embedding failed (circuit breaker tracking): {}", e);
                // Return fallback instead of error to maintain availability
                Ok(self.generate_fallback(text))
            }
        }
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // For batch operations, check circuit once and apply consistently
        if !self.should_allow_request() {
            self.total_rejections
                .fetch_add(texts.len() as u64, Ordering::Relaxed);
            return Ok(texts.iter().map(|t| self.generate_fallback(t)).collect());
        }

        // Process batch with individual tracking
        let mut results = Vec::with_capacity(texts.len());
        let mut any_success = false;
        let mut any_failure = false;

        for text in texts {
            self.total_calls.fetch_add(1, Ordering::Relaxed);
            match self.inner.encode(text) {
                Ok(embedding) => {
                    any_success = true;
                    results.push(embedding);
                }
                Err(_) => {
                    any_failure = true;
                    results.push(self.generate_fallback(text));
                }
            }
        }

        // Update circuit state based on batch results
        if any_failure && !any_success {
            self.record_failure();
        } else if any_success {
            self.record_success();
        }

        Ok(results)
    }
}

/// Metrics snapshot for monitoring
#[derive(Debug, Clone)]
pub struct CircuitBreakerMetrics {
    pub state: CircuitState,
    pub consecutive_failures: u32,
    pub consecutive_successes: u32,
    pub total_calls: u64,
    pub total_rejections: u64,
    pub total_fallbacks: u64,
    pub time_in_current_state: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn create_test_embedder() -> Arc<MiniLMEmbedder> {
        let config = super::super::minilm::EmbeddingConfig {
            model_path: PathBuf::from("dummy.onnx"),
            tokenizer_path: PathBuf::from("dummy.json"),
            max_length: 256,
            use_quantized: true,
            embed_timeout_ms: 5000,
        };
        Arc::new(MiniLMEmbedder::new_simplified(config).unwrap())
    }

    #[test]
    fn test_circuit_breaker_starts_closed() {
        let embedder = create_test_embedder();
        let resilient = ResilientEmbedder::with_defaults(embedder);
        assert_eq!(resilient.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_metrics() {
        let embedder = create_test_embedder();
        let resilient = ResilientEmbedder::with_defaults(embedder);

        let _ = resilient.encode("test");
        let metrics = resilient.metrics();

        assert!(metrics.total_calls >= 1);
        assert_eq!(metrics.state, CircuitState::Closed);
    }

    #[test]
    fn test_fallback_generates_valid_embedding() {
        let embedder = create_test_embedder();
        let resilient = ResilientEmbedder::with_defaults(embedder);

        let fallback = resilient.generate_fallback("test input");
        assert_eq!(fallback.len(), 384);

        // Check normalization
        let norm: f32 = fallback.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5 || norm == 0.0);
    }
}
