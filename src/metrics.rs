//! Production-grade metrics with Prometheus
//!
//! Exposes key operational metrics for monitoring and alerting:
//! - Request rates and latencies
//! - Memory usage and resource consumption
//! - Vector index performance
//! - Error rates and types
//!
//! NOTE: We intentionally avoid user_id in metric labels to prevent
//! high-cardinality explosion that can crash Prometheus.

use prometheus::{
    Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts,
    Registry,
};
use std::sync::{LazyLock, OnceLock};

/// Metrics initialization result
static METRICS_INIT: OnceLock<Result<(), MetricsError>> = OnceLock::new();

/// Error type for metrics initialization
#[derive(Debug, Clone)]
pub struct MetricsError {
    pub message: String,
}

impl std::fmt::Display for MetricsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Metrics initialization failed: {}", self.message)
    }
}

impl std::error::Error for MetricsError {}

/// Create histogram opts with standard latency buckets
fn latency_histogram_opts(name: &str, help: &str) -> HistogramOpts {
    HistogramOpts::new(name, help).buckets(vec![
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
    ])
}

/// Create histogram opts for fast operations (sub-millisecond)
fn fast_histogram_opts(name: &str, help: &str) -> HistogramOpts {
    HistogramOpts::new(name, help).buckets(vec![
        0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05,
    ])
}

/// Global metrics registry
pub static METRICS_REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::new);

// ============================================================================
// Request Metrics
// ============================================================================

/// HTTP request duration in seconds
pub static HTTP_REQUEST_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        latency_histogram_opts(
            "shodh_http_request_duration_seconds",
            "HTTP request duration in seconds",
        ),
        &["method", "endpoint", "status"],
    )
    .expect("HTTP_REQUEST_DURATION metric must be valid at compile time")
});

/// Total HTTP requests
pub static HTTP_REQUESTS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new("shodh_http_requests_total", "Total HTTP requests"),
        &["method", "endpoint", "status"],
    )
    .expect("HTTP_REQUESTS_TOTAL metric must be valid at compile time")
});

// ============================================================================
// Memory Operation Metrics
// NOTE: No user_id in labels to prevent cardinality explosion
// ============================================================================

/// Memory store operations (record)
pub static MEMORY_STORE_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new("shodh_memory_store_total", "Total memory store operations"),
        &["result"],
    )
    .expect("MEMORY_STORE_TOTAL metric must be valid at compile time")
});

/// Memory store duration
pub static MEMORY_STORE_DURATION: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "shodh_memory_store_duration_seconds",
            "Memory store operation duration",
        )
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]),
    )
    .expect("MEMORY_STORE_DURATION metric must be valid at compile time")
});

/// Memory retrieve operations
pub static MEMORY_RETRIEVE_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "shodh_memory_retrieve_total",
            "Total memory retrieve operations",
        ),
        &["retrieval_mode", "result"],
    )
    .expect("MEMORY_RETRIEVE_TOTAL metric must be valid at compile time")
});

/// Memory retrieve duration
pub static MEMORY_RETRIEVE_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "shodh_memory_retrieve_duration_seconds",
            "Memory retrieve operation duration",
        )
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
        &["retrieval_mode"],
    )
    .expect("MEMORY_RETRIEVE_DURATION metric must be valid at compile time")
});

/// Results returned per query
pub static MEMORY_RETRIEVE_RESULTS: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "shodh_memory_retrieve_results",
            "Number of results returned per query",
        )
        .buckets(vec![0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]),
        &["retrieval_mode"],
    )
    .expect("MEMORY_RETRIEVE_RESULTS metric must be valid at compile time")
});

// ============================================================================
// Embedding Metrics (P1.2: Instrument embed operations)
// ============================================================================

/// Embedding generation operations
pub static EMBEDDING_GENERATE_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "shodh_embedding_generate_total",
            "Total embedding generations",
        ),
        &["mode", "result"], // mode: "onnx" or "simplified"
    )
    .expect("EMBEDDING_GENERATE_TOTAL metric must be valid at compile time")
});

/// Embedding generation duration
pub static EMBEDDING_GENERATE_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "shodh_embedding_generate_duration_seconds",
            "Embedding generation duration",
        )
        .buckets(vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0,
        ]),
        &["mode"],
    )
    .expect("EMBEDDING_GENERATE_DURATION metric must be valid at compile time")
});

/// Embedding timeout count
pub static EMBEDDING_TIMEOUT_TOTAL: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::new(
        "shodh_embedding_timeout_total",
        "Total embedding generation timeouts",
    )
    .expect("EMBEDDING_TIMEOUT_TOTAL metric must be valid at compile time")
});

/// NER session lock timeout count
pub static NER_LOCK_TIMEOUT_TOTAL: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::new(
        "shodh_ner_lock_timeout_total",
        "Total NER session lock timeouts (degraded entity extraction)",
    )
    .expect("NER_LOCK_TIMEOUT_TOTAL metric must be valid at compile time")
});

// ============================================================================
// Memory Usage Metrics (aggregate, no per-user to avoid cardinality)
// ============================================================================

/// Active users in cache
pub static ACTIVE_USERS: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "shodh_active_users",
        "Number of users with active memory sessions",
    )
    .expect("ACTIVE_USERS metric must be valid at compile time")
});

/// Total memories stored by tier (aggregate across all users)
pub static MEMORIES_BY_TIER: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new("shodh_memories_by_tier", "Total memories by tier"),
        &["tier"], // tier: "working", "session", "longterm"
    )
    .expect("MEMORIES_BY_TIER metric must be valid at compile time")
});

/// Total memory system heap usage (estimated, aggregate)
pub static MEMORY_HEAP_BYTES_TOTAL: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "shodh_memory_heap_bytes_total",
        "Total estimated heap usage across all users",
    )
    .expect("MEMORY_HEAP_BYTES_TOTAL metric must be valid at compile time")
});

// ============================================================================
// Vector Index Metrics (aggregate)
// ============================================================================

/// Total vector index size (number of vectors across all users)
pub static VECTOR_INDEX_SIZE_TOTAL: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "shodh_vector_index_size_total",
        "Total number of vectors in all indices",
    )
    .expect("VECTOR_INDEX_SIZE_TOTAL metric must be valid at compile time")
});

/// Vector search operations
pub static VECTOR_SEARCH_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "shodh_vector_search_total",
            "Total vector search operations",
        ),
        &["result"],
    )
    .expect("VECTOR_SEARCH_TOTAL metric must be valid at compile time")
});

/// Vector search duration
pub static VECTOR_SEARCH_DURATION: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(fast_histogram_opts(
        "shodh_vector_search_duration_seconds",
        "Vector search duration",
    ))
    .expect("VECTOR_SEARCH_DURATION metric must be valid at compile time")
});

// ============================================================================
// Storage Metrics
// ============================================================================

/// RocksDB operations
pub static ROCKSDB_OPS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new("shodh_rocksdb_ops_total", "Total RocksDB operations"),
        &["operation", "result"], // operation: "get", "put", "delete"
    )
    .expect("ROCKSDB_OPS_TOTAL metric must be valid at compile time")
});

/// RocksDB operation duration
pub static ROCKSDB_OPS_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        fast_histogram_opts(
            "shodh_rocksdb_ops_duration_seconds",
            "RocksDB operation duration",
        ),
        &["operation"],
    )
    .expect("ROCKSDB_OPS_DURATION metric must be valid at compile time")
});

// ============================================================================
// Error Metrics
// ============================================================================

/// Total errors by type
pub static ERRORS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new("shodh_errors_total", "Total errors by type"),
        &["error_type", "endpoint"],
    )
    .expect("ERRORS_TOTAL metric must be valid at compile time")
});

/// Resource limit rejections
pub static RESOURCE_LIMIT_REJECTIONS: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "shodh_resource_limit_rejections",
            "Requests rejected due to resource limits",
        ),
        &["resource"],
    )
    .expect("RESOURCE_LIMIT_REJECTIONS metric must be valid at compile time")
});

// ============================================================================
// Concurrency Metrics (P0.8)
// ============================================================================

/// Current concurrent requests
pub static CONCURRENT_REQUESTS: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "shodh_concurrent_requests",
        "Current number of concurrent requests",
    )
    .expect("CONCURRENT_REQUESTS metric must be valid at compile time")
});

/// Request queue size (if queuing implemented)
pub static REQUEST_QUEUE_SIZE: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new("shodh_request_queue_size", "Number of queued requests")
        .expect("REQUEST_QUEUE_SIZE metric must be valid at compile time")
});

// ============================================================================
// Hebbian Learning Metrics
// ============================================================================

/// Hebbian reinforcement operations
pub static HEBBIAN_REINFORCE_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "shodh_hebbian_reinforce_total",
            "Total Hebbian reinforcement operations",
        ),
        &["outcome", "result"], // outcome: "helpful", "misleading", "neutral"
    )
    .expect("HEBBIAN_REINFORCE_TOTAL metric must be valid at compile time")
});

/// Hebbian reinforcement duration
pub static HEBBIAN_REINFORCE_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "shodh_hebbian_reinforce_duration_seconds",
            "Hebbian reinforcement operation duration",
        )
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]),
        &["outcome"],
    )
    .expect("HEBBIAN_REINFORCE_DURATION metric must be valid at compile time")
});

// ============================================================================
// Consolidation Metrics
// ============================================================================

/// Memory consolidation operations
pub static CONSOLIDATE_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "shodh_consolidate_total",
            "Total memory consolidation operations",
        ),
        &["result"],
    )
    .expect("CONSOLIDATE_TOTAL metric must be valid at compile time")
});

/// Memory consolidation duration
pub static CONSOLIDATE_DURATION: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "shodh_consolidate_duration_seconds",
            "Memory consolidation operation duration",
        )
        .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
    )
    .expect("CONSOLIDATE_DURATION metric must be valid at compile time")
});

// ============================================================================
// Batch Operation Metrics
// ============================================================================

/// Batch store duration
pub static BATCH_STORE_DURATION: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "shodh_batch_store_duration_seconds",
            "Batch memory store operation duration",
        )
        .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]),
    )
    .expect("BATCH_STORE_DURATION metric must be valid at compile time")
});

/// Batch store size
pub static BATCH_STORE_SIZE: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "shodh_batch_store_size",
            "Number of memories in batch store operations",
        )
        .buckets(vec![
            1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0,
        ]),
    )
    .expect("BATCH_STORE_SIZE metric must be valid at compile time")
});

// ============================================================================
// Embedding Cache Metrics (SHO-68)
// ============================================================================

/// Embedding cache operations (query cache)
pub static EMBEDDING_CACHE_QUERY: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "shodh_embedding_cache_query_total",
            "Query embedding cache operations",
        ),
        &["result"], // result: "hit" or "miss"
    )
    .expect("EMBEDDING_CACHE_QUERY metric must be valid at compile time")
});

/// Embedding cache operations (content cache)
pub static EMBEDDING_CACHE_CONTENT: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "shodh_embedding_cache_content_total",
            "Content embedding cache operations",
        ),
        &["result"], // result: "hit" or "miss"
    )
    .expect("EMBEDDING_CACHE_CONTENT metric must be valid at compile time")
});

/// Current cache size (query cache)
pub static EMBEDDING_CACHE_QUERY_SIZE: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "shodh_embedding_cache_query_size",
        "Current number of entries in query embedding cache",
    )
    .expect("EMBEDDING_CACHE_QUERY_SIZE metric must be valid at compile time")
});

/// Current cache size (content cache)
pub static EMBEDDING_CACHE_CONTENT_SIZE: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "shodh_embedding_cache_content_size",
        "Current number of entries in content embedding cache",
    )
    .expect("EMBEDDING_CACHE_CONTENT_SIZE metric must be valid at compile time")
});

/// Register all metrics with the global registry
///
/// # Returns
/// - `Ok(())` if all metrics registered successfully
/// - `Err(MetricsError)` if any metric fails to register
///
/// # Behavior
/// - Registration is idempotent - calling multiple times is safe
/// - On failure, server should log warning and continue (degraded mode)
/// - Prometheus scraping will simply return empty metrics if registration failed
pub fn register_metrics() -> Result<(), MetricsError> {
    // Check if already initialized
    if let Some(result) = METRICS_INIT.get() {
        return result.clone();
    }

    let result = do_register_metrics();
    let _ = METRICS_INIT.set(result.clone());
    result
}

fn do_register_metrics() -> Result<(), MetricsError> {
    let mut errors = Vec::new();

    // Helper macro to reduce boilerplate
    macro_rules! register {
        ($metric:expr, $name:expr) => {
            if let Err(e) = METRICS_REGISTRY.register(Box::new($metric.clone())) {
                errors.push(format!("{}: {}", $name, e));
            }
        };
    }

    // Request metrics
    register!(HTTP_REQUEST_DURATION, "HTTP_REQUEST_DURATION");
    register!(HTTP_REQUESTS_TOTAL, "HTTP_REQUESTS_TOTAL");

    // Memory operation metrics
    register!(MEMORY_STORE_TOTAL, "MEMORY_STORE_TOTAL");
    register!(MEMORY_STORE_DURATION, "MEMORY_STORE_DURATION");
    register!(MEMORY_RETRIEVE_TOTAL, "MEMORY_RETRIEVE_TOTAL");
    register!(MEMORY_RETRIEVE_DURATION, "MEMORY_RETRIEVE_DURATION");
    register!(MEMORY_RETRIEVE_RESULTS, "MEMORY_RETRIEVE_RESULTS");

    // Embedding metrics
    register!(EMBEDDING_GENERATE_TOTAL, "EMBEDDING_GENERATE_TOTAL");
    register!(EMBEDDING_GENERATE_DURATION, "EMBEDDING_GENERATE_DURATION");
    register!(EMBEDDING_TIMEOUT_TOTAL, "EMBEDDING_TIMEOUT_TOTAL");
    register!(NER_LOCK_TIMEOUT_TOTAL, "NER_LOCK_TIMEOUT_TOTAL");

    // Memory usage metrics (aggregate)
    register!(ACTIVE_USERS, "ACTIVE_USERS");
    register!(MEMORIES_BY_TIER, "MEMORIES_BY_TIER");
    register!(MEMORY_HEAP_BYTES_TOTAL, "MEMORY_HEAP_BYTES_TOTAL");

    // Vector index metrics (aggregate)
    register!(VECTOR_INDEX_SIZE_TOTAL, "VECTOR_INDEX_SIZE_TOTAL");
    register!(VECTOR_SEARCH_TOTAL, "VECTOR_SEARCH_TOTAL");
    register!(VECTOR_SEARCH_DURATION, "VECTOR_SEARCH_DURATION");

    // Storage metrics
    register!(ROCKSDB_OPS_TOTAL, "ROCKSDB_OPS_TOTAL");
    register!(ROCKSDB_OPS_DURATION, "ROCKSDB_OPS_DURATION");

    // Error metrics
    register!(ERRORS_TOTAL, "ERRORS_TOTAL");
    register!(RESOURCE_LIMIT_REJECTIONS, "RESOURCE_LIMIT_REJECTIONS");

    // Concurrency metrics
    register!(CONCURRENT_REQUESTS, "CONCURRENT_REQUESTS");
    register!(REQUEST_QUEUE_SIZE, "REQUEST_QUEUE_SIZE");

    // Hebbian learning metrics
    register!(HEBBIAN_REINFORCE_TOTAL, "HEBBIAN_REINFORCE_TOTAL");
    register!(HEBBIAN_REINFORCE_DURATION, "HEBBIAN_REINFORCE_DURATION");

    // Consolidation metrics
    register!(CONSOLIDATE_TOTAL, "CONSOLIDATE_TOTAL");
    register!(CONSOLIDATE_DURATION, "CONSOLIDATE_DURATION");

    // Batch operation metrics
    register!(BATCH_STORE_DURATION, "BATCH_STORE_DURATION");
    register!(BATCH_STORE_SIZE, "BATCH_STORE_SIZE");

    // Embedding cache metrics (SHO-68)
    register!(EMBEDDING_CACHE_QUERY, "EMBEDDING_CACHE_QUERY");
    register!(EMBEDDING_CACHE_CONTENT, "EMBEDDING_CACHE_CONTENT");
    register!(EMBEDDING_CACHE_QUERY_SIZE, "EMBEDDING_CACHE_QUERY_SIZE");
    register!(EMBEDDING_CACHE_CONTENT_SIZE, "EMBEDDING_CACHE_CONTENT_SIZE");

    if errors.is_empty() {
        Ok(())
    } else {
        Err(MetricsError {
            message: errors.join("; "),
        })
    }
}

/// Helper to time operations with histogram (RAII pattern)
/// Usage: let _timer = Timer::new(SOME_HISTOGRAM.clone());
pub struct Timer {
    histogram: Histogram,
    start: std::time::Instant,
}

impl Timer {
    /// Create timer that records duration to histogram on drop
    pub fn new(histogram: Histogram) -> Self {
        Self {
            histogram,
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let duration = self.start.elapsed().as_secs_f64();
        self.histogram.observe(duration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::core::Metric;

    #[test]
    fn test_metrics_registration_is_idempotent() {
        // First registration should succeed
        let result1 = register_metrics();
        // Second registration should also succeed (returns cached result)
        let result2 = register_metrics();

        // Both should have same result
        assert_eq!(result1.is_ok(), result2.is_ok());
    }

    #[test]
    fn test_timer_records_duration() {
        // Create a test histogram
        let histogram = Histogram::with_opts(HistogramOpts::new(
            "test_timer_histogram",
            "Test histogram for timer",
        ))
        .unwrap();

        {
            let _timer = Timer::new(histogram.clone());
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Histogram should have recorded one observation
        let metric = histogram.metric();
        assert_eq!(metric.get_histogram().get_sample_count(), 1);
        // Duration should be at least 10ms
        assert!(metric.get_histogram().get_sample_sum() >= 0.01);
    }
}
