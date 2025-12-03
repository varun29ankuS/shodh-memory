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

use lazy_static::lazy_static;
use prometheus::{
    Histogram, HistogramOpts, HistogramVec,
    IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry,
};

lazy_static! {
    /// Global metrics registry
    pub static ref METRICS_REGISTRY: Registry = Registry::new();

    // ============================================================================
    // Request Metrics
    // ============================================================================

    /// HTTP request duration in seconds
    pub static ref HTTP_REQUEST_DURATION: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "shodh_http_request_duration_seconds",
            "HTTP request duration in seconds"
        )
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]),
        &["method", "endpoint", "status"]
    ).unwrap();

    /// Total HTTP requests
    pub static ref HTTP_REQUESTS_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("shodh_http_requests_total", "Total HTTP requests"),
        &["method", "endpoint", "status"]
    ).unwrap();

    // ============================================================================
    // Memory Operation Metrics
    // NOTE: No user_id in labels to prevent cardinality explosion
    // ============================================================================

    /// Memory store operations (record)
    pub static ref MEMORY_STORE_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("shodh_memory_store_total", "Total memory store operations"),
        &["result"]
    ).unwrap();

    /// Memory store duration
    pub static ref MEMORY_STORE_DURATION: Histogram = Histogram::with_opts(
        HistogramOpts::new(
            "shodh_memory_store_duration_seconds",
            "Memory store operation duration"
        )
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5])
    ).unwrap();

    /// Memory retrieve operations
    pub static ref MEMORY_RETRIEVE_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("shodh_memory_retrieve_total", "Total memory retrieve operations"),
        &["retrieval_mode", "result"]
    ).unwrap();

    /// Memory retrieve duration
    pub static ref MEMORY_RETRIEVE_DURATION: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "shodh_memory_retrieve_duration_seconds",
            "Memory retrieve operation duration"
        )
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
        &["retrieval_mode"]
    ).unwrap();

    /// Results returned per query
    pub static ref MEMORY_RETRIEVE_RESULTS: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "shodh_memory_retrieve_results",
            "Number of results returned per query"
        )
        .buckets(vec![0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]),
        &["retrieval_mode"]
    ).unwrap();

    // ============================================================================
    // Embedding Metrics (P1.2: Instrument embed operations)
    // ============================================================================

    /// Embedding generation operations
    pub static ref EMBEDDING_GENERATE_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("shodh_embedding_generate_total", "Total embedding generations"),
        &["mode", "result"]  // mode: "onnx" or "simplified"
    ).unwrap();

    /// Embedding generation duration
    pub static ref EMBEDDING_GENERATE_DURATION: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "shodh_embedding_generate_duration_seconds",
            "Embedding generation duration"
        )
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0]),
        &["mode"]
    ).unwrap();

    /// Embedding timeout count
    pub static ref EMBEDDING_TIMEOUT_TOTAL: IntCounter = IntCounter::new(
        "shodh_embedding_timeout_total",
        "Total embedding generation timeouts"
    ).unwrap();

    // ============================================================================
    // Memory Usage Metrics (aggregate, no per-user to avoid cardinality)
    // ============================================================================

    /// Active users in cache
    pub static ref ACTIVE_USERS: IntGauge = IntGauge::new(
        "shodh_active_users",
        "Number of users with active memory sessions"
    ).unwrap();

    /// Total memories stored by tier (aggregate across all users)
    pub static ref MEMORIES_BY_TIER: IntGaugeVec = IntGaugeVec::new(
        Opts::new("shodh_memories_by_tier", "Total memories by tier"),
        &["tier"]  // tier: "working", "session", "longterm"
    ).unwrap();

    /// Total memory system heap usage (estimated, aggregate)
    pub static ref MEMORY_HEAP_BYTES_TOTAL: IntGauge = IntGauge::new(
        "shodh_memory_heap_bytes_total",
        "Total estimated heap usage across all users"
    ).unwrap();

    // ============================================================================
    // Vector Index Metrics (aggregate)
    // ============================================================================

    /// Total vector index size (number of vectors across all users)
    pub static ref VECTOR_INDEX_SIZE_TOTAL: IntGauge = IntGauge::new(
        "shodh_vector_index_size_total",
        "Total number of vectors in all indices"
    ).unwrap();

    /// Vector search operations
    pub static ref VECTOR_SEARCH_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("shodh_vector_search_total", "Total vector search operations"),
        &["result"]
    ).unwrap();

    /// Vector search duration
    pub static ref VECTOR_SEARCH_DURATION: Histogram = Histogram::with_opts(
        HistogramOpts::new(
            "shodh_vector_search_duration_seconds",
            "Vector search duration"
        )
        .buckets(vec![0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05])
    ).unwrap();

    // ============================================================================
    // Storage Metrics
    // ============================================================================

    /// RocksDB operations
    pub static ref ROCKSDB_OPS_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("shodh_rocksdb_ops_total", "Total RocksDB operations"),
        &["operation", "result"]  // operation: "get", "put", "delete"
    ).unwrap();

    /// RocksDB operation duration
    pub static ref ROCKSDB_OPS_DURATION: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "shodh_rocksdb_ops_duration_seconds",
            "RocksDB operation duration"
        )
        .buckets(vec![0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05]),
        &["operation"]
    ).unwrap();

    // ============================================================================
    // Error Metrics
    // ============================================================================

    /// Total errors by type
    pub static ref ERRORS_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("shodh_errors_total", "Total errors by type"),
        &["error_type", "endpoint"]
    ).unwrap();

    /// Resource limit rejections
    pub static ref RESOURCE_LIMIT_REJECTIONS: IntCounterVec = IntCounterVec::new(
        Opts::new("shodh_resource_limit_rejections", "Requests rejected due to resource limits"),
        &["resource"]
    ).unwrap();

    // ============================================================================
    // Concurrency Metrics (P0.8)
    // ============================================================================

    /// Current concurrent requests
    pub static ref CONCURRENT_REQUESTS: IntGauge = IntGauge::new(
        "shodh_concurrent_requests",
        "Current number of concurrent requests"
    ).unwrap();

    /// Request queue size (if queuing implemented)
    pub static ref REQUEST_QUEUE_SIZE: IntGauge = IntGauge::new(
        "shodh_request_queue_size",
        "Number of queued requests"
    ).unwrap();
}

/// Register all metrics with the global registry
pub fn register_metrics() -> Result<(), prometheus::Error> {
    // Request metrics
    METRICS_REGISTRY.register(Box::new(HTTP_REQUEST_DURATION.clone()))?;
    METRICS_REGISTRY.register(Box::new(HTTP_REQUESTS_TOTAL.clone()))?;

    // Memory operation metrics
    METRICS_REGISTRY.register(Box::new(MEMORY_STORE_TOTAL.clone()))?;
    METRICS_REGISTRY.register(Box::new(MEMORY_STORE_DURATION.clone()))?;
    METRICS_REGISTRY.register(Box::new(MEMORY_RETRIEVE_TOTAL.clone()))?;
    METRICS_REGISTRY.register(Box::new(MEMORY_RETRIEVE_DURATION.clone()))?;
    METRICS_REGISTRY.register(Box::new(MEMORY_RETRIEVE_RESULTS.clone()))?;

    // Embedding metrics
    METRICS_REGISTRY.register(Box::new(EMBEDDING_GENERATE_TOTAL.clone()))?;
    METRICS_REGISTRY.register(Box::new(EMBEDDING_GENERATE_DURATION.clone()))?;
    METRICS_REGISTRY.register(Box::new(EMBEDDING_TIMEOUT_TOTAL.clone()))?;

    // Memory usage metrics (aggregate)
    METRICS_REGISTRY.register(Box::new(ACTIVE_USERS.clone()))?;
    METRICS_REGISTRY.register(Box::new(MEMORIES_BY_TIER.clone()))?;
    METRICS_REGISTRY.register(Box::new(MEMORY_HEAP_BYTES_TOTAL.clone()))?;

    // Vector index metrics (aggregate)
    METRICS_REGISTRY.register(Box::new(VECTOR_INDEX_SIZE_TOTAL.clone()))?;
    METRICS_REGISTRY.register(Box::new(VECTOR_SEARCH_TOTAL.clone()))?;
    METRICS_REGISTRY.register(Box::new(VECTOR_SEARCH_DURATION.clone()))?;

    // Storage metrics
    METRICS_REGISTRY.register(Box::new(ROCKSDB_OPS_TOTAL.clone()))?;
    METRICS_REGISTRY.register(Box::new(ROCKSDB_OPS_DURATION.clone()))?;

    // Error metrics
    METRICS_REGISTRY.register(Box::new(ERRORS_TOTAL.clone()))?;
    METRICS_REGISTRY.register(Box::new(RESOURCE_LIMIT_REJECTIONS.clone()))?;

    // Concurrency metrics
    METRICS_REGISTRY.register(Box::new(CONCURRENT_REQUESTS.clone()))?;
    METRICS_REGISTRY.register(Box::new(REQUEST_QUEUE_SIZE.clone()))?;

    Ok(())
}

/// Helper to time operations with histogram (RAII pattern)
/// Usage: let _timer = Timer::new(SOME_HISTOGRAM.clone());
#[allow(unused)]  // Public API utility for metrics consumers
pub struct Timer {
    histogram: Histogram,
    start: std::time::Instant,
}

#[allow(unused)]  // Public API utility
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
