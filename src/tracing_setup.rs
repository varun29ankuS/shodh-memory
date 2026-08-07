//! P1.6: Distributed tracing with OpenTelemetry (OPTIONAL)
//!
//! Enables distributed tracing for production observability:
//! - End-to-end request tracking across services
//! - Performance bottleneck identification
//! - Latency analysis per operation
//! - Integration with Jaeger, Tempo, or any OTLP-compatible backend
//!
//! **Feature flag: `telemetry`**
//! - Enable with: `cargo build --features telemetry`
//! - Disabled by default for edge devices (saves ~200 packages)

#[cfg(feature = "telemetry")]
use opentelemetry::{global, trace::TracerProvider as _, KeyValue};
#[cfg(feature = "telemetry")]
use opentelemetry_otlp::{SpanExporter, WithExportConfig};
#[cfg(feature = "telemetry")]
use opentelemetry_sdk::{
    trace::{RandomIdGenerator, Sampler, SdkTracerProvider},
    Resource,
};
#[cfg(feature = "telemetry")]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Global tracer provider for shutdown
#[cfg(feature = "telemetry")]
static TRACER_PROVIDER: std::sync::OnceLock<SdkTracerProvider> = std::sync::OnceLock::new();

/// Initialize distributed tracing with OpenTelemetry
///
/// Configuration via environment variables:
/// - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (default: http://localhost:4317)
/// - OTEL_SERVICE_NAME: Service name (default: shodh-memory)
/// - OTEL_TRACE_SAMPLER: Sampling strategy (default: parentbased_always_on)
/// - RUST_LOG: Log level filter (default: info)
#[cfg(feature = "telemetry")]
pub fn init_tracing() -> Result<(), Box<dyn std::error::Error>> {
    // Read configuration from environment
    let otlp_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .unwrap_or_else(|_| "http://localhost:4317".to_string());

    let service_name =
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "shodh-memory".to_string());

    // Build span exporter using the new builder API (opentelemetry 0.27+)
    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&otlp_endpoint)
        .build()?;

    // Build resource with service metadata
    let resource = Resource::builder()
        .with_service_name(service_name.clone())
        .with_attribute(KeyValue::new("service.version", env!("CARGO_PKG_VERSION")))
        .build();

    // Build tracer provider using the new builder API
    // (0.28+ batch exporter picks the async runtime via the rt-tokio feature)
    let tracer_provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_sampler(Sampler::ParentBased(Box::new(Sampler::AlwaysOn)))
        .with_id_generator(RandomIdGenerator::default())
        .with_resource(resource)
        .build();

    // Get tracer from provider
    let tracer = tracer_provider.tracer("shodh-memory");

    // Store provider for shutdown
    let _ = TRACER_PROVIDER.set(tracer_provider.clone());

    // Set as global provider
    global::set_tracer_provider(tracer_provider);

    // Create OpenTelemetry tracing layer
    let telemetry_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    // Configure log filter (RUST_LOG env var or default to info)
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // Set up tracing subscriber with both console and OpenTelemetry layers
    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .with(telemetry_layer)
        .init();

    tracing::info!(
        service_name = %service_name,
        otlp_endpoint = %otlp_endpoint,
        "OpenTelemetry tracing initialized"
    );

    Ok(())
}

/// Shutdown tracing and flush remaining spans
///
/// Call this during graceful shutdown to ensure all traces are exported
#[cfg(feature = "telemetry")]
pub fn shutdown_tracing() {
    tracing::info!("Shutting down OpenTelemetry tracing");
    if let Some(provider) = TRACER_PROVIDER.get() {
        if let Err(e) = provider.shutdown() {
            tracing::error!("Error shutting down tracer provider: {:?}", e);
        }
    }
}

/// Middleware for propagating trace context through HTTP headers
///
/// Extracts trace context from incoming requests and injects it into outgoing requests
/// following W3C Trace Context specification (traceparent/tracestate headers)
#[cfg(feature = "telemetry")]
pub mod trace_propagation {
    use axum::{extract::Request, middleware::Next, response::Response};
    use opentelemetry::global;
    use opentelemetry::propagation::Extractor;
    use tracing::Span;
    use tracing_opentelemetry::OpenTelemetrySpanExt;

    /// HTTP header extractor for OpenTelemetry context propagation
    struct HeaderExtractor<'a> {
        headers: &'a axum::http::HeaderMap,
    }

    impl<'a> Extractor for HeaderExtractor<'a> {
        fn get(&self, key: &str) -> Option<&str> {
            self.headers.get(key)?.to_str().ok()
        }

        fn keys(&self) -> Vec<&str> {
            self.headers.keys().map(|k| k.as_str()).collect()
        }
    }

    /// Middleware to extract trace context from HTTP headers
    pub async fn propagate_trace_context(req: Request, next: Next) -> Response {
        // Extract parent trace context from headers
        let extractor = HeaderExtractor {
            headers: req.headers(),
        };
        let parent_cx =
            global::get_text_map_propagator(|propagator| propagator.extract(&extractor));

        // Set parent context on current span
        let current_span = Span::current();
        if let Err(e) = current_span.set_parent(parent_cx) {
            tracing::debug!("failed to set parent trace context: {e}");
        }

        // Continue processing
        next.run(req).await
    }
}

#[cfg(all(test, feature = "telemetry"))]
mod tests {
    use super::*;

    #[test]
    fn test_tracing_init_no_panic() {
        // Test that init doesn't panic even if OTLP endpoint is unavailable
        // In production, traces will be buffered and retried
        let _ = init_tracing();
    }
}
