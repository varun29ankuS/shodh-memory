//! Server bootstrap module — starts the Shodh-Memory HTTP API server.
//!
//! Extracted from `main.rs` so that both `shodh-memory-server` (standalone)
//! and `shodh server` (unified CLI) can start the server with identical behavior.

use anyhow::Result;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tower::ServiceBuilder;
use tower_governor::{governor::GovernorConfigBuilder, GovernorLayer};
use tower_http::timeout::TimeoutLayer;
use tracing::{error, info};

use crate::{
    auth,
    config::ServerConfig,
    embeddings::minilm::pre_init_ort_runtime,
    handlers::{self, AppState, MultiUserMemoryManager},
    metrics, middleware,
};

#[cfg(feature = "telemetry")]
use crate::tracing_setup;

use crate::constants::{
    DATABASE_FLUSH_TIMEOUT_SECS, GRACEFUL_SHUTDOWN_TIMEOUT_SECS, VECTOR_INDEX_SAVE_TIMEOUT_SECS,
};

// Timeout for draining in-flight requests (server-specific, not in constants.rs)
const SERVER_DRAIN_TIMEOUT_SECS: u64 = 5;

// =============================================================================
// PUBLIC API
// =============================================================================

/// Configuration for starting the server via [`run`].
pub struct ServerRunConfig {
    pub host: String,
    pub port: u16,
    pub storage_path: PathBuf,
    pub production: bool,
    pub rate_limit: u64,
    pub max_concurrent: usize,
}

/// Start the shodh-memory HTTP server.
///
/// This is a **blocking** call that runs until a shutdown signal (Ctrl-C / SIGTERM).
/// It sets environment variables, pre-initialises the ONNX runtime, builds a tokio
/// runtime, and then enters the async server loop.
///
/// # Safety
/// Environment variables are set **before** the tokio runtime is created, so no
/// threads exist yet. This avoids the `set_var` unsoundness on multi-threaded runtimes.
pub fn run(config: ServerRunConfig) -> Result<()> {
    // SAFETY: These set_var calls run before any threads are spawned — the tokio
    // runtime is not yet built, and pre_init_ort_runtime (below) is also single-threaded.
    // `std::env::set_var` is marked unsafe starting in Rust 2024 edition because it is
    // unsound to call concurrently with `std::env::var` in other threads. Here, this
    // process is single-threaded, so the invariant holds.
    unsafe {
        std::env::set_var("SHODH_HOST", &config.host);
        std::env::set_var("SHODH_PORT", config.port.to_string());
        std::env::set_var(
            "SHODH_MEMORY_PATH",
            config.storage_path.to_string_lossy().to_string(),
        );
        if config.production {
            std::env::set_var("SHODH_ENV", "production");
        }
        std::env::set_var("SHODH_RATE_LIMIT", config.rate_limit.to_string());
        std::env::set_var("SHODH_MAX_CONCURRENT", config.max_concurrent.to_string());
    }

    // Pre-initialize ORT_DYLIB_PATH before any threads are spawned.
    pre_init_ort_runtime(false);

    // SAFETY: Still single-threaded — setting default log level before runtime construction.
    if std::env::var("RUST_LOG").is_err() {
        unsafe {
            std::env::set_var("RUST_LOG", "shodh_memory=info,tower_http=warn");
        }
    }

    // Load .env file if present (won't override CLI-set vars)
    let _ = dotenvy::dotenv();

    // Build and enter the tokio runtime
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to build tokio runtime")
        .block_on(async_main())
}

// =============================================================================
// ASYNC MAIN
// =============================================================================

async fn async_main() -> Result<()> {
    // Initialize tracing
    #[cfg(feature = "telemetry")]
    {
        tracing_setup::init_tracing().expect("Failed to initialize tracing");
    }
    #[cfg(not(feature = "telemetry"))]
    {
        tracing_subscriber::fmt::init();
    }

    // Print startup banner
    print_banner();

    // Log security/authentication status
    auth::log_security_status();

    // Register Prometheus metrics
    metrics::register_metrics().expect("Failed to register metrics");

    // Load configuration
    let server_config = ServerConfig::from_env();
    print_config(&server_config);

    // Create memory manager
    let manager: AppState = Arc::new(MultiUserMemoryManager::new(
        server_config.storage_path.clone(),
        server_config.clone(),
    )?);

    // Print storage stats
    print_storage_stats(&server_config.storage_path);

    // Keep reference for shutdown cleanup
    let manager_for_shutdown = Arc::clone(&manager);

    // Start background maintenance scheduler
    start_maintenance_scheduler(
        Arc::clone(&manager),
        server_config.maintenance_interval_secs,
    );

    // Start active reminder scheduler (checks every 60s for due reminders)
    start_reminder_scheduler(Arc::clone(&manager));

    // Start backup scheduler if enabled
    if server_config.backup_enabled && server_config.backup_interval_secs > 0 {
        start_backup_scheduler(
            Arc::clone(&manager),
            server_config.backup_interval_secs,
            server_config.backup_max_count,
        );
    }

    // Start Zenoh transport if feature-enabled and configured
    #[cfg(feature = "zenoh")]
    let zenoh_handle = {
        let zenoh_config = crate::zenoh_transport::ZenohConfig::from_env();
        if zenoh_config.enabled {
            match crate::zenoh_transport::start(Arc::clone(&manager), zenoh_config).await {
                Ok(handle) => {
                    info!("Zenoh transport started successfully");
                    Some(handle)
                }
                Err(e) => {
                    error!("Failed to start Zenoh transport: {}. HTTP server will continue without Zenoh.", e);
                    None
                }
            }
        } else {
            info!("Zenoh transport: disabled (set SHODH_ZENOH_ENABLED=true to enable)");
            None
        }
    };

    // Configure rate limiting (0 = disabled, for localhost/embedded use)
    let rate_limit_enabled = server_config.rate_limit_per_second > 0;
    let governor_layer = if rate_limit_enabled {
        let rps = server_config.rate_limit_per_second.max(1);
        let cell_interval = std::time::Duration::from_nanos(1_000_000_000 / rps);
        let governor_conf = GovernorConfigBuilder::default()
            .period(cell_interval)
            .burst_size(server_config.rate_limit_burst)
            .finish()
            .expect("Failed to build governor rate limiter configuration");
        info!(
            "Rate limiting: {} req/sec (cell interval: {:?}), burst of {}",
            server_config.rate_limit_per_second, cell_interval, server_config.rate_limit_burst
        );
        Some(GovernorLayer::new(governor_conf))
    } else {
        info!("Rate limiting: disabled (SHODH_RATE_LIMIT=0)");
        None
    };

    // Build CORS layer
    let cors = server_config.cors.to_layer();

    // Build routes using handlers module
    let public_routes = handlers::build_public_routes(Arc::clone(&manager)).route(
        "/",
        axum::routing::get(|| async {
            axum::Json(serde_json::json!({
                "name": "shodh-memory",
                "version": env!("CARGO_PKG_VERSION"),
                "description": "Cognitive Memory for AI Agents",
                "health": "/health",
                "api": {
                    "remember": "POST /api/remember",
                    "recall": "POST /api/recall",
                    "forget": "POST /api/forget",
                    "todos": "GET /api/todos",
                    "graph": "GET /api/graph/stats"
                },
                "docs": "https://github.com/varun29ankuS/shodh-memory"
            }))
        }),
    );

    let protected_routes = if let Some(governor) = governor_layer {
        handlers::build_protected_routes(Arc::clone(&manager))
            .layer(axum::middleware::from_fn(auth::auth_middleware))
            .layer(governor)
    } else {
        handlers::build_protected_routes(Arc::clone(&manager))
            .layer(axum::middleware::from_fn(auth::auth_middleware))
    };

    // Combine routes with global middleware
    let request_timeout = std::time::Duration::from_secs(server_config.request_timeout_secs);
    let app = axum::Router::new()
        .merge(public_routes)
        .merge(protected_routes)
        .layer(
            ServiceBuilder::new()
                .layer(axum::middleware::from_fn(middleware::security_headers))
                .layer(axum::middleware::from_fn(middleware::track_metrics))
                .layer(TimeoutLayer::with_status_code(
                    axum::http::StatusCode::REQUEST_TIMEOUT,
                    request_timeout,
                ))
                .layer(tower::limit::ConcurrencyLimitLayer::new(
                    server_config.max_concurrent_requests,
                ))
                .layer(cors),
        );

    // Conditionally add trace propagation
    #[cfg(feature = "telemetry")]
    let app = app.layer(axum::middleware::from_fn(
        tracing_setup::trace_propagation::propagate_trace_context,
    ));

    // Start server
    let host = &server_config.host;
    let port = server_config.port;
    let addr: SocketAddr = format!("{}:{}", host, port).parse().unwrap_or_else(|_| {
        tracing::warn!("Invalid SHODH_HOST '{}', falling back to 127.0.0.1", host);
        SocketAddr::from(([127, 0, 0, 1], port))
    });

    // Small delay for log flush
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    print_ready_message(addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;

    // Use a notify to signal the server to stop accepting new connections
    let shutdown_notify = Arc::new(tokio::sync::Notify::new());
    let shutdown_listener = shutdown_notify.clone();

    let server = axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        shutdown_listener.notified().await;
    });

    let mut server_handle = tokio::spawn(async move { server.await });

    // Wait for shutdown signal (Ctrl+C / SIGTERM)
    shutdown_signal_with_drain().await;

    // Tell the server to stop accepting new connections
    shutdown_notify.notify_one();

    // Give the server a brief moment to finish in-flight requests
    info!(
        "Waiting up to {}s for in-flight requests...",
        SERVER_DRAIN_TIMEOUT_SECS
    );
    match tokio::time::timeout(
        std::time::Duration::from_secs(SERVER_DRAIN_TIMEOUT_SECS),
        &mut server_handle,
    )
    .await
    {
        Ok(Ok(Ok(()))) => info!("Server stopped gracefully"),
        Ok(Ok(Err(e))) => error!("Server error: {}", e),
        Ok(Err(e)) => error!("Server task panicked: {}", e),
        Err(_) => {
            info!(
                "Server drain timed out after {}s, aborting server task",
                SERVER_DRAIN_TIMEOUT_SECS
            );
            server_handle.abort();
        }
    }

    // Shut down Zenoh transport before flushing databases
    #[cfg(feature = "zenoh")]
    if let Some(handle) = zenoh_handle {
        handle.shutdown().await;
    }

    // Graceful shutdown with cleanup (flush databases, save indices)
    run_shutdown_cleanup(manager_for_shutdown).await;

    Ok(())
}

// =============================================================================
// Background Schedulers
// =============================================================================

fn start_maintenance_scheduler(manager: AppState, interval_secs: u64) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));

        // Skip first immediate tick — let server warm up before running maintenance
        interval.tick().await;

        loop {
            interval.tick().await;

            // Cleanup stale streaming sessions
            let extractor = manager.streaming_extractor().clone();
            let cleaned = extractor.cleanup_stale_sessions().await;
            if cleaned > 0 {
                tracing::debug!("Cleaned {} stale streaming sessions", cleaned);
            }

            // Cleanup stale user sessions
            let session_cleaned = manager.session_store().cleanup_stale_sessions();
            if session_cleaned > 0 {
                tracing::debug!("Ended {} stale user sessions", session_cleaned);
            }

            // Run maintenance in blocking thread pool
            let manager_clone = Arc::clone(&manager);
            tokio::task::spawn_blocking(move || {
                manager_clone.run_maintenance_all_users();
            });
        }
    });

    info!(
        "Background maintenance scheduler started (interval: {}s)",
        interval_secs
    );
}

fn start_backup_scheduler(manager: AppState, interval_secs: u64, max_backups: usize) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));

        // Skip first immediate tick
        interval.tick().await;

        loop {
            interval.tick().await;

            info!("Starting scheduled backup run...");
            let manager_clone = Arc::clone(&manager);
            let backed_up = tokio::task::spawn_blocking(move || {
                manager_clone.run_backup_all_users(max_backups)
            })
            .await
            .unwrap_or(0);

            if backed_up > 0 {
                info!("Scheduled backup completed: {} users backed up", backed_up);
            }
        }
    });

    info!(
        "Automatic backup scheduler started (interval: {}h, keep: {} backups)",
        interval_secs / 3600,
        max_backups
    );
}

fn start_reminder_scheduler(manager: AppState) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));

        // Skip first immediate tick — let server warm up
        interval.tick().await;

        loop {
            interval.tick().await;

            let manager_clone = Arc::clone(&manager);
            match tokio::task::spawn_blocking(move || manager_clone.check_and_emit_due_reminders())
                .await
            {
                Ok(triggered) => {
                    if triggered > 0 {
                        info!("Active reminder check: {} reminder(s) triggered", triggered);
                    }
                }
                Err(e) => {
                    error!("Reminder scheduler task panicked: {}", e);
                }
            }
        }
    });

    info!("Active reminder scheduler started (interval: 60s)");
}

// =============================================================================
// Shutdown Handling
// =============================================================================

/// Wait for shutdown signal (Ctrl+C or SIGTERM on Unix).
async fn shutdown_signal_with_drain() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("Shutdown signal received, starting graceful shutdown");
}

async fn run_shutdown_cleanup(manager: AppState) {
    info!("Proceeding with database flush...");

    let cleanup_future = async {
        // Flush databases (blocking operation, must use spawn_blocking)
        let manager_for_flush = Arc::clone(&manager);
        let flush_handle =
            tokio::task::spawn_blocking(move || manager_for_flush.flush_all_databases());

        match tokio::time::timeout(
            std::time::Duration::from_secs(DATABASE_FLUSH_TIMEOUT_SECS),
            flush_handle,
        )
        .await
        {
            Ok(Ok(Ok(()))) => info!("Databases flushed successfully"),
            Ok(Ok(Err(e))) => error!("Failed to flush databases: {}", e),
            Ok(Err(e)) => error!("Flush task panicked: {}", e),
            Err(_) => error!(
                "Database flush timed out after {}s",
                DATABASE_FLUSH_TIMEOUT_SECS
            ),
        }

        // Save vector indices (blocking operation, must use spawn_blocking)
        info!("Persisting vector indices...");
        let manager_for_save = Arc::clone(&manager);
        let save_handle =
            tokio::task::spawn_blocking(move || manager_for_save.save_all_vector_indices());

        match tokio::time::timeout(
            std::time::Duration::from_secs(VECTOR_INDEX_SAVE_TIMEOUT_SECS),
            save_handle,
        )
        .await
        {
            Ok(Ok(Ok(()))) => info!("Vector indices saved successfully"),
            Ok(Ok(Err(e))) => error!("Failed to save vector indices: {}", e),
            Ok(Err(e)) => error!("Save task panicked: {}", e),
            Err(_) => error!(
                "Vector index save timed out after {}s",
                VECTOR_INDEX_SAVE_TIMEOUT_SECS
            ),
        }

        #[cfg(feature = "telemetry")]
        tracing_setup::shutdown_tracing();
    };

    match tokio::time::timeout(
        std::time::Duration::from_secs(GRACEFUL_SHUTDOWN_TIMEOUT_SECS),
        cleanup_future,
    )
    .await
    {
        Ok(()) => info!("Server shutdown complete"),
        Err(_) => {
            error!(
                "Graceful shutdown timed out after {}s, forcing exit",
                GRACEFUL_SHUTDOWN_TIMEOUT_SECS
            );
            std::process::exit(1);
        }
    }
}

// =============================================================================
// Startup Output
// =============================================================================

fn print_banner() {
    eprintln!();
    eprintln!("  ╔═══════════════════════════════════════════════════╗");
    eprintln!(
        "  ║         🧠 Shodh-Memory Server v{}          ║",
        env!("CARGO_PKG_VERSION")
    );
    eprintln!("  ║       Cognitive Memory for AI Agents              ║");
    eprintln!("  ╚═══════════════════════════════════════════════════╝");
    eprintln!();
}

fn print_config(config: &ServerConfig) {
    eprintln!("  Configuration:");
    eprintln!(
        "     Mode:    {}",
        if config.is_production {
            "PRODUCTION"
        } else {
            "Development"
        }
    );
    eprintln!("     Host:    {}", config.host);
    eprintln!("     Port:    {}", config.port);
    eprintln!("     Storage: {}", config.storage_path.display());
    eprintln!();
}

fn print_storage_stats(storage_path: &std::path::Path) {
    if storage_path.exists() {
        let disk_usage = calculate_dir_size(storage_path);
        let user_count = count_user_directories(storage_path);
        eprintln!("  💾 Storage Statistics:");
        eprintln!(
            "     Location:  {}",
            storage_path
                .canonicalize()
                .unwrap_or_else(|_| storage_path.to_path_buf())
                .display()
        );
        eprintln!("     Disk used: {}", format_bytes(disk_usage));
        eprintln!("     Users:     {}", user_count);
        eprintln!();
    } else {
        eprintln!("  💾 Storage: New database (no existing data)");
        eprintln!();
    }
}

fn print_ready_message(addr: SocketAddr) {
    use std::io::Write;
    let _ = std::io::stderr().flush();
    eprintln!();
    eprintln!("  🚀 Server ready!");
    eprintln!("     API:       http://{}", addr);
    eprintln!("     Health:    http://{}/health", addr);
    eprintln!("     Stream:    ws://{}/api/stream", addr);
    #[cfg(feature = "zenoh")]
    {
        let zenoh_enabled = std::env::var("SHODH_ZENOH_ENABLED")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);
        if zenoh_enabled {
            let prefix =
                std::env::var("SHODH_ZENOH_PREFIX").unwrap_or_else(|_| "shodh".to_string());
            eprintln!(
                "     Zenoh:     {}/*/{{remember,recall,forget,stream,mission}}",
                prefix
            );
            eprintln!("     Fleet:     {}/fleet/*", prefix);
        }
    }
    eprintln!();
    eprintln!("  Press Ctrl+C to stop");
    eprintln!();
    let _ = std::io::stderr().flush();
}

// =============================================================================
// Helper Functions
// =============================================================================

fn calculate_dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                total += calculate_dir_size(&path);
            } else if let Ok(metadata) = entry.metadata() {
                total += metadata.len();
            }
        }
    }
    total
}

fn count_user_directories(path: &std::path::Path) -> usize {
    std::fs::read_dir(path)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| {
                    let name = e.file_name();
                    let name_str = name.to_string_lossy();
                    e.path().is_dir()
                        && name_str != "audit_logs"
                        && name_str != "backups"
                        && name_str != "feedback"
                        && name_str != "semantic_facts"
                        && name_str != "files"
                        && name_str != "prospective"
                        && name_str != "todos"
                })
                .count()
        })
        .unwrap_or(0)
}

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}
