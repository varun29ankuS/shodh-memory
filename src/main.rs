//! Shodh-Memory Server - Cognitive Memory for AI Agents
//!
//! Clean entry point using modular handlers architecture.
//! All HTTP handlers are in src/handlers/ modules.

use anyhow::Result;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::signal;
use tower::ServiceBuilder;
use tower_governor::{governor::GovernorConfigBuilder, GovernorLayer};
use tower_http::services::ServeDir;
use tracing::info;

use shodh_memory::{
    auth,
    config::ServerConfig,
    handlers::{self, AppState, MultiUserMemoryManager},
    metrics,
    middleware,
};

#[cfg(feature = "telemetry")]
use shodh_memory::tracing_setup;

// Timeout constants for graceful shutdown
const DATABASE_FLUSH_TIMEOUT_SECS: u64 = 30;
const VECTOR_INDEX_SAVE_TIMEOUT_SECS: u64 = 60;
const GRACEFUL_SHUTDOWN_TIMEOUT_SECS: u64 = 120;

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file if present
    let _ = dotenvy::dotenv();

    // Initialize tracing
    #[cfg(feature = "telemetry")]
    {
        tracing_setup::init_tracing().expect("Failed to initialize tracing");
    }
    #[cfg(not(feature = "telemetry"))]
    {
        if std::env::var("RUST_LOG").is_err() {
            std::env::set_var("RUST_LOG", "shodh_memory=info,tower_http=warn");
        }
        tracing_subscriber::fmt::init();
    }

    // Print startup banner
    print_banner();

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
    start_maintenance_scheduler(Arc::clone(&manager), server_config.maintenance_interval_secs);

    // Start backup scheduler if enabled
    if server_config.backup_enabled && server_config.backup_interval_secs > 0 {
        start_backup_scheduler(
            Arc::clone(&manager),
            server_config.backup_interval_secs,
            server_config.backup_max_count,
        );
    }

    // Configure rate limiting
    let governor_conf = GovernorConfigBuilder::default()
        .per_second(server_config.rate_limit_per_second)
        .burst_size(server_config.rate_limit_burst)
        .finish()
        .expect("Failed to build governor rate limiter configuration");
    let governor_layer = GovernorLayer::new(governor_conf);

    info!(
        "Rate limiting: {} req/sec, burst of {}",
        server_config.rate_limit_per_second, server_config.rate_limit_burst
    );

    // Build CORS layer
    let cors = server_config.cors.to_layer();

    // Build routes using handlers module
    let public_routes = handlers::build_public_routes(Arc::clone(&manager))
        .nest_service("/static", ServeDir::new("static"))
        .route(
            "/",
            axum::routing::get(|| async {
                axum::response::Redirect::permanent("/static/live.html")
            }),
        );

    let protected_routes = handlers::build_protected_routes(Arc::clone(&manager))
        .layer(axum::middleware::from_fn(auth::auth_middleware))
        .layer(governor_layer);

    // Combine routes with global middleware
    // Note: Routes already have state from build_public_routes/build_protected_routes
    let app = axum::Router::new()
        .merge(public_routes)
        .merge(protected_routes)
        .layer(
            ServiceBuilder::new()
                .layer(axum::middleware::from_fn(middleware::track_metrics))
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
    let port = server_config.port;
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    // Small delay for log flush
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    print_ready_message(addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;

    // Run server until shutdown signal
    tokio::select! {
        result = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        ) => {
            if let Err(e) = result {
                tracing::error!("Server error: {}", e);
            }
        }
        _ = shutdown_signal() => {
            info!("Shutdown signal received");
        }
    }

    // Graceful shutdown with cleanup
    run_shutdown_cleanup(manager_for_shutdown).await;

    Ok(())
}

// =============================================================================
// Background Schedulers
// =============================================================================

fn start_maintenance_scheduler(manager: AppState, interval_secs: u64) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));

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
                if let Err(e) = manager_clone.flush_all_databases() {
                    tracing::warn!("Periodic flush failed: {}", e);
                }
            });
        }
    });

    info!("Background maintenance scheduler started (interval: {}s)", interval_secs);
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

// =============================================================================
// Shutdown Handling
// =============================================================================

async fn shutdown_signal() {
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
        // Flush databases
        let flush_future = async { manager.flush_all_databases() };
        match tokio::time::timeout(
            std::time::Duration::from_secs(DATABASE_FLUSH_TIMEOUT_SECS),
            flush_future,
        )
        .await
        {
            Ok(Ok(())) => info!("Databases flushed successfully"),
            Ok(Err(e)) => tracing::error!("Failed to flush databases: {}", e),
            Err(_) => tracing::error!(
                "Database flush timed out after {}s",
                DATABASE_FLUSH_TIMEOUT_SECS
            ),
        }

        // Save vector indices
        info!("Persisting vector indices...");
        let save_future = async { manager.save_all_vector_indices() };
        match tokio::time::timeout(
            std::time::Duration::from_secs(VECTOR_INDEX_SAVE_TIMEOUT_SECS),
            save_future,
        )
        .await
        {
            Ok(Ok(())) => info!("Vector indices saved successfully"),
            Ok(Err(e)) => tracing::error!("Failed to save vector indices: {}", e),
            Err(_) => tracing::error!(
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
            tracing::error!(
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
    eprintln!("     HTTP:      http://{}", addr);
    eprintln!("     Health:    http://{}/health", addr);
    eprintln!("     Dashboard: http://{}/static/live.html", addr);
    eprintln!("     Stream:    ws://{}/api/stream", addr);
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
