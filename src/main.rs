//! Shodh-Memory Server — standalone binary entry point.
//!
//! Subcommands:
//!   serve     Start the memory server (default if no subcommand given)
//!   migrate   Migrate RocksDB data from bincode/msgpack to postcard
//!
//! Usage:
//!   shodh-memory-server [OPTIONS]                          # serve (default)
//!   shodh-memory-server serve [OPTIONS]                    # explicit serve
//!   shodh-memory-server migrate --storage <PATH> [--dry-run]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

const LONG_ABOUT: &str = r#"
Shodh-Memory is a cognitive memory system for AI agents, featuring:

  • 3-tier memory (Working → Session → LongTerm) with automatic promotion
  • Hebbian learning - memories that help get stronger, misleading ones decay
  • Knowledge graph with spreading activation for associative retrieval
  • Vector search (MiniLM embeddings + Vamana/DiskANN index)
  • 100% offline - no cloud, no API keys needed for core functionality

The server exposes a REST API for memory operations. After starting:

  Health check:  curl http://localhost:3030/health
  Store memory:  curl -X POST http://localhost:3030/api/remember \
                   -H "Content-Type: application/json" \
                   -H "X-API-Key: sk-shodh-dev-local-testing-key" \
                   -d '{"user_id":"test","content":"Hello world"}'
  Search:        curl -X POST http://localhost:3030/api/recall \
                   -H "Content-Type: application/json" \
                   -H "X-API-Key: sk-shodh-dev-local-testing-key" \
                   -d '{"user_id":"test","query":"hello"}'
"#;

const AFTER_HELP: &str = r#"
INTEGRATION:
  Unified CLI:   shodh server | shodh tui | shodh serve
  Claude Code:   claude mcp add shodh-memory -- npx -y @shodh/memory-mcp
  Python:        pip install shodh-memory
  TUI:           shodh tui

EXAMPLES:
  shodh-memory-server                          # Start with defaults
  shodh-memory-server serve -H 0.0.0.0 -p 8080  # Custom host and port
  shodh-memory-server migrate -s ./data --dry-run  # Preview migration

DOCUMENTATION:
  GitHub:  https://github.com/varun29ankuS/shodh-memory
"#;

/// Shodh-Memory Server - Cognitive Memory for AI Agents
#[derive(Parser)]
#[command(name = "shodh-memory-server")]
#[command(version, about, long_about = LONG_ABOUT, after_help = AFTER_HELP)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    // ── Flat args for backward compat (no subcommand = serve) ──
    /// Bind address (use 0.0.0.0 for network access)
    #[arg(
        short = 'H',
        long,
        env = "SHODH_HOST",
        default_value = "127.0.0.1",
        global = true
    )]
    host: String,

    /// Port number to listen on
    #[arg(short, long, env = "SHODH_PORT", default_value_t = 3030, global = true)]
    port: u16,

    /// Storage directory for RocksDB data
    #[arg(
        short,
        long = "storage",
        env = "SHODH_MEMORY_PATH",
        default_value_os_t = shodh_memory::config::default_storage_path(),
        global = true,
    )]
    storage_path: PathBuf,

    /// Production mode: stricter CORS, automatic backups enabled
    #[arg(long, env = "SHODH_ENV", global = true)]
    production: bool,

    /// Rate limit: max requests per second per client
    #[arg(long, env = "SHODH_RATE_LIMIT", default_value_t = 4000, global = true)]
    rate_limit: u64,

    /// Maximum concurrent requests before load shedding
    #[arg(
        long,
        env = "SHODH_MAX_CONCURRENT",
        default_value_t = 200,
        global = true
    )]
    max_concurrent: usize,
}

#[derive(Subcommand)]
enum Command {
    /// Start the memory server (default)
    Serve,
    /// Migrate RocksDB data from bincode/msgpack to postcard
    Migrate {
        /// Report what would be migrated without writing any data
        #[arg(long)]
        dry_run: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        None | Some(Command::Serve) => {
            shodh_memory::server::run(shodh_memory::server::ServerRunConfig {
                host: cli.host,
                port: cli.port,
                storage_path: cli.storage_path,
                production: cli.production,
                rate_limit: cli.rate_limit,
                max_concurrent: cli.max_concurrent,
            })
        }
        Some(Command::Migrate { dry_run }) => {
            eprintln!(
                "Shodh-Memory: migrating storage at {}{}",
                cli.storage_path.display(),
                if dry_run { " (dry run)" } else { "" }
            );

            let report = shodh_memory::migration::migrate_all(&cli.storage_path, dry_run)?;
            eprintln!("{report}");

            if !report.errors.is_empty() {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}
