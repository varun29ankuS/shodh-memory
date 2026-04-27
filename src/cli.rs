//! Unified shodh CLI — server, TUI, MCP, hooks, and management commands.
//!
//! Usage:
//!   shodh server              - Start the HTTP API server
//!   shodh tui                 - Launch the TUI dashboard
//!   shodh serve               - Run as MCP server (stdio transport)
//!   shodh init                - First-time setup wizard
//!   shodh status              - Check server health
//!   shodh doctor              - Diagnose common issues
//!   shodh hook session-start  - Output session start hook JSON
//!   shodh hook prompt <msg>   - Output prompt submit hook JSON
//!   shodh claude [args...]    - Launch Claude Code with Shodh memory
//!   shodh setup-hooks         - Print instructions for Claude Code hooks
//!   shodh version             - Print version and build info

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        CallToolResult, Content, ErrorCode, Implementation, ProtocolVersion, ServerCapabilities,
        ServerInfo,
    },
    schemars, tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, path::PathBuf, sync::Arc};

// =============================================================================
// CLI STRUCTURE
// =============================================================================

const LONG_ABOUT: &str = "\
Shodh — cognitive memory for AI agents.

One binary for everything: run the server, launch the TUI, serve MCP tools,
manage configuration, and diagnose issues.

Getting started:
  shodh init        Set up storage directory, API key, and ONNX runtime
  shodh server      Start the HTTP API server on localhost:3030
  shodh tui         Launch the terminal dashboard
  shodh status      Check if the server is running";

#[derive(Parser)]
#[command(name = "shodh")]
#[command(about = "Shodh — cognitive memory for AI agents")]
#[command(long_about = LONG_ABOUT)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the HTTP API server
    Server {
        /// Bind address (use 0.0.0.0 for network access)
        #[arg(short = 'H', long, env = "SHODH_HOST", default_value = "127.0.0.1")]
        host: String,

        /// Port number to listen on
        #[arg(short, long, env = "SHODH_PORT", default_value_t = 3030)]
        port: u16,

        /// Storage directory for RocksDB data
        #[arg(
            short,
            long = "storage",
            env = "SHODH_MEMORY_PATH",
            default_value_os_t = shodh_memory::config::default_storage_path()
        )]
        storage_path: PathBuf,

        /// Production mode: stricter CORS, automatic backups enabled
        #[arg(long, env = "SHODH_ENV")]
        production: bool,

        /// Rate limit: max requests per second per client (0 = disabled)
        #[arg(long, env = "SHODH_RATE_LIMIT", default_value_t = 4000)]
        rate_limit: u64,

        /// Maximum concurrent requests before load shedding
        #[arg(long, env = "SHODH_MAX_CONCURRENT", default_value_t = 200)]
        max_concurrent: usize,
    },

    /// Launch the TUI dashboard
    Tui {
        /// API URL for the memory server
        #[arg(
            long,
            env = "SHODH_SERVER_URL",
            default_value = "http://127.0.0.1:3030"
        )]
        api_url: String,

        /// API key for authentication
        #[arg(
            long,
            env = "SHODH_API_KEY",
            default_value = "sk-shodh-dev-local-testing-key"
        )]
        api_key: String,
    },

    /// Run as MCP server (stdio transport)
    Serve {
        /// API URL for the memory server
        #[arg(long, env = "SHODH_API_URL", default_value = "http://127.0.0.1:3030")]
        api_url: String,

        /// API key for authentication
        #[arg(
            long,
            env = "SHODH_API_KEY",
            default_value = "sk-shodh-dev-local-testing-key"
        )]
        api_key: String,

        /// User ID for memory operations
        #[arg(long, env = "SHODH_USER_ID", default_value = "claude-code")]
        user_id: String,
    },

    /// First-time setup — create config, generate API key, download models
    Init,

    /// Check server health and status
    Status {
        /// API URL for the memory server
        #[arg(long, env = "SHODH_API_URL", default_value = "http://127.0.0.1:3030")]
        api_url: String,

        /// API key for authentication
        #[arg(
            long,
            env = "SHODH_API_KEY",
            default_value = "sk-shodh-dev-local-testing-key"
        )]
        api_key: String,
    },

    /// Diagnose common issues (storage, ONNX, port, server health)
    Doctor,

    /// Output Claude Code hook JSON
    Hook {
        #[command(subcommand)]
        hook_type: HookType,
    },

    /// Launch Claude Code with Shodh memory proxy
    Claude {
        /// Port for the shodh-memory server
        #[arg(long, default_value = "3030")]
        port: u16,

        /// Additional arguments to pass to claude
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },

    /// Export knowledge graph as JSON or GEXF
    ExportGraph {
        /// User ID whose graph to export
        user_id: String,

        /// Output format: json or gexf
        #[arg(long, default_value = "json")]
        format: String,

        /// Node types to include (comma-separated: entities,memories,episodes)
        #[arg(long, default_value = "entities,memories,episodes")]
        include: String,

        /// Minimum importance threshold for memory nodes
        #[arg(long, default_value_t = 0.0)]
        min_importance: f32,

        /// Include embedding vectors
        #[arg(long)]
        include_embeddings: bool,

        /// Output file (stdout if omitted)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// API URL for the memory server
        #[arg(long, env = "SHODH_API_URL", default_value = "http://127.0.0.1:3030")]
        api_url: String,

        /// API key for authentication
        #[arg(
            long,
            env = "SHODH_API_KEY",
            default_value = "sk-shodh-dev-local-testing-key"
        )]
        api_key: String,
    },

    /// Print instructions to set up Claude Code hooks for automatic memory
    SetupHooks {
        /// Also print the settings.json snippet as raw JSON (for piping)
        #[arg(long)]
        json: bool,
    },

    /// Print version and build information
    Version,
}

#[derive(Subcommand)]
enum HookType {
    /// Session start hook - restore context
    SessionStart {
        /// API URL for the memory server
        #[arg(long, env = "SHODH_API_URL", default_value = "http://127.0.0.1:3030")]
        api_url: String,

        /// API key for authentication
        #[arg(
            long,
            env = "SHODH_API_KEY",
            default_value = "sk-shodh-dev-local-testing-key"
        )]
        api_key: String,

        /// User ID for memory operations
        #[arg(long, env = "SHODH_USER_ID", default_value = "claude-code")]
        user_id: String,

        /// Project directory (from CLAUDE_PROJECT_DIR)
        #[arg(long, env = "CLAUDE_PROJECT_DIR")]
        project_dir: Option<String>,
    },

    /// User prompt submit hook - inject relevant context
    Prompt {
        /// The user's message
        message: String,

        /// API URL for the memory server
        #[arg(long, env = "SHODH_API_URL", default_value = "http://127.0.0.1:3030")]
        api_url: String,

        /// API key for authentication
        #[arg(
            long,
            env = "SHODH_API_KEY",
            default_value = "sk-shodh-dev-local-testing-key"
        )]
        api_key: String,

        /// User ID for memory operations
        #[arg(long, env = "SHODH_USER_ID", default_value = "claude-code")]
        user_id: String,
    },
}

// =============================================================================
// MAIN
// =============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        // =====================================================================
        // NEW: shodh server — start the HTTP API server in-process
        // =====================================================================
        Commands::Server {
            host,
            port,
            storage_path,
            production,
            rate_limit,
            max_concurrent,
        } => {
            // server::run() builds its own tokio runtime, so we need to exit
            // the current one first. Drop the async context and call synchronously.
            // Since server::run() is blocking and builds its own runtime, we
            // spawn it in a blocking task and await.
            let result = tokio::task::spawn_blocking(move || {
                shodh_memory::server::run(shodh_memory::server::ServerRunConfig {
                    host,
                    port,
                    storage_path,
                    production,
                    rate_limit,
                    max_concurrent,
                })
            })
            .await?;
            result?;
        }

        // =====================================================================
        // NEW: shodh tui — launch the TUI dashboard
        // =====================================================================
        Commands::Tui { api_url, api_key } => {
            handle_tui(&api_url, &api_key)?;
        }

        // =====================================================================
        // EXISTING: shodh serve — MCP server
        // =====================================================================
        Commands::Serve {
            api_url,
            api_key,
            user_id,
        } => {
            eprintln!("Starting shodh MCP server...");
            eprintln!("  API URL: {}", api_url);
            eprintln!("  User ID: {}", user_id);

            let server = ShodhMcpServer::new(api_url, api_key, user_id);
            let service = server.serve(rmcp::transport::stdio()).await?;
            service.waiting().await?;
        }

        // =====================================================================
        // NEW: shodh init — first-time setup
        // =====================================================================
        Commands::Init => {
            handle_init()?;
        }

        // =====================================================================
        // NEW: shodh status — health check
        // =====================================================================
        Commands::Status { api_url, api_key } => {
            handle_status(&api_url, &api_key)?;
        }

        // =====================================================================
        // NEW: shodh doctor — diagnostics
        // =====================================================================
        Commands::Doctor => {
            handle_doctor()?;
        }

        // =====================================================================
        // EXISTING: shodh hook — Claude Code hooks
        // =====================================================================
        Commands::Hook { hook_type } => match hook_type {
            HookType::SessionStart {
                api_url,
                api_key,
                user_id,
                project_dir,
            } => {
                handle_session_start(&api_url, &api_key, &user_id, project_dir.as_deref());
            }

            HookType::Prompt {
                message,
                api_url,
                api_key,
                user_id,
            } => {
                handle_prompt_submit(&api_url, &api_key, &user_id, &message);
            }
        },

        // =====================================================================
        // EXISTING: shodh claude — launch Claude Code
        // =====================================================================
        Commands::Claude { port, args } => {
            handle_claude_launch(port, args).await?;
        }

        // =====================================================================
        // NEW: shodh export-graph — export knowledge graph
        // =====================================================================
        Commands::ExportGraph {
            user_id,
            format,
            include,
            min_importance,
            include_embeddings,
            output,
            api_url,
            api_key,
        } => {
            let url = format!(
                "{}/api/graph/{}/export?format={}&include={}&min_importance={}&include_embeddings={}",
                api_url, user_id, format, include, min_importance, include_embeddings
            );

            let client = reqwest::Client::new();
            let resp = client
                .get(&url)
                .header("x-api-key", &api_key)
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to connect to shodh server: {e}"))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                anyhow::bail!("Export failed (HTTP {status}): {body}");
            }

            let body = resp
                .text()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to read response: {e}"))?;

            match output {
                Some(path) => {
                    std::fs::write(&path, &body).map_err(|e| {
                        anyhow::anyhow!("Failed to write to {}: {e}", path.display())
                    })?;
                    eprintln!("Exported to {}", path.display());
                }
                None => {
                    println!("{body}");
                }
            }
        }

        // =====================================================================
        // NEW: shodh version — version and build info
        // =====================================================================
        // =====================================================================
        // NEW: shodh setup-hooks — print Claude Code hooks setup guide
        // =====================================================================
        Commands::SetupHooks { json } => {
            handle_setup_hooks(json);
        }

        Commands::Version => {
            handle_version();
        }
    }

    Ok(())
}

// =============================================================================
// NEW COMMAND HANDLERS
// =============================================================================

fn handle_tui(api_url: &str, api_key: &str) -> Result<()> {
    let exe_path = std::env::current_exe()?;
    let exe_dir = exe_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine executable directory"))?;

    let tui_name = if cfg!(windows) {
        "shodh-tui.exe"
    } else {
        "shodh-tui"
    };

    let tui_binary = exe_dir.join(tui_name);

    if !tui_binary.exists() {
        // Try PATH lookup
        let from_path = which_binary(tui_name);
        if let Some(path) = from_path {
            return exec_tui(&path, api_url, api_key);
        }

        eprintln!("Error: shodh-tui not found");
        eprintln!();
        eprintln!("  Looked in: {}", exe_dir.display());
        eprintln!("  Also checked: PATH");
        eprintln!();
        eprintln!("  If installed via brew:");
        eprintln!("    brew reinstall shodh-memory");
        eprintln!();
        eprintln!("  If installed from GitHub releases:");
        eprintln!("    Download the release archive — it includes shodh-tui");
        eprintln!("    Place shodh-tui alongside the shodh binary");
        std::process::exit(1);
    }

    exec_tui(&tui_binary, api_url, api_key)
}

fn exec_tui(tui_binary: &std::path::Path, api_url: &str, api_key: &str) -> Result<()> {
    let status = std::process::Command::new(tui_binary)
        .env("SHODH_SERVER_URL", api_url)
        .env("SHODH_API_KEY", api_key)
        .status()?;

    std::process::exit(status.code().unwrap_or(1));
}

fn which_binary(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let full = dir.join(name);
            if full.is_file() {
                Some(full)
            } else {
                None
            }
        })
    })
}

fn handle_init() -> Result<()> {
    eprintln!();
    eprintln!("  Shodh-Memory Setup");
    eprintln!("  ══════════════════");
    eprintln!();

    // 1. Create storage directory
    let storage = shodh_memory::config::default_storage_path();
    std::fs::create_dir_all(&storage)?;
    eprintln!("  ✓ Storage directory: {}", storage.display());

    // 2. Create config directory + generate API key
    let config_dir = config_directory();
    std::fs::create_dir_all(&config_dir)?;

    let config_path = config_dir.join("config.toml");
    if config_path.exists() {
        eprintln!("  ✓ Config exists: {}", config_path.display());
    } else {
        let api_key = generate_api_key();
        let config_content = format!(
            "# Shodh-Memory Configuration\n\
             # Generated by `shodh init`\n\
             \n\
             api_key = \"{api_key}\"\n\
             host = \"127.0.0.1\"\n\
             port = 3030\n\
             # storage = \"{}\"  # Uncomment to override default\n",
            storage.display()
        );
        std::fs::write(&config_path, config_content)?;
        eprintln!("  ✓ Config created: {}", config_path.display());
        eprintln!("  ✓ API key generated: {}", api_key);
    }

    // 3. Pre-download ONNX runtime + embedding model
    eprintln!();
    eprintln!("  Downloading ONNX runtime and embedding model...");
    eprintln!("  (this only happens once, ~40MB total)");
    eprintln!();
    shodh_memory::embeddings::minilm::pre_init_ort_runtime(false);
    eprintln!("  ✓ ONNX runtime ready");

    // 4. Print next steps
    eprintln!();
    eprintln!("  ╔═══════════════════════════════════════════╗");
    eprintln!("  ║           Setup complete!                  ║");
    eprintln!("  ╚═══════════════════════════════════════════╝");
    eprintln!();
    eprintln!("  Next steps:");
    eprintln!();
    eprintln!("    shodh server        Start the memory server");
    eprintln!("    shodh tui           Launch the TUI dashboard");
    eprintln!("    shodh status        Check server health");
    eprintln!();
    eprintln!("  For Claude Code / Cursor:");
    eprintln!();
    eprintln!("    claude mcp add shodh-memory -- npx -y @shodh/memory-mcp");
    eprintln!();
    eprintln!("  Documentation: https://www.shodh-memory.com/docs");
    eprintln!();

    Ok(())
}

fn handle_status(api_url: &str, api_key: &str) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()?;

    let resp = client
        .get(format!("{api_url}/health"))
        .header("X-API-Key", api_key)
        .send();

    match resp {
        Ok(r) if r.status().is_success() => {
            let body: serde_json::Value = r.json()?;

            eprintln!();
            eprintln!("  Shodh-Memory Server: RUNNING");
            eprintln!("  ════════════════════════════");
            eprintln!();

            if let Some(version) = body.get("version").and_then(|v| v.as_str()) {
                eprintln!("  Version:  {}", version);
            }
            if let Some(uptime) = body.get("uptime").and_then(|v| v.as_str()) {
                eprintln!("  Uptime:   {}", uptime);
            }
            if let Some(users) = body.get("active_users").and_then(|v| v.as_u64()) {
                eprintln!("  Users:    {}", users);
            }
            if let Some(memories) = body.get("total_memories").and_then(|v| v.as_u64()) {
                eprintln!("  Memories: {}", memories);
            }

            eprintln!("  URL:      {}", api_url);
            eprintln!();
        }
        Ok(r) => {
            eprintln!("  Server returned: {}", r.status());
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!();
            eprintln!("  Shodh-Memory Server: NOT RUNNING");
            eprintln!("  ════════════════════════════════");
            eprintln!();
            eprintln!("  Could not connect to {}", api_url);
            eprintln!();
            eprintln!("  Start it with:");
            eprintln!("    shodh server");
            eprintln!();
            eprintln!("  Or diagnose with:");
            eprintln!("    shodh doctor");
            eprintln!();
            std::process::exit(1);
        }
    }

    Ok(())
}

fn handle_doctor() -> Result<()> {
    eprintln!();
    eprintln!("  Shodh-Memory Doctor");
    eprintln!("  ═══════════════════");
    eprintln!();

    let mut all_ok = true;

    // 1. Storage directory
    let storage = shodh_memory::config::default_storage_path();
    if storage.exists() {
        eprintln!("  ✓ Storage directory exists: {}", storage.display());
    } else {
        match std::fs::create_dir_all(&storage) {
            Ok(()) => {
                eprintln!("  ✓ Storage directory created: {}", storage.display());
                // Clean up — we just tested writability
                let _ = std::fs::remove_dir(&storage);
            }
            Err(e) => {
                eprintln!(
                    "  ✗ Storage directory NOT writable: {} ({})",
                    storage.display(),
                    e
                );
                all_ok = false;
            }
        }
    }

    // 2. Config
    let config_dir = config_directory();
    let config_path = config_dir.join("config.toml");
    if config_path.exists() {
        eprintln!("  ✓ Config file exists: {}", config_path.display());
    } else {
        eprintln!("  ⚠ No config file (run `shodh init` to create one)");
    }

    // 3. ONNX runtime
    eprintln!("  … Checking ONNX runtime");
    shodh_memory::embeddings::minilm::pre_init_ort_runtime(false);
    eprintln!("  ✓ ONNX runtime loads OK");

    // 4. Port availability
    let port: u16 = std::env::var("SHODH_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3030);

    let addr = format!("127.0.0.1:{port}");
    match std::net::TcpListener::bind(&addr) {
        Ok(_listener) => {
            eprintln!("  ✓ Port {port} is available");
        }
        Err(_) => {
            // Port in use — could be our server or something else
            eprintln!("  ⚠ Port {port} is in use");
        }
    }

    // 5. Server health (if running)
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;

    match client.get(format!("http://127.0.0.1:{port}/health")).send() {
        Ok(r) if r.status().is_success() => {
            let body: serde_json::Value = r.json().unwrap_or_default();
            let version = body
                .get("version")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            eprintln!("  ✓ Server is running (v{version})");
        }
        Ok(r) => {
            eprintln!("  ⚠ Port {port} responds but returned {}", r.status());
            all_ok = false;
        }
        Err(_) => {
            eprintln!("  ⚠ Server is not running on port {port}");
        }
    }

    // Summary
    eprintln!();
    if all_ok {
        eprintln!("  All checks passed.");
    } else {
        eprintln!("  Some issues found. Run `shodh init` to fix setup issues.");
    }
    eprintln!();

    Ok(())
}

fn handle_version() {
    eprintln!("shodh {}", env!("CARGO_PKG_VERSION"));
    eprintln!("  Platform: {}", std::env::consts::OS);
    eprintln!("  Arch:     {}", std::env::consts::ARCH);
    eprintln!("  License:  Apache-2.0");
    eprintln!("  Repo:     https://github.com/varun29ankuS/shodh-memory");
}

fn handle_setup_hooks(json: bool) {
    let version = env!("CARGO_PKG_VERSION");
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"));
    let claude_dir = home.join(".claude");
    let hooks_dest = claude_dir
        .join("hooks")
        .join("shodh-memory")
        .join("memory-hook.ts");
    let settings_path = claude_dir.join("settings.json");

    // Use forward slashes everywhere for cross-platform hook commands
    let hook_path_str = hooks_dest.to_string_lossy().replace('\\', "/");

    let hook_url = format!(
        "https://raw.githubusercontent.com/varun29ankuS/shodh-memory/v{version}/hooks/memory-hook.ts"
    );

    if json {
        // Machine-readable: print just the settings.json hooks snippet
        let snippet = build_hooks_json(&hook_path_str);
        println!("{snippet}");
        return;
    }

    eprintln!();
    eprintln!("  Shodh-Memory — Claude Code Hooks Setup");
    eprintln!("  ═══════════════════════════════════════");
    eprintln!();
    eprintln!("  Follow these steps to enable automatic memory capture.");
    eprintln!();

    // Step 1: Prerequisites
    eprintln!("  1. Install bun (needed to run the hook script):");
    eprintln!();
    if cfg!(target_os = "windows") {
        eprintln!("     powershell -c \"irm bun.sh/install.ps1 | iex\"");
    } else {
        eprintln!("     curl -fsSL https://bun.sh/install | bash");
    }
    eprintln!();

    // Step 2: Download hook file
    eprintln!("  2. Download the hook script:");
    eprintln!();
    eprintln!("     mkdir -p {}", hooks_dest.parent().unwrap().display());
    if cfg!(target_os = "windows") {
        eprintln!("     curl -fsSL -o \"{}\" \\", hooks_dest.to_string_lossy());
    } else {
        eprintln!("     curl -fsSL -o {} \\", hooks_dest.display());
    }
    eprintln!("       {hook_url}");
    eprintln!();

    // Step 3: settings.json
    eprintln!("  3. Add hooks to {}", settings_path.display());
    eprintln!();
    eprintln!("     Merge the following into your settings.json \"hooks\" object.");
    eprintln!("     (Run `shodh setup-hooks --json` for copy-paste JSON.)");
    eprintln!();

    // Print compact hook overview
    let events = [
        "SessionStart",
        "UserPromptSubmit",
        "Stop",
        "PreToolUse",
        "PostToolUse",
        "SubagentStop",
    ];
    for event in &events {
        eprintln!("     {event}: bun run {hook_path_str} {event}");
    }
    eprintln!();

    // Step 4: Verify
    eprintln!("  4. Verify:");
    eprintln!();
    eprintln!("     Start a new Claude Code session — you should see");
    eprintln!("     \"SessionStart hook success\" in the system output.");
    eprintln!();

    // Quick path
    eprintln!("  ─── Or use the npm installer (does all steps automatically) ───");
    eprintln!();
    eprintln!("     npx @shodh/memory-mcp setup-hooks");
    eprintln!();
}

fn build_hooks_json(hook_path: &str) -> String {
    let events = [
        ("SessionStart", "{}"),
        ("UserPromptSubmit", "{}"),
        ("Stop", "{}"),
        ("PreToolUse", r#"{"tool_name": ["Edit", "Write", "Bash"]}"#),
        (
            "PostToolUse",
            r#"{"tool_name": ["Edit", "Write", "Bash", "TodoWrite", "Read", "Task"]}"#,
        ),
        ("SubagentStop", "{}"),
    ];

    let mut entries = Vec::new();
    for (event, matcher) in &events {
        entries.push(format!(
            r#"    "{event}": [
      {{
        "matcher": {matcher},
        "hooks": [{{"type": "command", "command": "bun run {hook_path} {event}"}}]
      }}
    ]"#
        ));
    }

    format!("{{\n  \"hooks\": {{\n{}\n  }}\n}}", entries.join(",\n"))
}

// =============================================================================
// UTILITY HELPERS
// =============================================================================

fn config_directory() -> PathBuf {
    dirs::config_dir()
        .map(|d| d.join("shodh"))
        .unwrap_or_else(|| PathBuf::from(".shodh"))
}

fn generate_api_key() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    // Simple deterministic key — good enough for local dev, user can change later
    format!("sk-shodh-{:x}", timestamp)
}

// =============================================================================
// API CLIENT
// =============================================================================

/// HTTP client for the shodh-memory API (async version for MCP tools)
#[derive(Clone, Debug)]
struct AsyncApiClient {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    user_id: String,
}

impl AsyncApiClient {
    fn new(base_url: String, api_key: String, user_id: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url,
            api_key,
            user_id,
        }
    }

    async fn post<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        body: &T,
    ) -> Result<R> {
        let url = format!("{}{endpoint}", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("X-API-Key", &self.api_key)
            .json(body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("API error {status}: {text}");
        }

        Ok(resp.json().await?)
    }
}

/// HTTP client for the shodh-memory API (blocking version for hooks)
#[derive(Clone, Debug)]
struct BlockingApiClient {
    client: reqwest::blocking::Client,
    base_url: String,
    api_key: String,
}

impl BlockingApiClient {
    fn new(base_url: String, api_key: String) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url,
            api_key,
        }
    }

    fn post<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        body: &T,
    ) -> Result<R> {
        let url = format!("{}{endpoint}", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("X-API-Key", &self.api_key)
            .json(body)
            .send()?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            anyhow::bail!("API error {status}: {text}");
        }

        Ok(resp.json()?)
    }
}

// =============================================================================
// API REQUEST/RESPONSE TYPES
// =============================================================================

#[derive(Serialize)]
struct ProactiveContextRequest {
    user_id: String,
    context: String,
    max_results: u32,
    auto_ingest: bool,
}

#[derive(Deserialize)]
struct ProactiveContextResponse {
    memories: Vec<SurfacedMemory>,
}

#[derive(Deserialize)]
struct SurfacedMemory {
    id: String,
    content: String,
    memory_type: String,
    relevance_score: f32,
}

#[derive(Serialize)]
struct ListTodosRequest {
    user_id: String,
    status: Vec<String>,
}

#[derive(Deserialize)]
struct ListTodosResponse {
    todos: Vec<Todo>,
}

#[derive(Deserialize)]
struct Todo {
    #[allow(dead_code)]
    id: String,
    content: String,
    status: String,
    priority: Option<String>,
    #[allow(dead_code)]
    project: Option<String>,
}

#[derive(Serialize)]
struct RememberRequest {
    user_id: String,
    content: String,
    memory_type: Option<String>,
    tags: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct RememberResponse {
    id: String,
    message: String,
}

#[derive(Serialize)]
struct RecallRequest {
    user_id: String,
    query: String,
    limit: Option<u32>,
    mode: Option<String>,
}

#[derive(Deserialize)]
struct RecallResponse {
    memories: Vec<RecalledMemory>,
}

#[derive(Deserialize)]
struct RecalledMemory {
    id: String,
    content: String,
    memory_type: String,
    similarity: f32,
    #[allow(dead_code)]
    tags: Vec<String>,
}

// =============================================================================
// HOOK OUTPUT
// =============================================================================

#[derive(Serialize)]
struct HookOutput {
    #[serde(rename = "hookSpecificOutput")]
    hook_specific_output: HookSpecificOutput,
}

#[derive(Serialize)]
struct HookSpecificOutput {
    #[serde(rename = "hookEventName")]
    hook_event_name: String,
    #[serde(rename = "additionalContext")]
    additional_context: String,
}

fn output_hook(event_name: &str, context: &str) {
    let output = HookOutput {
        hook_specific_output: HookSpecificOutput {
            hook_event_name: event_name.to_string(),
            additional_context: context.to_string(),
        },
    };
    println!("{}", serde_json::to_string(&output).unwrap());
}

// =============================================================================
// HOOK HANDLERS
// =============================================================================

fn handle_session_start(api_url: &str, api_key: &str, user_id: &str, project_dir: Option<&str>) {
    let client = BlockingApiClient::new(api_url.to_string(), api_key.to_string());

    let dir_name = project_dir
        .and_then(|p| std::path::Path::new(p).file_name())
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Get proactive context
    let context_result: Result<ProactiveContextResponse> = client.post(
        "/api/proactive_context",
        &ProactiveContextRequest {
            user_id: user_id.to_string(),
            context: format!("Starting session in {dir_name}"),
            max_results: 3,
            auto_ingest: false,
        },
    );

    // Get pending todos
    let todos_result: Result<ListTodosResponse> = client.post(
        "/api/todos",
        &ListTodosRequest {
            user_id: user_id.to_string(),
            status: vec!["todo".to_string(), "in_progress".to_string()],
        },
    );

    // Build context string
    let mut context_parts = vec!["## Shodh Memory Context Restored\n".to_string()];

    if let Ok(ctx) = context_result {
        if !ctx.memories.is_empty() {
            context_parts.push("### Relevant Memories:".to_string());
            for mem in ctx.memories.iter().take(3) {
                context_parts.push(format!(
                    "- [{}] {}: {}",
                    mem.memory_type,
                    &mem.id[..8.min(mem.id.len())],
                    mem.content.chars().take(200).collect::<String>()
                ));
            }
            context_parts.push(String::new());
        }
    }

    if let Ok(todos) = todos_result {
        let in_progress: Vec<_> = todos
            .todos
            .iter()
            .filter(|t| t.status == "in_progress")
            .collect();
        let pending: Vec<_> = todos.todos.iter().filter(|t| t.status == "todo").collect();

        if !in_progress.is_empty() || !pending.is_empty() {
            context_parts.push("### Pending Todos:".to_string());

            if !in_progress.is_empty() {
                context_parts.push("**In Progress:**".to_string());
                for todo in in_progress.iter().take(5) {
                    context_parts.push(format!("- ⏳ {}", todo.content));
                }
            }

            if !pending.is_empty() {
                context_parts.push("**Todo:**".to_string());
                for todo in pending.iter().take(5) {
                    let priority = todo.priority.as_deref().unwrap_or("");
                    let prefix = match priority {
                        "urgent" => "🔴",
                        "high" => "🟠",
                        "medium" => "🟡",
                        _ => "⚪",
                    };
                    context_parts.push(format!("- {} {}", prefix, todo.content));
                }
            }
        }
    }

    output_hook("SessionStart", &context_parts.join("\n"));
}

fn handle_prompt_submit(api_url: &str, api_key: &str, user_id: &str, message: &str) {
    let client = BlockingApiClient::new(api_url.to_string(), api_key.to_string());

    // Get proactive context based on user message
    let context_result: Result<ProactiveContextResponse> = client.post(
        "/api/proactive_context",
        &ProactiveContextRequest {
            user_id: user_id.to_string(),
            context: message.to_string(),
            max_results: 5,
            auto_ingest: true,
        },
    );

    let mut context_parts = Vec::new();

    if let Ok(ctx) = context_result {
        if !ctx.memories.is_empty() {
            context_parts.push("## Relevant Memories (auto-surfaced)\n".to_string());
            for mem in ctx.memories.iter() {
                let relevance = (mem.relevance_score * 100.0) as u32;
                context_parts.push(format!(
                    "- [{}%] **{}**: {}",
                    relevance,
                    mem.memory_type,
                    mem.content.chars().take(300).collect::<String>()
                ));
            }
        }
    }

    if !context_parts.is_empty() {
        output_hook("UserPromptSubmit", &context_parts.join("\n"));
    } else {
        output_hook("UserPromptSubmit", "");
    }
}

// =============================================================================
// MCP TOOL PARAMETER TYPES
// =============================================================================

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RememberParams {
    /// The content to remember
    content: String,
    /// Type of memory (Observation, Decision, Learning, etc.)
    #[serde(rename = "type")]
    memory_type: Option<String>,
    /// Optional tags for categorization
    tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RecallParams {
    /// Natural language search query
    query: String,
    /// Maximum number of results (default: 5)
    limit: Option<u32>,
    /// Retrieval mode: semantic, associative, or hybrid
    mode: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ProactiveContextParams {
    /// Current conversation context
    context: String,
    /// Maximum memories to surface (default: 5)
    max_results: Option<u32>,
    /// Auto-store context for feedback (default: true)
    auto_ingest: Option<bool>,
}

// =============================================================================
// LINEAGE MCP TOOL PARAMETERS
// =============================================================================

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LineageTraceParams {
    /// Memory ID to trace lineage from
    memory_id: String,
    /// Direction: "backward" (find causes), "forward" (find effects), "both"
    direction: Option<String>,
    /// Maximum depth to traverse (default: 10)
    max_depth: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LineageConfirmParams {
    /// ID of the inferred edge to confirm
    edge_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LineageRejectParams {
    /// ID of the inferred edge to reject
    edge_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LineageLinkParams {
    /// Source memory ID (the cause/origin)
    from_memory_id: String,
    /// Target memory ID (the effect/result)
    to_memory_id: String,
    /// Relation type: Caused, ResolvedBy, InformedBy, SupersededBy, TriggeredBy, BranchedFrom, RelatedTo
    relation: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LineageStatsParams {
    /// Optional - leave empty to get stats for current user
    #[serde(default)]
    _placeholder: Option<String>,
}

// Lineage API request types
#[derive(Serialize)]
struct LineageTraceRequest {
    user_id: String,
    memory_id: String,
    direction: String,
    max_depth: u32,
}

#[derive(Serialize)]
struct LineageEdgeRequest {
    user_id: String,
    edge_id: String,
}

#[derive(Serialize)]
struct LineageAddEdgeRequest {
    user_id: String,
    from_memory_id: String,
    to_memory_id: String,
    relation: String,
}

#[derive(Serialize)]
struct LineageStatsRequest {
    user_id: String,
}

// Lineage API response types
#[derive(Deserialize)]
struct LineageTraceResponse {
    root: String,
    direction: String,
    edges: Vec<LineageEdgeInfo>,
    path: Vec<String>,
    depth: usize,
}

#[derive(Deserialize)]
struct LineageEdgeInfo {
    #[allow(dead_code)]
    id: String,
    from: String,
    to: String,
    relation: String,
    confidence: f32,
    source: String,
}

#[derive(Deserialize)]
struct LineageConfirmResponse {
    message: String,
    edge_id: String,
}

#[derive(Deserialize)]
struct LineageRejectResponse {
    message: String,
    #[allow(dead_code)]
    deleted: bool,
}

#[derive(Deserialize)]
struct LineageAddResponse {
    message: String,
    edge_id: String,
}

#[derive(Deserialize)]
struct LineageStatsResponse {
    total_edges: usize,
    inferred_edges: usize,
    confirmed_edges: usize,
    explicit_edges: usize,
    total_branches: usize,
    active_branches: usize,
    edges_by_relation: std::collections::HashMap<String, usize>,
    avg_confidence: f32,
}

// =============================================================================
// MCP SERVER
// =============================================================================

#[derive(Debug, Clone)]
struct ShodhMcpServer {
    client: Arc<AsyncApiClient>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl ShodhMcpServer {
    fn new(api_url: String, api_key: String, user_id: String) -> Self {
        Self {
            client: Arc::new(AsyncApiClient::new(api_url, api_key, user_id)),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Store a memory for future recall. Use this to remember important information, decisions, user preferences, project context, or anything you want to recall later."
    )]
    async fn remember(
        &self,
        Parameters(params): Parameters<RememberParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<RememberResponse> = self
            .client
            .post(
                "/api/remember",
                &RememberRequest {
                    user_id: self.client.user_id.clone(),
                    content: params.content,
                    memory_type: params.memory_type,
                    tags: params.tags,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Stored memory: {} ({})",
                resp.id, resp.message
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Search memories using semantic, associative, or hybrid retrieval. Modes: 'semantic' (vector similarity), 'associative' (graph traversal), 'hybrid' (combines both)."
    )]
    async fn recall(
        &self,
        Parameters(params): Parameters<RecallParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<RecallResponse> = self
            .client
            .post(
                "/api/recall",
                &RecallRequest {
                    user_id: self.client.user_id.clone(),
                    query: params.query,
                    limit: params.limit,
                    mode: params.mode,
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = format!("Found {} memories:\n\n", resp.memories.len());
                for mem in resp.memories {
                    output.push_str(&format!(
                        "**[{}]** {} (similarity: {:.0}%)\n{}\n\n",
                        mem.memory_type,
                        &mem.id[..8.min(mem.id.len())],
                        mem.similarity * 100.0,
                        mem.content
                    ));
                }
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "REQUIRED: Call this to surface relevant memories based on current context. Enables automatic memory surfacing and implicit feedback learning."
    )]
    async fn proactive_context(
        &self,
        Parameters(params): Parameters<ProactiveContextParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<ProactiveContextResponse> = self
            .client
            .post(
                "/api/proactive_context",
                &ProactiveContextRequest {
                    user_id: self.client.user_id.clone(),
                    context: params.context,
                    max_results: params.max_results.unwrap_or(5),
                    auto_ingest: params.auto_ingest.unwrap_or(true),
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = format!("Surfaced {} relevant memories:\n\n", resp.memories.len());
                for mem in resp.memories {
                    output.push_str(&format!(
                        "- [{}%] **{}**: {}\n",
                        (mem.relevance_score * 100.0) as u32,
                        mem.memory_type,
                        mem.content.chars().take(200).collect::<String>()
                    ));
                }
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    // =========================================================================
    // LINEAGE TOOLS - Causal Memory Tracking
    // =========================================================================

    #[tool(
        description = "Trace the causal lineage of a memory. Find what caused it (backward), what it led to (forward), or both. Useful for understanding 'why' something happened."
    )]
    async fn lineage_trace(
        &self,
        Parameters(params): Parameters<LineageTraceParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageTraceResponse> = self
            .client
            .post(
                "/api/lineage/trace",
                &LineageTraceRequest {
                    user_id: self.client.user_id.clone(),
                    memory_id: params.memory_id,
                    direction: params.direction.unwrap_or_else(|| "backward".to_string()),
                    max_depth: params.max_depth.unwrap_or(10),
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = format!(
                    "**Lineage Trace** ({})\n\nRoot: {}\nDepth: {}\n\n",
                    resp.direction, resp.root, resp.depth
                );

                if resp.edges.is_empty() {
                    output.push_str("No causal connections found.\n");
                } else {
                    output.push_str("**Causal Chain:**\n");
                    for edge in &resp.edges {
                        let confidence = (edge.confidence * 100.0) as u32;
                        let source_icon = match edge.source.as_str() {
                            "Confirmed" => "✓",
                            "Explicit" => "⚡",
                            _ => "?",
                        };
                        output.push_str(&format!(
                            "  {} --[{} {}% {}]--> {}\n",
                            &edge.from[..8.min(edge.from.len())],
                            edge.relation,
                            confidence,
                            source_icon,
                            &edge.to[..8.min(edge.to.len())]
                        ));
                    }

                    output.push_str(&format!("\n**Path:** {}\n", resp.path.join(" → ")));
                }

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Confirm an inferred causal relationship between memories. This improves the system's confidence and learning."
    )]
    async fn lineage_confirm(
        &self,
        Parameters(params): Parameters<LineageConfirmParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageConfirmResponse> = self
            .client
            .post(
                "/api/lineage/confirm",
                &LineageEdgeRequest {
                    user_id: self.client.user_id.clone(),
                    edge_id: params.edge_id,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "✓ Confirmed edge: {} - {}",
                resp.edge_id, resp.message
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Reject an incorrectly inferred causal relationship. This helps the system learn better inference patterns."
    )]
    async fn lineage_reject(
        &self,
        Parameters(params): Parameters<LineageRejectParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageRejectResponse> = self
            .client
            .post(
                "/api/lineage/reject",
                &LineageEdgeRequest {
                    user_id: self.client.user_id.clone(),
                    edge_id: params.edge_id,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "✗ Rejected edge: {}",
                resp.message
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Create an explicit causal link between two memories. Relations: Caused (Error→Todo), ResolvedBy (Todo→Learning), InformedBy, SupersededBy, TriggeredBy, BranchedFrom, RelatedTo."
    )]
    async fn lineage_link(
        &self,
        Parameters(params): Parameters<LineageLinkParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageAddResponse> = self
            .client
            .post(
                "/api/lineage/link",
                &LineageAddEdgeRequest {
                    user_id: self.client.user_id.clone(),
                    from_memory_id: params.from_memory_id,
                    to_memory_id: params.to_memory_id,
                    relation: params.relation,
                },
            )
            .await;

        match result {
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "⚡ Created link: {} - {}",
                resp.edge_id, resp.message
            ))])),
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }

    #[tool(
        description = "Get statistics about the causal lineage graph - edge counts, relation types, confidence distribution."
    )]
    async fn lineage_stats(
        &self,
        Parameters(_params): Parameters<LineageStatsParams>,
    ) -> Result<CallToolResult, McpError> {
        let result: Result<LineageStatsResponse> = self
            .client
            .post(
                "/api/lineage/stats",
                &LineageStatsRequest {
                    user_id: self.client.user_id.clone(),
                },
            )
            .await;

        match result {
            Ok(resp) => {
                let mut output = "**Lineage Graph Statistics**\n\n".to_string();
                output.push_str(&format!("**Edges:** {}\n", resp.total_edges));
                output.push_str(&format!("  ✓ Confirmed: {}\n", resp.confirmed_edges));
                output.push_str(&format!("  ? Inferred: {}\n", resp.inferred_edges));
                output.push_str(&format!("  ⚡ Explicit: {}\n", resp.explicit_edges));
                output.push_str(&format!(
                    "Average Confidence: {:.1}%\n\n",
                    resp.avg_confidence * 100.0
                ));
                output.push_str(&format!(
                    "**Branches:** {} total, {} active\n\n",
                    resp.total_branches, resp.active_branches
                ));

                if !resp.edges_by_relation.is_empty() {
                    output.push_str("**By Relation Type:**\n");
                    let mut relations: Vec<_> = resp.edges_by_relation.iter().collect();
                    relations.sort_by(|a, b| b.1.cmp(a.1));
                    for (relation, count) in relations {
                        output.push_str(&format!("  {}: {}\n", relation, count));
                    }
                }

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError {
                code: ErrorCode::INTERNAL_ERROR,
                message: Cow::from(e.to_string()),
                data: None,
            }),
        }
    }
}

#[tool_handler]
impl ServerHandler for ShodhMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some(
                "Shodh Memory - persistent cognitive memory with causal reasoning. \
                 Use proactive_context at session start to surface relevant memories. \
                 Use remember to store decisions, learnings, errors. \
                 Use recall to search memories. \
                 Use lineage_trace to understand 'why' - trace causal chains backward/forward. \
                 Use lineage_link to explicitly connect cause→effect memories. \
                 Use lineage_confirm/reject to improve inference accuracy."
                    .to_string(),
            ),
        }
    }
}

// =============================================================================
// CLAUDE LAUNCH
// =============================================================================

/// Launch Claude Code with Shodh Cortex proxy
async fn handle_claude_launch(port: u16, args: Vec<String>) -> Result<()> {
    let server_url = format!("http://127.0.0.1:{port}");

    // Check if server is running
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;

    let health_url = format!("{server_url}/health");
    let server_running = client.get(&health_url).send().await.is_ok();

    if !server_running {
        eprintln!("🐘 Starting shodh-memory server on port {port}...");

        // Start server in background
        let exe_path = std::env::current_exe()?;
        let server_binary = exe_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Cannot find executable directory"))?
            .join("shodh-memory-server");

        #[cfg(windows)]
        let server_binary = server_binary.with_extension("exe");

        if !server_binary.exists() {
            eprintln!("⚠️  shodh-memory-server not found at {:?}", server_binary);
            eprintln!("   Please ensure shodh-memory-server is installed and in PATH");
            std::process::exit(1);
        }

        let mut cmd = std::process::Command::new(&server_binary);
        cmd.env("SHODH_PORT", port.to_string());

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            const DETACHED_PROCESS: u32 = 0x00000008;
            cmd.creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS);
        }

        #[allow(clippy::zombie_processes)]
        cmd.spawn().expect("Failed to start shodh-memory-server");

        // Wait for server to be ready
        eprintln!("   Waiting for server to be ready...");
        let mut server_ready = false;
        for _ in 0..30 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            if client.get(&health_url).send().await.is_ok() {
                eprintln!("   ✓ Server ready");
                server_ready = true;
                break;
            }
        }
        if !server_ready {
            eprintln!("   ✗ Server failed to start within 3 seconds on port {port}");
            std::process::exit(1);
        }
    } else {
        eprintln!("🐘 Shodh-memory server already running on port {port}");
    }

    // Launch claude with ANTHROPIC_API_BASE pointing to Cortex proxy
    eprintln!("🚀 Launching Claude Code with Shodh Cortex...");
    eprintln!("   ANTHROPIC_API_BASE={}", server_url);
    eprintln!();

    let mut claude_cmd = std::process::Command::new("claude");
    claude_cmd.env("ANTHROPIC_API_BASE", &server_url);
    claude_cmd.args(&args);

    // Replace current process with claude
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = claude_cmd.exec();
        eprintln!("Failed to exec claude: {}", err);
        std::process::exit(1);
    }

    #[cfg(windows)]
    {
        let mut cmd = std::process::Command::new("cmd");
        cmd.arg("/c").arg("claude");
        cmd.env("ANTHROPIC_API_BASE", &server_url);
        cmd.args(&args);
        let status = cmd.status()?;
        std::process::exit(status.code().unwrap_or(1));
    }
}
