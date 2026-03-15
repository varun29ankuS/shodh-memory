<p align="center">
  <img src="https://raw.githubusercontent.com/varun29ankuS/shodh-memory/main/assets/logo.png" width="120" alt="Shodh-Memory">
</p>

<h1 align="center">Shodh-Memory MCP Server</h1>

<p align="center">
  <strong>v0.1.90</strong> | Persistent cognitive memory for AI agents
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/@shodh/memory-mcp"><img src="https://img.shields.io/npm/v/@shodh/memory-mcp" alt="npm"></a>
  <a href="https://github.com/varun29ankuS/shodh-memory/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</p>

<p align="center">
  <a href="https://www.shodh-rag.com/memory">Documentation</a> |
  <a href="https://github.com/varun29ankuS/shodh-memory">GitHub</a> |
  <a href="https://pypi.org/project/shodh-memory/">Python SDK</a> |
  <a href="https://crates.io/crates/shodh-memory">Rust Crate</a>
</p>

---

## Features

- **Cognitive Architecture**: 3-tier memory (working, session, long-term) based on Cowan's model
- **Hebbian Learning**: "Neurons that fire together wire together" - associations strengthen with use
- **Semantic Search**: Find memories by meaning using MiniLM-L6 embeddings
- **Knowledge Graph**: Entity extraction and relationship tracking
- **Memory Consolidation**: Automatic decay, replay, and strengthening
- **Idempotent**: Content-hash dedup — identical memories are never stored twice
- **1-Click Install**: Auto-downloads native server binary for your platform
- **Offline-First**: All models auto-downloaded on first run (~38MB total), no internet required after
- **Fast**: <200ms API response, sub-millisecond graph lookup, 30-50ms semantic search
- **GTD Task Management**: Full todo system with projects, subtasks, comments, and reminders

## Installation

Add to your MCP client config:

**Claude Desktop / Claude Code** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "shodh-memory": {
      "command": "npx",
      "args": ["-y", "@shodh/memory-mcp"],
      "env": {
        "SHODH_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Config file locations:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Codex CLI** (`.codex/config.toml`):
```toml
[mcp_servers.shodh-memory]
startup_timeout_sec = 60
command = "npx"
args = ["-y", "@shodh/memory-mcp"]
env = { SHODH_API_KEY = "your-api-key-here" }
```

> **Note**: First run downloads the server binary (~15MB) plus embedding model (~23MB). The `startup_timeout_sec = 60` ensures enough time for initial setup.

**For Cursor/other MCP clients**: Similar configuration with the npx command.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SHODH_API_KEY` | **Required**. API key for authentication | - |
| `SHODH_API_URL` | Backend server URL | `http://127.0.0.1:3030` |
| `SHODH_USER_ID` | User ID for memory isolation | `claude-code` |
| `SHODH_NO_AUTO_SPAWN` | Set to `true` to disable auto-starting the backend | `false` |
| `SHODH_STREAM` | Enable/disable streaming ingestion | `true` |
| `SHODH_PROACTIVE` | Enable/disable proactive memory surfacing | `true` |

## MCP Tools (47 total)

<details>
<summary><b>Memory</b> — Store, search, and manage memories</summary>

| Tool | Description |
|------|-------------|
| `remember` | Store a memory with optional type, tags, and metadata |
| `recall` | Semantic search to find relevant memories |
| `proactive_context` | Auto-surface relevant memories for current context |
| `context_summary` | Get categorized context for session bootstrap |
| `list_memories` | List all stored memories |
| `read_memory` | Read full content of a specific memory by ID |
| `forget` | Delete a specific memory by ID |
| `reinforce` | Reinforce a memory (boost importance) |
</details>

<details>
<summary><b>Todos (GTD)</b> — Task management with projects and subtasks</summary>

| Tool | Description |
|------|-------------|
| `add_todo` | Create a task with priority, due date, project, contexts |
| `list_todos` | List/search todos with semantic or GTD-style filtering |
| `update_todo` | Update task properties (status, priority, notes) |
| `complete_todo` | Mark a task as done (auto-creates next for recurring) |
| `delete_todo` | Permanently delete a task |
| `reorder_todo` | Move a task up or down within its status group |
| `list_subtasks` | List subtasks of a parent todo |
| `add_todo_comment` | Add a comment to a task (progress, resolution) |
| `list_todo_comments` | List all comments on a task |
| `update_todo_comment` | Edit an existing comment |
| `delete_todo_comment` | Delete a comment |
| `todo_stats` | Get todo statistics by status, overdue items |
</details>

<details>
<summary><b>Projects</b> — Organize todos into groups</summary>

| Tool | Description |
|------|-------------|
| `add_project` | Create a project with optional parent (sub-projects) |
| `list_projects` | List all projects with todo counts |
| `archive_project` | Archive a project (hidden but restorable) |
| `delete_project` | Permanently delete a project |
</details>

<details>
<summary><b>Reminders</b> — Time, duration, and context-triggered reminders</summary>

| Tool | Description |
|------|-------------|
| `set_reminder` | Set a reminder (time, duration, or keyword trigger) |
| `list_reminders` | List pending/triggered/dismissed reminders |
| `dismiss_reminder` | Acknowledge a triggered reminder |
</details>

<details>
<summary><b>System</b> — Health, backups, and diagnostics</summary>

| Tool | Description |
|------|-------------|
| `memory_stats` | Get statistics about stored memories |
| `verify_index` | Check vector index integrity |
| `repair_index` | Re-index orphaned memories |
| `token_status` | Get current session token usage |
| `reset_token_session` | Reset token counter for new session |
| `consolidation_report` | View memory consolidation activity |
| `backup_create` | Create a backup of all memories |
| `backup_list` | List available backups |
| `backup_verify` | Verify backup integrity (SHA-256) |
| `backup_restore` | Restore from a backup |
| `backup_purge` | Purge old backups, keep most recent N |
</details>

## REST API (for Developers)

The server exposes a REST API at `http://127.0.0.1:3030`:

```javascript
// Store a memory
const res = await fetch("http://127.0.0.1:3030/api/remember", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": "your-api-key"
  },
  body: JSON.stringify({
    user_id: "my-app",
    content: "User prefers dark mode",
    memory_type: "Observation",
    tags: ["preferences", "ui"]
  })
});

// Semantic search
const results = await fetch("http://127.0.0.1:3030/api/recall", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": "your-api-key"
  },
  body: JSON.stringify({
    user_id: "my-app",
    query: "user preferences",
    limit: 5
  })
});
```

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/remember` | POST | Store a memory |
| `/api/recall` | POST | Semantic search |
| `/api/recall/tags` | POST | Search by tags |
| `/api/recall/date` | POST | Search by date range |
| `/api/memories` | POST | List all memories |
| `/api/memory/{id}` | GET/PUT/DELETE | CRUD operations |
| `/api/context_summary` | POST | Get context summary |
| `/api/relevant` | POST | Proactive context surfacing |
| `/api/batch_remember` | POST | Store multiple memories |
| `/api/upsert` | POST | Create or update by external_id |
| `/api/graph/{user_id}/stats` | GET | Knowledge graph statistics |
| `/api/consolidation/report` | POST | Memory consolidation report |
| `/api/index/verify` | POST | Verify index integrity |
| `/metrics` | GET | Prometheus metrics |

## Cognitive Features

### Hebbian Learning
Memories that are frequently accessed together form stronger associations. The system automatically:
- Forms edges between co-retrieved memories
- Strengthens connections with repeated co-activation
- Enables Long-Term Potentiation (LTP) for permanent associations

### Memory Consolidation
Background processes maintain memory health:
- **Decay**: Unused memories gradually lose activation
- **Replay**: High-value memories are periodically replayed
- **Pruning**: Weak associations are removed
- **Promotion**: Important memories move to long-term storage

### 3-Tier Architecture
Based on Cowan's working memory model:
1. **Working Memory**: Recent, highly active memories
2. **Session Memory**: Current session context
3. **Long-Term Memory**: Persistent storage with vector indexing

## How It Works

1. **Install**: `npx -y @shodh/memory-mcp` downloads the package
2. **Auto-spawn**: On first run, downloads the native server binary (~15MB) and embedding model (~23MB)
3. **Connect**: MCP client connects to the server via stdio
4. **Ready**: Start using `remember` and `recall` tools

The backend server runs locally and stores all data on your machine. No cloud dependency.

## Usage Examples

```
"Remember that the user prefers Rust over Python for systems programming"
"Recall what I know about user's programming preferences"
"What context do you have about this project?"
"List my recent memories"
"Show me the consolidation report"
```

## Platform Support

| Platform | Architecture | Status |
|----------|--------------|--------|
| Linux | x64 | Supported |
| macOS | x64 | Supported |
| macOS | ARM64 (M1/M2) | Supported |
| Windows | x64 | Supported |

## Related Packages

- **Python SDK**: `pip install shodh-memory` - Native Python bindings
- **Rust Crate**: `cargo add shodh-memory` - Use as a library

## Links

- [Documentation](https://www.shodh-rag.com/memory)
- [GitHub Repository](https://github.com/varun29ankuS/shodh-memory)
- [Issue Tracker](https://github.com/varun29ankuS/shodh-memory/issues)

## License

Apache-2.0
