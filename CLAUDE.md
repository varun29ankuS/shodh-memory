# Shodh-Memory Project Instructions

## MUST Rules
- MUST NOT run `cargo build` or `cargo run`. Only `cargo check`, `cargo clippy`, and `cargo test` are allowed.
- MUST NOT add "Co-Authored-By" or "Generated with Claude Code" to commits. Clean commit messages only.
- MUST NOT comment on GitHub issues/PRs without showing user a draft first.

## Code Standards
- IMPORTANT: Production grade code only. No TODOs, no placeholders, no mocks, no stubs.
- IMPORTANT: Understand architecture and data flow before fixing anything. Read files before editing.
- All changes go through PR workflow: branch → commit → push → PR → merge.

## Persistent Memory (shodh-memory MCP)

You have session continuity via shodh-memory hooks. Key behaviors:
- Relevant memories surface automatically — don't manually "check" memory
- Use `remember` sparingly (high-importance only), `forget` for corrections
- Use `list_todos` at session start for pending work
- The memory system you're using IS this codebase

## Codebase Map
- `src/` — Rust core (memory, API server, embeddings, graph, vector search)
- `mcp-server/` — TypeScript MCP server (45 tools, index.ts)
- `tui/` — Rust TUI dashboard
- `hooks/` — Claude Code hooks for automatic memory
- `python/` — Python bindings (PyO3/maturin)
- Architecture: RocksDB + Vamana/SPANN vector search + knowledge graph with Hebbian learning
