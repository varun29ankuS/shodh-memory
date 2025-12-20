<p align="center">
  <img src="https://raw.githubusercontent.com/varun29ankuS/shodh-memory/main/assets/logo.png" width="120" alt="Shodh-Memory">
</p>

<h1 align="center">Shodh-Memory</h1>

<p align="center">
  <a href="https://github.com/varun29ankuS/shodh-memory/actions"><img src="https://github.com/varun29ankuS/shodh-memory/workflows/CI/badge.svg" alt="build"></a>
  <a href="https://registry.modelcontextprotocol.io/v0/servers?search=shodh"><img src="https://img.shields.io/badge/MCP-Registry-green" alt="MCP Registry"></a>
  <a href="https://crates.io/crates/shodh-memory"><img src="https://img.shields.io/crates/v/shodh-memory.svg" alt="crates.io"></a>
  <a href="https://www.npmjs.com/package/@shodh/memory-mcp"><img src="https://img.shields.io/npm/v/@shodh/memory-mcp.svg?logo=npm" alt="npm"></a>
  <a href="https://pypi.org/project/shodh-memory/"><img src="https://img.shields.io/pypi/v/shodh-memory.svg" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

---

<p align="center"><i>Persistent memory for AI agents. Single binary. Local-first. Runs offline.</i></p>

---

> **For AI Agents** — Claude, Cursor, GPT, LangChain, AutoGPT, robotic systems, or your custom agents.
> Give them memory that persists across sessions, learns from experience, and runs entirely on your hardware.

---

We built this because AI agents forget everything between sessions. They make the same mistakes, ask the same questions, lose context constantly.

Shodh-Memory fixes that. It's a cognitive memory system—Hebbian learning, activation decay, semantic consolidation—packed into a single ~17MB binary that runs offline. Deploy on cloud, edge devices, or air-gapped systems.

## Quick Start

Choose your platform:

| Platform | Install | Documentation |
|----------|---------|---------------|
| **Claude / Cursor** | `claude mcp add shodh-memory -- npx -y @shodh/memory-mcp` | [MCP Setup](#claude--cursor-mcp) |
| **Python** | `pip install shodh-memory` | [Python Docs](https://pypi.org/project/shodh-memory/) |
| **Rust** | `cargo add shodh-memory` | [Rust Docs](https://crates.io/crates/shodh-memory) |
| **npm (MCP)** | `npx -y @shodh/memory-mcp` | [npm Docs](https://www.npmjs.com/package/@shodh/memory-mcp) |

## TUI Dashboard

<p align="center">
  <img src="https://raw.githubusercontent.com/varun29ankuS/shodh-memory/main/assets/dashboard.jpg" width="700" alt="Shodh Dashboard">
</p>

<p align="center"><i>Real-time activity feed, memory tiers, and detailed inspection</i></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/varun29ankuS/shodh-memory/main/assets/graph-map.jpg" width="700" alt="Shodh Graph Map">
</p>

<p align="center"><i>Knowledge graph visualization — entity connections across memories</i></p>

## How It Works

Experiences flow through three tiers based on Cowan's working memory model:

```
Working Memory ──overflow──▶ Session Memory ──importance──▶ Long-Term Memory
   (100 items)                  (500 MB)                      (RocksDB)
```

**Cognitive Processing:**
- **Hebbian learning** — Co-retrieved memories form stronger connections
- **Activation decay** — Unused memories fade: A(t) = A₀ · e^(-λt)
- **Long-term potentiation** — Frequently-used connections become permanent
- **Entity extraction** — TinyBERT NER identifies people, orgs, locations
- **Spreading activation** — Queries activate related memories through the graph
- **Memory replay** — Important memories replay during maintenance (like sleep)

## Claude / Cursor (MCP)

**Claude Code (CLI):**
```bash
claude mcp add shodh-memory -- npx -y @shodh/memory-mcp
```

**Claude Desktop / Cursor config:**
```json
{
  "mcpServers": {
    "shodh-memory": {
      "command": "npx",
      "args": ["-y", "@shodh/memory-mcp"],
      "env": {
        "SHODH_API_KEY": "your-api-key"
      }
    }
  }
}
```

Config file locations:

| Editor | Path |
|--------|------|
| Claude Desktop (macOS) | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude Desktop (Windows) | `%APPDATA%\Claude\claude_desktop_config.json` |
| Cursor | `~/.cursor/mcp.json` |

## Python

```bash
pip install shodh-memory
```

```python
from shodh_memory import Memory

memory = Memory(storage_path="./my_data")
memory.remember("User prefers dark mode", memory_type="Decision")
results = memory.recall("user preferences", limit=5)
```

[Full Python documentation →](https://pypi.org/project/shodh-memory/)

## Rust

```toml
[dependencies]
shodh-memory = "0.1"
```

```rust
use shodh_memory::{MemorySystem, MemoryConfig};

let memory = MemorySystem::new(MemoryConfig::default())?;
memory.remember("user-1", "User prefers dark mode", MemoryType::Decision, vec![])?;
let results = memory.recall("user-1", "user preferences", 5)?;
```

[Full Rust documentation →](https://crates.io/crates/shodh-memory)

## Performance

| Operation | Latency |
|-----------|---------|
| Store memory | 55-60ms |
| Semantic search | 34-58ms |
| Tag search | ~1ms |
| Entity lookup | 763ns |
| Graph traversal (3-hop) | 30µs |

## Compared to Alternatives

| | Shodh-Memory | Mem0 | Cognee |
|---|---|---|---|
| **Deployment** | Single 17MB binary | Cloud API | Neo4j + Vector DB |
| **Offline** | 100% | No | Partial |
| **Learning** | Hebbian + decay + LTP | Vector similarity | Knowledge graphs |
| **Latency** | Sub-millisecond | Network-bound | Database-bound |

## Platform Support

| Platform | Status |
|----------|--------|
| Linux x86_64 | Supported |
| macOS ARM64 (Apple Silicon) | Supported |
| macOS x86_64 (Intel) | Supported |
| Windows x86_64 | Supported |
| Linux ARM64 | Coming soon |

## Community Implementations

| Project | Description | Author |
|---------|-------------|--------|
| [SHODH on Cloudflare](https://github.com/doobidoo/shodh-cloudflare) | Edge-native implementation on Cloudflare Workers with D1, Vectorize, and Workers AI | [@doobidoo](https://github.com/doobidoo) |

Have an implementation? [Open a discussion](https://github.com/varun29ankuS/shodh-memory/discussions) to get it listed.

## References

[1] Cowan, N. (2010). The Magical Mystery Four: How is Working Memory Capacity Limited, and Why? *Current Directions in Psychological Science*.

[2] Magee, J.C., & Grienberger, C. (2020). Synaptic Plasticity Forms and Functions. *Annual Review of Neuroscience*.

[3] Subramanya, S.J., et al. (2019). DiskANN: Fast Accurate Billion-point Nearest Neighbor Search. *NeurIPS 2019*.

## License

Apache 2.0

---

[MCP Registry](https://registry.modelcontextprotocol.io/v0/servers?search=shodh) · [PyPI](https://pypi.org/project/shodh-memory/) · [npm](https://www.npmjs.com/package/@shodh/memory-mcp) · [crates.io](https://crates.io/crates/shodh-memory) · [Docs](https://www.shodh-rag.com/memory)
