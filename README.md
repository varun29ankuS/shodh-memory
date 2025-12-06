<p align="center">
  <img src="https://raw.githubusercontent.com/varun29ankuS/shodh-memory/main/assets/logo.png" width="120" alt="Shodh-Memory">
</p>

<h1 align="center">Shodh-Memory</h1>

<p align="center">
  <a href="https://registry.modelcontextprotocol.io/servers/io.github.varun29ankuS/shodh-memory"><img src="https://img.shields.io/badge/MCP-Registry-green" alt="MCP Registry"></a>
  <a href="https://www.npmjs.com/package/@shodh/memory-mcp"><img src="https://img.shields.io/npm/v/@shodh/memory-mcp.svg?logo=npm" alt="npm"></a>
  <a href="https://pypi.org/project/shodh-memory/"><img src="https://img.shields.io/pypi/v/shodh-memory.svg" alt="PyPI"></a>
  <a href="https://pepy.tech/project/shodh-memory"><img src="https://static.pepy.tech/badge/shodh-memory" alt="Downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

---

<p align="center"><i>Memory that learns. Single binary. Runs offline.</i></p>

---

We built this because AI agents forget everything between sessions. They make the same mistakes, ask the same questions, lose context constantly.

Shodh-Memory fixes that. It's a cognitive memory system—Hebbian learning, activation decay, semantic consolidation—packed into a single 8MB binary that runs offline.

**How it works:**

Experiences flow through three tiers based on Cowan's working memory model [1]. New information enters capacity-limited working memory, overflows into session storage, and consolidates into long-term memory based on importance. When memories are retrieved together successfully, their connections strengthen—classic Hebbian learning [2]. After enough co-activations, those connections become permanent. Unused memories naturally fade. The system learns what matters to *you*.

**What you get:**

Your decisions, errors, and patterns—searchable and private. No cloud. No API keys. Your memory, your machine.

```
Working Memory ──overflow──▶ Session Memory ──importance──▶ Long-Term Memory
   (100 items)                  (500 MB)                      (RocksDB)
```

### Architecture

**Storage & Retrieval**

- Vamana graph index for approximate nearest neighbor search [3]
- MiniLM-L6 embeddings (384-dim) for semantic similarity
- RocksDB for durable persistence across restarts
- User isolation — each agent gets independent memory space

**Cognitive Processing**

- *Activation decay* — exponential decay A(t) = A₀ · e^(-λt) applied each maintenance cycle (λ configurable)
- *Hebbian strengthening* — co-retrieved memories form graph edges; edge weight w increases as w' = w + α(1 - w) on each co-activation
- *Long-term potentiation* — edges surviving threshold co-activations (default: 5) become permanent, exempt from decay
- *Importance scoring* — composite score from memory type, content length, entity density, technical terms, access frequency

**Semantic Consolidation**

- Episodic memories older than 7 days compress into semantic facts
- Entity extraction preserves key information during compression
- Original experiences archived, compressed form used for retrieval

**Context Bootstrapping**

- `context_summary()` provides categorized session context on startup
- Returns decisions, learnings, patterns, errors — structured for LLM consumption
- `brain_state()` exposes full 3-tier visualization data

### Use cases

**Local LLM memory** — Give Claude, GPT, or any local model persistent memory across sessions. Remember user preferences, past decisions, learned patterns.

**Robotics & drones** — On-device experience accumulation. A robot that remembers which actions worked, which failed, without cloud round-trips.

**Edge AI** — Run on Jetson, Raspberry Pi, industrial PCs. Sub-millisecond retrieval, zero network dependency.

**Personal knowledge base** — Your own searchable memory. Decisions, learnings, discoveries—private and local.

### Compared to alternatives

| | Shodh-Memory | Mem0 | Cognee |
|---|---|---|---|
| **Deployment** | Single 8MB binary | Cloud API | Neo4j + Vector DB |
| **Offline** | 100% | No | Partial |
| **Learning** | Hebbian + decay + LTP | Vector similarity | Knowledge graphs |
| **Latency** | Sub-millisecond | Network-bound | Database-bound |
| **Best for** | Local-first, edge, privacy | Cloud scale | Enterprise ETL |

### Installation

**Claude Code / Claude Desktop:**

Add to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "shodh-memory": {
      "command": "npx",
      "args": ["-y", "@shodh/memory-mcp"]
    }
  }
}
```

Config file locations:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Python:**
```
pip install shodh-memory
```

**From source:**
```
cargo build --release
./target/release/shodh-memory-server
```

### Usage

**Python**

```python
from shodh_memory import Memory

memory = Memory(user_id="my-agent")

# Store
memory.remember("User prefers dark mode", memory_type="Decision")
memory.remember("JWT tokens expire after 24h", memory_type="Learning")

# Search
results = memory.recall("user preferences", limit=5)

# Session bootstrap - get categorized context
summary = memory.context_summary()
# Returns: decisions, learnings, patterns, errors
```

**REST API**

```
# Store
curl -X POST http://localhost:3030/api/record \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "user_id": "agent-1",
    "experience": {
      "content": "Deployment requires Docker 24+",
      "experience_type": "Learning"
    }
  }'

# Search
curl -X POST http://localhost:3030/api/retrieve \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"user_id": "agent-1", "query": "deployment requirements", "limit": 5}'
```

### Memory types

Different types get different importance weights in the scoring model:

- **Decision** (+0.30) — choices, preferences, conclusions
- **Learning** (+0.25) — new knowledge, facts learned
- **Error** (+0.25) — mistakes, things to avoid
- **Discovery**, **Pattern** (+0.20) — findings, recurring behaviors
- **Task** (+0.15) — work items
- **Context**, **Observation** (+0.10) — general info

Importance also increases with: content length, entity density, technical terms, and access frequency.

### API reference

**Python client**

| Method | What it does |
|--------|--------------|
| `remember(content, memory_type, tags)` | Store a memory |
| `recall(query, limit)` | Semantic search |
| `context_summary()` | Categorized context for session start |
| `brain_state()` | 3-tier visualization data |
| `stats()` | Memory statistics |
| `delete(memory_id)` | Remove a memory |

**REST endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/record` | POST | Store memory |
| `/api/retrieve` | POST | Semantic search |
| `/api/memories` | POST | List memories |
| `/api/memory/{id}` | GET/DELETE | Single memory operations |
| `/api/users/{id}/stats` | GET | User statistics |
| `/api/brain/{user_id}` | GET | 3-tier state |
| `/health` | GET | Health check |

### Configuration

```
SHODH_PORT=3030                    # Default: 3030
SHODH_MEMORY_PATH=./data           # Default: ./shodh_memory_data
SHODH_API_KEYS=key1,key2           # Required in production
SHODH_MAINTENANCE_INTERVAL=300     # Decay cycle (seconds)
SHODH_ACTIVATION_DECAY=0.95        # Decay factor per cycle
```

### Platform support

| Platform | Status | Use case |
|----------|--------|----------|
| Linux x86_64 | ✓ | Servers, workstations |
| macOS ARM64 | ✓ | Development (Apple Silicon) |
| Windows x86_64 | ✓ | Development, industrial PCs |
| Linux ARM64 | Coming soon | Jetson, Raspberry Pi, drones |

### References

[1] Cowan, N. (2010). The Magical Mystery Four: How is Working Memory Capacity Limited, and Why? *Current Directions in Psychological Science*, 19(1), 51-57. https://pmc.ncbi.nlm.nih.gov/articles/PMC4207727/

[2] Magee, J.C., & Grienberger, C. (2020). Synaptic Plasticity Forms and Functions. *Annual Review of Neuroscience*, 43, 95-117. https://pmc.ncbi.nlm.nih.gov/articles/PMC10410470/

[3] Subramanya, S.J., et al. (2019). DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node. *NeurIPS 2019*. https://papers.nips.cc/paper/9527-diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node

[4] Dudai, Y., Karni, A., & Born, J. (2015). The Consolidation and Transformation of Memory. *Neuron*, 88(1), 20-32. https://pmc.ncbi.nlm.nih.gov/articles/PMC4183265/

### License

Apache 2.0

---

[MCP Registry](https://registry.modelcontextprotocol.io/servers/io.github.varun29ankuS/shodh-memory) · [PyPI](https://pypi.org/project/shodh-memory/) · [npm](https://www.npmjs.com/package/@shodh/memory-mcp) · [GitHub](https://github.com/varun29ankuS/shodh-memory)
