<p align="center">
  <img src="https://raw.githubusercontent.com/varun29ankuS/shodh-memory/main/assets/logo.png" width="120" alt="Shodh-Memory">
</p>

<h1 align="center">Shodh-Memory</h1>

<p align="center">
  <strong>Local-first AI memory for robotics, drones, and edge devices</strong>
</p>

<p align="center">
  <a href="https://www.shodh-rag.com/memory"><img src="https://img.shields.io/badge/Website-shodh--rag.com-blue" alt="Website"></a>
  <a href="https://pypi.org/project/shodh-memory/"><img src="https://img.shields.io/pypi/v/shodh-memory.svg" alt="PyPI"></a>
  <a href="https://pepy.tech/project/shodh-memory"><img src="https://static.pepy.tech/badge/shodh-memory" alt="Downloads"></a>
  <a href="https://github.com/varun29ankuS/shodh-memory/actions"><img src="https://github.com/varun29ankuS/shodh-memory/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

---

Offline AI memory system for robotics and edge devices. Rust backend with Python bindings.

## How Memory Works

Shodh-Memory uses language structure to decide what to remember:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: "The drone detected a critical obstacle near the hangar"            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. PARSE                                                                   │
│                                                                             │
│     Nouns → Entities        Verbs → Relationships      Adjectives → Weight  │
│     ─────────────────       ──────────────────────     ───────────────────  │
│     "drone"                 "detected"                 "critical" (+1.5x)   │
│     "obstacle"              "near"                                          │
│     "hangar"                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. STORE AS GRAPH                                                          │
│                                                                             │
│     [drone] ──detected──▶ [obstacle] ◀── "critical" boosts importance       │
│                               │                                             │
│                            ──near──▶ [hangar]                               │
│                                                                             │
│     Each entity tracks: mention_count, last_accessed, importance            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. STRENGTHEN ON USE (Hebbian Learning)                                    │
│                                                                             │
│     Co-activation: [1] → [2] → [3] → STRONG (resists decay 10x)            │
│                                                                             │
│     Connections used together get stronger.                                 │
│     After 3 co-activations, they become long-term.                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. DECAY OVER TIME                                                         │
│                                                                             │
│     Important memories decay slowly. Unimportant ones fade fast.            │
│                                                                             │
│     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐            │
│     │ WORKING │ ──▶ │ SESSION │ ──▶ │  LONG   │ ──▶ │ ARCHIVE │            │
│     │ (now)   │     │ (task)  │     │  TERM   │     │ (gist)  │            │
│     └─────────┘     └─────────┘     └─────────┘     └─────────┘            │
│                                                                             │
│     Promoted by: action verbs, proper nouns, frequent access                │
│     Demoted by: low access, no connections, time without use                │
└─────────────────────────────────────────────────────────────────────────────┘
```

Based on [construction grammar](https://direct.mit.edu/coli/article/50/4/1375/123787), [Hebbian learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10410470/), [memory consolidation](https://pmc.ncbi.nlm.nih.gov/articles/PMC4183265/), and [working memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC4207727/) research.

## Installation

```bash
pip install shodh-memory
```

## Quick Start

```python
from shodh_memory import MemorySystem

# Initialize (models download automatically on first use)
memory = MemorySystem("./robot_memory")

# Store experiences
memory.record(
    content="Detected obstacle at coordinates X=5.2, Y=10.1",
    experience_type="observation",
    tags=["obstacle", "navigation"]
)

memory.record(
    content="Battery level critical at 15%",
    experience_type="sensor",
    tags=["battery", "warning"]
)

# Semantic search - finds relevant memories
results = memory.retrieve("obstacle near position 5", limit=5)
for mem in results:
    print(f"[{mem['relevance']:.2f}] {mem['content']}")

# Output:
# [0.89] Detected obstacle at coordinates X=5.2, Y=10.1
```

## API Reference

### MemorySystem

| Method | Description |
|--------|-------------|
| `record(content, experience_type, ...)` | Store a memory |
| `retrieve(query, limit)` | Search memories |
| `get_stats()` | Get statistics |
| `flush()` | Persist to disk |

### Experience Types

| Type | Use Case |
|------|----------|
| `observation` | What was seen/detected |
| `action` | What was done |
| `sensor` | Raw sensor readings |
| `navigation` | Position/movement |

## REST API Server

For microservice architectures, run the HTTP server:

```bash
# From source
cargo build --release
./target/release/shodh-memory-server

# Environment variables
PORT=3030                           # Server port
STORAGE_PATH=./shodh_memory_data    # Data directory
RUST_LOG=info                       # Log level
```

```bash
# Store memory (simple)
curl -X POST http://localhost:3030/api/remember \
  -H "Content-Type: application/json" \
  -d '{"user_id": "robot-001", "content": "Obstacle detected at entrance"}'

# Search (simple)
curl -X POST http://localhost:3030/api/recall \
  -H "Content-Type: application/json" \
  -d '{"user_id": "robot-001", "query": "obstacle", "limit": 5}'

# Store with full metadata
curl -X POST http://localhost:3030/api/record \
  -H "Content-Type: application/json" \
  -d '{"user_id": "robot-001", "content": "Obstacle detected", "experience_type": "observation"}'

# Search with filters
curl -X POST http://localhost:3030/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"user_id": "robot-001", "query": "obstacle", "max_results": 5}'
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/remember` | POST | Store memory (simple) |
| `/api/recall` | POST | Search memories (simple) |
| `/api/record` | POST | Store with full metadata |
| `/api/retrieve` | POST | Search with filters |
| `/api/batch_remember` | POST | Batch store |
| `/api/memory/{id}` | GET/PUT/DELETE | CRUD operations |
| `/api/users/{id}/stats` | GET | User statistics |
| `/api/graph/{user_id}/stats` | GET | Graph statistics |
| `/health` | GET | Health check |

## Platform Support

| Platform | Status |
|----------|--------|
| Linux x86_64 | Supported |
| macOS ARM64 (Apple Silicon) | Supported |
| Windows x86_64 | Supported |
| Linux ARM64 (Jetson, Pi) | Coming soon |

## Development

```bash
# Build from source
git clone https://github.com/varun29ankuS/shodh-memory
cd shodh-memory
cargo build --release

# Run tests
cargo test

# Build Python wheel
pip install maturin
maturin build --release
```

## Roadmap

### Near-term
- [ ] ARM64 Linux builds (Jetson Nano, Raspberry Pi 4/5)
- [ ] Temporal queries ("what happened yesterday", "last mission")
- [ ] Memory consolidation for long-term storage compression
- [ ] TypeScript/JavaScript client SDK

### Future
- [ ] ROS2 integration package
- [ ] Fleet memory sync with sharding (multi-robot coordination)
- [ ] Alternative embedding models (BGE-small, E5)

## License

Apache 2.0

## Links

- [Website](https://www.shodh-rag.com/memory)
- [PyPI Package](https://pypi.org/project/shodh-memory/)
- [GitHub](https://github.com/varun29ankuS/shodh-memory)
- [Issues](https://github.com/varun29ankuS/shodh-memory/issues)
- Email: 29.varuns@gmail.com
