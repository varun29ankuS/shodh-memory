# Shodh Memory API Reference

Complete documentation of shodh-memory API endpoints, features, and performance benchmarks.

---

## API Endpoints

### Memory Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/remember` | POST | Store memory with semantic indexing |
| `/api/recall` | POST | Semantic search across memories |
| `/api/recall/tags` | POST | Filter memories by tags |
| `/api/recall/date` | POST | Filter memories by date range |
| `/api/proactive_context` | POST | Auto-surface relevant memories |
| `/api/context_summary` | POST | Overview of learnings/decisions |
| `/api/memories` | POST | List all memories |
| `/api/memories/{id}` | DELETE | Delete memory by ID |
| `/api/forget/tags` | POST | Delete memories by tags |
| `/api/forget/date` | POST | Delete memories by date range |

### System Health
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with stats |
| `/api/users/{id}/stats` | GET | Memory statistics |
| `/api/index/verify` | POST | Index health check |
| `/api/index/repair` | POST | Fix index issues |
| `/api/consolidation_report` | POST | Memory evolution report |

### Prospective Memory (Reminders)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/reminders` | POST | Create reminder |
| `/api/reminders/list` | POST | List active reminders |
| `/api/reminders/{id}/dismiss` | POST | Dismiss reminder |
| `/api/reminders/check` | POST | Check due reminders |

### GTD Todo System
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/todos` | POST | Create todo |
| `/api/todos/list` | POST | List todos with filters |
| `/api/todos/{id}/update` | POST | Update todo |
| `/api/todos/{id}/complete` | POST | Mark todo complete |
| `/api/todos/{id}` | DELETE | Delete todo |
| `/api/todos/due` | POST | List due/overdue todos |
| `/api/todos/stats` | POST | Todo statistics |
| `/api/projects` | POST | Create project |
| `/api/projects/list` | POST | List projects |

### Streaming
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stream` | WebSocket | Real-time memory streaming |

---

## MCP Tools (26 total)

### Memory Tools
- `remember` - Store memories with semantic indexing
- `recall` - Semantic search across memories
- `recall_by_tags` - Tag-based filtering
- `recall_by_date` - Date range filtering
- `proactive_context` - Auto-surface relevant context for current conversation
- `context_summary` - Quick overview of learnings/decisions
- `list_memories` - List all stored memories
- `forget` - Delete memory by ID
- `forget_by_tags` - Delete memories matching tags
- `forget_by_date` - Delete memories in date range

### System Tools
- `memory_stats` - Get memory statistics
- `verify_index` - Check index health
- `repair_index` - Fix index issues
- `streaming_status` - WebSocket connection status
- `consolidation_report` - Memory evolution report

### Reminder Tools
- `set_reminder` - Create time/context-triggered reminder
- `list_reminders` - List active reminders
- `dismiss_reminder` - Mark reminder complete

### GTD Todo Tools
- `add_todo` - Create task with GTD fields
- `list_todos` - List with filters (status, context, project, due)
- `update_todo` - Modify task properties
- `complete_todo` - Mark done, handle recurrence
- `delete_todo` - Remove task
- `add_project` - Create project to group tasks
- `list_projects` - List projects with task counts
- `todo_stats` - Get todo statistics

---

## Performance Benchmarks

### Latency (measured on Windows, i7-12700H)
| Operation | Latency |
|-----------|---------|
| Health check | ~1ms |
| Remember (store) | 34-58ms |
| Recall (semantic search) | 34-58ms |
| Graph lookup | <1μs |
| Proactive context | 40-80ms |
| Todo operations | 5-15ms |

### Resource Usage
- Binary size: ~15MB
- Memory footprint: ~50-100MB (depends on data)
- Startup time: <2s (includes model loading)

### Scalability
- Tested with 10,000+ memories
- Sub-linear search time with HNSW index
- RocksDB provides O(log n) lookups

---

## Key Features

### Cognitive Architecture
- **3-tier memory** - Based on Cowan's working memory model
- **Hebbian learning** - Connections strengthen with use ("neurons that fire together wire together")
- **Hybrid decay** - Exponential + power-law decay based on cognitive research
- **Memory consolidation** - Replay and interference detection like biological memory
- **Spreading activation** - Knowledge graph with activation propagation

### GTD Todo System
Linear-style UI for task management:

```
SHO · My Todos                                                       3 items

In Progress ◐ ──────────────────────────────────────────────────────
  ◐ !   SHO-c6c7  Write unit tests for todo module
                  @computer

Todo ○ ─────────────────────────────────────────────────────────────
  ○ !!! SHO-c121  Add recurring tasks support           shodh-memory
                  @computer · Due Wed

Blocked ⊘ ──────────────────────────────────────────────────────────
  ⊘ !!  SHO-ad30  Review PR #456
                  @computer · Blocked on waiting for @varun approval
```

**Status Icons:**
- ◌ Backlog - Not started, someday/maybe
- ○ Todo - Ready to do
- ◐ In Progress - Actively working
- ⊘ Blocked - Waiting for someone/something
- ● Done - Completed
- ⊗ Cancelled - Won't do

**Priority Indicators:**
- `!!!` Urgent (P1)
- `!!` High (P2)
- `!` Medium (P3)
- (none) Low (P4)

**GTD Concepts Supported:**
- Projects - Multi-step outcomes with subtasks
- Contexts - @computer, @phone, @errands, @home
- Due dates - With natural parsing (tomorrow, next week)
- Recurring tasks - Daily, weekly, monthly patterns
- Blocked/waiting status - Track dependencies

### Edge Deployment
- Single binary (~15MB), no cloud dependency
- Works completely offline
- Runs on Raspberry Pi, Jetson, air-gapped systems
- Sub-millisecond graph lookups

---

## Example Usage

### Store a Memory
```bash
curl -X POST http://127.0.0.1:3030/api/remember \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{
    "user_id": "claude-code",
    "content": "User prefers functional programming style",
    "memory_type": "Learning",
    "tags": ["preferences", "coding-style"]
  }'
```

### Search Memories
```bash
curl -X POST http://127.0.0.1:3030/api/recall \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{
    "user_id": "claude-code",
    "query": "coding preferences",
    "limit": 5
  }'
```

### Create a Todo
```bash
curl -X POST http://127.0.0.1:3030/api/todos \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{
    "user_id": "claude-code",
    "content": "Review PR #456",
    "priority": "high",
    "contexts": ["@computer"],
    "project": "shodh-memory"
  }'
```

### List Todos
```bash
curl -X POST http://127.0.0.1:3030/api/todos/list \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{
    "user_id": "claude-code",
    "status": ["todo", "in_progress"]
  }'
```

---

## Linear Issue

Tracking: [SHO-117](https://linear.app/shodh-memory/issue/SHO-117/api-endpoints-and-benchmarks-documentation)
