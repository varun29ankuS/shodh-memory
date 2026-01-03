# SEP-002: Memory Interchange Format (MIF)

| Field | Value |
|-------|-------|
| SEP | 002 |
| Title | Memory Interchange Format |
| Author | Shodh Team |
| Status | Draft |
| Created | 2026-01-03 |
| Updated | 2026-01-03 |

## Abstract

This SEP defines an open, portable format for exporting and importing AI agent memory. The Memory Interchange Format (MIF) enables:

1. **Portability** - Move memory between different AI systems
2. **Backup/Restore** - Full-fidelity memory preservation
3. **Interoperability** - Share memory across agents (with consent)
4. **Auditability** - Inspect what an AI "knows" about you
5. **Privacy** - User owns and controls their data

## Motivation

As AI agents become persistent collaborators, they accumulate memory about users, projects, and decisions. Currently, this memory is locked in proprietary formats:

- OpenAI's memory is not exportable
- Anthropic's Claude has no persistent memory API
- Mem0, Zep, and others use incompatible internal formats
- No standard exists for memory portability

Sundar Pichai (Google CEO) recently stated: *"If it's my memory, I should be able to take it somewhere else."*

This specification addresses that need.

## Specification

### 1. File Format

MIF files use JSON with optional YAML frontmatter for human readability.

**File Extensions:**
- `.mif.json` - JSON format (canonical)
- `.mif.yaml` - YAML format (human-friendly)
- `.mif.jsonl` - JSON Lines for streaming/large exports

**MIME Type:** `application/vnd.memory-interchange+json`

### 2. Document Structure

```json
{
  "$schema": "https://shodh-memory.dev/schemas/mif-v1.json",
  "mif_version": "1.0",
  "generator": {
    "name": "shodh-memory",
    "version": "0.1.61"
  },
  "export": {
    "id": "exp_a1b2c3d4e5f6",
    "created_at": "2026-01-03T10:45:00.000Z",
    "user_id": "claude-code",
    "checksum": "sha256:abcdef1234567890..."
  },
  "memories": [...],
  "todos": [...],
  "graph": {...},
  "metadata": {...}
}
```

### 3. Memory Object

Each memory is a self-contained unit of knowledge:

```json
{
  "id": "mem_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "content": "User prefers Rust for systems programming due to memory safety guarantees",
  "type": "Learning",
  "importance": 0.85,
  "created_at": "2026-01-02T14:30:00.000Z",
  "updated_at": "2026-01-02T14:30:00.000Z",
  "accessed_at": "2026-01-03T09:15:00.000Z",
  "access_count": 7,
  "decay_rate": 0.1,
  "tags": ["rust", "preference", "programming"],
  "source": {
    "type": "conversation",
    "session_id": "sess_xyz789",
    "agent": "claude-code"
  },
  "entities": [
    {
      "text": "Rust",
      "type": "TECHNOLOGY",
      "confidence": 0.95
    }
  ],
  "embedding": {
    "model": "minilm-l6-v2",
    "dimensions": 384,
    "vector": [0.123, -0.456, 0.789, ...]
  },
  "relations": {
    "related_memories": ["mem_b2c3d4e5..."],
    "related_todos": ["todo_123..."],
    "blocks": [],
    "blocked_by": []
  }
}
```

#### 3.1 Memory Types

| Type | Description |
|------|-------------|
| `Observation` | Factual observation about user/context |
| `Decision` | Choice or commitment made |
| `Learning` | Knowledge acquired |
| `Error` | Bug, issue, or problem encountered |
| `Discovery` | Finding or insight |
| `Pattern` | Recurring behavior or preference |
| `Context` | Session/project context |
| `Task` | Work item or action taken |
| `CodeEdit` | Code modification |
| `FileAccess` | File read/write operation |
| `Search` | Search query and results |
| `Command` | Shell command executed |
| `Conversation` | Dialogue snippet |

#### 3.2 Importance Score

Float between 0.0 and 1.0:
- `1.0` - Critical, never forget (explicit user instruction)
- `0.8-0.99` - High importance (decisions, key learnings)
- `0.5-0.79` - Medium importance (preferences, patterns)
- `0.2-0.49` - Low importance (observations, context)
- `0.0-0.19` - Ephemeral (may be pruned)

#### 3.3 Embedding Format

Embeddings are optional but enable semantic search portability:

```json
{
  "model": "minilm-l6-v2",
  "dimensions": 384,
  "vector": [0.123, -0.456, ...],
  "normalized": true
}
```

Supported models:
- `minilm-l6-v2` (384d) - Default, lightweight
- `bge-small-en-v1.5` (384d)
- `text-embedding-3-small` (1536d) - OpenAI
- `text-embedding-3-large` (3072d) - OpenAI
- `voyage-3` (1024d) - Voyage AI

### 4. Todo Object

```json
{
  "id": "todo_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "short_id": "PROJ-123",
  "content": "Implement rate limiting on API endpoints",
  "status": "in_progress",
  "priority": "high",
  "created_at": "2026-01-01T09:00:00.000Z",
  "updated_at": "2026-01-02T15:30:00.000Z",
  "due_date": "2026-01-05T17:00:00.000Z",
  "completed_at": null,
  "project": {
    "id": "proj_xyz",
    "name": "Backend",
    "prefix": "BACK"
  },
  "contexts": ["@computer", "@deep-work"],
  "tags": ["api", "security"],
  "notes": "Use token bucket algorithm",
  "parent_id": null,
  "subtask_ids": ["todo_child1", "todo_child2"],
  "blocked_on": null,
  "recurrence": null,
  "related_memory_ids": ["mem_a1b2..."],
  "embedding": {
    "model": "minilm-l6-v2",
    "dimensions": 384,
    "vector": [...]
  },
  "comments": [
    {
      "id": "comment_abc",
      "content": "Started implementation",
      "type": "progress",
      "created_at": "2026-01-02T10:00:00.000Z"
    }
  ]
}
```

#### 4.1 Status Values

| Status | Description |
|--------|-------------|
| `backlog` | Not yet scheduled |
| `todo` | Ready to work on |
| `in_progress` | Currently being worked on |
| `blocked` | Waiting on something/someone |
| `done` | Completed |
| `cancelled` | Will not be done |

#### 4.2 Priority Values

| Priority | Description |
|----------|-------------|
| `urgent` | Do immediately |
| `high` | Important, do soon |
| `medium` | Normal priority (default) |
| `low` | Do when time permits |
| `none` | No priority assigned |

### 5. Knowledge Graph

The graph captures learned associations between memories:

```json
{
  "graph": {
    "format": "adjacency_list",
    "node_count": 1234,
    "edge_count": 5678,
    "nodes": [
      {
        "id": "mem_a1b2c3d4...",
        "type": "memory"
      },
      {
        "id": "entity:Rust",
        "type": "entity",
        "entity_type": "TECHNOLOGY"
      }
    ],
    "edges": [
      {
        "source": "mem_a1b2c3d4...",
        "target": "mem_e5f6g7h8...",
        "weight": 0.72,
        "type": "semantic_association",
        "created_at": "2026-01-02T14:30:00.000Z",
        "strengthened_count": 3
      },
      {
        "source": "mem_a1b2c3d4...",
        "target": "entity:Rust",
        "weight": 0.95,
        "type": "entity_mention"
      }
    ],
    "hebbian_config": {
      "learning_rate": 0.1,
      "decay_rate": 0.05,
      "ltp_threshold": 0.8,
      "max_weight": 1.0
    }
  }
}
```

#### 5.1 Edge Types

| Type | Description |
|------|-------------|
| `semantic_association` | Learned from co-retrieval (Hebbian) |
| `entity_mention` | Memory mentions this entity |
| `temporal_sequence` | Memory B follows memory A |
| `causal` | Memory A caused memory B |
| `user_linked` | Explicitly linked by user |
| `todo_memory` | Todo relates to memory |

### 6. Metadata

```json
{
  "metadata": {
    "total_memories": 1234,
    "total_todos": 56,
    "date_range": {
      "earliest": "2025-06-15T00:00:00.000Z",
      "latest": "2026-01-03T10:45:00.000Z"
    },
    "memory_types": {
      "Learning": 342,
      "Decision": 128,
      "Observation": 567,
      "Error": 89,
      "Context": 108
    },
    "top_entities": [
      {"text": "Rust", "count": 45},
      {"text": "TypeScript", "count": 38},
      {"text": "shodh-memory", "count": 32}
    ],
    "projects": [
      {"id": "proj_1", "name": "Backend", "todo_count": 23},
      {"id": "proj_2", "name": "Frontend", "todo_count": 18}
    ],
    "sessions": {
      "total": 89,
      "average_duration_minutes": 45
    },
    "privacy": {
      "pii_detected": false,
      "secrets_detected": false,
      "redacted_fields": []
    }
  }
}
```

### 7. Privacy & Security

#### 7.1 Redaction

Before export, sensitive data should be redacted:

```json
{
  "content": "API key is [REDACTED:api_key]",
  "redactions": [
    {
      "type": "api_key",
      "original_length": 32,
      "position": [12, 44]
    }
  ]
}
```

#### 7.2 Encryption (Optional)

For sensitive exports:

```json
{
  "encryption": {
    "algorithm": "aes-256-gcm",
    "key_derivation": "argon2id",
    "encrypted_payload": "base64...",
    "iv": "base64...",
    "auth_tag": "base64..."
  }
}
```

#### 7.3 Consent Tracking

```json
{
  "consent": {
    "exported_by": "user@example.com",
    "purpose": "backup",
    "retention_days": 30,
    "sharing_allowed": false,
    "timestamp": "2026-01-03T10:45:00.000Z"
  }
}
```

### 8. Versioning & Compatibility

#### 8.1 Version String

Format: `MAJOR.MINOR`

- **MAJOR** - Breaking changes (new required fields, schema restructure)
- **MINOR** - Backwards-compatible additions (new optional fields)

Current version: `1.0`

#### 8.2 Migration

Importers MUST:
1. Check `mif_version` before processing
2. Reject major version mismatches
3. Ignore unknown fields for forward compatibility
4. Provide clear error messages for validation failures

### 9. Streaming Format (JSONL)

For large exports, use JSON Lines:

```jsonl
{"type": "header", "mif_version": "1.0", "export": {...}}
{"type": "memory", "data": {...}}
{"type": "memory", "data": {...}}
{"type": "todo", "data": {...}}
{"type": "graph_edge", "data": {...}}
{"type": "footer", "checksum": "sha256:...", "counts": {...}}
```

### 10. API Endpoints

Recommended REST API for MIF support:

```
POST /api/export
  ?format=json|yaml|jsonl
  ?include=memories,todos,graph,embeddings
  ?since=2026-01-01T00:00:00Z
  ?until=2026-01-03T00:00:00Z
  ?types=Learning,Decision
  ?encrypt=true

POST /api/import
  Content-Type: application/vnd.memory-interchange+json
  Body: <MIF document>

  Response: {
    "imported": {"memories": 100, "todos": 10, "edges": 250},
    "skipped": {"duplicates": 5, "invalid": 2},
    "warnings": ["Edge target mem_xyz not found"]
  }
```

### 11. MCP Integration

MIF export/import can be exposed as MCP tools:

```typescript
{
  name: "export_memory",
  description: "Export all memories to portable MIF format",
  inputSchema: {
    type: "object",
    properties: {
      format: { enum: ["json", "yaml", "jsonl"] },
      include_embeddings: { type: "boolean" },
      since: { type: "string", format: "date-time" }
    }
  }
}

{
  name: "import_memory",
  description: "Import memories from MIF format",
  inputSchema: {
    type: "object",
    properties: {
      data: { type: "string", description: "MIF JSON string" },
      merge_strategy: { enum: ["skip_duplicates", "overwrite", "rename"] }
    }
  }
}
```

## Implementation

### Phase 1: Core Export (v1.0)
- [ ] JSON export with memories, todos, metadata
- [ ] Checksum validation
- [ ] Basic import with duplicate detection

### Phase 2: Graph & Embeddings (v1.1)
- [ ] Full graph export
- [ ] Embedding portability
- [ ] JSONL streaming for large exports

### Phase 3: Privacy & Interop (v1.2)
- [ ] PII detection and redaction
- [ ] Encryption support
- [ ] Import from other formats (Mem0, Zep)

### Phase 4: Ecosystem (v2.0)
- [ ] MCP tools for export/import
- [ ] A2A memory sharing protocol
- [ ] Federated memory queries

## Security Considerations

1. **Export access** - Only the memory owner can export
2. **Import validation** - Validate checksums, reject malformed data
3. **Embedding attacks** - Don't trust imported embeddings for security decisions
4. **PII leakage** - Scan for secrets before export
5. **Size limits** - Prevent DoS via massive imports

## Alternatives Considered

1. **RDF/JSON-LD** - Too complex for simple memory, overkill for most use cases
2. **SQLite dump** - Not human-readable, schema-dependent
3. **Protocol Buffers** - Binary format hurts auditability
4. **Agent Spec extension** - Different scope (workflows vs memory)

## References

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io)
- [Agent2Agent Protocol (A2A)](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- [Open Agent Specification](https://arxiv.org/abs/2510.04173v3)
- [JSON Schema](https://json-schema.org/)
- [GDPR Article 20 - Data Portability](https://gdpr-info.eu/art-20-gdpr/)

## Copyright

This specification is released under CC0 1.0 Universal (Public Domain).
