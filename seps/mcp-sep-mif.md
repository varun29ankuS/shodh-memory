# SEP-XXXX: Memory Interchange Format (MIF)

- **Status**: Draft
- **Type**: Standards Track
- **Created**: 2026-01-22
- **Author(s)**: @varun29ankuS
- **Sponsor**: None (seeking sponsor)
- **Related**: SEP-1975 (Conversation Event Subscriptions), RFC #2043

## Abstract

This SEP defines the Memory Interchange Format (MIF), an open, portable format for exporting and importing AI agent memory. MIF enables memory portability across different AI systems, addressing GDPR Article 20 (data portability) requirements and user data ownership.

Unlike storage-layer proposals that dictate *where* memory lives, MIF specifies *what format* memory takes when moved between systems — enabling interoperability regardless of underlying storage implementation.

## Motivation

As AI agents become persistent collaborators, they accumulate memory about users, projects, and decisions. This data is valuable personal information that:

1. **Users should own** — Not locked to a single provider
2. **Should be portable** — Move memory when switching AI tools
3. **Should be auditable** — Inspect what an AI "knows"
4. **Must comply with regulations** — GDPR Article 20 requires data portability

Currently, no standard exists for AI memory portability:

- OpenAI's memory is not exportable
- Anthropic's Claude has no persistent memory API
- Memory providers (Mem0, Zep, basic-memory) use incompatible internal formats
- Each MCP memory server implements its own storage mechanism

Sundar Pichai (Google CEO) recently stated: *"If it's my memory, I should be able to take it somewhere else."*

This specification addresses that need with a format-agnostic, human-readable interchange standard.

### Why a Format Standard (Not Storage)?

Storage layer proposals (e.g., embedded databases) solve *persistence*. MIF solves *portability*.

| Concern | Storage Layer | MIF |
|---------|--------------|-----|
| Where data lives | Database files | Agnostic |
| How to query | SQL/API | N/A (export/import) |
| Cross-system transfer | Requires same DB | Any system can read JSON |
| Human auditability | Requires tooling | Open JSON/YAML in editor |
| Regulatory compliance | Implementation detail | First-class concern |

MIF is complementary to storage proposals — any storage backend can export to MIF.

## Specification

### 1. File Format

MIF uses JSON as the canonical format, with optional alternatives for different use cases.

**File Extensions:**
- `.mif.json` — JSON format (canonical, machine-readable)
- `.mif.yaml` — YAML format (human-friendly)
- `.mif.jsonl` — JSON Lines (streaming, large exports)

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
    "user_id": "user-123",
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
  "content": "User prefers Rust for systems programming",
  "type": "Learning",
  "importance": 0.85,
  "created_at": "2026-01-02T14:30:00.000Z",
  "updated_at": "2026-01-02T14:30:00.000Z",
  "accessed_at": "2026-01-03T09:15:00.000Z",
  "access_count": 7,
  "tags": ["rust", "preference"],
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
    "vector": [0.123, -0.456, 0.789, "..."]
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
| `Conversation` | Dialogue snippet |

#### 3.2 Importance Score

Float between 0.0 and 1.0:
- `0.8-1.0` — Critical (explicit user instruction, key decisions)
- `0.5-0.79` — Medium (preferences, patterns)
- `0.2-0.49` — Low (observations, context)
- `0.0-0.19` — Ephemeral (may be pruned)

#### 3.3 Embedding Format

Embeddings enable semantic search portability:

```json
{
  "model": "minilm-l6-v2",
  "dimensions": 384,
  "vector": [0.123, -0.456, ...],
  "normalized": true
}
```

Importers SHOULD re-embed if the model differs from their native embedding model.

### 4. Todo Object (Optional)

MIF supports task/todo export for agents with GTD-style task management:

```json
{
  "id": "todo_a1b2c3d4...",
  "short_id": "PROJ-123",
  "content": "Implement rate limiting",
  "status": "in_progress",
  "priority": "high",
  "created_at": "2026-01-01T09:00:00.000Z",
  "due_date": "2026-01-05T17:00:00.000Z",
  "project": {
    "id": "proj_xyz",
    "name": "Backend"
  },
  "contexts": ["@computer", "@deep-work"],
  "tags": ["api", "security"]
}
```

Status values: `backlog`, `todo`, `in_progress`, `blocked`, `done`, `cancelled`

Priority values: `urgent`, `high`, `medium`, `low`, `none`

### 5. Knowledge Graph

MIF preserves learned associations between memories:

```json
{
  "graph": {
    "format": "adjacency_list",
    "node_count": 1234,
    "edge_count": 5678,
    "edges": [
      {
        "source": "mem_a1b2c3d4...",
        "target": "mem_e5f6g7h8...",
        "weight": 0.72,
        "type": "semantic_association",
        "created_at": "2026-01-02T14:30:00.000Z"
      }
    ],
    "hebbian_config": {
      "learning_rate": 0.1,
      "decay_rate": 0.05,
      "ltp_threshold": 0.8
    }
  }
}
```

Edge types:
- `semantic_association` — Learned from co-retrieval (Hebbian)
- `entity_mention` — Memory mentions entity
- `temporal_sequence` — Memory B follows memory A
- `causal` — Memory A caused memory B
- `user_linked` — Explicitly linked by user

### 6. Privacy & Security

#### 6.1 PII Detection and Redaction

Before export, implementations SHOULD detect and redact sensitive data:

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

Detectable PII types:
- API keys, tokens, secrets
- Email addresses
- Phone numbers
- Credit card numbers
- Social security numbers
- Passwords

#### 6.2 Encryption

For sensitive exports, MIF supports envelope encryption:

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

#### 6.3 Consent Tracking

```json
{
  "consent": {
    "exported_by": "user@example.com",
    "purpose": "backup",
    "retention_days": 30,
    "sharing_allowed": false,
    "gdpr_basis": "user_request",
    "timestamp": "2026-01-03T10:45:00.000Z"
  }
}
```

### 7. Metadata

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
      "Observation": 567
    },
    "privacy": {
      "pii_detected": false,
      "secrets_detected": false,
      "redacted_fields": []
    }
  }
}
```

### 8. MCP Tool Interface

MIF export/import exposed as MCP tools:

#### `export_memory`

```json
{
  "name": "export_memory",
  "description": "Export memories to portable MIF format",
  "inputSchema": {
    "type": "object",
    "properties": {
      "format": {
        "type": "string",
        "enum": ["json", "yaml", "jsonl"],
        "default": "json"
      },
      "include_embeddings": {
        "type": "boolean",
        "default": true
      },
      "include_graph": {
        "type": "boolean",
        "default": true
      },
      "since": {
        "type": "string",
        "format": "date-time",
        "description": "Export memories created after this date"
      },
      "redact_pii": {
        "type": "boolean",
        "default": true
      }
    }
  }
}
```

#### `import_memory`

```json
{
  "name": "import_memory",
  "description": "Import memories from MIF format",
  "inputSchema": {
    "type": "object",
    "properties": {
      "data": {
        "type": "string",
        "description": "MIF JSON string or file path"
      },
      "merge_strategy": {
        "type": "string",
        "enum": ["skip_duplicates", "overwrite", "rename"],
        "default": "skip_duplicates"
      }
    },
    "required": ["data"]
  }
}
```

### 9. Versioning

Format: `MAJOR.MINOR`

- **MAJOR** — Breaking changes (new required fields, restructure)
- **MINOR** — Backwards-compatible additions (new optional fields)

Current version: `1.0`

Importers MUST:
1. Check `mif_version` before processing
2. Reject unsupported major versions
3. Ignore unknown fields for forward compatibility

## Rationale

### Why JSON over Binary?

1. **Human auditability** — Users can inspect exports in any text editor
2. **Debugging** — Easier to troubleshoot import failures
3. **Regulatory compliance** — GDPR auditors can read the format
4. **Universal tooling** — Every language has JSON support

Binary formats (Protocol Buffers, MessagePack) were rejected for these reasons.

### Why Include Graph Edges?

Memory systems that use Hebbian learning, spreading activation, or knowledge graphs lose critical information if only nodes are exported. Edge weights represent learned associations that took time to develop.

### Why PII Detection is Required

Memory systems inherently store personal information. Without redaction, exports become liability vectors. Making PII detection a first-class concern (not an afterthought) protects users and implementers.

### Why Embeddings are Optional

Embeddings are large (384-3072 floats per memory) and model-specific. Including them enables faster import (no re-embedding) but increases file size 10-100x. The trade-off should be user-controlled.

## Backward Compatibility

MIF is a new format — no backward compatibility concerns.

For servers with existing proprietary export formats:
1. Continue supporting old format for existing users
2. Add MIF export as new option
3. Deprecate old format after migration period

## Reference Implementation

**shodh-memory** provides a complete reference implementation:

- **Export**: `export_memory` MCP tool
- **Import**: `import_memory` MCP tool with duplicate detection
- **JSON Schema**: Validation for all MIF documents
- **PII Detection**: Built-in redaction for secrets and personal data
- **Encryption**: AES-256-GCM with Argon2id key derivation

Repository: https://github.com/varun29ankuS/shodh-memory

Schema: https://github.com/varun29ankuS/shodh-memory/blob/main/seps/mif-v1.schema.json

Example: https://github.com/varun29ankuS/shodh-memory/blob/main/seps/example.mif.json

## Security Considerations

1. **Export authorization** — Only memory owners can export their data
2. **Import validation** — Validate checksums, reject malformed data
3. **Embedding attacks** — Don't trust imported embeddings for security decisions; re-embed if security-critical
4. **PII leakage** — Scan for secrets before export; warn users
5. **Size limits** — Prevent DoS via massive imports (recommend 100MB limit)
6. **Encryption key management** — Users responsible for key backup

## Alternatives Considered

### RDF/JSON-LD
Too complex for simple memory export. Overkill for most use cases. Poor tooling support.

### SQLite Dump
Not human-readable. Schema-dependent. Requires SQLite tooling to inspect.

### Protocol Buffers
Binary format hurts auditability. Requires schema compilation. Overkill for interchange.

### Proprietary Database Formats (chDB, DuckDB)
Ties portability to specific database. Defeats the purpose of interchange format.

## Prior Art

- **GDPR Article 20** — Right to data portability
- **Agent2Agent Protocol (A2A)** — Google's agent interoperability work
- **Open Agent Specification** — ArXiv 2510.04173v3
- **JSON Schema** — Validation standard

## Open Questions

1. **Federated queries** — Should MIF support query-over-export for large memory sets?
2. **Streaming imports** — How to handle partial imports on failure?
3. **Cross-agent consent** — Protocol for sharing memory between different users' agents?

## Acknowledgments

This specification was developed as part of the shodh-memory project, informed by:
- MCP community feedback on persistent storage needs
- GDPR compliance requirements for AI systems
- Prior art from knowledge graph and memory systems

## Copyright

This specification is released under CC0 1.0 Universal (Public Domain).
