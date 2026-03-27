# M1: Graph Snapshot Export

## Goal

Bulk export the full shodh knowledge graph as structured JSON (or GEXF for Gephi). Provides visibility into what's alive, what's dead weight, and where the Hebbian structure is. Independently useful for analysis and debugging; also the prerequisite input format for M2 (Curriculum Builder).

## API

### Endpoint

`GET /api/graph/{user_id}/export`

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | `json` \| `gexf` | `json` | Output format |
| `include` | comma-separated | `entities,memories,episodes` | Node types to include |
| `min_importance` | f32 | `0.0` | Floor filter for memory nodes |
| `include_embeddings` | bool | `false` | Include 384-dim embedding vectors |

### Response

Streamed if payload is large. JSON structure:

```json
{
  "metadata": {
    "exported_at": "2026-03-26T14:30:00Z",
    "user_id": "uuid",
    "node_count": 4821,
    "edge_count": 12340,
    "node_counts_by_type": { "entity": 1200, "memory": 3500, "episode": 121 },
    "edge_counts_by_type": { "relationship": 8000, "entity_ref": 4200 }
  },
  "nodes": [ ... ],
  "edges": [ ... ]
}
```

## Node Schema

All nodes share: `id` (UUID string), `type`, `label`, `attributes` (object).

### Entity Nodes

Sourced from `CF_ENTITIES` (EntityNode).

```json
{
  "id": "uuid",
  "type": "entity",
  "label": "UNS WordPress",
  "attributes": {
    "salience": 0.8,
    "mention_count": 12,
    "is_proper_noun": true,
    "labels": ["Organization"],
    "created_at": "2026-03-10T08:00:00Z",
    "last_seen_at": "2026-03-25T14:00:00Z",
    "summary": "Indonesian university WordPress installation..."
  }
}
```

Optional (with `include_embeddings=true`): `attributes.name_embedding: float[384]`

### Memory Nodes

Sourced from `CF_DEFAULT` (Memory).

```json
{
  "id": "uuid",
  "type": "memory",
  "label": "Staff blogs at *.staff.uns.ac.id compromised with gambling SEO...",
  "attributes": {
    "content": "Full memory content text...",
    "importance": 0.7,
    "tier": "LongTerm",
    "access_count": 8,
    "last_accessed": "2026-03-25T10:00:00Z",
    "temporal_relevance": 0.4,
    "activation": 0.15,
    "experience_type": "Discovery",
    "created_at": "2026-03-12T09:00:00Z",
    "agent_id": "optional-uuid",
    "run_id": "optional-uuid"
  }
}
```

Optional (with `include_embeddings=true`): `attributes.embedding: float[384]`

### Episode Nodes

Sourced from `CF_EPISODES` (EpisodicNode).

```json
{
  "id": "uuid",
  "type": "episode",
  "label": "Run 32 discovery",
  "attributes": {
    "content": "Episode content...",
    "source": "Event",
    "valid_at": "2026-03-20T12:00:00Z",
    "created_at": "2026-03-20T12:00:00Z"
  }
}
```

## Edge Schema

All edges share: `id` (UUID string or synthetic), `source`, `target`, `type`, `attributes` (object).

### Relationship Edges

Sourced from `CF_RELATIONSHIPS` (RelationshipEdge). Connect entity↔entity.

```json
{
  "id": "uuid",
  "source": "entity-uuid",
  "target": "entity-uuid",
  "type": "relationship",
  "label": "RelatedTo",
  "attributes": {
    "strength": 0.85,
    "relation_type": "RelatedTo",
    "ltp_status": "Full",
    "tier": "L3Semantic",
    "activation_count": 14,
    "last_activated": "2026-03-25T09:00:00Z",
    "created_at": "2026-03-10T08:00:00Z",
    "valid_at": "2026-03-10T08:00:00Z",
    "invalidated_at": null,
    "entity_confidence": 0.9
  }
}
```

### Entity Ref Edges

Synthesized from `entity_refs` on Memory and EpisodicNode records. Connect memory→entity or episode→entity.

```json
{
  "id": "synthetic-uuid",
  "source": "memory-uuid",
  "target": "entity-uuid",
  "type": "entity_ref",
  "attributes": {
    "relation": "mentioned"
  }
}
```

### Co-Retrieval Edges

Implicit in the current storage — co-retrieval strengthens entity↔entity RelationshipEdges rather than creating direct memory↔memory links. These are already represented in the `relationship` edges via `activation_count` and `strength`. No separate edge type needed unless we find direct memory↔memory co-retrieval records during implementation.

## GEXF Output

When `format=gexf`, the same data is serialized as GEXF XML. Mapping:

| JSON field | GEXF element |
|-----------|-------------|
| `node.type` | `<node>` with `<attvalue>` for class (Gephi partitioning/coloring) |
| `node.label` | `<node label="...">` |
| Numeric attributes (`importance`, `strength`, `salience`, `activation_count`) | `<attvalue>` (Gephi sizing/coloring) |
| `tier`, `ltp_status` | `<attvalue>` categorical (Gephi partition) |
| `created_at`, `last_activated` | GEXF dynamic attributes (Gephi timeline) |
| `edge.source/target` | `<edge source="..." target="...">` |
| `edge.attributes.strength` | `<edge weight="...">` (Gephi edge thickness) |

GEXF attribute declarations go in `<attributes>` blocks, typed as `float`, `string`, or `long` as appropriate.

## CLI

Thin HTTP client calling the API endpoint:

```
shodh export-graph <user_id> [--format json|gexf] [--include entities,memories,episodes] [--min-importance 0.0] [--include-embeddings] [-o output.json]
```

Writes to stdout by default. `-o` for file output.

## Implementation Notes

- Handler iterates all RocksDB column families in a single snapshot read for consistency
- Entity ref edges are synthesized by walking `entity_refs` on each memory/episode node
- Node labels for memories are content truncated to ~100 characters
- GEXF serialization uses lightweight XML writing (no heavy XML framework needed)
- Streaming response for large graphs to avoid holding the full serialized payload in memory

## What This Enables

- **Gephi visualization**: Size nodes by importance/salience, color by type, weight edges by Hebbian strength, partition by tier/LTP status, animate by timeline
- **Dead weight detection**: Disconnected or low-activity entity nodes reveal whether NER removal left orphans
- **Graph health metrics**: Density, clustering coefficient, degree distribution — all computable from the export
- **M2 input**: The JSON format is the canonical input for the Curriculum Builder (filtered by `type=memory` for training data, edge attributes for curriculum decisions)
