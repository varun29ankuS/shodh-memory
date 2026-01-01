# Hebbian Learning & Long-Term Potentiation

Deep-dive into shodh-memory's learning mechanisms with implementation details.

## The Core Insight

Most memory systems treat storage and retrieval as separate operations. Shodh treats **retrieval as learning**—every time you access memories, the system learns which associations matter.

```
Traditional RAG:
  Store → Index → Retrieve → Done
  (static, doesn't learn)

Shodh:
  Store → Index → Retrieve → Strengthen → Store
                     ↑           ↓
                     └───────────┘
                    (learning loop)
```

## Hebbian Learning: The Mechanism

### What Happens on Co-Activation

When two memories are accessed in the same context (e.g., retrieved together for the same query), the edge between them is strengthened:

**Code:** `src/graph_memory.rs:174-180`

```rust
pub fn strengthen(&mut self) {
    self.activation_count += 1;
    self.last_activated = Utc::now();

    // Hebbian strengthening: diminishing returns as strength approaches 1.0
    let boost = LTP_LEARNING_RATE * (1.0 - self.strength);
    self.strength = (self.strength + boost).min(1.0);
}
```

### The Learning Rate

**Constant:** `LTP_LEARNING_RATE = 0.1` (`src/constants.rs:452`)

**Why 0.1?**

| Rate | Effect | Problem |
|------|--------|---------|
| 0.01 | Very slow learning | Takes 100+ co-accesses to reach strength 0.5 |
| **0.1** | Balanced | Reaches 0.5 strength in ~7 co-accesses |
| 0.5 | Fast learning | Overfits to recent patterns, unstable |

The 0.1 rate matches empirical synaptic LTP rates from Bi & Poo (1998).

### Asymptotic Saturation

The formula `boost = η × (1 - strength)` ensures:

1. **New associations strengthen quickly** — When strength is low (0.1), boost is high
2. **Strong associations stabilize** — When strength is high (0.9), boost is small
3. **No unbounded growth** — Strength can never exceed 1.0

```
Strength progression with 0.1 learning rate:
  Access 1: 0.10 → 0.19  (+0.09)
  Access 2: 0.19 → 0.27  (+0.08)
  Access 3: 0.27 → 0.34  (+0.07)
  ...
  Access 10: 0.59 → 0.63 (+0.04)
  ...
  Access 20: 0.86 → 0.87 (+0.01)
```

## Long-Term Potentiation (LTP)

### The Phase Transition

After `LTP_THRESHOLD` (10) co-activations, the synapse undergoes a qualitative change:

**Code:** `src/graph_memory.rs:182-187`

```rust
if !self.potentiated && self.activation_count >= LTP_THRESHOLD {
    self.potentiated = true;
    self.strength = (self.strength + 0.2).min(1.0); // LTP bonus
}
```

### What Changes After Potentiation

| Property | Before LTP | After LTP |
|----------|------------|-----------|
| Decay rate | Normal (power-law β=0.5) | 10x slower (β=0.3) |
| Minimum strength | Can reach 0.01 (prunable) | Floor at 0.1 (persistent) |
| Pruning | Yes, if strength too low | Never pruned |

**Constants:** `src/constants.rs:469-488`
- `LTP_THRESHOLD = 10`
- `LTP_DECAY_FACTOR = 0.1` (10x slower decay)
- `LTP_MIN_STRENGTH = 0.01`

### Why This Matters

Without LTP, even frequently-used associations eventually fade. With LTP:

```
Scenario: User asks about "Rust ownership" 15 times over 6 months

Without LTP:
  Month 1: strength 0.9 (high)
  Month 3: strength 0.4 (decayed)
  Month 6: strength 0.1 (nearly forgotten)

With LTP (potentiated at access 10):
  Month 1: strength 0.9 (high)
  Month 3: strength 0.8 (stable)
  Month 6: strength 0.7 (preserved)
```

## The Feedback Loop API

### Tracked Recall

**Endpoint:** `POST /api/recall/tracked`

Returns memories with tracking IDs for later feedback:

```json
{
  "user_id": "user-123",
  "query": "How do I fix the auth bug?",
  "limit": 5
}
```

Response includes `tracking_id` for each memory.

### Reinforce (Hebbian Feedback)

**Endpoint:** `POST /api/reinforce`

After the agent completes a task, it reports which memories were helpful:

```json
{
  "user_id": "user-123",
  "feedback": {
    "helpful": ["mem-id-1", "mem-id-2"],
    "misleading": ["mem-id-3"],
    "neutral": ["mem-id-4"]
  }
}
```

**Code:** `src/main.rs:6335-6420`

**Effects:**

| Category | Importance Δ | Association Δ | Graph Effect |
|----------|--------------|---------------|--------------|
| `helpful` | +0.05 | Strengthen all pairs | Edges between helpful memories boosted |
| `misleading` | -0.10 | None | No strengthening |
| `neutral` | 0 | Mild | Small boost to pairs |

**Constants:** `src/constants.rs:24-37`
- `HEBBIAN_BOOST_HELPFUL = 0.05`
- `HEBBIAN_DECAY_MISLEADING = 0.10`

## Graph Traversal Learning

When spreading activation traverses the graph, every edge visited is strengthened:

**Code:** `src/graph_memory.rs:860-897`

```rust
// During traversal
for edge in outgoing_edges {
    if edge.effective_strength() >= min_strength {
        // Record this edge for Hebbian strengthening
        edges_to_strengthen.push(edge.uuid);
        // ... continue traversal
    }
}

// After traversal completes
for edge_uuid in edges_to_strengthen {
    if let Some(edge) = self.get_edge_by_uuid_mut(&edge_uuid) {
        edge.strengthen();
    }
}
```

This means the graph **learns from how it's used**, not just from explicit feedback.

## Memory Consolidation Learning

During background consolidation (every 5 minutes), the system:

1. **Replays recent memories** — Re-activates important associations
2. **Extracts facts** — Repeated patterns become semantic knowledge
3. **Prunes weak edges** — Low-strength, non-potentiated edges removed

**Code:** `src/memory/replay.rs`

### Replay Strengthening

```rust
// During replay, we strengthen co-occurring memory pairs
for pair in memory_pairs_in_episode {
    if let Some(edge) = graph.get_or_create_edge(pair.0, pair.1) {
        edge.strengthen();
    }
}
```

**Constant:** `REPLAY_STRENGTH_BOOST = 0.08` (`src/constants.rs:711`)

This simulates sleep-dependent memory consolidation.

## Metrics & Observability

### Prometheus Metrics

```
shodh_hebbian_reinforce_total{outcome="helpful|misleading|neutral"}
shodh_hebbian_reinforce_duration_seconds
```

**Code:** `src/metrics.rs:287-307`

### Consolidation Report

**Endpoint:** `POST /api/consolidation/report`

Returns what the system learned:

```json
{
  "statistics": {
    "memories_strengthened": 42,
    "edges_strengthened": 156,
    "edges_pruned": 23,
    "facts_extracted": 5
  },
  "strengthened_memories": [...],
  "strengthened_associations": [...]
}
```

## Comparison to Other Approaches

### vs. RAG (No Learning)

| Aspect | RAG | Shodh |
|--------|-----|-------|
| Retrieval affects storage | No | Yes (Hebbian) |
| Frequently-used items boosted | No | Yes |
| Unused items decay | No | Yes |
| Associations learned | No | Yes |

### vs. ACT-R Base-Level Learning

| Aspect | ACT-R | Shodh |
|--------|-------|-------|
| Learning signal | Access count only | Co-activation + feedback |
| Activation formula | `ln(Σ t_j^-d)` | Hebbian with saturation |
| Potentiation | None | Yes (LTP after 10 accesses) |
| Decay resistance | All equal | Potentiated edges protected |

### vs. Memory Networks (End-to-End)

| Aspect | MemNets | Shodh |
|--------|---------|-------|
| Learning | Backprop through attention | Explicit Hebbian updates |
| Interpretability | Black box | Inspectable edges |
| Requires training | Yes | No (online learning) |
| Works with any LLM | No | Yes |

## Configuration

```rust
// src/constants.rs - All tunable

// Hebbian Learning
pub const LTP_LEARNING_RATE: f32 = 0.1;
pub const LTP_THRESHOLD: u32 = 10;
pub const LTP_DECAY_FACTOR: f32 = 0.1;
pub const LTP_MIN_STRENGTH: f32 = 0.01;

// Feedback
pub const HEBBIAN_BOOST_HELPFUL: f32 = 0.05;
pub const HEBBIAN_DECAY_MISLEADING: f32 = 0.10;

// Replay
pub const REPLAY_STRENGTH_BOOST: f32 = 0.08;
```

## Testing

```bash
# Hebbian strengthening
cargo test strengthen

# LTP threshold
cargo test potentiation

# Feedback loop
cargo test reinforce

# Full integration
cargo test --test consolidation_tests
```
