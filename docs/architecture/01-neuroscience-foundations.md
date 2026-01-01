# Neuroscience Foundations

This document maps shodh-memory's features to their cognitive science basis, with code references proving implementation.

## Why Not Just Use ACT-R?

ACT-R (Adaptive Control of Thought—Rational) is the gold standard cognitive architecture from CMU. We studied it extensively. Here's why we built something different:

| Aspect | ACT-R | Shodh-Memory | Why We Differ |
|--------|-------|--------------|---------------|
| **Memory representation** | Symbolic chunks | Vector embeddings | AI agents process natural language, not pre-encoded symbols |
| **Retrieval** | Exact chunk matching + spreading | Semantic similarity + spreading | "borrowing" should find "ownership" even without explicit link |
| **Learning mechanism** | Base-level learning (frequency) | Hebbian + LTP | We need co-activation learning, not just access counting |
| **Decay model** | Power-law only | Hybrid exponential→power-law | Fast consolidation filtering matters for noisy agent contexts |
| **Deployment** | Lisp runtime, research tool | Single 15MB binary, production API | Edge deployment, air-gapped environments |
| **Integration** | Requires symbolic encoding | REST API, MCP, Python SDK | Works with any LLM, no pre-processing |

**ACT-R's limitation for AI agents:** It requires pre-encoding all knowledge into symbolic chunks with explicit slot-value pairs. An AI agent receiving free-form text would need a complex encoding layer. Shodh works directly with natural language via embeddings.

## Implemented Features (With Code References)

### 1. Hebbian Learning: "Neurons That Fire Together, Wire Together"

**What it is:** When two memories are accessed together, the connection between them strengthens.

**Implementation:** `src/graph_memory.rs:166-188`

```rust
pub fn strengthen(&mut self) {
    self.activation_count += 1;
    self.last_activated = Utc::now();

    // Hebbian strengthening: diminishing returns as strength approaches 1.0
    let boost = LTP_LEARNING_RATE * (1.0 - self.strength);
    self.strength = (self.strength + boost).min(1.0);

    // Check for Long-Term Potentiation threshold
    if !self.potentiated && self.activation_count >= LTP_THRESHOLD {
        self.potentiated = true;
        self.strength = (self.strength + 0.2).min(1.0);
    }
}
```

**Why this formula:** The `(1.0 - self.strength)` term creates asymptotic approach to 1.0, matching biological synaptic saturation. ACT-R uses additive base-level learning which can grow unboundedly.

**Constants:** `src/constants.rs:443-452`
- `LTP_LEARNING_RATE = 0.1` — Matches empirical synaptic LTP rates (Bi & Poo, 1998)
- `LTP_THRESHOLD = 10` — Activations needed for potentiation

### 2. Long-Term Potentiation (LTP)

**What it is:** After repeated co-activation, synapses undergo a phase transition—they become resistant to decay. This models the biological consolidation from labile to stable memory traces.

**Implementation:** `src/graph_memory.rs:182-187`

```rust
if !self.potentiated && self.activation_count >= LTP_THRESHOLD {
    self.potentiated = true;
    self.strength = (self.strength + 0.2).min(1.0); // LTP bonus
}
```

**Why LTP matters:** ACT-R doesn't model potentiation—all memories decay at the same rate regardless of reinforcement history. This means frequently-accessed knowledge can still fade. With LTP, truly important associations become permanent.

**Constants:** `src/constants.rs:469-479`
- `LTP_THRESHOLD = 10` — Matches biological LTP threshold (~10-100 activations)
- `LTP_DECAY_FACTOR = 0.1` — Potentiated synapses decay 10x slower

### 3. Hybrid Decay Model (Exponential → Power-Law)

**What it is:** Memory strength decays over time, but the decay function changes:
- **First 3 days:** Exponential decay (fast consolidation filtering)
- **After 3 days:** Power-law decay (heavy tail for long-term retention)

**Implementation:** `src/decay.rs:74-108`

```rust
pub fn hybrid_decay_factor(days_elapsed: f64, potentiated: bool) -> f32 {
    let beta = if potentiated { POWERLAW_BETA_POTENTIATED } else { POWERLAW_BETA };
    let lambda = if potentiated { DECAY_LAMBDA_CONSOLIDATION * 0.5 } else { DECAY_LAMBDA_CONSOLIDATION };

    if days_elapsed < DECAY_CROSSOVER_DAYS {
        // Consolidation phase: exponential decay w(t) = w₀ × e^(-λt)
        (-lambda * days_elapsed).exp() as f32
    } else {
        // Long-term phase: power-law decay A(t) = A_cross × (t / t_cross)^(-β)
        let value_at_crossover = (-lambda * DECAY_CROSSOVER_DAYS).exp();
        let power_law_factor = (days_elapsed / DECAY_CROSSOVER_DAYS).powf(-beta);
        (value_at_crossover * power_law_factor) as f32
    }
}
```

**Why hybrid, not just power-law (like ACT-R)?**

ACT-R uses pure power-law: `B_i = ln(Σ t_j^-d)`. This preserves everything with a long tail.

For AI agents, this is problematic:
1. **Noise accumulation:** Agents generate lots of low-quality context. Pure power-law retains it all.
2. **No consolidation window:** There's no fast-filtering period to identify what's worth keeping.

Our hybrid model:
- Fast exponential decay (days 0-3) filters noise aggressively
- Power-law (day 3+) preserves what survives consolidation

**Research basis:**
- Wixted & Ebbesen (1991) "On the Form of Forgetting"
- Wixted (2004) "The psychology and neuroscience of forgetting"

**Constants:** `src/constants.rs:50-51`
- `DECAY_CROSSOVER_DAYS = 3.0`
- `POWERLAW_BETA = 0.5` — Standard forgetting curve exponent

### 4. Spreading Activation

**What it is:** When a memory is retrieved, activation spreads to connected memories through the knowledge graph.

**Implementation:** `src/graph_memory.rs:825-897`

```rust
/// Implements Hebbian learning: edges traversed during retrieval are strengthened.
pub fn traverse_with_activation(&mut self, start_entities: &[Uuid], depth: usize, min_strength: f32)
    -> Vec<(Uuid, f32)>
{
    let mut edges_to_strengthen = Vec::new();
    // ... traversal logic ...

    // Apply Hebbian strengthening to all traversed edges atomically
    if !edges_to_strengthen.is_empty() {
        for edge_uuid in edges_to_strengthen {
            if let Some(edge) = self.get_edge_by_uuid_mut(&edge_uuid) {
                edge.strengthen();
            }
        }
    }
}
```

**Key difference from ACT-R:** Our spreading activation is bidirectional on the knowledge graph AND strengthens edges during traversal. ACT-R's spreading activation is read-only—it doesn't modify the network.

### 5. 3-Tier Memory Architecture

**What it is:** Memories exist in three nested tiers with different capacities and decay rates.

**Implementation:** `src/memory/types.rs:1991-2177`

| Tier | Capacity | Decay | Code Reference |
|------|----------|-------|----------------|
| Working | 7±2 items | Seconds | `WorkingMemory` struct, line 1991 |
| Session | ~100 items | Minutes | `SessionMemory` struct, line 2171 |
| Long-term | Unlimited | Days→permanent | `MemoryStorage`, persisted to RocksDB |

**Why not 2-tier like ACT-R?** ACT-R has declarative (facts) and procedural (rules) memory. We have three tiers because:
1. **Working memory capacity constraint** (Miller's 7±2) forces prioritization—critical for noisy agent contexts
2. **Session boundary** enables episodic segmentation without explicit markers
3. **Long-term** with LTP allows truly permanent storage of validated knowledge

**Research basis:** Cowan, N. (2001) "The magical number 4 in short-term memory"

## What We DON'T Implement (Honestly)

| ACT-R Feature | Status in Shodh | Reason |
|---------------|-----------------|--------|
| Procedural memory | Not implemented | AI agents have their own reasoning—we just provide declarative memory |
| Production rules | Not implemented | Conflates memory with reasoning; let the LLM reason |
| Goal buffer | Not implemented | Agent frameworks handle goals; we're infrastructure |
| Utility learning | Partially (via Hebbian) | We boost helpful memories, but don't learn action utilities |
| Imaginal buffer | Not implemented | Would require image embedding; future roadmap |

## Validation

All neuroscience-grounded features have tests:

```
src/decay.rs:166-240           — 6 tests for hybrid decay model
src/graph_memory.rs            — Hebbian strengthening tested in integration tests
tests/chunking_retrieval_tests.rs — End-to-end retrieval with scoring
```

Run: `cargo test decay` / `cargo test hebbian` / `cargo test consolidation`

## References

1. **Hebbian Learning:** Hebb, D.O. (1949). *The Organization of Behavior*
2. **LTP:** Bliss, T.V.P. & Lømo, T. (1973). "Long-lasting potentiation of synaptic transmission"
3. **LTP Rate Constants:** Bi, G.Q. & Poo, M.M. (1998). "Synaptic modifications in cultured hippocampal neurons"
4. **Power-Law Decay:** Wixted, J.T. & Ebbesen, E.B. (1991). "On the form of forgetting"
5. **Working Memory:** Miller, G.A. (1956). "The Magical Number Seven, Plus or Minus Two"
6. **Cowan's Model:** Cowan, N. (2001). "The magical number 4 in short-term memory"
7. **ACT-R:** Anderson, J.R. (2007). *How Can the Human Mind Occur in the Physical Universe?*
