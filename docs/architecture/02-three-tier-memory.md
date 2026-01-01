# 3-Tier Memory Model

Shodh-memory implements a biologically-inspired three-tier memory system based on Cowan's embedded-processes model.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      LONG-TERM MEMORY                        │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │                  SESSION MEMORY                      │    │   │
│  │  │  ┌─────────────────────────────────────────────┐    │    │   │
│  │  │  │              WORKING MEMORY                  │    │    │   │
│  │  │  │                                              │    │    │   │
│  │  │  │    Capacity: 7±2 items                       │    │    │   │
│  │  │  │    Decay: seconds                            │    │    │   │
│  │  │  │    Activation: highest                       │    │    │   │
│  │  │  │                                              │    │    │   │
│  │  │  └─────────────────────────────────────────────┘    │    │   │
│  │  │                                                      │    │   │
│  │  │    Capacity: ~100 items                              │    │   │
│  │  │    Decay: minutes to hours                           │    │   │
│  │  │    Activation: medium                                │    │   │
│  │  │                                                      │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │                                                              │   │
│  │    Capacity: unlimited                                       │   │
│  │    Decay: days to permanent                                  │   │
│  │    Activation: lowest (requires retrieval cue)               │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Tier 1: Working Memory

The innermost tier—highly activated, severely capacity-limited.

### Characteristics

| Property | Value | Biological Basis |
|----------|-------|------------------|
| Capacity | 7±2 items | Miller's Law |
| Decay rate | τ = 30 seconds | Rapid without rehearsal |
| Activation | 1.0 (maximum) | Focus of attention |
| Retrieval | Immediate | No search required |

### Implementation

```rust
pub struct WorkingMemory {
    /// Bounded queue enforcing capacity limit
    items: VecDeque<MemoryId>,
    /// Maximum capacity (default: 7)
    capacity: usize,
    /// Activation levels for each item
    activations: HashMap<MemoryId, f32>,
}

impl WorkingMemory {
    pub fn push(&mut self, id: MemoryId) {
        // If at capacity, oldest item is displaced to session memory
        if self.items.len() >= self.capacity {
            let displaced = self.items.pop_front();
            self.promote_to_session(displaced);
        }
        self.items.push_back(id);
        self.activations.insert(id, 1.0);
    }
}
```

### Displacement

When working memory is full, the oldest item is **displaced** (not deleted). It moves to session memory with reduced activation:

```
Working Memory (full):
  [A] [B] [C] [D] [E] [F] [G]   ← capacity 7

New item X arrives:
  [A] → displaced to Session Memory (activation 0.7)
  [B] [C] [D] [E] [F] [G] [X]   ← X takes position
```

## Tier 2: Session Memory

Middle tier—larger capacity, slower decay, requires mild retrieval effort.

### Characteristics

| Property | Value | Biological Basis |
|----------|-------|------------------|
| Capacity | ~100 items | Short-term store |
| Decay rate | τ = 30 minutes | Without rehearsal |
| Activation | 0.3 - 0.7 | Primed but not focal |
| Retrieval | Fast | Indexed search |

### Implementation

```rust
pub struct SessionMemory {
    /// Active session memories
    items: HashSet<MemoryId>,
    /// Decay timestamps
    last_access: HashMap<MemoryId, DateTime<Utc>>,
    /// Session boundary (for episodic segmentation)
    session_start: DateTime<Utc>,
}

impl SessionMemory {
    pub fn decay_check(&mut self, now: DateTime<Utc>) {
        let threshold = Duration::minutes(30);

        self.items.retain(|id| {
            let last = self.last_access.get(id).unwrap_or(&self.session_start);
            let age = now - *last;

            if age > threshold {
                // Promote to long-term if important enough
                if self.should_promote(id) {
                    self.promote_to_longterm(id);
                }
                false // Remove from session
            } else {
                true // Keep in session
            }
        });
    }
}
```

### Promotion Criteria

Items are promoted to long-term memory based on:

1. **Access frequency**: Accessed 3+ times in session
2. **Importance score**: Above threshold (default: 0.5)
3. **Semantic connections**: Linked to existing long-term memories
4. **Explicit marking**: User marked as important

```rust
fn should_promote(&self, id: &MemoryId) -> bool {
    let memory = self.get(id);

    memory.access_count >= 3
        || memory.importance >= 0.5
        || self.has_longterm_connections(id)
        || memory.explicit_important
}
```

## Tier 3: Long-Term Memory

Outermost tier—unlimited capacity, slow decay, requires retrieval cue.

### Characteristics

| Property | Value | Biological Basis |
|----------|-------|------------------|
| Capacity | Unlimited | Distributed storage |
| Decay rate | Power-law (t^-0.5) | Wixted's model |
| Activation | 0.0 - 0.3 | Requires retrieval |
| Retrieval | Slow | Semantic search |

### Sub-types

Long-term memory contains two sub-types:

#### Episodic Memory
- Specific events with temporal context
- "What happened when"
- Decays faster

#### Semantic Memory
- General facts extracted from episodes
- "What is true"
- Decays slower (consolidated knowledge)

```rust
pub enum LongTermMemoryType {
    /// Specific events: "User fixed bug in auth.rs on Tuesday"
    Episodic {
        timestamp: DateTime<Utc>,
        context: EpisodeContext,
    },
    /// General facts: "User prefers Rust for systems code"
    Semantic {
        confidence: f32,
        source_episodes: Vec<MemoryId>,
    },
}
```

### Retrieval

Long-term retrieval uses multiple pathways:

1. **Semantic search**: Vector similarity (Vamana HNSW)
2. **Graph traversal**: Spreading activation from cue
3. **Temporal cues**: "What happened last week"
4. **Tag filtering**: Categorical retrieval

```rust
fn retrieve_longterm(&self, query: &Query) -> Vec<Memory> {
    let mut results = Vec::new();

    // Pathway 1: Semantic similarity
    let semantic = self.vector_search(query.embedding, query.limit);
    results.extend(semantic);

    // Pathway 2: Graph activation
    let activated = self.spread_activation(query.cue_memories);
    results.extend(activated);

    // Pathway 3: Temporal
    if let Some(time_range) = query.time_range {
        let temporal = self.temporal_search(time_range);
        results.extend(temporal);
    }

    // Deduplicate and rank
    self.rank_and_dedupe(results)
}
```

## Tier Transitions

### Promotion (Inward)

```
Long-Term → Session → Working
   │           │          │
   └───────────┴──────────┘
        retrieval cue
```

When a long-term memory is retrieved, it's **promoted** inward:
- Enters session memory (if not already there)
- May enter working memory (if highly relevant)
- Activation increases

### Displacement (Outward)

```
Working → Session → Long-Term
   │          │          │
   └──────────┴──────────┘
      capacity overflow
```

When inner tiers overflow, items are **displaced** outward:
- Working → Session: Oldest item displaced
- Session → Long-Term: Decayed or session-end

## Memory Flow Example

```
Time 0: User asks about "Rust borrowing"
  Working: [query: Rust borrowing]
  Session: []
  Long-term: [500 memories]

Time 1: Semantic search retrieves relevant memories
  Working: [query, mem_42, mem_108]  ← retrieved
  Session: [mem_42, mem_108]         ← also in session
  Long-term: [500 memories]

Time 2: User asks follow-up about "lifetimes"
  Working: [lifetimes_query, mem_42, mem_201]
  Session: [query, mem_42, mem_108, mem_201]  ← expanded
  Long-term: [500 memories]

Time 3: User context-switches to different topic
  Working: [new_topic]               ← displaced others
  Session: [query, mem_42, mem_108, mem_201, lifetimes_query]
  Long-term: [500 memories]

Time 30min: Session timeout
  Working: []
  Session: []                        ← cleared
  Long-term: [500 + promoted memories]
```

## Configuration

```rust
pub struct MemoryTierConfig {
    /// Working memory capacity (default: 7)
    pub working_capacity: usize,

    /// Session memory timeout (default: 30 minutes)
    pub session_timeout: Duration,

    /// Promotion threshold (default: 0.5)
    pub promotion_importance: f32,

    /// Minimum accesses for auto-promotion (default: 3)
    pub promotion_access_count: usize,
}
```

## Metrics

The system tracks tier transitions for observability:

```rust
pub struct TierMetrics {
    pub working_memory_count: usize,
    pub session_memory_count: usize,
    pub long_term_memory_count: usize,
    pub promotions_to_session: u64,
    pub promotions_to_longterm: u64,
    pub displacements_from_working: u64,
}
```

These metrics are exposed via `/api/users/{user_id}/stats`.
