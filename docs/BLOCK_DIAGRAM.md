# SHODH-MEMORY SYSTEM ARCHITECTURE
## Block Diagram & Flowchart for Drone Memory System

**Version:** 1.0
**Date:** November 2025
**Application:** Drone Challenge Submission

---

## COMPLETE SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT SOURCES                              │
├─────────────────────────────────────────────────────────────────────┤
│  • Drone Sensors (Lidar, Camera, GPS, IMU)                          │
│  • Mission Logs (Flight data, waypoints, telemetry)                 │
│  • Operator Commands (Voice/text annotations)                       │
│  • Environmental Data (Weather, obstacles, terrain)                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE (REST API)                    │
├─────────────────────────────────────────────────────────────────────┤
│  POST /api/record                                                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Input: {                                                     │   │
│  │   user_id: "drone_fleet_01",                                 │   │
│  │   content: "Lidar detected red obstacle at waypoint 5",      │   │
│  │   metadata: {                                                │   │
│  │     gps: [lat, lon],                                         │   │
│  │     altitude: 50.2,                                          │   │
│  │     timestamp: "2025-11-22T14:30:00Z",                       │   │
│  │     sensor: "lidar_primary",                                 │   │
│  │     importance: 0.85                                         │   │
│  │   }                                                          │   │
│  │ }                                                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PARALLEL PROCESSING LAYER                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────────┐  ┌────────────────────┐  ┌──────────────┐   │
│  │ Entity Extraction  │  │ Embedding          │  │ Memory       │   │
│  │ (NLP Heuristics)   │  │ Generation         │  │ Storage      │   │
│  │                    │  │ (MiniLM-L6-v2)     │  │ (RocksDB)    │   │
│  │ IC Weighting:      │  │                    │  │              │   │
│  │ • "lidar"   →2.3   │  │ Text → 384-dim     │  │ Persistent   │   │
│  │ • "red"     →1.7   │  │ vector embedding   │  │ key-value    │   │
│  │ • "obstacle"→2.3   │  │                    │  │ store        │   │
│  │ • "waypoint"→2.3   │  │ Cosine similarity  │  │ ACID         │   │
│  │ • "detected"→1.0   │  │ optimized          │  │ guarantees   │   │
│  └────────┬───────────┘  └─────────┬──────────┘  └──────┬───────┘   │
│           │                        │                     │          │
│           └────────────────────────┼─────────────────────┘          │
│                                    │                                │
└────────────────────────────────────┼─────────────────────────────── ┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH DATABASE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ENTITY NODES (Semantic Objects)                            │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │  │
│  │  │  Lidar   │  │ Obstacle │  │ Waypoint │  │  Robot   │    │  │
│  │  │ (Sensor) │  │ (Object) │  │(Location)│  │ (Agent)  │    │  │
│  │  │ UUID: a1 │  │ UUID: b2 │  │ UUID: c3 │  │ UUID: d4 │    │  │
│  │  │ Type:Tech│  │ Type:Obj │  │ Type:Loc │  │ Type:Agt │    │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │  │
│  │       │             │              │             │          │  │
│  └───────┼─────────────┼──────────────┼─────────────┼──────────┘  │
│          │             │              │             │             │
│  ┌───────┼─────────────┼──────────────┼─────────────┼──────────┐  │
│  │  RELATIONSHIP EDGES (Weighted Connections)                  │  │
│  │       │             │              │             │          │  │
│  │  ┌────▼─────┐  ┌────▼──────┐  ┌───▼──────┐  ┌──▼───────┐  │  │
│  │  │ "detects"│  │ "locatedAt│  │ "uses"   │  │"WorksAt" │  │  │
│  │  │ strength:│  │ strength: │  │ strength:│  │ strength:│  │  │
│  │  │   0.90   │  │   0.80    │  │   0.70   │  │   0.85   │  │  │
│  │  │ created: │  │ created:  │  │ created: │  │ created: │  │  │
│  │  │ 2025-11  │  │ 2025-11   │  │ 2025-11  │  │ 2025-11  │  │  │
│  │  └──────────┘  └───────────┘  └──────────┘  └──────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  EPISODIC NODES (Memory Episodes)                           │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │ Episode #42                                             │ │  │
│  │  │ UUID: e42-uuid                                          │ │  │
│  │  │ Content: "Lidar detected red obstacle at waypoint 5"    │ │  │
│  │  │ Timestamp: 2025-11-22T14:30:00Z                         │ │  │
│  │  │ Importance: 0.85                                        │ │  │
│  │  │ Entity Refs: [a1, b2, c3, d4]                           │ │  │
│  │  │ Embedding: [0.12, -0.34, 0.56, ..., 0.21] (384-dim)    │ │  │
│  │  │ Metadata: {gps: [...], altitude: 50.2, sensor: "..."}  │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL PIPELINE                              │
├─────────────────────────────────────────────────────────────────────┤
│  POST /api/retrieve                                                 │
│  Input: {                                                           │
│    query_text: "red obstacle near waypoint",                        │
│    mode: "hybrid",                                                  │
│    max_results: 10                                                  │
│  }                                                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: LINGUISTIC QUERY ANALYSIS                                  │
│  (Based on Lioma & Ounis 2006: IC-weighted POS classification)     │
├─────────────────────────────────────────────────────────────────────┤
│  Input: "red obstacle near waypoint"                                │
│                                                                     │
│  Tokenization & Classification:                                     │
│  ┌──────────┬──────────────┬─────────────┬────────────────────┐   │
│  │  Token   │  POS Type    │  IC Weight  │  Classification    │   │
│  ├──────────┼──────────────┼─────────────┼────────────────────┤   │
│  │  "red"   │  Adjective   │    1.7      │  Modifier          │   │
│  │"obstacle"│  Noun        │    2.3      │  Focal Entity      │   │
│  │  "near"  │  Preposition │    1.0      │  Relation          │   │
│  │"waypoint"│  Noun        │    2.3      │  Focal Entity      │   │
│  └──────────┴──────────────┴─────────────┴────────────────────┘   │
│                                                                     │
│  Output: QueryAnalysis {                                            │
│    focal_entities: ["obstacle", "waypoint"],                       │
│    discriminative_modifiers: ["red"],                              │
│    relational_context: ["near"],                                   │
│    total_weight: 2.3 + 2.3 + 1.7 + 1.0 = 7.3                      │
│  }                                                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: INITIAL ENTITY ACTIVATION                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Look up entities in knowledge graph:                               │
│                                                                     │
│  • "obstacle" found → Entity b2 (activation = IC_NOUN = 2.3)       │
│  • "waypoint" found → Entity c3 (activation = IC_NOUN = 2.3)       │
│  • "red" (modifier) → Boosts related entities by 1.7               │
│                                                                     │
│  Initial Activation Map:                                            │
│  {                                                                  │
│    b2 (obstacle): 2.3,                                             │
│    c3 (waypoint): 2.3                                              │
│  }                                                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: SPREADING ACTIVATION ALGORITHM                             │
│  (Based on Anderson & Pirolli 1984)                                │
├─────────────────────────────────────────────────────────────────────┤
│  Formula: A(d) = A₀ × e^(-λ×d) × edge_strength                     │
│  where: λ = 0.5 (decay rate), d = hop distance (1-3)               │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ HOP 1 (decay factor = e^(-0.5×1) = 0.606)                   │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ From b2 (obstacle, A₀=2.3):                                 │   │
│  │   → a1 (lidar)  : 2.3 × 0.606 × 0.9 = 1.25                 │   │
│  │   → d4 (robot)  : 2.3 × 0.606 × 0.8 = 1.12                 │   │
│  │                                                              │   │
│  │ From c3 (waypoint, A₀=2.3):                                 │   │
│  │   → loc1 (location): 2.3 × 0.606 × 0.7 = 0.98              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ HOP 2 (decay factor = e^(-0.5×2) = 0.368)                   │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ From a1 (lidar, A₀=1.25):                                   │   │
│  │   → sen1 (sensor): 1.25 × 0.368 × 0.8 = 0.37               │   │
│  │                                                              │   │
│  │ From d4 (robot, A₀=1.12):                                   │   │
│  │   → nav1 (navigation): 1.12 × 0.368 × 0.6 = 0.25           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ HOP 3 (decay factor = e^(-0.5×3) = 0.223)                   │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ From sen1 (sensor, A₀=0.37):                                │   │
│  │   → det1 (detection): 0.37 × 0.223 × 0.5 = 0.041           │   │
│  │                                                              │   │
│  │ From nav1 (navigation, A₀=0.25):                            │   │
│  │   → path1 (path): 0.25 × 0.223 × 0.4 = 0.022               │   │
│  │   ❌ PRUNED (below threshold 0.01)                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Prune activations below threshold (0.01)                          │
│                                                                     │
│  Final Activation Map:                                              │
│  {                                                                  │
│    b2 (obstacle):    2.30,                                         │
│    c3 (waypoint):    2.30,                                         │
│    a1 (lidar):       1.25,                                         │
│    d4 (robot):       1.12,                                         │
│    loc1 (location):  0.98,                                         │
│    sen1 (sensor):    0.37,                                         │
│    nav1 (navigation):0.25,                                         │
│    det1 (detection): 0.04                                          │
│  }                                                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: EPISODE RETRIEVAL FROM ACTIVATED ENTITIES                  │
├─────────────────────────────────────────────────────────────────────┤
│  For each activated entity, retrieve connected episodes:            │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Episode #42: "Lidar detected red obstacle at waypoint 5"    │   │
│  │ Connected to: [a1, b2, c3, d4]                              │   │
│  │ Graph Activation = 2.30 + 2.30 + 1.25 + 1.12 = 6.97        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Episode #108: "Red traffic cone blocking path"              │   │
│  │ Connected to: [b2]                                           │   │
│  │ Graph Activation = 2.30                                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Episode #73: "Robot navigating around obstacle zone"        │   │
│  │ Connected to: [d4, nav1]                                     │   │
│  │ Graph Activation = 1.12 + 0.25 = 1.37                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: HYBRID SCORING (Xiong et al. 2017)                        │
├─────────────────────────────────────────────────────────────────────┤
│  For each episode, compute three independent scores:                │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Episode #42 Scoring Breakdown:                               │   │
│  │                                                              │   │
│  │ 1️⃣ GRAPH ACTIVATION SCORE (60% weight)                       │   │
│  │    Raw activation: 6.97                                      │   │
│  │    Normalized (max=8.0): 6.97/8.0 = 0.871                   │   │
│  │    Component: 0.60 × 0.871 = 0.523                          │   │
│  │                                                              │   │
│  │ 2️⃣ SEMANTIC SIMILARITY SCORE (25% weight)                    │   │
│  │    Query embedding: encode("red obstacle near waypoint")     │   │
│  │    Episode embedding: [0.12, -0.34, 0.56, ..., 0.21]        │   │
│  │    Cosine similarity: 0.847                                  │   │
│  │    Component: 0.25 × 0.847 = 0.212                          │   │
│  │                                                              │   │
│  │ 3️⃣ LINGUISTIC MATCH SCORE (15% weight)                       │   │
│  │    Matched entities: "obstacle" (1.0), "waypoint" (1.0)     │   │
│  │    Matched modifiers: "red" (0.5)                            │   │
│  │    Matched relations: "near" (0.2)                           │   │
│  │    Total: 2.7 points                                         │   │
│  │    Max possible: 2.3 + 2.3 + 1.7 + 1.0 = 7.3               │   │
│  │    Normalized: 2.7/7.3 = 0.370                              │   │
│  │    Component: 0.15 × 0.370 = 0.056                          │   │
│  │                                                              │   │
│  │ FINAL SCORE = 0.523 + 0.212 + 0.056 = 0.791                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Similarly compute for Episode #108: final_score = 0.628           │
│  Similarly compute for Episode #73:  final_score = 0.492           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6: RANKING & RESULT PREPARATION                               │
├─────────────────────────────────────────────────────────────────────┤
│  Sort episodes by final_score (descending):                         │
│                                                                     │
│  Ranked Results:                                                    │
│  ┌─────┬────────────┬──────────┬───────────────────────────────┐   │
│  │ Rank│ Episode ID │  Score   │ Content Preview              │   │
│  ├─────┼────────────┼──────────┼───────────────────────────────┤   │
│  │  1  │   #42      │  0.791   │ "Lidar detected red obsta..." │   │
│  │  2  │   #108     │  0.628   │ "Red traffic cone blocki..."  │   │
│  │  3  │   #73      │  0.492   │ "Robot navigating around..."  │   │
│  └─────┴────────────┴──────────┴───────────────────────────────┘   │
│                                                                     │
│  Apply max_results limit (10)                                      │
│  Enrich with metadata (GPS, timestamp, importance)                 │
│  Format as JSON response                                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        API RESPONSE                                 │
├─────────────────────────────────────────────────────────────────────┤
│  {                                                                  │
│    "memories": [                                                    │
│      {                                                              │
│        "id": "e42-uuid",                                            │
│        "content": "Lidar detected red obstacle at waypoint 5",     │
│        "score": 0.791,                                              │
│        "importance": 0.85,                                          │
│        "created_at": "2025-11-22T14:30:00Z",                       │
│        "metadata": {                                                │
│          "gps": [12.9716, 77.5946],                                │
│          "altitude": 50.2,                                          │
│          "sensor": "lidar_primary"                                  │
│        }                                                            │
│      },                                                             │
│      ...                                                            │
│    ],                                                               │
│    "count": 3,                                                      │
│    "query_time_ms": 147                                             │
│  }                                                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  INTEGRATION LAYER (Consumers)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │ Flight           │  │ Mission          │  │ Operator        │  │
│  │ Controller       │  │ Planner          │  │ Dashboard       │  │
│  │                  │  │                  │  │                 │  │
│  │ • Real-time      │  │ • Historical     │  │ • Web UI        │  │
│  │   obstacle       │  │   analysis       │  │ • Query         │  │
│  │   context        │  │ • Route          │  │   interface     │  │
│  │ • Adaptive       │  │   optimization   │  │ • Analytics     │  │
│  │   navigation     │  │ • Risk           │  │ • Visualization │  │
│  │                  │  │   assessment     │  │                 │  │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘  │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │ Analytics        │  │ Alert            │  │ Training        │  │
│  │ Platform         │  │ System           │  │ Simulator       │  │
│  │                  │  │                  │  │                 │  │
│  │ • Performance    │  │ • Anomaly        │  │ • Historical    │  │
│  │   metrics        │  │   detection      │  │   replay        │  │
│  │ • Trend          │  │ • Safety         │  │ • Scenario      │  │
│  │   analysis       │  │   warnings       │  │   testing       │  │
│  │ • Reports        │  │ • Notifications  │  │ • Validation    │  │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│              SUPPORTING INFRASTRUCTURE & SERVICES                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐   │
│  │ AUDIT & COMPLIANCE   │  │ MULTI-USER ISOLATION             │   │
│  │                      │  │                                  │   │
│  │ • Event logging      │  │ • Per-drone memory namespace     │   │
│  │ • 30-day retention   │  │ • GDPR-compliant deletion        │   │
│  │ • RocksDB persistent │  │ • API key authentication         │   │
│  │ • Tamper-proof logs  │  │ • Rate limiting (50 req/s)       │   │
│  │ • Compliance reports │  │ • User statistics tracking       │   │
│  └──────────────────────┘  └──────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐   │
│  │ BACKGROUND SERVICES  │  │ MONITORING & HEALTH              │   │
│  │                      │  │                                  │   │
│  │ • Compression        │  │ • Health check endpoint          │   │
│  │   pipeline           │  │ • Performance metrics            │   │
│  │ • Index optimization │  │ • Resource monitoring            │   │
│  │ • Log rotation       │  │ • Error tracking                 │   │
│  │ • Backup scheduling  │  │ • Uptime reporting               │   │
│  │ • Graceful shutdown  │  │ • Prometheus integration         │   │
│  └──────────────────────┘  └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## KEY PERFORMANCE METRICS

| Metric | Value | Notes |
|--------|-------|-------|
| **Retrieval Latency** | <200ms | End-to-end query processing |
| **Retrieval Accuracy** | 100% | 14/14 benchmark queries correct |
| **Score Diversity** | σ=0.18 | vs 0.00 for static scoring |
| **Throughput** | 50 req/s | Sustained, 100 burst |
| **Storage Backend** | RocksDB | ACID guarantees, crash recovery |
| **Embedding Model** | MiniLM-L6-v2 | 384-dim, 22M params |
| **Graph Hops** | 3 max | Configurable decay rate |
| **Multi-tenancy** | ✓ | Per-user isolation |
| **Persistence** | ✓ | Vector indices + graph + audit |

---

## RESEARCH FOUNDATIONS

This architecture implements algorithms from peer-reviewed research:

1. **Anderson & Pirolli (1984)** - Spreading activation decay formula
2. **Lioma & Ounis (2006)** - Information Content (IC) weighting for POS
3. **Xiong et al. (2017)** - Hybrid semantic ranking with graph embeddings
4. **Bendersky & Croft (2008)** - Key concept extraction from verbose queries
5. **Collins & Loftus (1975)** - Spreading activation theory foundations

---

## TECHNOLOGY STACK

- **Language**: Rust (production-grade, memory-safe)
- **Storage**: RocksDB (persistent key-value store)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Graph**: Custom in-memory + RocksDB persistence
- **API**: Axum web framework (REST)
- **Concurrency**: Tokio async runtime
- **Serialization**: Bincode + Serde

---

**Document Version:** 1.0
**Last Updated:** November 2025
**Prepared for:** Drone Challenge Application
