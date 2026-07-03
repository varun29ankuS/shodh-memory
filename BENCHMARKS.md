# Shodh-Memory Performance Benchmarks

Comprehensive performance measurements for shodh-memory (last measured on v0.1.5; current release v0.2.0).

**Test Environment:** Windows 11, Intel i7-1355U (10 cores, 12 threads), Rust 1.75, Release build with LTO

---

## Quick Summary

| Component | Latency | Verdict |
|-----------|---------|---------|
| Entity lookup (1000 entities) | **763 ns** | ⚡ Sub-microsecond |
| Graph stats | **26 ns - 7 μs** | ⚡ O(1) |
| Vector search (1000 vectors) | **2-5 ms** | ✅ Fast |
| Memory store (with embedding) | **150-250 ms** | 🟡 ONNX bottleneck |
| Hebbian edge update | **<1 ms** | ⚡ Negligible overhead |

---

## 1. Hebbian Learning (December 2025)

### Graph Stats (O(1) cache reads)

| Operation | Time |
|-----------|------|
| Empty graph | **26 ns** |
| 10 memories | **148 ns** |
| 50 memories | **446 ns** |
| 100 memories | **604 ns** |
| 500 memories | **7.1 μs** |

### Edge Formation (includes embedding generation)

| Memories | Time | Edges Formed |
|----------|------|--------------|
| 2 | **37.6 ms** | 1 edge |
| 5 | **47.5 ms** | 10 edges |
| 10 | **49.6 ms** | 45 edges |
| 20 | **52.3 ms** | 190 edges |

### Edge Strengthening (LTP pathway)

| Coactivations | Time |
|---------------|------|
| 1x | **43 ms** |
| 3x | **56.5 ms** |
| 5x | **139.5 ms** |
| 10x | **44.9 ms** |

### Graph Persistence (save + reload)

| Edges | Time |
|-------|------|
| 10 | **86.7 ms** |
| 50 | **92.4 ms** |
| 100 | **103.3 ms** |

### Key Operations

| Operation | Time |
|-----------|------|
| Associative retrieval | **107 ms** |
| LTP threshold (5 coact) | **48 ms** |

**Conclusion:** Hebbian learning adds **<1ms overhead**. Embedding generation (ONNX) is the bottleneck.

---

## 2. Graph Memory (Knowledge Graph)

### Entity Operations

| Operation | Scale | Latency | Throughput |
|-----------|-------|---------|------------|
| Entity Get | 1000 entities | **763 ns** | ~1.3M ops/sec |
| Entity Search | 1000 entities | **775 ns** | ~1.3M ops/sec |
| Entity Add | 100 entities | 23.6 ms | ~4 ops/sec |

### Relationship Operations

| Operation | Scale | Latency |
|-----------|-------|---------|
| Query | 1000e/500r | **2.2 μs** |
| Add | 100 rels | 13.5 ms |

### Graph Traversal

| Depth | Latency | Notes |
|-------|---------|-------|
| 1-hop | **10 μs** | Direct neighbors |
| 2-hop | **20 μs** | Friends of friends |
| 3-hop | **30 μs** | Extended network |

### Salience & Hebbian Plasticity

| Operation | Scale | Latency |
|-----------|-------|---------|
| Hebbian Strengthen | 10 entities | **5.70 μs** |
| Hebbian Strengthen | 100 entities | **7.54 μs** |
| Hebbian Strengthen | 500 entities | **6.03 μs** |
| Salience Update | 1 entity | **8.49 μs** |
| Salience Update | 100 entities | **894 μs** |

---

## 3. Vector Search (Vamana/DiskANN)

| Query Type | Graph Size | Latency |
|------------|------------|---------|
| Top-5 ANN | 100 vectors | **~2 ms** |
| Top-5 ANN | 1000 vectors | **~5 ms** |
| Top-10 ANN | 1000 vectors | **~8 ms** |

### Embedding Generation (MiniLM-L6)

| Text Length | Latency |
|-------------|---------|
| Short (~10 words) | 15-25 ms |
| Medium (~50 words) | 20-30 ms |
| Batch (10 texts) | 50-80 ms |

---

## 4. Associative Retrieval (Density-Dependent Hybrid)

### Cross-Domain Association Discovery

| Scenario | Latency |
|----------|---------|
| coffee → AWS | **176 μs** |
| AWS → coffee | **189 μs** |
| Jeff → Seattle | **194 μs** |
| Temporal chain | **140 μs** |
| Team discovery | **227 μs** |

### Density-Dependent Weights

| Graph Density | Graph Weight | Latency |
|---------------|--------------|---------|
| Sparse (d=0.5) | 10% | **80 μs** |
| Medium (d=2.0) | 44% | **504 μs** |
| Dense (d=5.0) | ~50% | **2.73 ms** |

---

## 5. Proactive Memory Surfacing

Target: **<30ms** for real-time use

### Entity Extraction (Rule-based NER)

| Context Length | Time |
|----------------|------|
| Short (~6 words) | **1.33 μs** |
| Medium (~15 words) | **2.83 μs** |
| Long (~50 words) | **10.68 μs** |

### Full Relevance Pipeline

| Database Size | Latency | vs 30ms Target |
|---------------|---------|----------------|
| 10 memories | **61 μs** | ✅ 492x under |
| 50 memories | **630 μs** | ✅ 48x under |
| 100 memories | **374 μs** | ✅ 80x under |
| 200 memories | **660 μs** | ✅ 45x under |

---

## 6. Streaming Memory Extraction

### Content Hashing (Optimized)

| Benchmark | Latency |
|-----------|---------|
| hash_short | **36.58 ns** |
| hash_medium | **330.80 ns** |
| hash_long | **280.46 ns** |
| dedup_check_100_items | **37.70 ns** |

### Importance Calculation

| Benchmark | Latency |
|-----------|---------|
| short_neutral | **215.43 ns** |
| short_important | **59.20 ns** |
| with_entities | **77.25 ns** |
| error_content | **63.22 ns** |

---

## 7. External Integrations (Linear & GitHub)

### Linear Integration

| Operation | Latency |
|-----------|---------|
| Payload Parsing | **2.34 μs** |
| HMAC-SHA256 Verification | **864 ns** |
| Content Transformation | **903 ns** |
| Tag Extraction | **305 ns** |
| **Full Pipeline** | **6.06 μs** |

### GitHub Integration

| Operation | Latency |
|-----------|---------|
| Payload Parsing | **1.86 μs** |
| HMAC-SHA256 Verification | **842 ns** |
| Issue Transformation | **615 ns** |
| PR Transformation | **1.00 μs** |
| **Full Pipeline** | **4.40 μs** |

**Throughput:** ~150K/sec (Linear), ~225K/sec (GitHub) per core

---

## 8. Competitive Comparison

### vs. Neo4j

| Metric | Shodh-Memory | Neo4j | Winner |
|--------|--------------|-------|--------|
| Entity Lookup | **763 ns** | 1-5 ms | Shodh (1000x) |
| Memory Footprint | ~50 MB | 1-4 GB | Shodh |
| Cold Start | ~200 ms | 5-30 s | Shodh |

### vs. Pinecone

| Metric | Shodh-Memory | Pinecone | Winner |
|--------|--------------|----------|--------|
| Latency | **2-5 ms** | 20-100 ms | Shodh (10x) |
| Cost | $0 | $70+/month | Shodh |
| Privacy | Local | Cloud | Shodh |

### vs. Mem0

| Metric | Shodh-Memory | Mem0 | Winner |
|--------|--------------|------|--------|
| Latency | **1-10 ms** | 100-500 ms | Shodh (100x) |
| Hebbian Learning | Yes | No | Shodh |
| Edge Deployment | Yes | No | Shodh |

---

## Running Benchmarks

```bash
# All benchmarks
ORT_DYLIB_PATH=/path/to/onnxruntime.dll cargo bench

# Specific benchmark
cargo bench --bench hebbian_benchmarks
cargo bench --bench memory_benchmarks
cargo bench --bench graph_benchmarks
```

---

*Last updated: December 16, 2025*
