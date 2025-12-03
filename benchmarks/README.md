# Shodh-Memory Benchmarks

Comprehensive benchmark suite for validating performance claims and comparing against mem0.

## Quick Start

### 1. Benchmark Shodh-Memory

```bash
# Start server first
cargo run --release

# In another terminal, run benchmarks
python benchmarks/benchmark_shodh.py
```

### 2. Benchmark mem0 (optional)

```bash
# Install mem0
pip install mem0ai

# Run mem0 benchmarks
python benchmarks/benchmark_mem0.py
```

### 3. Compare Results

```bash
python benchmarks/compare.py
```

## Benchmark Suite

### Operations Tested

1. **ADD** - Memory creation (100 iterations)
2. **SEARCH** - Memory retrieval (100 iterations)
3. **GET_ALL** - Fetch all memories (50 iterations)
4. **UPDATE** - Memory modification (100 iterations)
5. **DELETE** - Memory deletion (50 iterations)

### Metrics Collected

- **Mean Latency** - Average response time
- **Median Latency** - 50th percentile
- **P95 Latency** - 95th percentile
- **P99 Latency** - 99th percentile
- **Min/Max** - Best and worst case

## Claims Under Test

From README.md:

| Claim | Target | Test |
|-------|--------|------|
| "10x faster than Python" | 10x speedup | Compare mean latencies |
| "<1ms ADD" | <1ms | Measure ADD mean |
| "<1ms SEARCH (working)" | <1ms | Measure SEARCH median |
| "<10ms semantic search" | <10ms | Measure with embeddings |
| "<100ms startup" | <100ms | Measure server startup |

## Results Structure

```
benchmarks/results/
├── shodh_benchmark_TIMESTAMP.json    # Shodh-Memory results
├── mem0_benchmark_TIMESTAMP.json     # mem0 results (if run)
├── comparison_TIMESTAMP.json         # Side-by-side comparison
└── BENCHMARK_REPORT.md              # Human-readable report
```

## Running in CI/CD

Benchmarks run automatically on every push to main:

```yaml
# See .github/workflows/benchmark.yml
```

View results at: [GitHub Actions](../../actions)

## Interpreting Results

### Good Performance Indicators

- **ADD**: <1ms mean (claim: <1ms) ✅
- **SEARCH**: <1ms median for working memory ✅
- **P95**: <10ms for most operations ✅
- **Speedup vs mem0**: >5x average ✅

### Performance Degradation Signs

- **High P99**: >100ms indicates outliers ⚠️
- **High Stdev**: Inconsistent performance ⚠️
- **Increasing over time**: Performance regression ⚠️

## Test Environment

**Recommended specs:**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: SSD
- OS: Linux/macOS (Windows supported but slower)

**Dependencies:**
- Python 3.8+
- requests library
- mem0ai (optional, for comparison)

## Continuous Benchmarking

Track performance over time:

```bash
# Run benchmark
python benchmarks/benchmark_shodh.py

# Results saved to benchmarks/results/ with timestamp
# Compare with previous runs to detect regressions
```

## Troubleshooting

### Server Not Starting

```bash
# Check if server is running
curl http://localhost:3030/health

# Start server manually
cargo run --release
```

### mem0 Not Installed

```bash
pip install mem0ai

# Or skip mem0 comparison
python benchmarks/benchmark_shodh.py
```

### Inconsistent Results

- Close other applications
- Run multiple times and average
- Ensure server has warmed up (run once, discard, run again)

## Contributing

To add new benchmarks:

1. Add test to `benchmark_shodh.py`
2. Add corresponding test to `benchmark_mem0.py`
3. Update `compare.py` to include new operation
4. Update this README with new claim/target

## License

Apache 2.0
