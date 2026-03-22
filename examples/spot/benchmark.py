"""
Spot Memory Performance Benchmark

Measures real latency numbers at robotics scale — the kind of evidence
a BD or Oxford engineer needs before integrating a new dependency.

Tests:
  - Store latency at N = 100, 500, 1000, 5000, 10000 obstacles
  - Recall latency (semantic search by tag)
  - Spatial recall latency (Euclidean post-filter on stored positions)
  - Memory footprint (RocksDB directory size on disk)

Run:
    pip install shodh-memory
    python benchmark.py
"""

import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

from shodh_spot_bridge import SpotMemoryBridge


def get_dir_size_mb(path: str) -> float:
    """Get total size of a directory in megabytes."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def random_position(bounds: float = 100.0) -> Tuple[float, float, float]:
    """Generate a random 3D position within bounds."""
    return (
        random.uniform(-bounds, bounds),
        random.uniform(-bounds, bounds),
        random.uniform(-bounds / 10, bounds / 10),
    )


def percentile(sorted_values: List[float], pct: float) -> float:
    """Calculate percentile from a sorted list."""
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * pct / 100.0)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


def benchmark_store(bridge: SpotMemoryBridge, n: int) -> List[float]:
    """Store N obstacles and return per-operation latencies in ms."""
    latencies = []
    for i in range(n):
        pos = random_position()
        desc = f"Obstacle_{i:05d} at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"

        t0 = time.perf_counter()
        bridge.record_obstacle(description=desc, position=pos, confidence=0.85)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    return latencies


def benchmark_recall_semantic(bridge: SpotMemoryBridge, n_queries: int = 100) -> List[float]:
    """Measure semantic recall latency (tag-filtered hybrid search)."""
    latencies = []
    for _ in range(n_queries):
        t0 = time.perf_counter()
        bridge.memory.recall(
            query="obstacle hazard",
            limit=10,
            mode="hybrid",
            tags=["obstacle"],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
    return latencies


def benchmark_recall_spatial(
    bridge: SpotMemoryBridge,
    n_queries: int = 100,
    radius: float = 10.0,
) -> List[float]:
    """Measure spatial recall latency (Euclidean post-filter)."""
    latencies = []
    for _ in range(n_queries):
        pos = random_position()
        t0 = time.perf_counter()
        bridge.recall_obstacles_nearby(position=pos, radius=radius, limit=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
    return latencies


def format_latencies(latencies: List[float]) -> str:
    """Format latency stats as a compact string."""
    s = sorted(latencies)
    p50 = percentile(s, 50)
    p95 = percentile(s, 95)
    p99 = percentile(s, 99)
    return f"p50={p50:6.1f}ms  p95={p95:6.1f}ms  p99={p99:6.1f}ms"


def run_benchmark():
    print("=" * 74)
    print("  SHODH-MEMORY PERFORMANCE BENCHMARK — Robotics Scale")
    print("=" * 74)
    print()

    demo_path = "./spot_benchmark_data"
    scales = [100, 500, 1000, 5000, 10000]

    results: List[Dict] = []

    for n in scales:
        # Clean slate for each scale
        if Path(demo_path).exists():
            shutil.rmtree(demo_path)

        bridge = SpotMemoryBridge(storage_path=demo_path, robot_id="bench_robot")
        bridge.start_mission(f"benchmark_{n}")

        print(f"  N = {n:,} obstacles")
        print(f"  {'─' * 66}")

        # Store
        store_lat = benchmark_store(bridge, n)
        bridge.flush()
        print(f"    Store:          {format_latencies(store_lat)}")

        # Recall (semantic)
        n_queries = min(100, n)
        recall_lat = benchmark_recall_semantic(bridge, n_queries)
        print(f"    Recall (semantic): {format_latencies(recall_lat)}")

        # Recall (spatial)
        spatial_lat = benchmark_recall_spatial(bridge, n_queries)
        print(f"    Recall (spatial):  {format_latencies(spatial_lat)}")

        # Disk footprint
        size_mb = get_dir_size_mb(demo_path)
        print(f"    Disk footprint: {size_mb:.2f} MB")
        print()

        bridge.end_mission()
        bridge.flush()
        del bridge

        results.append({
            "n": n,
            "store_p50": percentile(sorted(store_lat), 50),
            "store_p95": percentile(sorted(store_lat), 95),
            "recall_p50": percentile(sorted(recall_lat), 50),
            "recall_p95": percentile(sorted(recall_lat), 95),
            "spatial_p50": percentile(sorted(spatial_lat), 50),
            "spatial_p95": percentile(sorted(spatial_lat), 95),
            "disk_mb": size_mb,
        })

    # Summary table
    print("=" * 74)
    print("  SUMMARY")
    print("=" * 74)
    print()
    print(f"  {'N':>7s} | {'Store p50':>10s} {'p95':>7s} | {'Recall p50':>11s} {'p95':>7s} | {'Spatial p50':>12s} {'p95':>7s} | {'Disk':>6s}")
    print(f"  {'─' * 7} | {'─' * 18} | {'─' * 19} | {'─' * 20} | {'─' * 6}")

    for r in results:
        print(
            f"  {r['n']:>7,} | "
            f"{r['store_p50']:>7.1f}ms {r['store_p95']:>6.1f}ms | "
            f"{r['recall_p50']:>8.1f}ms {r['recall_p95']:>6.1f}ms | "
            f"{r['spatial_p50']:>9.1f}ms {r['spatial_p95']:>6.1f}ms | "
            f"{r['disk_mb']:>5.1f}M"
        )

    print()
    print("  Notes:")
    print("    - Store: includes embedding generation (MiniLM-L6-v2 via ONNX Runtime)")
    print("    - Recall: hybrid mode (vector similarity + tag filter)")
    print("    - Spatial: hybrid recall + Euclidean distance post-filter")
    print("    - Disk: RocksDB with MessagePack serialization")
    print("    - Hardware: your current machine (run on Jetson for edge numbers)")
    print()

    # Cleanup
    if Path(demo_path).exists():
        shutil.rmtree(demo_path)


if __name__ == "__main__":
    run_benchmark()
