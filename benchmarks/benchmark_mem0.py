#!/usr/bin/env python3
"""
Benchmark mem0 performance for comparison
Requires: pip install mem0ai
"""

import time
import statistics
import json
from typing import List, Dict
from datetime import datetime

try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    print("⚠️  mem0 not installed. Install with: pip install mem0ai")

NUM_ITERATIONS = 100

class Mem0Benchmark:
    def __init__(self):
        if not MEM0_AVAILABLE:
            raise ImportError("mem0 not available")

        self.user_id = f"benchmark_user_{int(time.time())}"
        self.memory = Memory()
        self.results = {}
        self.memory_ids = []

    def benchmark_add(self, num_ops: int = NUM_ITERATIONS) -> Dict[str, float]:
        """Benchmark memory addition"""
        print(f"\n📝 Benchmarking ADD operation ({num_ops} iterations)...")

        latencies = []
        for i in range(num_ops):
            content = f"Benchmark memory {i}: This is a test memory for benchmarking purposes"

            start = time.perf_counter()
            result = self.memory.add(
                content,
                user_id=self.user_id,
                metadata={"iteration": i, "benchmark": True}
            )
            latency = (time.perf_counter() - start) * 1000

            # Store memory ID for later operations
            if isinstance(result, dict) and "id" in result:
                self.memory_ids.append(result["id"])

            latencies.append(latency)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{num_ops}")

        return self._calculate_stats("ADD", latencies)

    def benchmark_search(self, num_ops: int = NUM_ITERATIONS) -> Dict[str, float]:
        """Benchmark memory search"""
        print(f"\n🔍 Benchmarking SEARCH operation ({num_ops} iterations)...")

        latencies = []
        queries = ["benchmark", "test", "memory", "iteration", "purposes"]

        for i in range(num_ops):
            query = queries[i % len(queries)]

            start = time.perf_counter()
            results = self.memory.search(query, user_id=self.user_id, limit=10)
            latency = (time.perf_counter() - start) * 1000

            latencies.append(latency)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{num_ops}")

        return self._calculate_stats("SEARCH", latencies)

    def benchmark_get_all(self, num_ops: int = 50) -> Dict[str, float]:
        """Benchmark get all memories"""
        print(f"\n📚 Benchmarking GET_ALL operation ({num_ops} iterations)...")

        latencies = []

        for i in range(num_ops):
            start = time.perf_counter()
            results = self.memory.get_all(user_id=self.user_id)
            latency = (time.perf_counter() - start) * 1000

            latencies.append(latency)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{num_ops}")

        return self._calculate_stats("GET_ALL", latencies)

    def benchmark_update(self, num_ops: int = NUM_ITERATIONS) -> Dict[str, float]:
        """Benchmark memory update"""
        print(f"\n✏️ Benchmarking UPDATE operation ({num_ops} iterations)...")

        if not self.memory_ids:
            print("✗ No memory IDs available for update")
            return {}

        latencies = []

        for i in range(min(num_ops, len(self.memory_ids))):
            memory_id = self.memory_ids[i]
            new_content = f"Updated benchmark memory {i}"

            start = time.perf_counter()
            self.memory.update(memory_id, data=new_content)
            latency = (time.perf_counter() - start) * 1000

            latencies.append(latency)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(latencies)}")

        return self._calculate_stats("UPDATE", latencies)

    def benchmark_delete(self, num_ops: int = 50) -> Dict[str, float]:
        """Benchmark memory deletion"""
        print(f"\n🗑️ Benchmarking DELETE operation ({num_ops} iterations)...")

        if not self.memory_ids:
            print("✗ No memory IDs available for deletion")
            return {}

        latencies = []

        for i in range(min(num_ops, len(self.memory_ids))):
            memory_id = self.memory_ids[i]

            start = time.perf_counter()
            self.memory.delete(memory_id)
            latency = (time.perf_counter() - start) * 1000

            latencies.append(latency)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(latencies)}")

        return self._calculate_stats("DELETE", latencies)

    def _calculate_stats(self, operation: str, latencies: List[float]) -> Dict[str, float]:
        """Calculate statistics from latency measurements"""
        if not latencies:
            return {}

        stats = {
            "operation": operation,
            "count": len(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "p95": self._percentile(latencies, 95),
            "p99": self._percentile(latencies, 99),
            "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0
        }

        self.results[operation] = stats

        print(f"\n  Results for {operation}:")
        print(f"    Mean:   {stats['mean']:.2f}ms")
        print(f"    Median: {stats['median']:.2f}ms")
        print(f"    P95:    {stats['p95']:.2f}ms")
        print(f"    P99:    {stats['p99']:.2f}ms")
        print(f"    Min:    {stats['min']:.2f}ms")
        print(f"    Max:    {stats['max']:.2f}ms")

        return stats

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        size = len(data)
        return sorted(data)[int(size * percentile / 100)]

    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("=" * 70)
        print("🚀 MEM0 BENCHMARK SUITE")
        print("=" * 70)
        print(f"User ID: {self.user_id}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print()

        # Run benchmarks
        self.benchmark_add(NUM_ITERATIONS)
        self.benchmark_search(NUM_ITERATIONS)
        self.benchmark_get_all(50)
        self.benchmark_update(NUM_ITERATIONS)
        self.benchmark_delete(50)

        # Summary
        print("\n" + "=" * 70)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 70)

        summary = []
        for op, stats in self.results.items():
            summary.append({
                "operation": op,
                "mean_ms": round(stats["mean"], 2),
                "median_ms": round(stats["median"], 2),
                "p95_ms": round(stats["p95"], 2),
                "iterations": stats["count"]
            })

        # Print table
        print(f"\n{'Operation':<15} {'Mean':<12} {'Median':<12} {'P95':<12} {'Iterations':<12}")
        print("-" * 70)
        for s in summary:
            print(f"{s['operation']:<15} {s['mean_ms']:<12} {s['median_ms']:<12} {s['p95_ms']:<12} {s['iterations']:<12}")

        # Save results
        results_file = f"benchmarks/results/mem0_benchmark_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "system": "mem0",
                "results": self.results,
                "summary": summary
            }, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")

        return self.results

if __name__ == "__main__":
    if not MEM0_AVAILABLE:
        print("❌ Cannot run benchmark: mem0 not installed")
        exit(1)

    benchmark = Mem0Benchmark()
    benchmark.run_full_benchmark()
