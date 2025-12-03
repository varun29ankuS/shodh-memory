#!/usr/bin/env python3
"""
Benchmark Shodh-Memory performance
Tests: add, search, get_all, update, delete operations
"""

import requests
import time
import statistics
import json
import sys
from typing import List, Dict, Any
from datetime import datetime

API_BASE = "http://localhost:3030"
NUM_ITERATIONS = 100
NUM_MEMORIES = 1000

class ShodhBenchmark:
    def __init__(self, api_base: str, api_key: str = "shodh-dev-key-change-in-production"):
        self.api_base = api_base
        self.user_id = f"benchmark_user_{int(time.time())}"
        self.results = {}
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }

    def wait_for_server(self, timeout: int = 30):
        """Wait for server to be ready"""
        print(f"Waiting for server at {self.api_base}...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(f"{self.api_base}/health", timeout=1)
                if response.status_code == 200:
                    print("✓ Server is ready")
                    return True
            except requests.exceptions.RequestException:
                time.sleep(0.5)
        print(f"✗ Server not ready after {timeout}s")
        return False

    def benchmark_add(self, num_ops: int = NUM_ITERATIONS) -> Dict[str, float]:
        """Benchmark memory addition"""
        print(f"\n📝 Benchmarking ADD operation ({num_ops} iterations)...")

        latencies = []
        for i in range(num_ops):
            payload = {
                "user_id": self.user_id,
                "content": f"Benchmark memory {i}: This is a test memory for benchmarking purposes",
                "experience_type": "task",
                "entities": ["benchmark", "test", "memory"],
                "metadata": {
                    "iteration": str(i),
                    "benchmark": "true"
                }
            }

            start = time.perf_counter()
            response = requests.post(f"{self.api_base}/api/record", json=payload, headers=self.headers)
            latency = (time.perf_counter() - start) * 1000  # Convert to ms

            if response.status_code != 200:
                print(f"✗ Error on iteration {i}: {response.status_code}")
                continue

            latencies.append(latency)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{num_ops}")

        return self._calculate_stats("ADD", latencies)

    def benchmark_search(self, num_ops: int = NUM_ITERATIONS) -> Dict[str, float]:
        """Benchmark memory search"""
        print(f"\n🔍 Benchmarking SEARCH operation ({num_ops} iterations)...")

        latencies = []
        queries = ["benchmark", "test", "memory", "task", "iteration"]

        for i in range(num_ops):
            query = queries[i % len(queries)]
            payload = {
                "user_id": self.user_id,
                "query": query,
                "max_results": 10
            }

            start = time.perf_counter()
            response = requests.post(f"{self.api_base}/api/retrieve", json=payload, headers=self.headers)
            latency = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                print(f"✗ Error on iteration {i}: {response.status_code}")
                continue

            latencies.append(latency)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{num_ops}")

        return self._calculate_stats("SEARCH", latencies)

    def benchmark_get_all(self, num_ops: int = 50) -> Dict[str, float]:
        """Benchmark get all memories"""
        print(f"\n📚 Benchmarking GET_ALL operation ({num_ops} iterations)...")

        latencies = []

        for i in range(num_ops):
            payload = {
                "user_id": self.user_id,
                "limit": 1000
            }

            start = time.perf_counter()
            response = requests.post(f"{self.api_base}/api/memories", json=payload, headers=self.headers)
            latency = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                print(f"✗ Error on iteration {i}: {response.status_code}")
                continue

            latencies.append(latency)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{num_ops}")

        return self._calculate_stats("GET_ALL", latencies)

    def benchmark_update(self, num_ops: int = NUM_ITERATIONS) -> Dict[str, float]:
        """Benchmark memory update"""
        print(f"\n✏️ Benchmarking UPDATE operation ({num_ops} iterations)...")

        # First, get some memory IDs to update
        response = requests.post(f"{self.api_base}/api/memories", json={
            "user_id": self.user_id,
            "limit": num_ops
        }, headers=self.headers)

        if response.status_code != 200:
            print("✗ Failed to fetch memories for update benchmark")
            return {}

        memories = response.json().get("memories", [])
        if not memories:
            print("✗ No memories found to update")
            return {}

        latencies = []

        for i in range(min(num_ops, len(memories))):
            memory_id = memories[i]["id"]
            payload = {
                "content": f"Updated benchmark memory {i}",
                "importance": 0.85
            }

            start = time.perf_counter()
            response = requests.put(f"{self.api_base}/api/memory/{memory_id}", json=payload, headers=self.headers)
            latency = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                print(f"✗ Error on iteration {i}: {response.status_code}")
                continue

            latencies.append(latency)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(latencies)}")

        return self._calculate_stats("UPDATE", latencies)

    def benchmark_delete(self, num_ops: int = 50) -> Dict[str, float]:
        """Benchmark memory deletion"""
        print(f"\n🗑️ Benchmarking DELETE operation ({num_ops} iterations)...")

        # Get memory IDs to delete
        response = requests.post(f"{self.api_base}/api/memories", json={
            "user_id": self.user_id,
            "limit": num_ops
        }, headers=self.headers)

        if response.status_code != 200:
            print("✗ Failed to fetch memories for delete benchmark")
            return {}

        memories = response.json().get("memories", [])
        latencies = []

        for i in range(min(num_ops, len(memories))):
            memory_id = memories[i]["id"]

            start = time.perf_counter()
            response = requests.delete(f"{self.api_base}/api/memory/{memory_id}", headers=self.headers)
            latency = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                print(f"✗ Error on iteration {i}: {response.status_code}")
                continue

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
        print("🚀 SHODH-MEMORY BENCHMARK SUITE")
        print("=" * 70)
        print(f"Target: {self.api_base}")
        print(f"User ID: {self.user_id}")
        print(f"Timestamp: {datetime.now().isoformat()}")

        if not self.wait_for_server():
            print("\n❌ Server not available. Exiting.")
            sys.exit(1)

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
        results_file = f"benchmarks/results/shodh_benchmark_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "system": "shodh-memory",
                "results": self.results,
                "summary": summary
            }, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")

        return self.results

if __name__ == "__main__":
    benchmark = ShodhBenchmark(API_BASE)
    benchmark.run_full_benchmark()
