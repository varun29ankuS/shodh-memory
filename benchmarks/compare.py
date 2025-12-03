#!/usr/bin/env python3
"""
Compare Shodh-Memory vs mem0 performance
Generates comparison report with speedup calculations
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

def load_latest_results(system: str) -> Dict:
    """Load most recent benchmark results for a system"""
    results_dir = Path("benchmarks/results")

    if not results_dir.exists():
        print(f"✗ Results directory not found: {results_dir}")
        return None

    # Find latest results file
    pattern = f"{system}_benchmark_*.json"
    files = sorted(results_dir.glob(pattern), reverse=True)

    if not files:
        print(f"✗ No benchmark results found for {system}")
        return None

    with open(files[0], 'r') as f:
        return json.load(f)

def calculate_speedup(shodh_ms: float, mem0_ms: float) -> float:
    """Calculate speedup factor"""
    if mem0_ms == 0:
        return 0
    return mem0_ms / shodh_ms

def generate_comparison_report(shodh_results: Dict, mem0_results: Dict):
    """Generate detailed comparison report"""

    print("=" * 80)
    print("🏆 SHODH-MEMORY vs MEM0 PERFORMANCE COMPARISON")
    print("=" * 80)
    print()
    print(f"Shodh-Memory: {shodh_results.get('timestamp', 'Unknown')}")
    print(f"mem0:         {mem0_results.get('timestamp', 'Unknown')}")
    print()

    # Extract results
    shodh = {r['operation']: r for r in shodh_results['summary']}
    mem0_data = {r['operation']: r for r in mem0_results['summary']}

    # Comparison table
    print(f"{'Operation':<15} {'Shodh (ms)':<15} {'mem0 (ms)':<15} {'Speedup':<15} {'Status':<15}")
    print("-" * 80)

    comparisons = []
    for op in ['ADD', 'SEARCH', 'GET_ALL', 'UPDATE', 'DELETE']:
        if op not in shodh or op not in mem0_data:
            continue

        shodh_mean = shodh[op]['mean_ms']
        mem0_mean = mem0_data[op]['mean_ms']
        speedup = calculate_speedup(shodh_mean, mem0_mean)

        # Determine status
        if speedup > 1.5:
            status = "✅ Faster"
        elif speedup > 0.8:
            status = "≈ Similar"
        else:
            status = "⚠️  Slower"

        comparisons.append({
            'operation': op,
            'shodh_ms': shodh_mean,
            'mem0_ms': mem0_mean,
            'speedup': speedup,
            'status': status
        })

        print(f"{op:<15} {shodh_mean:<15.2f} {mem0_mean:<15.2f} {speedup:<15.1f}x {status:<15}")

    # Overall summary
    print()
    print("=" * 80)
    print("📊 OVERALL SUMMARY")
    print("=" * 80)
    print()

    avg_speedup = sum(c['speedup'] for c in comparisons) / len(comparisons)
    faster_count = sum(1 for c in comparisons if c['speedup'] > 1.2)

    print(f"Average Speedup:  {avg_speedup:.1f}x")
    print(f"Operations Faster: {faster_count}/{len(comparisons)}")
    print()

    # Claims validation
    print("=" * 80)
    print("✅ CLAIMS VALIDATION")
    print("=" * 80)
    print()

    claims = [
        ("10x faster claim", avg_speedup >= 10, f"Actual: {avg_speedup:.1f}x"),
        ("Faster ADD", shodh['ADD']['mean_ms'] < mem0_data['ADD']['mean_ms'],
         f"Shodh: {shodh['ADD']['mean_ms']:.2f}ms vs mem0: {mem0_data['ADD']['mean_ms']:.2f}ms"),
        ("Faster SEARCH", shodh['SEARCH']['mean_ms'] < mem0_data['SEARCH']['mean_ms'],
         f"Shodh: {shodh['SEARCH']['mean_ms']:.2f}ms vs mem0: {mem0_data['SEARCH']['mean_ms']:.2f}ms"),
        ("Sub-millisecond ADD", shodh['ADD']['mean_ms'] < 1.0,
         f"Actual: {shodh['ADD']['mean_ms']:.2f}ms"),
        ("Sub-millisecond SEARCH (working)", shodh['SEARCH']['median_ms'] < 1.0,
         f"Actual median: {shodh['SEARCH']['median_ms']:.2f}ms"),
    ]

    for claim, validated, detail in claims:
        status = "✅" if validated else "⚠️ "
        print(f"{status} {claim:<30} - {detail}")

    # Save comparison
    comparison_data = {
        "timestamp": datetime.now().isoformat(),
        "shodh_timestamp": shodh_results['timestamp'],
        "mem0_timestamp": mem0_results['timestamp'],
        "comparisons": comparisons,
        "summary": {
            "average_speedup": round(avg_speedup, 2),
            "operations_faster": faster_count,
            "total_operations": len(comparisons)
        },
        "claims_validation": [
            {"claim": c[0], "validated": c[1], "detail": c[2]}
            for c in claims
        ]
    }

    output_file = f"benchmarks/results/comparison_{int(datetime.now().timestamp())}.json"
    with open(output_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)

    print()
    print(f"✓ Comparison saved to {output_file}")

    # Generate markdown report
    generate_markdown_report(comparison_data)

def generate_markdown_report(comparison_data: Dict):
    """Generate markdown benchmark report"""

    md = []
    md.append("# Benchmark Results: Shodh-Memory vs mem0")
    md.append("")
    md.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    md.append("")
    md.append("## Performance Comparison")
    md.append("")
    md.append("| Operation | Shodh-Memory (ms) | mem0 (ms) | Speedup | Status |")
    md.append("|-----------|-------------------|-----------|---------|--------|")

    for comp in comparison_data['comparisons']:
        md.append(f"| {comp['operation']} | {comp['shodh_ms']:.2f} | {comp['mem0_ms']:.2f} | {comp['speedup']:.1f}x | {comp['status']} |")

    md.append("")
    md.append("## Summary")
    md.append("")
    summary = comparison_data['summary']
    md.append(f"- **Average Speedup:** {summary['average_speedup']:.1f}x")
    md.append(f"- **Operations Faster:** {summary['operations_faster']}/{summary['total_operations']}")
    md.append("")

    md.append("## Claims Validation")
    md.append("")
    for claim in comparison_data['claims_validation']:
        status = "✅" if claim['validated'] else "⚠️"
        md.append(f"- {status} **{claim['claim']}**: {claim['detail']}")

    md.append("")
    md.append("## Test Environment")
    md.append("")
    md.append("- **Iterations:** 100 per operation")
    md.append("- **Hardware:** [To be specified]")
    md.append("- **Shodh-Memory Version:** [To be specified]")
    md.append("- **mem0 Version:** [To be specified]")

    output_file = "benchmarks/results/BENCHMARK_REPORT.md"
    with open(output_file, 'w') as f:
        f.write('\n'.join(md))

    print(f"✓ Markdown report saved to {output_file}")

def main():
    print("Loading benchmark results...")

    shodh_results = load_latest_results("shodh")
    mem0_results = load_latest_results("mem0")

    if not shodh_results:
        print("❌ No Shodh-Memory results found. Run benchmark_shodh.py first.")
        sys.exit(1)

    if not mem0_results:
        print("⚠️  No mem0 results found. Comparison not possible.")
        print("To benchmark mem0: pip install mem0ai && python benchmarks/benchmark_mem0.py")
        sys.exit(1)

    generate_comparison_report(shodh_results, mem0_results)

if __name__ == "__main__":
    main()
