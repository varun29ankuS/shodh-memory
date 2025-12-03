#!/bin/bash
# Automated benchmark runner for Shodh-Memory

set -e

echo "========================================"
echo "🚀 Shodh-Memory Benchmark Runner"
echo "========================================"
echo ""

# Check if server is running
if ! curl -s http://localhost:3030/health > /dev/null 2>&1; then
    echo "❌ Server not running at localhost:3030"
    echo "Please start server with: cargo run --release"
    exit 1
fi

echo "✓ Server is ready"
echo ""

# Create results directory
mkdir -p benchmarks/results

# Run Shodh-Memory benchmarks
echo "📊 Running Shodh-Memory benchmarks..."
python benchmarks/benchmark_shodh.py

# Check if mem0 is available
if python -c "import mem0" 2>/dev/null; then
    echo ""
    echo "📊 Running mem0 benchmarks for comparison..."
    python benchmarks/benchmark_mem0.py

    echo ""
    echo "🔄 Generating comparison report..."
    python benchmarks/compare.py
else
    echo ""
    echo "⚠️  mem0 not installed - skipping comparison"
    echo "To compare: pip install mem0ai && ./benchmarks/run_benchmarks.sh"
fi

echo ""
echo "========================================"
echo "✓ Benchmarks complete!"
echo "========================================"
echo ""
echo "Results saved to: benchmarks/results/"
echo ""
ls -lt benchmarks/results/ | head -5
