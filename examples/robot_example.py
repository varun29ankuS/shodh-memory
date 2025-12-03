"""
Example: Warehouse robot using shodh-memory for local AI memory

This demonstrates how a warehouse robot can use shodh-memory to remember
obstacles, navigation patterns, and task history completely offline.
"""

import shodh_memory

# Initialize memory system (fully offline, no network required)
memory = shodh_memory.MemorySystem(storage_path="./robot_memory")

print("🤖 Warehouse Robot Memory System\n")

# Record obstacle detection
print("Recording obstacle detection...")
obs_id = memory.record(
    content="Detected obstacle at grid coordinates (10, 20) in warehouse zone A",
    experience_type="observation",
    entities=["obstacle_147", "zone_a", "grid_10_20"],
    metadata={
        "sensor": "lidar",
        "confidence": "0.95",
        "timestamp": "2025-01-15T10:30:00Z"
    }
)
print(f"✓ Recorded observation: {obs_id}\n")

# Record navigation action
print("Recording navigation action...")
action_id = memory.record(
    content="Rerouted around obstacle using path planning algorithm A*",
    experience_type="action",
    entities=["obstacle_147", "path_alt_5"],
    metadata={
        "algorithm": "a_star",
        "cost": "1.2",
        "success": "true"
    }
)
print(f"✓ Recorded action: {action_id}\n")

# Record task completion
print("Recording task completion...")
outcome_id = memory.record(
    content="Successfully delivered package to station B3",
    experience_type="outcome",
    entities=["station_b3", "package_12345"],
    metadata={
        "task_id": "task_789",
        "delivery_time": "45.2s"
    }
)
print(f"✓ Recorded outcome: {outcome_id}\n")

# Query relevant memories (uses cached embeddings for speed)
print("Querying: 'obstacles in zone A'")
results = memory.retrieve(
    query="obstacles in zone A",
    max_results=5,
    mode="hybrid"  # Combines semantic + temporal + graph
)

print(f"\nFound {len(results)} relevant memories:")
for i, mem in enumerate(results, 1):
    print(f"\n{i}. [{mem['experience_type']}] (importance: {mem['importance']:.2f})")
    print(f"   {mem['content']}")
    print(f"   Entities: {', '.join(mem['entities'])}")
    print(f"   Accessed: {mem['access_count']} times")

# Get system statistics
print("\n" + "="*60)
stats = memory.get_stats()
print("Memory System Statistics:")
print(f"  Working memory: {stats['working_count']} items")
print(f"  Session memory: {stats['session_count']} items")
print(f"  Total records: {stats['total_records']}")
print(f"  Total retrievals: {stats['total_retrievals']}")
print(f"  Cache hit rate: {stats['cache_hits']}/{stats['cache_hits'] + stats['cache_misses']}")

# Flush to disk before shutdown
print("\nFlushing to disk...")
memory.flush()
print("✓ All data persisted\n")

print("🎯 Performance characteristics:")
print("  • Retrieval latency: 10-30ms (with cache: <5ms)")
print("  • 100% offline operation")
print("  • Zero network dependency")
print("  • Privacy-first (data never leaves device)")
