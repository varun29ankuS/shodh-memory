#!/usr/bin/env python3
"""Basic usage example for Shodh-Memory"""

from shodh_memory import Memory

def main():
    # Create memory for user
    memory = Memory(user_id="demo_user")

    # Add different types of memories
    print("📝 Adding memories...")

    memory.add(
        "Learned about transformer architecture",
        experience_type="learning",
        entities=["transformers", "attention", "NLP"]
    )

    memory.add(
        "Fixed bug in authentication flow",
        experience_type="task",
        metadata={"priority": "high", "project": "auth"}
    )

    memory.add(
        "TypeError in vector.py line 42",
        experience_type="error",
        entities=["TypeError", "vector.py"]
    )

    # Search memories
    print("\n🔍 Searching for 'transformers'...")
    results = memory.search("transformers", max_results=5)

    for result in results:
        print(f"\n[Importance: {result['importance']:.2f}]")
        print(f"Content: {result['experience']['content']}")
        print(f"Type: {result['experience']['experience_type']}")

    # Get statistics
    print("\n📊 Memory Statistics:")
    stats = memory.stats()
    print(f"Total memories: {stats.total_memories}")
    print(f"Working memory: {stats.working_memory_count}")
    print(f"Session memory: {stats.session_memory_count}")
    print(f"Long-term memory: {stats.longterm_memory_count}")

if __name__ == "__main__":
    main()
