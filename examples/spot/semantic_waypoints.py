"""
Semantic Waypoints — Learned Annotations on GraphNav Maps

Spot's GraphNav waypoints support only two annotation fields:
  - name: a human-readable label
  - client_metadata: arbitrary bytes (opaque to the SDK)

There's no built-in way to semantically annotate waypoints with learned
knowledge like "this area is usually congested at 9am" or "charging
station with 2-hour wait time."

This example adds a rich semantic layer on top of GraphNav waypoints
using shodh-memory. Annotations accumulate over time and are queryable
by natural language.

Run:
    pip install shodh-memory
    python semantic_waypoints.py
"""

import shutil
from pathlib import Path

from shodh_spot_bridge import SpotMemoryBridge


# =============================================================================
# Facility waypoints with positions
# =============================================================================

WAYPOINTS = {
    "W1": {"name": "main_entrance", "pos": (0.0, 0.0, 0.0)},
    "W2": {"name": "corridor_A", "pos": (5.0, 0.0, 0.0)},
    "W3": {"name": "charging_dock", "pos": (10.0, 0.0, 0.0)},
    "W4": {"name": "server_room", "pos": (10.0, 5.0, 0.0)},
    "W5": {"name": "hazmat_storage", "pos": (10.0, 10.0, 0.0)},
    "W6": {"name": "cafeteria", "pos": (5.0, 10.0, 0.0)},
    "W7": {"name": "loading_bay", "pos": (0.0, 10.0, 0.0)},
    "W8": {"name": "parking_lot", "pos": (0.0, 5.0, 0.0)},
}


# =============================================================================
# Simulated visit data (10 visits with environmental observations)
# =============================================================================

VISIT_LOG = [
    # (visit_num, waypoint_id, observation, sensor_data)
    (1, "W1", "Normal foot traffic, 3 people near entrance", {"people_count": 3, "temperature": 22.0}),
    (1, "W3", "Dock occupied by Spot Beta, waited 5 minutes", {"wait_time_min": 5, "dock_occupied": 1}),
    (1, "W4", "Server room temperature elevated: 38C", {"temperature": 38.0, "humidity": 45.0}),
    (1, "W5", "Chemical smell detected near storage", {"gas_sensor_ppm": 12.0, "temperature": 20.0}),
    (2, "W1", "Heavy foot traffic, 8 people, congested", {"people_count": 8, "temperature": 23.0}),
    (2, "W3", "Dock free, charged to 95%", {"wait_time_min": 0, "dock_occupied": 0}),
    (2, "W4", "Server room temperature normal: 24C", {"temperature": 24.0, "humidity": 40.0}),
    (2, "W7", "Forklift operating near loading bay — caution", {"forklift_detected": 1, "noise_db": 85.0}),
    (3, "W1", "Moderate traffic, 5 people", {"people_count": 5, "temperature": 21.0}),
    (3, "W3", "Dock occupied, waited 12 minutes", {"wait_time_min": 12, "dock_occupied": 1}),
    (3, "W5", "Elevated gas readings: 15ppm — hazard flag", {"gas_sensor_ppm": 15.0, "temperature": 19.0}),
    (3, "W6", "Cafeteria — wet floor sign present", {"wet_floor": 1, "people_count": 12}),
    (4, "W1", "Light traffic, 2 people", {"people_count": 2, "temperature": 22.0}),
    (4, "W4", "Server room temperature elevated again: 36C", {"temperature": 36.0, "humidity": 50.0}),
    (4, "W5", "Gas sensor back to normal: 3ppm", {"gas_sensor_ppm": 3.0, "temperature": 20.0}),
    (4, "W7", "Loading bay clear, no forklifts", {"forklift_detected": 0, "noise_db": 40.0}),
]


def run_demo():
    print("=" * 70)
    print("  SEMANTIC WAYPOINTS — Learned Annotations on GraphNav")
    print("=" * 70)
    print()

    # Clean up previous run data for reproducible results
    demo_path = Path("./spot_semantic_wp_demo")
    if demo_path.exists():
        shutil.rmtree(demo_path)

    bridge = SpotMemoryBridge(
        storage_path=str(demo_path),
        robot_id="spot_alpha",
    )

    # ─────────────────────────────────────────────────────────────
    # Phase 1: Add static semantic annotations
    # ─────────────────────────────────────────────────────────────
    print("[Phase 1] Adding static semantic annotations")
    print("-" * 50)

    static_annotations = [
        ("W3", "Charging station — dock ID 520, 110V"),
        ("W4", "Server room — restricted access, temperature monitoring required"),
        ("W5", "Hazardous materials storage — gas sensor required"),
        ("W6", "Cafeteria — high foot traffic 12:00-13:00"),
        ("W7", "Loading bay — forklift operations 08:00-17:00"),
    ]

    for wp_id, label in static_annotations:
        wp = WAYPOINTS[wp_id]
        bridge.annotate_waypoint(
            waypoint_id=wp_id,
            label=label,
            position=wp["pos"],
        )
        print(f"  {wp_id} ({wp['name']:20s}): {label[:50]}")

    # ─────────────────────────────────────────────────────────────
    # Phase 2: Record visit observations with sensor data
    # ─────────────────────────────────────────────────────────────
    print()
    print("[Phase 2] Recording 16 visit observations across 4 patrols")
    print("-" * 50)

    for visit_num, wp_id, observation, sensors in VISIT_LOG:
        wp = WAYPOINTS[wp_id]

        # Record the visit
        bridge.record_waypoint_visit(
            waypoint_id=wp_id,
            status="visited",
            position=wp["pos"],
            sensor_data=sensors,
        )

        # Record sensor readings
        bridge.record_sensor_reading(
            sensor_name=f"{wp_id}_inspection",
            readings=sensors,
            position=wp["pos"],
            is_anomaly=(
                sensors.get("temperature", 0) > 35
                or sensors.get("gas_sensor_ppm", 0) > 10
            ),
        )

        # Annotate with learned observation
        bridge.annotate_waypoint(
            waypoint_id=wp_id,
            label=observation,
            position=wp["pos"],
            metadata={"visit": str(visit_num), "source": "autonomous_observation"},
        )

        print(f"  Visit {visit_num}, {wp_id}: {observation[:55]}")

    # ─────────────────────────────────────────────────────────────
    # Phase 3: Semantic queries — natural language waypoint search
    # ─────────────────────────────────────────────────────────────
    print()
    print("[Phase 3] Semantic queries on waypoint knowledge")
    print("-" * 50)

    queries = [
        ("hazardous areas", "Where are the dangerous zones?"),
        ("charging station dock", "Where can I charge?"),
        ("congested high traffic", "Which areas are crowded?"),
        ("temperature elevated hot", "Where are thermal anomalies?"),
        ("forklift machinery", "Where is heavy equipment?"),
    ]

    for query, description in queries:
        print(f"\n  Query: \"{query}\"  ({description})")

        results = bridge.memory.recall(
            query=query,
            limit=3,
            mode="hybrid",
            tags=["waypoint"],
        )

        if results:
            for r in results:
                content = r.get("content", "")
                score = r.get("score", 0.0)
                print(f"    -> {content[:60]:60s} [{score:.3f}]")
        else:
            print("    -> No results")

    # ─────────────────────────────────────────────────────────────
    # Phase 4: Waypoint history — accumulated knowledge per location
    # ─────────────────────────────────────────────────────────────
    print()
    print()
    print("[Phase 4] Accumulated knowledge per waypoint")
    print("-" * 50)

    for wp_id in ["W3", "W4", "W5"]:
        wp = WAYPOINTS[wp_id]
        history = bridge.recall_waypoint_history(wp_id, limit=10)
        print(f"\n  {wp_id} ({wp['name']}) — {len(history)} memories:")
        for h in history[:5]:
            content = h.get("content", "")
            print(f"    - {content[:65]}")

    # ─────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────
    print()
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print("  Static annotations:     5 (manually labeled)")
    print(f"  Visit observations:     {len(VISIT_LOG)} (autonomously recorded)")
    print(f"  Total waypoint memories: {len(VISIT_LOG) + len(static_annotations)}+")
    print()
    print("  GraphNav SDK provides: waypoint name + opaque bytes")
    print("  Shodh-memory provides: semantic search, sensor history,")
    print("                         learned patterns, natural language queries")
    print()

    stats = bridge.get_stats()
    print(f"  Storage stats: {stats}")
    bridge.flush()


if __name__ == "__main__":
    run_demo()
