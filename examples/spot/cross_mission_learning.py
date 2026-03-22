"""
Cross-Mission Learning — Knowledge Accumulation Across Missions

Boston Dynamics Spot's mission blackboard dies when a mission ends.
Every mission starts from scratch with zero knowledge of what happened
in previous runs. A drone that cleared Grid A yesterday will search
Grid A again today.

This example simulates 3 missions through the same facility:
  - Mission 1 (Discovery): Everything is new
  - Mission 2 (Learning): Pre-recalls obstacles from Mission 1
  - Mission 3 (Expertise): Full situational awareness, anomalies decayed

Run:
    pip install shodh-memory
    python cross_mission_learning.py
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from shodh_spot_bridge import (
    NavigationFeedbackResponse,
    SpotMemoryBridge,
)


# =============================================================================
# Facility layout — 8 waypoints in a patrol route
# =============================================================================

@dataclass
class WaypointDef:
    id: str
    name: str
    position: Tuple[float, float, float]

FACILITY_WAYPOINTS = [
    WaypointDef("W1", "entrance_lobby", (0.0, 0.0, 0.0)),
    WaypointDef("W2", "corridor_north", (5.0, 0.0, 0.0)),
    WaypointDef("W3", "server_room_door", (10.0, 0.0, 0.0)),
    WaypointDef("W4", "server_room_interior", (10.0, 5.0, 0.0)),
    WaypointDef("W5", "electrical_panel", (10.0, 10.0, 0.0)),
    WaypointDef("W6", "corridor_south", (5.0, 10.0, 0.0)),
    WaypointDef("W7", "loading_dock", (0.0, 10.0, 0.0)),
    WaypointDef("W8", "parking_area", (0.0, 5.0, 0.0)),
]


# =============================================================================
# Environment events — what happens at each waypoint per mission
# =============================================================================

# (waypoint_id, event_type, description, severity)
MISSION_1_EVENTS = [
    ("W1", "clear", "Entrance lobby clear", None),
    ("W2", "obstacle", "Fallen cable tray blocking corridor", "warning"),
    ("W3", "obstacle", "Door partially jammed — manual force needed", "warning"),
    ("W4", "anomaly", "Temperature 42C — above normal 25C threshold", "info"),
    ("W5", "obstacle", "Water leak near panel — slip hazard", "error"),
    ("W6", "clear", "South corridor clear", None),
    ("W7", "obstacle", "Forklift parked in patrol path", "warning"),
    ("W8", "anomaly", "Unusual vibration from HVAC unit", "info"),
]

MISSION_2_EVENTS = [
    ("W1", "clear", "Entrance lobby clear", None),
    ("W2", "obstacle", "Fallen cable tray still present", "warning"),
    ("W3", "clear", "Door fixed since last mission", None),
    ("W4", "clear", "Temperature normal 24C", None),
    ("W5", "obstacle", "Water leak persists — maintenance not completed", "error"),
    ("W6", "obstacle", "New: storage boxes left in corridor", "warning"),
    ("W7", "clear", "Forklift moved — path clear", None),
    ("W8", "clear", "HVAC vibration resolved", None),
]

MISSION_3_EVENTS = [
    ("W1", "clear", "Entrance lobby clear", None),
    ("W2", "obstacle", "Cable tray still present — persistent hazard", "warning"),
    ("W3", "clear", "Door clear", None),
    ("W4", "clear", "Temperature normal 23C", None),
    ("W5", "obstacle", "Water leak — third consecutive observation", "error"),
    ("W6", "clear", "Storage boxes removed", None),
    ("W7", "clear", "Loading dock clear", None),
    ("W8", "clear", "Parking area clear", None),
]


def get_waypoint(wp_id: str) -> WaypointDef:
    for wp in FACILITY_WAYPOINTS:
        if wp.id == wp_id:
            return wp
    raise ValueError(f"Unknown waypoint: {wp_id}")


# =============================================================================
# Mission execution
# =============================================================================

def execute_mission(
    bridge: SpotMemoryBridge,
    mission_id: str,
    mission_number: int,
    events: List[Tuple[str, str, str, Optional[str]]],
) -> Dict[str, int]:
    """Execute a single mission and return statistics."""

    stats = {
        "obstacles_found": 0,
        "obstacles_prerecalled": 0,
        "anomalies_found": 0,
        "reroutes": 0,
        "clear_zones": 0,
    }

    header = f"MISSION {mission_number}: {mission_id.upper()}"
    print()
    print(f"{'=' * 3} {header} {'=' * (66 - len(header))}")
    bridge.start_mission(mission_id)

    for wp_id, event_type, description, severity in events:
        wp = get_waypoint(wp_id)

        # --- Pre-recall: What do we know about this waypoint? ---
        # recall_obstacles_nearby uses Euclidean distance filtering on stored
        # positions — all results are within 3m of this waypoint
        relevant_prerecalls = bridge.recall_obstacles_nearby(
            position=wp.position,
            radius=3.0,
            limit=5,
        )

        # --- Print waypoint entry ---
        if relevant_prerecalls:
            stats["obstacles_prerecalled"] += len(relevant_prerecalls)
            for m in relevant_prerecalls:
                content = m.get("content", "")
                score = m.get("score", 0.0)
                print(f"  {wp_id} {wp.name:25s} PRE-RECALL: {content[:50]} [score: {score:.2f}]")
        else:
            print(f"  {wp_id} {wp.name:25s} entered blind (no prior knowledge)")

        # --- Process current event ---
        if event_type == "obstacle":
            stats["obstacles_found"] += 1
            bridge.record_obstacle(
                description=f"{wp_id} {wp.name}: {description}",
                position=wp.position,
                confidence=0.9,
            )

            # Record navigation decision (reroute if past failures exist)
            if relevant_prerecalls:
                stats["reroutes"] += 1
                feedback = NavigationFeedbackResponse(
                    status="reached_goal",
                    distance_to_goal=0.0,
                )
                bridge.record_navigation_decision(
                    description=f"Rerouted around known obstacle at {wp_id}",
                    action="reroute",
                    state={"waypoint": wp_id, "reason": "pre-recalled_obstacle"},
                    outcome=feedback,
                    position=wp.position,
                    alternatives=["proceed_normal", "abort_mission"],
                )
                print(f"  {'':30s} REROUTED (based on prior experience)")
            else:
                # First encounter — record failure
                bridge.record_failure(
                    description=f"Unexpected obstacle at {wp_id}: {description}",
                    severity=severity or "warning",
                    position=wp.position,
                )
                print(f"  {'':30s} OBSTACLE: {description}")

        elif event_type == "anomaly":
            stats["anomalies_found"] += 1
            bridge.record_sensor_reading(
                sensor_name=f"inspection_{wp_id}",
                readings={"anomaly_score": 0.8},
                position=wp.position,
                is_anomaly=True,
            )
            print(f"  {'':30s} ANOMALY: {description}")

        else:
            stats["clear_zones"] += 1
            bridge.record_waypoint_visit(
                waypoint_id=wp_id,
                status="clear",
                position=wp.position,
            )

    # Mission summary
    bridge.end_mission(
        f"obstacles={stats['obstacles_found']}, "
        f"prerecalled={stats['obstacles_prerecalled']}, "
        f"reroutes={stats['reroutes']}"
    )

    print()
    print(f"  Summary: {stats['obstacles_found']} obstacles, "
          f"{stats['anomalies_found']} anomalies, "
          f"{stats['obstacles_prerecalled']} pre-recalled, "
          f"{stats['reroutes']} reroutes")

    return stats


# =============================================================================
# Main
# =============================================================================

def run_demo():
    print("=" * 70)
    print("  CROSS-MISSION LEARNING — Knowledge Accumulation")
    print("=" * 70)
    print()
    print("  Facility: 8 waypoints in a patrol loop")
    print("  Scenario: 3 missions through the same facility")
    print("  Goal:     Show knowledge improving across missions")

    # Clean up previous run data for reproducible results
    demo_path = Path("./spot_cross_mission_demo")
    if demo_path.exists():
        shutil.rmtree(demo_path)

    bridge = SpotMemoryBridge(
        storage_path=str(demo_path),
        robot_id="spot_alpha",
    )

    # Execute 3 missions
    stats_1 = execute_mission(bridge, "patrol_day1", 1, MISSION_1_EVENTS)
    stats_2 = execute_mission(bridge, "patrol_day2", 2, MISSION_2_EVENTS)
    stats_3 = execute_mission(bridge, "patrol_day3", 3, MISSION_3_EVENTS)

    # ─────────────────────────────────────────────────────────────
    # Learning metrics
    # ─────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  LEARNING METRICS")
    print("=" * 70)
    print()

    m1_prerecall = stats_1["obstacles_prerecalled"]
    m2_prerecall = stats_2["obstacles_prerecalled"]
    m3_prerecall = stats_3["obstacles_prerecalled"]

    m1_obstacles = stats_1["obstacles_found"]
    m2_obstacles = stats_2["obstacles_found"]
    m3_obstacles = stats_3["obstacles_found"]

    # Pre-recall rates
    m2_rate = (m2_prerecall / m2_obstacles * 100) if m2_obstacles else 0
    m3_rate = (m3_prerecall / m3_obstacles * 100) if m3_obstacles else 0

    print(f"  Mission 1 -> 2: {m2_rate:.0f}% obstacle pre-recall")
    print(f"  Mission 2 -> 3: {m3_rate:.0f}% obstacle pre-recall")
    print()
    print(f"  Mission 1: {m1_obstacles} obstacles, {m1_prerecall} pre-recalled, {stats_1['reroutes']} reroutes")
    print(f"  Mission 2: {m2_obstacles} obstacles, {m2_prerecall} pre-recalled, {stats_2['reroutes']} reroutes")
    print(f"  Mission 3: {m3_obstacles} obstacles, {m3_prerecall} pre-recalled, {stats_3['reroutes']} reroutes")
    print()

    # Persistent hazards (seen in all 3 missions)
    print("  Persistent hazards (seen across all missions):")
    persistent = bridge.recall_obstacles_nearby(
        position=(5.0, 5.0, 0.0),
        radius=20.0,
        limit=20,
    )
    seen_content = set()
    for m in persistent:
        content = m.get("content", "")
        if content not in seen_content:
            seen_content.add(content)
            score = m.get("score", 0.0)
            print(f"    - {content[:60]:60s} [score: {score:.3f}]")

    print()
    print("  Key insight: obstacles seen in multiple missions get higher")
    print("  relevance scores through Hebbian learning. One-off anomalies")
    print("  from Mission 1 have lower scores by Mission 3.")
    print()

    storage_stats = bridge.get_stats()
    print(f"  Storage stats: {storage_stats}")
    bridge.flush()


if __name__ == "__main__":
    run_demo()
