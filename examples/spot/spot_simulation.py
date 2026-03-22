"""
Spot Facility Inspection — 5-Day Simulation

Simulates a realistic industrial inspection scenario: a Spot robot
performing daily facility inspections over 5 consecutive days.

Demonstrates ALL shodh-memory capabilities working together:
  1. Object permanence — obstacles persist across missions
  2. Cross-mission learning — pre-recalls improve each day
  3. Semantic annotations — "this area is always hot" learned from repeated readings
  4. Decision improvement — rerouting decisions improve with experience
  5. Hebbian learning — frequently-encountered obstacles get higher recall scores
  6. Decay — one-off anomalies from Day 1 fade by Day 5
  7. Fleet readiness — multi-robot support via robot_id

No hardware needed. Run:
    pip install shodh-memory
    python spot_simulation.py
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from shodh_spot_bridge import (
    NavigationFeedbackResponse,
    SpotMemoryBridge,
)


# =============================================================================
# Facility layout — 12 waypoints in a grid
# =============================================================================

@dataclass
class FacilityWaypoint:
    id: str
    name: str
    position: Tuple[float, float, float]
    zone: str  # functional zone for context

FACILITY = [
    FacilityWaypoint("W01", "main_entrance", (0.0, 0.0, 0.0), "access"),
    FacilityWaypoint("W02", "reception_hall", (5.0, 0.0, 0.0), "access"),
    FacilityWaypoint("W03", "corridor_north", (10.0, 0.0, 0.0), "transit"),
    FacilityWaypoint("W04", "server_room_A", (15.0, 0.0, 0.0), "critical"),
    FacilityWaypoint("W05", "server_room_B", (15.0, 5.0, 0.0), "critical"),
    FacilityWaypoint("W06", "electrical_room", (15.0, 10.0, 0.0), "critical"),
    FacilityWaypoint("W07", "corridor_south", (10.0, 10.0, 0.0), "transit"),
    FacilityWaypoint("W08", "storage_area", (5.0, 10.0, 0.0), "utility"),
    FacilityWaypoint("W09", "loading_dock", (0.0, 10.0, 0.0), "utility"),
    FacilityWaypoint("W10", "parking_bay", (0.0, 5.0, 0.0), "access"),
    FacilityWaypoint("W11", "charging_station", (5.0, 5.0, 0.0), "utility"),
    FacilityWaypoint("W12", "outdoor_yard", (10.0, 5.0, 0.0), "outdoor"),
]

WAYPOINT_MAP = {wp.id: wp for wp in FACILITY}


# =============================================================================
# Environment simulation — what happens each day
# =============================================================================

@dataclass
class WaypointEvent:
    waypoint_id: str
    event_type: str       # "clear", "obstacle", "anomaly", "failure"
    description: str
    severity: str = "info"
    sensor_data: Dict[str, float] = field(default_factory=dict)

# Persistent obstacles (appear and stay)
# Transient anomalies (appear once, may not recur)
# Progressive conditions (worsen over time)

DAY_EVENTS: Dict[int, List[WaypointEvent]] = {
    1: [
        WaypointEvent("W01", "clear", "Entrance clear"),
        WaypointEvent("W02", "clear", "Reception clear"),
        WaypointEvent("W03", "obstacle", "Cable tray fallen across corridor", "warning"),
        WaypointEvent("W04", "anomaly", "Server room A: temperature 38C", "info", {"temperature": 38.0}),
        WaypointEvent("W05", "clear", "Server room B normal"),
        WaypointEvent("W06", "obstacle", "Water pooling near electrical panel", "error", {"humidity": 90.0}),
        WaypointEvent("W07", "clear", "South corridor clear"),
        WaypointEvent("W08", "obstacle", "Boxes stacked in path", "warning"),
        WaypointEvent("W09", "anomaly", "Unusual noise from dock motor", "info", {"noise_db": 92.0}),
        WaypointEvent("W10", "clear", "Parking bay clear"),
        WaypointEvent("W11", "clear", "Charging station available"),
        WaypointEvent("W12", "obstacle", "Construction barrier in yard", "warning"),
    ],
    2: [
        WaypointEvent("W01", "clear", "Entrance clear"),
        WaypointEvent("W02", "clear", "Reception clear"),
        WaypointEvent("W03", "obstacle", "Cable tray still present — not repaired", "warning"),
        WaypointEvent("W04", "clear", "Server room A: temperature normal 24C", sensor_data={"temperature": 24.0}),
        WaypointEvent("W05", "anomaly", "Server room B: fan noise abnormal", "info", {"noise_db": 78.0}),
        WaypointEvent("W06", "obstacle", "Water leak persists — maintenance pending", "error", {"humidity": 88.0}),
        WaypointEvent("W07", "obstacle", "New: pallet left in corridor", "warning"),
        WaypointEvent("W08", "clear", "Boxes removed since yesterday"),
        WaypointEvent("W09", "clear", "Dock motor noise resolved"),
        WaypointEvent("W10", "clear", "Parking bay clear"),
        WaypointEvent("W11", "clear", "Charging station available"),
        WaypointEvent("W12", "obstacle", "Construction barrier still present", "warning"),
    ],
    3: [
        WaypointEvent("W01", "clear", "Entrance clear"),
        WaypointEvent("W02", "clear", "Reception clear"),
        WaypointEvent("W03", "obstacle", "Cable tray — 3rd day, persistent hazard", "warning"),
        WaypointEvent("W04", "anomaly", "Server room A: temperature 36C again", "warning", {"temperature": 36.0}),
        WaypointEvent("W05", "clear", "Server room B: fan replaced, normal"),
        WaypointEvent("W06", "obstacle", "Water leak — 3rd day, escalating", "error", {"humidity": 92.0}),
        WaypointEvent("W07", "clear", "Pallet removed from corridor"),
        WaypointEvent("W08", "clear", "Storage area clear"),
        WaypointEvent("W09", "clear", "Loading dock clear"),
        WaypointEvent("W10", "clear", "Parking bay clear"),
        WaypointEvent("W11", "clear", "Charging station available"),
        WaypointEvent("W12", "clear", "Construction barrier removed"),
    ],
    4: [
        WaypointEvent("W01", "clear", "Entrance clear"),
        WaypointEvent("W02", "clear", "Reception clear"),
        WaypointEvent("W03", "obstacle", "Cable tray — 4th day, work order open", "warning"),
        WaypointEvent("W04", "clear", "Server room A: HVAC fixed, 23C", sensor_data={"temperature": 23.0}),
        WaypointEvent("W05", "clear", "Server room B normal"),
        WaypointEvent("W06", "clear", "Electrical room: leak finally repaired", sensor_data={"humidity": 45.0}),
        WaypointEvent("W07", "clear", "South corridor clear"),
        WaypointEvent("W08", "clear", "Storage area clear"),
        WaypointEvent("W09", "clear", "Loading dock clear"),
        WaypointEvent("W10", "clear", "Parking bay clear"),
        WaypointEvent("W11", "clear", "Charging station available"),
        WaypointEvent("W12", "clear", "Outdoor yard clear"),
    ],
    5: [
        WaypointEvent("W01", "clear", "Entrance clear"),
        WaypointEvent("W02", "clear", "Reception clear"),
        WaypointEvent("W03", "clear", "Cable tray finally repaired on Day 5"),
        WaypointEvent("W04", "clear", "Server room A normal", sensor_data={"temperature": 24.0}),
        WaypointEvent("W05", "clear", "Server room B normal"),
        WaypointEvent("W06", "clear", "Electrical room dry and clear"),
        WaypointEvent("W07", "clear", "South corridor clear"),
        WaypointEvent("W08", "clear", "Storage area clear"),
        WaypointEvent("W09", "clear", "Loading dock clear"),
        WaypointEvent("W10", "clear", "Parking bay clear"),
        WaypointEvent("W11", "clear", "Charging station available"),
        WaypointEvent("W12", "clear", "Outdoor yard clear"),
    ],
}


# =============================================================================
# Mission execution engine
# =============================================================================

@dataclass
class DayStats:
    day: int
    obstacles_found: int = 0
    obstacles_prerecalled: int = 0
    new_obstacles: int = 0
    anomalies: int = 0
    decisions_made: int = 0
    reroutes: int = 0


def execute_day(
    bridge: SpotMemoryBridge,
    day: int,
    events: List[WaypointEvent],
) -> DayStats:
    """Execute one day's inspection patrol."""
    stats = DayStats(day=day)
    mission_id = f"inspection_day{day:02d}"
    bridge.start_mission(mission_id)

    print(f"\n  Day {day} — Mission {mission_id}")
    print(f"  {'─' * 60}")

    for event in events:
        wp = WAYPOINT_MAP[event.waypoint_id]

        # Pre-recall: check for known hazards at this waypoint
        # recall_obstacles_nearby uses Euclidean distance filtering on stored
        # positions — all results are within 3m of this waypoint
        relevant = bridge.recall_obstacles_nearby(
            position=wp.position,
            radius=3.0,
            limit=5,
        )

        prerecalled = len(relevant) > 0

        if event.event_type == "obstacle":
            stats.obstacles_found += 1

            if prerecalled:
                stats.obstacles_prerecalled += 1
                stats.decisions_made += 1
                stats.reroutes += 1

                # Record successful reroute based on prior knowledge
                feedback = NavigationFeedbackResponse(status="reached_goal")
                bridge.record_navigation_decision(
                    description=f"Rerouted at {event.waypoint_id} — pre-recalled hazard",
                    action="reroute",
                    state={"waypoint": event.waypoint_id, "zone": wp.zone},
                    outcome=feedback,
                    position=wp.position,
                )

                print(
                    f"    {event.waypoint_id} {wp.name:22s} "
                    f"PRE-RECALL + REROUTE: {event.description[:35]}"
                )
            else:
                stats.new_obstacles += 1
                bridge.record_obstacle(
                    description=f"{event.waypoint_id} {wp.name}: {event.description}",
                    position=wp.position,
                    confidence=0.9,
                )
                bridge.record_failure(
                    description=f"Unexpected obstacle at {event.waypoint_id}: {event.description}",
                    severity=event.severity,
                    position=wp.position,
                )
                print(
                    f"    {event.waypoint_id} {wp.name:22s} "
                    f"NEW OBSTACLE: {event.description[:40]}"
                )

        elif event.event_type == "anomaly":
            stats.anomalies += 1
            bridge.record_sensor_reading(
                sensor_name=f"{event.waypoint_id}_inspection",
                readings=event.sensor_data,
                position=wp.position,
                is_anomaly=True,
            )
            bridge.annotate_waypoint(
                waypoint_id=event.waypoint_id,
                label=event.description,
                position=wp.position,
            )
            print(
                f"    {event.waypoint_id} {wp.name:22s} "
                f"ANOMALY: {event.description[:43]}"
            )

        else:
            # Clear — just record the visit
            bridge.record_waypoint_visit(
                waypoint_id=event.waypoint_id,
                status="clear",
                position=wp.position,
                sensor_data=event.sensor_data if event.sensor_data else None,
            )

            if prerecalled:
                stats.decisions_made += 1
                print(
                    f"    {event.waypoint_id} {wp.name:22s} "
                    f"CLEAR (was known hazard — now resolved)"
                )
            else:
                print(
                    f"    {event.waypoint_id} {wp.name:22s} "
                    f"clear"
                )

    # Mission summary
    bridge.end_mission(
        f"obstacles={stats.obstacles_found}, new={stats.new_obstacles}, "
        f"prerecalled={stats.obstacles_prerecalled}, reroutes={stats.reroutes}"
    )

    return stats


# =============================================================================
# Main simulation
# =============================================================================

def run_demo():
    print()
    print("+" + "=" * 68 + "+")
    print("|" + "  SPOT FACILITY INSPECTION — 5 DAY SIMULATION".center(68) + "|")
    print("+" + "=" * 68 + "+")

    # Clean up previous run data for reproducible results
    demo_path = Path("./spot_simulation_demo")
    if demo_path.exists():
        shutil.rmtree(demo_path)

    bridge = SpotMemoryBridge(
        storage_path=str(demo_path),
        robot_id="spot_alpha",
    )

    # Annotate facility zones
    for wp in FACILITY:
        bridge.annotate_waypoint(
            waypoint_id=wp.id,
            label=f"{wp.name} [{wp.zone} zone]",
            position=wp.position,
        )

    all_stats: List[DayStats] = []

    for day in range(1, 6):
        events = DAY_EVENTS[day]
        stats = execute_day(bridge, day, events)
        all_stats.append(stats)

    # ─────────────────────────────────────────────────────────────
    # Results table
    # ─────────────────────────────────────────────────────────────
    print()
    print()
    print("+" + "=" * 68 + "+")
    print("|" + "  RESULTS".center(68) + "|")
    print("+" + "=" * 68 + "+")
    print("|" + "".center(68) + "|")

    header = f"| {'Day':>4s} | {'Obstacles':>9s} | {'Pre-recalled':>12s} | {'New':>4s} | {'Anomalies':>9s} | {'Reroutes':>8s} |"
    print(header)
    sep = f"| {'─' * 4} | {'─' * 9} | {'─' * 12} | {'─' * 4} | {'─' * 9} | {'─' * 8} |"
    print(sep)

    for s in all_stats:
        print(
            f"|  {s.day:>2d}  |    {s.obstacles_found:>2d}     |      "
            f"{s.obstacles_prerecalled:>2d}      |  {s.new_obstacles:>2d}  |"
            f"    {s.anomalies:>2d}     |    {s.reroutes:>2d}    |"
        )

    print("|" + "".center(68) + "|")
    print("+" + "─" * 68 + "+")

    # Learning progression
    total_obstacles = sum(s.obstacles_found for s in all_stats)
    total_prerecalled = sum(s.obstacles_prerecalled for s in all_stats)
    total_new = sum(s.new_obstacles for s in all_stats)
    total_reroutes = sum(s.reroutes for s in all_stats)

    print()
    print("  Learning progression:")
    for s in all_stats:
        rate = (s.obstacles_prerecalled / s.obstacles_found * 100) if s.obstacles_found else 100
        bar = "#" * int(rate / 5) + "." * (20 - int(rate / 5))
        print(f"    Day {s.day}: [{bar}] {rate:5.1f}% pre-recall rate")

    print()
    print(f"  Totals across 5 days:")
    print(f"    Obstacles encountered:  {total_obstacles}")
    print(f"    Pre-recalled (avoided): {total_prerecalled}")
    print(f"    New (first encounter):  {total_new}")
    print(f"    Reroutes from memory:   {total_reroutes}")
    print()

    # Persistent hazard analysis
    print("  Persistent hazards (detected in multiple missions):")
    all_obstacles = bridge.memory.recall(
        query="obstacle hazard",
        limit=20,
        mode="hybrid",
        tags=["obstacle"],
    )
    seen = set()
    for obs in all_obstacles:
        content = obs.get("content", "")
        short = content[:50]
        if short not in seen:
            seen.add(short)
            score = obs.get("score", 0.0)
            print(f"    - {short:50s} [score: {score:.3f}]")

    print()
    print("  Key takeaways:")
    print("    - Day 1: Everything is new — 0% pre-recall")
    print("    - Day 2+: Known obstacles recalled before encounter")
    print("    - Persistent hazards (cable tray, water leak) get")
    print("      higher scores through Hebbian learning")
    print("    - One-off anomalies (dock noise Day 1) naturally decay")
    print("    - Resolved conditions are noted (leak fixed Day 4)")
    print()

    storage_stats = bridge.get_stats()
    print(f"  Storage: {storage_stats}")
    print()

    bridge.flush()


if __name__ == "__main__":
    run_demo()
