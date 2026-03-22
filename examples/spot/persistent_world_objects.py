"""
Persistent World Objects — Solving Spot's 15-Second TTL

Boston Dynamics Spot's WorldObjectService has a fundamental limitation:
detected objects expire after ~15 seconds. A fiducial marker seen at
timestamp T is gone from ListWorldObjects by T+15s. This means:

  - Spot can't remember objects it saw 30 seconds ago
  - Returning to an area yields zero prior knowledge
  - Object permanence doesn't exist across the SDK

This example demonstrates how shodh-memory gives Spot permanent object
memory. Objects persist across missions, reboots, and power cycles.

Run:
    pip install shodh-memory
    python persistent_world_objects.py
"""

import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from shodh_spot_bridge import (
    SE3Pose,
    SpotMemoryBridge,
    Vec3,
    WorldObject,
)


# =============================================================================
# Simulated Spot world object state (mimics the 15-sec TTL)
# =============================================================================

class SimulatedWorldObjectService:
    """Simulates Spot's WorldObjectService with its 15-second TTL.

    Objects are added with a timestamp. After 15 seconds, they disappear
    from list_world_objects() — exactly like the real SDK.
    """

    TTL_SECONDS = 15.0

    def __init__(self):
        self._objects: Dict[int, Tuple[WorldObject, float]] = {}
        self._next_id = 1

    def add_object(self, obj: WorldObject) -> int:
        """Add a detected object (starts the 15-second TTL clock)."""
        obj.id = self._next_id
        obj.acquisition_time = time.time()
        self._objects[obj.id] = (obj, time.time())
        self._next_id += 1
        return obj.id

    def list_world_objects(self, current_time: Optional[float] = None) -> List[WorldObject]:
        """Return only objects within the TTL window (Spot SDK behavior)."""
        now = current_time or time.time()
        active = []
        for obj, created_at in self._objects.values():
            if now - created_at <= self.TTL_SECONDS:
                active.append(obj)
        return active

    def advance_time(self, seconds: float) -> None:
        """Simulate time passing (for demo purposes)."""
        # Shift all creation times back to simulate elapsed time
        shifted = {}
        for obj_id, (obj, created_at) in self._objects.items():
            shifted[obj_id] = (obj, created_at - seconds)
        self._objects = shifted


# =============================================================================
# Demo scenario
# =============================================================================

def create_test_objects() -> List[WorldObject]:
    """Create 5 objects Spot might detect during a facility inspection."""
    return [
        WorldObject(
            name="fiducial_tag_301",
            object_type=WorldObject.WORLD_OBJECT_APRILTAG,
            transforms_snapshot={
                "body": SE3Pose(position=Vec3(x=3.2, y=1.5, z=0.0)),
            },
            additional_properties={"tag_id": "301", "tag_family": "tag36h11"},
        ),
        WorldObject(
            name="obstacle_pipe_junction",
            object_type=WorldObject.WORLD_OBJECT_UNKNOWN,
            transforms_snapshot={
                "odom": SE3Pose(position=Vec3(x=7.8, y=4.2, z=0.5)),
            },
            additional_properties={"height_cm": "120", "material": "steel"},
        ),
        WorldObject(
            name="tracked_person_A",
            object_type=WorldObject.WORLD_OBJECT_TRACKED_ENTITY,
            transforms_snapshot={
                "body": SE3Pose(position=Vec3(x=5.0, y=3.0, z=0.0)),
            },
            additional_properties={"confidence": "0.94"},
        ),
        WorldObject(
            name="dock_station_main",
            object_type=WorldObject.WORLD_OBJECT_DOCK,
            transforms_snapshot={
                "odom": SE3Pose(position=Vec3(x=0.5, y=0.5, z=0.0)),
            },
            additional_properties={"dock_id": "520"},
        ),
        WorldObject(
            name="staircase_level2",
            object_type=WorldObject.WORLD_OBJECT_STAIRCASE,
            transforms_snapshot={
                "body": SE3Pose(position=Vec3(x=12.0, y=8.0, z=0.0)),
            },
            additional_properties={"floors": "2", "direction": "ascending"},
        ),
    ]


def run_demo():
    print("=" * 70)
    print("  PERSISTENT WORLD OBJECTS — Solving Spot's 15-Second TTL")
    print("=" * 70)
    print()

    # Clean up previous run data for reproducible results
    demo_path = Path("./spot_world_objects_demo")
    if demo_path.exists():
        shutil.rmtree(demo_path)

    # Initialize both systems
    bridge = SpotMemoryBridge(storage_path=str(demo_path), robot_id="spot_alpha")
    spot_service = SimulatedWorldObjectService()
    bridge.start_mission("inspection_001")

    objects = create_test_objects()

    # ─────────────────────────────────────────────────────────────
    # Phase 1: Spot detects 5 objects and we persist them
    # ─────────────────────────────────────────────────────────────
    print("[Phase 1] Spot detects 5 objects during patrol")
    print("-" * 50)

    for obj in objects:
        # Spot's native service registers the object (starts 15-sec TTL)
        spot_service.add_object(obj)

        # We persist it in shodh-memory (permanent)
        memory_id = bridge.persist_world_object_from_proto(obj)
        print(f"  Detected: {obj.name:30s} → persisted as {memory_id[:8]}...")

    # Verify both systems agree right now
    spot_objects = spot_service.list_world_objects()
    shodh_objects = bridge.recall_world_objects(limit=10)

    print()
    print(f"  Spot native:  {len(spot_objects)} objects (within 15-sec TTL)")
    print(f"  Shodh-memory: {len(shodh_objects)} objects (permanent)")

    # ─────────────────────────────────────────────────────────────
    # Phase 2: Time passes — Spot's objects expire
    # ─────────────────────────────────────────────────────────────
    print()
    print("[Phase 2] 30 seconds pass — Spot's TTL expires")
    print("-" * 50)

    spot_service.advance_time(30.0)  # 30 seconds later

    spot_objects_after = spot_service.list_world_objects()
    shodh_objects_after = bridge.recall_world_objects(limit=10)

    print(f"  Spot native:  {len(spot_objects_after)} objects (all expired)")
    print(f"  Shodh-memory: {len(shodh_objects_after)} objects (all retained)")

    # ─────────────────────────────────────────────────────────────
    # Phase 3: Robot returns to area — queries for nearby objects
    # ─────────────────────────────────────────────────────────────
    print()
    print("[Phase 3] Robot returns to same area — queries past objects")
    print("-" * 50)

    current_position = (4.0, 2.0, 0.0)
    recalled = bridge.recall_world_objects(
        position=current_position,
        radius=10.0,
        limit=10,
    )

    print(f"  Current position: {current_position}")
    print(f"  Recalled {len(recalled)} objects from memory:")
    for mem in recalled:
        content = mem.get("content", "")
        score = mem.get("score", 0.0)
        print(f"    - {content:50s} [relevance: {score:.3f}]")

    # ─────────────────────────────────────────────────────────────
    # Phase 4: Filtered recall — only specific object types
    # ─────────────────────────────────────────────────────────────
    print()
    print("[Phase 4] Filtered recall — fiducials only")
    print("-" * 50)

    fiducials = bridge.recall_world_objects(
        object_type="WORLD_OBJECT_APRILTAG",
        limit=5,
    )
    print(f"  AprilTag fiducials recalled: {len(fiducials)}")
    for mem in fiducials:
        print(f"    - {mem.get('content', '')}")

    # ─────────────────────────────────────────────────────────────
    # Phase 5: Cross-mission — end mission, start new one, recall
    # ─────────────────────────────────────────────────────────────
    print()
    print("[Phase 5] New mission — objects persist across missions")
    print("-" * 50)

    bridge.end_mission("5 objects detected, all persisted")
    bridge.start_mission("inspection_002")

    cross_mission = bridge.recall_world_objects(limit=10)
    print(f"  Mission 002 can recall {len(cross_mission)} objects from Mission 001")
    for mem in cross_mission:
        content = mem.get("content", "")
        print(f"    - {content}")

    # ─────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print("  Spot SDK (native):  Objects lost after 15 seconds")
    print("  Shodh-memory:       Objects persist permanently")
    print()
    print("  After 30 seconds:")
    print(f"    Spot native:  {len(spot_objects_after)} objects")
    print(f"    Shodh-memory: {len(shodh_objects_after)} objects")
    print()
    print("  Cross-mission:")
    print(f"    Mission 002 recalled {len(cross_mission)} objects from Mission 001")
    print()
    stats = bridge.get_stats()
    print(f"  Storage stats: {stats}")
    print()

    bridge.end_mission()
    bridge.flush()


if __name__ == "__main__":
    run_demo()
