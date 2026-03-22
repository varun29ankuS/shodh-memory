"""
Fleet Memory — Multi-Robot Knowledge Sharing

Spot's Orbit (fleet management) aggregates data from multiple robots,
but each robot operates with its own isolated world view. Robot A
discovering an obstacle doesn't help Robot B avoid it.

This example shows two Spot robots sharing knowledge through a
common shodh-memory store. The key insight: for fleet-wide sharing,
robots use the SAME identity (or no robot_id filter) so all memories
are visible to all fleet members. Robot attribution is done via tags.

In production, both robots connect to the same shodh-memory HTTP
server (port 3030). Here we simulate the handoff by using the same
storage_path sequentially (RocksDB requires exclusive access).

    # On fleet server
    shodh server --port 3030

    # Each robot connects via HTTP client
    from shodh_memory.client import ShodhClient
    alpha = ShodhClient(base_url="http://fleet-server:3030", user_id="fleet")
    beta  = ShodhClient(base_url="http://fleet-server:3030", user_id="fleet")

Run:
    pip install shodh-memory
    python fleet_memory.py
"""

import shutil
from pathlib import Path
from typing import List, Tuple

from shodh_memory import Position
from shodh_spot_bridge import SpotMemoryBridge


# =============================================================================
# Fleet bridge — wraps SpotMemoryBridge without robot_id filtering
# =============================================================================

class FleetSpotBridge:
    """Fleet-aware wrapper that tags memories with robot name instead of
    using robot_id as a query filter.

    shodh-memory's robot_id is a hard filter on recall() — memories from
    robot_id="spot_alpha" are invisible to robot_id="spot_beta". For fleet
    sharing, we avoid setting robot_id and instead tag each memory with
    the originating robot's name.
    """

    def __init__(self, storage_path: str, robot_name: str):
        # Default robot_id="spot_01" shared by all fleet instances, so
        # recall() returns memories from any robot using this bridge
        self._bridge = SpotMemoryBridge(storage_path=storage_path)
        self.robot_name = robot_name

    def start_mission(self, mission_id: str) -> None:
        self._bridge.start_mission(mission_id)

    def end_mission(self, summary: str = "") -> None:
        self._bridge.end_mission(summary or None)

    def record_obstacle(
        self,
        description: str,
        position: Tuple[float, float, float],
        confidence: float = 0.9,
    ) -> str:
        """Record obstacle with robot attribution tag."""
        return self._bridge.memory.remember(
            f"[{self.robot_name}] obstacle: {description} "
            f"at position {position[0]:.1f} {position[1]:.1f} {position[2]:.1f}",
            memory_type="error",
            tags=["obstacle", "fleet", self.robot_name],
            entities=[self.robot_name] + description.split()[:1],
            sensor_data={"detection_confidence": confidence},
            position=Position(x=position[0], y=position[1], z=position[2]),
        )

    def recall_fleet_obstacles(self, limit: int = 20) -> List[dict]:
        """Recall all fleet-wide obstacles (no robot_id filter)."""
        return self._bridge.memory.recall(
            query="obstacle hazard danger",
            limit=limit,
            mode="hybrid",
            tags=["obstacle"],
        )

    def flush(self) -> None:
        self._bridge.flush()


def run_demo():
    print("=" * 70)
    print("  FLEET MEMORY — Multi-Robot Knowledge Sharing")
    print("=" * 70)
    print()

    # Clean up previous run data for reproducible results
    shared_path = "./spot_fleet_demo"
    demo_path = Path(shared_path)
    if demo_path.exists():
        shutil.rmtree(demo_path)

    # ─────────────────────────────────────────────────────────────
    # Phase 1: Robot Alpha patrols and discovers hazards
    # ─────────────────────────────────────────────────────────────
    print("[Phase 1] Robot Alpha — Morning patrol")
    print("-" * 50)

    alpha = FleetSpotBridge(storage_path=shared_path, robot_name="spot_alpha")
    alpha.start_mission("alpha_morning_001")

    discoveries_alpha = [
        ("Fallen cable tray at corridor junction", (5.0, 0.0, 0.0)),
        ("Water leak near electrical panel", (10.0, 10.0, 0.0)),
        ("Loose floor tile at server room entrance", (10.0, 3.0, 0.0)),
    ]

    for desc, pos in discoveries_alpha:
        mem_id = alpha.record_obstacle(description=desc, position=pos, confidence=0.9)
        print(f"  Alpha discovered: {desc}")
        print(f"    Position: {pos}, Memory: {mem_id[:8]}...")

    alpha.end_mission("3 obstacles discovered")
    alpha.flush()
    del alpha  # Release RocksDB lock

    # ─────────────────────────────────────────────────────────────
    # Phase 2: Robot Beta queries fleet knowledge before patrol
    # ─────────────────────────────────────────────────────────────
    print()
    print("[Phase 2] Robot Beta — Pre-patrol briefing from fleet memory")
    print("-" * 50)

    beta = FleetSpotBridge(storage_path=shared_path, robot_name="spot_beta")
    beta.start_mission("beta_afternoon_001")

    # Beta queries all known obstacles — sees Alpha's discoveries
    fleet_obstacles = beta.recall_fleet_obstacles(limit=10)

    print(f"  Beta received {len(fleet_obstacles)} hazard warnings from fleet memory:")
    for obs in fleet_obstacles:
        content = obs.get("content", "")
        score = obs.get("score", 0.0)
        print(f"    - {content[:55]:55s} [score: {score:.3f}]")

    # ─────────────────────────────────────────────────────────────
    # Phase 3: Robot Beta discovers additional hazards
    # ─────────────────────────────────────────────────────────────
    print()
    print("[Phase 3] Robot Beta — Afternoon patrol (new discoveries)")
    print("-" * 50)

    discoveries_beta = [
        ("Forklift blocking loading dock entrance", (0.0, 10.0, 0.0)),
        ("Spilled liquid near cafeteria exit", (5.0, 10.0, 0.0)),
    ]

    for desc, pos in discoveries_beta:
        mem_id = beta.record_obstacle(description=desc, position=pos, confidence=0.85)
        print(f"  Beta discovered: {desc}")
        print(f"    Position: {pos}, Memory: {mem_id[:8]}...")

    beta.end_mission("2 new obstacles discovered, 3 pre-recalled from Alpha")
    beta.flush()
    del beta  # Release RocksDB lock

    # ─────────────────────────────────────────────────────────────
    # Phase 4: Robot Alpha's next mission gets Beta's discoveries
    # ─────────────────────────────────────────────────────────────
    print()
    print("[Phase 4] Robot Alpha — Evening patrol (receives Beta's discoveries)")
    print("-" * 50)

    alpha = FleetSpotBridge(storage_path=shared_path, robot_name="spot_alpha")
    alpha.start_mission("alpha_evening_001")

    all_fleet_obstacles = alpha.recall_fleet_obstacles(limit=20)

    print(f"  Alpha received {len(all_fleet_obstacles)} total hazard warnings:")
    for obs in all_fleet_obstacles:
        content = obs.get("content", "")
        score = obs.get("score", 0.0)
        print(f"    - {content[:55]:55s} [score: {score:.3f}]")

    alpha.end_mission()
    alpha.flush()

    # ─────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  FLEET KNOWLEDGE SHARING RESULTS")
    print("=" * 70)
    print()
    print(f"  Robot Alpha discovered:  {len(discoveries_alpha)} obstacles (morning)")
    print(f"  Robot Beta pre-recalled: {len(fleet_obstacles)} obstacles from Alpha")
    print(f"  Robot Beta discovered:   {len(discoveries_beta)} new obstacles (afternoon)")
    print(f"  Robot Alpha received:    {len(all_fleet_obstacles)} total (evening)")
    print()
    print("  How it works:")
    print("    - Both robots share the same shodh-memory store (no robot_id filter)")
    print("    - Robot attribution via tags: ['obstacle', 'fleet', 'spot_alpha']")
    print("    - recall() returns all fleet memories regardless of origin")
    print("    - robot_id IS available for isolated queries when needed")
    print("    - Production: both connect to shodh server on port 3030")
    print("    - No Orbit server needed, no cloud dependency")
    print()
    print("  Without fleet memory: Each robot discovers obstacles independently")
    print("  With fleet memory:    One discovery benefits all robots instantly")
    print()


if __name__ == "__main__":
    run_demo()
