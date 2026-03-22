"""
Area Callback with Memory-Backed Decisions

Spot's Area Callbacks let you define custom behavior when the robot
enters specific regions. But each callback starts fresh — no memory
of what happened last time in that region.

This example implements a MemoryAreaCallback that:
  1. Queries past experiences before entering a region
  2. Adjusts speed/behavior based on historical outcomes
  3. Records the traversal result for future callbacks

After 3 traversals, the callback has learned which regions are
dangerous and automatically adjusts behavior.

Run:
    pip install shodh-memory
    python area_callback_memory.py
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from shodh_spot_bridge import SpotMemoryBridge


# =============================================================================
# Speed policies (mirrors Spot SDK locomotion hint speed values)
# =============================================================================

POLICY_NORMAL = "NORMAL"
POLICY_CAUTION = "CAUTION"
POLICY_SLOW = "SLOW_DOWN"


# =============================================================================
# Region definitions
# =============================================================================

@dataclass
class Region:
    id: str
    name: str
    center: Tuple[float, float, float]
    radius: float = 3.0


REGIONS = [
    Region("R1", "server_room_entrance", (10.0, 3.0, 0.0)),
    Region("R2", "wet_floor_zone", (5.0, 8.0, 0.0)),
    Region("R3", "loading_dock_crossing", (0.0, 10.0, 0.0)),
    Region("R4", "narrow_corridor", (5.0, 0.0, 0.0)),
]


# =============================================================================
# Memory-backed Area Callback
# =============================================================================

class MemoryAreaCallback:
    """Area Callback that remembers past region traversals.

    On a real Spot, this would implement the Area Callback gRPC service
    (UpdateCallback RPC). Here we simulate the callback lifecycle:

        1. on_begin(region) → queries memory → returns speed policy
        2. on_traverse(region) → robot traverses → stuff happens
        3. on_end(region, outcome) → records result for future
    """

    def __init__(self, bridge: SpotMemoryBridge):
        self.bridge = bridge
        self.traversal_count: Dict[str, int] = {}

    def on_begin(self, region: Region) -> str:
        """Called when robot is about to enter a region.

        Queries memory for past experiences in this region and returns
        the appropriate speed policy.
        """
        # Query past events in this region
        past = self.bridge.recall_region_history(region.id, limit=10)

        # Count past outcomes
        failures = 0
        successes = 0
        for m in past:
            content = m.get("content", "").lower()
            if "failure" in content or "obstacle" in content or "slip" in content:
                failures += 1
            elif "success" in content or "clear" in content:
                successes += 1

        # Determine policy based on history
        if failures >= 2:
            return POLICY_SLOW
        elif failures >= 1:
            return POLICY_CAUTION
        elif successes >= 3:
            return POLICY_NORMAL
        else:
            return POLICY_NORMAL  # No history — proceed normally

    def on_end(
        self,
        region: Region,
        outcome: str,
        details: str = "",
    ) -> None:
        """Called after traversing a region. Records the outcome.

        Args:
            region: The region that was traversed
            outcome: "success", "failure", or "partial"
            details: Description of what happened
        """
        self.traversal_count[region.id] = self.traversal_count.get(region.id, 0) + 1

        self.bridge.record_area_callback_event(
            region_id=region.id,
            stage="completed",
            action_taken=details or f"traversal_{outcome}",
            outcome_type=outcome,
            position=region.center,
        )

        # Also record failures as obstacles for cross-system recall
        if outcome == "failure":
            self.bridge.record_failure(
                description=f"Region {region.id} ({region.name}): {details}",
                severity="warning",
                position=region.center,
            )


# =============================================================================
# Simulation — 3 rounds of traversals
# =============================================================================

# (region_id, outcome, details)
ROUND_1_EVENTS = [
    ("R1", "success", "Server room entry clear, door open"),
    ("R2", "failure", "Slipped on wet floor — recovery needed"),
    ("R3", "success", "Loading dock crossing clear"),
    ("R4", "success", "Narrow corridor traversed normally"),
]

ROUND_2_EVENTS = [
    ("R1", "success", "Server room entry clear"),
    ("R2", "failure", "Wet floor again — obstacle avoidance triggered"),
    ("R3", "partial", "Loading dock — had to wait for forklift"),
    ("R4", "success", "Narrow corridor clear"),
]

ROUND_3_EVENTS = [
    ("R1", "success", "Server room clear"),
    ("R2", "success", "Wet floor — SLOW speed prevented slip"),
    ("R3", "success", "Loading dock clear, no forklifts"),
    ("R4", "success", "Narrow corridor clear"),
]


def get_region(region_id: str) -> Region:
    for r in REGIONS:
        if r.id == region_id:
            return r
    raise ValueError(f"Unknown region: {region_id}")


def run_demo():
    print("=" * 70)
    print("  AREA CALLBACK WITH MEMORY — Learning Region Behavior")
    print("=" * 70)

    # Clean up previous run data for reproducible results
    demo_path = Path("./spot_area_callback_demo")
    if demo_path.exists():
        shutil.rmtree(demo_path)

    bridge = SpotMemoryBridge(
        storage_path=str(demo_path),
        robot_id="spot_alpha",
    )
    callback = MemoryAreaCallback(bridge)
    bridge.start_mission("area_callback_demo")

    all_rounds = [
        (1, ROUND_1_EVENTS),
        (2, ROUND_2_EVENTS),
        (3, ROUND_3_EVENTS),
    ]

    decision_log: List[Dict] = []

    for round_num, events in all_rounds:
        print()
        print(f"{'=' * 3} ROUND {round_num} {'=' * 60}")

        for region_id, outcome, details in events:
            region = get_region(region_id)

            # Step 1: Area callback decides speed policy
            policy = callback.on_begin(region)

            # Step 2: Record the traversal
            callback.on_end(region, outcome, details)

            # Log it
            decision_log.append({
                "round": round_num,
                "region": region_id,
                "policy": policy,
                "outcome": outcome,
            })

            policy_indicator = {
                POLICY_NORMAL: "  ",
                POLICY_CAUTION: "! ",
                POLICY_SLOW: "!!",
            }.get(policy, "  ")

            outcome_symbol = {
                "success": "OK",
                "failure": "FAIL",
                "partial": "PART",
            }.get(outcome, "??")

            print(
                f"  [{policy_indicator}] {region_id} ({region.name:25s}) "
                f"policy={policy:8s} outcome={outcome_symbol:4s}  {details}"
            )

    # ─────────────────────────────────────────────────────────────
    # Decision evolution table
    # ─────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  DECISION EVOLUTION")
    print("=" * 70)
    print()
    print(f"  {'Region':8s} {'Round 1':12s} {'Round 2':12s} {'Round 3':12s}")
    print(f"  {'─' * 8} {'─' * 12} {'─' * 12} {'─' * 12}")

    for region in REGIONS:
        entries = [d for d in decision_log if d["region"] == region.id]
        row = f"  {region.id:8s}"
        for round_num in [1, 2, 3]:
            entry = next((d for d in entries if d["round"] == round_num), None)
            if entry:
                row += f" {entry['policy']:12s}"
            else:
                row += f" {'N/A':12s}"
        print(row)

    print()
    print("  Key observations:")
    print("  - R2 (wet_floor_zone): NORMAL -> CAUTION -> SLOW_DOWN")
    print("    Two failures triggered maximum caution. Round 3 succeeded")
    print("    because the callback slowed the robot down.")
    print("  - R1, R4: Always NORMAL — no failures, no policy change")
    print("  - R3: NORMAL -> NORMAL -> NORMAL (partial doesn't trigger caution)")
    print()
    print("  Without memory: robot slips on wet floor EVERY time")
    print("  With memory:    robot learns to slow down after 2 slips")
    print()

    bridge.end_mission()
    bridge.flush()


if __name__ == "__main__":
    run_demo()
