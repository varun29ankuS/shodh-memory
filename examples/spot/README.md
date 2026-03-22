# Shodh-Memory for Boston Dynamics Spot

Persistent cognitive memory for Spot robots. Solves 6 fundamental limitations in the Spot SDK.

## The Problem

Spot's SDK is stateless by design:

| Limitation | Impact |
|---|---|
| **World objects expire after ~15 seconds** | Robot forgets obstacles it saw 30 seconds ago |
| **GraphNav maps don't survive reboot** | No learning between power cycles |
| **Mission blackboard dies when mission ends** | Every mission starts from scratch |
| **Area Callbacks are stateless** | Can't learn from past region traversals |
| **No semantic waypoint annotations** | Only name + opaque bytes, no queryable knowledge |
| **No fleet knowledge sharing** | Robot A's discovery doesn't help Robot B |

A Spot doing daily facility inspections will rediscover the same fallen cable tray every single day. An Area Callback that caused a slip yesterday runs at full speed today. A fleet of 3 robots each independently discovers the same obstacle.

## The Solution

Shodh-memory adds a persistent cognitive layer underneath Spot's existing services:

```
┌─────────────────────────────────────────────────────────────┐
│                     SPOT ROBOT                              │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ GraphNav │  │  World   │  │ Mission  │  │   Area    │  │
│  │ Service  │  │ Objects  │  │ Service  │  │ Callbacks │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬─────┘  │
│       │              │             │              │         │
│       └──────────────┴──────┬──────┴──────────────┘         │
│                             │                               │
│                   ┌─────────┴─────────┐                     │
│                   │ SpotMemoryBridge  │                     │
│                   │                   │                     │
│                   │  Dual-mode:       │                     │
│                   │  Simulated types  │                     │
│                   │  + Real protobufs │                     │
│                   │                   │                     │
│                   │  Euclidean spatial│                     │
│                   │  post-filter      │                     │
│                   └─────────┬─────────┘                     │
│                             │                               │
│                   ┌─────────┴─────────┐                     │
│                   │   shodh-memory    │                     │
│                   │  (native engine)  │                     │
│                   │                   │                     │
│                   │  Hebbian learning │                     │
│                   │  Vector search    │                     │
│                   │  Knowledge graph  │                     │
│                   │  Hybrid decay     │                     │
│                   │                   │                     │
│                   │  17MB · Offline   │                     │
│                   └───────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
pip install shodh-memory
cd examples/spot

# Run the full 5-day simulation
python spot_simulation.py

# Run individual examples
python persistent_world_objects.py    # 15-second TTL fix
python cross_mission_learning.py      # Knowledge accumulation
python semantic_waypoints.py          # Semantic annotations
python area_callback_memory.py        # Memory-backed callbacks
python fleet_memory.py                # Multi-robot sharing

# Run the performance benchmark
python benchmark.py
```

No `bosdyn-client` required. All Spot SDK types are simulated with compatible interfaces. When `bosdyn-client` is installed, the bridge automatically detects it and provides real protobuf conversion methods.

## What Each Example Demonstrates

### `persistent_world_objects.py` — Solving the 15-Second TTL

Spot detects 5 objects. After 15 seconds, Spot's native service returns 0. Shodh-memory returns all 5 with original positions, types, and metadata.

```
  Spot native:  0 objects (all expired)
  Shodh-memory: 5 objects (all retained)
```

### `cross_mission_learning.py` — Knowledge Accumulation

3 missions through the same facility. Mission 1 is blind — zero prior knowledge. Missions 2 and 3 pre-recall previously-seen obstacles before encountering them. Pre-recall uses real Euclidean distance filtering on stored positions, not text matching.

```
  Mission 1: obstacles discovered blind, 0 pre-recalled
  Mission 2: persistent obstacles pre-recalled, reroutes applied
  Mission 3: full situational awareness, one-off anomalies decayed
```

### `semantic_waypoints.py` — Queryable Waypoint Knowledge

Natural language queries on waypoint annotations. GraphNav gives you: waypoint name + opaque bytes. Shodh-memory gives you: semantic search across all observations.

### `area_callback_memory.py` — Learning Region Behavior

An Area Callback that remembers past traversals. Region R2 (wet floor) causes a slip in Round 1, another in Round 2. By Round 3, the callback automatically slows down — and the slip doesn't happen.

```
  R2 (wet_floor_zone): NORMAL → CAUTION → SLOW_DOWN
  Without memory: slips every time
  With memory:    learns to slow down after 2 failures
```

### `fleet_memory.py` — Multi-Robot Knowledge Sharing

Two robots, shared storage. Robot Alpha discovers 3 obstacles in the morning. Robot Beta gets all 3 warnings before its afternoon patrol. No Orbit server, no cloud, no network — just a shared filesystem.

### `spot_simulation.py` — Full 5-Day Simulation

Combines everything into a realistic industrial inspection:
- 12 waypoints across 5 facility zones
- 5 days of patrol with evolving conditions
- Persistent obstacles (cable tray: Days 1-4), progressive issues (water leak: Days 1-3), transient anomalies (dock noise: Day 1 only)
- Pre-recall rate improves from 0% (Day 1) toward 100% as knowledge accumulates

### `benchmark.py` — Performance Measurement

Measures real latency at robotics scale (100 to 10,000 obstacles):

```
  N = 100, 500, 1,000, 5,000, 10,000

  Per operation:
    Store:          embedding + NER + graph + storage per obstacle
    Recall:         hybrid search (semantic + graph + BM25) with tag filter
    Spatial recall: hybrid search + Euclidean distance post-filter

  Reports: p50, p95, p99 latencies + RocksDB disk footprint
```

Run `python benchmark.py` to get numbers for your hardware.

## Spatial Recall

The bridge provides real Euclidean distance filtering on stored positions. Every memory in shodh-memory stores `local_position: [f32; 3]`, which is returned as the `"position"` key in recall results.

`recall_obstacles_nearby()` and `recall_world_objects()` use this:

1. Fetch extra results via tag-filtered hybrid search (5x the limit)
2. Extract stored position from each result
3. Compute Euclidean distance to query point
4. Filter by radius, sort by distance (closest first)
5. Truncate to requested limit

Each returned memory includes `distance_meters` — the actual Euclidean distance from the query point.

```python
# All results are within 3m of this position, sorted closest-first
nearby = bridge.recall_obstacles_nearby(
    position=(10.0, 5.0, 0.0),
    radius=3.0,
    limit=5,
)
for obs in nearby:
    print(f"  {obs['content']} — {obs['distance_meters']:.2f}m away")
```

## Integration Points

| Spot SDK Pain Point | Shodh Solution | Method |
|---|---|---|
| World objects expire (15s TTL) | Permanent persistence | `bridge.persist_world_object()` |
| No cross-mission learning | Accumulated knowledge | `bridge.recall_obstacles_nearby()` |
| Static waypoint annotations | Semantic layer | `bridge.annotate_waypoint()` |
| Stateless Area Callbacks | Memory-backed decisions | `bridge.recall_region_history()` |
| No fleet sharing | Shared storage | Same `storage_path`, tag-based attribution |
| Mission blackboard ephemeral | Persistent decisions | `bridge.record_navigation_decision()` |

## SpotMemoryBridge API

```python
from shodh_spot_bridge import SpotMemoryBridge

bridge = SpotMemoryBridge(
    storage_path="./spot_memory",  # RocksDB storage directory
    robot_id="spot_01",            # Unique robot ID
)

# Mission lifecycle
bridge.start_mission("inspection_001")
bridge.end_mission("summary text")

# World objects (15-second TTL fix)
bridge.persist_world_object(name="obstacle_A", object_type="WORLD_OBJECT_UNKNOWN",
                             position=(3.2, 1.5, 0.0))
bridge.recall_world_objects(position=(3.0, 1.0, 0.0), radius=5.0)

# Waypoints (semantic annotations)
bridge.annotate_waypoint("W3", "charging station — 2hr typical wait")
bridge.recall_waypoint_history("W3")

# Obstacles (Euclidean distance filtering)
bridge.record_obstacle("Fallen cable tray", position=(5.0, 0.0, 0.0))
nearby = bridge.recall_obstacles_nearby(position=(5.0, 0.0, 0.0), radius=3.0)

# Decisions (action-outcome learning)
bridge.record_navigation_decision(
    description="Rerouted around known obstacle",
    action="reroute",
    state={"waypoint": "W3"},
    outcome=NavigationFeedbackResponse(status="reached_goal"),
)

# Sensors and anomalies
bridge.record_sensor_reading("thermal_cam", {"temperature": 38.0}, is_anomaly=True)
bridge.record_failure("Slip on wet floor", severity="warning")

# Area Callbacks
bridge.record_area_callback_event("R2", "completed", "slow_traverse", "success")
bridge.recall_region_history("R2")
```

## Moving to Real Spot Hardware

The bridge is **dual-mode**: it detects `bosdyn-client` at import time and provides real protobuf conversion methods alongside the simulated types.

```python
import bosdyn.client
# bosdyn-client detected → _HAS_BOSDYN = True
# All _from_proto() methods are now available
```

### 1. Persist real WorldObject protobufs

```python
from shodh_spot_bridge import SpotMemoryBridge

bridge = SpotMemoryBridge(storage_path="/opt/spot/memory", robot_id="spot_01")

# From Spot's WorldObjectClient
world_object_client = robot.ensure_client("world-object")
objects = world_object_client.list_world_objects().world_objects

for obj in objects:
    # Extracts position from transforms_snapshot, maps object_type enum,
    # pulls apriltag_properties / dock_properties as metadata
    bridge.persist_world_object_from_real_proto(obj)
```

### 2. Convert real navigation feedback

```python
# From GraphNavClient navigation command
nav_response = graph_nav_client.navigate_to(waypoint_id)
feedback = graph_nav_client.navigation_feedback()

# Maps real protobuf status enums (STATUS_REACHED_GOAL, STATUS_STUCK, etc.)
outcome = bridge.nav_feedback_from_proto(feedback)

bridge.record_navigation_decision(
    description="Navigate to server room",
    action="navigate_to",
    state={"target": waypoint_id},
    outcome=outcome,  # Real Outcome from proto conversion
    position=(10.0, 5.0, 0.0),
)
```

### 3. Convert real robot state

```python
# From RobotStateClient
state_client = robot.ensure_client("robot-state")
robot_state = state_client.get_robot_state()

# Extracts battery %, e-stop state, motor power from real protobuf
environment = bridge.robot_state_from_proto(robot_state)
```

### 4. Persist real GraphNav waypoints

```python
# From GraphNavClient
graph = graph_nav_client.download_graph()
for waypoint in graph.waypoints:
    # Extracts position from waypoint_tform_ko, annotations.name
    bridge.waypoint_from_proto(waypoint)
```

### 5. Wire into Area Callback gRPC service

```python
class MyAreaCallback(AreaCallbackServiceServicer):
    def __init__(self, bridge):
        self.bridge = bridge

    def UpdateCallback(self, request, context):
        history = self.bridge.recall_region_history(request.region_id)
        failures = [h for h in history if "failure" in h.get("content", "")]

        if len(failures) >= 2:
            # Slow down — this region has caused problems before
            speed_hint = 0.3
        else:
            speed_hint = 1.0

        # ... return response with speed_hint
```

### Available proto conversion methods

| Method | Input | Output |
|---|---|---|
| `persist_world_object_from_real_proto(proto)` | `world_object_pb2.WorldObject` | Memory ID |
| `se3pose_from_proto(proto)` | `geometry_pb2.SE3Pose` | `Position(x, y, z)` |
| `heading_from_proto_quaternion(proto)` | `geometry_pb2.Quaternion` | Heading in degrees |
| `nav_feedback_from_proto(proto)` | `graph_nav_pb2.NavigationFeedbackResponse` | `Outcome` |
| `robot_state_from_proto(proto)` | `robot_state_pb2.RobotState` | `Environment` |
| `waypoint_from_proto(proto)` | `map_pb2.Waypoint` | Memory ID |

All raise `ImportError` with an install instruction if `bosdyn-client` is not available.

## Performance

Run `python benchmark.py` to measure on your hardware. Typical numbers:

| Operation | p50 | Notes |
|---|---|---|
| Store obstacle | ~50ms | Embedding (MiniLM-L6-v2 ONNX) + NER + graph + RocksDB |
| Recall (semantic) | ~40ms | Hybrid search: vector similarity + BM25 + tag filter |
| Recall (spatial) | ~50ms | Hybrid search + Euclidean distance post-filter |
| Tag-based lookup | <1ms | Direct index |
| Graph traversal (3-hop) | <1ms | In-memory adjacency |

Scales to 10,000+ obstacles with sub-100ms recall. Single 17MB binary. No GPU. Runs on Jetson Nano, Raspberry Pi 4, any x86_64/ARM64.

## How It Learns

```
Object seen once      → stored with base score
Object seen 3+ times  → Hebbian boost: easier to recall
Object seen 10+ times → Long-term potentiation: quasi-permanent
Object not seen again  → Exponential decay (first 3 days), power-law after
```

Obstacles you encounter repeatedly become part of the robot's permanent knowledge base. One-off anomalies naturally fade. No manual cleanup needed.

## License

Apache 2.0 — [shodh-memory](https://github.com/varun29ankuS/shodh-memory)
