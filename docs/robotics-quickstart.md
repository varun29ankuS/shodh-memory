# Robotics Quickstart

Shodh-memory gives robots persistent, queryable memory that survives power cycles. Store experiences with GPS coordinates, mission context, reward signals, and sensor data — then recall them by location, mission, or outcome.

## Setup

```bash
docker run -d -p 3030:3030 -v shodh_data:/data varunankuS/shodh-memory:latest
```

The server is now running at `http://localhost:3030`.

## Store a Robotics Memory

```bash
curl -X POST http://localhost:3030/api/remember -H "Content-Type: application/json" -d '{
  "user_id": "robot_001",
  "content": "Navigated to charging station via corridor B. Obstacle detected at waypoint 3, rerouted through corridor C.",
  "type": "Task",
  "importance": 0.7,
  "tags": ["navigation", "obstacle-avoidance"],
  "robot_id": "robot_001",
  "mission_id": "patrol_night_shift_042",
  "action_type": "navigate",
  "reward": 0.8,
  "outcome_type": "success",
  "terrain_type": "indoor",
  "geo_location": [37.7749, -122.4194, 2.5],
  "heading": 127.3,
  "local_position": [12.5, 3.2, 0.0],
  "sensor_data": {
    "battery": 23.5,
    "temperature": 22.1,
    "lidar_range": 4.7
  }
}'
```

### Fields Reference

| Field | Type | Description |
|-------|------|-------------|
| `robot_id` | string | Robot identifier (for multi-robot fleets) |
| `mission_id` | string | Mission/task grouping (e.g., "patrol_night_042") |
| `action_type` | string | Action performed (navigate, grasp, dock, inspect) |
| `reward` | float | RL reward signal, -1.0 to 1.0 |
| `outcome_type` | string | success, failure, partial, aborted, timeout |
| `terrain_type` | string | indoor, outdoor, urban, rural, water, aerial |
| `geo_location` | [lat, lon, alt] | WGS84 GPS coordinates |
| `heading` | float | Heading in degrees, 0-360 |
| `local_position` | [x, y, z] | Local frame position in meters |
| `sensor_data` | object | Arbitrary sensor readings (battery, temp, etc.) |

## Recall by Location (Spatial Mode)

Find memories near a GPS coordinate. Uses geohash indexing for sub-meter precision.

```bash
curl -X POST http://localhost:3030/api/recall -H "Content-Type: application/json" -d '{
  "user_id": "robot_001",
  "query": "obstacle near corridor",
  "mode": "spatial",
  "geo_lat": 37.7749,
  "geo_lon": -122.4194,
  "geo_radius_meters": 50.0,
  "limit": 10
}'
```

Results are sorted by distance (closest first).

## Recall by Mission

Replay all memories from a specific mission in chronological order.

```bash
curl -X POST http://localhost:3030/api/recall -H "Content-Type: application/json" -d '{
  "user_id": "robot_001",
  "query": "what happened during patrol",
  "mode": "mission",
  "mission_id": "patrol_night_shift_042",
  "limit": 50
}'
```

## Recall by Outcome (Reinforcement Learning)

Find memories with specific reward outcomes. Useful for learning from success/failure.

```bash
# What actions led to positive rewards?
curl -X POST http://localhost:3030/api/recall -H "Content-Type: application/json" -d '{
  "user_id": "robot_001",
  "query": "successful navigation actions",
  "mode": "action_outcome",
  "reward_min": 0.5,
  "reward_max": 1.0,
  "limit": 20
}'

# What went wrong? (failures only)
curl -X POST http://localhost:3030/api/recall -H "Content-Type: application/json" -d '{
  "user_id": "robot_001",
  "query": "failed actions",
  "mode": "action_outcome",
  "reward_min": -1.0,
  "reward_max": -0.3,
  "failures_only": true,
  "limit": 20
}'
```

Results are sorted by reward (highest first).

## Combined Filters

All filters compose. Query by location within a mission, or by action type with outcome filters:

```bash
curl -X POST http://localhost:3030/api/recall -H "Content-Type: application/json" -d '{
  "user_id": "robot_001",
  "query": "grasp attempts",
  "mode": "action_outcome",
  "action_type": "grasp",
  "robot_id": "robot_001",
  "terrain_type": "indoor",
  "reward_min": 0.0,
  "reward_max": 1.0,
  "limit": 10
}'
```

## Reward-Based Learning

Shodh uses reward signals for Hebbian edge strengthening in the knowledge graph:

- **Positive reward** (e.g., +0.8): Strengthens associations between entities in the memory. "Battery low" → "navigate to charger" becomes a stronger association.
- **Negative reward** (e.g., -0.7): Weakens associations. "Take corridor A during rush hour" becomes less preferred.
- **Neutral** (0.0): No modulation, standard edge strength.

This happens automatically at store time — no extra API calls needed. Over time, the graph learns which action-context pairs produce good outcomes.

## Multi-Robot Fleets

Use `robot_id` to partition memories per robot, and `mission_id` to group shared missions:

```bash
# Store from robot 2
curl -X POST http://localhost:3030/api/remember -H "Content-Type: application/json" -d '{
  "user_id": "fleet_warehouse_01",
  "content": "Aisle 7 blocked by fallen pallet at position (15.2, 7.1, 0.0)",
  "robot_id": "robot_002",
  "mission_id": "inventory_scan_043",
  "action_type": "inspect",
  "reward": -0.3,
  "outcome_type": "partial",
  "geo_location": [37.7750, -122.4195, 0.0],
  "local_position": [15.2, 7.1, 0.0],
  "sensor_data": {"battery": 67.2}
}'

# Robot 3 queries: "anything blocking the aisles?"
curl -X POST http://localhost:3030/api/recall -H "Content-Type: application/json" -d '{
  "user_id": "fleet_warehouse_01",
  "query": "blocked aisles obstacles",
  "mode": "spatial",
  "geo_lat": 37.7750,
  "geo_lon": -122.4195,
  "geo_radius_meters": 200.0
}'
```

## Python SDK

```python
from shodh_memory import ShodhMemory

mem = ShodhMemory(storage_path="/data/robot_memory")

# Store
mem.remember(
    user_id="robot_001",
    content="Docked successfully at station B",
    experience_type="Task",
    robot_id="robot_001",
    mission_id="patrol_042",
    reward=0.9,
    geo_location=[37.7749, -122.4194, 2.5],
)

# Recall by location
results = mem.recall(
    user_id="robot_001",
    query="docking experiences",
    mode="spatial",
    geo_lat=37.7749,
    geo_lon=-122.4194,
    geo_radius_meters=10.0,
)
```

## Zenoh Transport (ROS2 / DDS)

For real-time robotics middleware, shodh-memory supports Zenoh pub/sub transport. See `src/zenoh_transport/` for the implementation. Zenoh enables:

- Sub-millisecond memory store/recall over local network
- No HTTP overhead — direct binary protocol
- Compatible with ROS2 via zenoh-bridge-ros2dds

## Architecture Notes

- **Storage**: RocksDB with column families, persists across power cycles
- **Spatial indexing**: Geohash precision 10 (~1.2m × 0.6m cells), 9-cell neighbor search
- **Reward indexing**: Bucketed (-1.0 to 1.0 → 21 integer buckets) for O(1) range queries
- **Mission indexing**: Prefix-scanned by mission_id for chronological replay
- **Knowledge graph**: Entities from memories form a Hebbian graph with reward-modulated edge strengths
- **Embeddings**: MiniLM-L6-v2 (384-dim) via ONNX Runtime, runs locally on CPU/GPU
- **No cloud dependency**: All processing is local, no API keys, no data leaves the device
