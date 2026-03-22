"""
Spot SDK <> Shodh-Memory Bridge

Type translation layer that maps Boston Dynamics Spot SDK protobuf types
to shodh-memory's cognitive memory primitives.

This bridge enables persistent memory for Spot robots — solving the SDK's
fundamental limitation that world objects expire after 15 seconds, maps
don't survive reboot, and mission state dies when a mission ends.

Dual-mode: when bosdyn-client is installed, provides real protobuf
conversion methods. Without it, simulated types let you run all examples
standalone.

Usage:
    from shodh_spot_bridge import SpotMemoryBridge

    bridge = SpotMemoryBridge(storage_path="./spot_memory", robot_id="spot_01")
    bridge.start_mission("inspection_2026_03_22")

    # Persist a world object (survives beyond 15-second TTL)
    bridge.persist_world_object(name="obstacle_A", object_type="obstacle",
                                position=(3.2, 1.5, 0.0))

    # Recall objects near a position — real Euclidean distance filtering
    objects = bridge.recall_world_objects(position=(3.0, 1.0, 0.0), radius=5.0)

Type Mapping:
    Spot SDK Type              -> shodh-memory Type
    ──────────────────────────────────────────────────
    SE3Pose.position (Vec3)    -> Position(x, y, z)
    GPS (lat/lon/alt)          -> GeoLocation(latitude, longitude, altitude)
    WorldObject                -> remember() with entity tags + position
    Waypoint.Annotations       -> remember() with waypoint_id tag + metadata
    NavigationFeedbackResponse -> Outcome(success/failure/partial)
    RobotState                 -> Environment(battery, terrain, lighting)
    Area Callback stage        -> remember() with action_type="area_callback"
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from shodh_memory import (
    DecisionContext,
    Environment,
    GeoFilter,
    GeoLocation,
    MemorySystem,
    Outcome,
    Position,
)


# =============================================================================
# Detect real Spot SDK (optional dependency)
# =============================================================================

try:
    from bosdyn.api import robot_state_pb2 as _robot_state_pb2
    from bosdyn.api import world_object_pb2 as _world_object_pb2
    from bosdyn.api.graph_nav import graph_nav_pb2 as _graph_nav_pb2
    _HAS_BOSDYN = True
except ImportError:
    _HAS_BOSDYN = False


# =============================================================================
# Simulated Spot SDK types (for running without bosdyn-client)
# =============================================================================
# These mirror the real Spot SDK protobuf structures so examples work
# standalone. On a real Spot, use the _from_proto() methods instead.

@dataclass
class Vec3:
    """Simulates bosdyn.api.geometry_pb2.Vec3"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    """Simulates bosdyn.api.geometry_pb2.Quaternion"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class SE3Pose:
    """Simulates bosdyn.api.geometry_pb2.SE3Pose"""
    position: Vec3 = field(default_factory=Vec3)
    rotation: Quaternion = field(default_factory=Quaternion)


@dataclass
class WorldObject:
    """Simulates bosdyn.api.world_object_pb2.WorldObject"""
    id: int = 0
    name: str = ""
    object_type: str = "WORLD_OBJECT_UNKNOWN"
    acquisition_time: float = 0.0
    transforms_snapshot: Optional[Dict[str, SE3Pose]] = None
    additional_properties: Optional[Dict[str, Any]] = None

    WORLD_OBJECT_UNKNOWN = "WORLD_OBJECT_UNKNOWN"
    WORLD_OBJECT_APRILTAG = "WORLD_OBJECT_APRILTAG"
    WORLD_OBJECT_DOCK = "WORLD_OBJECT_DOCK"
    WORLD_OBJECT_TRACKED_ENTITY = "WORLD_OBJECT_TRACKED_ENTITY"
    WORLD_OBJECT_USER_NOGO = "WORLD_OBJECT_USER_NOGO"
    WORLD_OBJECT_STAIRCASE = "WORLD_OBJECT_STAIRCASE"


@dataclass
class Waypoint:
    """Simulates bosdyn.api.graph_nav.map_pb2.Waypoint"""
    id: str = ""
    name: str = ""
    seed_tform_waypoint: SE3Pose = field(default_factory=SE3Pose)
    annotations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NavigationFeedbackResponse:
    """Simulates bosdyn.api.graph_nav.graph_nav_pb2.NavigationFeedbackResponse"""
    STATUS_REACHED_GOAL = "reached_goal"
    STATUS_FOLLOWING_ROUTE = "following_route"
    STATUS_STUCK = "stuck"
    STATUS_ROBOT_LOST = "robot_lost"
    STATUS_COMMAND_TIMED_OUT = "timed_out"

    status: str = "reached_goal"
    remaining_route_length: float = 0.0
    distance_to_goal: float = 0.0


@dataclass
class RobotState:
    """Simulates bosdyn.api.robot_state_pb2.RobotState"""
    battery_percentage: float = 100.0
    battery_charge_state: str = "full"
    estop_state: str = "not_estopped"
    is_powered_on: bool = True
    terrain_type: str = "indoor"
    lighting: str = "bright"


# =============================================================================
# SpotMemoryBridge — Main integration class
# =============================================================================

class SpotMemoryBridge:
    """Bridge between Spot SDK and shodh-memory cognitive system.

    Provides persistent memory for Spot robots, solving:
    - 15-second world object TTL -> permanent object persistence
    - No cross-mission learning -> accumulated knowledge across missions
    - Static map annotations -> semantic waypoint knowledge
    - No fleet sharing -> multi-robot knowledge via shared storage

    Args:
        storage_path: Directory for shodh-memory's RocksDB storage
        robot_id: Unique identifier for this robot (enables fleet memory)
    """

    OBJECT_TYPE_TAGS = {
        "WORLD_OBJECT_UNKNOWN": ["world_object", "unknown"],
        "WORLD_OBJECT_APRILTAG": ["world_object", "apriltag", "fiducial"],
        "WORLD_OBJECT_DOCK": ["world_object", "dock", "charging"],
        "WORLD_OBJECT_TRACKED_ENTITY": ["world_object", "tracked", "entity"],
        "WORLD_OBJECT_USER_NOGO": ["world_object", "nogo", "restricted"],
        "WORLD_OBJECT_STAIRCASE": ["world_object", "staircase", "stairs"],
    }

    def __init__(
        self,
        storage_path: str = "./spot_memory",
        robot_id: str = "spot_01",
    ):
        self.memory = MemorySystem(storage_path=storage_path, robot_id=robot_id)
        self.robot_id = robot_id
        self._mission_id: Optional[str] = None

    # =========================================================================
    # Euclidean spatial post-filter
    # =========================================================================

    @staticmethod
    def _euclidean_distance(
        pos_a: Tuple[float, float, float],
        pos_b: List[float],
    ) -> float:
        """Euclidean distance between a query position and a stored position."""
        dx = pos_a[0] - pos_b[0]
        dy = pos_a[1] - pos_b[1]
        dz = pos_a[2] - pos_b[2] if len(pos_b) > 2 else 0.0
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @classmethod
    def _spatial_post_filter(
        cls,
        memories: List[dict],
        position: Tuple[float, float, float],
        radius: float,
        limit: int,
    ) -> List[dict]:
        """Filter recalled memories by Euclidean distance to a query position.

        shodh-memory stores local_position on every memory and returns it as
        the "position" key in recall results. This method uses it for real
        distance-based spatial filtering — not text matching.

        Args:
            memories: Raw recall results (each dict may have "position" key)
            position: Query center point (x, y, z) in meters
            radius: Maximum distance in meters
            limit: Maximum results to return

        Returns:
            Memories within radius, sorted by distance (closest first)
        """
        scored = []
        for mem in memories:
            stored_pos = mem.get("position")
            if stored_pos and isinstance(stored_pos, (list, tuple)) and len(stored_pos) >= 2:
                dist = cls._euclidean_distance(position, stored_pos)
                if dist <= radius:
                    mem["distance_meters"] = round(dist, 3)
                    scored.append((dist, mem))

        scored.sort(key=lambda x: x[0])
        return [mem for _, mem in scored[:limit]]

    # =========================================================================
    # Mission lifecycle
    # =========================================================================

    def start_mission(self, mission_id: str) -> None:
        """Begin a named mission. All memories are tagged with this mission_id."""
        self._mission_id = mission_id
        self.memory.start_mission(mission_id)
        self.memory.remember(
            f"Mission started: {mission_id}",
            memory_type="task",
            tags=["mission", "start", mission_id],
            entities=[mission_id],
        )

    def end_mission(self, summary: Optional[str] = None) -> None:
        """End the current mission with optional summary."""
        if self._mission_id:
            content = f"Mission ended: {self._mission_id}"
            if summary:
                content += f" — {summary}"
            self.memory.remember(
                content,
                memory_type="task",
                tags=["mission", "end", self._mission_id],
                entities=[self._mission_id],
            )
        self.memory.end_mission()
        self._mission_id = None

    @property
    def current_mission(self) -> Optional[str]:
        return self._mission_id

    # =========================================================================
    # Type conversions: Simulated Spot types -> shodh-memory
    # =========================================================================

    @staticmethod
    def se3pose_to_position(pose: SE3Pose) -> Position:
        """Convert Spot's SE3Pose to shodh Position (extracts translation)."""
        return Position(
            x=float(pose.position.x),
            y=float(pose.position.y),
            z=float(pose.position.z),
        )

    @staticmethod
    def vec3_to_position(vec: Vec3) -> Position:
        """Convert Spot's Vec3 to shodh Position."""
        return Position(x=float(vec.x), y=float(vec.y), z=float(vec.z))

    @staticmethod
    def gps_to_geolocation(latitude: float, longitude: float, altitude: float = 0.0) -> GeoLocation:
        """Convert GPS coordinates to shodh GeoLocation."""
        return GeoLocation(latitude=latitude, longitude=longitude, altitude=altitude)

    @staticmethod
    def robot_state_to_environment(state: RobotState) -> Environment:
        """Convert Spot's RobotState to shodh Environment."""
        return Environment(
            weather={"battery": f"{state.battery_percentage:.0f}%"},
            terrain_type=state.terrain_type,
            lighting=state.lighting,
            nearby_agents=[],
        )

    @staticmethod
    def navigation_result_to_outcome(feedback: NavigationFeedbackResponse) -> Outcome:
        """Convert navigation feedback to shodh Outcome."""
        status_map = {
            "reached_goal": ("success", 1.0),
            "following_route": ("partial", 0.5),
            "stuck": ("failure", -0.5),
            "robot_lost": ("failure", -1.0),
            "timed_out": ("timeout", -0.3),
        }
        outcome_type, reward = status_map.get(feedback.status, ("failure", -0.5))
        return Outcome(
            outcome_type=outcome_type,
            reward=reward,
            details=f"status={feedback.status}, remaining={feedback.remaining_route_length:.1f}m",
        )

    # =========================================================================
    # Type conversions: Real Spot SDK protobufs -> shodh-memory
    # =========================================================================

    @staticmethod
    def se3pose_from_proto(proto_pose: Any) -> Position:
        """Convert a real bosdyn.api.geometry_pb2.SE3Pose to shodh Position.

        Requires bosdyn-client: pip install bosdyn-client>=4.0.0
        """
        if not _HAS_BOSDYN:
            raise ImportError(
                "bosdyn-client is required for protobuf conversion. "
                "Install with: pip install bosdyn-client>=4.0.0"
            )
        return Position(
            x=float(proto_pose.position.x),
            y=float(proto_pose.position.y),
            z=float(proto_pose.position.z),
        )

    @staticmethod
    def heading_from_proto_quaternion(proto_quat: Any) -> float:
        """Extract yaw (heading) from a real bosdyn Quaternion in degrees."""
        if not _HAS_BOSDYN:
            raise ImportError("bosdyn-client is required for protobuf conversion.")
        siny_cosp = 2.0 * (proto_quat.w * proto_quat.z + proto_quat.x * proto_quat.y)
        cosy_cosp = 1.0 - 2.0 * (proto_quat.y * proto_quat.y + proto_quat.z * proto_quat.z)
        return math.degrees(math.atan2(siny_cosp, cosy_cosp))

    def persist_world_object_from_real_proto(self, proto_obj: Any) -> str:
        """Persist a real bosdyn.api.WorldObject protobuf to shodh-memory.

        Call this with objects from:
            world_object_client.list_world_objects().world_objects

        Args:
            proto_obj: A bosdyn.api.world_object_pb2.WorldObject instance

        Returns:
            Memory ID for the persisted object
        """
        if not _HAS_BOSDYN:
            raise ImportError("bosdyn-client is required for protobuf conversion.")

        # Map protobuf WorldObjectType enum to string
        type_map = {
            _world_object_pb2.WORLD_OBJECT_UNKNOWN: "WORLD_OBJECT_UNKNOWN",
            _world_object_pb2.WORLD_OBJECT_DRAWABLE: "WORLD_OBJECT_UNKNOWN",
            _world_object_pb2.WORLD_OBJECT_APRILTAG: "WORLD_OBJECT_APRILTAG",
            _world_object_pb2.WORLD_OBJECT_IMAGE_COORDINATES: "WORLD_OBJECT_UNKNOWN",
            _world_object_pb2.WORLD_OBJECT_DOCK: "WORLD_OBJECT_DOCK",
            _world_object_pb2.WORLD_OBJECT_TRACKED_ENTITY: "WORLD_OBJECT_TRACKED_ENTITY",
            _world_object_pb2.WORLD_OBJECT_USER_NOGO: "WORLD_OBJECT_USER_NOGO",
            _world_object_pb2.WORLD_OBJECT_STAIRCASE: "WORLD_OBJECT_STAIRCASE",
        }
        obj_type = type_map.get(proto_obj.object_type, "WORLD_OBJECT_UNKNOWN")

        # Extract position from transforms_snapshot frame tree
        position = None
        if proto_obj.HasField("transforms_snapshot"):
            snapshot = proto_obj.transforms_snapshot
            for edge in snapshot.child_to_parent_edge_map.values():
                if edge.HasField("parent_tform_child"):
                    pose = edge.parent_tform_child
                    position = (
                        float(pose.position.x),
                        float(pose.position.y),
                        float(pose.position.z),
                    )
                    break

        # Extract additional properties as metadata
        metadata = {"spot_object_id": str(proto_obj.id)}
        if proto_obj.name:
            metadata["spot_name"] = proto_obj.name
        if proto_obj.HasField("apriltag_properties"):
            tag = proto_obj.apriltag_properties
            metadata["tag_id"] = str(tag.tag_id)
            metadata["tag_family"] = str(tag.frame_name_fiducial)
        if proto_obj.HasField("dock_properties"):
            metadata["dock_id"] = str(proto_obj.dock_properties.dock_id)

        return self.persist_world_object(
            name=proto_obj.name or f"object_{proto_obj.id}",
            object_type=obj_type,
            position=position,
            metadata=metadata,
        )

    def nav_feedback_from_proto(self, proto_feedback: Any) -> Outcome:
        """Convert a real NavigationFeedbackResponse protobuf to shodh Outcome.

        Maps the real status enum values to outcome types.
        """
        if not _HAS_BOSDYN:
            raise ImportError("bosdyn-client is required for protobuf conversion.")

        status_map = {
            _graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL: ("success", 1.0),
            _graph_nav_pb2.NavigationFeedbackResponse.STATUS_FOLLOWING_ROUTE: ("partial", 0.5),
            _graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK: ("failure", -0.5),
            _graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST: ("failure", -1.0),
            _graph_nav_pb2.NavigationFeedbackResponse.STATUS_COMMAND_TIMED_OUT: ("timeout", -0.3),
            _graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED: ("failure", -0.8),
            _graph_nav_pb2.NavigationFeedbackResponse.STATUS_CONSTRAINT_FAULT: ("failure", -0.6),
        }
        outcome_type, reward = status_map.get(proto_feedback.status, ("failure", -0.5))
        return Outcome(
            outcome_type=outcome_type,
            reward=reward,
            details=f"proto_status={proto_feedback.status}",
        )

    def robot_state_from_proto(self, proto_state: Any) -> Environment:
        """Convert a real bosdyn.api.RobotState protobuf to shodh Environment.

        Extracts battery percentage, e-stop state, and power state.
        """
        if not _HAS_BOSDYN:
            raise ImportError("bosdyn-client is required for protobuf conversion.")

        battery_pct = 0.0
        if proto_state.battery_states:
            battery_pct = proto_state.battery_states[0].charge_percentage.value

        estop_active = any(
            es.state == _robot_state_pb2.EStopState.STATE_ESTOPPED
            for es in proto_state.estop_states
        )

        return Environment(
            weather={
                "battery": f"{battery_pct:.0f}%",
                "estop": "active" if estop_active else "clear",
                "powered": str(proto_state.power_state.motor_power_state != 0),
            },
            terrain_type="unknown",
            lighting="unknown",
            nearby_agents=[],
        )

    def waypoint_from_proto(self, proto_waypoint: Any) -> str:
        """Persist a real GraphNav Waypoint protobuf to shodh-memory.

        Call with waypoints from graph_nav_client.download_graph().
        Returns memory_id.
        """
        if not _HAS_BOSDYN:
            raise ImportError("bosdyn-client is required for protobuf conversion.")

        position = None
        if proto_waypoint.HasField("waypoint_tform_ko"):
            pose = proto_waypoint.waypoint_tform_ko
            position = (
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            )

        name = ""
        if proto_waypoint.HasField("annotations") and proto_waypoint.annotations.name:
            name = proto_waypoint.annotations.name

        return self.annotate_waypoint(
            waypoint_id=proto_waypoint.id,
            label=name or f"waypoint_{proto_waypoint.id[:8]}",
            position=position,
            metadata={"snapshot_id": proto_waypoint.snapshot_id},
        )

    # =========================================================================
    # World Object persistence (solves 15-second TTL)
    # =========================================================================

    def persist_world_object(
        self,
        name: str,
        object_type: str = "WORLD_OBJECT_UNKNOWN",
        position: Optional[Tuple[float, float, float]] = None,
        geo_location: Optional[Tuple[float, float, float]] = None,
        metadata: Optional[Dict[str, str]] = None,
        confidence: Optional[float] = None,
    ) -> str:
        """Persist a Spot WorldObject beyond its 15-second TTL.

        Args:
            name: Object name/identifier
            object_type: WorldObjectType enum string
            position: (x, y, z) in robot's local frame
            geo_location: (latitude, longitude, altitude) for GPS-enabled
            metadata: Additional key-value data
            confidence: Detection confidence (0.0-1.0)

        Returns:
            Memory ID for the persisted object
        """
        tags = self.OBJECT_TYPE_TAGS.get(object_type, ["world_object"])
        tags = list(tags)  # copy to avoid mutation

        pos = Position(x=position[0], y=position[1], z=position[2]) if position else None
        geo = GeoLocation(latitude=geo_location[0], longitude=geo_location[1],
                         altitude=geo_location[2]) if geo_location else None

        sensor_data = {}
        if confidence is not None:
            sensor_data["detection_confidence"] = confidence

        memory_id = self.memory.remember(
            content=f"World object: {name} ({object_type})",
            memory_type="discovery",
            position=pos,
            geo_location=geo,
            tags=tags + [name],
            entities=[name, object_type],
            sensor_data=sensor_data if sensor_data else None,
            metadata=metadata or {},
        )
        return memory_id

    def persist_world_object_from_proto(self, obj: WorldObject) -> str:
        """Persist a WorldObject from simulated Spot SDK types.

        For real Spot SDK protobufs, use persist_world_object_from_real_proto().
        """
        position = None
        if obj.transforms_snapshot:
            for frame_name, pose in obj.transforms_snapshot.items():
                if "body" in frame_name or "odom" in frame_name:
                    position = (pose.position.x, pose.position.y, pose.position.z)
                    break

        memory_id = self.persist_world_object(
            name=obj.name or f"object_{obj.id}",
            object_type=obj.object_type,
            position=position,
            metadata={"spot_object_id": str(obj.id)},
        )
        return memory_id

    def recall_world_objects(
        self,
        position: Optional[Tuple[float, float, float]] = None,
        geo_location: Optional[Tuple[float, float, float]] = None,
        radius: float = 10.0,
        object_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[dict]:
        """Recall persisted world objects near a position.

        Uses real Euclidean distance filtering on stored positions.
        Unlike Spot's native ListWorldObjects (15-second window), this returns
        ALL objects ever seen near this position, across all missions.
        """
        geo_filter = None
        if geo_location:
            geo_filter = GeoFilter(
                latitude=geo_location[0],
                longitude=geo_location[1],
                radius_meters=radius,
            )

        query = "world objects"
        if position:
            query = f"world objects near position {position[0]:.1f} {position[1]:.1f}"
        if object_type:
            query = f"{object_type} {query}"

        tags = ["world_object"]
        if object_type:
            tag = self.OBJECT_TYPE_TAGS.get(object_type, [object_type.lower()])
            tags = list(tag)

        # Fetch extra results for spatial post-filtering
        fetch_limit = limit * 5 if position else limit
        raw = self.memory.recall(
            query=query,
            limit=fetch_limit,
            mode="hybrid",
            geo_filter=geo_filter,
            tags=tags,
        )

        if position:
            return self._spatial_post_filter(raw, position, radius, limit)
        return raw[:limit]

    # =========================================================================
    # Waypoint persistence (semantic annotations on GraphNav)
    # =========================================================================

    def annotate_waypoint(
        self,
        waypoint_id: str,
        label: str,
        position: Optional[Tuple[float, float, float]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Add a semantic annotation to a GraphNav waypoint.

        The Spot SDK only supports name + client_metadata on waypoints.
        This adds arbitrary semantic labels that persist across sessions.
        """
        pos = Position(x=position[0], y=position[1], z=position[2]) if position else None
        return self.memory.remember(
            content=f"Waypoint {waypoint_id}: {label}",
            memory_type="observation",
            position=pos,
            tags=["waypoint", waypoint_id, "annotation"],
            entities=[waypoint_id],
            metadata=metadata or {},
        )

    def record_waypoint_visit(
        self,
        waypoint_id: str,
        status: str = "reached",
        position: Optional[Tuple[float, float, float]] = None,
        geo_location: Optional[Tuple[float, float, float]] = None,
        sensor_data: Optional[Dict[str, float]] = None,
        environment: Optional[RobotState] = None,
    ) -> str:
        """Record a waypoint visit with sensor data and environment.

        Uses record_waypoint for the core visit, then stores sensor data
        and environment as a separate annotated memory if provided.
        """
        pos = Position(x=position[0], y=position[1], z=position[2]) if position else None
        geo = GeoLocation(latitude=geo_location[0], longitude=geo_location[1],
                         altitude=geo_location[2]) if geo_location else None

        visit_id = self.memory.record_waypoint(
            waypoint_id=waypoint_id,
            status=status,
            position=pos,
            geo_location=geo,
        )

        # Store sensor data and environment as supplementary memory
        if sensor_data or environment:
            env = self.robot_state_to_environment(environment) if environment else None
            self.memory.remember(
                content=f"Waypoint {waypoint_id} visit: {status}, sensors={sensor_data}",
                memory_type="observation",
                position=pos,
                geo_location=geo,
                sensor_data=sensor_data,
                environment=env,
                tags=["waypoint", waypoint_id, "visit"],
                entities=[waypoint_id],
            )

        return visit_id

    def recall_waypoint_history(self, waypoint_id: str, limit: int = 20) -> List[dict]:
        """Get all memories associated with a waypoint across all missions."""
        return self.memory.recall(
            query=f"waypoint {waypoint_id}",
            limit=limit,
            mode="hybrid",
            tags=["waypoint", waypoint_id],
        )

    # =========================================================================
    # Obstacle and hazard persistence
    # =========================================================================

    def record_obstacle(
        self,
        description: str,
        position: Tuple[float, float, float],
        distance: Optional[float] = None,
        confidence: Optional[float] = None,
        geo_location: Optional[Tuple[float, float, float]] = None,
    ) -> str:
        """Record an obstacle detection with spatial data."""
        pos = Position(x=position[0], y=position[1], z=position[2])
        geo = GeoLocation(latitude=geo_location[0], longitude=geo_location[1],
                         altitude=geo_location[2]) if geo_location else None
        return self.memory.record_obstacle(
            description=description,
            distance=distance,
            confidence=confidence,
            position=pos,
            geo_location=geo,
        )

    def recall_obstacles_nearby(
        self,
        position: Tuple[float, float, float],
        radius: float = 5.0,
        limit: int = 10,
        geo_location: Optional[Tuple[float, float, float]] = None,
    ) -> List[dict]:
        """Recall known obstacles near a position using Euclidean distance.

        Fetches obstacle memories, then filters by real distance to the
        stored position coordinates. Results are sorted closest-first.

        Args:
            position: Query center point (x, y, z) in meters
            radius: Maximum distance in meters
            limit: Maximum results to return
            geo_location: Optional GPS coordinates for spatial mode
        """
        geo_filter = None
        if geo_location:
            geo_filter = GeoFilter(
                latitude=geo_location[0],
                longitude=geo_location[1],
                radius_meters=radius,
            )

        # Fetch extra results for post-filtering by distance
        raw = self.memory.recall(
            query=f"obstacle near position {position[0]:.1f} {position[1]:.1f}",
            limit=limit * 5,
            mode="hybrid",
            tags=["obstacle"],
            geo_filter=geo_filter,
        )

        return self._spatial_post_filter(raw, position, radius, limit)

    # =========================================================================
    # Decision recording (for action-outcome learning)
    # =========================================================================

    def record_navigation_decision(
        self,
        description: str,
        action: str,
        state: Dict[str, str],
        outcome: NavigationFeedbackResponse,
        position: Optional[Tuple[float, float, float]] = None,
        alternatives: Optional[List[str]] = None,
    ) -> str:
        """Record a navigation decision with its outcome for learning."""
        ctx = DecisionContext(
            state=state,
            action_params={"action": action},
            alternatives=alternatives or [],
        )
        out = self.navigation_result_to_outcome(outcome)
        pos = Position(x=position[0], y=position[1], z=position[2]) if position else None

        return self.memory.record_decision(
            description=description,
            action_type="navigation",
            decision_context=ctx,
            outcome=out,
            position=pos,
        )

    def recall_past_decisions(
        self,
        action_type: str = "navigation",
        limit: int = 10,
    ) -> List[dict]:
        """Recall past navigation decisions for learning."""
        return self.memory.find_similar_decisions(
            action_type=action_type,
            max_results=limit,
        )

    # =========================================================================
    # Sensor and anomaly recording
    # =========================================================================

    def record_sensor_reading(
        self,
        sensor_name: str,
        readings: Dict[str, float],
        position: Optional[Tuple[float, float, float]] = None,
        is_anomaly: bool = False,
    ) -> str:
        """Record sensor readings with optional anomaly flag."""
        pos = Position(x=position[0], y=position[1], z=position[2]) if position else None
        return self.memory.record_sensor(
            sensor_name=sensor_name,
            readings=readings,
            is_anomaly=is_anomaly,
            position=pos,
        )

    def record_failure(
        self,
        description: str,
        severity: str = "error",
        root_cause: Optional[str] = None,
        recovery_action: Optional[str] = None,
        position: Optional[Tuple[float, float, float]] = None,
    ) -> str:
        """Record a failure event for future avoidance."""
        pos = Position(x=position[0], y=position[1], z=position[2]) if position else None
        return self.memory.record_failure(
            description=description,
            severity=severity,
            root_cause=root_cause,
            recovery_action=recovery_action,
            position=pos,
        )

    # =========================================================================
    # Area Callback support
    # =========================================================================

    def record_area_callback_event(
        self,
        region_id: str,
        stage: str,
        action_taken: str,
        outcome_type: str = "success",
        position: Optional[Tuple[float, float, float]] = None,
    ) -> str:
        """Record an Area Callback event for future region-based decisions."""
        ctx = DecisionContext(
            state={"region": region_id, "stage": stage},
            action_params={"action": action_taken},
        )
        out = Outcome(outcome_type=outcome_type)
        pos = Position(x=position[0], y=position[1], z=position[2]) if position else None

        return self.memory.record_decision(
            description=f"Area callback: region {region_id}, stage {stage}, action {action_taken}",
            action_type="area_callback",
            decision_context=ctx,
            outcome=out,
            position=pos,
        )

    def recall_region_history(self, region_id: str, limit: int = 10) -> List[dict]:
        """Recall all past events in a specific region."""
        return self.memory.recall(
            query=f"area callback region {region_id}",
            limit=limit,
            mode="hybrid",
            tags=["decision"],
        )

    # =========================================================================
    # Utility
    # =========================================================================

    def flush(self) -> None:
        """Persist all in-memory data to disk."""
        self.memory.flush()

    def get_stats(self) -> Dict[str, int]:
        """Get memory system statistics."""
        return self.memory.get_stats()
