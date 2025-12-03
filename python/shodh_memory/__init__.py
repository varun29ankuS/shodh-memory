"""
Shodh-Memory: AI Memory System for Autonomous Robots & Drones

Native Python bindings for high-performance memory operations
optimized for robotics, drones, and offline AI applications.

Features:
- Position(x, y, z) - Local robot coordinates in meters
- GeoLocation(lat, lon, alt) - GPS for drones & outdoor robots
- GeoFilter - Spatial queries by radius
- DecisionContext - For action-outcome learning (what conditions -> what action)
- Outcome - Result of decisions (success/failure/partial + reward signal)
- Environment - Weather, terrain, lighting, nearby agents
- Failure tracking - Severity, root cause, recovery actions
- Anomaly detection - Track unusual sensor readings
- Pattern learning - Match situations to learned patterns
- 100% offline operation - No cloud, no API keys
"""

from .shodh_memory import (
    MemorySystem,
    # Location types
    Position,
    GeoLocation,
    GeoFilter,
    # Decision & Learning types
    DecisionContext,
    Outcome,
    Environment,
    # Version
    __version__,
)

__all__ = [
    # Core
    "MemorySystem",
    # Location types
    "Position",
    "GeoLocation",
    "GeoFilter",
    # Decision & Learning types
    "DecisionContext",
    "Outcome",
    "Environment",
    # Version
    "__version__",
]
