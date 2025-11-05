"""Core data classes for Zone, Obstacle, Road, and World entities.

This module defines the fundamental data structures for the 2D driving simulation.
All classes use type-safe dataclasses with semantic properties for RL integration.

No physics or rendering logic - pure data representation and management.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set
from uuid import UUID, uuid4
import numpy as np


# ==============================================================================
# ENUMERATIONS & TYPE DEFINITIONS
# ==============================================================================

class TerrainType(Enum):
    """Terrain classification for zones."""
    ASPHALT = auto()      # Drivable surface with high grip
    GRASS = auto()        # Natural terrain, lower friction
    WATER = auto()        # Impassable obstacle
    GRAVEL = auto()        # Drivable but slippery
    CONCRETE = auto()     # Alternative drivable surface
    SAND = auto()         # Drivable with very low friction


class RoadType(Enum):
    """Classification for road segments."""
    MAIN_STREET = auto()  # Primary route
    SECONDARY = auto()    # Secondary route
    RESIDENTIAL = auto()  # Low-speed residential street
    HIGHWAY = auto()      # High-speed route
    PARKING = auto()      # Parking area


class ObstacleType(Enum):
    """Classification for static obstacles."""
    WALL = auto()         # Hard barrier
    TREE = auto()         # Natural obstacle
    ROCK = auto()         # Large rock/boulder
    BUILDING = auto()     # Building structure
    FENCE = auto()        # Fencing/barrier
    CAR = auto()          # Parked vehicle
    DEBRIS = auto()       # Generic debris


# ==============================================================================
# SEMANTIC PROPERTIES (Immutable configurations)
# ==============================================================================

@dataclass(frozen=True)
class TerrainProperties:
    """Physical properties associated with each terrain type."""
    friction_coefficient: float      # [0.1, 1.0] - affects grip
    max_speed_kmh: float            # km/h - speed limit for terrain
    damage_per_second: float        # Damage accumulated on this terrain
    traversability: float            # [0, 1] - 0=impassable, 1=fully passable
    slip_factor: float              # [0, 1] - 0=no slip, 1=full drift
    
    def __post_init__(self) -> None:
        """Validate properties are in valid ranges."""
        assert 0 <= self.friction_coefficient <= 1.0, "friction must be [0, 1]"
        assert self.max_speed_kmh > 0, "max_speed must be positive"
        assert self.damage_per_second >= 0, "damage cannot be negative"
        assert 0 <= self.traversability <= 1.0, "traversability must be [0, 1]"
        assert 0 <= self.slip_factor <= 1.0, "slip_factor must be [0, 1]"


# Predefined terrain properties (semantic mappings)
TERRAIN_PROPERTIES: Dict[TerrainType, TerrainProperties] = {
    TerrainType.ASPHALT: TerrainProperties(
        friction_coefficient=0.95,
        max_speed_kmh=150.0,
        damage_per_second=0.0,
        traversability=1.0,
        slip_factor=0.0
    ),
    TerrainType.GRASS: TerrainProperties(
        friction_coefficient=0.65,
        max_speed_kmh=60.0,
        damage_per_second=0.5,
        traversability=1.0,
        slip_factor=0.15
    ),
    TerrainType.WATER: TerrainProperties(
        friction_coefficient=0.2,
        max_speed_kmh=30.0,
        damage_per_second=5.0,
        traversability=0.0,
        slip_factor=1.0
    ),
    TerrainType.GRAVEL: TerrainProperties(
        friction_coefficient=0.65,
        max_speed_kmh=80.0,
        damage_per_second=1.0,
        traversability=1.0,
        slip_factor=0.3
    ),
    TerrainType.CONCRETE: TerrainProperties(
        friction_coefficient=0.9,
        max_speed_kmh=120.0,
        damage_per_second=0.0,
        traversability=1.0,
        slip_factor=0.05
    ),
    TerrainType.SAND: TerrainProperties(
        friction_coefficient=0.4,
        max_speed_kmh=40.0,
        damage_per_second=2.0,
        traversability=1.0,
        slip_factor=0.5
    ),
}


# ==============================================================================
# ZONE DATA CLASS
# ==============================================================================

@dataclass
class Zone:
    """
    Semantic region with environmental properties.
    
    A zone represents a contiguous area of the world with consistent properties
    like terrain type, friction, and speed limits. Used for RL reward shaping
    and vehicle physics calculations.
    
    Attributes:
        zone_id: Unique identifier (auto-generated UUID)
        name: Human-readable name
        position: [x, y] center position in world coordinates
        width: Zone width in pixels
        height: Zone height in pixels
        terrain_type: Classification of terrain
        speed_limit_kmh: Maximum legal speed in this zone
        polygon_vertices: Optional list of vertices for non-rectangular zones
        metadata: Custom key-value attributes for extensibility
    """
    
    zone_id: UUID = field(default_factory=uuid4)
    name: str = field(default="unnamed_zone")
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    width: float = 100.0
    height: float = 100.0
    terrain_type: TerrainType = TerrainType.ASPHALT
    speed_limit_kmh: float = 80.0
    polygon_vertices: Optional[List[Tuple[float, float]]] = None
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate zone properties."""
        # Ensure position is numpy array
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float32)
        
        assert self.width > 0, "Zone width must be positive"
        assert self.height > 0, "Zone height must be positive"
        assert self.speed_limit_kmh > 0, "Speed limit must be positive"
    
    def get_terrain_properties(self) -> TerrainProperties:
        """Get semantic terrain properties for this zone."""
        return TERRAIN_PROPERTIES[self.terrain_type]
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) in world coordinates."""
        x, y = self.position
        return (
            x - self.width / 2,
            y - self.height / 2,
            x + self.width / 2,
            y + self.height / 2
        )
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is within zone bounds (AABB check)."""
        x_min, y_min, x_max, y_max = self.get_bounds()
        px, py = point[0], point[1]
        return x_min <= px <= x_max and y_min <= py <= y_max
    
    def get_semantic_data(self) -> Dict[str, any]:
        """Return all semantic data for RL observations."""
        props = self.get_terrain_properties()
        return {
            'zone_id': str(self.zone_id),
            'name': self.name,
            'terrain_type': self.terrain_type.name,
            'speed_limit_kmh': self.speed_limit_kmh,
            'friction': props.friction_coefficient,
            'damage_per_second': props.damage_per_second,
            'slip_factor': props.slip_factor,
            'position': self.position.tolist(),
        }


# ==============================================================================
# ROAD DATA CLASS
# ==============================================================================

@dataclass
class Road:
    """
    Road segment with traffic rules and topology.
    
    Represents a drivable path with direction, lanes, and semantic properties.
    Roads connect the world and can be used to guide procedural generation
    or provide rewards for following lanes.
    
    Attributes:
        road_id: Unique identifier
        name: Human-readable name
        start_point: [x, y] starting position
        end_point: [x, y] ending position
        road_type: Classification (main, secondary, etc.)
        lanes: Number of lanes
        width: Road width in pixels
        speed_limit_kmh: Speed limit for this road
        is_bidirectional: Whether traffic flows both directions
        is_one_way: Direction of one-way traffic (if applicable)
        metadata: Custom attributes
    """
    
    road_id: UUID = field(default_factory=uuid4)
    name: str = field(default="unnamed_road")
    start_point: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    end_point: np.ndarray = field(default_factory=lambda: np.array([100.0, 100.0]))
    road_type: RoadType = RoadType.MAIN_STREET
    lanes: int = 2
    width: float = 40.0
    speed_limit_kmh: float = 80.0
    is_bidirectional: bool = True
    is_one_way: bool = False
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate road properties."""
        # Ensure endpoints are numpy arrays
        if not isinstance(self.start_point, np.ndarray):
            self.start_point = np.array(self.start_point, dtype=np.float32)
        if not isinstance(self.end_point, np.ndarray):
            self.end_point = np.array(self.end_point, dtype=np.float32)
        
        assert self.lanes > 0, "Must have at least 1 lane"
        assert self.width > 0, "Width must be positive"
        assert self.speed_limit_kmh > 0, "Speed limit must be positive"
        assert not (self.is_bidirectional and self.is_one_way), \
            "Road cannot be both bidirectional and one-way"
    
    @property
    def length(self) -> float:
        """Calculate road segment length."""
        return float(np.linalg.norm(self.end_point - self.start_point))
    
    @property
    def center(self) -> np.ndarray:
        """Calculate midpoint of road."""
        return (self.start_point + self.end_point) / 2.0
    
    @property
    def direction(self) -> np.ndarray:
        """Get normalized direction vector from start to end."""
        delta = self.end_point - self.start_point
        length = np.linalg.norm(delta)
        if length == 0:
            return np.array([1.0, 0.0])
        return delta / length
    
    @property
    def heading_radians(self) -> float:
        """Get heading angle in radians (0 = east, π/2 = north)."""
        direction = self.direction
        return float(np.arctan2(direction[1], direction[0]))
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Return bounding box (x_min, y_min, x_max, y_max)."""
        xs = [self.start_point[0], self.end_point[0]]
        ys = [self.start_point[1], self.end_point[1]]
        margin = self.width / 2
        return (min(xs) - margin, min(ys) - margin, 
                max(xs) + margin, max(ys) + margin)
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """Calculate perpendicular distance from point to road centerline."""
        # Project point onto line segment
        pa = point - self.start_point
        ab = self.end_point - self.start_point
        ab_squared = np.dot(ab, ab)
        
        if ab_squared == 0:
            return float(np.linalg.norm(pa))
        
        t = np.clip(np.dot(pa, ab) / ab_squared, 0, 1)
        projection = self.start_point + t * ab
        return float(np.linalg.norm(point - projection))
    
    def is_point_on_road(self, point: np.ndarray, tolerance: float = 0.0) -> bool:
        """Check if point is on/near the road."""
        distance = self.distance_to_point(point)
        return distance <= (self.width / 2 + tolerance)
    
    def get_semantic_data(self) -> Dict[str, any]:
        """Return all semantic data for RL observations."""
        return {
            'road_id': str(self.road_id),
            'name': self.name,
            'road_type': self.road_type.name,
            'length': self.length,
            'lanes': self.lanes,
            'speed_limit_kmh': self.speed_limit_kmh,
            'center': self.center.tolist(),
            'heading_radians': self.heading_radians,
            'is_bidirectional': self.is_bidirectional,
        }


# ==============================================================================
# OBSTACLE DATA CLASS
# ==============================================================================

@dataclass
class Obstacle:
    """
    Static obstacle in the world.
    
    Represents impassable objects that vehicles must navigate around.
    Used for collision penalties and navigation challenges in RL.
    
    Attributes:
        obstacle_id: Unique identifier
        name: Human-readable name
        position: [x, y] center position
        width: Width in pixels
        height: Height in pixels
        obstacle_type: Classification
        is_passable: Whether vehicle can pass through
        damage_on_collision: Penalty for hitting obstacle
        rotation_degrees: Rotation angle for oriented obstacles
        metadata: Custom attributes
    """
    
    obstacle_id: UUID = field(default_factory=uuid4)
    name: str = field(default="unnamed_obstacle")
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    width: float = 50.0
    height: float = 50.0
    obstacle_type: ObstacleType = ObstacleType.WALL
    is_passable: bool = False
    damage_on_collision: float = 10.0
    rotation_degrees: float = 0.0
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate obstacle properties."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float32)
        
        assert self.width > 0, "Width must be positive"
        assert self.height > 0, "Height must be positive"
        assert self.damage_on_collision >= 0, "Damage cannot be negative"
        assert 0 <= self.rotation_degrees < 360, "Rotation must be [0, 360)"
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Return axis-aligned bounding box (no rotation considered)."""
        x, y = self.position
        return (
            x - self.width / 2,
            y - self.height / 2,
            x + self.width / 2,
            y + self.height / 2
        )
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is within obstacle bounds (AABB check)."""
        x_min, y_min, x_max, y_max = self.get_bounds()
        px, py = point[0], point[1]
        return x_min <= px <= x_max and y_min <= py <= y_max
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """Calculate minimum distance from point to obstacle."""
        x_min, y_min, x_max, y_max = self.get_bounds()
        px, py = point[0], point[1]
        
        # Clamp point to rectangle bounds
        closest_x = np.clip(px, x_min, x_max)
        closest_y = np.clip(py, y_min, y_max)
        
        return float(np.sqrt((px - closest_x)**2 + (py - closest_y)**2))
    
    def get_semantic_data(self) -> Dict[str, any]:
        """Return all semantic data."""
        return {
            'obstacle_id': str(self.obstacle_id),
            'name': self.name,
            'obstacle_type': self.obstacle_type.name,
            'passable': self.is_passable,
            'damage_on_collision': self.damage_on_collision,
            'position': self.position.tolist(),
        }


# ==============================================================================
# WORLD DATA CLASS
# ==============================================================================

@dataclass
class World:
    """
    Container and manager for all world entities.
    
    Provides efficient lookup, spatial queries, and management of zones,
    roads, and obstacles. Forms the core of the simulation world.
    
    Attributes:
        world_id: Unique identifier
        name: World name
        width: World width in pixels
        height: World height in pixels
        zones: Mapping of zone_id -> Zone
        roads: Mapping of road_id -> Road
        obstacles: Mapping of obstacle_id -> Obstacle
        metadata: Custom world attributes
    """
    
    world_id: UUID = field(default_factory=uuid4)
    name: str = field(default="unnamed_world")
    width: int = 1200
    height: int = 800
    zones: Dict[UUID, Zone] = field(default_factory=dict)
    roads: Dict[UUID, Road] = field(default_factory=dict)
    obstacles: Dict[UUID, Obstacle] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate world properties."""
        assert self.width > 0, "World width must be positive"
        assert self.height > 0, "World height must be positive"
    
    # ========================================================================
    # ZONE MANAGEMENT
    # ========================================================================
    
    def add_zone(self, zone: Zone) -> UUID:
        """
        Add zone to world.
        
        Args:
            zone: Zone to add
        
        Returns:
            Zone ID for future reference
        """
        self.zones[zone.zone_id] = zone
        return zone.zone_id
    
    def remove_zone(self, zone_id: UUID) -> bool:
        """Remove zone by ID. Returns True if found and removed."""
        if zone_id in self.zones:
            del self.zones[zone_id]
            return True
        return False
    
    def get_zone(self, zone_id: UUID) -> Optional[Zone]:
        """Get zone by ID."""
        return self.zones.get(zone_id)
    
    def find_zones_at_point(self, point: np.ndarray) -> List[Zone]:
        """Find all zones containing a point."""
        return [z for z in self.zones.values() if z.contains_point(point)]
    
    def find_zones_in_radius(self, point: np.ndarray, radius: float) -> List[Zone]:
        """Find all zones within radius of point."""
        result = []
        for zone in self.zones.values():
            # Check if zone bounds intersect circle
            dx = np.clip(point[0], zone.get_bounds()[0], zone.get_bounds()[2]) - point[0]
            dy = np.clip(point[1], zone.get_bounds()[1], zone.get_bounds()[3]) - point[1]
            distance = np.sqrt(dx*dx + dy*dy)
            if distance <= radius:
                result.append(zone)
        return result
    
    # ========================================================================
    # ROAD MANAGEMENT
    # ========================================================================
    
    def add_road(self, road: Road) -> UUID:
        """Add road to world."""
        self.roads[road.road_id] = road
        return road.road_id
    
    def remove_road(self, road_id: UUID) -> bool:
        """Remove road by ID."""
        if road_id in self.roads:
            del self.roads[road_id]
            return True
        return False
    
    def get_road(self, road_id: UUID) -> Optional[Road]:
        """Get road by ID."""
        return self.roads.get(road_id)
    
    def find_nearby_roads(self, point: np.ndarray, radius: float) -> List[Road]:
        """Find roads within radius of point."""
        result = []
        for road in self.roads.values():
            distance = road.distance_to_point(point)
            if distance <= radius:
                result.append(road)
        return sorted(result, key=lambda r: road.distance_to_point(point))
    
    def find_roads_by_type(self, road_type: RoadType) -> List[Road]:
        """Find all roads of specific type."""
        return [r for r in self.roads.values() if r.road_type == road_type]
    
    # ========================================================================
    # OBSTACLE MANAGEMENT
    # ========================================================================
    
    def add_obstacle(self, obstacle: Obstacle) -> UUID:
        """Add obstacle to world."""
        self.obstacles[obstacle.obstacle_id] = obstacle
        return obstacle.obstacle_id
    
    def remove_obstacle(self, obstacle_id: UUID) -> bool:
        """Remove obstacle by ID."""
        if obstacle_id in self.obstacles:
            del self.obstacles[obstacle_id]
            return True
        return False
    
    def get_obstacle(self, obstacle_id: UUID) -> Optional[Obstacle]:
        """Get obstacle by ID."""
        return self.obstacles.get(obstacle_id)
    
    def find_obstacles_near_point(self, point: np.ndarray, radius: float) -> List[Obstacle]:
        """Find obstacles within radius of point."""
        result = []
        for obstacle in self.obstacles.values():
            distance = obstacle.distance_to_point(point)
            if distance <= radius:
                result.append(obstacle)
        return sorted(result, key=lambda o: obstacle.distance_to_point(point))
    
    def find_obstacles_by_type(self, obstacle_type: ObstacleType) -> List[Obstacle]:
        """Find all obstacles of specific type."""
        return [o for o in self.obstacles.values() if o.obstacle_type == obstacle_type]
    
    # ========================================================================
    # GLOBAL QUERIES
    # ========================================================================
    
    def get_world_bounds(self) -> Tuple[int, int, int, int]:
        """Return world bounds (0, 0, width, height)."""
        return (0, 0, self.width, self.height)
    
    def point_in_world(self, point: np.ndarray) -> bool:
        """Check if point is within world bounds."""
        return (0 <= point[0] <= self.width and 
                0 <= point[1] <= self.height)
    
    def get_entity_count(self) -> Dict[str, int]:
        """Get counts of all entities."""
        return {
            'zones': len(self.zones),
            'roads': len(self.roads),
            'obstacles': len(self.obstacles),
            'total': len(self.zones) + len(self.roads) + len(self.obstacles)
        }
    
    def get_all_entities_semantic_data(self) -> Dict[str, List[Dict]]:
        """Get semantic data for all entities (useful for RL observations)."""
        return {
            'zones': [z.get_semantic_data() for z in self.zones.values()],
            'roads': [r.get_semantic_data() for r in self.roads.values()],
            'obstacles': [o.get_semantic_data() for o in self.obstacles.values()],
        }
    
    def clear(self) -> None:
        """Remove all entities from world."""
        self.zones.clear()
        self.roads.clear()
        self.obstacles.clear()
    
    def summary(self) -> str:
        """Generate world summary for debugging."""
        counts = self.get_entity_count()
        return (f"World '{self.name}' ({self.width}x{self.height}px) - "
                f"Zones: {counts['zones']}, Roads: {counts['roads']}, "
                f"Obstacles: {counts['obstacles']}")
