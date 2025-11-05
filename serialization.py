"""
Serialization utilities for World objects.

Provides JSON/YAML serialization with complete metadata and reproducibility.
Enables world saving for later analysis, replay, and training dataset management.

Philosophy: Worlds are fully reconstructable from saved files.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from uuid import UUID
import json
import yaml
import numpy as np

from dataclasses_core import World, Zone, Road, Obstacle
from dataclasses_core import TerrainType, RoadType, ObstacleType


# ==============================================================================
# SERIALIZATION METADATA
# ==============================================================================

@dataclass
class WorldMetadata:
    """Metadata for serialized world."""
    
    version: str = "1.0"                          # Schema version
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    seed: Optional[int] = None                    # Generation seed (if procedural)
    generator_config: Optional[Dict[str, Any]] = None  # Generator config
    world_name: str = "Generated World"
    world_width: int = 1200
    world_height: int = 800
    entity_counts: Dict[str, int] = field(default_factory=dict)
    notes: str = ""                               # User notes
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "WorldMetadata":
        """Create from dictionary."""
        return WorldMetadata(**data)


# ==============================================================================
# SERIALIZATION CONVERTERS
# ==============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder supporting numpy types."""
    
    def default(self, obj: Any) -> Any:
        """Handle numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, (TerrainType, RoadType, ObstacleType)):
            return obj.name
        return super().default(obj)


def serialize_zone(zone: Zone) -> Dict[str, Any]:
    """Serialize Zone to dictionary."""
    return {
        'zone_id': str(zone.zone_id),
        'name': zone.name,
        'position': zone.position.tolist(),
        'width': float(zone.width),
        'height': float(zone.height),
        'terrain_type': zone.terrain_type.name,
        'speed_limit_kmh': float(zone.speed_limit_kmh),
        'polygon_vertices': zone.polygon_vertices,
        'metadata': zone.metadata,
    }


def deserialize_zone(data: Dict[str, Any]) -> Zone:
    """Deserialize Zone from dictionary."""
    return Zone(
        zone_id=UUID(data['zone_id']),
        name=data['name'],
        position=np.array(data['position'], dtype=np.float32),
        width=float(data['width']),
        height=float(data['height']),
        terrain_type=TerrainType[data['terrain_type']],
        speed_limit_kmh=float(data['speed_limit_kmh']),
        polygon_vertices=data.get('polygon_vertices'),
        metadata=data.get('metadata', {}),
    )


def serialize_road(road: Road) -> Dict[str, Any]:
    """Serialize Road to dictionary."""
    return {
        'road_id': str(road.road_id),
        'name': road.name,
        'start_point': road.start_point.tolist(),
        'end_point': road.end_point.tolist(),
        'road_type': road.road_type.name,
        'lanes': int(road.lanes),
        'width': float(road.width),
        'speed_limit_kmh': float(road.speed_limit_kmh),
        'is_bidirectional': bool(road.is_bidirectional),
        'is_one_way': bool(road.is_one_way),
        'metadata': road.metadata,
    }


def deserialize_road(data: Dict[str, Any]) -> Road:
    """Deserialize Road from dictionary."""
    return Road(
        road_id=UUID(data['road_id']),
        name=data['name'],
        start_point=np.array(data['start_point'], dtype=np.float32),
        end_point=np.array(data['end_point'], dtype=np.float32),
        road_type=RoadType[data['road_type']],
        lanes=int(data['lanes']),
        width=float(data['width']),
        speed_limit_kmh=float(data['speed_limit_kmh']),
        is_bidirectional=bool(data.get('is_bidirectional', True)),
        is_one_way=bool(data.get('is_one_way', False)),
        metadata=data.get('metadata', {}),
    )


def serialize_obstacle(obstacle: Obstacle) -> Dict[str, Any]:
    """Serialize Obstacle to dictionary."""
    return {
        'obstacle_id': str(obstacle.obstacle_id),
        'name': obstacle.name,
        'position': obstacle.position.tolist(),
        'width': float(obstacle.width),
        'height': float(obstacle.height),
        'obstacle_type': obstacle.obstacle_type.name,
        'is_passable': bool(obstacle.is_passable),
        'damage_on_collision': float(obstacle.damage_on_collision),
        'rotation_degrees': float(obstacle.rotation_degrees),
        'metadata': obstacle.metadata,
    }


def deserialize_obstacle(data: Dict[str, Any]) -> Obstacle:
    """Deserialize Obstacle from dictionary."""
    return Obstacle(
        obstacle_id=UUID(data['obstacle_id']),
        name=data['name'],
        position=np.array(data['position'], dtype=np.float32),
        width=float(data['width']),
        height=float(data['height']),
        obstacle_type=ObstacleType[data['obstacle_type']],
        is_passable=bool(data.get('is_passable', False)),
        damage_on_collision=float(data['damage_on_collision']),
        rotation_degrees=float(data.get('rotation_degrees', 0.0)),
        metadata=data.get('metadata', {}),
    )


# ==============================================================================
# WORLD SERIALIZATION
# ==============================================================================

class WorldSerializer:
    """Handles serialization and deserialization of World objects."""
    
    @staticmethod
    def serialize_world(world: World, metadata: Optional[WorldMetadata] = None) -> Dict[str, Any]:
        """
        Serialize world to dictionary.
        
        Args:
            world: World to serialize
            metadata: Optional metadata
        
        Returns:
            Serialized world dictionary
        """
        if metadata is None:
            metadata = WorldMetadata(
                world_name=world.name,
                world_width=world.width,
                world_height=world.height,
                entity_counts=world.get_entity_count(),
            )
        
        return {
            'metadata': metadata.to_dict(),
            'world': {
                'world_id': str(world.world_id),
                'name': world.name,
                'width': int(world.width),
                'height': int(world.height),
                'metadata': world.metadata,
            },
            'zones': [serialize_zone(zone) for zone in world.zones.values()],
            'roads': [serialize_road(road) for road in world.roads.values()],
            'obstacles': [serialize_obstacle(obs) for obs in world.obstacles.values()],
        }
    
    @staticmethod
    def deserialize_world(data: Dict[str, Any]) -> Tuple[World, WorldMetadata]:
        """
        Deserialize world from dictionary.
        
        Args:
            data: Serialized world dictionary
        
        Returns:
            (World, Metadata) tuple
        """
        # Extract metadata
        metadata = WorldMetadata.from_dict(data['metadata'])
        
        # Create world
        world_data = data['world']
        world = World(
            world_id=UUID(world_data['world_id']),
            name=world_data['name'],
            width=world_data['width'],
            height=world_data['height'],
            metadata=world_data.get('metadata', {}),
        )
        
        # Add zones
        for zone_data in data.get('zones', []):
            zone = deserialize_zone(zone_data)
            world.add_zone(zone)
        
        # Add roads
        for road_data in data.get('roads', []):
            road = deserialize_road(road_data)
            world.add_road(road)
        
        # Add obstacles
        for obs_data in data.get('obstacles', []):
            obstacle = deserialize_obstacle(obs_data)
            world.add_obstacle(obstacle)
        
        return world, metadata
    
    @staticmethod
    def to_json(world: World, metadata: Optional[WorldMetadata] = None, indent: int = 2) -> str:
        """
        Serialize world to JSON string.
        
        Args:
            world: World to serialize
            metadata: Optional metadata
            indent: JSON indentation
        
        Returns:
            JSON string
        """
        data = WorldSerializer.serialize_world(world, metadata)
        return json.dumps(data, cls=NumpyEncoder, indent=indent)
    
    @staticmethod
    def from_json(json_str: str) -> Tuple[World, WorldMetadata]:
        """
        Deserialize world from JSON string.
        
        Args:
            json_str: JSON string
        
        Returns:
            (World, Metadata) tuple
        """
        data = json.loads(json_str)
        return WorldSerializer.deserialize_world(data)
    
    @staticmethod
    def to_yaml(world: World, metadata: Optional[WorldMetadata] = None) -> str:
        """
        Serialize world to YAML string.
        
        Args:
            world: World to serialize
            metadata: Optional metadata
        
        Returns:
            YAML string
        """
        data = WorldSerializer.serialize_world(world, metadata)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def from_yaml(yaml_str: str) -> Tuple[World, WorldMetadata]:
        """
        Deserialize world from YAML string.
        
        Args:
            yaml_str: YAML string
        
        Returns:
            (World, Metadata) tuple
        """
        data = yaml.safe_load(yaml_str)
        return WorldSerializer.deserialize_world(data)


# ==============================================================================
# FILE I/O UTILITIES
# ==============================================================================

class WorldFileIO:
    """Handle file operations for world serialization."""
    
    @staticmethod
    def save_json(world: World, path: str, metadata: Optional[WorldMetadata] = None) -> None:
        """
        Save world to JSON file.
        
        Args:
            world: World to save
            path: File path
            metadata: Optional metadata
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        json_str = WorldSerializer.to_json(world, metadata)
        
        with open(path_obj, 'w') as f:
            f.write(json_str)
    
    @staticmethod
    def load_json(path: str) -> Tuple[World, WorldMetadata]:
        """
        Load world from JSON file.
        
        Args:
            path: File path
        
        Returns:
            (World, Metadata) tuple
        """
        with open(path, 'r') as f:
            json_str = f.read()
        
        return WorldSerializer.from_json(json_str)
    
    @staticmethod
    def save_yaml(world: World, path: str, metadata: Optional[WorldMetadata] = None) -> None:
        """
        Save world to YAML file.
        
        Args:
            world: World to save
            path: File path
            metadata: Optional metadata
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        yaml_str = WorldSerializer.to_yaml(world, metadata)
        
        with open(path_obj, 'w') as f:
            f.write(yaml_str)
    
    @staticmethod
    def load_yaml(path: str) -> Tuple[World, WorldMetadata]:
        """
        Load world from YAML file.
        
        Args:
            path: File path
        
        Returns:
            (World, Metadata) tuple
        """
        with open(path, 'r') as f:
            yaml_str = f.read()
        
        return WorldSerializer.from_yaml(yaml_str)
    
    @staticmethod
    def save(world: World, path: str, format: str = "json", metadata: Optional[WorldMetadata] = None) -> None:
        """
        Save world to file (auto-detect format or specify).
        
        Args:
            world: World to save
            path: File path
            format: "json" or "yaml" (auto-detected from extension if not specified)
            metadata: Optional metadata
        """
        path_obj = Path(path)
        
        # Auto-detect format from extension
        if format == "auto":
            ext = path_obj.suffix.lower()
            if ext in ['.yaml', '.yml']:
                format = "yaml"
            else:
                format = "json"
        
        if format == "yaml":
            WorldFileIO.save_yaml(world, path, metadata)
        else:
            WorldFileIO.save_json(world, path, metadata)
    
    @staticmethod
    def load(path: str, format: str = "auto") -> Tuple[World, WorldMetadata]:
        """
        Load world from file (auto-detect format or specify).
        
        Args:
            path: File path
            format: "json" or "yaml" (auto-detected from extension if not specified)
        
        Returns:
            (World, Metadata) tuple
        """
        path_obj = Path(path)
        
        # Auto-detect format from extension
        if format == "auto":
            ext = path_obj.suffix.lower()
            if ext in ['.yaml', '.yml']:
                format = "yaml"
            else:
                format = "json"
        
        if format == "yaml":
            return WorldFileIO.load_yaml(path)
        else:
            return WorldFileIO.load_json(path)


# ==============================================================================
# WORLD EXTENSION METHODS
# ==============================================================================

def add_serialization_methods(world_class: type) -> None:
    """
    Add serialization methods to World class.
    
    This injects save/load methods into the World class.
    """
    
    def save(self, path: str, format: str = "auto", seed: Optional[int] = None) -> None:
        """Save world to file."""
        metadata = WorldMetadata(
            world_name=self.name,
            world_width=self.width,
            world_height=self.height,
            entity_counts=self.get_entity_count(),
            seed=seed,
        )
        WorldFileIO.save(self, path, format, metadata)
    
    @staticmethod
    def load(path: str, format: str = "auto") -> Tuple[World, WorldMetadata]:
        """Load world from file."""
        return WorldFileIO.load(path, format)
    
    def to_json_str(self, seed: Optional[int] = None) -> str:
        """Convert world to JSON string."""
        metadata = WorldMetadata(
            world_name=self.name,
            world_width=self.width,
            world_height=self.height,
            entity_counts=self.get_entity_count(),
            seed=seed,
        )
        return WorldSerializer.to_json(self, metadata)
    
    def to_yaml_str(self, seed: Optional[int] = None) -> str:
        """Convert world to YAML string."""
        metadata = WorldMetadata(
            world_name=self.name,
            world_width=self.width,
            world_height=self.height,
            entity_counts=self.get_entity_count(),
            seed=seed,
        )
        return WorldSerializer.to_yaml(self, metadata)
    
    # Attach methods to class
    world_class.save = save
    world_class.load = load
    world_class.to_json_str = to_json_str
    world_class.to_yaml_str = to_yaml_str


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def create_world_metadata(
    world: World,
    seed: Optional[int] = None,
    generator_config: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> WorldMetadata:
    """
    Create metadata for a world.
    
    Args:
        world: World instance
        seed: Generation seed
        generator_config: Generator configuration dict
        notes: User notes
    
    Returns:
        WorldMetadata instance
    """
    return WorldMetadata(
        world_name=world.name,
        world_width=world.width,
        world_height=world.height,
        entity_counts=world.get_entity_count(),
        seed=seed,
        generator_config=generator_config,
        notes=notes,
    )


def estimate_serialization_size(world: World) -> Dict[str, int]:
    """
    Estimate serialization size.
    
    Args:
        world: World instance
    
    Returns:
        Dictionary with size estimates
    """
    json_str = WorldSerializer.to_json(world)
    yaml_str = WorldSerializer.to_yaml(world)
    
    return {
        'json_bytes': len(json_str.encode('utf-8')),
        'json_mb': len(json_str.encode('utf-8')) / (1024 * 1024),
        'yaml_bytes': len(yaml_str.encode('utf-8')),
        'yaml_mb': len(yaml_str.encode('utf-8')) / (1024 * 1024),
        'num_zones': len(world.zones),
        'num_roads': len(world.roads),
        'num_obstacles': len(world.obstacles),
    }


# Initialize serialization methods on World class
add_serialization_methods(World)
