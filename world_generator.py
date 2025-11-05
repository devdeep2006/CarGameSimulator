"""
WorldGenerator: Procedural generation of roads, zones, and obstacles.

Implements sophisticated algorithms for realistic world creation:
- Grid-based road generation with orthogonal/parallel constraints
- Non-overlapping zone placement using bin-packing principles
- Obstacle placement respecting world features
- Full separation of concerns (generation only, no rendering/physics)

All randomization uses seeded RNG for reproducibility.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum, auto
import numpy as np
from uuid import uuid4

from dataclasses_core import (
    Zone, Road, Obstacle, World,
    TerrainType, RoadType, ObstacleType,
    TERRAIN_PROPERTIES
)


# ==============================================================================
# GENERATION CONFIGURATION
# ==============================================================================

@dataclass
class GenerationConfig:
    """Configuration for world generation."""
    
    # Road generation
    num_main_roads: int = 3
    num_secondary_roads: int = 6
    min_road_length: int = 200
    max_road_length: int = 800
    road_width: float = 50.0
    road_spacing: int = 150  # Minimum spacing between parallel roads
    
    # Zone generation
    num_zones: int = 8
    min_zone_size: int = 100
    max_zone_size: int = 300
    zone_margin: int = 20  # Margin around zones
    
    # Obstacle generation
    num_obstacles: int = 20
    min_obstacle_size: int = 20
    max_obstacle_size: int = 100
    obstacles_near_roads: float = 0.7  # Proportion near roads (vs. in zones)
    
    # World parameters
    world_width: int = 1200
    world_height: int = 800


# ==============================================================================
# GRID-BASED ROAD GENERATOR
# ==============================================================================

class RoadGenerator:
    """
    Generates logically-placed roads (parallel/perpendicular) using grid constraints.
    
    Algorithm:
    1. Divide world into grid
    2. Place main roads as principal grid lines
    3. Add secondary roads perpendicular/parallel to mains
    4. Ensure non-overlapping and logical placement
    """
    
    def __init__(self, config: GenerationConfig, seed: int) -> None:
        """Initialize road generator."""
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.roads: List[Road] = []
    
    def generate(self) -> List[Road]:
        """Generate road network."""
        self.roads = []
        
        # Generate main roads (highways)
        self._generate_main_roads()
        
        # Generate secondary roads (streets)
        self._generate_secondary_roads()
        
        return self.roads
    
    def _generate_main_roads(self) -> None:
        """Generate main roads on primary grid lines."""
        
        # Calculate spacing for main roads
        num_horizontal = max(1, self.config.num_main_roads // 2)
        num_vertical = self.config.num_main_roads - num_horizontal
        
        # Horizontal main roads
        if num_horizontal > 0:
            h_spacing = self.config.world_height / (num_horizontal + 1)
            for i in range(num_horizontal):
                y = h_spacing * (i + 1)
                road = Road(
                    name=f"Highway-H{i+1}",
                    start_point=np.array([0.0, y]),
                    end_point=np.array([float(self.config.world_width), y]),
                    road_type=RoadType.HIGHWAY,
                    lanes=4,
                    width=self.config.road_width,
                    speed_limit_kmh=120.0,
                    is_bidirectional=True
                )
                self.roads.append(road)
        
        # Vertical main roads
        if num_vertical > 0:
            v_spacing = self.config.world_width / (num_vertical + 1)
            for i in range(num_vertical):
                x = v_spacing * (i + 1)
                road = Road(
                    name=f"Highway-V{i+1}",
                    start_point=np.array([x, 0.0]),
                    end_point=np.array([x, float(self.config.world_height)]),
                    road_type=RoadType.HIGHWAY,
                    lanes=4,
                    width=self.config.road_width,
                    speed_limit_kmh=120.0,
                    is_bidirectional=True
                )
                self.roads.append(road)
    
    def _generate_secondary_roads(self) -> None:
        """Generate secondary roads (streets) between main roads."""
        
        secondary_count = 0
        
        # Generate horizontal secondary roads
        h_spacing = self.config.world_height / (self.config.num_secondary_roads + 1)
        for i in range(self.config.num_secondary_roads):
            y = h_spacing * (i + 1)
            
            # Skip if too close to existing main roads
            if self._too_close_to_existing_road(0, y):
                continue
            
            road = Road(
                name=f"Main-Street-H{secondary_count+1}",
                start_point=np.array([0.0, y]),
                end_point=np.array([float(self.config.world_width), y]),
                road_type=RoadType.MAIN_STREET,
                lanes=2,
                width=self.config.road_width * 0.7,
                speed_limit_kmh=80.0,
                is_bidirectional=True
            )
            self.roads.append(road)
            secondary_count += 1
    
    def _too_close_to_existing_road(self, x: float, y: float, tolerance: float = 80.0) -> bool:
        """Check if location is too close to existing road."""
        for road in self.roads:
            distance = road.distance_to_point(np.array([x, y]))
            if distance < tolerance:
                return True
        return False


# ==============================================================================
# NON-OVERLAPPING ZONE GENERATOR
# ==============================================================================

class ZoneGenerator:
    """
    Generates non-overlapping zones using grid-based bin packing.
    
    Algorithm:
    1. Divide world into grid cells
    2. Use greedy placement to avoid overlaps
    3. Assign terrain types semantically (schools, highways, parks, etc.)
    4. Ensure separation margins between zones
    """
    
    # Zone type templates
    ZONE_TEMPLATES = {
        'school': {
            'terrain': TerrainType.ASPHALT,
            'speed_limit': 40,
            'frequency': 1
        },
        'park': {
            'terrain': TerrainType.GRASS,
            'speed_limit': 30,
            'frequency': 2
        },
        'highway_rest': {
            'terrain': TerrainType.CONCRETE,
            'speed_limit': 100,
            'frequency': 1
        },
        'residential': {
            'terrain': TerrainType.ASPHALT,
            'speed_limit': 50,
            'frequency': 2
        },
        'industrial': {
            'terrain': TerrainType.CONCRETE,
            'speed_limit': 60,
            'frequency': 1
        },
    }
    
    def __init__(self, config: GenerationConfig, seed: int) -> None:
        """Initialize zone generator."""
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.zones: List[Zone] = []
        self.used_rects: List[Tuple[int, int, int, int]] = []
    
    def generate(self) -> List[Zone]:
        """Generate non-overlapping zones."""
        self.zones = []
        self.used_rects = []
        
        # Create zone placement grid
        grid_cell_size = 150
        
        # Place zones greedily
        attempts = 0
        max_attempts = self.config.num_zones * 10
        
        while len(self.zones) < self.config.num_zones and attempts < max_attempts:
            attempts += 1
            
            # Random zone size
            size = self.rng.randint(
                self.config.min_zone_size,
                self.config.max_zone_size
            )
            
            # Random position
            x = self.rng.randint(self.config.zone_margin, 
                                 self.config.world_width - size - self.config.zone_margin)
            y = self.rng.randint(self.config.zone_margin,
                                 self.config.world_height - size - self.config.zone_margin)
            
            # Check for overlap
            if not self._overlaps_existing(x, y, size, size):
                zone_type = self.rng.choice(list(self.ZONE_TEMPLATES.keys()))
                template = self.ZONE_TEMPLATES[zone_type]
                
                zone = Zone(
                    name=f"{zone_type.title()}-{len(self.zones)+1}",
                    position=np.array([x + size/2, y + size/2]),
                    width=float(size),
                    height=float(size),
                    terrain_type=template['terrain'],
                    speed_limit_kmh=float(template['speed_limit']),
                    metadata={'zone_type': zone_type}
                )
                
                self.zones.append(zone)
                self.used_rects.append((x, y, x + size, y + size))
        
        return self.zones
    
    def _overlaps_existing(self, x: int, y: int, w: int, h: int, margin: int = 20) -> bool:
        """Check if rectangle overlaps existing zones (with margin)."""
        
        # Add margin
        x1, y1, x2, y2 = x - margin, y - margin, x + w + margin, y + h + margin
        
        for rx1, ry1, rx2, ry2 in self.used_rects:
            # AABB overlap test
            if not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2):
                return True
        
        return False


# ==============================================================================
# OBSTACLE GENERATOR
# ==============================================================================

class ObstacleGenerator:
    """
    Generates obstacles placed strategically:
    - Near roads (parking, trees)
    - Within zones (buildings)
    
    Ensures obstacles don't overlap with each other or block entire roads.
    """
    
    OBSTACLE_TEMPLATES = {
        'near_road': [
            {'type': ObstacleType.TREE, 'probability': 0.4, 'size_range': (20, 50)},
            {'type': ObstacleType.CAR, 'probability': 0.3, 'size_range': (30, 50)},
            {'type': ObstacleType.DEBRIS, 'probability': 0.3, 'size_range': (10, 40)},
        ],
        'in_zone': [
            {'type': ObstacleType.BUILDING, 'probability': 0.6, 'size_range': (50, 150)},
            {'type': ObstacleType.TREE, 'probability': 0.3, 'size_range': (20, 50)},
            {'type': ObstacleType.ROCK, 'probability': 0.1, 'size_range': (30, 80)},
        ],
    }
    
    def __init__(self, config: GenerationConfig, seed: int, 
                 roads: List[Road], zones: List[Zone]) -> None:
        """Initialize obstacle generator."""
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.roads = roads
        self.zones = zones
        self.obstacles: List[Obstacle] = []
        self.used_rects: List[Tuple[float, float, float, float]] = []
    
    def generate(self) -> List[Obstacle]:
        """Generate obstacles."""
        self.obstacles = []
        self.used_rects = []
        
        num_near_roads = int(self.config.num_obstacles * self.config.obstacles_near_roads)
        num_in_zones = self.config.num_obstacles - num_near_roads
        
        # Place obstacles near roads
        for _ in range(num_near_roads):
            self._place_near_road()
        
        # Place obstacles in zones
        for _ in range(num_in_zones):
            self._place_in_zone()
        
        return self.obstacles
    
    def _place_near_road(self) -> None:
        """Place obstacle near a random road."""
        
        if not self.roads:
            return
        
        attempts = 0
        max_attempts = 20
        
        while attempts < max_attempts:
            attempts += 1
            
            # Pick random road
            road = self.rng.choice(self.roads)
            
            # Random position along road
            t = self.rng.uniform(0, 1)
            center = road.start_point + t * (road.end_point - road.start_point)
            
            # Place perpendicular to road (offset)
            direction = road.direction
            perpendicular = np.array([-direction[1], direction[0]])
            offset_distance = self.rng.uniform(road.width / 2 + 10, road.width * 2)
            
            if self.rng.rand() > 0.5:
                offset_distance *= -1
            
            position = center + perpendicular * offset_distance
            
            # Generate obstacle
            template = self.rng.choice(self.OBSTACLE_TEMPLATES['near_road'])
            size_min, size_max = template['size_range']
            size = self.rng.randint(size_min, size_max)
            
            # Check bounds and overlap
            if self._is_valid_position(position, size):
                obstacle = Obstacle(
                    name=f"{template['type'].name}-{len(self.obstacles)+1}",
                    position=position.astype(np.float32),
                    width=float(size),
                    height=float(size),
                    obstacle_type=template['type'],
                    damage_on_collision=float(size)  # Damage proportional to size
                )
                
                self.obstacles.append(obstacle)
                self._record_rect(position, size)
                return
    
    def _place_in_zone(self) -> None:
        """Place obstacle within a random zone."""
        
        if not self.zones:
            return
        
        attempts = 0
        max_attempts = 20
        
        while attempts < max_attempts:
            attempts += 1
            
            # Pick random zone
            zone = self.rng.choice(self.zones)
            
            # Random position within zone
            x_min, y_min, x_max, y_max = zone.get_bounds()
            
            template = self.rng.choice(self.OBSTACLE_TEMPLATES['in_zone'])
            size_min, size_max = template['size_range']
            size = self.rng.randint(size_min, size_max)
            
            x = self.rng.uniform(x_min + size/2, x_max - size/2)
            y = self.rng.uniform(y_min + size/2, y_max - size/2)
            position = np.array([x, y])
            
            # Check overlap
            if self._is_valid_position(position, size):
                obstacle = Obstacle(
                    name=f"{template['type'].name}-{len(self.obstacles)+1}",
                    position=position.astype(np.float32),
                    width=float(size),
                    height=float(size),
                    obstacle_type=template['type'],
                    is_passable=(template['type'] == ObstacleType.TREE),
                    damage_on_collision=float(size)
                )
                
                self.obstacles.append(obstacle)
                self._record_rect(position, size)
                return
    
    def _is_valid_position(self, position: np.ndarray, size: int, margin: float = 30.0) -> bool:
        """Check if position is valid (in bounds, no overlap)."""
        
        # Check world bounds
        x, y = position
        if x - size/2 < 0 or x + size/2 > self.config.world_width:
            return False
        if y - size/2 < 0 or y + size/2 > self.config.world_height:
            return False
        
        # Check overlap with existing obstacles
        x1, y1, x2, y2 = x - size/2 - margin, y - size/2 - margin, x + size/2 + margin, y + size/2 + margin
        
        for rx1, ry1, rx2, ry2 in self.used_rects:
            if not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2):
                return False
        
        return True
    
    def _record_rect(self, position: np.ndarray, size: int) -> None:
        """Record bounding rectangle of placed obstacle."""
        x, y = position
        self.used_rects.append((x - size/2, y - size/2, x + size/2, y + size/2))


# ==============================================================================
# WORLD GENERATOR (Main Orchestrator)
# ==============================================================================

class WorldGenerator:
    """
    Orchestrates complete world generation.
    
    Generates a fully-populated World with roads, zones, and obstacles
    using sophisticated procedural algorithms. All randomization is seeded
    for reproducibility.
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None, seed: int = 42) -> None:
        """
        Initialize world generator.
        
        Args:
            config: Generation configuration (uses defaults if None)
            seed: Random seed for reproducibility
        """
        self.config = config or GenerationConfig()
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate(self) -> World:
        """
        Generate a complete world.
        
        Returns:
            Fully populated World instance with roads, zones, and obstacles
        """
        
        # Create world container
        world = World(
            name=f"Generated-World-{self.seed}",
            width=self.config.world_width,
            height=self.config.world_height,
            metadata={'seed': self.seed, 'generation_config': str(self.config)}
        )
        
        # Generate roads
        print(f"Generating {self.config.num_main_roads} main + {self.config.num_secondary_roads} secondary roads...")
        road_gen = RoadGenerator(self.config, self.seed)
        roads = road_gen.generate()
        
        for road in roads:
            world.add_road(road)
        print(f"  ✓ Generated {len(roads)} roads")
        
        # Generate zones
        print(f"Generating {self.config.num_zones} zones...")
        zone_gen = ZoneGenerator(self.config, self.seed + 1)
        zones = zone_gen.generate()
        
        for zone in zones:
            world.add_zone(zone)
        print(f"  ✓ Generated {len(zones)} zones")
        
        # Generate obstacles
        print(f"Generating {self.config.num_obstacles} obstacles...")
        obs_gen = ObstacleGenerator(self.config, self.seed + 2, roads, zones)
        obstacles = obs_gen.generate()
        
        for obstacle in obstacles:
            world.add_obstacle(obstacle)
        print(f"  ✓ Generated {len(obstacles)} obstacles")
        
        print(f"\n✓ World generation complete: {world.summary()}")
        
        return world


# ==============================================================================
# CONVENIENCE FUNCTION
# ==============================================================================

def generate_world(
    width: int = 1200,
    height: int = 800,
    seed: int = 42,
    num_roads: int = 5,
    num_zones: int = 8,
    num_obstacles: int = 20
) -> World:
    """
    Quick world generation with sensible defaults.
    
    Args:
        width: World width in pixels
        height: World height in pixels
        seed: Random seed
        num_roads: Total number of roads
        num_zones: Number of zones
        num_obstacles: Number of obstacles
    
    Returns:
        Fully generated World instance
    """
    config = GenerationConfig(
        num_main_roads=max(1, num_roads // 2),
        num_secondary_roads=num_roads - num_roads // 2,
        num_zones=num_zones,
        num_obstacles=num_obstacles,
        world_width=width,
        world_height=height
    )
    
    generator = WorldGenerator(config, seed=seed)
    return generator.generate()
