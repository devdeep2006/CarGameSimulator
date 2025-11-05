from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple, Any
from enum import Enum
import numpy as np
import pymunk

from dataclasses_core import World, Zone, Road, Obstacle


class PhysicsCollisionType(Enum):
    """Collision type categories."""
    VEHICLE = 1
    ROAD_SENSOR = 2
    OBSTACLE = 3
    ZONE_SENSOR = 4
    BOUNDARY = 5


@dataclass
class CollisionEvent:
    """Collision event information."""
    event_type: str
    arbiter: pymunk.Arbiter
    shape_a: pymunk.Shape
    shape_b: pymunk.Shape
    collision_type_a: PhysicsCollisionType
    collision_type_b: PhysicsCollisionType
    entity_a_id: Optional[Any] = None
    entity_b_id: Optional[Any] = None
    metadata_a: Optional[Dict] = None
    metadata_b: Optional[Dict] = None


class PhysicsWorld:
    """Simplified physics world wrapper for Pymunk."""
    
    def __init__(
        self,
        world: World,
        gravity: Tuple[float, float] = (0.0, 0.0),
        damping: float = 0.8,
    ) -> None:
        """Initialize physics world."""
        self.world = world
        
        # Create Pymunk space
        self.space = pymunk.Space()
        self.space.gravity = gravity
        self.space.damping = damping
        
        # Collision tracking
        self.collision_events: List[CollisionEvent] = []
        self.collision_handlers: Dict[str, List[Callable]] = {
            'begin': [],
            'pre_solve': [],
            'post_solve': [],
            'separate': [],
        }
        
        # Add world entities to physics
        self._add_obstacles_to_physics()
        self._add_zones_to_physics()
        self._add_roads_to_physics()
        self._add_boundaries()
    
    def _add_obstacles_to_physics(self) -> None:
        """Convert obstacles to static physics bodies."""
        for obstacle in self.world.obstacles.values():
            # Create static body at obstacle position
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            # FIX: Convert numpy array to tuple
            body.position = tuple(obstacle.position)
            
            # Create box shape
            shape = pymunk.Poly.create_box(body, (obstacle.width, obstacle.height))
            shape.friction = 1.0
            shape.elasticity = 0.5
            shape.sensor = False
            
            # Store metadata
            shape.user_data = {
                'entity_id': obstacle.obstacle_id,
                'entity_type': 'obstacle',
                'collision_type': PhysicsCollisionType.OBSTACLE,
                'name': obstacle.name,
                'damage': obstacle.damage_on_collision,
            }
            
            # Add to space
            self.space.add(body, shape)
    
    def _add_zones_to_physics(self) -> None:
        """Convert zones to kinematic sensor bodies."""
        for zone in self.world.zones.values():
            # Create kinematic body
            body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            # FIX: Convert numpy array to tuple
            body.position = tuple(zone.position)
            
            # Create box shape
            shape = pymunk.Poly.create_box(body, (zone.width, zone.height))
            shape.friction = 0.0
            shape.elasticity = 0.0
            shape.sensor = True  # Sensor - no collision response
            
            # Store metadata
            shape.user_data = {
                'entity_id': zone.zone_id,
                'entity_type': 'zone',
                'collision_type': PhysicsCollisionType.ZONE_SENSOR,
                'name': zone.name,
                'terrain_type': zone.terrain_type.name,
            }
            
            # Add to space
            self.space.add(body, shape)
    
    def _add_roads_to_physics(self) -> None:
        """Convert roads to kinematic sensor bodies."""
        for road in self.world.roads.values():
            # Create kinematic body at midpoint
            midpoint = (road.start_point + road.end_point) / 2
            body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            # FIX: Convert numpy array to tuple
            
            body.position = tuple(midpoint)
            
            # Create line segment shape
            start_rel = road.start_point - midpoint
            end_rel = road.end_point - midpoint
            
            # FIX: Convert numpy arrays to tuples
            shape = pymunk.Segment(
                body,
                tuple(start_rel),
                tuple(end_rel),
                road.width / 2
            )
            shape.friction = 1.0
            shape.elasticity = 0.0
            shape.sensor = True
            
            # Store metadata
            shape.user_data = {
                'entity_id': road.road_id,
                'entity_type': 'road',
                'collision_type': PhysicsCollisionType.ROAD_SENSOR,
                'name': road.name,
                'road_type': road.road_type.name,
            }
            
            # Add to space
            self.space.add(body, shape)
    
    def _add_boundaries(self) -> None:
        """Add world boundaries."""
        # Create static body for boundaries
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.space.add(body)
        # Add four boundary lines
        boundaries = [
            # Left wall
            pymunk.Segment(body, (0, 0), (0, self.world.height), 1),
            # Right wall
            pymunk.Segment(body, (self.world.width, 0), (self.world.width, self.world.height), 1),
            # Top wall
            pymunk.Segment(body, (0, 0), (self.world.width, 0), 1),
            # Bottom wall
            pymunk.Segment(body, (0, self.world.height), (self.world.width, self.world.height), 1),
        ]
        
        for shape in boundaries:
            shape.friction = 1.0
            shape.elasticity = 0.5
            shape.sensor = False
            shape.user_data = {
                'entity_type': 'boundary',
                'collision_type': PhysicsCollisionType.BOUNDARY,
            }
            self.space.add(shape)
        
    
    def step(self, dt: float) -> None:
        """Step physics simulation."""
        self.space.step(dt)
    
    def register_collision_callback(
        self,
        event_type: str,
        callback: Callable[[CollisionEvent], None]
    ) -> None:
        """Register collision callback."""
        if event_type in self.collision_handlers:
            self.collision_handlers[event_type].append(callback)
    
    def get_collision_events(self) -> List[CollisionEvent]:
        """Get and clear collision events."""
        events = self.collision_events.copy()
        self.collision_events.clear()
        return events
    
    def get_entities_in_radius(
        self,
        position: np.ndarray,
        radius: float = 300.0
    ) -> List[Tuple]:
        """Query entities near position."""
        results = []
        
        # FIX: Ensure position is numpy array
        if isinstance(position, (list, tuple)):
            position = np.array(position)
        
        for shape in self.space.shapes:
            if shape.user_data is None:
                continue
            
            entity_pos = np.array(shape.body.position)
            distance = np.linalg.norm(entity_pos - position)
            
            if distance <= radius:
                entity_id = shape.user_data.get('entity_id')
                entity_type = shape.user_data.get('entity_type')
                results.append((entity_id, entity_type, distance))
        
        # Sort by distance
        results.sort(key=lambda x: x[2])
        return results
    
    def get_entity_at_position(
        self,
        position: np.ndarray
    ) -> Optional[Tuple]:
        """Get nearest entity at position."""
        nearest = None
        min_dist = float('inf')
        
        # FIX: Ensure position is numpy array
        if isinstance(position, (list, tuple)):
            position = np.array(position)
        
        for shape in self.space.shapes:
            if shape.user_data is None:
                continue
            
            entity_pos = np.array(shape.body.position)
            distance = np.linalg.norm(entity_pos - position)
            
            if distance < min_dist:
                min_dist = distance
                entity_id = shape.user_data.get('entity_id')
                entity_type = shape.user_data.get('entity_type')
                nearest = (entity_id, entity_type)
        
        return nearest
    
    def segment_query(
        self,
        start: np.ndarray,
        end: np.ndarray
    ) -> Optional[Tuple]:
        """Ray casting query."""
        # FIX: Convert to tuples for Pymunk
        start_tuple = tuple(start) if isinstance(start, np.ndarray) else start
        end_tuple = tuple(end) if isinstance(end, np.ndarray) else end
        
        query_result = self.space.segment_query(start_tuple, end_tuple, 0.0)
        
        if query_result:
            info = query_result[0]
            return (
                info.shape.user_data.get('entity_id') if info.shape.user_data else None,
                info.shape.user_data.get('entity_type') if info.shape.user_data else None,
                info.point,
                info.alpha
            )
        return None
    
    def get_space(self) -> pymunk.Space:
        """Get underlying Pymunk space."""
        return self.space
    
    def cleanup(self) -> None:
        """Clean up physics world."""
        self.space = None
    
    def summary(self) -> str:
        """Get physics world summary."""
        return (
            f"PhysicsWorld: "
            f"Bodies: {len(self.space.bodies)}, "
            f"Shapes: {len(self.space.shapes)}, "
            f"Gravity: {self.space.gravity}, "
            f"Damping: {self.space.damping}"
        )