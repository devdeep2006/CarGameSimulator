"""
Semantic Extensions: Rich semantic parameters for Zone and Obstacle.

Implements data-driven physics effects, penalty systems, and modifier registries.
All effects are configuration-based and read from registries - no hard-coded logic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import numpy as np

from dataclasses_core import Zone, Obstacle, World


# ==============================================================================
# SEMANTIC PARAMETER DEFINITIONS
# ==============================================================================

@dataclass
class SemanticModifier:
    """
    A modifier that affects vehicle behavior in specific zones/obstacles.
    
    All parameters are data-driven and applied consistently.
    """
    name: str
    friction_multiplier: float = 1.0           # 0.0-2.0: friction modifier
    max_speed_kmh: Optional[float] = None       # Max speed in this zone
    min_speed_kmh: Optional[float] = None       # Min speed in this zone
    acceleration_multiplier: float = 1.0       # 0.0-2.0: accel/braking modifier
    penalty_per_second: float = 0.0            # Damage/penalty while inside
    drag_multiplier: float = 1.0               # Air resistance multiplier
    grip_multiplier: float = 1.0               # Tire grip multiplier
    visibility_multiplier: float = 1.0         # Sensor visibility modifier
    sound_multiplier: float = 1.0              # Sound volume modifier
    damage_on_collision: float = 0.0           # Collision damage override
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def copy(self) -> "SemanticModifier":
        """Create a copy of this modifier."""
        return SemanticModifier(
            name=self.name,
            friction_multiplier=self.friction_multiplier,
            max_speed_kmh=self.max_speed_kmh,
            min_speed_kmh=self.min_speed_kmh,
            acceleration_multiplier=self.acceleration_multiplier,
            penalty_per_second=self.penalty_per_second,
            drag_multiplier=self.drag_multiplier,
            grip_multiplier=self.grip_multiplier,
            visibility_multiplier=self.visibility_multiplier,
            sound_multiplier=self.sound_multiplier,
            damage_on_collision=self.damage_on_collision,
            metadata=self.metadata.copy(),
        )
    
    def apply_to_zone(self, zone: Zone) -> None:
        """Apply modifier to zone's semantic data."""
        if self.friction_multiplier != 1.0:
            props = zone.get_terrain_properties()
            new_friction = props.friction_coefficient * self.friction_multiplier
            zone.metadata['friction_override'] = new_friction
        
        if self.max_speed_kmh is not None:
            zone.metadata['max_speed_override'] = self.max_speed_kmh
        
        if self.penalty_per_second != 0.0:
            zone.metadata['penalty_per_second'] = self.penalty_per_second
        
        # Store all modifier data
        zone.metadata['semantic_modifier'] = self
    
    def apply_to_obstacle(self, obstacle: Obstacle) -> None:
        """Apply modifier to obstacle's semantic data."""
        if self.damage_on_collision > 0:
            obstacle.damage_on_collision = self.damage_on_collision
        
        obstacle.metadata['semantic_modifier'] = self


# ==============================================================================
# SEMANTIC MODIFIER REGISTRY
# ==============================================================================

class SemanticModifierRegistry:
    """
    Global registry for semantic modifiers.
    
    Data-driven approach: all modifiers defined here, not hard-coded elsewhere.
    """
    
    # Predefined modifier templates
    MODIFIERS: Dict[str, SemanticModifier] = {
        # Road conditions
        'dry_road': SemanticModifier(
            name='dry_road',
            friction_multiplier=1.0,
            penalty_per_second=0.0,
            drag_multiplier=1.0,
        ),
        'wet_road': SemanticModifier(
            name='wet_road',
            friction_multiplier=0.7,          # Slippery
            acceleration_multiplier=0.85,     # Reduced braking/accel
            penalty_per_second=0.1,           # Minor penalty
            grip_multiplier=0.8,
        ),
        'icy_road': SemanticModifier(
            name='icy_road',
            friction_multiplier=0.4,          # Very slippery
            acceleration_multiplier=0.6,
            penalty_per_second=0.5,           # Increased penalty
            grip_multiplier=0.5,
            max_speed_kmh=40.0,               # Reduced max speed
        ),
        'gravel_road': SemanticModifier(
            name='gravel_road',
            friction_multiplier=0.8,
            acceleration_multiplier=0.9,
            penalty_per_second=0.2,           # Dust/wear
            drag_multiplier=1.15,
        ),
        
        # Zone types
        'school_zone': SemanticModifier(
            name='school_zone',
            max_speed_kmh=30.0,
            penalty_per_second=1.0,           # Heavy penalty for speeding
            drag_multiplier=1.1,
        ),
        'construction': SemanticModifier(
            name='construction',
            friction_multiplier=0.75,
            acceleration_multiplier=0.7,
            max_speed_kmh=25.0,
            penalty_per_second=2.0,           # Very hazardous
            visibility_multiplier=0.8,
        ),
        'parking': SemanticModifier(
            name='parking',
            friction_multiplier=0.9,
            max_speed_kmh=15.0,
            penalty_per_second=0.05,
        ),
        
        # Obstacle types
        'fragile': SemanticModifier(
            name='fragile',
            damage_on_collision=10.0,
        ),
        'heavy': SemanticModifier(
            name='heavy',
            damage_on_collision=50.0,
        ),
        'immovable': SemanticModifier(
            name='immovable',
            damage_on_collision=100.0,
        ),
    }
    
    def __init__(self) -> None:
        """Initialize registry with default modifiers."""
        self.custom_modifiers: Dict[str, SemanticModifier] = {}
    
    def register(self, modifier: SemanticModifier) -> None:
        """Register a custom modifier."""
        self.custom_modifiers[modifier.name] = modifier
    
    def get(self, name: str) -> Optional[SemanticModifier]:
        """Get a modifier by name."""
        if name in self.MODIFIERS:
            return self.MODIFIERS[name].copy()
        elif name in self.custom_modifiers:
            return self.custom_modifiers[name].copy()
        return None
    
    def list_all(self) -> List[str]:
        """List all available modifier names."""
        return list(self.MODIFIERS.keys()) + list(self.custom_modifiers.keys())


# Global registry instance
semantic_registry = SemanticModifierRegistry()


# ==============================================================================
# EXTENDED ZONE WITH SEMANTIC EFFECTS
# ==============================================================================

def apply_semantic_modifier_to_zone(zone: Zone, modifier_name: str) -> None:
    """
    Apply semantic modifier to a zone.
    
    This is the extensible hook for adding dynamic effects.
    """
    modifier = semantic_registry.get(modifier_name)
    if modifier:
        modifier.apply_to_zone(zone)
        zone.metadata['modifier_name'] = modifier_name


def apply_semantic_modifier_to_obstacle(obstacle: Obstacle, modifier_name: str) -> None:
    """
    Apply semantic modifier to an obstacle.
    """
    modifier = semantic_registry.get(modifier_name)
    if modifier:
        modifier.apply_to_obstacle(obstacle)
        obstacle.metadata['modifier_name'] = modifier_name


# ==============================================================================
# SEMANTIC EFFECTS CALCULATOR
# ==============================================================================

class SemanticEffects:
    """
    Calculates combined semantic effects at a position.
    
    Used by PhysicsWorld or Simulator to apply effects dynamically.
    """
    
    def __init__(self, world: World) -> None:
        """Initialize effects calculator."""
        self.world = world
    
    def get_effects_at_position(self, position: np.ndarray, radius: float = 50.0) -> Dict[str, Any]:
        """
        Get combined semantic effects at a position.
        
        Returns:
            Dict with combined effects from all nearby zones/obstacles
        """
        effects = {
            'friction_multiplier': 1.0,
            'max_speed_kmh': None,
            'min_speed_kmh': None,
            'acceleration_multiplier': 1.0,
            'penalty_per_second': 0.0,
            'drag_multiplier': 1.0,
            'grip_multiplier': 1.0,
            'visibility_multiplier': 1.0,
            'sound_multiplier': 1.0,
            'damage_on_collision': 0.0,
            'contributing_zones': [],
            'contributing_obstacles': [],
        }
        
        # Query zones
        zones = self.world.find_zones_in_radius(position, radius)
        for zone in zones:
            if 'semantic_modifier' in zone.metadata:
                modifier: SemanticModifier = zone.metadata['semantic_modifier']
                
                # Apply modifier effects (averaged)
                effects['friction_multiplier'] *= modifier.friction_multiplier
                effects['acceleration_multiplier'] *= modifier.acceleration_multiplier
                effects['drag_multiplier'] *= modifier.drag_multiplier
                effects['grip_multiplier'] *= modifier.grip_multiplier
                effects['visibility_multiplier'] *= modifier.visibility_multiplier
                effects['sound_multiplier'] *= modifier.sound_multiplier
                effects['penalty_per_second'] += modifier.penalty_per_second
                
                # Speed limits (most restrictive wins)
                if modifier.max_speed_kmh is not None:
                    if effects['max_speed_kmh'] is None:
                        effects['max_speed_kmh'] = modifier.max_speed_kmh
                    else:
                        effects['max_speed_kmh'] = min(effects['max_speed_kmh'], modifier.max_speed_kmh)
                
                if modifier.min_speed_kmh is not None:
                    if effects['min_speed_kmh'] is None:
                        effects['min_speed_kmh'] = modifier.min_speed_kmh
                    else:
                        effects['min_speed_kmh'] = max(effects['min_speed_kmh'], modifier.min_speed_kmh)
                
                effects['contributing_zones'].append(zone.name)
        
        # Query obstacles (if at position)
        obstacles = self.world.find_obstacles_near_point(position, radius)
        for obstacle in obstacles:
            if 'semantic_modifier' in obstacle.metadata:
                modifier: SemanticModifier = obstacle.metadata['semantic_modifier']
                
                if modifier.damage_on_collision > effects['damage_on_collision']:
                    effects['damage_on_collision'] = modifier.damage_on_collision
                
                effects['contributing_obstacles'].append(obstacle.name)
        
        return effects
    
    def get_zone_penalty_at_position(self, position: np.ndarray, dt: float) -> float:
        """
        Get accumulated penalty from zones at position over time dt.
        
        Returns:
            Accumulated penalty amount
        """
        effects = self.get_effects_at_position(position)
        return effects['penalty_per_second'] * dt
    
    def get_speed_constraint_at_position(self, position: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Get speed limits at position.
        
        Returns:
            (min_speed_kmh, max_speed_kmh) where None means no constraint
        """
        effects = self.get_effects_at_position(position)
        return (effects['min_speed_kmh'], effects['max_speed_kmh'])
    
    def get_friction_at_position(self, position: np.ndarray) -> float:
        """Get effective friction coefficient at position."""
        effects = self.get_effects_at_position(position)
        base_friction = 0.95  # Default asphalt
        return base_friction * effects['friction_multiplier']


# ==============================================================================
# PHYSICS APPLICATION HOOKS
# ==============================================================================

def apply_semantic_effects_to_physics(
    physics_world: "PhysicsWorld",
    world: World,
    vehicle_position: np.ndarray,
    dt: float
) -> Dict[str, Any]:
    """
    Apply semantic effects to physics simulation.
    
    This hook is called by PhysicsWorld/Simulator each frame.
    
    Returns:
        Dict with applied effects for telemetry/debugging
    """
    effects_calc = SemanticEffects(world)
    effects = effects_calc.get_effects_at_position(vehicle_position)
    
    # Apply friction multiplier to vehicle (if vehicle exists)
    # TODO: When vehicle is implemented, modify pymunk body here
    
    applied = {
        'friction_multiplier': effects['friction_multiplier'],
        'penalty_accumulated': effects_calc.get_zone_penalty_at_position(vehicle_position, dt),
        'speed_limits': effects_calc.get_speed_constraint_at_position(vehicle_position),
        'contributing_zones': effects['contributing_zones'],
    }
    
    return applied


# ==============================================================================
# CONFIGURATION-BASED WORLD SETUP
# ==============================================================================

def setup_semantic_world_from_config(world: World, config: Dict[str, Any]) -> None:
    """
    Set up semantic modifiers for world based on configuration.
    
    Example config:
    {
        'zones': {
            'school_zone': ['School-1', 'School-2'],
            'wet_road': ['Road-1'],
        },
        'obstacles': {
            'heavy': ['Building-1', 'Building-2'],
            'fragile': ['Tree-1', 'Tree-2'],
        }
    }
    """
    # Apply zone modifiers
    if 'zones' in config:
        for modifier_name, zone_names in config['zones'].items():
            for zone_name in zone_names:
                # Find zone by name
                for zone in world.zones.values():
                    if zone.name == zone_name:
                        apply_semantic_modifier_to_zone(zone, modifier_name)
    
    # Apply obstacle modifiers
    if 'obstacles' in config:
        for modifier_name, obstacle_names in config['obstacles'].items():
            for obstacle_name in obstacle_names:
                # Find obstacle by name
                for obstacle in world.obstacles.values():
                    if obstacle.name == obstacle_name:
                        apply_semantic_modifier_to_obstacle(obstacle, modifier_name)


# ==============================================================================
# TELEMETRY & DEBUGGING
# ==============================================================================

class SemanticTelemetry:
    """Track semantic effects during simulation."""
    
    def __init__(self) -> None:
        """Initialize telemetry."""
        self.effects_history: List[Dict[str, Any]] = []
        self.penalty_accumulated: float = 0.0
        self.max_friction_multiplier: float = 1.0
        self.min_friction_multiplier: float = 1.0
    
    def record_effects(self, step: int, effects: Dict[str, Any]) -> None:
        """Record effects at a step."""
        self.effects_history.append({
            'step': step,
            **effects,
        })
        
        # Track statistics
        friction = effects['friction_multiplier']
        self.max_friction_multiplier = max(self.max_friction_multiplier, friction)
        self.min_friction_multiplier = min(self.min_friction_multiplier, friction)
        self.penalty_accumulated += effects.get('penalty_accumulated', 0.0)
    
    def summary(self) -> str:
        """Get telemetry summary."""
        if not self.effects_history:
            return "No telemetry recorded"
        
        return (
            f"Semantic Telemetry:\n"
            f"  Steps recorded: {len(self.effects_history)}\n"
            f"  Friction range: {self.min_friction_multiplier:.2f}x - {self.max_friction_multiplier:.2f}x\n"
            f"  Total penalty: {self.penalty_accumulated:.1f}\n"
        )
