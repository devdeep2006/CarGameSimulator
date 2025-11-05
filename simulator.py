"""
Simulator: Main orchestrator for the driving simulation.

Coordinates PhysicsWorld, Renderer, and World entities.
Provides clean event loop, reset/run mechanisms, and extensible hooks for agents.

Philosophy: Simulator is pure orchestration - NO agent logic.
All decision-making deferred to external agents via hooks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any
from enum import Enum, auto
import numpy as np
from datetime import datetime

from dataclasses_core import World
from world_generator import WorldGenerator, GenerationConfig
from physics_world import PhysicsWorld, CollisionEvent, PhysicsCollisionType
from renderer import Renderer, ColorScheme


# ==============================================================================
# SIMULATION STATE
# ==============================================================================

class SimulatorState(Enum):
    """Simulator operational state."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()


# ==============================================================================
# EPISODE STATISTICS
# ==============================================================================

@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    episode_number: int = 0
    start_time: float = 0.0
    duration: float = 0.0
    total_steps: int = 0
    total_collisions: int = 0
    collision_details: List[Dict[str, Any]] = field(default_factory=list)
    max_speed: float = 0.0
    avg_speed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def record_collision(self, event: CollisionEvent) -> None:
        """Record a collision event."""
        self.total_collisions += 1
        self.collision_details.append({
            'step': self.total_steps,
            'type_a': event.collision_type_a.name,
            'type_b': event.collision_type_b.name,
            'entity_a': event.metadata_a.get('name') if event.metadata_a else None,
            'entity_b': event.metadata_b.get('name') if event.metadata_b else None,
        })
    
    def summary(self) -> str:
        """Get statistics summary."""
        return (
            f"Episode {self.episode_number}: "
            f"Duration {self.duration:.2f}s, "
            f"Steps {self.total_steps}, "
            f"Collisions {self.total_collisions}"
        )


# ==============================================================================
# SIMULATOR
# ==============================================================================

class Simulator:
    """
    Main simulation orchestrator.
    
    Coordinates world generation, physics, rendering, and event handling.
    Provides hooks for agent control without embedding agent logic.
    """
    
    def __init__(
        self,
        world: Optional[World] = None,
        generator_config: Optional[GenerationConfig] = None,
        render: bool = True,
        render_width: int = 1200,
        render_height: int = 800,
        render_fps: int = 60,
        physics_fps: int = 60,
        gravity: Tuple[float, float] = (0.0, 0.0),
        physics_damping: float = 0.8,
    ) -> None:
        """
        Initialize simulator.
        
        Args:
            world: Existing World instance (generates if None)
            generator_config: Config for procedural generation
            render: Enable Pygame rendering
            render_width: Render width
            render_height: Render height
            render_fps: Rendering FPS
            physics_fps: Physics simulation FPS
            gravity: Physics gravity vector
            physics_damping: Physics damping factor
        """
        self.render_enabled = render
        self.physics_fps = physics_fps
        self.dt = 1.0 / physics_fps  # Physics timestep
        
        # World creation
        if world is None:
            gen_config = generator_config or GenerationConfig()
            generator = WorldGenerator(gen_config, seed=42)
            self.world = generator.generate()
        else:
            self.world = world
        
        # Physics
        self.physics_world = PhysicsWorld(
            self.world,
            gravity=gravity,
            damping=physics_damping
        )
        
        # Rendering
        self.renderer = None
        if render:
            self.renderer = Renderer(
                width=render_width,
                height=render_height,
                fps=render_fps,
                title="Driving Simulator"
            )
            self.renderer.camera.reset(
                self.world.width,
                self.world.height,
                render_width,
                render_height
            )
        
        # State management
        self.state = SimulatorState.IDLE
        self.running = False
        self.episode_number = 0
        self.step_count = 0
        self.episode_stats = EpisodeStats()
        
        # Agent hooks (no logic here - just placeholders)
        self.agent_action_hook: Optional[Callable[[], np.ndarray]] = None
        self.agent_observation_hook: Optional[Callable[[Dict], None]] = None
        self.agent_collision_hook: Optional[Callable[[CollisionEvent], None]] = None
        self.agent_step_hook: Optional[Callable[[int], None]] = None
        
        # Callbacks for events
        self.on_step_callbacks: List[Callable[[int], None]] = []
        self.on_collision_callbacks: List[Callable[[CollisionEvent], None]] = []
        self.on_episode_end_callbacks: List[Callable[[EpisodeStats], None]] = []
    
    # ========================================================================
    # AGENT HOOKS
    # ========================================================================
    
    def set_action_hook(self, hook: Callable[[], np.ndarray]) -> None:
        """
        Set hook for agent to provide actions.
        
        Hook should return action array or None (simulator ignores).
        This is where agent decision-making happens.
        """
        self.agent_action_hook = hook
    
    def set_observation_hook(self, hook: Callable[[Dict], None]) -> None:
        """
        Set hook to provide observations to agent.
        
        Hook receives observation dict each step.
        """
        self.agent_observation_hook = hook
    
    def set_collision_hook(self, hook: Callable[[CollisionEvent], None]) -> None:
        """
        Set hook to notify agent of collisions.
        
        Hook receives CollisionEvent.
        """
        self.agent_collision_hook = hook
    
    def set_step_hook(self, hook: Callable[[int], None]) -> None:
        """
        Set hook called each simulation step.
        
        Hook receives current step number.
        """
        self.agent_step_hook = hook
    
    # ========================================================================
    # CALLBACK REGISTRATION
    # ========================================================================
    
    def register_step_callback(self, callback: Callable[[int], None]) -> None:
        """Register callback called each simulation step."""
        self.on_step_callbacks.append(callback)
    
    def register_collision_callback(self, callback: Callable[[CollisionEvent], None]) -> None:
        """Register callback for collisions."""
        self.on_collision_callbacks.append(callback)
    
    def register_episode_end_callback(self, callback: Callable[[EpisodeStats], None]) -> None:
        """Register callback when episode ends."""
        self.on_episode_end_callbacks.append(callback)
    
    # ========================================================================
    # EPISODE CONTROL
    # ========================================================================
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset simulation to initial state.
        
        Returns:
            Initial observation dict for agent
        """
        # Reset physics
        self.physics_world.cleanup()
        self.physics_world = PhysicsWorld(
            self.world,
            gravity=(0.0, 0.0),
            damping=0.8
        )
        
        # Reset stats
        self.step_count = 0
        self.episode_number += 1
        self.episode_stats = EpisodeStats(episode_number=self.episode_number)
        
        # Generate observation
        observation = self._get_observation()
        
        return observation
    
    def reset_world(self, seed: Optional[int] = None) -> None:
        """
        Regenerate world with new procedural seed.
        
        Args:
            seed: Random seed for generation
        """
        if seed is None:
            seed = int(datetime.now().timestamp() * 1000) % 10000
        
        config = GenerationConfig()
        generator = WorldGenerator(config, seed=seed)
        self.world = generator.generate()
        
        # Reset physics
        self.physics_world.cleanup()
        self.physics_world = PhysicsWorld(self.world)
        
        # Update renderer camera
        if self.renderer:
            self.renderer.camera.reset(
                self.world.width,
                self.world.height,
                self.renderer.width,
                self.renderer.height
            )
    
    def run_episode(
        self,
        max_steps: Optional[int] = None,
        render: bool = True
    ) -> EpisodeStats:
        """
        Run a complete episode.
        
        Args:
            max_steps: Maximum steps (None = unlimited)
            render: Whether to render during episode
        
        Returns:
            Episode statistics
        """
        # Reset to clean state
        self.reset()
        
        # Episode loop
        self.state = SimulatorState.RUNNING
        self.running = True
        
        start_time = datetime.now()
        
        while self.running:
            # Check max steps
            if max_steps and self.step_count >= max_steps:
                break
            
            # Single simulation step
            self._step(render=render)
            
            # Check for quit (only if rendering)
            if render and self.renderer:
                if not self.renderer.handle_input(self.world):
                    self.running = False
        
        # Episode complete
        self.state = SimulatorState.STOPPED
        self.episode_stats.duration = (datetime.now() - start_time).total_seconds()
        
        # Run callbacks
        for callback in self.on_episode_end_callbacks:
            callback(self.episode_stats)
        
        return self.episode_stats
    
    def stop_episode(self) -> None:
        """Stop current episode."""
        self.running = False
        self.state = SimulatorState.STOPPED
    
    # ========================================================================
    # SIMULATION STEP
    # ========================================================================
    
    def _step(self, render: bool = True) -> None:
        """
        Execute single simulation step.
        
        Args:
            render: Whether to render this frame
        """
        # Get agent action (if hook registered)
        action = None
        if self.agent_action_hook:
            action = self.agent_action_hook()
        
        # Apply action (placeholder - agent integration point)
        if action is not None:
            # Action processing would happen here
            # (vehicle control not yet implemented)
            pass
        
        # Step physics
        self.physics_world.step(self.dt)
        
        # Get collision events
        collision_events = self.physics_world.get_collision_events()
        
        # Process collisions
        for event in collision_events:
            self.episode_stats.record_collision(event)
            
            # Call collision hook
            if self.agent_collision_hook:
                self.agent_collision_hook(event)
            
            # Call registered callbacks
            for callback in self.on_collision_callbacks:
                callback(event)
        
        # Get observation
        observation = self._get_observation()
        
        # Provide observation to agent
        if self.agent_observation_hook:
            self.agent_observation_hook(observation)
        
        # Call step hook
        if self.agent_step_hook:
            self.agent_step_hook(self.step_count)
        
        # Call step callbacks
        for callback in self.on_step_callbacks:
            callback(self.step_count)
        
        # Render (if enabled)
        if render and self.renderer:
            self.renderer.draw(self.world)
            self.renderer.limit_fps()
        
        # Update state
        self.step_count += 1
        self.episode_stats.total_steps += 1
    
    # ========================================================================
    # OBSERVATIONS
    # ========================================================================
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        Generate observation dict for agent.
        
        Returns:
            Observation dictionary
        """
        observation = {
            'step': self.step_count,
            'episode': self.episode_number,
            'world_state': {
                'zones': len(self.world.zones),
                'roads': len(self.world.roads),
                'obstacles': len(self.world.obstacles),
            },
            'physics_state': {
                'bodies': len(self.physics_world.space.bodies),
                'shapes': len(self.physics_world.space.shapes),
                'gravity': self.physics_world.space.gravity,
                'damping': self.physics_world.space.damping,
            },
            'collisions': {
                'total': self.episode_stats.total_collisions,
                'recent': self.episode_stats.collision_details[-5:] if self.episode_stats.collision_details else [],
            },
        }
        
        return observation
    
    # ========================================================================
    # QUERIES (for agent use)
    # ========================================================================
    
    def query_nearby_zones(self, position: np.ndarray, radius: float = 300.0) -> List[Dict]:
        """
        Query zones near position.
        
        Args:
            position: [x, y] in world coordinates
            radius: Search radius
        
        Returns:
            List of zone info dicts
        """
        zones = self.world.find_zones_in_radius(position, radius)
        return [z.get_semantic_data() for z in zones]
    
    def query_nearby_roads(self, position: np.ndarray, radius: float = 400.0) -> List[Dict]:
        """
        Query roads near position.
        
        Args:
            position: [x, y] in world coordinates
            radius: Search radius
        
        Returns:
            List of road info dicts
        """
        roads = self.world.find_nearby_roads(position, radius)
        return [r.get_semantic_data() for r in roads]
    
    def query_nearby_obstacles(self, position: np.ndarray, radius: float = 300.0) -> List[Dict]:
        """
        Query obstacles near position.
        
        Args:
            position: [x, y] in world coordinates
            radius: Search radius
        
        Returns:
            List of obstacle info dicts
        """
        obstacles = self.world.find_obstacles_near_point(position, radius)
        return [o.get_semantic_data() for o in obstacles]
    
    # ========================================================================
    # STATE & INFO
    # ========================================================================
    
    def get_state(self) -> str:
        """Get current simulator state."""
        return self.state.name
    
    def get_stats(self) -> EpisodeStats:
        """Get current episode statistics."""
        return self.episode_stats
    
    def summary(self) -> str:
        """Get simulator summary."""
        summary_lines = [
            f"Simulator State: {self.state.name}",
            f"World: {self.world.summary()}",
            f"Physics: {self.physics_world.summary()}",
            f"Episode: {self.episode_number}, Step: {self.step_count}",
            f"Stats: {self.episode_stats.summary()}",
        ]
        return "\n".join(summary_lines)
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    
    def quit(self) -> None:
        """Clean up and quit simulator."""
        self.running = False
        self.state = SimulatorState.IDLE
        
        if self.renderer:
            self.renderer.quit()
        
        self.physics_world.cleanup()


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def run_interactive_simulator(
    world: Optional[World] = None,
    max_episodes: int = 10,
    max_steps_per_episode: int = 1000
) -> None:
    """
    Run interactive simulator (no agent).
    
    Use keyboard controls in Pygame window:
    - Arrow keys: Camera pan
    - +/-: Zoom
    - R: Reset world
    - Q: Quit
    
    Args:
        world: World to simulate
        max_episodes: Maximum episodes to run
        max_steps_per_episode: Max steps per episode
    """
    simulator = Simulator(world=world, render=True)
    
    print("Starting interactive simulator...")
    print("Controls: Arrow keys = pan, +/- = zoom, R = reset, ESC = quit")
    
    for episode in range(max_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        stats = simulator.run_episode(
            max_steps=max_steps_per_episode,
            render=True
        )
        print(stats.summary())
        
        if not simulator.running:
            break
    
    simulator.quit()
    print("Simulator closed.")


def run_headless_simulator(
    num_episodes: int = 5,
    steps_per_episode: int = 1000
) -> List[EpisodeStats]:
    """
    Run simulator without rendering (fast for data collection).
    
    Args:
        num_episodes: Number of episodes to run
        steps_per_episode: Steps per episode
    
    Returns:
        List of episode statistics
    """
    simulator = Simulator(render=False)
    
    all_stats = []
    
    for episode in range(num_episodes):
        stats = simulator.run_episode(
            max_steps=steps_per_episode,
            render=False
        )
        all_stats.append(stats)
        
        print(f"Episode {episode + 1}: {stats.summary()}")
    
    simulator.quit()
    return all_stats
