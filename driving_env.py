"""
DrivingWorldEnv: Gymnasium environment for 2D driving simulation.

Cleanly integrates World, PhysicsWorld, Renderer, and Simulator.
Provides gym.Env interface with reset(), step(), and render().

Ready for integration with PPO, DQN, and other RL algorithms.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from dataclasses_core import World
from world_generator import WorldGenerator, GenerationConfig
from physics_world import PhysicsWorld, CollisionEvent, PhysicsCollisionType
from renderer import Renderer
from semantic_extensions import SemanticEffects, SemanticTelemetry


# ==============================================================================
# ENVIRONMENT CONSTANTS
# ==============================================================================

# Action space: continuous control (throttle, steering, brake)
# Throttle: [-1, 1] (negative = brake, positive = accelerate)
# Steering: [-1, 1] (negative = left, positive = right)
ACTION_DIM = 2

# Observation space (placeholder for now)
# Will include: vehicle state, nearby entities, sensor readings
OBS_DIM = 32  # Placeholder dimension


# ==============================================================================
# GYMNASIUM ENVIRONMENT
# ==============================================================================

class DrivingWorldEnv(gym.Env):
    """
    Gymnasium environment for 2D driving simulation.
    
    Integrates:
    - World: Semantic world data (zones, roads, obstacles)
    - PhysicsWorld: Physics simulation (Pymunk)
    - Renderer: Visualization (Pygame)
    - Simulator: Orchestration and telemetry
    
    Compatible with standard RL training frameworks (stable-baselines3, etc.)
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    
    def __init__(
        self,
        world: Optional[World] = None,
        render_mode: Optional[str] = None,
        render_size: Tuple[int, int] = (1200, 800),
        physics_fps: int = 60,
        max_episode_steps: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize DrivingWorldEnv.
        
        Args:
            world: Existing World instance (generates if None)
            render_mode: "human" for display, "rgb_array" for pixel array, None for no rendering
            render_size: (width, height) for rendering
            physics_fps: Physics simulation FPS
            max_episode_steps: Maximum steps per episode
            seed: Random seed for reproducibility
        """
        self.render_mode = render_mode
        self.render_size = render_size
        self.physics_fps = physics_fps
        self.max_episode_steps = max_episode_steps
        self.seed_value = seed
        
        # World and physics
        if world is None:
            config = GenerationConfig()
            generator = WorldGenerator(config, seed=seed or 42)
            self.world = generator.generate()
        else:
            self.world = world
        
        self.physics_world = PhysicsWorld(self.world)
        self.dt = 1.0 / physics_fps  # Physics timestep
        
        # Renderer (lazy initialization)
        self.renderer: Optional[Renderer] = None
        self._init_renderer_on_demand()
        
        # Semantic effects
        self.effects_calc = SemanticEffects(self.world)
        self.telemetry = SemanticTelemetry()
        
        # Episode state
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_collision_count = 0
        self.episode_penalty_accumulated = 0.0
        
        # Vehicle state (placeholder - no vehicle dynamics yet)
        # TODO: Implement vehicle body in physics
        self.vehicle_position = np.array([
            self.world.width / 2,
            self.world.height / 2,
        ], dtype=np.float32)
        self.vehicle_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.vehicle_heading = 0.0
        
        # Define action and observation spaces
        self._define_spaces()
        
        # Collision tracking
        self.recent_collisions: list = []
        self.physics_world.register_collision_callback(
            'begin',
            self._on_collision
        )
    
    # ========================================================================
    # GYMNASIUM INTERFACE
    # ========================================================================
    
    def _define_spaces(self) -> None:
        """Define action and observation spaces."""
        
        # Action space: continuous control
        # [throttle, steering]
        # throttle: [-1, 1] (negative=brake, positive=accelerate)
        # steering: [-1, 1] (negative=left, positive=right)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space (placeholder for now)
        # TODO: Implement full observation with vehicle state, sensors, etc.
        # Placeholder: 32-dimensional vector
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32
        )
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
        
        Returns:
            (observation, info) tuple
        """
        # Set seed
        if seed is not None:
            self.seed_value = seed
            # Regenerate world if seed provided
            if hasattr(self, 'world'):
                config = GenerationConfig()
                generator = WorldGenerator(config, seed=seed)
                self.world = generator.generate()
                
                # Recreate physics
                self.physics_world.cleanup()
                self.physics_world = PhysicsWorld(self.world)
                self.effects_calc = SemanticEffects(self.world)
                
                # Re-register collision callback
                self.physics_world.register_collision_callback(
                    'begin',
                    self._on_collision
                )
        
        # Reset vehicle position to world center
        self.vehicle_position = np.array([
            self.world.width / 2.0,
            self.world.height / 2.0,
        ], dtype=np.float32)
        self.vehicle_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.vehicle_heading = 0.0
        
        # Reset episode state
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_collision_count = 0
        self.episode_penalty_accumulated = 0.0
        self.recent_collisions = []
        
        # Reset telemetry
        self.telemetry = SemanticTelemetry()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute single environment step.
        
        Args:
            action: Action array [throttle, steering]
        
        Returns:
            (observation, reward, terminated, truncated, info) tuple
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        throttle, steering = action[0], action[1]
        
        # TODO: Apply throttle/steering to vehicle body when implemented
        # For now, this is a placeholder
        self._apply_action(throttle, steering)
        
        # Step physics
        self.physics_world.step(self.dt)
        
        # Update vehicle state (placeholder)
        self._update_vehicle_state()
        
        # Get semantic effects at vehicle position
        effects = self.effects_calc.get_effects_at_position(self.vehicle_position)
        penalty = self.effects_calc.get_zone_penalty_at_position(self.vehicle_position, self.dt)
        
        # Record telemetry
        self.telemetry.record_effects(self.episode_step, {
            'friction_multiplier': effects['friction_multiplier'],
            'penalty_accumulated': penalty,
        })
        
        # Accumulate penalty
        self.episode_penalty_accumulated += penalty
        
        # Calculate reward (placeholder - no RL logic here)
        reward = self._calculate_reward(throttle, steering, penalty)
        
        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self.episode_step >= self.max_episode_steps - 1
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Update episode state
        self.episode_step += 1
        self.episode_reward += reward
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render environment.
        
        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise
        """
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        if self.renderer:
            self.renderer.quit()
        
        self.physics_world.cleanup()
    
    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================
    
    def _init_renderer_on_demand(self) -> None:
        """Initialize renderer only if rendering is enabled."""
        if self.render_mode in ["human", "rgb_array"]:
            self.renderer = Renderer(
                width=self.render_size[0],
                height=self.render_size[1],
                fps=self.metadata["render_fps"],
                title="DrivingWorldEnv - RL",
                show_labels=True,
                show_grid=False
            )
            self.renderer.camera.reset(
                self.world.width,
                self.world.height,
                self.render_size[0],
                self.render_size[1]
            )
    
    def _apply_action(self, throttle: float, steering: float) -> None:
        """
        Apply action to vehicle.
        
        Args:
            throttle: Throttle input [-1, 1]
            steering: Steering input [-1, 1]
        """
        # TODO: When vehicle is implemented, apply to pymunk body
        # For now, update velocity based on action
        
        # Simple kinematic update (placeholder)
        max_speed = 100.0  # pixels/sec
        max_turn_rate = 5.0  # radians/sec
        
        # Update heading
        self.vehicle_heading += steering * max_turn_rate * self.dt
        
        # Update velocity
        target_speed = throttle * max_speed
        acceleration = 50.0  # pixels/sec^2
        
        current_speed = np.linalg.norm(self.vehicle_velocity)
        if abs(target_speed - current_speed) > acceleration * self.dt:
            direction = np.sign(target_speed - current_speed)
            current_speed += direction * acceleration * self.dt
        else:
            current_speed = target_speed
        
        self.vehicle_velocity = current_speed * np.array([
            np.cos(self.vehicle_heading),
            np.sin(self.vehicle_heading)
        ], dtype=np.float32)
    
    def _update_vehicle_state(self) -> None:
        """Update vehicle position and velocity."""
        # Update position
        self.vehicle_position += self.vehicle_velocity * self.dt
        
        # Clamp to world bounds
        self.vehicle_position[0] = np.clip(
            self.vehicle_position[0],
            0.0,
            float(self.world.width)
        )
        self.vehicle_position[1] = np.clip(
            self.vehicle_position[1],
            0.0,
            float(self.world.height)
        )
    
    def _on_collision(self, event: CollisionEvent) -> None:
        """Handle collision event."""
        self.recent_collisions.append({
            'step': self.episode_step,
            'type_a': event.collision_type_a.name,
            'type_b': event.collision_type_b.name,
            'damage': event.metadata_b.get('damage', 0) if event.metadata_b else 0,
        })
        self.episode_collision_count += 1
    
    def _calculate_reward(self, throttle: float, steering: float, penalty: float) -> float:
        """
        Calculate reward for this step.
        
        PLACEHOLDER: No RL-specific logic here.
        Agents will implement their own reward shaping.
        
        Args:
            throttle: Throttle action
            steering: Steering action
            penalty: Semantic penalty from zones
        
        Returns:
            Reward value
        """
        # Base reward: negative penalty
        reward = -penalty
        
        # Add collision penalty (if any)
        if self.recent_collisions:
            damage = self.recent_collisions[-1]['damage']
            reward -= damage
        
        # Small step cost (encourage efficiency)
        reward -= 0.01
        
        return float(reward)
    
    def _check_terminated(self) -> bool:
        """
        Check if episode should terminate.
        
        Returns:
            True if episode should end, False otherwise
        """
        # TODO: Implement termination conditions
        # Examples: vehicle out of bounds, too many collisions, etc.
        return False
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        PLACEHOLDER: Returns dummy observation.
        
        Returns:
            Observation vector
        """
        # Create dummy observation (32-dimensional)
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        
        # TODO: Implement proper observation space
        # Should include:
        # - Vehicle state (position, velocity, heading)
        # - Nearby entities (zones, roads, obstacles)
        # - Sensor readings (distance to obstacles, etc.)
        # - World state (number of entities, etc.)
        
        # For now, fill with normalized vehicle state
        if OBS_DIM >= 3:
            obs[0] = self.vehicle_position[0] / self.world.width
            obs[1] = self.vehicle_position[1] / self.world.height
            obs[2] = self.vehicle_heading / (2 * np.pi)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get info dict.
        
        Returns:
            Info dictionary with debug/status information
        """
        info = {
            'step': self.episode_step,
            'vehicle_position': self.vehicle_position.copy(),
            'vehicle_velocity': self.vehicle_velocity.copy(),
            'vehicle_heading': float(self.vehicle_heading),
            'episode_reward': float(self.episode_reward),
            'collisions': self.episode_collision_count,
            'penalty_accumulated': float(self.episode_penalty_accumulated),
            'recent_collisions': self.recent_collisions[-5:],  # Last 5 collisions
        }
        
        return info
    
    def _render_human(self) -> None:
        """Render to display (human-visible)."""
        if self.renderer:
            self.renderer.draw(self.world)
            self.renderer.limit_fps()
    
    def _render_rgb_array(self) -> Optional[np.ndarray]:
        """
        Render to RGB array (for recording/analysis).
        
        Returns:
            RGB pixel array or None
        """
        if not self.renderer:
            return None
        
        # TODO: Implement pixel array extraction from Pygame surface
        # For now, just render to display
        self._render_human()
        
        # Convert pygame surface to numpy array
        surface = self.renderer.screen
        if surface:
            # Create RGB array from pygame surface
            # (implementation would depend on pygame version)
            pixels = np.transpose(
                pygame.surfarray.array3d(surface),
                axes=(1, 0, 2)
            )
            return pixels
        
        return None


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def make_driving_env(
    render_mode: Optional[str] = None,
    max_episode_steps: int = 1000,
    seed: Optional[int] = None,
) -> DrivingWorldEnv:
    """
    Create a DrivingWorldEnv instance.
    
    Convenience function for creating environments.
    
    Args:
        render_mode: "human", "rgb_array", or None
        max_episode_steps: Maximum steps per episode
        seed: Random seed
    
    Returns:
        DrivingWorldEnv instance
    """
    return DrivingWorldEnv(
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )


# ==============================================================================
# GYMNASIUM WRAPPER (for compatibility)
# ==============================================================================

def create_env_with_wrapper(
    render_mode: Optional[str] = None,
    max_episode_steps: int = 1000,
) -> gym.Env:
    """
    Create environment with optional gymnasium wrappers.
    
    Useful for integrating with stable-baselines3 and other RL libraries.
    
    Args:
        render_mode: Render mode
        max_episode_steps: Max steps per episode
    
    Returns:
        Wrapped environment
    """
    env = DrivingWorldEnv(
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )
    
    # Optionally add wrappers
    # from gymnasium.wrappers import TimeLimit, NormalizeObservation, NormalizeReward
    # env = TimeLimit(env, max_episode_steps=max_episode_steps)
    # env = NormalizeObservation(env)
    # env = NormalizeReward(env)
    
    return env
