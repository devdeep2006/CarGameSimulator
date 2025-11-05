"""
Renderer: Pygame-based visualization for World objects.

Pure visualization layer - completely separate from physics, logic, or data model.
Supports panning, zooming, efficient rendering with caching, and semantic visualization.

Key features:
- Clean separation from world model and physics
- Efficient drawing with layer caching
- Camera control (pan/zoom)
- Color-coded entity types with optional labels
- Frame-based rendering with configurable FPS
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from enum import Enum
import numpy as np
import pygame
import math

from dataclasses_core import World, Zone, Road, Obstacle
from dataclasses_core import TerrainType, RoadType, ObstacleType


# ==============================================================================
# COLOR DEFINITIONS
# ==============================================================================

@dataclass
class ColorScheme:
    """Color scheme for rendering different entity types."""
    
    # Terrain types
    terrain_colors: Dict[TerrainType, Tuple[int, int, int]] = field(default_factory=lambda: {
        TerrainType.ASPHALT: (64, 64, 64),        # Dark gray
        TerrainType.GRASS: (34, 139, 34),         # Forest green
        TerrainType.WATER: (30, 144, 255),        # Dodger blue
        TerrainType.GRAVEL: (160, 160, 160),      # Light gray
        TerrainType.CONCRETE: (192, 192, 192),    # Silver
        TerrainType.SAND: (238, 214, 175),        # Burlywood
    })
    
    # Road types
    road_colors: Dict[RoadType, Tuple[int, int, int]] = field(default_factory=lambda: {
        RoadType.HIGHWAY: (255, 165, 0),          # Orange
        RoadType.MAIN_STREET: (255, 215, 0),      # Gold
        RoadType.RESIDENTIAL: (205, 133, 63),     # Peru
        RoadType.SECONDARY: (184, 134, 11),       # Dark goldenrod
        RoadType.PARKING: (119, 136, 153),        # Light slate gray
    })
    
    # Obstacle types
    obstacle_colors: Dict[ObstacleType, Tuple[int, int, int]] = field(default_factory=lambda: {
        ObstacleType.WALL: (139, 69, 19),         # Saddle brown
        ObstacleType.TREE: (34, 139, 34),         # Forest green
        ObstacleType.ROCK: (128, 128, 128),       # Gray
        ObstacleType.BUILDING: (139, 0, 0),       # Dark red
        ObstacleType.FENCE: (160, 82, 45),        # Sienna
        ObstacleType.CAR: (255, 0, 0),            # Red
        ObstacleType.DEBRIS: (169, 169, 169),     # Dark gray
    })
    
    # UI colors
    background: Tuple[int, int, int] = (20, 20, 30)  # Dark blue-black
    text: Tuple[int, int, int] = (255, 255, 255)     # White
    grid: Tuple[int, int, int] = (80, 80, 80)        # Gray
    selection: Tuple[int, int, int] = (255, 255, 0)  # Yellow


# ==============================================================================
# CAMERA CONTROL
# ==============================================================================

@dataclass
class Camera:
    """Camera for panning and zooming."""
    
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0
    min_zoom: float = 0.1
    max_zoom: float = 10.0
    pan_speed: float = 10.0
    zoom_speed: float = 0.1
    
    def pan(self, dx: float, dy: float) -> None:
        """Pan camera by (dx, dy)."""
        self.x += dx * self.pan_speed
        self.y += dy * self.pan_speed
    
    def zoom_in(self) -> None:
        """Zoom in."""
        self.zoom = min(self.zoom + self.zoom_speed, self.max_zoom)
    
    def zoom_out(self) -> None:
        """Zoom out."""
        self.zoom = max(self.zoom - self.zoom_speed, self.min_zoom)
    
    def reset(self, world_width: int, world_height: int, screen_width: int, screen_height: int) -> None:
        """Reset camera to center world view."""
        self.x = (world_width - screen_width / self.zoom) / 2
        self.y = (world_height - screen_height / self.zoom) / 2
        self.zoom = 1.0
    
    def world_to_screen(self, x: float, y: float, screen_width: int, screen_height: int) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int((x - self.x) * self.zoom)
        screen_y = int((y - self.y) * self.zoom)
        return (screen_x, screen_y)
    
    def screen_to_world(self, screen_x: int, screen_y: int, screen_width: int, screen_height: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        world_x = (screen_x / self.zoom) + self.x
        world_y = (screen_y / self.zoom) + self.y
        return (world_x, world_y)


# ==============================================================================
# RENDERER
# ==============================================================================

class Renderer:
    """
    Pygame-based renderer for World visualization.
    
    Pure visualization layer - completely decoupled from world model.
    Handles rendering, camera control, and UI overlays.
    """
    
    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
        fps: int = 60,
        title: str = "Driving Simulator",
        color_scheme: Optional[ColorScheme] = None,
        show_grid: bool = False,
        show_labels: bool = True,
        label_distance: int = 50
    ) -> None:
        """
        Initialize renderer.
        
        Args:
            width: Screen width in pixels
            height: Screen height in pixels
            fps: Target frames per second
            title: Window title
            color_scheme: Custom color scheme (uses defaults if None)
            show_grid: Show background grid
            show_labels: Show entity labels
            label_distance: Minimum pixels to show label (zoom-dependent)
        """
        pygame.init()
        
        self.width = width
        self.height = height
        self.fps = fps
        self.show_grid = show_grid
        self.show_labels = show_labels
        self.label_distance = label_distance
        
        # Initialize display
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_small = pygame.font.Font(None, 16)
        self.font_large = pygame.font.Font(None, 24)
        
        # Colors and styling
        self.colors = color_scheme or ColorScheme()
        
        # Camera
        self.camera = Camera()
        
        # Stats
        self.show_stats = True
        self.frame_count = 0
        self.fps_value = 0.0
    
    # ========================================================================
    # DRAWING METHODS
    # ========================================================================
    
    def draw(self, world: World) -> None:
        """
        Draw entire world.
        
        Args:
            world: World instance to render
        """
        # Clear screen
        self.screen.fill(self.colors.background)
        
        # Draw background grid (optional)
        if self.show_grid:
            self._draw_grid(world)
        
        # Draw world layers (back to front)
        self._draw_zones(world)
        self._draw_roads(world)
        self._draw_obstacles(world)
        
        # Draw UI overlay
        if self.show_stats:
            self._draw_stats()
        
        # Update display
        pygame.display.flip()
        self.frame_count += 1
        self.fps_value = self.clock.get_fps()
    
    def _draw_grid(self, world: World) -> None:
        """Draw background grid."""
        grid_spacing = 100
        
        for x in range(0, world.width, grid_spacing):
            start_screen = self.camera.world_to_screen(x, 0, self.width, self.height)
            end_screen = self.camera.world_to_screen(x, world.height, self.width, self.height)
            pygame.draw.line(
                self.screen,
                self.colors.grid,
                start_screen,
                end_screen,
                1
            )
        
        for y in range(0, world.height, grid_spacing):
            start_screen = self.camera.world_to_screen(0, y, self.width, self.height)
            end_screen = self.camera.world_to_screen(world.width, y, self.width, self.height)
            pygame.draw.line(
                self.screen,
                self.colors.grid,
                start_screen,
                end_screen,
                1
            )
    
    def _draw_zones(self, world: World) -> None:
        """Draw zones."""
        for zone in world.zones.values():
            color = self.colors.terrain_colors.get(zone.terrain_type, (128, 128, 128))
            
            # Get bounds
            x_min, y_min, x_max, y_max = zone.get_bounds()
            width = x_max - x_min
            height = y_max - y_min
            
            # Convert to screen coordinates
            screen_pos = self.camera.world_to_screen(x_min, y_min, self.width, self.height)
            screen_width = int(width * self.camera.zoom)
            screen_height = int(height * self.camera.zoom)
            
            # Draw rectangle with transparency
            rect = pygame.Rect(screen_pos[0], screen_pos[1], screen_width, screen_height)
            pygame.draw.rect(self.screen, color, rect, width=2)
            
            # Draw label if enabled and zoomed enough
            if self.show_labels and self.camera.zoom > 0.5:
                label_surface = self.font_small.render(zone.name, True, self.colors.text)
                label_pos = (screen_pos[0] + 5, screen_pos[1] + 5)
                self.screen.blit(label_surface, label_pos)
    
    def _draw_roads(self, world: World) -> None:
        """Draw roads."""
        for road in world.roads.values():
            color = self.colors.road_colors.get(road.road_type, (255, 255, 255))
            
            # Convert to screen coordinates
            start_screen = self.camera.world_to_screen(
                road.start_point[0], road.start_point[1],
                self.width, self.height
            )
            end_screen = self.camera.world_to_screen(
                road.end_point[0], road.end_point[1],
                self.width, self.height
            )
            
            # Draw road as thick line
            width = max(1, int(road.width * self.camera.zoom))
            pygame.draw.line(self.screen, color, start_screen, end_screen, width)
            
            # Draw direction arrow
            if road.length > 50:
                mid = (
                    (road.start_point + road.end_point) / 2
                )
                mid_screen = self.camera.world_to_screen(
                    mid[0], mid[1],
                    self.width, self.height
                )
                
                # Draw arrow at midpoint
                arrow_size = 8
                direction = road.direction
                perp = np.array([-direction[1], direction[0]])
                
                arrow_end = np.array(mid_screen) + direction * arrow_size * self.camera.zoom
                pygame.draw.line(
                    self.screen,
                    (255, 255, 255),
                    mid_screen,
                    tuple(arrow_end.astype(int)),
                    2
                )
            
            # Draw label if enabled
            if self.show_labels and self.camera.zoom > 0.7:
                mid = (road.start_point + road.end_point) / 2
                mid_screen = self.camera.world_to_screen(mid[0], mid[1], self.width, self.height)
                label_surface = self.font_small.render(road.name, True, self.colors.text)
                label_pos = (mid_screen[0] + 5, mid_screen[1] + 5)
                self.screen.blit(label_surface, label_pos)
    
    def _draw_obstacles(self, world: World) -> None:
        """Draw obstacles."""
        for obstacle in world.obstacles.values():
            color = self.colors.obstacle_colors.get(obstacle.obstacle_type, (100, 100, 100))
            
            # Convert to screen coordinates
            screen_pos = self.camera.world_to_screen(
                obstacle.position[0], obstacle.position[1],
                self.width, self.height
            )
            
            # Draw obstacle as rectangle
            screen_width = int(obstacle.width * self.camera.zoom)
            screen_height = int(obstacle.height * self.camera.zoom)
            
            rect = pygame.Rect(
                screen_pos[0] - screen_width // 2,
                screen_pos[1] - screen_height // 2,
                screen_width,
                screen_height
            )
            
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)  # Border
            
            # Draw label if enabled
            if self.show_labels and self.camera.zoom > 0.8:
                label_surface = self.font_small.render(obstacle.name, True, (0, 0, 0))
                label_pos = (
                    screen_pos[0] - label_surface.get_width() // 2,
                    screen_pos[1] - label_surface.get_height() // 2
                )
                self.screen.blit(label_surface, label_pos)
    
    def _draw_stats(self) -> None:
        """Draw statistics overlay."""
        stats = [
            f"FPS: {self.fps_value:.1f}",
            f"Zoom: {self.camera.zoom:.2f}x",
            f"Pos: ({self.camera.x:.0f}, {self.camera.y:.0f})",
            "Press R to reset | +/- to zoom | Arrow keys to pan",
        ]
        
        y_offset = 10
        for stat in stats:
            surface = self.font_small.render(stat, True, self.colors.text)
            self.screen.blit(surface, (10, y_offset))
            y_offset += 20
    
    # ========================================================================
    # CAMERA & INPUT CONTROL
    # ========================================================================
    
    def handle_input(self, world: World) -> bool:
        """
        Handle user input for camera control.
        
        Returns:
            False if user closed window, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.camera.pan(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    self.camera.pan(1, 0)
                elif event.key == pygame.K_UP:
                    self.camera.pan(0, -1)
                elif event.key == pygame.K_DOWN:
                    self.camera.pan(0, 1)
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.camera.zoom_in()
                elif event.key == pygame.K_MINUS:
                    self.camera.zoom_out()
                elif event.key == pygame.K_r:
                    self.camera.reset(world.width, world.height, self.width, self.height)
                elif event.key == pygame.K_s:
                    self.show_stats = not self.show_stats
                elif event.key == pygame.K_l:
                    self.show_labels = not self.show_labels
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
        
        return True
    
    def limit_fps(self) -> None:
        """Limit frame rate."""
        self.clock.tick(self.fps)
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    
    def quit(self) -> None:
        """Clean up and close renderer."""
        pygame.quit()


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def render_world_interactive(world: World, fps: int = 60, title: str = "Driving Simulator") -> None:
    """
    Render world with interactive camera control.
    
    Controls:
    - Arrow keys: Pan camera
    - +/-: Zoom in/out
    - R: Reset view
    - S: Toggle statistics
    - L: Toggle labels
    - G: Toggle grid
    - ESC/Close: Quit
    
    Args:
        world: World to render
        fps: Target FPS
        title: Window title
    """
    renderer = Renderer(fps=fps, title=title)
    renderer.camera.reset(world.width, world.height, renderer.width, renderer.height)
    
    running = True
    while running:
        running = renderer.handle_input(world)
        renderer.draw(world)
        renderer.limit_fps()
    
    renderer.quit()
