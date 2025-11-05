from world_generator import generate_world
from simulator import run_interactive_simulator

print("🎮 DRIVING SIMULATION GAME")
print("\nControls:")
print("  Arrow Keys: Move camera")
print("  +/-: Zoom in/out")
print("  R: Reset view")
print("  L: Toggle labels")
print("  G: Toggle grid")
print("  S: Toggle stats")
print("  ESC: Quit\n")

print("Generating world...")
world = generate_world(seed=42, num_zones=8, num_roads=5, num_obstacles=20)

print(f"✓ {world.summary()}\n")
print("Starting game... Press ESC to quit\n")

run_interactive_simulator(world)

print("✓ Game ended")
