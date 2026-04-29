import pygame
import math

# 1. SETUP: Start the engine and create the window
pygame.init()
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Colors (Red, Green, Blue)
WHITE = (255, 255, 255)
BLUE  = (100, 149, 237)
RED   = (255, 100, 100)
BLACK = (0, 0, 0)

# 2. THE RULES: Physics constants
G = 0.5           # How strong gravity is
EARTH_MASS = 5000 # How heavy Earth is (more mass = stronger pull)
dt = 0.1          # "Delta Time" - how much time passes in each frame

# 3. THE PLAYERS: Earth and Satellite
earth_pos = pygame.Vector2(WIDTH // 2, HEIGHT // 2) # Center of screen
sat_pos   = pygame.Vector2(WIDTH // 2 + 200, HEIGHT // 2) # Start 200px to the right

# --- THE SECRET SAUCE: Perfect Orbit Math ---
# To stay in orbit, the satellite needs a specific speed.
# We find the distance (r), then use the formula: v = sqrt(GM/r)
r_vec = sat_pos - earth_pos
r = r_vec.length()
v_mag = math.sqrt(G * EARTH_MASS / r)

# We want the speed to point UP (perpendicular to Earth) so it doesn't just crash
# Normalize makes the direction 1 unit long, then we multiply by our speed
sat_vel = pygame.Vector2(-r_vec.y, r_vec.x).normalize() * (v_mag * 0.8)

# A list to store old positions so we can draw a line behind the satellite
trail = []

# 4. THE GAME LOOP: This runs 60 times every second
running = True
while running:
    clock.tick(60) # Keep the game at 60 FPS
    screen.fill(BLACK) # Clear the screen so we don't see old frames

    # Check if the user clicked the [X] button
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- STEP A: Calculate Gravity ---
    # Find the vector pointing from Satellite to Earth
    to_earth = earth_pos - sat_pos
    distance = to_earth.length()

    if distance > 0:
        # Gravity Formula: (G * M) / distance squared
        force_magnitude = (G * EARTH_MASS) / (distance**2)
        # Pull the satellite toward Earth
        acceleration = to_earth.normalize() * force_magnitude
    else:
        acceleration = pygame.Vector2(0, 0)

    # --- STEP B: Update Movement ---
    sat_vel += acceleration * dt  # Gravity changes the velocity
    sat_pos += sat_vel * dt       # Velocity changes the position

    # --- STEP C: Draw the Trail ---
    trail.append((int(sat_pos.x), int(sat_pos.y)))
    if len(trail) > 600: # Limit trail length so the computer doesn't lag
        trail.pop(0)

    if len(trail) > 2:
        pygame.draw.lines(screen, WHITE, False, trail, 1)

    # --- STEP D: Draw Earth and Satellite ---
    pygame.draw.circle(screen, BLUE, (int(earth_pos.x), int(earth_pos.y)), 20) # Earth
    pygame.draw.circle(screen, RED, (int(sat_pos.x), int(sat_pos.y)), 5)       # Satellite

    # Update the actual monitor
    pygame.display.flip()

pygame.quit()