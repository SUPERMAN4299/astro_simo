"""
constants.py — Simulation Constants & Configuration
════════════════════════════════════════════════════

All tunable parameters live here. No magic numbers elsewhere.

SCALING RATIONALE
─────────────────
Real Earth:  GM  = 3.986 × 10¹⁴ m³/s²,  R = 6.371 × 10⁶ m
Sim Earth:   GM  = 2.5  (sim units),      R = 0.25   (sim units)

Scale factor k = R_sim / R_real = 0.25 / 6.371e6 ≈ 3.93e-8
A circular LEO orbit (~400 km altitude, r ≈ 1.063 R_earth) maps to
r_sim ≈ 0.266.  We use r=2.0 for clarity; the physics ratios are exact.
"""

# ── Gravitational parameters ─────────────────────────────────────────
GM_EARTH: float  = 2.5       # G × M_earth  (sim units)
GM_MOON:  float  = 0.031     # G × M_moon   (~1/81 of Earth's GM in reality)

# ── Body radii (sim units) ───────────────────────────────────────────
EARTH_RADIUS:    float = 0.25   # collision / render radius
MOON_RADIUS:     float = 0.068  # ~27% of Earth radius (realistic ratio)
SATELLITE_SCALE: float = 0.06   # render scale for satellite mesh

# ── Moon orbital parameters ──────────────────────────────────────────
MOON_ORBIT_RADIUS:   float = 4.5    # sim units (realistic ratio ≈ 60 × R_earth)
MOON_ORBIT_PERIOD:   float = 60.0   # sim seconds (compressed for visibility)
MOON_INCLINATION_DEG: float = 5.145 # realistic orbital inclination (degrees)

# ── Simulation bounds ────────────────────────────────────────────────
ESCAPE_DISTANCE: float = 18.0   # beyond this → satellite escaped
MIN_ALTITUDE:    float = 0.01   # minimum clearance above Earth surface

# ── Integrator settings ──────────────────────────────────────────────
DT_BASE:          float = 0.008  # base physics timestep (sim seconds)
STEPS_PER_FRAME:  int   = 8      # Verlet sub-steps per render frame
DT_MULT_MIN:      float = 0.1
DT_MULT_MAX:      float = 10.0
# Adaptive timestep: dt is halved when v > VEL_THRESHOLD or r < R_THRESHOLD
VEL_THRESHOLD:    float = 8.0    # sim units / sim second
R_THRESHOLD:      float = 0.5    # sim units (close-approach guard)

# ── Collision response ────────────────────────────────────────────────
COLLISION_RESTITUTION: float = 0.35   # bounce coefficient (0 = stop, 1 = elastic)

# ── Thruster ─────────────────────────────────────────────────────────
THRUST_ACCEL:    float = 0.15   # sim units/s² (continuous thrust)
IMPULSE_DV:      float = 0.08   # instantaneous Δv per key-press (sim units/s)

# ── Trail settings ───────────────────────────────────────────────────
TRAIL_MAX_POINTS: int   = 2400
TRAIL_SAMPLE_INTERVAL: int = 2   # record every N physics steps (downsampling)

# ── Hohmann transfer ─────────────────────────────────────────────────
# Δv₁ and Δv₂ are computed analytically; this just guards the minimum orbit r
HOHMANN_MIN_TARGET_R: float = 0.35

# ── Window / renderer ────────────────────────────────────────────────
WINDOW_WIDTH:  int   = 1280
WINDOW_HEIGHT: int   = 800
WINDOW_TITLE:  str   = "Orbital Mechanics Engine v3 — Perturbations"
CAMERA_NEAR:   float = 0.01
CAMERA_FAR:    float = 200.0
CAMERA_FOV:    float = 45.0
MSAA_SAMPLES:  int   = 4

# ── Perturbations ─────────────────────────────────────────────────────
# J2/J4 scaled ×500 so nodal precession is visible in sim time
PERTURB_J2:              float = 1.08263e-3 * 500.0
PERTURB_J4:              float = -1.60e-6   * 500.0
PERTURB_DRAG_RHO0:       float = 8.0e-3     # sea-level density (sim units)
PERTURB_DRAG_H:          float = 0.50       # scale height (sim units)
PERTURB_DRAG_CdAm:       float = 0.04       # ballistic coefficient
PERTURB_SRP_P:           float = 3.0e-5     # solar radiation pressure
PERTURB_SRP_CrAm:        float = 0.010      # reflectivity × area/mass
PERTURB_SUN_GM:          float = 800.0      # Sun GM (sim units)
PERTURB_SUN_DIST:        float = 53.75      # Earth-Sun distance (sim units)
PERTURB_SUN_ROT_PERIOD:  float = 120.0      # Sun rotation period (sim seconds)