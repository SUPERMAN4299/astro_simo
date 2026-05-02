"""
physics.py — 3D Orbital Mechanics Engine
═════════════════════════════════════════

All positions are in simulation units (1 unit ≈ Earth radius in concept).
G·M is tuned so circular orbit at r=2.0 has period ≈ a few seconds — fast
enough to be visually satisfying without being a blur.

GRAVITY in 3D:
  a = -GM/|r|³ · r_vec      (inverse square, full 3-vector)
  Note the |r|³ denominator: |r|² for magnitude, |r| to normalize.

VELOCITY VERLET in 3D (identical math, just np.array):
  r_new = r + v·dt + ½·a·dt²
  a_new = gravity(r_new)
  v_new = v + ½·(a + a_new)·dt

ANGULAR MOMENTUM (conserved → Kepler 2nd Law):
  L = r × v    (cross product — always perpendicular to orbital plane)
  |L| = const  → speed increases when |r| decreases
"""

import numpy as np
import math


# ── Constants ────────────────────────────────────────────────────────
GM         = 2.5        # G × M_earth (sim units) — tuned for good visual speed
EARTH_R    = 0.25       # Earth sphere radius (render units)
ESCAPE_DIST = 15.0      # Beyond this → satellite "escaped"
TRAIL_MAX   = 2000      # Max trail points stored


# ── Presets ─────────────────────────────────────────────────────────
#   Each preset: (pos, vel) in sim units
#   Circular v = sqrt(GM/r) perpendicular to radius
def _circular_vel(pos):
    r     = np.linalg.norm(pos)
    v_mag = math.sqrt(GM / r)
    # Perpendicular in XY plane, then normalize
    perp = np.array([-pos[1], pos[0], 0.0])
    return perp / np.linalg.norm(perp) * v_mag

PRESETS = {
    "circular": {
        "pos": np.array([2.0, 0.0, 0.0]),
        "vel": lambda p: _circular_vel(p),
    },
    "elliptical": {
        "pos": np.array([2.0, 0.0, 0.0]),
        "vel": lambda p: _circular_vel(p) * 0.72,   # slower → more eccentric ellipse
    },
    "escape": {
        "pos": np.array([2.0, 0.0, 0.0]),
        "vel": lambda p: _circular_vel(p) * 1.42,   # √2 × v_circ = escape
    },
    "inclined": {
        "pos": np.array([2.0, 0.0, 0.0]),
        "vel": lambda p: np.array([0.0,
                                    math.sqrt(GM / np.linalg.norm(p)) * math.cos(math.radians(45)),
                                    math.sqrt(GM / np.linalg.norm(p)) * math.sin(math.radians(45))]),
    },
}


# ── Orbital Body ─────────────────────────────────────────────────────
class OrbitalBody:
    """
    A single satellite in 3D space.

    Attributes:
        pos  (np.array shape 3): position
        vel  (np.array shape 3): velocity
        acc  (np.array shape 3): current acceleration
        trail (list of np.array): historical positions (capped)
        escaped (bool): True when satellite leaves simulation bounds
    """

    def __init__(self, pos: np.ndarray, vel: np.ndarray):
        self.pos     = pos.astype(np.float64).copy()
        self.vel     = vel.astype(np.float64).copy()
        self.acc     = np.zeros(3, dtype=np.float64)
        self.trail   = []
        self.escaped = False
        self._compute_acc()

    # ── Gravity ────────────────────────────────────────────────────
    def _compute_acc(self):
        """
        Newtonian gravity toward origin (Earth center).
        a = -GM / |r|³  ×  r_vec
        Using |r|³ = |r|² × |r| avoids an extra sqrt for normalization.
        """
        r2 = float(np.dot(self.pos, self.pos))   # |r|²
        if r2 < 1e-6:
            self.acc = np.zeros(3)
            return
        r      = math.sqrt(r2)
        factor = -GM / (r2 * r)                  # -GM / |r|³
        self.acc = self.pos * factor

    # ── Integration: Velocity Verlet ───────────────────────────────
    def step(self, dt: float):
        """
        Velocity Verlet — symplectic, energy-conserving.
        Works identically in 3D (just numpy array math).
        """
        a0 = self.acc.copy()

        # Update position
        self.pos += self.vel * dt + 0.5 * a0 * dt * dt

        # Recompute acceleration at new position
        self._compute_acc()

        # Update velocity using average of old + new acceleration
        self.vel += 0.5 * (a0 + self.acc) * dt

        # Record trail
        self.trail.append(self.pos.copy())
        if len(self.trail) > TRAIL_MAX:
            self.trail.pop(0)

        # Check escape
        if np.linalg.norm(self.pos) > ESCAPE_DIST:
            self.escaped = True

    def clear_trail(self):
        self.trail.clear()
        self.escaped = False

    # ── Derived quantities ─────────────────────────────────────────
    def speed(self):
        return float(np.linalg.norm(self.vel))

    def distance(self):
        return float(np.linalg.norm(self.pos))

    def kinetic_energy(self):
        return 0.5 * float(np.dot(self.vel, self.vel))   # ½|v|²

    def potential_energy(self):
        r = self.distance()
        return -GM / r if r > 1e-6 else -1e9             # -GM/r

    def total_energy(self):
        return self.kinetic_energy() + self.potential_energy()

    def circular_speed(self):
        r = self.distance()
        return math.sqrt(GM / r) if r > 1e-6 else 0.0

    def escape_speed(self):
        r = self.distance()
        return math.sqrt(2 * GM / r) if r > 1e-6 else 0.0

    def angular_momentum(self):
        """L = r × v  (vector, perpendicular to orbital plane)."""
        return np.cross(self.pos, self.vel)

    def orbit_type(self):
        e = self.total_energy()
        if e < -0.05:
            v_c = self.circular_speed()
            ratio = self.speed() / v_c if v_c > 1e-6 else 1.0
            if abs(ratio - 1.0) < 0.05:
                return "CIRCULAR"
            return "ELLIPTICAL"
        return "ESCAPE"

    # ── Velocity manipulation ──────────────────────────────────────
    def rotate_vel_in_plane(self, deg: float):
        """
        Rotate velocity around the angular momentum axis (L = r×v).
        This keeps the satellite in its orbital plane while changing direction.
        """
        L = self.angular_momentum()
        Ln = np.linalg.norm(L)
        if Ln < 1e-10:
            # Fallback: rotate around Z
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = L / Ln
        self.vel = _rotate_vec(self.vel, axis, math.radians(deg))

    def tilt_orbit(self, deg: float):
        """
        Tilt velocity out of its current orbital plane.
        Adds a component perpendicular to both r and v (i.e., along L direction),
        then re-normalizes to the same speed.
        """
        spd  = self.speed()
        perp = np.cross(self.pos, self.vel)   # ≈ L direction
        pn   = np.linalg.norm(perp)
        if pn < 1e-10:
            return
        perp /= pn
        tilt  = math.radians(deg)
        self.vel = self.vel + perp * (spd * math.sin(tilt))
        # Re-normalize to preserve speed
        new_spd = np.linalg.norm(self.vel)
        if new_spd > 1e-8:
            self.vel = self.vel * (spd / new_spd)

    def snap_circular(self):
        """Set speed to exact circular orbit speed, keeping direction."""
        v_c = self.circular_speed()
        spd = self.speed()
        if spd > 1e-8:
            self.vel = self.vel * (v_c / spd)
        self.clear_trail()

    def snap_escape(self):
        """Set speed to exact escape velocity, keeping direction."""
        v_e = self.escape_speed()
        spd = self.speed()
        if spd > 1e-8:
            self.vel = self.vel * (v_e / spd)
        self.clear_trail()


# ── Helper ────────────────────────────────────────────────────────────
def _rotate_vec(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rodrigues' rotation formula:
      v_rot = v·cos(θ) + (axis × v)·sin(θ) + axis·(axis·v)·(1 - cos(θ))
    Rotates vector v by angle around unit axis.
    """
    c = math.cos(angle)
    s = math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * (np.dot(axis, v) * (1 - c))


# ── Physics Engine ────────────────────────────────────────────────────
class PhysicsEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.load_preset("circular")

    def load_preset(self, name: str):
        if name not in PRESETS:
            return
        p   = PRESETS[name]
        pos = p["pos"].copy()
        vel = p["vel"](pos)
        self.satellite = OrbitalBody(pos, vel)

    def step(self, dt: float):
        self.satellite.step(dt)
