"""
physics.py — Production Orbital Mechanics Engine (v2)
══════════════════════════════════════════════════════

Architecture
────────────
  GravitationalBody   – a massive body (Earth, Moon) with fixed or dynamic orbit
  Satellite           – the controlled spacecraft; integrates under N-body gravity
  ThrusterSystem      – manages continuous and impulse thrust inputs
  OrbitalElements     – data-class for Keplerian elements derived from state vectors
  PhysicsEngine       – top-level coordinator; owns all bodies, steps the sim

Integration: Velocity Verlet (symplectic)
─────────────────────────────────────────
  r(t+dt) = r(t) + v(t)·dt + ½·a(t)·dt²
  a(t+dt) = ΣF(r(t+dt)) / m
  v(t+dt) = v(t) + ½·[a(t) + a(t+dt)]·dt

  Symplectic integrators conserve a modified Hamiltonian → no secular energy
  drift even over thousands of orbits.

Adaptive Timestep
─────────────────
  dt_eff = dt_base / max(1, v/V_THRESH, R_THRESH/r)
  Near Earth or at high speed the timestep is automatically halved.

Orbital Elements (from state vectors)
──────────────────────────────────────
  h     = r × v
  e_vec = (v×h)/GM - r̂     (Laplace-Runge-Lenz vector)
  e     = |e_vec|
  a     = -GM / (2ε)        (vis-viva)
  i     = acos(h_y / |h|)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from perturbations import PerturbationEngine, PerturbationConfig
from constants import (
    GM_EARTH, GM_MOON,
    EARTH_RADIUS, MOON_RADIUS,
    MOON_ORBIT_RADIUS, MOON_ORBIT_PERIOD, MOON_INCLINATION_DEG,
    ESCAPE_DISTANCE, MIN_ALTITUDE,
    VEL_THRESHOLD, R_THRESHOLD,
    COLLISION_RESTITUTION,
    THRUST_ACCEL, IMPULSE_DV,
    TRAIL_MAX_POINTS, TRAIL_SAMPLE_INTERVAL,
    HOHMANN_MIN_TARGET_R,
)


# ── Math helpers ──────────────────────────────────────────────────────

def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def _unit(v: np.ndarray) -> np.ndarray:
    n = _norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)

def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation: rotate v by angle radians around unit axis."""
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1 - c)


# ══════════════════════════════════════════════════════════════════════
# ORBITAL ELEMENTS
# ══════════════════════════════════════════════════════════════════════

@dataclass
class OrbitalElements:
    """Keplerian orbital elements derived from Cartesian state vectors."""
    semi_major_axis:   float = 0.0
    eccentricity:      float = 0.0
    inclination_deg:   float = 0.0
    raan_deg:          float = 0.0
    arg_periapsis_deg: float = 0.0
    true_anomaly_deg:  float = 0.0
    periapsis_r:       float = 0.0
    apoapsis_r:        float = float('inf')
    period:            float = float('inf')
    orbit_type:        str   = "UNKNOWN"

    @classmethod
    def from_state(cls, pos: np.ndarray, vel: np.ndarray, gm: float) -> "OrbitalElements":
        """Derive all elements from position and velocity vectors under GM."""
        oe    = cls()
        r_mag = _norm(pos)
        v_mag = _norm(vel)
        if r_mag < 1e-9 or v_mag < 1e-9:
            return oe

        # Specific angular momentum
        h_vec = np.cross(pos, vel)
        h_mag = _norm(h_vec)

        # Eccentricity vector (Laplace-Runge-Lenz)
        e_vec = np.cross(vel, h_vec) / gm - pos / r_mag
        e_mag = float(_norm(e_vec))
        oe.eccentricity = e_mag

        # Specific orbital energy
        eps = 0.5 * v_mag**2 - gm / r_mag

        # Semi-major axis
        oe.semi_major_axis = float(-gm / (2.0 * eps)) if abs(eps) > 1e-12 else float('inf')

        # Orbit type
        if eps < -1e-4:
            oe.orbit_type = "CIRCULAR" if e_mag < 0.05 else "ELLIPTICAL"
        else:
            oe.orbit_type = "ESCAPE"

        # Periapsis / apoapsis / period
        if oe.orbit_type != "ESCAPE":
            a = oe.semi_major_axis
            oe.periapsis_r = a * (1.0 - e_mag)
            oe.apoapsis_r  = a * (1.0 + e_mag)
            oe.period      = 2.0 * math.pi * math.sqrt(max(a, 0.0)**3 / gm)
        else:
            oe.periapsis_r = r_mag

        # Inclination (Y is "up" axis)
        if h_mag > 1e-12:
            cos_i = max(-1.0, min(1.0, float(h_vec[1] / h_mag)))
            oe.inclination_deg = math.degrees(math.acos(cos_i))

        # Ascending node vector N = ĵ × h
        N_vec = np.cross(np.array([0., 1., 0.]), h_vec)
        N_mag = _norm(N_vec)

        # RAAN
        if N_mag > 1e-12:
            cos_raan = max(-1.0, min(1.0, float(N_vec[0] / N_mag)))
            oe.raan_deg = math.degrees(math.acos(cos_raan))
            if N_vec[2] < 0:
                oe.raan_deg = 360.0 - oe.raan_deg

        # Argument of periapsis
        if N_mag > 1e-12 and e_mag > 1e-6:
            cos_w = max(-1.0, min(1.0, float(np.dot(N_vec, e_vec) / (N_mag * e_mag))))
            oe.arg_periapsis_deg = math.degrees(math.acos(cos_w))
            if e_vec[1] < 0:
                oe.arg_periapsis_deg = 360.0 - oe.arg_periapsis_deg

        # True anomaly
        if e_mag > 1e-6:
            cos_nu = max(-1.0, min(1.0, float(np.dot(e_vec, pos) / (e_mag * r_mag))))
            oe.true_anomaly_deg = math.degrees(math.acos(cos_nu))
            if float(np.dot(pos, vel)) < 0:
                oe.true_anomaly_deg = 360.0 - oe.true_anomaly_deg

        return oe


# ══════════════════════════════════════════════════════════════════════
# GRAVITATIONAL BODY
# ══════════════════════════════════════════════════════════════════════

class GravitationalBody:
    """
    A massive attractor (Earth, Moon).
    Stationary if parent is None; otherwise orbits parent in a circle.
    """

    def __init__(self, name: str, gm: float, radius: float,
                 pos: np.ndarray, parent: Optional["GravitationalBody"] = None,
                 orbit_radius: float = 0.0, orbit_period: float = 0.0,
                 orbit_inclination_deg: float = 0.0):
        self.name      = name
        self.gm        = gm
        self.radius    = radius
        self.pos       = pos.astype(np.float64).copy()
        self._parent   = parent
        self._orbit_r  = orbit_radius
        self._orbit_T  = orbit_period
        self._inc_rad  = math.radians(orbit_inclination_deg)
        self._sim_time = 0.0

    def advance(self, dt: float) -> None:
        """Move this body along its prescribed circular orbit."""
        if self._parent is None or self._orbit_T < 1e-9:
            return
        self._sim_time += dt
        omega = 2.0 * math.pi / self._orbit_T
        phase = omega * self._sim_time
        ci, si = math.cos(self._inc_rad), math.sin(self._inc_rad)
        x = self._orbit_r * math.cos(phase)
        z = self._orbit_r * math.sin(phase)
        self.pos = self._parent.pos + np.array([x, z * si, z * ci], dtype=np.float64)

    def gravity_at(self, pos: np.ndarray) -> np.ndarray:
        """Gravitational acceleration at pos: a = GM(r_body - r_pos) / |...|³"""
        delta = self.pos - pos
        d2    = float(np.dot(delta, delta))
        if d2 < 1e-9:
            return np.zeros(3, dtype=np.float64)
        return delta * (self.gm / (d2 * math.sqrt(d2)))


# ══════════════════════════════════════════════════════════════════════
# THRUSTER SYSTEM
# ══════════════════════════════════════════════════════════════════════

class ThrusterSystem:
    """
    Manages continuous (held-key) and impulse (single-press) thrust.
    Directions are expressed as named strings; velocity-frame vectors
    are computed each step from current pos/vel.
    """

    PROGRADE   = "prograde"
    RETROGRADE = "retrograde"
    RADIAL_OUT = "radial_out"
    RADIAL_IN  = "radial_in"
    NORMAL_P   = "normal_pos"
    NORMAL_N   = "normal_neg"

    def __init__(self):
        self._active: set = set()
        self.firing: bool = False

    def press(self, direction: str) -> None:
        self._active.add(direction)
        self.firing = bool(self._active)

    def release(self, direction: str) -> None:
        self._active.discard(direction)
        self.firing = bool(self._active)

    def release_all(self) -> None:
        self._active.clear()
        self.firing = False

    def compute_dv(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
        """Δv from continuous thrust this timestep."""
        if not self._active:
            return np.zeros(3, dtype=np.float64)
        v_hat = _unit(vel)
        r_hat = _unit(pos)
        h_hat = _unit(np.cross(pos, vel))
        dv = np.zeros(3, dtype=np.float64)
        a  = THRUST_ACCEL
        for d in self._active:
            if   d == self.PROGRADE:    dv += v_hat *  a * dt
            elif d == self.RETROGRADE:  dv += v_hat * -a * dt
            elif d == self.RADIAL_OUT:  dv += r_hat *  a * dt
            elif d == self.RADIAL_IN:   dv += r_hat * -a * dt
            elif d == self.NORMAL_P:    dv += h_hat *  a * dt
            elif d == self.NORMAL_N:    dv += h_hat * -a * dt
        return dv

    @staticmethod
    def impulse(vel: np.ndarray, direction: str, dv_mag: float = IMPULSE_DV) -> np.ndarray:
        """Apply instantaneous Δv; return new velocity."""
        v_hat = _unit(vel)
        if   direction == ThrusterSystem.PROGRADE:   return vel + v_hat *  dv_mag
        elif direction == ThrusterSystem.RETROGRADE: return vel + v_hat * -dv_mag
        return vel


# ══════════════════════════════════════════════════════════════════════
# SATELLITE
# ══════════════════════════════════════════════════════════════════════

class Satellite:
    """
    The controlled spacecraft.  Integrates under N-body gravity + thrust.
    """

    def __init__(self, pos: np.ndarray, vel: np.ndarray):
        self.pos      = pos.astype(np.float64).copy()
        self.vel      = vel.astype(np.float64).copy()
        self.acc      = np.zeros(3, dtype=np.float64)
        self.thruster = ThrusterSystem()

        self.escaped:         bool = False
        self.collided:        bool = False
        self.collision_body:  Optional[str] = None

        self._trail:         List[np.ndarray] = []
        self._trail_counter: int = 0

        self.elements:      OrbitalElements = OrbitalElements()
        self._e0:           Optional[float] = None
        self.energy_drift:  float = 0.0

        # Perturbation engine (each satellite owns one; shareable config)
        self.perturbations: PerturbationEngine = PerturbationEngine()

    # ── Trail ──────────────────────────────────────────────────────
    @property
    def trail(self) -> List[np.ndarray]:
        return self._trail

    def clear_trail(self) -> None:
        self._trail.clear()
        self._trail_counter = 0
        self.escaped  = False
        self.collided = False
        self._e0      = None
        self.perturbations.total_drag_dv = 0.0

    def _record_trail(self) -> None:
        self._trail_counter += 1
        if self._trail_counter % TRAIL_SAMPLE_INTERVAL:
            return
        self._trail.append(self.pos.copy())
        if len(self._trail) > TRAIL_MAX_POINTS:
            self._trail.pop(0)

    # ── Derived quantities ─────────────────────────────────────────
    def speed(self) -> float:
        return float(_norm(self.vel))

    def distance_to(self, body: GravitationalBody) -> float:
        return float(_norm(self.pos - body.pos))

    def distance_to_origin(self) -> float:
        return float(_norm(self.pos))

    def kinetic_energy(self) -> float:
        return 0.5 * float(np.dot(self.vel, self.vel))

    def potential_energy(self, bodies: List[GravitationalBody]) -> float:
        pe = 0.0
        for body in bodies:
            d = self.distance_to(body)
            if d > 1e-9:
                pe -= body.gm / d
        return pe

    def total_energy(self, bodies: List[GravitationalBody]) -> float:
        return self.kinetic_energy() + self.potential_energy(bodies)

    def angular_momentum(self) -> np.ndarray:
        return np.cross(self.pos, self.vel)

    def circular_speed_at(self, r: float, gm: float = GM_EARTH) -> float:
        return math.sqrt(gm / r) if r > 1e-9 else 0.0

    def escape_speed_at(self, r: float, gm: float = GM_EARTH) -> float:
        return math.sqrt(2.0 * gm / r) if r > 1e-9 else 0.0

    # ── N-body gravity ─────────────────────────────────────────────
    def _gravity(self, pos: np.ndarray, bodies: List[GravitationalBody]) -> np.ndarray:
        acc = np.zeros(3, dtype=np.float64)
        for body in bodies:
            acc += body.gravity_at(pos)
        return acc

    # ── Adaptive timestep ──────────────────────────────────────────
    @staticmethod
    def _adaptive_dt(dt: float, pos: np.ndarray, vel: np.ndarray,
                     bodies: List[GravitationalBody]) -> float:
        speed = _norm(vel)
        min_r = min((_norm(pos - b.pos) for b in bodies), default=_norm(pos))
        factor = max(1.0, speed / VEL_THRESHOLD, R_THRESHOLD / max(min_r, 1e-9))
        return dt / factor

    # ── Velocity Verlet step ───────────────────────────────────────
    def step(self, dt: float, bodies: List[GravitationalBody]) -> None:
        """Advance one timestep with adaptive Velocity Verlet + thrust + collision."""
        if self.escaped:
            return

        dt_eff = self._adaptive_dt(dt, self.pos, self.vel, bodies)
        a0     = self.acc.copy()

        # Update position
        self.pos = self.pos + self.vel * dt_eff + 0.5 * a0 * dt_eff**2

        # Collision check + response
        self._resolve_collisions(bodies)

        # New acceleration (gravity + perturbations + thrust)
        a_grav    = self._gravity(self.pos, bodies)
        a_pert    = self.perturbations.total(self.pos, self.vel)
        dv_thrust = self.thruster.compute_dv(self.pos, self.vel, dt_eff)
        a_thrust  = dv_thrust / dt_eff if dt_eff > 1e-12 else np.zeros(3)
        a_new     = a_grav + a_pert + a_thrust
        self.acc  = a_new
        # Rotate sun direction for SRP seasonal variation + accumulate diagnostics
        self.perturbations.accumulate_step(dt_eff)
        self.perturbations.rotate_sun(dt_eff)

        # Update velocity (Verlet)
        self.vel = self.vel + 0.5 * (a0 + a_new) * dt_eff

        # Energy drift monitoring
        e_now = self.total_energy(bodies)
        if self._e0 is None and abs(e_now) > 1e-9:
            self._e0 = e_now
        elif self._e0 is not None and abs(self._e0) > 1e-9:
            self.energy_drift = abs((e_now - self._e0) / self._e0)

        # Orbital elements (relative to Earth at origin)
        earth_pos = next((b.pos for b in bodies if b.name == "Earth"), np.zeros(3))
        earth_gm  = next((b.gm  for b in bodies if b.name == "Earth"), GM_EARTH)
        self.elements = OrbitalElements.from_state(self.pos - earth_pos, self.vel, earth_gm)

        # Trail + escape
        self._record_trail()
        if _norm(self.pos) > ESCAPE_DISTANCE:
            self.escaped = True

    # ── Collision resolution ───────────────────────────────────────
    def _resolve_collisions(self, bodies: List[GravitationalBody]) -> None:
        """Inelastic surface bounce: reflect radial velocity × restitution coefficient."""
        for body in bodies:
            delta = self.pos - body.pos
            d     = _norm(delta)
            clearance = body.radius + MIN_ALTITUDE
            if d < clearance:
                n_hat = _unit(delta) if d > 1e-12 else np.array([0., 1., 0.])
                # Push to surface
                self.pos = body.pos + n_hat * clearance
                # Reflect inward velocity component
                v_n = float(np.dot(self.vel, n_hat))
                if v_n < 0:
                    self.vel -= (1.0 + COLLISION_RESTITUTION) * v_n * n_hat
                self.collided      = True
                self.collision_body = body.name

    # ── Velocity manipulation ──────────────────────────────────────
    def rotate_vel_in_plane(self, deg: float) -> None:
        """Rotate velocity around angular momentum axis (stays in orbital plane)."""
        L    = self.angular_momentum()
        axis = _unit(L) if _norm(L) > 1e-10 else np.array([0., 1., 0.])
        self.vel = _rodrigues(self.vel, axis, math.radians(deg))

    def tilt_orbit(self, deg: float) -> None:
        """Tilt velocity out of orbital plane, preserving speed."""
        spd  = self.speed()
        perp = np.cross(self.pos, self.vel)
        pn   = _norm(perp)
        if pn < 1e-10:
            return
        perp /= pn
        self.vel = self.vel + perp * (spd * math.sin(math.radians(deg)))
        new_spd = _norm(self.vel)
        if new_spd > 1e-10:
            self.vel *= spd / new_spd

    def snap_circular(self, gm: float = GM_EARTH, ref_pos: Optional[np.ndarray] = None) -> None:
        r   = _norm(self.pos - (ref_pos if ref_pos is not None else np.zeros(3)))
        v_c = self.circular_speed_at(r, gm)
        spd = self.speed()
        if spd > 1e-10:
            self.vel *= v_c / spd
        self.clear_trail()

    def snap_escape(self, gm: float = GM_EARTH, ref_pos: Optional[np.ndarray] = None) -> None:
        r   = _norm(self.pos - (ref_pos if ref_pos is not None else np.zeros(3)))
        v_e = self.escape_speed_at(r, gm)
        spd = self.speed()
        if spd > 1e-10:
            self.vel *= v_e / spd
        self.clear_trail()

    def apply_impulse(self, direction: str) -> None:
        self.vel = ThrusterSystem.impulse(self.vel, direction)
        self.clear_trail()

    def hohmann_burn1(self, target_r: float, gm: float = GM_EARTH) -> Tuple[float, float]:
        """
        First Hohmann burn: prograde Δv to enter transfer ellipse.
        Returns (dv1, dv2) where dv2 is the second burn magnitude needed.
        """
        r1  = _norm(self.pos)
        r2  = max(target_r, HOHMANN_MIN_TARGET_R)
        a_t = (r1 + r2) / 2.0
        dv1 = math.sqrt(gm / r1) * (math.sqrt(2 * r2 / (r1 + r2)) - 1.0)
        dv2 = math.sqrt(gm / r2) * (1.0 - math.sqrt(2 * r1 / (r1 + r2)))
        self.vel += _unit(self.vel) * dv1
        self.clear_trail()
        return float(dv1), float(dv2)


# ══════════════════════════════════════════════════════════════════════
# PRESETS
# ══════════════════════════════════════════════════════════════════════

def _circ_vel(pos: np.ndarray, gm: float = GM_EARTH) -> np.ndarray:
    r = _norm(pos)
    perp = np.array([-pos[1], pos[0], 0.0])
    return _unit(perp) * math.sqrt(gm / r)


PRESETS: dict = {
    "circular":         {"pos": np.array([2.0, 0.0, 0.0]),
                         "vel": lambda p: _circ_vel(p)},
    "elliptical":       {"pos": np.array([2.0, 0.0, 0.0]),
                         "vel": lambda p: _circ_vel(p) * 0.70},
    "highly_elliptical":{"pos": np.array([0.8, 0.0, 0.0]),
                         "vel": lambda p: _circ_vel(p) * 1.28},
    "escape":           {"pos": np.array([2.0, 0.0, 0.0]),
                         "vel": lambda p: _circ_vel(p) * 1.415},
    "inclined_45":      {"pos": np.array([2.0, 0.0, 0.0]),
                         "vel": lambda p: np.array([0., math.sqrt(GM_EARTH / _norm(p)) * math.cos(math.radians(45)),
                                                        math.sqrt(GM_EARTH / _norm(p)) * math.sin(math.radians(45))])},
    "polar":            {"pos": np.array([2.0, 0.0, 0.0]),
                         "vel": lambda p: np.array([0., 0., math.sqrt(GM_EARTH / _norm(p))])},
    "retrograde":       {"pos": np.array([2.0, 0.0, 0.0]),
                         "vel": lambda p: -_circ_vel(p)},
}


# ══════════════════════════════════════════════════════════════════════
# PHYSICS ENGINE
# ══════════════════════════════════════════════════════════════════════

class PhysicsEngine:
    """
    Top-level simulation coordinator.
    Owns all gravitational bodies and the satellite.
    """

    def __init__(self):
        self._build_bodies()
        self.reset()

    def _build_bodies(self) -> None:
        self.earth = GravitationalBody(
            name="Earth", gm=GM_EARTH, radius=EARTH_RADIUS,
            pos=np.zeros(3, dtype=np.float64))
        self.moon = GravitationalBody(
            name="Moon", gm=GM_MOON, radius=MOON_RADIUS,
            pos=np.array([MOON_ORBIT_RADIUS, 0., 0.], dtype=np.float64),
            parent=self.earth,
            orbit_radius=MOON_ORBIT_RADIUS,
            orbit_period=MOON_ORBIT_PERIOD,
            orbit_inclination_deg=MOON_INCLINATION_DEG)
        self.bodies: List[GravitationalBody] = [self.earth, self.moon]

    def reset(self) -> None:
        self.load_preset("circular")

    def load_preset(self, name: str) -> None:
        if name not in PRESETS:
            name = "circular"
        p   = PRESETS[name]
        pos = p["pos"].astype(np.float64).copy()
        vel = p["vel"](pos).astype(np.float64)
        self.satellite = Satellite(pos, vel)
        self.satellite.acc = self.satellite._gravity(pos, self.bodies)

    def step(self, dt: float) -> None:
        for body in self.bodies:
            body.advance(dt)
        self.satellite.step(dt, self.bodies)
        # Clear one-shot collision flag after response
        if self.satellite.collided:
            self.satellite.collided = False

    # ── Query API ──────────────────────────────────────────────────
    def kinetic_energy(self)   -> float: return self.satellite.kinetic_energy()
    def potential_energy(self) -> float: return self.satellite.potential_energy(self.bodies)
    def total_energy(self)     -> float: return self.satellite.total_energy(self.bodies)
    def orbital_elements(self) -> OrbitalElements: return self.satellite.elements
    def moon_pos(self)         -> np.ndarray: return self.moon.pos.copy()

    def hohmann_to(self, target_r: float) -> Tuple[float, float]:
        return self.satellite.hohmann_burn1(target_r)