"""
perturbations.py — Orbital Perturbation Forces
════════════════════════════════════════════════════════════════════════

WHAT IS AN ORBITAL PERTURBATION?
─────────────────────────────────
A Keplerian orbit assumes a point-mass Earth and no other forces.
Real satellites deviate from this ideal because of:

  1. J2 Oblateness  — Earth is not a sphere; it bulges at the equator.
                      The equatorial radius is ~21 km larger than polar.
                      This extra mass causes the orbital plane to precess
                      (nod westward) and the argument of periapsis to drift.

  2. J4 Term        — Higher-order zonal harmonic; smaller but non-zero.
                      Makes polar orbits slightly more complex.

  3. Atmospheric Drag — Below ~800 km (real) satellites feel air resistance.
                         Drag removes energy → orbit spirals inward.
                         Modelled via exponential density: ρ(r) = ρ₀·e^(-h/H)

  4. Solar Radiation Pressure (SRP) — Photons carry momentum.
                         Tiny but persistent push away from Sun.
                         Distorts orbit shape (eccentricity vector rotates).

  5. Third-Body: Sun — Sun's gravity perturbs Earth-orbiting satellites.
                         Modelled as a distant point mass; causes long-period
                         oscillations in inclination and eccentricity (Kozai-Lidov).

  6. Thermal Re-radiation (Yarkovsky) — Not implemented; requires thermal model.

PERTURBATION HIERARCHY (typical LEO in sim units)
──────────────────────────────────────────────────
  Central gravity:           1.0       (reference)
  J2 oblateness (scaled):  ~ 1e-3      (dominant perturbation)
  Moon N-body:             ~ 1e-3      (already in physics.py)
  Atmospheric drag (LEO):  ~ 1e-4      (orbit-altitude dependent)
  J4 term:                 ~ 1e-6      (small correction to J2)
  Solar radiation:         ~ 1e-5      (area/mass dependent)
  Solar gravity:           ~ 1e-4      (long-period, 3rd-body)

ARCHITECTURE
────────────
  PerturbationConfig   — dataclass: toggle each effect on/off, store params
  PerturbationEngine   — computes total perturbation acceleration vector
  Each perturbation is a pure function: f(pos, vel, t, config) → ndarray[3]

  The Satellite.step() in physics.py calls:
      a_total = a_gravity + a_perturbations + a_thrust
  so perturbations slot cleanly into the existing Verlet integrator.

SIMULATION SCALING NOTES
─────────────────────────
  All parameters are SCALED from real SI values to match the sim unit system:
    Length: 1 sim unit ≈ 6371 km (1 Earth radius)
    GM_earth = 2.5 sim units³/s²
    J2 is boosted 500× so precession is visible within seconds of sim time.
    Drag rho0 is boosted to make spiral decay visible at r~0.4-1.0 sim units.
    SRP is boosted so long-term eccentricity drift is observable.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from constants import GM_EARTH, EARTH_RADIUS


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION  — all parameters in one place, each individually toggled
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PerturbationConfig:
    """
    Master configuration for all perturbation forces.
    Toggle each effect independently; tune magnitudes without touching code.

    Parameters
    ----------
    enable_j2 : bool
        Earth oblateness J2 zonal harmonic.
    enable_j4 : bool
        J4 zonal harmonic (higher-order oblateness correction).
    enable_drag : bool
        Atmospheric drag using exponential density model.
    enable_srp : bool
        Solar radiation pressure.
    enable_solar_gravity : bool
        Sun as a distant third-body perturber.

    j2 : float
        J2 coefficient. Real value = 1.08263e-3; scaled ×500 here so nodal
        precession is visible within a few orbital periods in real time.
    j4 : float
        J4 coefficient (negative). Real value ≈ -1.6e-6; scaled ×500.
    earth_radius : float
        Earth equatorial radius in sim units (used in zonal harmonics).

    drag_rho0 : float
        Sea-level atmospheric density in sim units (scaled from 1.225 kg/m³).
        Higher values → stronger drag → faster orbital decay.
    drag_scale_height : float
        Exponential atmosphere scale height H in sim units.
        ρ(r) = ρ₀ · exp(−(r − R_earth) / H)
    drag_Cd_Am : float
        Ballistic coefficient Cd·A/m [sim units²/kg_sim].
        Larger → more drag (bigger satellite cross-section or lower mass).

    srp_P : float
        Solar radiation pressure at sim-Earth distance [sim force/area].
        Scaled from real 4.56×10⁻⁶ N/m² at 1 AU.
    srp_Cr_Am : float
        SRP coefficient Cr·A/m [sim units²/kg_sim].
        Cr≈1.5 for mixed absorber/reflector; scale with effective area.
    sun_direction : np.ndarray
        Unit vector from Earth to Sun (fixed for simplicity).
        In a full model this rotates once per year.

    sun_gm : float
        Sun's gravitational parameter in sim units.
        Scaled so tidal acceleration ≈ 1% of central gravity at r=2.
    sun_distance : float
        Earth-Sun distance in sim units (≈ 215 Earth radii → sim ≈ 53.75).
    """

    # ── Toggles ───────────────────────────────────────────────────────
    enable_j2:             bool  = True
    enable_j4:             bool  = True
    enable_drag:           bool  = True
    enable_srp:            bool  = True
    enable_solar_gravity:  bool  = True

    # ── J2 / J4 parameters ────────────────────────────────────────────
    j2:          float = 1.08263e-3 * 500.0   # ×500 for visual precession
    j4:          float = -1.6e-6    * 500.0   # ×500 matching J2 scale
    earth_radius: float = EARTH_RADIUS         # 0.25 sim units

    # ── Atmospheric drag ──────────────────────────────────────────────
    drag_rho0:          float = 8.0e-3    # scaled sea-level density
    drag_scale_height:  float = 0.50      # sim units (~3185 km real)
    drag_Cd_Am:         float = 0.04      # ballistic coefficient

    # ── Solar radiation pressure ──────────────────────────────────────
    srp_P:      float = 3.0e-5            # radiation pressure (sim units)
    srp_Cr_Am:  float = 0.010             # reflectivity × area/mass
    sun_direction: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    # ── Third-body: Sun ───────────────────────────────────────────────
    sun_gm:       float = 800.0           # Sun GM in sim units
    sun_distance: float = 53.75           # Earth-Sun distance (sim units)

    def __post_init__(self):
        # Ensure sun_direction is a unit float64 array
        sd = np.asarray(self.sun_direction, dtype=np.float64)
        n = np.linalg.norm(sd)
        self.sun_direction = sd / n if n > 1e-12 else np.array([1., 0., 0.])


# ══════════════════════════════════════════════════════════════════════
# INDIVIDUAL PERTURBATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def _acc_j2(pos: np.ndarray, cfg: PerturbationConfig) -> np.ndarray:
    """
    J2 Oblateness Perturbation
    ──────────────────────────
    Earth's equatorial bulge is captured by the second zonal harmonic J2.
    In the Earth-Centred Inertial (ECI) frame with Y as the polar axis:

        U_J2 = -GM·J2·R²/(2r³) · (3·sin²φ − 1)
             = -GM·J2·R²/(2r⁵) · (3y² − r²)

    Taking the gradient gives the acceleration:

        a_x = -3·GM·J2·R²/(2r⁵) · x · (1 − 5y²/r²)
        a_y = -3·GM·J2·R²/(2r⁵) · y · (3 − 5y²/r²)
        a_z = -3·GM·J2·R²/(2r⁵) · z · (1 − 5y²/r²)

    (Y-axis is polar axis in this coordinate system.)

    Physical effect
    ───────────────
    • Nodal regression: the orbital plane's ascending node drifts westward.
      Rate: dΩ/dt = -3/2 · n · J2 · (Re/p)² · cos(i)
    • Apsidal precession: argument of periapsis rotates.
      Rate: dω/dt = 3/4 · n · J2 · (Re/p)² · (4 − 5·sin²i)
    • These are the dominant long-term perturbations for LEO satellites.
    """
    x, y, z = pos
    r2 = float(np.dot(pos, pos))
    if r2 < 1e-9:
        return np.zeros(3)
    r  = math.sqrt(r2)
    r5 = r2 * r2 * r   # r⁵

    Re = cfg.earth_radius
    coeff = -3.0 * GM_EARTH * cfg.j2 * Re * Re / (2.0 * r5)
    y2_r2 = y * y / r2

    return np.array([
        coeff * x * (1.0 - 5.0 * y2_r2),
        coeff * y * (3.0 - 5.0 * y2_r2),
        coeff * z * (1.0 - 5.0 * y2_r2),
    ], dtype=np.float64)


def _acc_j4(pos: np.ndarray, cfg: PerturbationConfig) -> np.ndarray:
    """
    J4 Oblateness Perturbation
    ──────────────────────────
    The fourth zonal harmonic adds a correction to J2, important for
    precise orbit propagation (GPS, geodetic satellites).

    Gradient of the J4 disturbing potential:

        a_x = 5·GM·J4·R⁴/(8r⁷) · x · (3 − 42y²/r² + 63y⁴/r⁴)
        a_y = 5·GM·J4·R⁴/(8r⁷) · y · (15 − 70y²/r² + 63y⁴/r⁴) − GM·J4·R⁴·15y/(r⁷)
        a_z = 5·GM·J4·R⁴/(8r⁷) · z · (3 − 42y²/r² + 63y⁴/r⁴)

    Physical effect
    ───────────────
    • Refines the J2 precession rate (corrections of order (Re/r)²).
    • Non-negligible for high-accuracy propagation over many orbits.
    """
    x, y, z = pos
    r2 = float(np.dot(pos, pos))
    if r2 < 1e-9:
        return np.zeros(3)
    r  = math.sqrt(r2)
    r7 = r2 * r2 * r2 * r   # r⁷

    Re = cfg.earth_radius
    Re4 = Re**4
    y2_r2 = y * y / r2
    y4_r4 = y2_r2 * y2_r2

    coeff = 5.0 * GM_EARTH * cfg.j4 * Re4 / (8.0 * r7)

    # Off-diagonal terms (full expansion)
    poly_xz = 3.0 - 42.0 * y2_r2 + 63.0 * y4_r4
    poly_y  = (15.0 - 70.0 * y2_r2 + 63.0 * y4_r4)

    return np.array([
        coeff * x * poly_xz,
        coeff * y * poly_y - 15.0 * GM_EARTH * cfg.j4 * Re4 * y / r7,
        coeff * z * poly_xz,
    ], dtype=np.float64)


def _acc_drag(pos: np.ndarray, vel: np.ndarray,
              cfg: PerturbationConfig) -> np.ndarray:
    """
    Atmospheric Drag
    ────────────────
    Drag decelerates the satellite in the direction opposite to velocity:

        a_drag = -½ · Cd·(A/m) · ρ(h) · v² · v̂

    where ρ(h) = ρ₀ · exp(−h/H) is the exponential atmosphere model,
    h = r − Re is the altitude above the surface, and H is the scale height.

    Physical effect
    ───────────────
    • Removes orbital energy → satellite spirals inward (circularises ellipse).
    • Dominant below ~800 km real altitude; negligible at GEO.
    • Causes orbit lifetime decay; used for satellite deorbit planning.

    Sim effect
    ──────────
    With rho0=8e-3, scale_height=0.5, you'll see:
    • Almost no drag at r=2.0 (standard orbit)
    • Significant spiral-in at r<0.7 (low orbit preset)
    • Orbit fully decays in ~seconds at r~0.35 (very low orbit)
    """
    r = math.sqrt(float(np.dot(pos, pos)))
    if r < 1e-9:
        return np.zeros(3)

    altitude = max(0.0, r - cfg.earth_radius)   # height above surface

    # Exponential atmosphere
    rho = cfg.drag_rho0 * math.exp(-altitude / cfg.drag_scale_height)

    v_mag = math.sqrt(float(np.dot(vel, vel)))
    if v_mag < 1e-12:
        return np.zeros(3)

    # F_drag = -½ Cd A/m ρ v² v̂  →  a = -½ Cd A/m ρ |v| v
    return -0.5 * cfg.drag_Cd_Am * rho * v_mag * vel


def _acc_srp(pos: np.ndarray, cfg: PerturbationConfig) -> np.ndarray:
    """
    Solar Radiation Pressure
    ────────────────────────
    Photons carry momentum p = E/c. A satellite absorbs or reflects photons,
    receiving a net force away from the Sun:

        a_srp = Cr · (A/m) · P_srp · ŝ

    where ŝ is the unit vector FROM the Sun TO the satellite,
    P_srp is the radiation pressure at this distance,
    Cr accounts for reflectivity (1 = absorb, 2 = perfect reflect).

    Physical effect
    ───────────────
    • Long-term eccentricity oscillation (eccentricity vector rotates).
    • Sun-synchronous orbit maintenance requires compensating ΔV.
    • Important for high area-to-mass objects (solar sails, debris).

    Shadow model (simplified)
    ─────────────────────────
    • Check if satellite is in Earth's shadow using conical umbra test:
      shadow when (pos · sun_dir) < 0 and the lateral distance < Re.
    • Full penumbra model not implemented (add if needed).
    """
    # Shadow check: is satellite behind Earth relative to Sun?
    sun_dir = cfg.sun_direction
    proj    = float(np.dot(pos, sun_dir))   # positive = sunward side
    if proj < 0.0:
        # Check lateral distance from Sun-Earth line
        shadow_perp = pos - proj * sun_dir
        lateral     = math.sqrt(float(np.dot(shadow_perp, shadow_perp)))
        if lateral < cfg.earth_radius:
            return np.zeros(3)   # in Earth's umbra → no SRP

    # Direction from Sun to satellite (away from Sun)
    return cfg.srp_Cr_Am * cfg.srp_P * sun_dir


def _acc_solar_gravity(pos: np.ndarray, cfg: PerturbationConfig) -> np.ndarray:
    """
    Third-Body: Solar Gravity (Tidal Component)
    ────────────────────────────────────────────
    The Sun pulls both Earth and the satellite. In an Earth-centred frame,
    the satellite feels the differential (tidal) acceleration:

        a_tidal = GM_sun · ( (r_sun_sat / |r_sun_sat|³)
                            − (r_sun_earth / |r_sun_earth|³) )

    For a distant Sun (r_sun >> r_sat), this simplifies to:

        a_tidal ≈ GM_sun / D³ · (pos − 3 · (pos·ŝ) · ŝ)

    where D = Earth-Sun distance, ŝ = Sun direction from Earth.

    Physical effect
    ───────────────
    • Causes Kozai-Lidov oscillations: coupled periodic evolution of
      inclination and eccentricity over long timescales.
    • Important for highly elliptical orbits and for lunar/solar perturbations
      of GPS/GNSS constellation.
    • Sun also shifts periapsis direction over the anomalistic year.
    """
    sun_dir = cfg.sun_direction
    D       = cfg.sun_distance
    D3      = D * D * D

    # Tidal approximation (valid when |pos| << D)
    # a = GM_sun/D³ · [ pos - 3·(pos·ŝ)·ŝ ]
    proj = float(np.dot(pos, sun_dir))
    return (cfg.sun_gm / D3) * (pos - 3.0 * proj * sun_dir)


# ══════════════════════════════════════════════════════════════════════
# PERTURBATION ENGINE
# ══════════════════════════════════════════════════════════════════════

class PerturbationEngine:
    """
    Computes the total perturbation acceleration vector from all enabled forces.

    Usage in Satellite.step():
        a_pert = engine.total(pos, vel, t)
        a_total = a_gravity + a_pert + a_thrust

    The engine is stateless between calls — all state lives in PerturbationConfig.
    This makes it trivial to:
        • Toggle effects on/off at runtime (just flip cfg.enable_*)
        • Adjust parameters live (e.g., inflate drag for low-orbit demo)
        • Unit-test each perturbation independently
    """

    def __init__(self, config: Optional[PerturbationConfig] = None):
        self.cfg = config if config is not None else PerturbationConfig()

        # Diagnostic: last computed contribution of each force (for HUD)
        self.last_j2:      float = 0.0
        self.last_j4:      float = 0.0
        self.last_drag:    float = 0.0
        self.last_srp:     float = 0.0
        self.last_solar:   float = 0.0
        self.last_total:   float = 0.0

        # Accumulated impulse from each perturbation (for long-term analysis)
        self._dt_accum:    float = 0.0
        self.total_drag_dv: float = 0.0   # cumulative drag ΔV (energy removed)

    def total(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """
        Return the total perturbation acceleration at position *pos*
        with velocity *vel*.

        Each enabled term is computed and summed. Disabled terms contribute
        exactly zero (no floating-point noise from disabled effects).
        """
        acc = np.zeros(3, dtype=np.float64)

        if self.cfg.enable_j2:
            a_j2 = _acc_j2(pos, self.cfg)
            self.last_j2 = float(np.linalg.norm(a_j2))
            acc += a_j2
        else:
            self.last_j2 = 0.0

        if self.cfg.enable_j4:
            a_j4 = _acc_j4(pos, self.cfg)
            self.last_j4 = float(np.linalg.norm(a_j4))
            acc += a_j4
        else:
            self.last_j4 = 0.0

        if self.cfg.enable_drag:
            a_drag = _acc_drag(pos, vel, self.cfg)
            self.last_drag = float(np.linalg.norm(a_drag))
            acc += a_drag
        else:
            a_drag = np.zeros(3, dtype=np.float64)
            self.last_drag = 0.0

        if self.cfg.enable_srp:
            a_srp = _acc_srp(pos, self.cfg)
            self.last_srp = float(np.linalg.norm(a_srp))
            acc += a_srp
        else:
            self.last_srp = 0.0

        if self.cfg.enable_solar_gravity:
            a_solar = _acc_solar_gravity(pos, self.cfg)
            self.last_solar = float(np.linalg.norm(a_solar))
            acc += a_solar
        else:
            self.last_solar = 0.0

        self.last_total = float(np.linalg.norm(acc))
        # Auto-accumulate drag ΔV for diagnostics (called with a nominal dt=1)
        # Actual per-step accumulation is done in rotate_sun call from Satellite.step
        self._last_drag_acc = a_drag if self.cfg.enable_drag else np.zeros(3)
        return acc

    def accumulate_drag(self, drag_acc: np.ndarray, dt: float) -> None:
        """Track total Δv lost to drag (diagnostic)."""
        self.total_drag_dv += float(np.linalg.norm(drag_acc)) * dt

    def accumulate_step(self, dt: float) -> None:
        """Call after total() each step to accumulate diagnostic integrals."""
        if hasattr(self, '_last_drag_acc'):
            self.total_drag_dv += float(np.linalg.norm(self._last_drag_acc)) * dt

    def rotate_sun(self, dt: float, period: float = 120.0) -> None:
        """
        Slowly rotate the Sun direction vector to simulate Earth's annual orbit.
        *period* is the full rotation period in sim seconds.
        A full rotation ≈ 120 sim seconds → visible SRP direction change.
        """
        angle = 2.0 * math.pi * dt / period
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        sd = self.cfg.sun_direction
        # Rotate in XZ plane (Y is polar axis)
        new_x = cos_a * sd[0] - sin_a * sd[2]
        new_z = sin_a * sd[0] + cos_a * sd[2]
        self.cfg.sun_direction = np.array([new_x, sd[1], new_z], dtype=np.float64)

    def summary(self) -> dict:
        """Return diagnostic dict of last-computed magnitudes."""
        return {
            "j2":    self.last_j2,
            "j4":    self.last_j4,
            "drag":  self.last_drag,
            "srp":   self.last_srp,
            "solar": self.last_solar,
            "total": self.last_total,
            "drag_dv_total": self.total_drag_dv,
        }
