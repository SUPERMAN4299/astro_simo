# 🛰️ Orbital Mechanics Engine v3

**A physically accurate, real-time 3D orbital mechanics simulator built in Python.**

> Simulates Keplerian orbits, N-body gravity, J2/J4 oblateness, atmospheric drag, solar radiation pressure, third-body perturbations, Hohmann transfers, inclined orbits, and satellite collision response — all rendered in OpenGL with a live telemetry HUD.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Installation & Setup](#2-installation--setup)
3. [File Structure](#3-file-structure)
4. [Controls Reference](#4-controls-reference)
5. [The Physics — Complete Theory](#5-the-physics--complete-theory)
   - 5.1 [Newton's Law of Gravitation](#51-newtons-law-of-gravitation)
   - 5.2 [The Two-Body Problem](#52-the-two-body-problem)
   - 5.3 [Kepler's Laws](#53-keplers-laws)
   - 5.4 [Orbital Energy & Vis-Viva Equation](#54-orbital-energy--vis-viva-equation)
   - 5.5 [Orbital Elements — The 6 Classical Parameters](#55-orbital-elements--the-6-classical-parameters)
   - 5.6 [Deriving Elements from State Vectors](#56-deriving-elements-from-state-vectors)
   - 5.7 [N-Body Gravity](#57-n-body-gravity)
   - 5.8 [Angular Momentum & Kepler's Second Law](#58-angular-momentum--keplers-second-law)
   - 5.9 [Escape Velocity & Orbit Classification](#59-escape-velocity--orbit-classification)
   - 5.10 [Hohmann Transfer Orbit](#510-hohmann-transfer-orbit)
6. [Perturbation Theory — Complete Reference](#6-perturbation-theory--complete-reference)
   - 6.1 [What Is a Perturbation?](#61-what-is-a-perturbation)
   - 6.2 [J2 Oblateness — Earth's Equatorial Bulge](#62-j2-oblateness--earths-equatorial-bulge)
   - 6.3 [J4 Zonal Harmonic](#63-j4-zonal-harmonic)
   - 6.4 [Atmospheric Drag](#64-atmospheric-drag)
   - 6.5 [Solar Radiation Pressure (SRP)](#65-solar-radiation-pressure-srp)
   - 6.6 [Third-Body Solar Gravity (Tidal)](#66-third-body-solar-gravity-tidal)
   - 6.7 [The Moon as an N-Body Perturber](#67-the-moon-as-an-n-body-perturber)
   - 6.8 [Perturbation Hierarchy](#68-perturbation-hierarchy)
7. [Numerical Integration](#7-numerical-integration)
   - 7.1 [Why Integration Matters](#71-why-integration-matters)
   - 7.2 [Euler Integration (What We Don't Use)](#72-euler-integration-what-we-dont-use)
   - 7.3 [Velocity Verlet (What We Use)](#73-velocity-verlet-what-we-use)
   - 7.4 [Symplectic Integrators & Energy Conservation](#74-symplectic-integrators--energy-conservation)
   - 7.5 [Adaptive Timestep](#75-adaptive-timestep)
8. [3D Mathematics](#8-3d-mathematics)
   - 8.1 [Coordinate System](#81-coordinate-system)
   - 8.2 [Rodrigues' Rotation Formula](#82-rodrigues-rotation-formula)
   - 8.3 [Camera Mathematics — LookAt & Perspective](#83-camera-mathematics--lookat--perspective)
   - 8.4 [Spherical Camera Coordinates](#84-spherical-camera-coordinates)
9. [Rendering Pipeline](#9-rendering-pipeline)
   - 9.1 [OpenGL & ModernGL](#91-opengl--moderngl)
   - 9.2 [Blinn-Phong Lighting](#92-blinn-phong-lighting)
   - 9.3 [Earth Texture Mapping](#93-earth-texture-mapping)
   - 9.4 [Satellite Model Geometry](#94-satellite-model-geometry)
10. [Simulation Scaling](#10-simulation-scaling)
11. [Code Architecture](#11-code-architecture)
12. [Extending the Simulator](#12-extending-the-simulator)
13. [References & Further Reading](#13-references--further-reading)

---

## 1. Project Overview

This simulator models the complete lifecycle of a spacecraft in Earth orbit — from launch through orbital maneuvers to atmospheric reentry. It is built on real astrodynamics equations used by space agencies, scaled and accelerated for real-time interactive visualization.

**What makes it physically accurate:**
- Gravity follows Newton's inverse-square law in full 3D
- Orbital elements (semi-major axis, eccentricity, inclination, RAAN, argument of periapsis, true anomaly) are computed every frame from state vectors using standard astrodynamics
- The Velocity Verlet integrator preserves orbital energy over thousands of steps with < 0.1% drift on unperturbed orbits
- Five independent perturbation forces act simultaneously, each derived from real physical models
- The Moon orbits Earth with correct inclination and attracts the satellite using full N-body gravity
- Collision with Earth's surface triggers an inelastic bounce response

**Stack:** Python 3.10+ · NumPy · ModernGL · GLFW · Pygame (HUD) · Pillow (textures)

---

## 2. Installation & Setup

### Prerequisites

```bash
pip install moderngl glfw numpy pygame Pillow
```

### Directory Structure Required

```
your_folder/
├── main.py
├── physics.py
├── constants.py
├── perturbations.py
├── camera.py
├── renderer.py
└── assets/
    ├── earth.jpg       # Earth diffuse texture (2048×1024)
    ├── clouds.png      # Cloud layer (RGBA)
    └── specular.jpg    # Ocean/land specular map
```

> The texture files are generated automatically the first time you run the simulator if they are missing. Alternatively, place any equirectangular Earth texture at `assets/earth.jpg`.

### Running

```bash
python main.py
```

### Tested On

| Platform | Python | Status |
|----------|--------|--------|
| Windows 10/11 | 3.10 – 3.13 | ✅ |
| Ubuntu 22.04 | 3.10 – 3.12 | ✅ |
| macOS 13+ (Apple Silicon) | 3.11 – 3.12 | ✅ (requires `OPENGL_FORWARD_COMPAT=True`) |

---

## 3. File Structure

| File | Role | Key Classes |
|------|------|-------------|
| `constants.py` | All simulation parameters in one place | — |
| `physics.py` | Orbital mechanics engine | `OrbitalElements`, `GravitationalBody`, `Satellite`, `ThrusterSystem`, `PhysicsEngine` |
| `perturbations.py` | Five perturbation force models | `PerturbationConfig`, `PerturbationEngine` |
| `camera.py` | Spherical orbit camera | `Camera` |
| `renderer.py` | ModernGL rendering pipeline | `Renderer` |
| `main.py` | GLFW window, event loop, input | — |

**Data flow:**

```
main.py
  │
  ├─ PhysicsEngine.step(dt)
  │     ├─ GravitationalBody.advance(dt)      ← Moon orbit
  │     └─ Satellite.step(dt, bodies)
  │           ├─ _gravity(pos, bodies)        ← N-body
  │           ├─ PerturbationEngine.total()   ← J2+J4+drag+SRP+solar
  │           ├─ ThrusterSystem.compute_dv()  ← thrust
  │           └─ Velocity Verlet integration
  │
  └─ Renderer.render(physics, camera, view, proj)
        ├─ Earth sphere (textured, Blinn-Phong)
        ├─ Moon sphere
        ├─ Orbital trail (line strip, color by energy)
        ├─ Satellite mesh (ISS geometry)
        ├─ Orbital plane ring
        └─ HUD overlay (orbital elements + perturbation panel)
```

---

## 4. Controls Reference

### Camera

| Input | Action |
|-------|--------|
| Left mouse drag | Orbit camera around Earth |
| Scroll wheel | Zoom in / out |
| `W` / `S` | Pan view up / down |
| `A` / `D` | Pan view left / right |

### Satellite Velocity

| Key | Action |
|-----|--------|
| `↑` (held) | Increase speed (scale velocity magnitude) |
| `↓` (held) | Decrease speed |
| `←` (held) | Rotate velocity clockwise in orbital plane |
| `→` (held) | Rotate velocity counter-clockwise in orbital plane |
| `I` (held) | Tilt orbit inclination positive (+Z) |
| `K` (held) | Tilt orbit inclination negative (−Z) |

### Thruster

| Key | Action |
|-----|--------|
| `Z` (held) | Continuous **prograde** thrust (accelerates along velocity direction) |
| `X` (held) | Continuous **retrograde** thrust (decelerates) |
| `Q` | Single **impulse** prograde burst (instantaneous Δv) |
| `C` | Snap to exact **circular orbit** speed at current radius |
| `E` | Snap to exact **escape velocity** at current radius |
| `H` | Execute **Hohmann transfer** first burn (target r = 4.0 sim units) |

### Presets

| Key | Orbit |
|-----|-------|
| `1` | Circular orbit (r = 2.0) |
| `2` | Elliptical (e ≈ 0.50) |
| `3` | Highly elliptical / Molniya-like (e > 0.7) |
| `4` | Escape trajectory |
| `5` | Inclined 45° |
| `6` | Polar orbit (i = 90°) |
| `7` | Retrograde |

### Simulation

| Key | Action |
|-----|--------|
| `Space` | Pause / Resume |
| `R` | Full reset |
| `F` | Speed up time (×1.5 per press, max ×10) |
| `S` | Slow down time (÷1.5 per press, min ×0.1) |
| `Esc` | Quit |

### Perturbation Toggles

| Key | Toggles |
|-----|---------|
| `P` | ALL perturbations on / off |
| `J` | J2 + J4 oblateness |
| `N` | Solar radiation pressure |
| `G` | Solar gravity (third-body) |
| `D` (while paused) | Atmospheric drag |

---

## 5. The Physics — Complete Theory

### 5.1 Newton's Law of Gravitation

Every pair of masses attracts each other with a force proportional to their masses and inversely proportional to the square of their separation:

```
F = G · M · m / r²
```

Where:
- `G` = gravitational constant = 6.674 × 10⁻¹¹ N·m²/kg²
- `M` = mass of the central body (Earth)
- `m` = mass of the satellite
- `r` = distance between their centers of mass

In vector form (direction matters):

```
F⃗ = -G · M · m / r³ · r⃗
```

The negative sign means the force points **toward** the central body. Dividing by `m` gives the gravitational **acceleration** experienced by the satellite:

```
a⃗ = -G·M / r³ · r⃗  =  -GM / |r|³ · r⃗
```

Note the use of `|r|³` in the denominator: it simultaneously provides the inverse-square magnitude (`GM/r²`) and the unit direction vector (`r̂ = r/|r|`). This is a computationally efficient form that avoids a separate normalization step.

In the simulation, `GM` is a single combined constant (the **gravitational parameter**, sometimes written `μ`). We never need `G` and `M` separately.

### 5.2 The Two-Body Problem

When you have only two bodies (Earth + satellite), the equation of motion is:

```
r̈ = -GM / r³ · r⃗
```

This is a second-order ordinary differential equation. It has an exact analytical solution — the solution is a **conic section** (circle, ellipse, parabola, or hyperbola) depending on the total energy. This is what Kepler described empirically in 1609–1619 before Newton provided the mathematical proof.

The two-body problem is **exactly solvable**. The moment you add a third body (Moon, Sun), it becomes the **three-body problem**, which has no general closed-form solution. The gravitational interactions become chaotic under certain configurations. This is why we use numerical integration.

### 5.3 Kepler's Laws

Johannes Kepler derived three empirical laws from Tycho Brahe's observations of Mars (1609–1619):

**Kepler's First Law** — *The Law of Ellipses*

Every planet (or satellite) moves in an ellipse with the central body at one **focus**. A circle is a special case of an ellipse where both foci coincide.

```
Ellipse geometry:
   a = semi-major axis (half the long axis)
   b = semi-minor axis (half the short axis)
   e = eccentricity = √(1 - b²/a²)
   c = ae = distance from center to focus

   At periapsis (closest point):  r_p = a(1 - e)
   At apoapsis (farthest point):  r_a = a(1 + e)
```

**Kepler's Second Law** — *The Law of Equal Areas*

A line segment joining the satellite to the central body sweeps equal areas in equal intervals of time. This means:
- The satellite moves **faster** near periapsis
- The satellite moves **slower** near apoapsis

This is not a separate physical law — it is a direct consequence of **conservation of angular momentum** (`L = r × mv = constant`). Since `|L| = r · v · sin(θ)`, when `r` decreases, `v` must increase to keep `|L|` constant.

```
Area swept in time dt:
   dA = ½ · |r × v| · dt = ½ · |L|/m · dt = constant
```

In the simulation, this law **emerges automatically** from the physics — no code is needed to enforce it. The symplectic integrator conserves angular momentum naturally.

**Kepler's Third Law** — *The Law of Periods*

The square of the orbital period `T` is proportional to the cube of the semi-major axis `a`:

```
T² = 4π²/GM · a³

Or equivalently:   T = 2π · √(a³/GM)
```

Consequence: larger orbits take longer to complete, and the relationship is the same for every orbit around the same central body. The International Space Station (a ≈ 6771 km) has a period of ~92 minutes; the Moon (a ≈ 384,400 km) has a period of ~27.3 days.

### 5.4 Orbital Energy & Vis-Viva Equation

The **specific orbital energy** (energy per unit mass) of a satellite is:

```
ε = KE + PE = ½v² - GM/r
```

Where `½v²` is the specific kinetic energy and `-GM/r` is the specific gravitational potential energy. The sign conventions matter enormously:
- PE is **negative** (work must be done to escape the gravitational well)
- KE is always **positive**
- Total energy `ε` determines the orbit type

The **vis-viva equation** relates speed to position on any conic orbit:

```
v² = GM · (2/r - 1/a)
```

This equation is the single most important formula in astrodynamics. From it:
- **Circular orbit** (`r = a`): `v_c = √(GM/r)`
- **Escape** (`a → ∞`, `ε = 0`): `v_esc = √(2GM/r)`
- Note that `v_esc = √2 · v_c` — escape velocity is always exactly √2 times circular velocity at the same radius

**Orbit type by total energy:**

```
ε < 0   →   Elliptical (bound orbit)   — satellite cannot escape
ε = 0   →   Parabolic  (barely escapes)
ε > 0   →   Hyperbolic (escape trajectory)
```

Semi-major axis from energy (via vis-viva):

```
a = -GM / (2ε)
```

For bound orbits, `ε < 0`, so `a > 0`. For escape trajectories, `ε > 0`, so `a` would be negative — which is the mathematical convention for hyperbolic orbits.

### 5.5 Orbital Elements — The 6 Classical Parameters

A Keplerian orbit requires **6 numbers** to fully describe it (3 for the shape and orientation of the orbit, 3 for the satellite's position on it). These are the **Classical Orbital Elements (COE)**:

| Symbol | Name | Describes |
|--------|------|-----------|
| `a` | Semi-major axis | Size of the orbit |
| `e` | Eccentricity | Shape (0 = circle, 0<e<1 = ellipse, e≥1 = escape) |
| `i` | Inclination | Tilt of orbital plane relative to equator [degrees] |
| `Ω` | RAAN (Right Ascension of Ascending Node) | Where orbital plane intersects equatorial plane |
| `ω` | Argument of periapsis | Orientation of ellipse within orbital plane |
| `ν` | True anomaly | Satellite's current position on the orbit |

**Inclination `i`:**
- 0° = prograde equatorial (geostationary belt)
- 90° = polar orbit (passes over poles, used for Earth observation)
- 180° = retrograde equatorial
- 63.4° = "critical inclination" where apsidal precession from J2 is zero (used for Molniya orbits)

**RAAN `Ω`:**
Measured eastward from the vernal equinox to the ascending node (where the satellite crosses the equatorial plane going north). Due to J2 oblateness, Ω drifts at a rate that depends on inclination. Sun-synchronous orbits use this drift to keep the orbit aligned with the Sun.

**Argument of periapsis `ω`:**
Measured in the orbital plane from the ascending node to the periapsis direction. J2 causes ω to precess at a rate that depends on inclination. At i = 63.4° (critical inclination), this precession rate is zero — allowing the periapsis to stay above the same latitude. The Molniya orbit used by Russia exploits this.

**True anomaly `ν`:**
The angle between the periapsis direction and the satellite's current position vector. `ν = 0` at periapsis (closest approach), `ν = 180°` at apoapsis (farthest point).

### 5.6 Deriving Elements from State Vectors

Given position `r⃗` and velocity `v⃗`, we can compute all orbital elements analytically. This is what `OrbitalElements.from_state()` does every physics step.

**Step 1 — Specific angular momentum:**
```
h⃗ = r⃗ × v⃗
```
`h⃗` is perpendicular to the orbital plane. Its magnitude is conserved (Kepler 2nd Law).

**Step 2 — Eccentricity vector (Laplace-Runge-Lenz vector):**
```
e⃗ = (v⃗ × h⃗) / GM  −  r̂
```
`e⃗` points from the focus toward periapsis. Its magnitude is the scalar eccentricity `e = |e⃗|`. The LRL vector is a conserved quantity in the pure two-body problem — perturbations cause it to drift slowly.

**Step 3 — Specific orbital energy:**
```
ε = v²/2 − GM/r
```

**Step 4 — Semi-major axis:**
```
a = −GM / (2ε)
```

**Step 5 — Inclination:**
With the Y-axis as the polar axis:
```
cos(i) = h_y / |h|
i = acos(h_y / |h|)
```

**Step 6 — RAAN (Ω):**
First compute the ascending node vector `N⃗ = ĵ × h⃗` (perpendicular to both the polar axis and the angular momentum vector). Then:
```
cos(Ω) = N_x / |N|
Ω = acos(N_x / |N|)     if N_z ≥ 0
Ω = 360° − acos(N_x/|N|)  if N_z < 0
```

**Step 7 — Argument of periapsis (ω):**
```
cos(ω) = (N⃗ · e⃗) / (|N| · |e|)
ω = acos(...)   if e_y ≥ 0
ω = 360° − acos(...)   if e_y < 0
```

**Step 8 — True anomaly (ν):**
```
cos(ν) = (e⃗ · r⃗) / (|e| · r)
ν = acos(...)   if r⃗ · v⃗ ≥ 0  (moving away from periapsis)
ν = 360° − acos(...)   if r⃗ · v⃗ < 0  (moving toward periapsis)
```

The dot product `r⃗ · v⃗ = |r||v|cos(flight path angle)` tells us whether we're on the inbound or outbound leg of the orbit.

### 5.7 N-Body Gravity

With multiple massive bodies (Earth + Moon), the satellite accelerates toward **all of them simultaneously**:

```
a⃗_sat = Σᵢ  GMᵢ · (r⃗ᵢ − r⃗_sat) / |r⃗ᵢ − r⃗_sat|³
```

Each body `i` contributes an independent gravitational pull. The total acceleration is the vector sum. This is the **N-body problem** in action — for N=3 (Earth, Moon, satellite), the satellite's trajectory cannot be predicted analytically.

**Moon perturbation effects:**
- Oscillates inclination and eccentricity with a period related to the Moon's orbital period
- Significant for highly elliptical orbits (apoapsis near Moon's orbit)
- Moon's gravitational sphere of influence extends to ~66,000 km (~10 Earth radii)
- Responsible for the "3-body resonance" effects in libration point missions (L1, L2, L3, L4, L5)

### 5.8 Angular Momentum & Kepler's Second Law

The specific angular momentum vector is:

```
h⃗ = r⃗ × v⃗
```

In the absence of torques (all gravitational forces pass through the center of mass), `dh⃗/dt = 0`. This means:
- `|h⃗|` is constant → Kepler's 2nd law
- The direction of `h⃗` is constant → orbital plane doesn't precess (without perturbations)
- J2 perturbation exerts a small torque → `h⃗` direction slowly drifts → nodal precession

The orbital plane normal is `h⃗ / |h⃗|`. This is displayed in the simulation as the blue axis arrow.

### 5.9 Escape Velocity & Orbit Classification

**Escape velocity** is the minimum speed needed to escape a gravitational body's influence starting from radius `r`:

```
v_esc = √(2GM/r)
```

This is derived by setting total energy to zero:
```
ε = ½v² − GM/r = 0   →   v = √(2GM/r)
```

At Earth's surface (r ≈ 6371 km), this is approximately 11.2 km/s.

**Orbit classification by speed** (at a given radius `r`):

```
v < v_c = √(GM/r)          →  Suborbital (will hit Earth)
v = v_c                    →  Circular orbit
v_c < v < v_esc            →  Elliptical orbit (bound)
v = v_esc = √(2GM/r)       →  Parabolic escape (barely)
v > v_esc                  →  Hyperbolic escape trajectory
```

The circular speed `v_c` and escape speed `v_esc` differ by exactly `√2`:
```
v_esc = √2 · v_c   ≈   1.414 · v_c
```

### 5.10 Hohmann Transfer Orbit

A **Hohmann transfer** is the most fuel-efficient two-impulse maneuver to move between two circular coplanar orbits. It uses an intermediate elliptical transfer orbit that is tangent to both circular orbits.

**Geometry:**
- Transfer ellipse periapsis = `r₁` (lower orbit radius)
- Transfer ellipse apoapsis = `r₂` (higher orbit radius)
- Semi-major axis of transfer ellipse: `a_t = (r₁ + r₂) / 2`

**First burn (at periapsis of transfer ellipse):**

The satellite is in circular orbit at `r₁` with speed:
```
v₁ = √(GM/r₁)
```

Speed at periapsis of transfer ellipse (from vis-viva with a = a_t):
```
v_t1 = √(GM · (2/r₁ − 1/a_t))
```

Required Δv₁ (prograde):
```
Δv₁ = v_t1 − v₁ = √(GM/r₁) · (√(2r₂/(r₁+r₂)) − 1)
```

**Second burn (at apoapsis, half an orbit later):**

Speed at apoapsis of transfer ellipse:
```
v_t2 = √(GM · (2/r₂ − 1/a_t))
```

Target circular speed at `r₂`:
```
v₂ = √(GM/r₂)
```

Required Δv₂ (prograde again):
```
Δv₂ = v₂ − v_t2 = √(GM/r₂) · (1 − √(2r₁/(r₁+r₂)))
```

**Total Δv:**
```
ΔV_total = |Δv₁| + |Δv₂|
```

The Hohmann transfer is optimal only for nearly circular coplanar orbits where `r₂/r₁ < 11.94`. Beyond that ratio, a bi-elliptic transfer becomes more efficient.

In the simulation, pressing `H` executes Δv₁ immediately (prograde impulse at current position). Δv₂ is printed to console for reference — you would need to fire it half an orbit later manually.

---

## 6. Perturbation Theory — Complete Reference

### 6.1 What Is a Perturbation?

In the ideal two-body problem, orbits are perfect, unchanging ellipses. In reality, they evolve continuously due to forces beyond the central point-mass gravity. These additional forces are called **perturbations**.

Perturbation theory separates the total force into:
```
a⃗_total = a⃗_kepler + a⃗_perturbations
```

Where `a⃗_kepler = −GM/r³ · r⃗` is the "main" gravity, and `a⃗_perturbations` is everything else. Perturbations are typically small (`|a_pert| << |a_kepler|`), but because they act continuously, their cumulative effect over many orbits can become large.

**Osculating elements:** At any instant, we can describe the satellite's trajectory as if it were a perfect Keplerian ellipse (the "osculating orbit"). Perturbations cause this osculating ellipse to slowly change — the elements drift. This is how mission controllers monitor orbital decay, precession, and station-keeping requirements.

### 6.2 J2 Oblateness — Earth's Equatorial Bulge

**Physical cause:**

Earth is not a perfect sphere. It is an **oblate spheroid** — slightly flattened at the poles and bulging at the equator. The equatorial radius is 6,378.1 km while the polar radius is 6,356.8 km, a difference of ~21 km. This extra mass at the equator creates an uneven gravitational field that cannot be described by a simple `GM/r²` point-mass formula.

**Gravitational potential expansion:**

The full gravitational potential of an oblate Earth is expressed as a series of **zonal harmonics** (spherical harmonic functions):

```
U(r, φ) = GM/r · [1 − Σₙ (Rₑ/r)ⁿ · Jₙ · Pₙ(sin φ)]
```

Where:
- `φ` = geocentric latitude (not geographic latitude)
- `Rₑ` = Earth's equatorial radius
- `Jₙ` = zonal harmonic coefficients (dimensionless constants determined by Earth's mass distribution)
- `Pₙ` = Legendre polynomials

The dominant term is **J2 = 1.08263 × 10⁻³** (by far the largest deviation from a perfect sphere). Higher terms (J3, J4, ...) exist but are progressively smaller.

**J2 acceleration formula:**

Taking the gradient of the J2 disturbing potential in ECI coordinates (Y = polar axis):

```
aₓ = −(3·GM·J2·Rₑ²) / (2r⁵) · x · (1 − 5y²/r²)
aᵧ = −(3·GM·J2·Rₑ²) / (2r⁵) · y · (3 − 5y²/r²)
a_z = −(3·GM·J2·Rₑ²) / (2r⁵) · z · (1 − 5y²/r²)
```

The `y` component (polar axis direction) has a different coefficient (3 vs 1) because the potential is asymmetric about the equatorial plane.

**Physical effects:**

*Nodal Regression (dΩ/dt):*

The orbital plane slowly rotates around Earth's spin axis:
```
dΩ/dt = −(3/2) · n · J2 · (Rₑ/p)² · cos(i)
```
Where `n = √(GM/a³)` is the mean motion (radians/second) and `p = a(1−e²)` is the semi-latus rectum.

- Prograde orbits (i < 90°): Ω drifts **westward** (negative rate)
- Retrograde orbits (i > 90°): Ω drifts **eastward** (positive rate)
- At i = 90° (polar): no nodal precession
- **Sun-synchronous orbits** use i ≈ 97.4° to make Ω precess eastward at exactly 0.9856°/day, matching Earth's revolution around the Sun — this keeps the orbital plane fixed relative to the Sun, ensuring consistent lighting for Earth-observation satellites.

*Apsidal Precession (dω/dt):*

The argument of periapsis rotates within the orbital plane:
```
dω/dt = (3/4) · n · J2 · (Rₑ/p)² · (4 − 5·sin²i)
```

- This is zero when `sin²i = 4/5`, i.e., i = **63.435°** (the critical inclination)
- **Molniya orbits** use i = 63.4° with high eccentricity (e ≈ 0.74) and a ≈ 26,560 km, so the apoapsis stays over Russia throughout the satellite's lifetime — no apsidal precession means the high-latitude coverage window is stable.

*Effect on inclination:*

Pure J2 produces only secular (steady) changes in Ω and ω. Inclination and semi-major axis are unchanged by J2 in the first-order analysis (they have only periodic oscillations that average to zero).

**In the simulation:**

J2 is scaled ×500 so nodal precession is visible within a few orbital periods. At standard orbit (r = 2.0, circular), you'll see the orbital plane ring slowly rotating.

### 6.3 J4 Zonal Harmonic

**Physical cause:**

J4 (= −1.6 × 10⁻⁶ for Earth) represents a further correction to the gravitational field from the fourth-degree Legendre polynomial. It arises from more subtle mass distribution irregularities. The J3 term (= −2.54 × 10⁻⁶) represents a north-south asymmetry (Earth is very slightly "pear-shaped") but is not included here.

**J4 acceleration formula:**

```
aₓ = (5·GM·J4·Rₑ⁴) / (8r⁷) · x · (3 − 42y²/r² + 63y⁴/r⁴)
aᵧ = (5·GM·J4·Rₑ⁴) / (8r⁷) · y · (15 − 70y²/r² + 63y⁴/r⁴) − 15·GM·J4·Rₑ⁴·y / r⁷
a_z = (5·GM·J4·Rₑ⁴) / (8r⁷) · z · (3 − 42y²/r² + 63y⁴/r⁴)
```

**Physical effects:**

J4 modifies the J2 precession rates by a correction of order `(Rₑ/r)² × J4/J2`. For GPS satellites at r ≈ 4.2 Earth radii, this correction is non-negligible for high-precision orbit determination. For LEO, it is a small but measurable refinement.

**In the simulation:**

J4 magnitude is about 0.003% of J2 at r=2.0. It is included for completeness and physical accuracy, not for visual effect.

### 6.4 Atmospheric Drag

**Physical cause:**

Below approximately 800 km altitude, Earth's atmosphere is dense enough to exert a meaningful drag force on satellites. The force acts opposite to the satellite's velocity and removes orbital energy, causing the orbit to spiral inward.

**Exponential atmosphere model:**

The atmospheric density decreases approximately exponentially with altitude `h`:

```
ρ(h) = ρ₀ · exp(−h / H)
```

Where:
- `ρ₀` = sea-level density (1.225 kg/m³ for real Earth)
- `h = r − Rₑ` = altitude above Earth's surface
- `H` = scale height ≈ 8.5 km at sea level for real Earth (varies with altitude and solar activity)

A more accurate model uses multiple exponential layers (the NRLMSISE-00 model used by NASA has ~100 layers), but the single exponential is adequate for demonstration.

**Drag acceleration:**

```
a⃗_drag = −½ · (Cd · A / m) · ρ(h) · |v| · v⃗
```

Where:
- `Cd` = drag coefficient (≈ 2.2 for typical satellites in free molecular flow at LEO)
- `A` = cross-sectional area facing the velocity vector (m²)
- `m` = satellite mass (kg)
- `Cd·A/m` = the **ballistic coefficient** (combined aerodynamic parameter)
- `ρ` = atmospheric density at current altitude
- `|v|` = speed (the `|v| · v⃗` term gives the `v²` force magnitude with correct direction)

**Physical effects:**

- **Orbit circularization:** Drag acts most strongly at periapsis (lowest altitude, densest atmosphere). This lowers the apoapsis over many orbits while the periapsis changes more slowly → orbit becomes more circular.
- **Orbital decay:** As the orbit circularizes, both periapsis and apoapsis approach Earth → satellite eventually deorbits.
- **Lifetime estimation:** The ISS (altitude ≈ 400 km) experiences ~2 km/day of altitude loss from drag and requires periodic reboost maneuvers.
- **Thermosphere variability:** Solar activity heats the thermosphere, expanding it. During solar maximum (sunspot cycle), drag at 400 km can increase by a factor of 10×, drastically reducing satellite lifetimes. The ROSAT satellite deorbited unexpectedly early due to elevated solar activity.
- **Historical note:** Sputnik 1's orbit decayed due to drag and it reentered after 92 days. This orbit decay was used to measure the atmospheric density at 200 km altitude.

**In the simulation:**

At r = 2.0 (standard circular orbit), drag is nearly invisible. Try preset 3 (highly elliptical) or manually lower the orbit below r = 0.5 to watch the spiral decay in real time. The cumulative drag Δv is displayed in the perturbation HUD panel.

### 6.5 Solar Radiation Pressure (SRP)

**Physical cause:**

Sunlight carries momentum. When photons are absorbed or reflected by a satellite's surface, they impart a small force. This is the same principle as a **solar sail** — just much smaller for typical satellites.

**Radiation pressure formula:**

At 1 AU from the Sun, the solar radiation pressure is:
```
P_srp = S/c = 1361 W/m² / (3×10⁸ m/s) = 4.56 × 10⁻⁶ N/m²
```

Where `S` is the solar constant (total solar irradiance) and `c` is the speed of light.

**Acceleration model:**

```
a⃗_srp = Cr · (A/m) · P_srp · ŝ
```

Where:
- `Cr` = reflectivity coefficient
  - `Cr = 1`: perfect absorber (black body)
  - `Cr = 2`: perfect specular reflector (mirror)
  - `Cr ≈ 1.5`: typical satellite (mixed)
- `A/m` = area-to-mass ratio (m²/kg)
- `ŝ` = unit vector from Sun to satellite (direction of photon flow)

Note: SRP acts **away from the Sun**, not toward Earth's center. It does not follow an inverse-square law (it does fall off as 1/r²_sun, but since we're at essentially constant 1 AU from the Sun during Earth orbit, it's treated as constant magnitude with varying direction).

**Physical effects:**

- **Eccentricity vector rotation:** SRP continuously pushes the orbit slightly off-center, causing the eccentricity vector `e⃗` to slowly rotate. This is most visible in geostationary and highly elliptical orbits.
- **Seasonal variation:** As Earth orbits the Sun, the Sun direction relative to the orbital plane changes, causing the perturbation to vary with a yearly period.
- **Solar sails:** Intentionally enlarged area-to-mass ratio. The IKAROS spacecraft by JAXA (2010) demonstrated this for deep space propulsion.
- **Debris:** High area-to-mass ratio debris is highly perturbed by SRP. Old rocket upper stages have been observed to evolve dramatically in eccentricity over years.
- **JWST:** The James Webb Space Telescope at L2 uses solar pressure for attitude control — its sunshield provides a persistent SRP force that must be balanced by small thrusters.

**Earth umbra shadow model:**

When the satellite passes behind Earth (relative to the Sun), SRP drops to zero. The shadow test:

1. Compute the projection of the satellite's position onto the Sun-direction axis: `proj = r⃗ · ŝ`
2. If `proj < 0` (satellite is on the night side of Earth):
3. Compute the perpendicular distance from the Sun-Earth line: `d_perp = |r⃗ − proj · ŝ|`
4. If `d_perp < Rₑ`: satellite is in Earth's **umbra** → SRP = 0

The full model includes a **penumbra** (partial shadow) region, but the cylindrical umbra approximation is standard for most orbit propagators.

**In the simulation:**

Watch the eccentricity readout in the HUD slowly oscillate as the Sun direction rotates (every 120 sim seconds). In real missions this would take one year.

### 6.6 Third-Body Solar Gravity (Tidal)

**Physical cause:**

The Sun also pulls the satellite. In the Earth-centered reference frame, this appears as a differential (tidal) force — the satellite and Earth are pulled by the Sun, but with slightly different magnitudes and directions.

**Full third-body acceleration:**

The complete gravitational acceleration on the satellite due to the Sun is:
```
a⃗_Sun = GM_sun · (r⃗_Sun − r⃗_sat) / |r⃗_Sun − r⃗_sat|³
```

But this includes the acceleration of the Earth's reference frame itself. In the Earth-centered frame, we subtract the acceleration of Earth due to the Sun:
```
a⃗_tidal = GM_sun · [(r⃗_Sun − r⃗_sat) / |r⃗_Sun − r⃗_sat|³  −  r⃗_Sun / |r⃗_Sun|³]
```

**Tidal approximation (used in simulation):**

When the satellite is much closer to Earth than to the Sun (`|r⃗_sat| << |r⃗_Sun| = D`), we can expand to first order:

```
a⃗_tidal ≈ GM_sun / D³ · (r⃗_sat − 3 · (r⃗_sat · ŝ) · ŝ)
```

Where `ŝ` is the unit vector from Earth to the Sun. This is the **tidal force** approximation — the same physics that causes ocean tides on Earth. The `−3(r⃗·ŝ)ŝ` term means the force is larger along the Earth-Sun line (stretching) and smaller perpendicular to it (compressing).

**Physical effects:**

- **Kozai-Lidov oscillations:** For highly inclined orbits around Earth, the Sun's tidal force causes long-period coupled oscillations of inclination and eccentricity. The inclination increases as the eccentricity decreases, then vice versa. Over thousands of orbits this can dramatically change the orbit shape.
- **GPS constellation perturbations:** GPS satellites are perturbed primarily by the Moon and Sun (they're too high for significant J2 or drag). Solar/lunar gravity causes eccentricity growth that requires periodic station-keeping.
- **Historical note:** The Kozai-Lidov mechanism was independently discovered by Yoshihide Kozai (1962) and Mikhail Lidov (1962) while studying artificial satellites and natural satellites respectively. It's now understood to drive many exoplanet and binary star evolution scenarios.

**Real-world magnitudes:**

```
At LEO (400 km):    a_solar_tidal ≈ 5.6 × 10⁻⁷ m/s²   (≈ 3 × 10⁻⁷ of central gravity)
At GEO (36,000 km): a_solar_tidal ≈ 5.6 × 10⁻⁷ m/s²   (same! because it's position-independent in tidal approx)
At Moon's orbit:    a_solar_tidal ≈ 5.6 × 10⁻⁷ m/s²
```

The tidal force is essentially constant in magnitude regardless of altitude (for Earth-orbiting objects), which is why it's equally important for GEO as for LEO, unlike drag (which drops off exponentially) or J2 (which drops as 1/r²).

### 6.7 The Moon as an N-Body Perturber

The Moon's gravity is handled as a full N-body term — not approximated. The satellite feels direct Newtonian attraction toward the Moon:

```
a⃗_moon = GM_moon · (r⃗_moon − r⃗_sat) / |r⃗_moon − r⃗_sat|³
```

The Moon moves in a circular orbit around Earth with:
- Semi-major axis: 384,400 km ≈ 60.3 Earth radii (sim: 4.5 units)
- Period: 27.3 days (sim: 60 seconds, compressed)
- Inclination: 5.145° to the ecliptic

This is the dominant third-body perturbation for most Earth-orbiting satellites. At LEO, Moon perturbations are smaller than J2 but comparable to other perturbations. At GPS altitude and above, the Moon is a primary perturbation driver.

**Sphere of influence:**

The Moon's gravitational sphere of influence (the region where Moon's gravity dominates over Earth's differential tidal pull on the satellite) extends to approximately:
```
r_SOI = a_moon · (M_moon / M_earth)^(2/5)  ≈  66,200 km  ≈  10.4 Earth radii
```

For satellites with apoapsis beyond this radius, the three-body dynamics become strongly chaotic — this is the regime of lunar transfer orbits and libration point missions.

### 6.8 Perturbation Hierarchy

The relative magnitudes at LEO (400 km real altitude, r ≈ 1.06 Rₑ):

```
Force                          Acceleration (m/s²)    Ratio to g
─────────────────────────────────────────────────────────────────
Central gravity (J0)           8.70                   1.0
J2 oblateness                  2.6 × 10⁻²             3.0 × 10⁻³
J3 pear-shape                  1.0 × 10⁻⁵             1.2 × 10⁻⁶
J4 higher harmonic             5.0 × 10⁻⁶             5.7 × 10⁻⁷
Atmospheric drag (LEO)         2.0 × 10⁻⁶             2.3 × 10⁻⁷  (varies ×10 with solar activity)
Lunar gravity                  5.0 × 10⁻⁶             5.7 × 10⁻⁷
Solar radiation pressure       1.0 × 10⁻⁷             1.1 × 10⁻⁸  (varies with A/m)
Solar gravity (tidal)          5.6 × 10⁻⁷             6.4 × 10⁻⁸
Relativistic corrections       3.5 × 10⁻⁸             4.0 × 10⁻⁹
Yarkovsky thermal              ~10⁻¹²                 negligible for metal spacecraft
─────────────────────────────────────────────────────────────────
```

For the simulation (scaled units), J2 is boosted ×500 to be visually observable. All ratios between perturbations are preserved.

---

## 7. Numerical Integration

### 7.1 Why Integration Matters

The equation of motion is a **second-order ordinary differential equation**:

```
r̈ = f(r, ṙ, t) = [gravity + perturbations + thrust] / m
```

It has no general closed-form solution (except for the pure two-body case). We must integrate it numerically — advancing the position and velocity in small timesteps.

The choice of integration method determines:
- **Accuracy:** How closely does the numerical trajectory follow the true trajectory?
- **Stability:** Does error accumulate unboundedly (the orbit spirals in/out) or stay bounded?
- **Efficiency:** How many function evaluations per unit of simulated time?

### 7.2 Euler Integration (What We Don't Use)

The simplest method — and the most commonly taught — is Forward Euler:

```python
v = v + a * dt       # update velocity
r = r + v * dt       # update position
```

**Why it fails for orbits:**

Euler is a first-order method. It adds systematic energy into the orbit every step:
- Truncation error ≈ O(dt²) per step
- Global error accumulates as O(dt) over a fixed time interval
- For orbital mechanics: energy drift causes the orbit to slowly spiral **outward** (energy injection) even with perfect gravity

If you run the simulation with Euler integration and a reasonable timestep (dt = 0.01s), the satellite's orbit radius will visibly grow over seconds. This is unacceptable for any realistic simulation.

**Comparison:**

| Method | Order | Energy drift | Steps/orbit for 0.1% accuracy |
|--------|-------|-------------|-------------------------------|
| Forward Euler | 1st | Secular (unbounded) | ~1,000,000 |
| Leapfrog/Verlet | 2nd | Bounded (oscillatory) | ~1,000 |
| RK4 | 4th | None (but not symplectic) | ~100 |
| Yoshida 6th order | 6th | None + symplectic | ~20 |

### 7.3 Velocity Verlet (What We Use)

The Velocity Verlet algorithm:

```python
# Step 1: Update position using current velocity AND current acceleration
r_new = r + v * dt + 0.5 * a * dt²

# Step 2: Compute acceleration at NEW position
a_new = gravity(r_new) + perturbations(r_new, v) + thrust

# Step 3: Update velocity using AVERAGE of old and new acceleration
v_new = v + 0.5 * (a + a_new) * dt
```

This is a **second-order method** — the local error is O(dt³) per step, and the global error is O(dt²). More importantly, it is **symplectic**.

Why it's better than standard Euler:
- Uses the acceleration at the beginning AND end of each step (average)
- Position update includes the `½a·dt²` quadratic correction
- Naturally conserves phase-space volume (Liouville's theorem)

### 7.4 Symplectic Integrators & Energy Conservation

A **symplectic integrator** preserves the geometric structure of Hamiltonian mechanics. The Hamiltonian is:

```
H(r, v) = ½|v|² − GM/r
```

For a conservative system, H = total energy = constant. Euler integration violates this. Velocity Verlet does not — it conserves a **modified Hamiltonian** `H̃` that is very close to the true `H` (differs by O(dt²)).

**Consequence:** The energy in a Velocity Verlet orbit oscillates slightly but never systematically grows. Over 10,000 orbits, the orbit remains stable. This is why all serious orbital mechanics software (STK, GMAT, Orekit) uses symplectic integrators.

**Comparison with Runge-Kutta 4 (RK4):**

RK4 is more accurate per step (4th order) but is NOT symplectic. Over long integration times, energy drifts slightly. For orbital mechanics over millions of steps, Verlet often outperforms RK4 in long-term stability even though RK4 is more accurate per step.

The simulation achieves **< 0.03% energy drift** over 2000 steps on an unperturbed circular orbit.

### 7.5 Adaptive Timestep

When the satellite approaches Earth closely (near periapsis of an elliptical orbit) or is moving very fast (just after a large thrust burn), a fixed timestep may be too large — the satellite "jumps over" a curve in the trajectory, accumulating large errors.

The adaptive timestep logic:

```python
factor = max(
    1.0,
    speed / V_THRESHOLD,         # reduce dt when moving fast
    R_THRESHOLD / min_distance    # reduce dt when close to a body
)
dt_effective = dt_base / factor
```

This means:
- When `speed < V_THRESHOLD` and `distance > R_THRESHOLD`: use full `dt_base`
- When passing close to Earth at high speed: `factor` can be 10–100, so `dt_effective` is 10–100× smaller

The trade-off: more accuracy in critical regions at the cost of more computation steps. Since we use 8 sub-steps per render frame, a factor of 8 would bring the effective sub-step count to 64 per frame during a close approach.

---

## 8. 3D Mathematics

### 8.1 Coordinate System

The simulation uses a **right-handed Cartesian coordinate system** with:
- **Y axis** = Earth's rotation axis (pointing "north")
- **X axis** = pointing toward the vernal equinox (standard ECI reference)
- **Z axis** = completes the right-handed system

This matches standard Earth-Centred Inertial (ECI) coordinates used in astrodynamics (J2000 frame), except we haven't implemented the Earth's actual rotation — the frame is inertially fixed.

All positions, velocities, and acceleration vectors are 3-element NumPy float64 arrays: `np.array([x, y, z])`.

### 8.2 Rodrigues' Rotation Formula

To rotate velocity in the orbital plane (← / → keys), we need to rotate a 3D vector around an arbitrary axis. The **Rodrigues' rotation formula** does this without computing a full rotation matrix:

```
v' = v·cos(θ) + (k̂ × v)·sin(θ) + k̂·(k̂·v)·(1 − cos(θ))
```

Where:
- `v` = vector to rotate
- `k̂` = unit rotation axis
- `θ` = rotation angle (radians)

In code:
```python
def _rodrigues(v, axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    return v*c + np.cross(axis, v)*s + axis*np.dot(axis, v)*(1 - c)
```

**Why it works:** The formula decomposes `v` into a component parallel to the axis (`k̂·(k̂·v)`, unchanged by rotation) and a perpendicular component (rotated by the standard 2D rotation). This avoids the 9-element rotation matrix computation.

For orbit tilting (I/K keys), we rotate velocity around the **perpendicular to the current velocity** — specifically around `r⃗ × v⃗` (the angular momentum direction), tilting the orbit plane.

### 8.3 Camera Mathematics — LookAt & Perspective

**LookAt matrix:**

Given the camera eye position `E`, the target point `T`, and the world "up" vector `U`:

```
forward  = normalize(T − E)
right    = normalize(forward × U)
up_true  = right × forward          (re-orthogonalised)
```

The 4×4 view matrix then transforms world coordinates into camera space:
```
[  right.x    right.y    right.z   −dot(right, E)  ]
[  up.x       up.y       up.z      −dot(up, E)     ]
[ −forward.x −forward.y −forward.z  dot(forward, E)]
[  0          0          0          1              ]
```

**Perspective projection:**

```
f = cot(FOV/2)

[  f/aspect  0     0                    0           ]
[  0         f     0                    0           ]
[  0         0    (far+near)/(near−far)  2·far·near/(near−far) ]
[  0         0    −1                    0           ]
```

The `−1` in the [3,2] position is what performs the perspective divide — dividing by the camera-space Z coordinate maps nearby objects to larger screen coordinates and distant objects to smaller ones.

The result is a **clip-space** coordinate. After the GPU performs the perspective divide (`w` divide), the result is Normalized Device Coordinates (NDC) in the range [−1, 1]³, which maps to screen pixels.

### 8.4 Spherical Camera Coordinates

The camera uses **spherical coordinates** (r, θ, φ) for intuitive drag-to-orbit:

```
eye.x = target.x + r · sin(φ) · cos(θ)
eye.y = target.y + r · cos(φ)          ← φ=0 is at top
eye.z = target.z + r · sin(φ) · sin(θ)
```

Where:
- `r` = distance from target (zoom)
- `θ` = horizontal angle (azimuth), changes with horizontal mouse drag
- `φ` = vertical angle (elevation), clamped [5°, 175°] to prevent gimbal lock at poles
- Target = the point the camera looks at (can be panned with WASD)

Mouse dragging increments `θ` and `φ`; scroll wheel increments `r`. Converting back to Cartesian and passing to `lookAt()` gives a smooth, gimbal-lock-free orbit camera.

---

## 9. Rendering Pipeline

### 9.1 OpenGL & ModernGL

The renderer uses **OpenGL 3.3 Core Profile** through the **ModernGL** Python wrapper. Core Profile removes legacy fixed-function state, forcing explicit shaders for all rendering.

**Per-frame render order (painter's algorithm — back to front):**
1. Stars (point sprites, depth test on)
2. Reference grid (XZ plane)
3. Earth (opaque, textured)
4. Orbit trail (semi-transparent line strip)
5. Orbital plane ring (transparent)
6. Moon
7. Satellite mesh
8. Velocity arrow / angular momentum arrow
9. HUD overlay (2D, depth test off)

**Vertex Array Objects (VAOs):**

Each drawable object owns a VAO that records how vertex buffer data maps to shader input attributes. This avoids resetting state every frame.

**Dynamic buffers:**

The orbit trail changes every physics step. Rather than creating a new GPU buffer each frame, we pre-allocate a large buffer and write only the trail data portion, expanding if needed.

### 9.2 Blinn-Phong Lighting

The Earth and satellite use the **Blinn-Phong** reflectance model:

```
I = I_ambient + I_diffuse + I_specular

I_ambient  = k_a · L_a · surface_color
I_diffuse  = k_d · max(0, N̂·L̂) · L_d · surface_color
I_specular = k_s · max(0, N̂·Ĥ)^n · L_s

where Ĥ = normalize(L̂ + V̂)   (the half-vector)
```

**Why Blinn instead of Phong?**

Classic Phong specular uses `max(0, R̂·V̂)^n` where `R̂` is the reflection of the light direction. Blinn's modification replaces this with the half-vector `Ĥ` between the view and light directions. This avoids computing the reflection and is physically more accurate for many materials. It's also slightly faster.

**Atmospheric rim glow:**

The blue atmospheric halo on Earth's edge is a simple fresnel-like approximation:
```glsl
float rim = 1.0 - max(dot(N, V), 0.0);   // 0 at face, 1 at edge
color += pow(rim, 3.5) * vec3(0.1, 0.32, 0.78) * 0.88;
```

This mimics Rayleigh scattering in the atmosphere (shorter wavelengths/blue scatter more), though it's purely artistic rather than physically derived.

**Night-side terminator:**

The transition from day to night uses a smooth step over a narrow latitude band:
```glsl
float night = smoothstep(-0.08, 0.14, dot(N, L));
color = mix(color * 0.07, color, night);
```

The `smoothstep` function provides a C2-continuous (smooth second derivative) transition, avoiding a hard edge.

### 9.3 Earth Texture Mapping

**UV mapping for a sphere:**

Each vertex on the UV-sphere has texture coordinates:
```
u = θ / (2π)    ← 0..1 mapping longitude (horizontal)
v = φ / π       ← 0..1 mapping latitude (vertical, 0=north pole)
```

This is the **equirectangular projection** — the same format as most online Earth texture maps (Google Earth exports, NASA Blue Marble, etc.).

**Three texture layers:**

1. **Diffuse texture** (`earth.jpg`): The land/ocean color map
2. **Specular map** (`specular.jpg`): Single-channel — white for ocean (shiny), black for land (matte). Multiplies the specular highlight intensity so oceans glint under the Sun but continents don't.
3. **Cloud layer** (`clouds.png`): RGBA texture blended over the surface. The cloud UV coordinates are offset slightly each frame: `u_cloud = fract(u + time × 0.0018)`, making clouds drift slowly westward.

**Mipmap filtering:**

All textures use `LINEAR_MIPMAP_LINEAR` (trilinear) filtering with anisotropy = 8. This prevents aliasing artefacts when the Earth is viewed at a steep angle or from far away.

### 9.4 Satellite Model Geometry

The satellite is an ISS-inspired procedural mesh built entirely from primitive shapes:

| Part | Geometry | Purpose |
|------|----------|---------|
| Main truss | Thin box (2.6 × 0.09 × 0.09) | Long horizontal backbone |
| Central habitat | Cylinder (r=0.165, h=0.52, rotated 90°) | Pressurised crew module |
| Forward module | Smaller cylinder | Connecting node |
| Docking port | Very small cylinder | Nose docking interface |
| Solar panels (×4) | Thin flat boxes, extending ±Y from truss ends | Power generation |
| Radiator panels (×2) | Thin boxes along Z | Thermal control |
| Antenna dish | Inverted cone | Communication |
| Antenna mast | Thin cylinder | Dish support |

**Satellite orientation:**

The model matrix is recomputed every frame to orient the satellite physically correctly:
- The truss X-axis aligns with the velocity vector (prograde)
- The Y-axis tracks the orbital normal (`L⃗/|L⃗|`)
- Z-axis completes the right-handed frame

This means the solar panels are always correctly oriented relative to the orbital plane regardless of inclination.

---

## 10. Simulation Scaling

The simulation does not use real SI units — everything is scaled for visual clarity and real-time performance.

**Unit mapping:**

| Quantity | Real Value | Sim Value | Scale Factor |
|----------|-----------|-----------|-------------|
| Earth radius | 6,371 km | 0.25 units | 1 unit = 25,484 km |
| ISS orbit radius | 6,771 km | ~0.266 units | |
| Moon distance | 384,400 km | 4.5 units | |
| Earth GM | 3.986 × 10¹⁴ m³/s² | 2.5 units³/s² | |
| ISS orbital period | 92 minutes | ~11 sim-seconds | ×500 acceleration |
| J2 coefficient | 1.083 × 10⁻³ | 5.41 × 10⁻¹ | ×500 for visual precession |

**Ratios are preserved:**
- Moon GM / Earth GM = 1/81 (same in sim)
- Moon orbit radius / Earth radius = 60.3 (same: 4.5 / 0.25 = 18 — slightly compressed for visibility)
- Escape velocity / circular velocity = √2 (exactly preserved)
- Orbital period ratios (Kepler's 3rd law) are exactly preserved

**Why scale J2 up?**

Real J2 causes nodal precession of about 7°/day for ISS. At 500× time acceleration, that's 3500°/day ≈ 10 full precessions/day. Visible but subtle. We also scale J2 by ×500 magnitude so precession is visible within seconds of sim time. The ratio J4/J2 is preserved.

**Why scale drag up?**

Real atmospheric drag at 400 km causes ~2 km/day of altitude loss. At 500× time acceleration, you'd see ~1 km per simulated second — invisible on the scale of our simulation. The scaled drag causes visible orbital decay at altitudes equivalent to real LEO, making the physics intuitive to observe.

---

## 11. Code Architecture

### Module Dependency Graph

```
constants.py          (no dependencies)
     │
perturbations.py ─────> constants.py
     │
physics.py ───────────> constants.py
           ───────────> perturbations.py
     │
camera.py ────────────> constants.py
     │
renderer.py ──────────> constants.py
             (runtime) > physics.py (reads satellite state)
     │
main.py ──────────────> all of the above
```

### Class Hierarchy

```
PhysicsEngine
├── earth: GravitationalBody        (stationary, GM=2.5, R=0.25)
├── moon:  GravitationalBody        (orbiting Earth, GM=0.031)
│   └── advance(dt) → circular orbit kinematics
└── satellite: Satellite
    ├── pos, vel, acc: np.array[3]
    ├── elements: OrbitalElements   (updated each step)
    ├── thruster: ThrusterSystem    (prograde/retro/radial/normal)
    ├── perturbations: PerturbationEngine
    │   └── cfg: PerturbationConfig (all toggles + parameters)
    └── step(dt, bodies)
        ├── adaptive_dt()
        ├── Velocity Verlet position update
        ├── _resolve_collisions()
        ├── _gravity() ← N-body sum
        ├── perturbations.total() ← J2+J4+drag+SRP+solar
        ├── thruster.compute_dv()
        └── Velocity Verlet velocity update
```

### Design Principles

1. **No global state:** All state is owned by class instances. `PhysicsEngine` owns `Satellite` owns `PerturbationEngine`. No module-level mutable variables.

2. **Constants module:** Every magic number has a name in `constants.py`. This makes tuning easy — one file controls everything.

3. **Pure perturbation functions:** Each perturbation (`_acc_j2`, `_acc_drag`, etc.) is a pure function of position and velocity. No hidden state. Trivially unit-testable.

4. **Renderer reads physics, never writes:** The renderer only reads `physics.satellite.pos`, `physics.satellite.trail`, etc. Physics and rendering are fully decoupled.

5. **Fixed-timestep physics with render-independent loop:** The physics runs at a fixed `DT_BASE` with 8 sub-steps per render frame. This ensures physics determinism regardless of rendering frame rate.

---

## 12. Extending the Simulator

### Add a New Perturbation

```python
# In perturbations.py

@dataclass
class PerturbationConfig:
    enable_yarkovsky: bool = True        # 1. Add toggle
    yarkovsky_coeff: float = 1e-9        # 2. Add parameter

def _acc_yarkovsky(pos, vel, cfg):       # 3. Write pure function
    # Thermal re-radiation: force along spin axis direction
    spin_axis = np.array([0., 1., 0.])
    return cfg.yarkovsky_coeff * spin_axis

class PerturbationEngine:
    def total(self, pos, vel):
        # ... existing forces ...
        if self.cfg.enable_yarkovsky:    # 4. Add to total()
            a_yark = _acc_yarkovsky(pos, vel, self.cfg)
            self.last_yarkovsky = float(np.linalg.norm(a_yark))
            acc += a_yark
        return acc
```

### Add a New Gravitational Body (e.g., the Sun at L2)

```python
# In PhysicsEngine._build_bodies():
self.sun = GravitationalBody(
    name="Sun",
    gm=800.0,          # large GM
    radius=2.0,        # render radius
    pos=np.array([53.75, 0., 0.]),
    parent=self.earth,
    orbit_radius=53.75,
    orbit_period=365.25 * 24 * sim_day_length,
    orbit_inclination_deg=0.0,
)
self.bodies.append(self.sun)
```

### Switch to RK4 Integration

```python
def _rk4_step(self, dt, bodies):
    def deriv(r, v):
        a = self._gravity(r, bodies) + self.perturbations.total(r, v)
        return v, a

    v1, a1 = deriv(self.pos, self.vel)
    v2, a2 = deriv(self.pos + 0.5*dt*v1, self.vel + 0.5*dt*a1)
    v3, a3 = deriv(self.pos + 0.5*dt*v2, self.vel + 0.5*dt*a2)
    v4, a4 = deriv(self.pos + dt*v3,     self.vel + dt*a3)

    self.pos += dt/6 * (v1 + 2*v2 + 2*v3 + v4)
    self.vel += dt/6 * (a1 + 2*a2 + 2*a3 + a4)
```

Note: RK4 uses 4 force evaluations per step (vs Verlet's 1), but has 4th-order accuracy. For very high eccentricity orbits, RK4 may produce more accurate close-approach trajectories.

### Add Multiple Satellites

```python
class PhysicsEngine:
    def __init__(self):
        self._build_bodies()
        self.satellites = []          # list instead of single
        self.reset()

    def add_satellite(self, pos, vel):
        sat = Satellite(pos, vel)
        sat.acc = sat._gravity(pos, self.bodies)
        self.satellites.append(sat)
        return sat

    def step(self, dt):
        for body in self.bodies:
            body.advance(dt)
        for sat in self.satellites:
            sat.step(dt, self.bodies)
```

### Export Orbital Data

```python
import csv

def export_trajectory(satellite, filename):
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y', 'z', 'semi_major_axis', 'eccentricity',
                    'inclination_deg', 'true_anomaly_deg'])
        for pos in satellite.trail:
            oe = OrbitalElements.from_state(pos, satellite.vel, GM_EARTH)
            w.writerow([*pos, oe.semi_major_axis, oe.eccentricity,
                        oe.inclination_deg, oe.true_anomaly_deg])
```

---

## 13. References & Further Reading

### Textbooks

- **Bate, Mueller & White** — *Fundamentals of Astrodynamics* (1971) — The classic USAF Academy text. Everything about two-body problem, orbital elements, maneuvers.
- **Vallado** — *Fundamentals of Astrodynamics and Applications* (2013) — Comprehensive modern reference including SGP4/SDP4 propagators and state vector conversions.
- **Battin** — *An Introduction to the Mathematics and Methods of Astrodynamics* (1987) — Advanced mathematical treatment including perturbation theory.
- **Montenbruck & Gill** — *Satellite Orbits: Models, Methods, Applications* (2001) — The definitive reference on numerical orbit propagation, perturbations, and observation models.

### Perturbation Theory

- **Kozai, Y.** (1959) — "The Motion of a Close Earth Satellite" — *The Astronomical Journal*, 64, 367. Original derivation of J2 secular effects.
- **Kozai, Y.** (1962) — "Secular Perturbations of Asteroids with High Inclination and Eccentricity" — First paper on Kozai-Lidov oscillations.
- **Lidov, M.L.** (1962) — "The evolution of orbits of artificial satellites of planets..." — *Planetary and Space Science*, 9, 719.
- **King-Hele, D.G.** (1987) — *Satellite Orbits in an Atmosphere* — Definitive treatment of atmospheric drag perturbations.

### Numerical Methods

- **Hairer, Lubich & Wanner** — *Geometric Numerical Integration* (2006) — Comprehensive treatment of symplectic integrators and their long-term energy conservation properties.
- **Yoshida, H.** (1990) — "Construction of higher order symplectic integrators" — *Physics Letters A*, 150(5-7), 262-268. If you need higher-order symplectic integration.

### Online Resources

- [NASA Orbital Mechanics Guide](https://www.grc.nasa.gov/WWW/k-12/rocket/orbmect.html) — Accessible introductions
- [ESA Space Debris Office](https://www.esa.int/Space_Safety/Space_Debris) — Real perturbation effects on tracked debris
- [Heavens-Above.com](https://heavens-above.com) — Live ISS tracking showing real J2 precession
- [GMAT (General Mission Analysis Tool)](https://gmat.gsfc.nasa.gov/) — NASA's open-source high-fidelity propagator
- [Orekit](https://www.orekit.org/) — Java-based open-source astrodynamics library used operationally

### Physics of Solar Radiation Pressure

- **Vokrouhlický & Farinella** (1998) — "The Yarkovsky seasonal effect on asteroidal fragments" — *Astronomy & Astrophysics*, 335, 351.
- **Milani et al.** (1987) — *Non-gravitational Perturbations and Satellite Geodesy* — Complete treatment of SRP and thermal forces.

### Historical Papers

- **Newton, I.** (1687) — *Philosophiæ Naturalis Principia Mathematica* — Mathematical proof that inverse-square gravity produces conic orbits (Book I, Prop. XI).
- **Kepler, J.** (1609) — *Astronomia Nova* — First and second laws derived empirically from Mars observations.
- **Kepler, J.** (1619) — *Harmonices Mundi* — Third law published here.
- **Hohmann, W.** (1925) — *Die Erreichbarkeit der Himmelskörper* (The Accessibility of Heavenly Bodies) — The optimal two-impulse transfer orbit still used today.
- **Laplace, P.S.** (1799) — Introduction of the eccentricity vector (Laplace-Runge-Lenz vector) in the context of celestial mechanics.

---

*Built with Python, NumPy, ModernGL, GLFW, Pygame, and Pillow.*

*The physics engine implements standard astrodynamics equations from Bate-Mueller-White (1971) and Vallado (2013). Perturbation models follow Montenbruck & Gill (2001). All equations scaled for real-time interactive visualization.*