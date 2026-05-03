"""
camera.py — Spherical Orbit Camera
════════════════════════════════════

Uses spherical coordinates (r, θ, φ) for intuitive mouse-drag orbiting.

SPHERICAL → CARTESIAN:
  x = r · sin(φ) · cos(θ)
  y = r · cos(φ)            ← Y is "up" in our world
  z = r · sin(φ) · sin(θ)

  θ (theta) = horizontal angle (azimuth)
  φ (phi)   = vertical angle  (polar / elevation)

LOOKÁT MATRIX:
  Given eye position E, target T, up vector U:
    forward = normalize(T - E)
    right   = normalize(forward × U)
    up_true = right × forward
  This builds an orthonormal camera frame → 4×4 view matrix.

PERSPECTIVE PROJECTION:
  Standard OpenGL perspective matrix from FOV, aspect, near, far planes.
  Projects 3D → 2D clip space. GPU then does divide-by-w.
"""

import numpy as np
import math


class Camera:
    def __init__(self, distance: float = 6.0):
        self.distance = distance
        self.theta    = 30.0   # horizontal angle (degrees)
        self.phi      = 55.0   # vertical angle (degrees, clamped 5..175)
        self.target   = np.array([0.0, 0.0, 0.0])  # look-at point
        self.fov      = 45.0
        self._aspect  = 1280 / 800

    # ── Camera movement ───────────────────────────────────────────
    def orbit(self, d_theta: float, d_phi: float):
        """Rotate camera around target by dragging mouse."""
        self.theta += d_theta
        self.phi    = max(5.0, min(175.0, self.phi + d_phi))

    def zoom(self, delta: float):
        """Scroll wheel zoom."""
        self.distance = max(1.5, min(20.0, self.distance + delta))

    def pan(self, dx: float, dy: float):
        """
        Pan target point in camera's right/up plane.
        This keeps the scene centered where the user wants.
        """
        eye    = self._eye()
        fwd    = _normalize(self.target - eye)
        right  = _normalize(np.cross(fwd, np.array([0.0, 1.0, 0.0])))
        up     = np.cross(right, fwd)
        self.target += right * dx + up * dy

    # ── Position ──────────────────────────────────────────────────
    def _eye(self) -> np.ndarray:
        """Convert spherical coordinates to Cartesian eye position."""
        t   = math.radians(self.theta)
        p   = math.radians(self.phi)
        sp  = math.sin(p)
        cp  = math.cos(p)
        st  = math.sin(t)
        ct  = math.cos(t)
        return self.target + self.distance * np.array([sp * ct, cp, sp * st])

    def position(self) -> np.ndarray:
        return self._eye()

    # ── Matrices ─────────────────────────────────────────────────
    def matrices(self):
        """Return (view_matrix, proj_matrix) as flat float32 arrays (column-major)."""
        view = _look_at(self._eye(), self.target, np.array([0.0, 1.0, 0.0]))
        proj = _perspective(self.fov, self._aspect, 0.01, 200.0)
        return view, proj

    def update_projection(self, w: int, h: int):
        self._aspect = w / h if h > 0 else 1.0


# ── Math helpers ─────────────────────────────────────────────────────
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Build a right-handed view matrix (column-major, OpenGL convention).
    Transforms world coordinates into camera space.
    """
    f = _normalize(target - eye)        # forward
    r = _normalize(np.cross(f, up))     # right
    u = np.cross(r, f)                  # true up (re-orthogonalized)

    # Rotation part (inverse = transpose for orthogonal)
    # Translation part (dot products)
    m = np.identity(4, dtype=np.float32)
    m[0, 0:3] =  r
    m[1, 0:3] =  u
    m[2, 0:3] = -f
    m[0, 3]   = -np.dot(r, eye)
    m[1, 3]   = -np.dot(u, eye)
    m[2, 3]   =  np.dot(f, eye)
    return m   # already column-major layout for GLSL


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """
    Standard OpenGL perspective projection matrix.
    Maps view-space frustum to NDC cube [-1,1]³.

      f = cot(fov/2)
      [f/aspect  0        0                  0           ]
      [0         f        0                  0           ]
      [0         0  (f+n)/(n-f)    2·f·n/(n-f)           ]
      [0         0       -1                  0           ]
    """
    f   = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    nf  = near - far
    m   = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / nf
    m[2, 3] = (2 * far * near) / nf
    m[3, 2] = -1.0
    return m
