"""
camera.py — Spherical Orbit Camera
════════════════════════════════════

Spherical coordinates (r, θ, φ) give gimbal-lock-free mouse-orbit.

  eye.x = target.x + r · sin(φ) · cos(θ)
  eye.y = target.y + r · cos(φ)
  eye.z = target.z + r · sin(φ) · sin(θ)

lookAt and perspective matrix implementations follow standard OpenGL convention.
"""

import numpy as np
import math
from constants import CAMERA_FOV, CAMERA_NEAR, CAMERA_FAR


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a right-handed view matrix (column-major, OpenGL convention)."""
    f = _normalize(target - eye)
    r = _normalize(np.cross(f, up))
    u = np.cross(r, f)
    m = np.identity(4, dtype=np.float32)
    m[0, 0:3] =  r;  m[0, 3] = -float(np.dot(r, eye))
    m[1, 0:3] =  u;  m[1, 3] = -float(np.dot(u, eye))
    m[2, 0:3] = -f;  m[2, 3] =  float(np.dot(f, eye))
    return m


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Standard OpenGL perspective projection matrix."""
    f  = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    nf = near - far
    m  = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / nf
    m[2, 3] = (2 * far * near) / nf
    m[3, 2] = -1.0
    return m


class Camera:
    """Arcball-style spherical orbit camera."""

    def __init__(self, distance: float = 6.0):
        self.distance = distance
        self.theta    = 30.0
        self.phi      = 55.0
        self.target   = np.array([0.0, 0.0, 0.0])
        self._aspect  = 1280.0 / 800.0

    def orbit(self, d_theta: float, d_phi: float) -> None:
        self.theta += d_theta
        self.phi    = max(5.0, min(175.0, self.phi + d_phi))

    def zoom(self, delta: float) -> None:
        self.distance = max(1.0, min(25.0, self.distance + delta))

    def pan(self, dx: float, dy: float) -> None:
        eye   = self._eye()
        fwd   = _normalize(self.target - eye)
        right = _normalize(np.cross(fwd, np.array([0., 1., 0.])))
        up    = np.cross(right, fwd)
        self.target += right * dx + up * dy

    def _eye(self) -> np.ndarray:
        t, p = math.radians(self.theta), math.radians(self.phi)
        return self.target + self.distance * np.array(
            [math.sin(p)*math.cos(t), math.cos(p), math.sin(p)*math.sin(t)])

    def position(self) -> np.ndarray:
        return self._eye()

    def matrices(self):
        view = _look_at(self._eye(), self.target, np.array([0., 1., 0.]))
        proj = _perspective(CAMERA_FOV, self._aspect, CAMERA_NEAR, CAMERA_FAR)
        return view, proj

    def update_projection(self, w: int, h: int) -> None:
        self._aspect = w / h if h > 0 else 1.0