"""
╔══════════════════════════════════════════════════════════════════════╗
║   ORBITAL MECHANICS ENGINE v2 — Production Grade                    ║
║   Python + NumPy + ModernGL + GLFW                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  SETUP:  pip install moderngl glfw numpy pygame Pillow              ║
║          python main.py                                             ║
╠══════════════════════════════════════════════════════════════════════╣
║  CAMERA                                                             ║
║    Left drag     Orbit camera                                       ║
║    Scroll        Zoom                                               ║
║    WASD          Pan target                                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  SATELLITE CONTROL                                                  ║
║    ↑ / ↓         Increase / decrease speed (rotate velocity)        ║
║    ← / →         Rotate velocity in orbital plane                  ║
║    I / K         Tilt inclination ±                                 ║
║    Z             Continuous prograde thrust (held)                  ║
║    X             Continuous retrograde thrust (held)                ║
║    Q             Impulse prograde burst                             ║
║    C             Snap to circular orbit speed                       ║
║    E             Snap to escape velocity                            ║
║    H             Hohmann transfer burn to r=4.0                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  SIMULATION                                                         ║
║    SPACE         Pause / Resume                                     ║
║    R             Reset to circular orbit                            ║
║    F / S         Speed up / slow down time                          ║
║    1             Preset: Circular                                   ║
║    2             Preset: Elliptical                                 ║
║    3             Preset: Highly Elliptical (Molniya-like)           ║
║    4             Preset: Escape trajectory                          ║
║    5             Preset: Inclined 45°                               ║
║    6             Preset: Polar orbit                                ║
║    7             Preset: Retrograde                                 ║
║    ESC           Quit                                               ║
╠══════════════════════════════════════════════════════════════════════╣
║  PERTURBATIONS                                                       ║
║    P             Toggle ALL perturbations on/off                     ║
║    J             Toggle J2 oblateness                                ║
║    D             Toggle atmospheric drag                             ║
║    N             Toggle solar radiation pressure                     ║
║    G             Toggle solar gravity (3rd body)                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys
import glfw
import moderngl
import numpy as np
import time

from constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE,
    DT_BASE, STEPS_PER_FRAME,
    DT_MULT_MIN, DT_MULT_MAX, MSAA_SAMPLES,
)
from physics  import PhysicsEngine, PRESETS
from camera   import Camera
from renderer import Renderer


# ── GLFW / GL initialisation ──────────────────────────────────────────

def _init_window(width: int, height: int, title: str):
    """Create and configure GLFW window with OpenGL 3.3 core profile."""
    if not glfw.init():
        print("[FATAL] GLFW init failed"); sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)   # macOS requirement
    glfw.window_hint(glfw.SAMPLES, MSAA_SAMPLES)

    win = glfw.create_window(width, height, title, None, None)
    if not win:
        glfw.terminate()
        print("[FATAL] Window creation failed"); sys.exit(1)

    glfw.make_context_current(win)
    glfw.swap_interval(1)   # vsync on
    return win


# ── Input state ───────────────────────────────────────────────────────

class InputState:
    """Centralises all raw input state to avoid scattered globals."""
    def __init__(self):
        self.mouse_dragging = False
        self.mouse_last_x   = 0.0
        self.mouse_last_y   = 0.0

        # Mutable simulation state (single-element lists so closures can mutate)
        self.paused   = [False]
        self.dt_mult  = [1.0]


# ── Callback factory ──────────────────────────────────────────────────

def _build_callbacks(window, physics: PhysicsEngine, camera: Camera,
                     inp: InputState, renderer: Renderer, ctx):
    """
    Return all GLFW callback functions as closures.
    All closures capture their dependencies explicitly — no module-level state.
    """

    # ── Mouse ──────────────────────────────────────────────────────
    def on_mouse_button(win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            inp.mouse_dragging = (action == glfw.PRESS)
            x, y = glfw.get_cursor_pos(win)
            inp.mouse_last_x, inp.mouse_last_y = x, y

    def on_cursor_pos(win, x, y):
        if inp.mouse_dragging:
            camera.orbit((x - inp.mouse_last_x) * 0.35,
                         (y - inp.mouse_last_y) * 0.35)
        inp.mouse_last_x, inp.mouse_last_y = x, y

    def on_scroll(win, _xoff, yoff):
        camera.zoom(-yoff * 0.35)

    def on_resize(win, w, h):
        if w > 0 and h > 0:
            ctx.viewport = (0, 0, w, h)
            camera.update_projection(w, h)
            renderer.update_projection(w, h)

    # ── Keyboard (one-shot) ────────────────────────────────────────
    _PRESET_KEYS = {
        glfw.KEY_1: "circular",
        glfw.KEY_2: "elliptical",
        glfw.KEY_3: "highly_elliptical",
        glfw.KEY_4: "escape",
        glfw.KEY_5: "inclined_45",
        glfw.KEY_6: "polar",
        glfw.KEY_7: "retrograde",
    }

    def on_key(win, key, _scancode, action, mods):
        sat = physics.satellite

        # ── Held-key thrust is handled in the loop; only one-shot here ──
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(win, True)

            elif key == glfw.KEY_SPACE:
                inp.paused[0] = not inp.paused[0]

            elif key == glfw.KEY_R:
                physics.reset()
                inp.dt_mult[0] = 1.0

            elif key == glfw.KEY_F:
                inp.dt_mult[0] = min(inp.dt_mult[0] * 1.5, DT_MULT_MAX)

            elif key == glfw.KEY_S and not (mods & glfw.MOD_SHIFT):
                inp.dt_mult[0] = max(inp.dt_mult[0] / 1.5, DT_MULT_MIN)

            elif key == glfw.KEY_C:
                sat.snap_circular()

            elif key == glfw.KEY_E:
                sat.snap_escape()

            elif key == glfw.KEY_Q:
                # Single prograde impulse burst
                from physics import ThrusterSystem
                sat.apply_impulse(ThrusterSystem.PROGRADE)

            elif key == glfw.KEY_H:
                # Hohmann transfer to r=4.0 sim units
                dv1, dv2 = physics.hohmann_to(4.0)
                print(f"[Hohmann] Δv₁={dv1:+.4f}  Δv₂(needed)={dv2:+.4f}")

            elif key in _PRESET_KEYS:
                physics.load_preset(_PRESET_KEYS[key])

            # ── Perturbation toggles ───────────────────────────────
            elif key == glfw.KEY_P:
                # Toggle ALL perturbations at once
                cfg = physics.satellite.perturbations.cfg
                all_on = (cfg.enable_j2 and cfg.enable_drag and
                          cfg.enable_srp and cfg.enable_solar_gravity)
                new_state = not all_on
                cfg.enable_j2            = new_state
                cfg.enable_j4            = new_state
                cfg.enable_drag          = new_state
                cfg.enable_srp           = new_state
                cfg.enable_solar_gravity = new_state
                status = "ON" if new_state else "OFF"
                print(f"[Perturbations] ALL → {status}")

            elif key == glfw.KEY_J:
                cfg = physics.satellite.perturbations.cfg
                cfg.enable_j2 = not cfg.enable_j2
                cfg.enable_j4 = cfg.enable_j2   # J4 follows J2
                print(f"[Perturbations] J2/J4 → {'ON' if cfg.enable_j2 else 'OFF'}")

            elif key == glfw.KEY_D and inp.paused[0]:
                # Only toggle drag when paused (avoid key conflict with pan)
                cfg = physics.satellite.perturbations.cfg
                cfg.enable_drag = not cfg.enable_drag
                print(f"[Perturbations] Drag → {'ON' if cfg.enable_drag else 'OFF'}")

            elif key == glfw.KEY_N:
                cfg = physics.satellite.perturbations.cfg
                cfg.enable_srp = not cfg.enable_srp
                print(f"[Perturbations] SRP → {'ON' if cfg.enable_srp else 'OFF'}")

            elif key == glfw.KEY_G:
                cfg = physics.satellite.perturbations.cfg
                cfg.enable_solar_gravity = not cfg.enable_solar_gravity
                print(f"[Perturbations] Solar Gravity → {'ON' if cfg.enable_solar_gravity else 'OFF'}")

        # Thrust key release (continuous thrust)
        if action == glfw.RELEASE:
            from physics import ThrusterSystem
            if key == glfw.KEY_Z:
                sat.thruster.release(ThrusterSystem.PROGRADE)
            elif key == glfw.KEY_X:
                sat.thruster.release(ThrusterSystem.RETROGRADE)

    return on_mouse_button, on_cursor_pos, on_scroll, on_resize, on_key


# ── Held-key poll (run every frame) ──────────────────────────────────

def _poll_held_keys(window, physics: PhysicsEngine, camera: Camera,
                    inp: InputState) -> None:
    """
    Check held keys each frame for smooth continuous inputs.
    Velocity manipulations clear the trail for visual clarity.
    """
    sat  = physics.satellite
    down = lambda k: glfw.get_key(window, k) == glfw.PRESS

    # Camera pan
    pan_speed = 0.025
    if down(glfw.KEY_A): camera.pan(-pan_speed, 0)
    if down(glfw.KEY_D): camera.pan( pan_speed, 0)
    if down(glfw.KEY_W): camera.pan(0,  pan_speed)

    if not inp.paused[0]:
        boost = 0.006

        # Speed up / down
        if down(glfw.KEY_UP):
            sat.vel = sat.vel * (1.0 + boost)
            sat.clear_trail()
        if down(glfw.KEY_DOWN):
            sat.vel = sat.vel * max(0.01, 1.0 - boost)
            sat.clear_trail()

        # Rotate velocity in orbital plane
        if down(glfw.KEY_LEFT):
            sat.rotate_vel_in_plane(-1.2)
            sat.clear_trail()
        if down(glfw.KEY_RIGHT):
            sat.rotate_vel_in_plane( 1.2)
            sat.clear_trail()

        # Inclination tilt
        if down(glfw.KEY_I):
            sat.tilt_orbit( 0.4)
            sat.clear_trail()
        if down(glfw.KEY_K):
            sat.tilt_orbit(-0.4)
            sat.clear_trail()

        # Continuous thruster (Z = prograde, X = retrograde)
        from physics import ThrusterSystem
        if down(glfw.KEY_Z):
            sat.thruster.press(ThrusterSystem.PROGRADE)
        else:
            sat.thruster.release(ThrusterSystem.PROGRADE)

        if down(glfw.KEY_X):
            sat.thruster.press(ThrusterSystem.RETROGRADE)
        else:
            sat.thruster.release(ThrusterSystem.RETROGRADE)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: initialise subsystems, run main loop, clean up."""

    window = _init_window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    physics  = PhysicsEngine()
    camera   = Camera(distance=6.0)
    renderer = Renderer(ctx, WINDOW_WIDTH, WINDOW_HEIGHT)
    inp      = InputState()

    # Register callbacks
    (on_mb, on_cp, on_sc, on_rz, on_key) = _build_callbacks(
        window, physics, camera, inp, renderer, ctx)
    glfw.set_mouse_button_callback(window, on_mb)
    glfw.set_cursor_pos_callback(window,   on_cp)
    glfw.set_scroll_callback(window,       on_sc)
    glfw.set_framebuffer_size_callback(window, on_rz)
    glfw.set_key_callback(window,          on_key)

    print(f"[INIT] {WINDOW_TITLE}")
    print(f"[INIT] Physics: Earth GM={physics.earth.gm}  Moon GM={physics.moon.gm}")
    print(f"[INIT] Presets available: {list(PRESETS.keys())}")

    # Main loop
    while not glfw.window_should_close(window):
        glfw.poll_events()

        # Held-key inputs (smooth per-frame)
        _poll_held_keys(window, physics, camera, inp)

        # Physics sub-steps
        sat = physics.satellite
        if not inp.paused[0] and not sat.escaped:
            dt = DT_BASE * inp.dt_mult[0]
            for _ in range(STEPS_PER_FRAME):
                physics.step(dt)

        # Render
        ctx.clear(0.02, 0.03, 0.06, 1.0)
        view, proj = camera.matrices()
        renderer.render(ctx, physics, camera, view, proj,
                        inp.paused[0], inp.dt_mult[0])
        glfw.swap_buffers(window)

    glfw.terminate()
    print("[EXIT] Simulation terminated cleanly.")


if __name__ == "__main__":
    main()