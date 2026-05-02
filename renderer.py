"""
renderer.py — ModernGL Renderer
════════════════════════════════

Renders:
  1. Star-field background (billboard points)
  2. Reference grid (XZ plane)
  3. Earth sphere (Blinn-Phong lighting)
  4. Orbit trail (line strip with fade)
  5. Satellite (small glowing sphere)
  6. Velocity vector arrow
  7. HUD overlay (2D text via pygame surface → texture)

SHADERS:
  • Sphere: vertex shader transforms positions + normals, fragment does
    Blinn-Phong (ambient + diffuse + specular).
  • Trail:  vertex shader applies color fade based on 'age' attribute.
  • Grid:   simple flat color with depth.
  • HUD:    fullscreen quad with orthographic projection.

BLINN-PHONG LIGHTING:
  ambient  = k_a · L_a
  diffuse  = k_d · max(0, N·L) · L_d
  specular = k_s · max(0, N·H)^shininess · L_s
  H = normalize(L + V)   ← half-vector (Blinn's approximation)

SPHERE GENERATION:
  Tessellate by latitude (stacks) and longitude (sectors).
  Each vertex: position = (sin(φ)cos(θ), cos(φ), sin(φ)sin(θ)) × radius
  Normal = same unit vector (sphere normals point outward from center).
"""

import numpy as np
import math
import moderngl
import time


# ── Shader sources ────────────────────────────────────────────────────

# ── Sphere ────────────────────────────────────────────────────────────
SPHERE_VERT = """
#version 330 core
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

in vec3 in_pos;
in vec3 in_norm;

out vec3 v_norm;
out vec3 v_frag_pos;

void main() {
    vec4 world_pos = model * vec4(in_pos, 1.0);
    v_frag_pos     = world_pos.xyz;
    v_norm         = mat3(transpose(inverse(model))) * in_norm;
    gl_Position    = proj * view * world_pos;
}
"""

SPHERE_FRAG = """
#version 330 core
uniform vec3 light_dir;     // world-space, normalized, points TO light
uniform vec3 light_color;
uniform vec3 ambient;
uniform vec3 obj_color;
uniform vec3 cam_pos;
uniform float specular_pow;

in vec3 v_norm;
in vec3 v_frag_pos;
out vec4 frag_color;

void main() {
    vec3 N = normalize(v_norm);
    vec3 L = normalize(light_dir);
    vec3 V = normalize(cam_pos - v_frag_pos);
    vec3 H = normalize(L + V);   // half-vector (Blinn-Phong)

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), specular_pow);

    vec3 color = ambient * obj_color
               + diff   * light_color * obj_color
               + spec   * light_color * 0.6;

    // Subtle atmospheric rim glow
    float rim = 1.0 - max(dot(N, V), 0.0);
    color += pow(rim, 4.0) * vec3(0.1, 0.3, 0.6) * 0.8;

    frag_color = vec4(color, 1.0);
}
"""

# ── Trail ─────────────────────────────────────────────────────────────
TRAIL_VERT = """
#version 330 core
uniform mat4 view;
uniform mat4 proj;

in vec3 in_pos;
in float in_age;    // 0.0 = oldest, 1.0 = newest

out float v_age;

void main() {
    v_age       = in_age;
    gl_Position = proj * view * vec4(in_pos, 1.0);
}
"""

TRAIL_FRAG = """
#version 330 core
uniform vec3 trail_color;

in float v_age;
out vec4 frag_color;

void main() {
    float alpha = pow(v_age, 1.6);  // non-linear fade: tail is very transparent
    frag_color = vec4(trail_color, alpha);
}
"""

# ── Flat colored geometry (grid, arrow) ───────────────────────────────
FLAT_VERT = """
#version 330 core
uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

in vec3 in_pos;

void main() {
    gl_Position = proj * view * model * vec4(in_pos, 1.0);
}
"""

FLAT_FRAG = """
#version 330 core
uniform vec4 color;
out vec4 frag_color;
void main() { frag_color = color; }
"""

# ── Stars (point sprites) ─────────────────────────────────────────────
STAR_VERT = """
#version 330 core
uniform mat4 view;
uniform mat4 proj;

in vec3 in_pos;
in float in_brightness;

out float v_brightness;

void main() {
    v_brightness = in_brightness;
    gl_PointSize = in_brightness * 2.5;
    gl_Position  = proj * view * vec4(in_pos, 1.0);
}
"""

STAR_FRAG = """
#version 330 core
in float v_brightness;
out vec4 frag_color;
void main() {
    // Circular soft point
    vec2  uv   = gl_PointCoord - 0.5;
    float dist = length(uv);
    if (dist > 0.5) discard;
    float alpha = (1.0 - dist * 2.0) * v_brightness;
    frag_color = vec4(1.0, 1.0, 1.0, alpha * 0.8);
}
"""

# ── HUD (2D overlay rendered onto a quad) ─────────────────────────────
HUD_VERT = """
#version 330 core
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

HUD_FRAG = """
#version 330 core
uniform sampler2D hud_tex;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    frag_color = texture(hud_tex, v_uv);
}
"""


# ── Geometry helpers ──────────────────────────────────────────────────
def _sphere_mesh(stacks: int = 32, sectors: int = 48, radius: float = 1.0):
    """
    Generate sphere vertices and triangle indices.
    Returns (verts float32 [N,6]: pos+normal, indices uint32 [M,3]).
    """
    verts   = []
    indices = []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks          # 0 → π
        sp, cp = math.sin(phi), math.cos(phi)
        for j in range(sectors + 1):
            theta = 2 * math.pi * j / sectors   # 0 → 2π
            st, ct = math.sin(theta), math.cos(theta)
            x, y, z = sp * ct, cp, sp * st      # unit sphere
            verts.extend([x * radius, y * radius, z * radius, x, y, z])

    for i in range(stacks):
        for j in range(sectors):
            a = i * (sectors + 1) + j
            b = a + sectors + 1
            indices.extend([a, b, a + 1, b, b + 1, a + 1])

    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.uint32)


def _grid_lines(size: float = 8.0, step: float = 1.0) -> np.ndarray:
    """XZ-plane grid lines."""
    verts = []
    x = -size
    while x <= size + 1e-6:
        verts.extend([x, 0, -size,  x, 0, size])
        verts.extend([-size, 0, x,  size, 0, x])
        x += step
    return np.array(verts, dtype=np.float32)


def _gen_stars(n: int = 1500, radius: float = 80.0) -> np.ndarray:
    """Random stars on a large sphere."""
    rng = np.random.default_rng(42)
    phi   = rng.uniform(0, math.pi, n)
    theta = rng.uniform(0, 2 * math.pi, n)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.cos(phi)
    z = radius * np.sin(phi) * np.sin(theta)
    bright = rng.uniform(0.3, 1.0, n).astype(np.float32)
    pos    = np.column_stack([x, y, z]).astype(np.float32)
    return pos, bright


def _identity() -> np.ndarray:
    return np.identity(4, dtype=np.float32)


def _translation(tx, ty, tz) -> np.ndarray:
    m = _identity()
    m[0, 3] = tx; m[1, 3] = ty; m[2, 3] = tz
    return m


def _scale_mat(s) -> np.ndarray:
    m = _identity()
    m[0, 0] = m[1, 1] = m[2, 2] = s
    return m


# ── Orbit type colors ─────────────────────────────────────────────────
TRAIL_COLORS = {
    "CIRCULAR":   np.array([0.35, 0.90, 0.60], dtype=np.float32),
    "ELLIPTICAL": np.array([0.95, 0.75, 0.20], dtype=np.float32),
    "ESCAPE":     np.array([0.95, 0.30, 0.25], dtype=np.float32),
}


# ── HUD Text (pygame → texture) ───────────────────────────────────────
try:
    import pygame
    _pygame_available = True
except ImportError:
    _pygame_available = False


def _make_hud_surface(physics, paused, dt_mult, width, height):
    """Render HUD info to a pygame surface, return RGBA bytes."""
    if not _pygame_available:
        return None

    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    sat = physics.satellite

    def col(name):
        return {
            "accent":   (80, 200, 255, 255),
            "good":     (80, 220, 130, 255),
            "warn":     (240, 190, 60, 255),
            "danger":   (240, 80,  80, 255),
            "gray":     (140, 155, 175, 255),
            "white":    (230, 240, 255, 255),
            "bg":       (10,  15,  28, 200),
        }[name]

    font = pygame.font.SysFont("Consolas", 15)
    font_b = pygame.font.SysFont("Consolas", 16, bold=True)

    otype   = sat.orbit_type()
    energy  = sat.total_energy()
    speed   = sat.speed()
    dist    = sat.distance()
    v_c     = sat.circular_speed()
    v_e     = sat.escape_speed()
    ke      = sat.kinetic_energy()
    pe      = sat.potential_energy()

    type_col = {"CIRCULAR": col("good"), "ELLIPTICAL": col("warn"), "ESCAPE": col("danger")}[otype]

    panel_w, panel_h = 290, 360
    px, py = width - panel_w - 12, 12

    # Background panel
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((8, 12, 25, 215))
    pygame.draw.rect(panel, (40, 55, 80, 255), panel.get_rect(), 1)
    surf.blit(panel, (px, py))

    def row(label, value, vc, y_off):
        lbl = font.render(label, True, col("gray"))
        val = font.render(value, True, vc)
        surf.blit(lbl, (px + 10, py + y_off))
        surf.blit(val, (px + panel_w - val.get_width() - 10, py + y_off))

    def sep(y_off):
        pygame.draw.line(surf, (35, 50, 70, 255),
                         (px + 8, py + y_off), (px + panel_w - 8, py + y_off))

    y = 14
    title = font_b.render("ORBITAL MECHANICS", True, col("accent"))
    surf.blit(title, (px + panel_w // 2 - title.get_width() // 2, py + y)); y += 28
    sep(y); y += 10

    otype_surf = font_b.render(otype, True, type_col)
    surf.blit(otype_surf, (px + panel_w // 2 - otype_surf.get_width() // 2, py + y)); y += 28
    sep(y); y += 10

    row("SPEED",      f"{speed:8.3f} u/s",  col("accent"),  y); y += 22
    row("v_circular", f"{v_c:8.3f} u/s",    col("gray"),    y); y += 22
    row("v_escape",   f"{v_e:8.3f} u/s",    col("gray"),    y); y += 22
    row("DISTANCE",   f"{dist:8.3f} u",      col("accent"),  y); y += 22
    sep(y); y += 10

    row("KINETIC E",  f"{ke:+8.3f}",         col("good"),    y); y += 22
    row("POTENT. E",  f"{pe:+8.3f}",         col("warn"),    y); y += 22
    ec = col("good") if energy < 0 else col("danger")
    row("TOTAL E",    f"{energy:+8.3f}",      ec,             y); y += 22
    sep(y); y += 10

    row("TIME SCALE", f"×{dt_mult:.1f}",      col("accent"),  y); y += 22

    st_txt = "PAUSED" if paused else ("ESCAPED" if sat.escaped else "ORBITING")
    st_col = col("warn") if paused else (col("danger") if sat.escaped else col("good"))
    row("STATUS",     st_txt,                 st_col,         y); y += 22

    # Energy bar at bottom of panel
    bar_y = py + panel_h - 28
    bar_x = px + 10
    bar_w = panel_w - 20
    bar_h = 12
    total = ke + abs(pe)
    ke_w  = int(bar_w * ke / total) if total > 0 else bar_w // 2
    pygame.draw.rect(surf, (80, 220, 130), (bar_x, bar_y, ke_w, bar_h))
    pygame.draw.rect(surf, (240, 190, 60), (bar_x + ke_w, bar_y, bar_w - ke_w, bar_h))
    pygame.draw.rect(surf, (60, 80, 110), (bar_x, bar_y, bar_w, bar_h), 1)
    ke_l = font.render("KE", True, (10, 15, 28))
    pe_l = font.render("|PE|", True, (10, 15, 28))
    surf.blit(ke_l, (bar_x + 2, bar_y))
    if bar_w - ke_w > 30:
        surf.blit(pe_l, (bar_x + ke_w + 2, bar_y))

    # Controls panel (bottom-left)
    ctrl = [
        ("Mouse drag", "Orbit camera"),
        ("Scroll",     "Zoom"),
        ("WASD",       "Pan view"),
        ("↑↓",         "Speed ±"),
        ("←→",         "Rotate velocity"),
        ("I / K",      "Tilt inclination"),
        ("C",          "Circular orbit"),
        ("E",          "Escape velocity"),
        ("1/2/3/4",    "Presets"),
        ("SPACE",      "Pause"),
        ("R",          "Reset"),
        ("F / S",      "Time ±"),
    ]
    cw, ch = 240, len(ctrl) * 20 + 12
    cp_x, cp_y = 12, height - ch - 12
    ctrl_bg = pygame.Surface((cw, ch), pygame.SRCALPHA)
    ctrl_bg.fill((8, 12, 25, 210))
    pygame.draw.rect(ctrl_bg, (40, 55, 80, 255), ctrl_bg.get_rect(), 1)
    surf.blit(ctrl_bg, (cp_x, cp_y))
    for i, (k, d) in enumerate(ctrl):
        ks = font.render(k, True, col("accent"))
        ds = font.render(d, True, col("gray"))
        surf.blit(ks, (cp_x + 8,  cp_y + 6 + i * 20))
        surf.blit(ds, (cp_x + 85, cp_y + 6 + i * 20))

    # Preset labels
    preset_txt = font.render("1=Circular  2=Elliptical  3=Escape  4=Inclined", True, col("gray"))
    surf.blit(preset_txt, (width // 2 - preset_txt.get_width() // 2, height - 20))

    if sat.escaped:
        msg = font_b.render("ESCAPE TRAJECTORY  —  Press R to reset", True, col("danger"))
        surf.blit(msg, (width // 2 - msg.get_width() // 2, height // 2 - 18))

    if paused:
        pm = font_b.render("[ PAUSED ]", True, col("warn"))
        surf.blit(pm, (width // 2 - pm.get_width() // 2, 14))

    return pygame.image.tostring(surf, "RGBA", True)


# ── Renderer ─────────────────────────────────────────────────────────
class Renderer:
    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx    = ctx
        self.width  = width
        self.height = height
        self._init_pygame(width, height)
        self._build_shaders(ctx)
        self._build_geometry(ctx)
        self._build_hud(ctx, width, height)
        self.update_projection(width, height)  # compute proj now hud_tex exists

    def _init_pygame(self, w, h):
        if _pygame_available:
            pygame.init()
            pygame.font.init()

    def _build_shaders(self, ctx):
        self.prog_sphere = ctx.program(vertex_shader=SPHERE_VERT, fragment_shader=SPHERE_FRAG)
        self.prog_trail  = ctx.program(vertex_shader=TRAIL_VERT,  fragment_shader=TRAIL_FRAG)
        self.prog_flat   = ctx.program(vertex_shader=FLAT_VERT,   fragment_shader=FLAT_FRAG)
        self.prog_star   = ctx.program(vertex_shader=STAR_VERT,   fragment_shader=STAR_FRAG)
        self.prog_hud    = ctx.program(vertex_shader=HUD_VERT,    fragment_shader=HUD_FRAG)

    def _build_geometry(self, ctx):
        # Earth sphere
        v, idx = _sphere_mesh(stacks=48, sectors=64, radius=0.25)
        self.earth_vbo = ctx.buffer(v.tobytes())
        self.earth_ibo = ctx.buffer(idx.tobytes())
        self.earth_vao = ctx.vertex_array(
            self.prog_sphere,
            [(self.earth_vbo, "3f 3f", "in_pos", "in_norm")],
            self.earth_ibo)

        # Satellite sphere (smaller)
        sv, sidx = _sphere_mesh(stacks=16, sectors=24, radius=0.07)
        self.sat_vbo = ctx.buffer(sv.tobytes())
        self.sat_ibo = ctx.buffer(sidx.tobytes())
        self.sat_vao = ctx.vertex_array(
            self.prog_sphere,
            [(self.sat_vbo, "3f 3f", "in_pos", "in_norm")],
            self.sat_ibo)

        # Grid
        gv = _grid_lines(size=8.0, step=1.0)
        self.grid_vbo = ctx.buffer(gv.tobytes())
        self.grid_vao = ctx.vertex_array(
            self.prog_flat,
            [(self.grid_vbo, "3f", "in_pos")])

        # Stars
        sp, sb = _gen_stars(n=1800)
        combined = np.column_stack([sp, sb.reshape(-1, 1)]).astype(np.float32)
        self.star_vbo = ctx.buffer(combined.tobytes())
        self.star_vao = ctx.vertex_array(
            self.prog_star,
            [(self.star_vbo, "3f 1f", "in_pos", "in_brightness")])
        self.n_stars = len(sp)

        # Trail (dynamic — rebuilt each frame)
        self.trail_buf  = ctx.buffer(reserve=8 * 4 * 4)  # pre-allocate
        self.trail_vao  = None  # built dynamically
        self._trail_buf_size = 0

        # Velocity arrow (2 points: tail + head)
        self.arrow_vbo = ctx.buffer(reserve=6 * 4)
        self.arrow_vao = ctx.vertex_array(
            self.prog_flat,
            [(self.arrow_vbo, "3f", "in_pos")])

        # Projection (computed once HUD is ready; set a placeholder here)
        self.proj = np.identity(4, dtype=np.float32)

    def _build_hud(self, ctx, w, h):
        """Fullscreen quad for HUD overlay."""
        quad = np.array([
            -1, -1,  0, 0,
             1, -1,  1, 0,
            -1,  1,  0, 1,
             1,  1,  1, 1,
        ], dtype=np.float32)
        self.hud_vbo = ctx.buffer(quad.tobytes())
        self.hud_vao = ctx.vertex_array(
            self.prog_hud,
            [(self.hud_vbo, "2f 2f", "in_pos", "in_uv")])
        # HUD texture (RGBA)
        self.hud_tex = ctx.texture((w, h), 4)
        self.hud_tex.filter = moderngl.LINEAR, moderngl.LINEAR
        self.hud_tex.swizzle = 'RGBA'

    def update_projection(self, w: int, h: int):
        from camera import _perspective
        self.width, self.height = w, h
        self.proj = _perspective(45.0, w / h if h > 0 else 1.0, 0.01, 200.0)
        # Rebuild HUD texture at new size (guard: only after _build_hud)
        if hasattr(self, 'hud_tex'):
            self.hud_tex.release()
            self.hud_tex = self.ctx.texture((w, h), 4)
            self.hud_tex.filter = moderngl.LINEAR, moderngl.LINEAR

    def _set_sphere_uniforms(self, prog, view, proj, model, color, cam_pos):
        prog["view"].write(view.T.astype(np.float32).tobytes())
        prog["proj"].write(proj.T.astype(np.float32).tobytes())
        prog["model"].write(model.T.astype(np.float32).tobytes())
        prog["light_dir"].value    = (0.7, 1.0, 0.5)
        prog["light_color"].value  = (1.0, 0.95, 0.85)
        prog["ambient"].value      = (0.08, 0.08, 0.12)
        prog["obj_color"].value    = tuple(color)
        prog["cam_pos"].value      = tuple(cam_pos)
        prog["specular_pow"].value = 64.0

    def render(self, ctx, physics, camera, view, proj, paused=False, dt_mult=1.0):
        sat = physics.satellite

        # ── Enable point sprites for stars ────────────────────────
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # ── Stars ─────────────────────────────────────────────────
        self.prog_star["view"].write(view.T.astype(np.float32).tobytes())
        self.prog_star["proj"].write(proj.T.astype(np.float32).tobytes())
        self.star_vao.render(moderngl.POINTS, vertices=self.n_stars)

        # ── Grid ──────────────────────────────────────────────────
        self.prog_flat["view"].write(view.T.astype(np.float32).tobytes())
        self.prog_flat["proj"].write(proj.T.astype(np.float32).tobytes())
        self.prog_flat["model"].write(_identity().T.tobytes())
        self.prog_flat["color"].value = (0.08, 0.12, 0.20, 0.6)
        self.grid_vao.render(moderngl.LINES)

        # ── Earth ─────────────────────────────────────────────────
        earth_color = np.array([0.22, 0.50, 0.88])
        self._set_sphere_uniforms(
            self.prog_sphere, view, proj, _identity(),
            earth_color, camera.position())
        self.earth_vao.render(moderngl.TRIANGLES)

        # ── Orbit trail ───────────────────────────────────────────
        trail = sat.trail
        if len(trail) >= 2:
            n = len(trail)
            pts   = np.array(trail, dtype=np.float32)       # [N, 3]
            ages  = np.linspace(0.0, 1.0, n, dtype=np.float32)  # 0=old, 1=new
            data  = np.column_stack([pts, ages])             # [N, 4]
            data_bytes = data.tobytes()

            # Resize buffer if needed
            if len(data_bytes) > self._trail_buf_size:
                self.trail_buf.release()
                self.trail_buf = ctx.buffer(reserve=len(data_bytes) * 2)
                self._trail_buf_size = len(data_bytes) * 2
                if self.trail_vao:
                    self.trail_vao.release()
                self.trail_vao = ctx.vertex_array(
                    self.prog_trail,
                    [(self.trail_buf, "3f 1f", "in_pos", "in_age")])

            self.trail_buf.write(data_bytes)

            otype = sat.orbit_type()
            tc    = TRAIL_COLORS[otype]
            self.prog_trail["view"].write(view.T.astype(np.float32).tobytes())
            self.prog_trail["proj"].write(proj.T.astype(np.float32).tobytes())
            self.prog_trail["trail_color"].value = tuple(tc)
            self.trail_vao.render(moderngl.LINE_STRIP, vertices=n)

        # ── Satellite ─────────────────────────────────────────────
        sat_pos = sat.pos.astype(np.float32)
        sat_model = _translation(*sat_pos)
        otype = sat.orbit_type()
        sat_color = TRAIL_COLORS[otype]
        self._set_sphere_uniforms(
            self.prog_sphere, view, proj, sat_model,
            sat_color, camera.position())
        self.prog_sphere["specular_pow"].value = 128.0
        self.sat_vao.render(moderngl.TRIANGLES)

        # ── Velocity arrow ────────────────────────────────────────
        arrow_tail = sat.pos.astype(np.float32)
        arrow_head = (sat.pos + sat.vel * 0.6).astype(np.float32)
        arrow_data = np.concatenate([arrow_tail, arrow_head]).astype(np.float32)
        self.arrow_vbo.write(arrow_data.tobytes())
        self.prog_flat["view"].write(view.T.astype(np.float32).tobytes())
        self.prog_flat["proj"].write(proj.T.astype(np.float32).tobytes())
        self.prog_flat["model"].write(_identity().T.tobytes())
        self.prog_flat["color"].value = (1.0, 1.0, 1.0, 0.9)
        self.arrow_vao.render(moderngl.LINES, vertices=2)

        # ── Angular momentum axis (orbital plane normal) ───────────
        L = sat.angular_momentum()
        Ln = np.linalg.norm(L)
        if Ln > 1e-6:
            Lhat = (L / Ln * 1.5).astype(np.float32)
            L_data = np.concatenate([
                np.zeros(3, dtype=np.float32), Lhat
            ])
            self.arrow_vbo.write(L_data.tobytes())
            self.prog_flat["color"].value = (0.4, 0.4, 1.0, 0.4)
            self.arrow_vao.render(moderngl.LINES, vertices=2)

        # ── HUD overlay ───────────────────────────────────────────
        if _pygame_available:
            rgba = _make_hud_surface(physics, paused, dt_mult, self.width, self.height)
            if rgba:
                self.hud_tex.write(rgba)
                ctx.disable(moderngl.DEPTH_TEST)
                self.hud_tex.use(0)
                self.prog_hud["hud_tex"].value = 0
                self.hud_vao.render(moderngl.TRIANGLE_STRIP)
                ctx.enable(moderngl.DEPTH_TEST)