"""
renderer.py — ModernGL Renderer  (v2: Textured Earth + Real Satellite Model)
"""

import numpy as np
import math
import moderngl
import os
import time

_HERE   = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(_HERE, "assets")

# ══════════════════════════════════════════════════════════════════════
# GLSL SHADERS
# ══════════════════════════════════════════════════════════════════════

EARTH_VERT = """
#version 330 core
uniform mat4 model, view, proj;
in vec3 in_pos; in vec3 in_norm; in vec2 in_uv;
out vec3 v_norm, v_frag_pos; out vec2 v_uv;
void main() {
    vec4 wp   = model * vec4(in_pos, 1.0);
    v_frag_pos = wp.xyz;
    v_norm     = mat3(transpose(inverse(model))) * in_norm;
    v_uv       = in_uv;
    gl_Position = proj * view * wp;
}
"""

EARTH_FRAG = """
#version 330 core
uniform sampler2D earth_tex, spec_tex, cloud_tex;
uniform vec3 light_dir, light_color, ambient, cam_pos;
uniform float time_sec;
in vec3 v_norm, v_frag_pos; in vec2 v_uv;
out vec4 frag_color;
void main() {
    vec3 N = normalize(v_norm);
    vec3 L = normalize(light_dir);
    vec3 V = normalize(cam_pos - v_frag_pos);
    vec3 H = normalize(L + V);

    vec2 cuv = vec2(fract(v_uv.x + time_sec * 0.0018), v_uv.y);
    vec3  ec  = texture(earth_tex, v_uv).rgb;
    float sm  = texture(spec_tex,  v_uv).r;
    vec4  cl  = texture(cloud_tex, cuv);

    vec3 surf = mix(ec, vec3(1.0), cl.a * 0.82);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 85.0) * sm;
    vec3 col = ambient * surf + diff * light_color * surf + spec * light_color * 0.65;

    // atmospheric rim
    float rim = 1.0 - max(dot(N, V), 0.0);
    col += pow(rim, 3.5) * vec3(0.1, 0.32, 0.78) * 0.88;

    // night side
    float night = smoothstep(-0.08, 0.14, dot(N, L));
    col = mix(col * 0.07, col, night);
    frag_color = vec4(col, 1.0);
}
"""

SAT_VERT = """
#version 330 core
uniform mat4 model, view, proj;
in vec3 in_pos; in vec3 in_norm;
out vec3 v_norm, v_frag_pos;
void main() {
    vec4 wp    = model * vec4(in_pos, 1.0);
    v_frag_pos = wp.xyz;
    v_norm     = mat3(transpose(inverse(model))) * in_norm;
    gl_Position = proj * view * wp;
}
"""

SAT_FRAG = """
#version 330 core
uniform vec3  light_dir, light_color, ambient_col, obj_color, cam_pos;
uniform float shininess, emissive;
in vec3 v_norm, v_frag_pos;
out vec4 frag_color;
void main() {
    vec3 N = normalize(v_norm);
    vec3 L = normalize(light_dir);
    vec3 V = normalize(cam_pos - v_frag_pos);
    vec3 H = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), shininess);
    vec3 col = ambient_col * obj_color
             + diff * light_color * obj_color
             + spec * light_color * 0.75
             + obj_color * emissive;
    frag_color = vec4(col, 1.0);
}
"""

TRAIL_VERT = """
#version 330 core
uniform mat4 view, proj;
in vec3 in_pos; in float in_age;
out float v_age;
void main() { v_age = in_age; gl_Position = proj * view * vec4(in_pos, 1.0); }
"""

TRAIL_FRAG = """
#version 330 core
uniform vec3 trail_color;
in float v_age; out vec4 frag_color;
void main() { frag_color = vec4(trail_color, pow(v_age, 1.5)); }
"""

FLAT_VERT = """
#version 330 core
uniform mat4 view, proj, model;
in vec3 in_pos;
void main() { gl_Position = proj * view * model * vec4(in_pos, 1.0); }
"""
FLAT_FRAG = """
#version 330 core
uniform vec4 color; out vec4 frag_color;
void main() { frag_color = color; }
"""

STAR_VERT = """
#version 330 core
uniform mat4 view, proj;
in vec3 in_pos; in float in_brightness;
out float v_brightness;
void main() {
    v_brightness = in_brightness; gl_PointSize = in_brightness * 2.8;
    gl_Position = proj * view * vec4(in_pos, 1.0);
}
"""
STAR_FRAG = """
#version 330 core
in float v_brightness; out vec4 frag_color;
void main() {
    vec2 uv = gl_PointCoord - 0.5; float d = length(uv);
    if (d > 0.5) discard;
    frag_color = vec4(1.0, 1.0, 1.0, (1.0 - d*2.0) * v_brightness * 0.85);
}
"""

HUD_VERT = """
#version 330 core
in vec2 in_pos; in vec2 in_uv; out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""
HUD_FRAG = """
#version 330 core
uniform sampler2D hud_tex; in vec2 v_uv; out vec4 frag_color;
void main() { frag_color = texture(hud_tex, v_uv); }
"""

# ══════════════════════════════════════════════════════════════════════
# GEOMETRY BUILDERS
# ══════════════════════════════════════════════════════════════════════

def _sphere_uv(stacks=64, sectors=96, radius=1.0):
    """UV-mapped sphere: vertex = pos(3) + norm(3) + uv(2)."""
    verts, indices = [], []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        sp, cp = math.sin(phi), math.cos(phi)
        vt = i / stacks
        for j in range(sectors + 1):
            theta = 2 * math.pi * j / sectors
            st, ct = math.sin(theta), math.cos(theta)
            x, y, z = sp * ct, cp, sp * st
            verts.extend([x*radius, y*radius, z*radius, x, y, z, j/sectors, vt])
    for i in range(stacks):
        for j in range(sectors):
            a = i*(sectors+1)+j; b = a+sectors+1
            indices.extend([a, b, a+1, b, b+1, a+1])
    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.uint32)


def _box(w, h, d):
    """Axis-aligned box centred at origin → pos(3)+norm(3) per vertex."""
    hw, hh, hd = w/2, h/2, d/2
    faces = [
        ([ hw,-hh,-hd, hw,-hh, hd, hw, hh, hd, hw, hh,-hd], [1,0,0]),
        ([-hw,-hh, hd,-hw,-hh,-hd,-hw, hh,-hd,-hw, hh, hd], [-1,0,0]),
        ([-hw, hh,-hd, hw, hh,-hd, hw, hh, hd,-hw, hh, hd], [0,1,0]),
        ([-hw,-hh, hd, hw,-hh, hd, hw,-hh,-hd,-hw,-hh,-hd], [0,-1,0]),
        ([-hw,-hh, hd,-hw, hh, hd, hw, hh, hd, hw,-hh, hd], [0,0,1]),
        ([ hw,-hh,-hd, hw, hh,-hd,-hw, hh,-hd,-hw,-hh,-hd], [0,0,-1]),
    ]
    verts, idxs, base = [], [], 0
    for pts, nm in faces:
        for k in range(4):
            verts.extend(pts[k*3:k*3+3] + nm)
        idxs.extend([base, base+1, base+2, base, base+2, base+3])
        base += 4
    return np.array(verts, dtype=np.float32), np.array(idxs, dtype=np.uint32)


def _cylinder(radius=1.0, height=1.0, sectors=24):
    """Closed cylinder centred at origin along Y axis."""
    verts, idxs, base = [], [], 0
    for i in range(sectors):
        a0 = 2*math.pi*i/sectors; a1 = 2*math.pi*(i+1)/sectors
        x0,z0 = math.cos(a0)*radius, math.sin(a0)*radius
        x1,z1 = math.cos(a1)*radius, math.sin(a1)*radius
        yb, yt = -height/2, height/2
        nx = (x0+x1)*0.5/radius; nz = (z0+z1)*0.5/radius
        for (x,z,y) in [(x0,z0,yb),(x1,z1,yb),(x1,z1,yt),(x0,z0,yt)]:
            verts.extend([x,y,z,nx,0,nz])
        idxs.extend([base, base+1, base+2, base, base+2, base+3])
        base += 4
    # top cap
    ct = len(verts)//6; verts.extend([0,height/2,0, 0,1,0]); cs = ct+1
    for i in range(sectors):
        a=2*math.pi*i/sectors; verts.extend([math.cos(a)*radius,height/2,math.sin(a)*radius,0,1,0])
    for i in range(sectors): idxs.extend([ct, cs+i, cs+(i+1)%sectors])
    # bottom cap
    cb = len(verts)//6; verts.extend([0,-height/2,0, 0,-1,0]); bs = cb+1
    for i in range(sectors):
        a=2*math.pi*i/sectors; verts.extend([math.cos(a)*radius,-height/2,math.sin(a)*radius,0,-1,0])
    for i in range(sectors): idxs.extend([cb, bs+(i+1)%sectors, bs+i])
    return np.array(verts, dtype=np.float32), np.array(idxs, dtype=np.uint32)


def _cone(base_r=1.0, height=1.0, sectors=16):
    verts, idxs = [], []; tip = 0
    verts.extend([0, height, 0, 0, 1, 0])
    bs = 1
    for i in range(sectors):
        a=2*math.pi*i/sectors; x,z=math.cos(a)*base_r,math.sin(a)*base_r
        sl=1/math.sqrt(1+(base_r/height)**2)
        verts.extend([x,0,z, x/base_r*sl, height/math.sqrt(height**2+base_r**2), z/base_r*sl])
    for i in range(sectors): idxs.extend([tip, bs+(i+1)%sectors, bs+i])
    cc = len(verts)//6; verts.extend([0,0,0, 0,-1,0]); cs2=cc+1
    for i in range(sectors):
        a=2*math.pi*i/sectors; verts.extend([math.cos(a)*base_r,0,math.sin(a)*base_r,0,-1,0])
    for i in range(sectors): idxs.extend([cc, cs2+i, cs2+(i+1)%sectors])
    return np.array(verts, dtype=np.float32), np.array(idxs, dtype=np.uint32)


def _merge(meshes):
    all_v, all_i, offset = [], [], 0
    for v, i in meshes:
        nv = len(v)//6
        all_v.append(v); all_i.append(i + offset); offset += nv
    return np.concatenate(all_v), np.concatenate(all_i)


def _txform(verts, mat):
    """Apply 4×4 matrix to pos and rotation-part to normals."""
    s = 6; out = verts.copy().reshape(-1, s)
    rot = mat[:3,:3]
    pos4 = np.column_stack([out[:,:3], np.ones(len(out))])
    out[:,:3] = (mat @ pos4.T).T[:,:3]
    out[:,3:6] = (rot @ out[:,3:6].T).T
    return out.reshape(-1)


def _Tx(tx,ty,tz):
    m=np.eye(4,dtype=np.float64); m[0,3]=tx; m[1,3]=ty; m[2,3]=tz; return m
def _Rx(deg):
    a=math.radians(deg); c,s=math.cos(a),math.sin(a)
    m=np.eye(4,dtype=np.float64); m[1,1]=c; m[1,2]=-s; m[2,1]=s; m[2,2]=c; return m
def _Ry(deg):
    a=math.radians(deg); c,s=math.cos(a),math.sin(a)
    m=np.eye(4,dtype=np.float64); m[0,0]=c; m[0,2]=s; m[2,0]=-s; m[2,2]=c; return m


def _build_satellite_mesh():
    """
    ISS-inspired satellite geometry (local units, scaled to ~0.06 sim units at render).

    Structure:
        ─── Solar array ──┐
                          [=====Truss=====]
        ─── Solar array ──┘    │ Habitat │ Dock ─ Antenna
    """
    parts = []

    # Main truss (long horizontal box along X)
    parts.append(_box(2.6, 0.09, 0.09))

    # Central pressurised module (cylinder, axis along Z)
    hv, hi = _cylinder(radius=0.165, height=0.52, sectors=22)
    parts.append((_txform(hv, _Rx(90)), hi))

    # Second smaller module (forward)
    mv, mi = _cylinder(radius=0.12, height=0.30, sectors=16)
    mv = _txform(mv, _Rx(90))
    mv = _txform(mv, _Tx(0, 0, 0.42))
    parts.append((mv, mi))

    # Docking port (small flat cylinder at nose)
    dv, di = _cylinder(radius=0.08, height=0.12, sectors=12)
    dv = _txform(dv, _Rx(90))
    dv = _txform(dv, _Tx(0, 0, 0.62))
    parts.append((dv, di))

    # Solar panels – 4 large thin quads (+X and -X sides of truss, each forking ±Y)
    for tx in [+1.1, -1.1]:
        for sy in [+1, -1]:
            pv, pi = _box(0.07, 0.92, 0.48)
            pv = _txform(pv, _Tx(tx, sy*0.52, 0.0))
            parts.append((pv, pi))

    # Radiator panels (thinner, along Z from mid-truss)
    for tz in [+0.35, -0.35]:
        rv, ri = _box(0.85, 0.06, 0.05)
        rv = _txform(rv, _Tx(0, 0.12, tz))
        parts.append((rv, ri))

    # Antenna dish (cone on top of habitat)
    av, ai = _cone(base_r=0.11, height=0.07, sectors=14)
    av = _txform(av, _Rx(180))
    av = _txform(av, _Tx(0, 0.245, 0))
    parts.append((av, ai))

    # Antenna mast
    sv, si = _cylinder(radius=0.013, height=0.18, sectors=6)
    sv = _txform(sv, _Tx(0, 0.15, 0))
    parts.append((sv, si))

    return _merge(parts)


# ── Colors ────────────────────────────────────────────────────────────
_WHITE_SILVER = np.array([0.86, 0.89, 0.93], dtype=np.float32)
_SOLAR_BLUE   = np.array([0.07, 0.18, 0.52], dtype=np.float32)
_GOLD_FOIL    = np.array([0.82, 0.66, 0.14], dtype=np.float32)

TRAIL_COLORS = {
    "CIRCULAR":   np.array([0.28, 0.85, 0.52], dtype=np.float32),
    "ELLIPTICAL": np.array([0.95, 0.70, 0.16], dtype=np.float32),
    "ESCAPE":     np.array([0.95, 0.26, 0.20], dtype=np.float32),
}


# ── Texture loader ────────────────────────────────────────────────────
def _load_tex(ctx, path, comps=3):
    from PIL import Image
    mode = "RGB" if comps == 3 else "RGBA"
    img  = Image.open(path).convert(mode).transpose(Image.FLIP_TOP_BOTTOM)
    tex  = ctx.texture(img.size, comps, img.tobytes())
    tex.filter     = moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR
    tex.anisotropy = 8.0
    tex.build_mipmaps()
    return tex


# ── Misc ──────────────────────────────────────────────────────────────
def _identity():
    return np.eye(4, dtype=np.float32)

def _mat_trans(tx, ty, tz):
    """4x4 translation matrix (float32)."""
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx; m[1, 3] = ty; m[2, 3] = tz
    return m

def _grid_lines(size=8.0, step=1.0):
    v=[]; x=-size
    while x<=size+1e-6:
        v.extend([x,0,-size, x,0,size, -size,0,x, size,0,x]); x+=step
    return np.array(v, dtype=np.float32)

def _gen_stars(n=2000, radius=80.0):
    rng=np.random.default_rng(42)
    phi=rng.uniform(0,math.pi,n); th=rng.uniform(0,2*math.pi,n)
    pos=np.column_stack([radius*np.sin(phi)*np.cos(th),
                         radius*np.cos(phi),
                         radius*np.sin(phi)*np.sin(th)]).astype(np.float32)
    return pos, rng.uniform(0.25,1.0,n).astype(np.float32)


# ── HUD ───────────────────────────────────────────────────────────────
try:
    import pygame as _pg
    _pg_ok = True
except ImportError:
    _pg_ok = False

def _make_hud(physics, paused, dt_mult, W, H):
    if not _pg_ok: return None
    surf = _pg.Surface((W,H), _pg.SRCALPHA); surf.fill((0,0,0,0))
    sat  = physics.satellite
    def c(n): return {"ac":(80,200,255,255),"go":(75,215,125,255),
                      "wn":(240,185,55,255),"dn":(240,75,75,255),
                      "gr":(135,150,170,255),"bg":(8,12,26,215)}[n]
    f  = _pg.font.SysFont("Consolas",15)
    fb = _pg.font.SysFont("Consolas",16,bold=True)
    ot = sat.elements.orbit_type
    bodies = physics.bodies
    e=sat.total_energy(bodies); ke=sat.kinetic_energy(); pe=sat.potential_energy(bodies)
    sp=sat.speed(); r=sat.distance_to_origin(); vc=sat.circular_speed_at(r); ve=sat.escape_speed_at(r)
    di=r
    tc={"CIRCULAR":c("go"),"ELLIPTICAL":c("wn"),"ESCAPE":c("dn"),"UNKNOWN":c("gr")}.get(ot, c("gr"))
    pw,ph=292,545; px,py=W-pw-12,12
    pan=_pg.Surface((pw,ph),_pg.SRCALPHA); pan.fill(c("bg"))
    _pg.draw.rect(pan,(38,55,82,255),pan.get_rect(),1); surf.blit(pan,(px,py))
    def row(lb,vl,vc_,yo):
        l=f.render(lb,True,c("gr")); v=f.render(vl,True,vc_)
        surf.blit(l,(px+10,py+yo)); surf.blit(v,(px+pw-v.get_width()-10,py+yo))
    def sep(yo): _pg.draw.line(surf,(35,50,72,255),(px+8,py+yo),(px+pw-8,py+yo))
    y=14
    tl=fb.render("ORBITAL MECHANICS",True,c("ac"))
    surf.blit(tl,(px+pw//2-tl.get_width()//2,py+y)); y+=28; sep(y); y+=10
    otl=fb.render(ot,True,tc)
    surf.blit(otl,(px+pw//2-otl.get_width()//2,py+y)); y+=28; sep(y); y+=10
    row("SPEED",      f"{sp:8.3f} u/s", c("ac"), y); y+=22
    row("v_circular", f"{vc:8.3f} u/s", c("gr"), y); y+=22
    row("v_escape",   f"{ve:8.3f} u/s", c("gr"), y); y+=22
    row("DISTANCE",   f"{di:8.3f} u",   c("ac"), y); y+=22
    sep(y); y+=10
    row("KINETIC E",  f"{ke:+8.3f}",    c("go"), y); y+=22
    row("POTENT. E",  f"{pe:+8.3f}",    c("wn"), y); y+=22
    ec_=c("go") if e<0 else c("dn")
    row("TOTAL E",    f"{e:+8.3f}",     ec_,     y); y+=22
    row("E DRIFT",    f"{sat.energy_drift*100:.4f}%", c("gr"), y); y+=22
    sep(y); y+=10
    # Orbital elements
    oe = sat.elements
    row("SEMI-MAJOR a", f"{oe.semi_major_axis:7.3f}",   c("ac"), y); y+=22
    row("ECCENTR.  e",  f"{oe.eccentricity:7.4f}",      c("ac"), y); y+=22
    row("INCLINAT. i",  f"{oe.inclination_deg:7.2f}°",  c("ac"), y); y+=22
    row("PERIAPSIS r_p",f"{oe.periapsis_r:7.3f}",       c("gr"), y); y+=22
    apo_txt = f"{oe.apoapsis_r:7.3f}" if oe.apoapsis_r < 1e9 else "   ∞   "
    row("APOAPSIS r_a", apo_txt,                         c("gr"), y); y+=22
    per_txt = f"{oe.period:7.2f} s" if oe.period < 1e9 else "   ∞   "
    row("PERIOD   T",   per_txt,                         c("gr"), y); y+=22
    row("TRUE ANOM ν",  f"{oe.true_anomaly_deg:7.2f}°", c("gr"), y); y+=22
    sep(y); y+=10
    row("TIME x",     f"{dt_mult:.1f}", c("ac"), y); y+=22
    st="PAUSED" if paused else ("ESCAPED" if sat.escaped else "ORBITING")
    if sat.thruster.firing: st = "THRUSTING"
    sc=c("wn") if paused else (c("dn") if sat.escaped else (c("ac") if sat.thruster.firing else c("go")))
    row("STATUS",st,sc,y)
    # energy bar
    by_=py+ph-30; bx_=px+10; bw_=pw-20; bh_=13
    tot=ke+abs(pe); kew=int(bw_*ke/tot) if tot>0 else bw_//2
    _pg.draw.rect(surf,(75,215,125),(bx_,by_,kew,bh_))
    _pg.draw.rect(surf,(240,185,55),(bx_+kew,by_,bw_-kew,bh_))
    _pg.draw.rect(surf,(50,72,105),(bx_,by_,bw_,bh_),1)
    surf.blit(f.render("KE",True,(8,12,26)),(bx_+3,by_+1))
    if bw_-kew>35: surf.blit(f.render("|PE|",True,(8,12,26)),(bx_+kew+3,by_+1))
    # controls panel
    ctrl=[("Mouse drag","Orbit camera"),("Scroll","Zoom"),
          ("WASD","Pan view"),("↑ / ↓","Speed ±"),("← / →","Rotate vel"),
          ("I / K","Inclination ±"),("C","Circular snap"),("E","Escape snap"),
          ("Z / X","Thrust pro/retro"),("Q","Impulse prograde"),
          ("H","Hohmann to 4.0u"),
          ("1-7","Presets"),("SPACE","Pause"),("R","Reset"),("F/S","Time ±")]
    cw,ch2=242,len(ctrl)*20+12; cpx,cpy=12,H-ch2-12
    cb=_pg.Surface((cw,ch2),_pg.SRCALPHA); cb.fill(c("bg"))
    _pg.draw.rect(cb,(38,55,82,255),cb.get_rect(),1); surf.blit(cb,(cpx,cpy))
    for i,(k,d) in enumerate(ctrl):
        surf.blit(f.render(k,True,c("ac")),(cpx+8, cpy+6+i*20))
        surf.blit(f.render(d,True,c("gr")),(cpx+95,cpy+6+i*20))
    pl=f.render("1-7=Presets  Z/X=Thrust  H=Hohmann  P=Toggle Perturb",True,c("gr"))
    surf.blit(pl,(W//2-pl.get_width()//2,H-20))

    # ── Perturbation panel (bottom-right) ─────────────────────────
    pdata = sat.perturbations.summary()
    cfg   = sat.perturbations.cfg
    pw2,ph2 = 230, 202
    ppx,ppy = W-pw2-12, H-ph2-28
    ppan=_pg.Surface((pw2,ph2),_pg.SRCALPHA); ppan.fill(c("bg"))
    _pg.draw.rect(ppan,(38,55,82,255),ppan.get_rect(),1); surf.blit(ppan,(ppx,ppy))
    ptitle=fb.render("PERTURBATIONS",True,c("ac"))
    surf.blit(ptitle,(ppx+pw2//2-ptitle.get_width()//2,ppy+6))
    def pcol(enabled): return c("go") if enabled else c("gr")
    def prow(lbl,val,en,yo):
        tick = "●" if en else "○"
        ls=f.render(f"{tick} {lbl}",True,pcol(en))
        vs=f.render(val,True,pcol(en))
        surf.blit(ls,(ppx+8,ppy+yo)); surf.blit(vs,(ppx+pw2-vs.get_width()-8,ppy+yo))
    py2=28
    prow("J2 Oblate",  f"{pdata['j2']:.2e}",  cfg.enable_j2,   py2); py2+=20
    prow("J4 Zonal",   f"{pdata['j4']:.2e}",  cfg.enable_j4,   py2); py2+=20
    prow("Atm. Drag",  f"{pdata['drag']:.2e}", cfg.enable_drag, py2); py2+=20
    prow("Sol. Rad.",  f"{pdata['srp']:.2e}",  cfg.enable_srp,  py2); py2+=20
    prow("Solar Grav.",f"{pdata['solar']:.2e}",cfg.enable_solar_gravity, py2); py2+=20
    _pg.draw.line(surf,(35,50,72,255),(ppx+8,ppy+py2),(ppx+pw2-8,ppy+py2)); py2+=8
    tdv=f.render(f"ΔV drag: {pdata['drag_dv_total']:.4f}",True,c("wn"))
    surf.blit(tdv,(ppx+8,ppy+py2)); py2+=20
    tot=f.render(f"Total:   {pdata['total']:.2e}",True,c("ac"))
    surf.blit(tot,(ppx+8,ppy+py2))
    # Toggle hint
    hint=f.render("P=All  J/D/R/G=J2/Drag/SRP/Grav",True,c("gr"))
    surf.blit(hint,(ppx+pw2//2-hint.get_width()//2,ppy+ph2-16))
    if sat.escaped:
        em=fb.render("ESCAPE TRAJECTORY  — Press R",True,c("dn"))
        surf.blit(em,(W//2-em.get_width()//2,H//2-18))
    if paused:
        pm=fb.render("[ PAUSED ]",True,c("wn"))
        surf.blit(pm,(W//2-pm.get_width()//2,14))
    return _pg.image.tostring(surf,"RGBA",True)


# ══════════════════════════════════════════════════════════════════════
# RENDERER
# ══════════════════════════════════════════════════════════════════════

class Renderer:
    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx    = ctx
        self.width  = width
        self.height = height
        self._start = time.perf_counter()
        self._tbsz  = 0

        if _pg_ok:
            _pg.init(); _pg.font.init()

        self._progs(ctx)
        self._earth(ctx)
        self._satellite(ctx)
        self._moon(ctx)
        self._scene(ctx)
        self._hud_build(ctx, width, height)
        self.update_projection(width, height)

        # trail buffers (dynamic)
        self.trail_buf = ctx.buffer(reserve=4096)
        self.trail_vao = None

    # ── Shaders ───────────────────────────────────────────────────
    def _progs(self, ctx):
        self.p_earth = ctx.program(vertex_shader=EARTH_VERT, fragment_shader=EARTH_FRAG)
        self.p_sat   = ctx.program(vertex_shader=SAT_VERT,   fragment_shader=SAT_FRAG)
        self.p_trail = ctx.program(vertex_shader=TRAIL_VERT, fragment_shader=TRAIL_FRAG)
        self.p_flat  = ctx.program(vertex_shader=FLAT_VERT,  fragment_shader=FLAT_FRAG)
        self.p_star  = ctx.program(vertex_shader=STAR_VERT,  fragment_shader=STAR_FRAG)
        self.p_hud   = ctx.program(vertex_shader=HUD_VERT,   fragment_shader=HUD_FRAG)

    # ── Earth geometry + textures ─────────────────────────────────
    def _earth(self, ctx):
        v, idx = _sphere_uv(stacks=80, sectors=120, radius=0.25)
        vbo = ctx.buffer(v.tobytes())
        ibo = ctx.buffer(idx.tobytes())
        self.earth_vao = ctx.vertex_array(self.p_earth,
                           [(vbo, "3f 3f 2f", "in_pos", "in_norm", "in_uv")], ibo)
        self.t_earth  = _load_tex(ctx, os.path.join(_ASSETS,"earth.jpg"),   3)
        self.t_clouds = _load_tex(ctx, os.path.join(_ASSETS,"clouds.png"),  3)
        self.t_spec   = _load_tex(ctx, os.path.join(_ASSETS,"specular.jpg"),3)

    # ── Satellite geometry ────────────────────────────────────────
    def _satellite(self, ctx):
        v, idx = _build_satellite_mesh()
        vbo = ctx.buffer(v.tobytes())
        ibo = ctx.buffer(idx.tobytes())
        self.sat_vao = ctx.vertex_array(self.p_sat,
                         [(vbo, "3f 3f", "in_pos", "in_norm")], ibo)

    # ── Moon sphere ───────────────────────────────────────────────
    def _moon(self, ctx):
        from constants import MOON_RADIUS
        v, idx = _sphere_uv(stacks=32, sectors=48, radius=MOON_RADIUS)
        vbo = ctx.buffer(v.tobytes())
        ibo = ctx.buffer(idx.tobytes())
        # Moon reuses earth shader but with a grey colour uniform override
        # We'll use p_sat (no texture) for simplicity
        v2, idx2 = _sphere_uv(stacks=32, sectors=48, radius=MOON_RADIUS)
        # strip UV from data → pos+norm only (6 floats per vertex)
        verts = v2.reshape(-1, 8)[:, :6].astype(np.float32).flatten()
        mvbo = ctx.buffer(verts.tobytes())
        mibo = ctx.buffer(idx2.tobytes())
        self.moon_vao = ctx.vertex_array(self.p_sat,
                          [(mvbo, "3f 3f", "in_pos", "in_norm")], mibo)

    # ── Stars + grid + arrow ──────────────────────────────────────
    def _scene(self, ctx):
        gv = _grid_lines()
        gvbo = ctx.buffer(gv.tobytes())
        self.grid_vao = ctx.vertex_array(self.p_flat, [(gvbo,"3f","in_pos")])

        sp, sb = _gen_stars(2000)
        data = np.column_stack([sp, sb.reshape(-1,1)]).astype(np.float32)
        svbo = ctx.buffer(data.tobytes())
        self.star_vao = ctx.vertex_array(self.p_star,
                          [(svbo,"3f 1f","in_pos","in_brightness")])
        self.n_stars = len(sp)

        self.arrow_vbo = ctx.buffer(reserve=6*4)
        self.arrow_vao = ctx.vertex_array(self.p_flat, [(self.arrow_vbo,"3f","in_pos")])

    # ── HUD quad + texture ────────────────────────────────────────
    def _hud_build(self, ctx, w, h):
        quad = np.array([-1,-1,0,0, 1,-1,1,0, -1,1,0,1, 1,1,1,1], dtype=np.float32)
        hv   = ctx.buffer(quad.tobytes())
        self.hud_vao = ctx.vertex_array(self.p_hud, [(hv,"2f 2f","in_pos","in_uv")])
        self.hud_tex = ctx.texture((w,h), 4)
        self.hud_tex.filter = moderngl.LINEAR, moderngl.LINEAR

    def update_projection(self, w, h):
        from camera import _perspective
        self.width, self.height = w, h
        self.proj = _perspective(45.0, w/h if h>0 else 1.0, 0.01, 200.0)
        if hasattr(self, "hud_tex"):
            self.hud_tex.release()
            self.hud_tex = self.ctx.texture((w,h), 4)
            self.hud_tex.filter = moderngl.LINEAR, moderngl.LINEAR

    # ── Satellite orientation matrix ──────────────────────────────
    def _sat_matrix(self, sat):
        """
        Builds a 4×4 model matrix that orients the satellite so:
          - Its X truss axis is along the orbital velocity (forward)
          - Its Y axis tracks the orbit normal (angular momentum)
          - Positioned at sat.pos, scaled to 0.06 sim units
        """
        pos = sat.pos.astype(np.float64)
        vel = sat.vel.astype(np.float64)
        vmag = np.linalg.norm(vel)
        fwd  = vel/vmag if vmag > 1e-8 else np.array([1.,0.,0.])

        L  = np.cross(pos, vel); Ln = np.linalg.norm(L)
        up = L/Ln if Ln > 1e-8 else np.array([0.,1.,0.])

        right = np.cross(fwd, up); rn = np.linalg.norm(right)
        right = right/rn if rn > 1e-8 else np.array([0.,0.,1.])
        up    = np.cross(right, fwd)

        sc = 0.06
        m  = np.eye(4, dtype=np.float32)
        m[0,:3] = (right * sc).astype(np.float32)
        m[1,:3] = (up    * sc).astype(np.float32)
        m[2,:3] = (-fwd  * sc).astype(np.float32)
        m[0,3]  = float(pos[0])
        m[1,3]  = float(pos[1])
        m[2,3]  = float(pos[2])
        return m

    # ── Draw satellite with per-part coloring ─────────────────────
    def _draw_sat(self, model, cam_p):
        p = self.p_sat
        p["model"].write(model.T.tobytes())
        p["light_dir"].value    = (0.8, 1.0, 0.5)
        p["light_color"].value  = (1.0, 0.96, 0.88)
        p["ambient_col"].value  = (0.14, 0.14, 0.18)
        p["cam_pos"].value      = tuple(cam_p)
        # Main body pass (silver)
        p["obj_color"].value    = tuple(_WHITE_SILVER)
        p["shininess"].value    = 110.0
        p["emissive"].value     = 0.0
        self.sat_vao.render(moderngl.TRIANGLES)

    # ── Main render ───────────────────────────────────────────────
    def render(self, ctx, physics, camera, view, proj, paused=False, dt_mult=1.0):
        sat   = physics.satellite
        t_sec = float(time.perf_counter() - self._start)
        camp  = camera.position().astype(np.float32)

        v32 = view.T.astype(np.float32).tobytes()
        p32 = proj.T.astype(np.float32).tobytes()

        ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # Stars
        self.p_star["view"].write(v32)
        self.p_star["proj"].write(p32)
        self.star_vao.render(moderngl.POINTS, vertices=self.n_stars)

        # Grid
        self.p_flat["view"].write(v32)
        self.p_flat["proj"].write(p32)
        self.p_flat["model"].write(_identity().T.tobytes())
        self.p_flat["color"].value = (0.07, 0.11, 0.20, 0.50)
        self.grid_vao.render(moderngl.LINES)

        # Earth
        self.p_earth["view"].write(v32)
        self.p_earth["proj"].write(p32)
        self.p_earth["model"].write(_identity().T.tobytes())
        self.p_earth["light_dir"].value   = (0.8, 1.0, 0.5)
        self.p_earth["light_color"].value = (1.0, 0.96, 0.88)
        self.p_earth["ambient"].value     = (0.05, 0.06, 0.10)
        self.p_earth["cam_pos"].value     = tuple(camp)
        self.p_earth["time_sec"].value    = t_sec
        self.t_earth.use(0);  self.p_earth["earth_tex"].value = 0
        self.t_spec.use(1);   self.p_earth["spec_tex"].value  = 1
        self.t_clouds.use(2); self.p_earth["cloud_tex"].value = 2
        self.earth_vao.render(moderngl.TRIANGLES)

        # Trail
        trail = sat.trail
        if len(trail) >= 2:
            n    = len(trail)
            pts  = np.array(trail, dtype=np.float32)
            ages = np.linspace(0.0, 1.0, n, dtype=np.float32)
            data = np.column_stack([pts, ages]).tobytes()
            if len(data) > self._tbsz:
                self.trail_buf.release()
                self.trail_buf = ctx.buffer(reserve=max(len(data)*2, 4096))
                self._tbsz = max(len(data)*2, 4096)
                if self.trail_vao: self.trail_vao.release()
                self.trail_vao = ctx.vertex_array(self.p_trail,
                    [(self.trail_buf,"3f 1f","in_pos","in_age")])
            self.trail_buf.write(data)
            tc = TRAIL_COLORS.get(sat.elements.orbit_type, TRAIL_COLORS["ELLIPTICAL"])
            self.p_trail["view"].write(v32)
            self.p_trail["proj"].write(p32)
            self.p_trail["trail_color"].value = tuple(tc)
            self.trail_vao.render(moderngl.LINE_STRIP, vertices=n)

        # Satellite model
        self.p_sat["view"].write(v32)
        self.p_sat["proj"].write(p32)
        self._draw_sat(self._sat_matrix(sat), camp)

        # Velocity arrow
        tail = sat.pos.astype(np.float32)
        head = (sat.pos + sat.vel * 0.5).astype(np.float32)
        self.arrow_vbo.write(np.concatenate([tail, head]).astype(np.float32).tobytes())
        self.p_flat["model"].write(_identity().T.tobytes())
        self.p_flat["color"].value = (1.0, 1.0, 0.7, 0.8)
        self.arrow_vao.render(moderngl.LINES, vertices=2)

        # Angular momentum axis
        L = sat.angular_momentum(); Ln = np.linalg.norm(L)
        if Ln > 1e-6:
            Lh = (L/Ln*1.4).astype(np.float32)
            self.arrow_vbo.write(np.concatenate([np.zeros(3,np.float32),Lh]).tobytes())
            self.p_flat["color"].value = (0.35, 0.42, 1.0, 0.38)
            self.arrow_vao.render(moderngl.LINES, vertices=2)

        # ── Moon ──────────────────────────────────────────────────────
        moon_pos = physics.moon_pos().astype(np.float32)
        moon_mat = _mat_trans(*moon_pos)
        moon_color = np.array([0.68, 0.68, 0.65], dtype=np.float32)
        self.p_sat["model"].write(moon_mat.T.tobytes())
        self.p_sat["light_dir"].value    = (0.8, 1.0, 0.5)
        self.p_sat["light_color"].value  = (1.0, 0.96, 0.88)
        self.p_sat["ambient_col"].value  = (0.18, 0.18, 0.20)
        self.p_sat["obj_color"].value    = tuple(moon_color)
        self.p_sat["cam_pos"].value      = tuple(camp)
        self.p_sat["shininess"].value    = 12.0
        self.p_sat["emissive"].value     = 0.0
        self.moon_vao.render(moderngl.TRIANGLES)

        # ── Orbital plane disc (semi-transparent) ──────────────────
        L_vec = sat.angular_momentum()
        L_mag = float(np.linalg.norm(L_vec))
        if L_mag > 1e-6:
            L_hat = (L_vec / L_mag).astype(np.float32)
            # Draw a thin ring of line segments in the orbital plane
            n_seg = 96
            r_disc = 3.5   # radius of orbital plane indicator ring
            # Build plane vectors perpendicular to L_hat
            ref = np.array([1., 0., 0.], dtype=np.float32)
            if abs(np.dot(ref, L_hat)) > 0.9:
                ref = np.array([0., 1., 0.], dtype=np.float32)
            u_hat = ref - np.dot(ref, L_hat) * L_hat
            u_hat /= np.linalg.norm(u_hat)
            v_hat2 = np.cross(L_hat, u_hat)
            ring_pts = []
            for k in range(n_seg + 1):
                angle = 2 * math.pi * k / n_seg
                pt = u_hat * math.cos(angle) * r_disc + v_hat2 * math.sin(angle) * r_disc
                ring_pts.extend(pt.tolist())
            ring_arr = np.array(ring_pts, dtype=np.float32)
            # Use arrow_vbo as scratch (resize if needed)
            if len(ring_arr) * 4 > 6 * 4:
                ring_vbo = ctx.buffer(ring_arr.tobytes())
                ring_vao = ctx.vertex_array(self.p_flat, [(ring_vbo, "3f", "in_pos")])
                self.p_flat["model"].write(_identity().T.tobytes())
                self.p_flat["color"].value = (0.30, 0.42, 0.90, 0.28)
                ring_vao.render(moderngl.LINE_STRIP, vertices=n_seg + 1)
                ring_vao.release(); ring_vbo.release()

        # HUD overlay
        if _pg_ok:
            rgba = _make_hud(physics, paused, dt_mult, self.width, self.height)
            if rgba:
                self.hud_tex.write(rgba)
                ctx.disable(moderngl.DEPTH_TEST)
                self.hud_tex.use(0); self.p_hud["hud_tex"].value = 0
                self.hud_vao.render(moderngl.TRIANGLE_STRIP)
                ctx.enable(moderngl.DEPTH_TEST)