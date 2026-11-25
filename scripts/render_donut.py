"""Render donut.svg: a studio-lit, raymarched ASCII donut spinning on one axis.

Local-only (needs numpy, scipy, pillow):  python scripts/render_donut.py
"""
import html
import math
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_SVG = os.path.join(REPO_DIR, "donut.svg")
CARD_W, CARD_H = 891, 340
PREVIEW_PNG = os.environ.get("DONUT_PREVIEW", "")

COLS, ROWS = 96, 48
SS = 3                       # 3x3 supersampling per character cell (anti-aliased silhouette)
N_FRAMES = 36                # torus is symmetric under 180deg Y-rotation -> 36 frames cover a full visual cycle
DURATION = 2.1               # same angular speed as before (360deg / 4.2s)
FONT_SIZE, LINE_H, X0, Y0 = 5.7, 6.0, 281, 31.0

R2, R1 = 1.92, 0.86
CAM_DIST, FOV = 14.0, 4.55
ASPECT = 1.143

# Gray levels (dark -> bright). Glyph density + glyph brightness = two layers of tonal resolution.
GRAYS = ["#2d333b", "#444c56", "#636e7b", "#8b949e", "#b4bec9", "#dce3ea", "#ffffff"]
GRAY_EDGES = [0.07, 0.16, 0.28, 0.42, 0.58, 0.78]
DEFAULT_LEVEL = 4


def build_ramp(n=16):
    """Measure real ink coverage of each glyph in Menlo and pick an evenly spaced, perceptually sorted ramp."""
    cand = " .,:;-~=+*xo#%&@"
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 48)
    dens = []
    for ch in cand:
        im = Image.new("L", (32, 60), 0)
        ImageDraw.Draw(im).text((2, 2), ch, fill=255, font=font)
        dens.append((np.asarray(im, dtype=np.float32).mean(), ch))
    dens.sort()
    if n >= len(dens):
        return "".join(ch for _, ch in dens)
    lo, hi = dens[0][0], dens[-1][0]
    ramp, used = [], set()
    for i in range(n):
        target = lo + (hi - lo) * i / (n - 1)
        best = min((d for d in dens if d[1] not in used), key=lambda d: abs(d[0] - target))
        used.add(best[1])
        ramp.append(best[1])
    return "".join(ramp)


RAMP = build_ramp()


def sdf(px, py, pz):
    q = np.sqrt(px * px + py * py) - R2
    return np.sqrt(q * q + pz * pz) - R1


def norm3(x, y, z):
    l = math.sqrt(x * x + y * y + z * z)
    return x / l, y / l, z / l


def render_frame(yaw):
    h, w = ROWS * SS, COLS * SS
    xs = (np.arange(w, dtype=np.float32) + 0.5) / w * 2.0 - 1.0
    ys = 1.0 - (np.arange(h, dtype=np.float32) + 0.5) / h * 2.0
    gx, gy = np.meshgrid(xs * ASPECT, ys)

    rdx, rdy, rdz = gx, gy, np.full_like(gx, -FOV)
    rl = np.sqrt(rdx * rdx + rdy * rdy + rdz * rdz)
    rdx, rdy, rdz = rdx / rl, rdy / rl, rdz / rl

    c, s = math.cos(-yaw), math.sin(-yaw)

    def to_local(x, y, z):  # inverse of a pure Y-axis rotation through the donut's exact center
        return x * c + z * s, y, -x * s + z * c

    rox, roy, roz = to_local(0.0, 0.0, CAM_DIST)
    dx, dy, dz = to_local(rdx, rdy, rdz)

    def to_world(x, y, z):
        return x * c - z * s, y, x * s + z * c

    # World-space studio rig (fixed while the donut spins)
    key = to_local(*norm3(0.80, 0.26, 0.54))
    fill = to_local(*norm3(-0.70, 0.00, 0.70))
    rim = to_local(*norm3(-0.90, 0.04, -0.42))

    # --- primary raymarch ---
    t = np.full_like(gx, CAM_DIST - R2 - R1 - 0.2)
    hit = np.zeros(gx.shape, dtype=bool)
    for _ in range(72):
        d = sdf(rox + dx * t, roy + dy * t, roz + dz * t)
        hit |= d < 0.0015
        t = np.where(~hit & (t < CAM_DIST + 4.0), t + d, t)

    px, py, pz = rox + dx * t, roy + dy * t, roz + dz * t
    rxy = np.sqrt(px * px + py * py) + 1e-8
    nx, ny, nz = px - R2 * px / rxy, py - R2 * py / rxy, pz
    nl = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx, ny, nz = nx / nl, ny / nl, nz / nl

    # --- soft shadows from the key light ---
    sh = np.ones_like(gx)
    st = np.full_like(gx, 0.04)
    ox, oy, oz = px + nx * 0.01, py + ny * 0.01, pz + nz * 0.01
    for _ in range(40):
        hd = sdf(ox + key[0] * st, oy + key[1] * st, oz + key[2] * st)
        sh = np.minimum(sh, 8.0 * hd / st)
        st = st + np.clip(hd, 0.02, 0.35)
    sh = np.clip(sh, 0.0, 1.0)
    sh = sh * sh * (3.0 - 2.0 * sh)

    # --- ambient occlusion ---
    occ, wgt = np.zeros_like(gx), 1.0
    for i in range(1, 6):
        hh = 0.12 * i
        occ += (hh - sdf(px + nx * hh, py + ny * hh, pz + nz * hh)) * wgt
        wgt *= 0.7
    ao = np.clip(1.0 - 1.4 * occ, 0.0, 1.0)

    # --- shading: chiaroscuro key, faint fill, back rim, glossy studio softbox reflections ---
    vx, vy, vz = -dx, -dy, -dz
    ndl = np.clip(nx * key[0] + ny * key[1] + nz * key[2], 0.0, 1.0)
    ndf = np.clip(nx * fill[0] + ny * fill[1] + nz * fill[2], 0.0, 1.0)
    ndr = np.clip(nx * rim[0] + ny * rim[1] + nz * rim[2] + 0.35, 0.0, 1.0)
    ndv = np.clip(nx * vx + ny * vy + nz * vz, 0.0, 1.0)
    fres = (1.0 - ndv) ** 3

    hx, hy, hz = key[0] + vx, key[1] + vy, key[2] + vz
    hl = np.sqrt(hx * hx + hy * hy + hz * hz) + 1e-8
    ndh = np.clip((nx * hx + ny * hy + nz * hz) / hl, 0.0, 1.0)
    spec = (ndh ** 90) * sh

    # reflection vector -> world space -> two tall softbox strips (highlights glide as it spins)
    vdn = vx * nx + vy * ny + vz * nz
    rx_, ry_, rz_ = 2 * vdn * nx - vx, 2 * vdn * ny - vy, 2 * vdn * nz - vz
    wx, wy, wz = to_world(rx_, ry_, rz_)
    az = np.arctan2(wx, wz)
    tall = np.clip(1.0 - (np.abs(wy) / 0.75) ** 4, 0.0, 1.0)
    env = (np.exp(-((az - 0.80) / 0.16) ** 2) + 0.45 * np.exp(-((az + 1.05) / 0.11) ** 2)) * tall
    refl = env * (0.10 + 0.90 * fres) * ao

    col = (
        0.025 * ao
        + 0.82 * ndl * sh
        + 0.05 * ndf * ao
        + 0.75 * fres * ndr * ao
        + 1.20 * spec
        + 0.60 * refl
    )
    v = 1.0 - np.exp(-1.3 * col)
    v = np.where(hit, v, 0.0)

    # --- downsample supersampled buffer to character cells ---
    v_cell = v.reshape(ROWS, SS, COLS, SS).mean(axis=(1, 3))
    cov = hit.reshape(ROWS, SS, COLS, SS).mean(axis=(1, 3))

    # --- soft bloom around the brightest highlights ---
    bright = np.clip(v_cell - 0.70, 0.0, None)
    glow = gaussian_filter(bright, sigma=(1.2, 2.2)) * 1.1
    v_cell = v_cell + glow
    cov = np.maximum(cov, np.clip(glow * 6.0, 0.0, 1.0))
    return v_cell / 0.78, cov


def cell_to_glyph(val, cov):
    if cov < 0.12:
        return " ", DEFAULT_LEVEL
    val = min(1.0, max(0.0, val))
    idx = max(1, min(len(RAMP) - 1, int(round((val ** 1.05) * (len(RAMP) - 1)))))
    lvl = sum(val > e for e in GRAY_EDGES)
    return RAMP[idx], lvl


def frame_rows(yaw):
    val, cov = render_frame(yaw)
    rows = []
    for r in range(ROWS):
        rows.append([cell_to_glyph(val[r, c], cov[r, c]) for c in range(COLS)])
    return rows


def row_to_svg(row):
    """Group runs of equal gray level into <tspan>s (default level inherits the group fill)."""
    while row and row[-1][0] == " ":
        row = row[:-1]
    if not row:
        return None
    out, buf, cur = [], "", None
    for ch, lvl in row:
        lvl = DEFAULT_LEVEL if ch == " " else lvl
        if ch == " " and cur is not None:
            lvl = cur  # keep spaces inside the current run to minimise tspans
        if lvl != cur and buf:
            out.append(buf if cur == DEFAULT_LEVEL else f'<tspan class="g{cur}">{buf}</tspan>')
            buf = ""
        cur = lvl
        buf += html.escape(ch)
    if buf:
        out.append(buf if cur == DEFAULT_LEVEL else f'<tspan class="g{cur}">{buf}</tspan>')
    return "".join(out)


def save_preview(all_frames, picks):
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 12)
    cw, chh = 7.2, 12.6
    pw, ph = int(COLS * cw) + 20, int(ROWS * chh) + 20
    sheet = Image.new("RGB", (pw * len(picks), ph), (13, 17, 23))
    d = ImageDraw.Draw(sheet)
    for i, k in enumerate(picks):
        for r, row in enumerate(all_frames[k]):
            for cidx, (ch, lvl) in enumerate(row):
                if ch != " ":
                    d.text((i * pw + 10 + cidx * cw, 10 + r * chh), ch, fill=GRAYS[lvl], font=font)
    sheet.save(PREVIEW_PNG)


def build_svg():
    pct = 100.0 / N_FRAMES
    step_s = DURATION / N_FRAMES
    gray_css = "".join(f" .g{i}{{fill:{g}}}" for i, g in enumerate(GRAYS) if i != DEFAULT_LEVEL)
    all_frames, groups = [], []
    for k in range(N_FRAMES):
        yaw = math.pi * k / N_FRAMES  # 0 -> 180deg; the torus repeats every half turn
        rows = frame_rows(yaw)
        all_frames.append(rows)
        delay = -((N_FRAMES - k) % N_FRAMES) * step_s
        els = []
        for r, row in enumerate(rows):
            body = row_to_svg(row)
            if body:
                els.append(f'<text x="{X0}" y="{Y0 + r * LINE_H:.1f}" xml:space="preserve">{body}</text>')
        groups.append(f'<g class="df" style="animation-delay:{delay:.3f}s">{"".join(els)}</g>')
    if PREVIEW_PNG:
        save_preview(all_frames, [0, N_FRAMES // 4, N_FRAMES // 2])
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CARD_W}" height="{CARD_H}" viewBox="0 0 {CARD_W} {CARD_H}" '
        'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">\n'
        "<style>\n"
        f"  .df {{ opacity: 0; animation: spin {DURATION:.2f}s linear infinite; }}\n"
        f"  @keyframes spin {{ 0%, {pct - 0.01:.3f}% {{ opacity: 1; }} {pct:.3f}%, 100% {{ opacity: 0; }} }}\n"
        f" {gray_css}\n"
        "</style>\n"
        f'<rect x="0.5" y="0.5" width="{CARD_W - 1}" height="{CARD_H - 1}" rx="10" fill="#0d1117" stroke="#30363d"/>\n'
        f'<g font-size="{FONT_SIZE}" fill="{GRAYS[DEFAULT_LEVEL]}">\n' + "\n".join(groups) + "\n</g>\n</svg>\n"
    )
    with open(OUT_SVG, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Wrote {OUT_SVG} ({len(svg) / 1024:.1f} KB, {N_FRAMES} frames)")


if __name__ == "__main__":
    build_svg()
