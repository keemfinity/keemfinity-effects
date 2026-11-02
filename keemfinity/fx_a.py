"""Camera & Light: punch-ins, pushes, lighting and grades."""
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

from . import kit
from .engine import effect, Face

GROUP = 'Camera & Light'


# ---------- private helpers ----------
def _inout(t):
    """Cubic ease in-out."""
    t = kit.clamp(t)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def _cam(s, z, tx=None, ty=None, k=1.0, blur=0.0):
    """Zoom about the face and slide the face toward (tx, ty) by k, never showing past the frame edge.
    blur > 0 adds a short zoom-blur trail (fraction of zoom)."""
    f = s.face; W, H = s.W, s.H
    z = max(1.0, float(z))
    if z < 1.0005 and k <= 0: return
    cx, cy = f.cx, f.cy
    tx = cx if tx is None else tx; ty = cy if ty is None else ty
    dx = kit.lerp(0, tx - cx, k); dy = kit.lerp(0, ty - cy, k)
    dx = min(max(dx, -(W - cx) * (z - 1)), cx * (z - 1))
    dy = min(max(dy, -(H - cy) * (z - 1)), cy * (z - 1))

    def M(zz):
        return np.float32([[zz, 0, cx - zz * cx + dx], [0, zz, cy - zz * cy + dy]])
    if abs(z - 1) < 1e-4 and abs(dx) < 0.01 and abs(dy) < 0.01: return
    src = s.rgb.copy()
    m0 = M(z)
    if blur > 0.002:
        acc = np.zeros_like(src)
        zs = [z * (1 - blur * j) for j in (0.0, 0.33, 0.66, 1.0)]
        for zz in zs:
            acc += cv2.warpAffine(src, M(max(zz, 1.0)), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        s.rgb[:] = acc / len(zs)
    else:
        s.rgb[:] = cv2.warpAffine(src, m0, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    s.alpha = cv2.warpAffine(s.alpha, m0, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    s.face = Face(z * f.cx + m0[0, 2], z * f.cy + m0[1, 2], f.w * z, f.h * z)


def _ss_line(d, pts, fill, width):
    d.line(pts, fill=fill, width=int(width), joint='curve')
    r = width / 2
    for x, y in (pts[0], pts[-1]):
        d.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def _pill(txt, f, fg=(255, 255, 255, 255), bg=(12, 12, 16, 170), padx=None, pady=None, outline=None):
    t = kit.text_layer(txt, f, fill=fg, pad=0)
    padx = int(f.size * 0.7 if padx is None else padx); pady = int(f.size * 0.45 if pady is None else pady)
    w, h = t.width + 2 * padx, t.height + 2 * pady
    im = kit.rounded(w, h, h / 2, fill=bg, outline=outline, width=1 if outline else 0)
    im.alpha_composite(t, (padx, pady))
    return im


def _bg_blur(rgb, alpha, sigma):
    """Blur only the background (normalised so the person does not bleed into it). Done at low res."""
    H, W = rgb.shape[:2]
    sc = 4 if sigma > 8 else 2
    w, h = max(8, W // sc), max(8, H // sc)
    bga = 1 - alpha
    small = cv2.resize(rgb * bga[..., None], (w, h), interpolation=cv2.INTER_AREA)
    sa = cv2.resize(bga, (w, h), interpolation=cv2.INTER_AREA)
    sg = max(0.5, sigma / sc)
    num = cv2.GaussianBlur(small, (0, 0), sg); den = cv2.GaussianBlur(sa, (0, 0), sg)
    out = num / np.maximum(den, 1e-3)[..., None]
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)


# =====================================================================================
@effect('locked-on', GROUP, 'corner brackets snap onto your face and the camera locks in', aliases=('lock',))
def locked_on(s, p):
    Z = kit.clamp(p.num('zoom', 1.35), 1.0, 2.2)
    tag = p.text('TARGET: KEEM').upper()
    u = s.u; t = s.t
    kin = _inout(kit.window(t, 0.05, 0.9))
    kout = 1 - _inout(kit.window(t, s.dur - 0.4, 0.4))
    k = kin * kout
    _cam(s, 1 + (Z - 1) * k, s.W / 2, s.H * 0.33, k)

    f = s.face
    a = kit.window(t, 0.0, 0.1) * kit.clamp((s.dur - t) / 0.2)
    if a <= 0: return
    snap = kit.expo_out(kit.window(t, 0.04, 0.3))
    sc = 1 + 1.5 * (1 - snap)
    breathe = 1 + 0.012 * math.sin(t * 5.5)
    bw = f.w * 1.32 * sc * breathe; bh = f.h * 1.55 * sc * breathe
    bcx, bcy = f.cx, f.cy - f.h * 0.12
    th = max(2, int(round(3.2 * u))); L = min(bw, bh) * 0.22
    pad = int(14 * u)
    im = Image.new('RGBA', (int(bw) + 2 * pad, int(bh) + 2 * pad), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = pad, pad, pad + bw, pad + bh
    for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([(cx, cy + sy * L), (cx, cy), (cx + sx * L, cy)], fill=(255, 255, 255, 255), width=th, joint='curve')
    # tiny centre ticks on each side
    tl = L * 0.18
    for (xa, ya, xb, yb) in (((x0 + x1) / 2, y0 - tl * 0.2, (x0 + x1) / 2, y0 + tl), ((x0 + x1) / 2, y1 + tl * 0.2, (x0 + x1) / 2, y1 - tl)):
        d.line([(xa, ya), (xb, yb)], fill=(255, 255, 255, 200), width=max(1, th - 1))
    sh, (ox, oy) = kit.shadow(im, blur=int(5 * u) + 1, alpha=0.55, offset=(0, int(2 * u)))
    s.over(sh, bcx - bw / 2 - pad + ox, bcy - bh / 2 - pad + oy, 'tl', alpha=a)

    # tag label typed beside the top-right bracket
    ka = kit.window(t, 0.22, 0.12)
    if ka > 0:
        n = int(round(len(tag) * kit.window(t, 0.24, 0.035 * len(tag))))
        f_ = kit.font('mono', 22 * u)
        def mk(txt):
            body = kit.text_layer(txt + ('_' if n < len(tag) else ''), f_, fill=(255, 255, 255, 255), pad=0)
            ph = int(f_.size * 1.75); dot = int(f_.size * 0.42); gap = int(f_.size * 0.5)
            w = int(f_.size * 0.6) + dot + gap + max(body.width, 1) + int(f_.size * 0.7)
            pl = kit.rounded(w, ph, int(4 * u), fill=(10, 10, 14, 185), outline=(255, 255, 255, 70), width=1)
            dd = ImageDraw.Draw(pl); dx0 = int(f_.size * 0.6); dy0 = (ph - dot) // 2
            dd.ellipse((dx0, dy0, dx0 + dot, dy0 + dot), fill=(255, 64, 72, 255))
            if txt: pl.alpha_composite(body, (dx0 + dot + gap, (ph - body.height) // 2))
            return pl
        pl = s.once(('tag', n), lambda: mk(tag[:n]))
        tx = bcx + bw / 2 + 12 * u; ty = bcy - bh / 2
        if tx + pl.width > s.W - 16 * u:            # no room: sit on top of the bracket instead
            tx = bcx + bw / 2 - pl.width; ty = bcy - bh / 2 - pl.height - 10 * u
        ty = max(16 * u, ty)
        slide = (1 - kit.expo_out(ka)) * 18 * u
        s.over(pl, tx - slide, ty, 'tl', alpha=a * kit.cubic_out(ka))
    # live readout under the bottom-left bracket
    kr = kit.window(t, 0.35, 0.2)
    if kr > 0:
        z_now = 1 + (Z - 1) * k
        txt = f'LOCK {int(100 * snap):3d}%  ZOOM {z_now:.2f}x'
        r = kit.text_layer(txt, kit.font('mono', 16 * u), fill=(255, 255, 255, 215), pad=int(4 * u))
        sh, (ox, oy) = kit.shadow(r, blur=int(4 * u) + 1, alpha=0.6, offset=(0, 1))
        s.over(sh, bcx - bw / 2 + ox, bcy + bh / 2 + 10 * u + oy, 'tl', alpha=a * kr)


# =====================================================================================
@effect('lean-in', GROUP, 'a fast punch-in on your face with a little overshoot')
def lean_in(s, p):
    Z = kit.clamp(p.num('zoom', 1.45), 1.0, 2.0)
    sp = max(0.25, p.num('speed', 1.0))
    hit = 0.12 / sp

    def zf(t):
        if t <= 0: return 1.0
        k = kit.back_out(t / (hit * 1.7), 2.6)
        drift = 0.03 * kit.clamp((t - hit) / max(0.1, s.dur - hit))
        out = 1 - _inout(kit.window(t, s.dur - 0.38 / sp ** 0.5, 0.38 / sp ** 0.5))
        return 1 + ((Z - 1) * k + drift * Z) * out
    z = min(2.0, zf(s.t))
    vel = abs(z - zf(s.t - 1 / s.fps))
    blur = kit.clamp(vel * 0.55, 0, 0.09)
    kpos = kit.clamp((z - 1) / max(1e-3, Z - 1))
    _cam(s, z, s.W / 2, s.H * 0.36, kit.clamp(kpos), blur=blur)


# =====================================================================================
@effect('slow-push', GROUP, 'slow cinematic push-in while the background melts into depth of field')
def slow_push(s, p):
    Z = kit.clamp(p.num('zoom', 1.6), 1.0, 2.5)
    B = max(0.0, p.num('blur', 14))
    k = 0.35 * s.p + 0.65 * _inout(s.p)          # moves from frame one, eases at the end
    _cam(s, 1 + (Z - 1) * k, s.W / 2, s.H * 0.34, k)
    kb = kit.cubic_out(kit.window(s.t, 0.1, s.dur * 0.8))
    sig = B * s.u * kb
    if sig > 0.6:
        bg = _bg_blur(s.rgb, s.alpha, sig)
        a = (1 - s.alpha)[..., None]
        # a touch darker and softer behind, so the person pops
        bg = bg * (1 - 0.12 * kb)
        s.rgb[:] = s.rgb * (1 - a) + bg * a


# =====================================================================================
def _stat_card(w, h, u, glass_im, kind, label, value, prog, accent, seed):
    """Glass card with a label, a number and a graph growing with prog (0..1)."""
    im = glass_im.copy()
    pad = int(w * 0.075)
    lab = kit.text_layer(label.upper(), kit.font('sans', w * 0.052, 650), fill=(255, 255, 255, 215), pad=0, tracking=int(w * 0.006))
    im.alpha_composite(lab, (pad, pad))
    num = kit.text_layer(value, kit.font('sans', w * 0.21, 800), fill=(255, 255, 255, 255), pad=0)
    ny = pad + lab.height + int(w * 0.035)
    im.alpha_composite(num, (pad, ny))
    # live dot
    dr = int(w * 0.014); dx = w - pad - dr * 2; dy = pad + lab.height // 2 - dr
    ImageDraw.Draw(im).ellipse((dx, dy, dx + 2 * dr, dy + 2 * dr), fill=accent + (255,))
    gy0 = ny + num.height + int(w * 0.07); gy1 = h - pad; gx0 = pad; gx1 = w - pad
    gw, gh = gx1 - gx0, gy1 - gy0
    if gh < 10: return im
    S = 2
    g = Image.new('RGBA', (gw * S, gh * S), (0, 0, 0, 0)); d = ImageDraw.Draw(g)
    rng = np.random.default_rng(seed)
    if kind == 'line':
        n = 10
        vals = [0.08 + 0.84 * (i / (n - 1)) ** 1.7 + (rng.uniform(-0.07, 0.07) if 0 < i < n - 1 else 0) for i in range(n)]
        pts = [(i / (n - 1) * gw * S, (1 - vals[i]) * gh * S) for i in range(n)]
        e = kit.clamp(prog) * (n - 1); m = int(e)
        use = pts[:m + 1]
        if m < n - 1:
            fr = e - m; a0, b0 = pts[m], pts[m + 1]
            use = use + [(a0[0] + (b0[0] - a0[0]) * fr, a0[1] + (b0[1] - a0[1]) * fr)]
        if len(use) >= 2:
            # fill under
            fill = Image.new('L', g.size, 0)
            ImageDraw.Draw(fill).polygon(use + [(use[-1][0], gh * S), (0, gh * S)], fill=255)
            grad = np.linspace(0.42, 0.0, g.size[1], dtype=np.float32)[:, None]
            fa = (np.asarray(fill, np.float32) * grad).astype(np.uint8)
            fl = Image.new('RGBA', g.size, accent + (0,)); fl.putalpha(Image.fromarray(fa))
            g.alpha_composite(fl)
            ln = Image.new('RGBA', g.size, (0, 0, 0, 0))
            _ss_line(ImageDraw.Draw(ln), use, accent + (255,), w * 0.016 * S)
            gl = ln.filter(ImageFilter.GaussianBlur(w * 0.02 * S))
            g.alpha_composite(gl); g.alpha_composite(gl); g.alpha_composite(ln)
            x, y = use[-1]; r = w * 0.02 * S
            d.ellipse((x - r * 1.9, y - r * 1.9, x + r * 1.9, y + r * 1.9), fill=accent + (70,))
            d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 255), outline=accent + (255,), width=int(w * 0.008 * S))
    else:
        n = 7
        vals = [0.22 + 0.74 * (i / (n - 1)) ** 1.35 + (rng.uniform(-0.06, 0.06) if 0 < i < n - 1 else 0) for i in range(n)]
        gap = gw * S * 0.035; bwid = (gw * S - gap * (n - 1)) / n
        for i in range(n):
            k = kit.back_out(kit.clamp((prog * 1.6 - i * 0.09) / 0.55), 1.4)
            if k <= 0: continue
            bh_ = max(bwid * 0.6, vals[i] * gh * S * k)
            x = i * (bwid + gap); top = gh * S - bh_
            col = accent + (255,) if i == n - 1 else (255, 255, 255, 95)
            d.rounded_rectangle((x, top, x + bwid, gh * S), radius=int(bwid * 0.28), fill=col)
        if prog > 0.4:
            hl = g.copy().filter(ImageFilter.GaussianBlur(w * 0.018 * S))
            g = Image.alpha_composite(hl, g)
    g = g.resize((gw, gh), Image.LANCZOS)
    im.alpha_composite(g, (gx0, gy0))
    return im


@effect('over-the-shoulder-zoom', GROUP, 'the camera pushes beside your head to reveal glass stat cards', aliases=('ots',))
def over_the_shoulder_zoom(s, p):
    Z = kit.clamp(p.num('zoom', 2.0), 1.1, 2.6)
    labels = (p.list('text', 'Views|Followers') + ['Views', 'Followers'])[:2]
    nums = (p.list('to', '197K|672K') + ['197K', '672K'])[:2]
    accent = p.color('color', (120, 220, 255))
    u, t, D = s.u, s.t, s.dur
    m = D / 2
    push = min(0.75, m * 0.55); back = min(0.32, m * 0.3)
    if t < m:
        side, ts, te = -1, 0.0, m
    else:
        side, ts, te = 1, m, D
    lt = t - ts; seg = te - ts
    w = kit.expo_out(lt / push) * 0.6 + _inout(lt / push) * 0.4
    w *= (1 - _inout(kit.window(lt, seg - back, back)))
    z = 1 + (Z - 1) * w
    tx = s.W * (0.71 if side < 0 else 0.29)
    _cam(s, z, tx, s.H * 0.37, w)

    f = s.face
    # card appears once the push has mostly landed
    c0 = min(0.3, seg * 0.22)
    kc = kit.window(lt, c0, 0.45)
    ko = 1 - kit.cubic_in(kit.window(lt, seg - back - 0.12, back * 0.8))
    if kc <= 0 or ko <= 0: return
    margin = 0.045 * s.W
    if side < 0:
        right = f.cx - f.w * 0.56
        cw = int(min(0.48 * s.W, right - margin)); cx0 = right - cw
    else:
        left = f.cx + f.w * 0.56
        cw = int(min(0.48 * s.W, s.W - margin - left)); cx0 = left
    if cw < 0.24 * s.W:            # face is too wide here: keep the card readable at the frame edge
        cw = int(0.30 * s.W); cx0 = margin if side < 0 else s.W - margin - cw
    ch = int(cw * 0.9)
    cy0 = f.cy - f.h * 0.62 - ch * 0.25
    cy0 = max(0.05 * s.H, min(cy0, s.H - ch - 0.05 * s.H))
    g = kit.glass(s.rgb, (cx0, cy0, cw, ch), int(30 * u), tint=(225, 232, 250), tint_a=0.10, blur=max(6, int(26 * u)), border_a=0.45)
    # cool inner wash + dark base so white text always reads
    ga = np.asarray(g, np.float32)
    ga[..., :3] = ga[..., :3] * 0.62 + np.float32([18, 22, 34]) * 0.38
    g = Image.fromarray(ga.astype(np.uint8))
    idx = 0 if side < 0 else 1
    prog = kit.cubic_out(kit.window(lt, c0 + 0.12, max(0.4, seg - back - c0 - 0.35)))
    val = kit.parse_number(nums[idx]) * kit.expo_out(kit.window(lt, c0 + 0.08, max(0.35, (seg - back - c0) * 0.75)))
    card = _stat_card(cw, ch, u, g, 'line' if idx == 0 else 'bar', labels[idx], kit.number_text(val, nums[idx]), prog, accent, 11 + idx)
    sc = 0.82 + 0.18 * kit.back_out(kc, 1.7)
    sc *= 1 + 0.02 * kit.window(lt, c0, seg)
    card = kit.scaled(card, sc)
    sh, (ox, oy) = kit.shadow(card, blur=int(26 * u), alpha=0.45, offset=(0, int(14 * u)))
    cxm = cx0 + cw / 2; cym = cy0 + ch / 2 + (1 - kit.expo_out(kc)) * 40 * u
    s.behind(sh, cxm - card.width / 2 + ox, cym - card.height / 2 + oy, 'tl', alpha=kit.cubic_out(kc) * ko)


# =====================================================================================
@effect('spotlight', GROUP, 'kill the lights: you stay lit in front of a glowing neon word')
def spotlight(s, p):
    word = p.text('KEEMFINITY')
    col = p.color('color', (255, 70, 170))
    push = kit.clamp(p.num('push', 1.06), 1.0, 1.5)
    u, t = s.u, s.t
    k = kit.cubic_out(s.env(0.4, 0.3))
    f0 = s.once('face0', lambda: (s.face.cx, s.face.cy, s.face.w, s.face.h))

    # lights out: room nearly black, a hint of it left; a warm pool behind the head
    bgk = 1 - 0.86 * k
    def lights(rgb):
        y = kit.luma(rgb)[..., None]
        out = (rgb * 0.6 + y * 0.4) * np.float32([0.92, 0.95, 1.05])
        return out * bgk + rgb * 0
    s.tint(lambda rgb: lights(rgb), alpha=k, where='background')
    s.tint(lambda rgb: rgb * bgk, alpha=1 - k, where='background')

    # neon word behind the head (on the wall: placed from the first frame, moves only with the camera)
    def mk():
        fnt = kit.fit_font(word, 'serif-italic', s.W * 0.88, s.W * 0.24)
        core = kit.text_layer(word, fnt, fill=(255, 240, 248, 255), pad=int(10 * u))
        g1, o1 = kit.glow(core, radius=int(10 * u) + 1, strength=2.2, color=col)
        big = Image.new('RGBA', core.size, (0, 0, 0, 0)); big.alpha_composite(core)
        g2, o2 = kit.glow(core, radius=int(42 * u) + 1, strength=1.6, color=col)
        g2.alpha_composite(g1, (o1[0] - o2[0], o1[1] - o2[1]))
        return g2
    neon = s.once('neon', mk)
    # flicker on
    fl = [0, 0.7, 0.1, 0.0, 0.9, 0.3, 1.0, 0.6, 1.0]
    tf = kit.window(t, 0.25, 0.45)
    fa = 1.0 if tf >= 1 else fl[min(len(fl) - 1, int(tf * len(fl)))]
    fa *= 0.94 + 0.06 * math.sin(t * 31) * math.sin(t * 7.3)
    fcx, fcy, fw, fh = f0
    crown = fcy - fh * 0.85
    wy = crown + fh * 0.3
    sc = 1 + 0.025 * s.p
    # light the wall around the word
    if fa * k > 0.01:
        s.behind(kit.scaled(neon, sc), s.W / 2, wy, 'c', alpha=k * fa)

    # warm rim on the person + gentle falloff away from the face
    a = s.alpha
    b = cv2.GaussianBlur(a, (0, 0), 9 * u)
    rim = np.clip((a - b) * 2.6, 0, 1)
    H = s.H
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    rim *= np.clip(1.15 - yy * 0.9, 0.2, 1)
    warm = np.float32([1.0, 0.72, 0.48])
    s.rgb += rim[..., None] * warm * 0.42 * k
    fall = 1 - 0.32 * k * np.clip((yy - fcy / H - 0.1) * 1.2, 0, 1)
    s.rgb[:] = s.rgb * (1 - a[..., None]) + s.rgb * fall[..., None] * a[..., None]
    np.clip(s.rgb, 0, 1, out=s.rgb)
    _cam(s, 1 + (push - 1) * _inout(s.p), None, None, 0)


# =====================================================================================
def _bulb_grid(ch, rows):
    """Character -> boolean grid (rows x cols) of bulbs, sampled from a heavy font."""
    if ch == ' ': return np.zeros((rows, max(2, rows // 3)), bool)
    f = kit.font('heavy', 220)
    im = kit.text_layer(ch, f, fill=(255, 255, 255, 255), pad=0)
    a = np.asarray(im.getchannel('A'), np.float32) / 255
    h, w = a.shape
    # cap height of the font -> rows; keep aspect
    cols = max(2, int(round(w / h * rows * 1.0)))
    cov = cv2.resize(a, (cols, rows), interpolation=cv2.INTER_AREA)
    return cov > 0.42


def _led_layout(lines, W, H, u):
    rows = 9 if len(lines) == 1 else 7
    grids = [[_bulb_grid(c, rows) for c in ln] for ln in lines]
    line_cols = [sum(g.shape[1] for g in gl) + max(0, len(gl) - 1) for gl in grids]
    maxc = max(line_cols)
    pitch = min(W * 0.9 / maxc, (H * 0.26 / len(lines)) / (rows + 1.6))
    return rows, grids, line_cols, pitch


@effect('led-sign', GROUP, 'the room dims and a marquee bulb sign types itself out behind you', aliases=('led',))
def led_sign(s, p):
    lines = [x.upper() for x in p.list('text', 'KEEM')] or ['KEEM']
    col = p.color('color', (255, 176, 72))
    push = kit.clamp(p.num('push', 1.05), 1.0, 1.5)
    u, t = s.u, s.t
    k = kit.cubic_out(s.env(0.35, 0.3))
    f0 = s.once('face0', lambda: (s.face.cx, s.face.cy, s.face.w, s.face.h))

    def build():
        rows, grids, line_cols, pitch = _led_layout(lines, s.W, s.H, u)
        r = pitch * 0.30
        lh = (rows + 1.6) * pitch
        sign_h = lh * len(lines)
        fcx, fcy, fw, fh = f0
        crown = fcy - fh * 0.85
        top = max(s.H * 0.035, crown - sign_h * 0.5)
        pad = int(pitch * 3)
        Hs = int(sign_h + 2 * pad); y_off = int(top) - pad
        rng = np.random.default_rng(5)
        letters = []          # (lit light float HxWx3 sparse box, x0,y0)
        off = Image.new('RGBA', (s.W, Hs), (0, 0, 0, 0)); do = ImageDraw.Draw(off)
        for li, gl in enumerate(grids):
            x = (s.W - line_cols[li] * pitch) / 2 + pitch / 2
            y0 = pad + li * lh + pitch * 0.8
            for g in gl:
                rr, cc = np.nonzero(g)
                pts = [(x + c * pitch, y0 + rr_ * pitch) for rr_, c in zip(rr, cc)]
                if pts:
                    xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
                    bx0 = int(max(0, min(xs) - pitch * 3)); bx1 = int(min(s.W, max(xs) + pitch * 3))
                    by0 = int(max(0, min(ys) - pitch * 3)); by1 = int(min(Hs, max(ys) + pitch * 3))
                    core = np.zeros((by1 - by0, bx1 - bx0), np.float32)
                    cm = Image.new('L', (bx1 - bx0, by1 - by0), 0); dc = ImageDraw.Draw(cm)
                    for (px, py) in pts:
                        v = int(255 * rng.uniform(0.82, 1.0))
                        dc.ellipse((px - bx0 - r, py - by0 - r, px - bx0 + r, py - by0 + r), fill=v)
                        do.ellipse((px - r, py - r, px + r, py + r), fill=(70, 52, 34, 150))
                        do.ellipse((px - r * 0.45, py - r * 0.6, px - r * 0.05, py - r * 0.2), fill=(150, 120, 90, 90))
                    core = np.asarray(cm, np.float32) / 255
                    c = np.float32(col) / 255
                    hot = np.float32([1.0, 0.96, 0.88])
                    g1 = cv2.GaussianBlur(core, (0, 0), r * 0.9)
                    g2 = cv2.GaussianBlur(core, (0, 0), r * 3.2)
                    light = core[..., None] * hot * 1.4 + g1[..., None] * c * 1.3 + g2[..., None] * c * 0.9
                    letters.append((light.astype(np.float32), bx0, by0))
                x += (g.shape[1] + 1) * pitch
        return dict(letters=letters, off=off, y_off=y_off, Hs=Hs, pitch=pitch)
    B = s.once('sign', build)

    # dim the room, keep the person a little brighter
    s.tint(lambda rgb: rgb * (1 - 0.75 * k) * np.float32([1.0, 0.97, 0.94]) ** k, where='background')
    s.tint(lambda rgb: rgb * (1 - 0.18 * k), where='person')

    # letter timing
    n = len(B['letters'])
    each = min(0.26, max(0.08, (s.dur * 0.5) / max(1, n)))
    fl = [0.9, 0.05, 0.7, 0.0, 1.0, 0.35, 1.0]
    y0 = B['y_off']; Hs = B['Hs']
    light = np.zeros((Hs, s.W, 3), np.float32)
    total = 0.0
    for i, (L, bx, by) in enumerate(B['letters']):
        tf = kit.window(t, 0.3 + i * each, 0.22)
        if tf <= 0: continue
        v = 1.0 if tf >= 1 else fl[min(len(fl) - 1, int(tf * len(fl)))]
        v *= 0.95 + 0.05 * math.sin(t * 13 + i * 1.7)
        hh, ww = L.shape[:2]
        light[by:by + hh, bx:bx + ww] += L * v
        total += v
    # paste unlit bulbs + light, behind the person
    ys0, ys1 = max(0, y0), min(s.H, y0 + Hs)
    if ys1 <= ys0: return
    sl = slice(ys0 - y0, ys1 - y0)
    bgm = (1 - s.alpha[ys0:ys1])[..., None]
    off = np.asarray(B['off'], np.float32)[sl] / 255
    oa = off[..., 3:4] * k * bgm
    reg = s.rgb[ys0:ys1]
    reg[:] = reg * (1 - oa) + off[..., :3] * oa
    reg += light[sl] * bgm * k
    # soft amber spill across the room and onto the top of the head
    if total > 0:
        small = cv2.resize(light[..., 0], (max(4, s.W // 8), max(4, Hs // 8)), interpolation=cv2.INTER_AREA)
        sp = cv2.GaussianBlur(small, (0, 0), 14)
        spf = np.zeros((s.H // 8 + 1, s.W // 8 + 1), np.float32)
        oy = max(0, y0 // 8); hh = min(sp.shape[0], spf.shape[0] - oy)
        spf[oy:oy + hh, :sp.shape[1]] = sp[:hh, :spf.shape[1]]
        spf = cv2.GaussianBlur(spf, (0, 0), 22)
        spf = cv2.resize(spf, (s.W, s.H), interpolation=cv2.INTER_LINEAR)
        c = np.float32(col) / 255
        s.rgb += (spf * 1.6)[..., None] * c * k * (1 - 0.5 * s.alpha[..., None])
    np.clip(s.rgb, 0, 1, out=s.rgb)
    _cam(s, 1 + (push - 1) * _inout(s.p), None, None, 0)


# =====================================================================================
@effect('moody', GROUP, 'a cinematic colour grade eases in with a soft vignette', aliases=('grade',))
def moody(s, p):
    look = str(p.get('grade', 'moody')).lower()
    k = kit.smooth(s.env(0.5, 0.35))
    _cam(s, 1 + 0.03 * _inout(s.p), None, None, 0)
    if k <= 0.001: return
    s.tint(lambda rgb: kit.tone(rgb, look), alpha=k)
    vg = s.once('vig', lambda: kit.vignette(np.ones((s.H, s.W, 3), np.float32), 0.5)[..., :1])
    s.rgb[:] = s.rgb * (1 - k + k * vg)


# =====================================================================================
@effect('swipe', GROUP, 'a before/after slider sweeps across to reveal the graded look', aliases=('before-after',))
def swipe(s, p):
    look = str(p.get('grade', 'pop')).lower()
    u, t, W, H = s.u, s.t, s.W, s.H
    a = s.env(0.2, 0.3)
    w_ = kit.window(t, 0.15, max(0.5, s.dur - 0.6))
    e = 0.55 * w_ + 0.45 * _inout(w_)
    x = W * (0.04 + 0.90 * e)
    # graded on the left, original on the right, 1.5 px feathered seam
    after = kit.tone(s.rgb, look)
    after = np.clip((after - 0.5) * 1.08 + 0.5, 0, 1)
    if look == 'pop': after = kit.vignette(after, 0.22)
    xs = np.arange(W, dtype=np.float32)
    m = np.clip((x - xs) / (1.5 * u) + 0.5, 0, 1)[None, :, None] * a
    s.rgb[:] = s.rgb * (1 - m) + after * m

    # the line
    lw = max(2, int(round(3.5 * u)))
    line = s.once('line', lambda: kit.shadow(kit.rounded(lw, H + 20, lw / 2), blur=int(8 * u) + 1, alpha=0.5, offset=(0, 0)))
    s.over(line[0], x + line[1][0] - lw / 2, line[1][1] - 10, 'tl', alpha=a)

    def handle():
        R = int(38 * u); S = 3
        im = Image.new('RGBA', (2 * R * S, 2 * R * S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        d.ellipse((0, 0, 2 * R * S - 1, 2 * R * S - 1), fill=(255, 255, 255, 255))
        c = R * S; q = R * S * 0.22; w_ = int(R * S * 0.09)
        for sgn in (-1, 1):
            tip = c + sgn * R * S * 0.42; back = tip - sgn * q
            d.line([(back, c - q), (tip, c), (back, c + q)], fill=(30, 30, 36, 255), width=w_, joint='curve')
        im = im.resize((2 * R, 2 * R), Image.LANCZOS)
        return kit.shadow(im, blur=int(12 * u) + 1, alpha=0.45, offset=(0, int(5 * u)))
    hd, (ox, oy) = s.once('handle', handle)
    hy = H * 0.56
    pop = kit.back_out(kit.window(t, 0.05, 0.35), 2.0)
    s.over(kit.scaled(hd, max(0.05, pop)), x, hy, 'c', alpha=a)

    # pills: pinned to the bottom corners, each cut by the line so it only shows on its own side
    f_ = kit.font('sans', 34 * u, 600)
    pa = s.once('p_after', lambda: _pill('after', f_, bg=(12, 12, 16, 170), outline=(255, 255, 255, 70)))
    pb = s.once('p_before', lambda: _pill('before', f_, bg=(12, 12, 16, 170), outline=(255, 255, 255, 70)))
    py = H * 0.87; edge = 40 * u
    ka = kit.cubic_out(kit.window(t, 0.12, 0.3)) * a
    ax = edge; vis = int(kit.clamp((x - ax) / pa.width) * pa.width)
    if vis > 0: s.over(pa.crop((0, 0, vis, pa.height)), ax, py, 'tl', alpha=ka)
    bx = W - edge - pb.width; cut = int(kit.clamp((x - bx) / pb.width) * pb.width)
    if cut < pb.width: s.over(pb.crop((cut, 0, pb.width, pb.height)), bx + cut, py, 'tl', alpha=ka)
