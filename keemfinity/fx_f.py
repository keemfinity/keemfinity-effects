"""Social UI, Claude-at-work props and a liquid-glass graph.

messages, documentary, comment-box, tap, todo, shortcut-rain, uninstall, liquid-glass-graph.
"""
import math, os
from functools import lru_cache
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from . import kit
from .engine import effect, Face

PINK = (255, 79, 163)
IG_BLUE = (0, 149, 246)
IG_GREY = (239, 239, 239)
WHITE = (255, 255, 255, 255)


# ---------- private helpers ----------
@lru_cache(maxsize=256)
def _font(role, size, weight=None):
    """kit.font, but Inter's weight really applies (its first variation axis is optical size, not weight)."""
    size = max(6, int(size))
    if role != 'sans' or not weight: return kit.font(role, size, weight)
    f = ImageFont.truetype(os.path.join(kit.FONT_DIR, kit.FONTS['sans']), size)
    try: f.set_variation_by_axes([min(32, max(14, size * 0.6)), int(weight)])
    except Exception: pass
    return f


def _fit(txt, role, width, start, weight=None, floor=10):
    sz = start
    while sz > floor and _font(role, int(sz), weight).getlength(txt) > width: sz *= 0.94
    return _font(role, int(sz), weight)


def _F(s, size, weight=600, role='sans'):
    return _font(role, int(round(size * s.u)), weight)


def _ss(w, h, draw, k=3):
    """Supersampled RGBA drawing: draw(ImageDraw, k) works at k x scale; result is (w, h) and anti-aliased."""
    w, h = max(1, int(round(w))), max(1, int(round(h)))
    im = Image.new('RGBA', (w * k, h * k), (0, 0, 0, 0))
    draw(ImageDraw.Draw(im), k)
    return im.resize((w, h), Image.LANCZOS)


def _rr(w, h, r, fill=WHITE, outline=None, width=0, k=3):
    return _ss(w, h, lambda d, k: d.rounded_rectangle((0, 0, w * k - 1, h * k - 1), r * k, fill=fill,
                                                        outline=outline, width=int(width * k)), k)


def _put(s, im, x, y, anchor='c', sc=1.0, alpha=1.0, behind=False):
    if im is None or alpha <= 0.003 or sc <= 0.02: return
    im = kit.scaled(im, sc)
    (s.behind if behind else s.over)(im, x, y, anchor, kit.clamp(alpha))


def _io(t):
    t = kit.clamp(t); return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def _bump(t, a, b):
    """0 outside [a, b], a smooth 0 -> 1 -> 0 hump inside."""
    if t <= a or t >= b: return 0.0
    return math.sin(math.pi * (t - a) / (b - a))


def _shadow_base(s, key, w, h, r, blur=26, alpha=0.38, dy=14):
    """Cached soft drop shadow for a rounded panel. Returns (RGBA image, pad) - the panel sits at (pad, pad)."""
    def make():
        base = _rr(w, h, r, fill=(0, 0, 0, 255))
        im, (ox, oy) = kit.shadow(base, blur=max(2, int(blur * s.u)), alpha=alpha, offset=(0, int(dy * s.u)))
        a = np.asarray(im).copy(); pad = -ox
        # keep only the shadow: punch out the panel itself so nothing dark peeks around the glass edge
        m = np.asarray(base.getchannel('A'), np.float32) / 255
        sub = a[pad:pad + base.height, pad:pad + base.width].astype(np.float32)
        sub[..., 3] *= (1 - m); a[pad:pad + base.height, pad:pad + base.width] = sub.astype(np.uint8)
        return Image.fromarray(a), pad
    return s.once(('shadow', key), make)


def _glass_panel(s, key, x, y, w, h, r, tint=(255, 255, 255), tint_a=0.2, blur=24, border_a=0.4,
                 sh_blur=26, sh_alpha=0.38, sh_dy=14):
    """Frosted panel sampled from the current frame, with an anti-aliased mask and a soft shadow.
    Returns (image, pad): the panel's own top-left is at (x, y) when the image is drawn at (x - pad, y - pad)."""
    w, h = int(round(w)), int(round(h))
    g = kit.glass(s.rgb, (x, y, w, h), r, tint=tint, tint_a=tint_a, blur=max(1, blur * s.u), border_a=0, sheen=True)
    mask = s.once(('mask', key, w, h), lambda: _rr(w, h, r).getchannel('A'))
    g.putalpha(mask)
    if border_a > 0:
        ring = s.once(('ring', key, w, h, border_a), lambda: _rr(w, h, r, fill=(0, 0, 0, 0),
                                                                outline=(255, 255, 255, int(255 * border_a)), width=max(1.5, 2 * s.u)))
        g.alpha_composite(ring)
    sh, pad = _shadow_base(s, (key, w, h), w, h, r, sh_blur, sh_alpha, sh_dy)
    out = sh.copy(); out.alpha_composite(g, (pad, pad))
    return out, pad


def _cursor_img(s, size=54):
    def make():
        c = kit.cursor(int(size * s.u))
        im, (ox, oy) = kit.shadow(c, blur=max(2, int(6 * s.u)), alpha=0.5, offset=(0, int(4 * s.u)))
        return im, -ox + 2
    return s.once(('cursor', size), make)


def _ring(rad, width, color, alpha):
    rad = max(2, rad); n = int(rad * 2 + width * 2 + 4)
    return _ss(n, n, lambda d, k: d.ellipse(((n / 2 - rad) * k, (n / 2 - rad) * k, (n / 2 + rad) * k, (n / 2 + rad) * k),
                                            outline=color + (int(255 * alpha),), width=int(width * k)), 2)


def _click(s, bx, by, at, enter=0.7):
    """A cursor glides to (bx, by), clicks at `at` (ripple), then drifts off. Draw after the button."""
    t = s.t
    if t < at - enter: return
    mv = _io((t - (at - enter)) / (enter - 0.12))
    sx, sy = s.W * 1.04, by + s.H * 0.2
    ex, ey = bx + 6 * s.u, by + 4 * s.u
    x = kit.lerp(sx, ex, mv); y = kit.lerp(sy, ey, mv) - math.sin(math.pi * mv) * 70 * s.u
    a = kit.cubic_out((t - (at - enter)) / 0.2)
    after = t - at
    if after > 0.3:
        k = kit.cubic_in((after - 0.3) / 0.7)
        x += k * 160 * s.u; y += k * 120 * s.u; a *= 1 - k
    # ripple
    if 0 <= after < 0.55:
        q = after / 0.55
        rg = _ring(kit.expo_out(q) * 110 * s.u + 8 * s.u, max(1.5, 5 * s.u * (1 - q)), (255, 255, 255), 0.9 * (1 - q))
        s.over(rg, bx, by, 'c')
        dot = _ss(60 * s.u, 60 * s.u, lambda d, k: d.ellipse((0, 0, 60 * s.u * k - 1, 60 * s.u * k - 1), fill=(255, 255, 255, 110)), 2)
        _put(s, dot, bx, by, 'c', sc=0.4 + 0.8 * kit.expo_out(q), alpha=1 - q)
    cur, off = _cursor_img(s)
    press = 1 - 0.14 * _bump(t, at - 0.06, at + 0.14)
    _put(s, cur, x - off * press, y - off * press, 'tl', sc=press, alpha=a)


def _press(t, at):
    return 1 - 0.09 * _bump(t, at - 0.05, at + 0.16) + 0.04 * _bump(t, at + 0.12, at + 0.4)


def _pill(s, key, label, bg, fg, h=96, padx=58, size=38, weight=700):
    def make():
        f = _F(s, size, weight); tw = f.getlength(label)
        w = tw + 2 * padx * s.u; hh = h * s.u
        im = _rr(w, hh, hh * 0.28, fill=tuple(bg) + (255,))
        txt = kit.text_layer(label, f, fill=tuple(fg) + (255,))
        im.alpha_composite(txt, (int((im.width - txt.width) / 2), int((im.height - txt.height) / 2)))
        out, _ = kit.shadow(im, blur=max(2, int(18 * s.u)), alpha=0.35, offset=(0, int(8 * s.u)))
        return out
    return s.once(('pill', key, label), make)


def _heart_pts(cx, cy, size, n=80):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x = 16 * math.sin(a) ** 3
        y = -(13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a))
        pts.append((cx + x * size / 34, cy + y * size / 34 + size * 0.03))
    return pts


def _heart(size, fill=None, outline=None, width=0):
    def dr(d, k):
        pts = _heart_pts(size * k / 2, size * k / 2, size * k * 0.92)
        if fill: d.polygon(pts, fill=fill)
        if outline: d.line(pts + [pts[0]], fill=outline, width=int(width * k), joint='curve')
    return _ss(size, size, dr, 4)


def _avatar(dia, top, bottom, letter=None, f=None, ring=None):
    dia = int(round(dia))
    g = kit.gradient(dia, dia, top, bottom)
    m = _ss(dia, dia, lambda d, k: d.ellipse((0, 0, dia * k - 1, dia * k - 1), fill=WHITE)).getchannel('A')
    g.putalpha(m)
    if letter and f is not None:
        t = kit.text_layer(letter, f, fill=WHITE)
        g.alpha_composite(t, ((dia - t.width) // 2, (dia - t.height) // 2))
    if ring:
        pad = max(3, dia // 14); n = dia + 4 * pad
        out = _ss(n, n, lambda d, k: d.ellipse((0, 0, n * k - 1, n * k - 1), fill=ring[0] + (255,)))
        rg = kit.gradient(n, n, ring[0], ring[1]); rg.putalpha(out.getchannel('A'))
        inner = _ss(n - 2 * pad, n - 2 * pad, lambda d, k: d.ellipse((0, 0, (n - 2 * pad) * k - 1, (n - 2 * pad) * k - 1), fill=WHITE))
        rg.alpha_composite(inner, (pad, pad)); rg.alpha_composite(g, (2 * pad, 2 * pad))
        return rg
    return g


# =====================================================================================
# 1. messages
# =====================================================================================
def _msg_icon(sz):
    def dr(d, k):
        S = sz * k
        bw, bh = S * 0.66, S * 0.52; cx, cy = S / 2, S * 0.47
        d.ellipse((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), fill=WHITE)
        d.polygon([(cx - bw * 0.30, cy + bh * 0.25), (cx - bw * 0.40, cy + bh * 0.66), (cx - bw * 0.02, cy + bh * 0.44)], fill=WHITE)
    im = kit.gradient(sz, sz, (104, 226, 110), (32, 186, 58))
    im = Image.fromarray(np.asarray(im)).convert('RGBA')
    m = _rr(sz, sz, sz * 0.235).getchannel('A'); im.putalpha(m)
    im.alpha_composite(_ss(sz, sz, dr, 4))
    return im


def _msg_content(s, title, body, w, h):
    u = s.u; im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    pad = int(28 * u); ic = int(h - 2 * pad + 4 * u)
    icon = _msg_icon(ic); im.alpha_composite(icon, (pad, (h - ic) // 2))
    tx = pad + ic + int(24 * u)
    ft = _F(s, 31, 700); fb = _F(s, 33, 500); fn = _F(s, 26, 500)
    asc_t = ft.getmetrics()[0]
    ty = int(h * 0.25)
    d.text((tx, ty), title, font=ft, fill=(18, 18, 20, 255), anchor='ls' if False else None)
    now = 'now'; d.text((w - pad - fn.getlength(now), ty + 4 * u), now, font=fn, fill=(112, 112, 118, 255))
    avail = w - tx - pad
    fb2 = _fit(body, 'sans', avail, 33 * u, 500)
    d.text((tx, ty + asc_t + 10 * u), body, font=fb2, fill=(40, 40, 44, 255))
    return im


@effect('messages', 'Social', 'iOS message notifications pop in and stack on your chest', aliases=('imessage',))
def messages(s, p):
    msgs = p.list('text', 'how did you edit this??|wait, which app is this|send me the effects pls|KEEM')
    title = p.get('title', 'Messages')
    u = s.u; W, H = s.W, s.H
    bw, bh = int(min(W - 70 * u, 940 * u)), int(150 * u); r = 46 * u; gap = 18 * u
    n = len(msgs)
    every = min(0.35, max(0.12, (s.dur - 0.9) / max(1, n)))
    apps = [0.12 + i * every for i in range(n)]
    contents = s.once('content', lambda: [_msg_content(s, title, m, bw, bh) for m in msgs])
    base = p.num('y', 0.61) * H - 14 * u * s.p                       # slow drift up
    lim = s.face.chin + 26 * u
    env = s.env(0.01, 0.3)
    x = W / 2
    for i in range(n):
        ti = s.t - apps[i]
        if ti <= 0: continue
        push = sum((bh + gap) * kit.expo_out((s.t - apps[j]) / 0.42) for j in range(i + 1, n) if s.t > apps[j])
        y = base - push + (1 - kit.expo_out(ti / 0.4)) * 34 * u
        a = kit.cubic_out(ti / 0.16) * env
        a *= kit.clamp(1 - (lim - y) / (bh * 0.55))                    # fade before reaching the face
        if a <= 0.003: continue
        sc = 0.84 + 0.16 * kit.spring(ti, freq=4.2, decay=7.0)
        depth = sum(1 for j in range(i + 1, n) if s.t > apps[j])
        sc *= 1 - 0.015 * depth
        panel, pad = _glass_panel(s, 'msg', x - bw / 2, y, bw, bh, r, tint=(251, 247, 240), tint_a=0.74, blur=28,
                                  border_a=0.55, sh_blur=30, sh_alpha=0.32, sh_dy=16)
        panel.alpha_composite(contents[i], (pad, pad))
        _put(s, panel, x, y + bh / 2, 'c', sc=sc, alpha=a)


# =====================================================================================
# 2. documentary
# =====================================================================================
def _doc_line(s, line, hl, f, color=(246, 240, 244), hl_col=PINK):
    parts = []
    if hl and hl in line:
        i = line.index(hl); parts = [(line[:i], color), (hl, hl_col), (line[i + len(hl):], color)]
    else:
        parts = [(line, color)]
    parts = [(t, c) for t, c in parts if t]
    w = int(sum(f.getlength(t) for t, _ in parts)); asc, desc = f.getmetrics()
    im = Image.new('RGBA', (w + 8, asc + desc + 8), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x = 4
    for t, c in parts:
        d.text((x, 4), t, font=f, fill=tuple(c) + (255,)); x += f.getlength(t)
    bb = im.getbbox(); return im.crop(bb) if bb else im


@effect('documentary', 'Social', 'docuseries look: black-and-white you on a pink-lit set with titles and a Follow click')
def documentary(s, p):
    u = s.u; W, H = s.W, s.H
    lines = p.list('text', 'ONE CLIP.|48 EFFECTS.|ZERO APPS.')
    hl = p.get('hl', 'CLIP.')
    tag = p.get('tag', 'THE KEEMFINITY AI SERIES · DAILY')
    name = p.get('name', 'KEEMFINITY'); handle = p.get('handle', '@keemfinity')
    at = p.time('at', s.dur * 0.6)
    m = min(kit.expo_out(s.t / 0.6), kit.cubic_out((s.dur - s.t) / 0.3))
    f0 = s.face
    sc = 1 - 0.25 * m
    dx = (W * 0.34 - f0.cx) * m
    M = np.float32([[sc, 0, f0.cx - sc * f0.cx + dx], [0, sc, H - sc * H]])
    src = s.rgb
    bw = kit.tone(src, 'bw') * np.float32([1.0, 0.985, 0.995])
    bw = np.clip((bw - 0.04) * 1.06, 0, 1)
    pr = src * (1 - m) + bw * m
    pr = cv2.warpAffine(pr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # feather the person where the source frame cut them off, so no hard vertical edge shows mid-frame
    feath = s.once('feather', lambda: np.clip(np.minimum(np.arange(W, dtype=np.float32), W - 1 - np.arange(W, dtype=np.float32)) / (W * 0.16), 0, 1) ** 1.5)
    src_a = s.alpha * (1 - m + m * feath[None, :])
    pa = cv2.warpAffine(src_a, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    nf = Face(sc * f0.cx + M[0, 2], sc * f0.cy + M[1, 2], f0.w * sc, f0.h * sc)

    # the set: dark magenta-black, pink halo + thin ring behind the head
    def base():
        b = np.empty((H, W, 3), np.float32); b[:] = np.float32((30, 15, 28)) / 255
        return kit.vignette(b, 0.55)
    st = s.once('set', base).copy()
    halo = s.once('halo', lambda: np.asarray(kit.radial(int(1.5 * W), int(1.5 * W), 0.75 * W, 0.75 * W, 0.75 * W, PINK, 0.62, 1.7), np.float32) / 255)
    hx, hy = nf.cx, nf.cy - nf.h * 0.1
    _add_sprite(st, halo, hx, hy)
    rr = nf.h * 1.18 * (1 + 0.015 * math.sin(s.t * 2))
    ring = _ring(rr, 3 * u, PINK, 0.95)
    ring_g, _ = kit.glow(ring, radius=max(2, int(10 * u)), strength=1.4, color=PINK)
    ri = np.asarray(ring_g, np.float32) / 255
    _blend_sprite(st, ri, hx, hy)
    bg = pr * (1 - m) + st * m
    a3 = pa[..., None]
    s.rgb[:] = bg * (1 - a3) + pr * a3
    s.alpha = pa; s.face = nf

    env = s.env(0.01, 0.3)
    # top tag
    ftag = _F(s, 24, 600)
    tg = s.once('tag', lambda: kit.text_layer(tag, ftag, fill=(255, 214, 232, 255), tracking=5 * u))
    k = kit.expo_out((s.t - 0.25) / 0.6)
    _put(s, tg, W / 2, H * 0.055 + (1 - k) * 18 * u, 'tc', alpha=k * env * 0.92)
    lw = tg.width * 0.5 * k
    dash = kit.rounded(max(1, int(46 * u * k)), max(1, int(2 * u)), 1 * u, fill=PINK + (255,))
    _put(s, dash, W / 2 - tg.width / 2 - 18 * u, H * 0.055 + tg.height / 2, 'cr', alpha=k * env)
    _put(s, dash, W / 2 + tg.width / 2 + 18 * u, H * 0.055 + tg.height / 2, 'cl', alpha=k * env)

    # three condensed lines, right aligned
    def mk_lines():
        widest = max(lines, key=len)
        f = _fit(widest, 'display', W * 0.5, 150 * u)
        return [_doc_line(s, ln, hl, f) for ln in lines]
    ims = s.once('lines', mk_lines)
    y = H * 0.17; rx = W - 62 * u
    for i, im in enumerate(ims):
        k = kit.expo_out((s.t - 0.4 - i * 0.13) / 0.55)
        if k > 0:
            _put(s, im, rx + (1 - k) * 60 * u, y, 'tr', alpha=k * env)
        y += im.height + 22 * u

    # name bar bottom-left
    def mk_name():
        fn = _F(s, 46, 800, 'sans'); t = kit.text_layer(name, fn, fill=WHITE)
        bar = _rr(t.width + 44 * u, t.height + 30 * u, 10 * u, fill=PINK + (255,))
        bar.alpha_composite(t, (int(22 * u), int(15 * u)))
        bar, _ = kit.shadow(bar, blur=max(2, int(18 * u)), alpha=0.4, offset=(0, int(8 * u)))
        hd = kit.text_layer(handle, _F(s, 30, 500), fill=(255, 236, 245, 235))
        return bar, hd
    bar, hd = s.once('name', mk_name)
    by = H * 0.80
    k = kit.expo_out((s.t - 0.7) / 0.6)
    pad = int(18 * u * 2.5) + int(8 * u)
    _put(s, bar, 56 * u - pad - (1 - k) * 80 * u, by - pad, 'tl', alpha=k * env)
    k2 = kit.expo_out((s.t - 0.82) / 0.6)
    _put(s, hd, 60 * u - (1 - k2) * 60 * u, by + bar.height - 2 * pad + 18 * u, 'tl', alpha=k2 * env)

    # follow button + click
    k3 = kit.back_out(kit.clamp((s.t - 0.9) / 0.45), 1.4)
    bx, byy = W - 64 * u, by + (bar.height - 2 * pad) / 2
    fol = _pill(s, 'doc-f', 'Follow', IG_BLUE, (255, 255, 255), h=86, padx=52, size=34)
    fing = _pill(s, 'doc-g', 'Following', IG_GREY, (12, 12, 12), h=86, padx=40, size=34)
    swap = kit.clamp((s.t - at) / 0.1)
    pr_ = _press(s.t, at) * (0.6 + 0.4 * k3)
    cxb = bx - fol.width / 2 + int(18 * u * 2.5)
    if k3 > 0:
        _put(s, fol, cxb, byy, 'c', sc=pr_, alpha=env * kit.clamp(k3 * 2) * (1 - swap))
        _put(s, fing, cxb, byy, 'c', sc=pr_, alpha=env * kit.clamp(k3 * 2) * swap)
    if env > 0.5: _click(s, cxb, byy, at)


def _add_sprite(dst, spr, cx, cy):
    """Additive-ish 'screen' blend of an RGBA float sprite centred at (cx, cy)."""
    H, W = dst.shape[:2]; h, w = spr.shape[:2]
    x, y = int(round(cx - w / 2)), int(round(cy - h / 2))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return
    sp = spr[y0 - y:y1 - y, x0 - x:x1 - x]; a = sp[..., 3:4]
    d = dst[y0:y1, x0:x1]; c = sp[..., :3] * a
    d[:] = 1 - (1 - d) * (1 - c)


def _blend_sprite(dst, spr, cx, cy):
    H, W = dst.shape[:2]; h, w = spr.shape[:2]
    x, y = int(round(cx - w / 2)), int(round(cy - h / 2))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return
    sp = spr[y0 - y:y1 - y, x0 - x:x1 - x]; a = sp[..., 3:4]
    d = dst[y0:y1, x0:x1]; d[:] = d * (1 - a) + sp[..., :3] * a


# =====================================================================================
# 3. comment-box
# =====================================================================================
@effect('comment-box', 'Social', 'an Instagram comment sheet: your word gets typed, posted, liked and answered')
def comment_box(s, p):
    u = s.u; W, H = s.W, s.H
    word = p.text('KEEM')
    reply = p.get('reply', 'Sent you the pack')
    chip = p.get('chip', '48 effects')
    sh_h = int(H * 0.40)
    # timeline (compressed for short clips)
    need = 0.5 + 0.2 + len(word) * 0.11 + 0.3 + 1.0 + 0.5
    c = min(1.0, (s.dur - 0.35) / need)
    t_type = 0.6 * c; per = 0.11 * c
    t_post = t_type + len(word) * per + 0.25 * c
    t_heart = t_post + 0.5 * c; t_reply = t_post + 0.95 * c
    t = s.t
    k_in = kit.expo_out(t / 0.5); k_out = kit.cubic_in(kit.clamp(1 - (s.dur - t) / 0.3))
    top = H - sh_h * k_in + sh_h * k_out
    typed = word[:max(0, min(len(word), int((t - t_type) / max(1e-6, per)) + 1))] if t >= t_type else ''
    if t >= t_post: typed = ''

    sheet = Image.new('RGBA', (W, sh_h), (0, 0, 0, 0))
    base = s.once('sheet', lambda: _rr(W, sh_h + 60 * u, 40 * u, fill=(255, 255, 255, 255)))
    sheet.alpha_composite(base.crop((0, 0, W, sh_h)))
    d = ImageDraw.Draw(sheet)
    d.rounded_rectangle((W / 2 - 42 * u, 16 * u, W / 2 + 42 * u, 26 * u), 5 * u, fill=(214, 214, 218, 255))
    fh = _F(s, 32, 700); d.text((W / 2, 70 * u), 'Comments', font=fh, fill=(14, 14, 16, 255), anchor='mm')
    d.line((0, 112 * u, W, 112 * u), fill=(234, 234, 236, 255), width=max(1, int(1.5 * u)))

    fname = _F(s, 27, 700); fmeta = _F(s, 25, 500); ftxt = _F(s, 31, 450); fsm = _F(s, 23, 600)
    av_you = s.once('av-you', lambda: _avatar(70 * u, (120, 132, 150), (70, 80, 98), 'Y', _F(s, 30, 700)))
    av_keem = s.once('av-keem', lambda: _avatar(58 * u, (40, 40, 44), (10, 10, 12), 'K', _F(s, 26, 800),
                                                ring=((254, 196, 60), (214, 41, 118))))
    av_sam = s.once('av-sam', lambda: _avatar(70 * u, (255, 170, 120), (226, 92, 120), 'S', _F(s, 30, 700)))
    heart_o = s.once('h-o', lambda: _heart(int(40 * u), outline=(150, 150, 156, 255), width=2.6 * u))
    heart_r = s.once('h-r', lambda: _heart(int(40 * u), fill=(255, 48, 64, 255)))

    def row(y, av, name_, meta, text, heart_k=None, likes=None, indent=0, chip_=None, alpha=1.0):
        x = int(36 * u + indent)
        layer = Image.new('RGBA', (W, int(200 * u)), (0, 0, 0, 0)); dl = ImageDraw.Draw(layer)
        layer.alpha_composite(av, (x, int(4 * u)))
        tx = x + av.width + int(22 * u)
        dl.text((tx, 2 * u), name_, font=fname, fill=(14, 14, 16, 255))
        dl.text((tx + fname.getlength(name_) + 12 * u, 4 * u), meta, font=fmeta, fill=(142, 142, 148, 255))
        dl.text((tx, 40 * u), text, font=ftxt, fill=(24, 24, 26, 255))
        yy = 88 * u
        if chip_:
            cw = fsm.getlength(chip_) + 64 * u
            dl.rounded_rectangle((tx, yy, tx + cw, yy + 46 * u), 23 * u, fill=(236, 244, 255, 255), outline=(0, 149, 246, 255), width=max(1, int(2 * u)))
            dl.ellipse((tx + 18 * u, yy + 17 * u, tx + 30 * u, yy + 29 * u), fill=IG_BLUE + (255,))
            dl.text((tx + 40 * u, yy + 23 * u), chip_, font=fsm, fill=(0, 112, 210, 255), anchor='lm')
            yy += 58 * u
        dl.text((tx, yy), 'Reply', font=fsm, fill=(142, 142, 148, 255))
        hx = W - 74 * u
        if heart_k is not None:
            if heart_k <= 0:
                layer.alpha_composite(heart_o, (int(hx), int(20 * u)))
            else:
                sc = 0.6 + 0.4 * kit.back_out(heart_k, 3.0)
                hr = kit.scaled(heart_r, sc)
                layer.alpha_composite(hr, (int(hx + (heart_o.width - hr.width) / 2), int(20 * u + (heart_o.height - hr.height) / 2)))
            if likes:
                dl.text((hx + heart_o.width / 2, 72 * u), likes, font=fsm, fill=(142, 142, 148, 255), anchor='mm')
        if alpha < 1: layer = kit.faded(layer, alpha)
        sheet.alpha_composite(layer, (0, int(y)))

    ly = 136 * u
    kp = kit.expo_out((t - t_post) / 0.45) if t >= t_post else 0.0
    kr = kit.expo_out((t - t_reply) / 0.45) if t >= t_reply else 0.0
    row_h = 130 * u; rep_h = 196 * u
    # new comment + reply slot in above the existing one
    y_new = ly
    y_rep = ly + row_h * kp
    y_old = ly + row_h * kp + rep_h * kr
    row(y_old, av_sam, 'sam.edits', '2m', 'this is actually insane', heart_k=0, likes='214')
    if kp > 0:
        hk = kit.clamp((t - t_heart) / 0.4) if t >= t_heart else 0.0
        row(y_new - (1 - kp) * 30 * u, av_you, '@you', 'now', word, heart_k=hk, likes='1' if hk > 0 else None, alpha=kp)
    if kr > 0:
        row(y_rep + (1 - kr) * 24 * u, av_keem, '@keemfinity', 'now', reply, heart_k=None, indent=96 * u, chip_=chip, alpha=kr)

    # input bar
    ib = sh_h - 118 * u
    d.rectangle((0, ib - 6 * u, W, sh_h), fill=(255, 255, 255, 255))
    d.line((0, ib - 6 * u, W, ib - 6 * u), fill=(234, 234, 236, 255), width=max(1, int(1.5 * u)))
    av_me = s.once('av-me', lambda: _avatar(64 * u, (120, 132, 150), (70, 80, 98), 'Y', _F(s, 28, 700)))
    sheet.alpha_composite(av_me, (int(32 * u), int(ib + 14 * u)))
    fx0 = 32 * u + av_me.width + 20 * u; fy0 = ib + 10 * u; fx1 = W - 32 * u; fy1 = ib + 86 * u
    d.rounded_rectangle((fx0, fy0, fx1, fy1), (fy1 - fy0) / 2, outline=(214, 214, 220, 255), width=max(1, int(2 * u)), fill=(255, 255, 255, 255))
    fin = _F(s, 31, 500); cy = (fy0 + fy1) / 2
    if typed:
        d.text((fx0 + 28 * u, cy), typed, font=fin, fill=(14, 14, 16, 255), anchor='lm')
        cx_ = fx0 + 28 * u + fin.getlength(typed) + 4 * u
    else:
        d.text((fx0 + 28 * u, cy), 'Add a comment for keemfinity...', font=fin, fill=(150, 150, 156, 255), anchor='lm')
        cx_ = fx0 + 28 * u
    typing = t_type - 0.1 <= t < t_post
    if typing and (t < t_type + len(word) * per or int(t * 4) % 2 == 0):
        d.rectangle((cx_, cy - 20 * u, cx_ + 3 * u, cy + 20 * u), fill=IG_BLUE + (255,))
    fpost = _F(s, 31, 700)
    post_col = IG_BLUE + (255,) if typed else (178, 216, 250, 255)
    pdim = 1 - 0.5 * _bump(t, t_post - 0.08, t_post + 0.2)
    post_col = tuple(int(c_ * pdim + 255 * (1 - pdim)) for c_ in post_col[:3]) + (255,)
    px = fx1 - 30 * u
    d.text((px, cy), 'Post', font=fpost, fill=post_col, anchor='rm')
    # finger tap ripple on Post
    pcx = px - fpost.getlength('Post') / 2
    if t_post - 0.12 <= t < t_post + 0.45:
        q = kit.clamp((t - (t_post - 0.12)) / 0.57)
        rad = (18 + 40 * kit.expo_out(q)) * u
        tapl = Image.new('RGBA', sheet.size, (0, 0, 0, 0))
        ImageDraw.Draw(tapl).ellipse((pcx - rad, cy - rad, pcx + rad, cy + rad), fill=(0, 0, 0, int(36 * (1 - q))))
        sheet.alpha_composite(tapl)

    # shadow on the video above the sheet
    full, _ = s.once('sheet-shadow', lambda: kit.shadow(_rr(W, sh_h, 40 * u, fill=(0, 0, 0, 255)), blur=max(2, int(30 * u)), alpha=0.45, offset=(0, 0)))
    pad = (full.width - W) // 2
    s.over(full, -pad, top - pad, 'tl', alpha=1.0)
    s.over(sheet, 0, top, 'tl')


# =====================================================================================
# 4. tap
# =====================================================================================
@effect('tap', 'Social', 'a Follow button pops in, a cursor clicks it and it flips to Following', aliases=('follow',))
def tap(s, p):
    u = s.u; W, H = s.W, s.H
    at = p.time('at', s.dur * 0.55)
    col = p.color('color', IG_BLUE)
    x = p.num('x', 0.72) * W
    y = p.num('y', 0) * H if p.get('y') is not None else kit.clamp(s.face.chin + s.face.h * 0.95, H * 0.3, H * 0.78)
    y += math.sin(s.t * 2.2) * 5 * u
    fol = _pill(s, 'tap-f', 'Follow', col, (255, 255, 255), h=112, padx=72, size=46)
    fing = _pill(s, 'tap-g', 'Following', IG_GREY, (12, 12, 12), h=112, padx=54, size=46)
    gl = s.once('tap-glow', lambda: kit.glow(_rr(fol.width * 0.7, 104 * u * 0.8, 30 * u, fill=col + (255,)), radius=max(2, int(26 * u)), strength=0.9, color=col)[0])
    k = kit.back_out(kit.clamp((s.t - 0.08) / 0.45), 1.7)
    env = s.env(0.01, 0.3)
    swap = kit.clamp((s.t - at) / 0.1)
    sc = (0.55 + 0.45 * k) * _press(s.t, at)
    a = kit.clamp(k * 2.5) * env
    _put(s, gl, x, y + 6 * u, 'c', sc=sc, alpha=a * 0.55 * (1 - swap))
    _put(s, fol, x, y, 'c', sc=sc, alpha=a * (1 - swap))
    _put(s, fing, x, y, 'c', sc=sc, alpha=a * swap)
    if env > 0.5: _click(s, x, y, at)


# =====================================================================================
# 5. todo
# =====================================================================================
@effect('todo', 'Claude At Work', 'a glass checklist beside your head ticks itself off', aliases=('to-do',))
def todo(s, p):
    u = s.u; W, H = s.W, s.H; f = s.face
    items = p.list('text', 'Shoot one clip|Type /effects|Post it')
    title = p.get('title', 'Today')
    n = len(items)
    tk = p.list('ticks', '')
    if tk:
        ticks = [float(x) - p.start for x in tk] + [1e9] * n
    else:
        span = max(0.6, s.dur - 1.3)
        ticks = [0.75 + i * span / max(1, n - 1) if n > 1 else 0.75 + span / 2 for i in range(n)]
    pad = 28 * u; rowh = 62 * u
    cw = 360 * u
    gapx = 30 * u + f.w * 0.3
    left_room = f.x - gapx - 24 * u
    right_room = W - (f.x + f.w) - gapx - 24 * u
    side = s.once('side', lambda: 'l' if left_room >= right_room or left_room >= 300 * u else 'r')
    room = left_room if side == 'l' else right_room
    cw = int(max(270 * u, min(cw, room)))
    chh = int(pad + 56 * u + 12 * u + n * rowh + pad * 0.6)
    x = (f.x - gapx - cw) if side == 'l' else (f.x + f.w + gapx)
    x = kit.clamp(x, 22 * u, W - cw - 22 * u)
    y = max(30 * u, f.crown + 10 * u)
    k = kit.expo_out(s.t / 0.55); env = s.env(0.01, 0.3)
    y += math.sin(s.t * 1.8) * 4 * u
    x += (1 - k) * (-50 * u if side == 'l' else 50 * u)
    panel, pp = _glass_panel(s, 'todo', x, y, cw, chh, 34 * u, tint=(14, 14, 18), tint_a=0.42, blur=22, border_a=0.28,
                             sh_blur=24, sh_alpha=0.35, sh_dy=12)
    d = ImageDraw.Draw(panel)
    ox = oy = pp
    ft = _F(s, 40, 800); fi = _fit(max(items, key=len), 'sans', cw - 2 * pad - 52 * u, 30 * u, 500)
    fc = _F(s, 22, 600)
    d.text((ox + pad, oy + pad - 4 * u), title, font=ft, fill=WHITE)
    done = sum(1 for tt in ticks[:n] if s.t >= tt)
    cnt = f'{done}/{n}'
    d.text((ox + cw - pad, oy + pad + 4 * u), cnt, font=fc, fill=(255, 255, 255, 150), anchor='ra')
    ry = oy + pad + 56 * u + 12 * u
    for i, it in enumerate(items):
        ka = kit.expo_out((s.t - 0.15 - i * 0.08) / 0.4)
        q = kit.clamp((s.t - ticks[i]) / 0.35) if s.t >= ticks[i] else 0.0
        cy = ry + rowh / 2; bx = ox + pad; bs = 36 * u
        al = int(255 * ka)
        if q <= 0:
            d.ellipse((bx, cy - bs / 2, bx + bs, cy + bs / 2), outline=(255, 255, 255, int(170 * ka)), width=max(1, int(2.5 * u)))
        else:
            sc = 0.5 + 0.5 * kit.back_out(q, 2.4); r_ = bs / 2 * sc
            d.ellipse((bx + bs / 2 - r_, cy - r_, bx + bs / 2 + r_, cy + r_), fill=(52, 199, 89, 255))
            cm = kit.check_mark(int(bs * 0.8 * sc) + 1, (255, 255, 255), width=max(2, 4 * u * sc))
            panel.alpha_composite(cm, (int(bx + bs / 2 - cm.width / 2), int(cy - cm.height / 2)))
        tx = bx + bs + 18 * u
        ta = int(al * (1 - 0.5 * q))
        d.text((tx, cy), it, font=fi, fill=(255, 255, 255, ta), anchor='lm')
        if q > 0:
            sw = fi.getlength(it) * kit.expo_out(q)
            d.line((tx - 2 * u, cy + 2 * u, tx - 2 * u + sw, cy + 2 * u), fill=(255, 255, 255, 220), width=max(1, int(3 * u)))
        ry += rowh
    _put(s, panel, x - pp, y - pp, 'tl', alpha=k * env)


# =====================================================================================
# 6. shortcut-rain
# =====================================================================================
def _keycap(s, label, h, wide):
    u = s.u; w = h * wide
    def dr(d, k):
        W_, H_ = w * k, h * k
        d.rounded_rectangle((0, 0, W_ - 1, H_ - 1), H_ * 0.2, fill=(150, 153, 162, 255))
        d.rounded_rectangle((W_ * 0.02, H_ * 0.02, W_ * 0.98, H_ * 0.96), H_ * 0.19, fill=(186, 189, 197, 255))
    im = _ss(w, h, dr, 3)
    tw, th = w * 0.84, h * 0.74; tx, ty = w * 0.08, h * 0.05
    top = kit.gradient(int(tw), int(th), (253, 253, 255), (224, 226, 232))
    top.putalpha(_rr(int(tw), int(th), h * 0.16).getchannel('A'))
    im.alpha_composite(top, (int(tx), int(ty)))
    hl = _rr(int(tw), int(th), h * 0.16, fill=(0, 0, 0, 0), outline=(255, 255, 255, 220), width=max(1, h * 0.015))
    im.alpha_composite(hl, (int(tx), int(ty)))
    sz = h * (0.40 if len(label) == 1 else 0.24)
    f = _font('sans', int(sz), 600)
    t = kit.text_layer(label, f, fill=(62, 64, 74, 255))
    im.alpha_composite(t, (int(tx + (tw - t.width) / 2), int(ty + (th - t.height) / 2)))
    out, _ = kit.shadow(im, blur=max(2, int(h * 0.09)), alpha=0.42, offset=(0, int(h * 0.1)))
    return out


@effect('shortcut-rain', 'Claude At Work', 'keyboard keycaps rain down behind you', aliases=('rain',))
def shortcut_rain(s, p):
    u = s.u; W, H = s.W, s.H
    labels = p.list('text', '⌘|/|E|F|↵|⇧|esc|K|⌥')
    count = int(p.num('count', 32))

    def make():
        rng = np.random.default_rng(48)
        keys = []
        for i in range(count):
            z = float(rng.uniform(0, 1)) ** 0.8
            lab = labels[i % len(labels)]
            h = (70 + 120 * z) * u
            wide = 1.45 if len(lab) > 1 or lab in ('↵', '⇧') else 1.0
            im = _keycap(s, lab, h, wide)
            br = (1 - z) * 3.2 * u
            if br > 0.4: im = kit.blurred(im, br)
            side = rng.uniform()          # most keys fall where the person doesn't cover them
            xf = rng.uniform(-0.05, 0.3) if side < 0.38 else rng.uniform(0.7, 1.05) if side < 0.76 else rng.uniform(0.2, 0.8)
            keys.append(dict(z=z, im=im, x=float(xf) * W, y0=float(rng.uniform(-1.1, 0.15)),
                             v=(0.22 + 0.42 * z) * H, rot=float(rng.uniform(-35, 35)), vr=float(rng.uniform(-70, 70)) * (1.1 - 0.6 * z),
                             sway=float(rng.uniform(0, 6.28))))
        keys.sort(key=lambda k: k['z'])
        return keys
    keys = s.once('keys', make)
    env = s.env(0.2, 0.35)
    s.tint(lambda r: r * 0.86, alpha=env, where='background')
    for k in keys:
        span = H + 2 * k['im'].height
        yy = (k['y0'] * H + k['v'] * s.t) % span - k['im'].height
        xx = k['x'] + math.sin(s.t * 1.3 + k['sway']) * 14 * u
        im = kit.rotated(k['im'], k['rot'] + k['vr'] * s.t)
        s.behind(im, xx, yy, 'c', alpha=env * (0.7 + 0.3 * k['z']))


# =====================================================================================
# 7. uninstall
# =====================================================================================
def _glitch(rgb, g, seed, u, hard=False):
    H, W = rgb.shape[:2]
    rng = np.random.default_rng(seed)
    out = rgb.copy()
    d = int((3 + 26 * g) * u * (2.0 if hard else 1.0))
    if d > 0:
        out[..., 0] = np.roll(rgb[..., 0], d, axis=1)
        out[..., 2] = np.roll(rgb[..., 2], -d, axis=1)
        if hard: out[..., 1] = np.roll(rgb[..., 1], int(d * 0.4), axis=0)
    ns = int(2 + 14 * g + (16 if hard else 0))
    for _ in range(ns):
        h = int(rng.uniform(4, 24 + 90 * g) * u)
        y0 = int(rng.uniform(0, H - h))
        off = int(rng.uniform(-1, 1) * (20 + 150 * g) * u * (2.2 if hard else 1))
        band = np.roll(out[y0:y0 + h], off, axis=1)
        if rng.uniform() < 0.3 * g + (0.4 if hard else 0):
            band = band[..., [rng.integers(0, 3), rng.integers(0, 3), rng.integers(0, 3)]]
        out[y0:y0 + h] = band
    if hard:
        for _ in range(7):
            bw, bh = int(rng.uniform(60, 420) * u), int(rng.uniform(10, 90) * u)
            x0, y0 = int(rng.uniform(0, W - bw)), int(rng.uniform(0, H - bh))
            if rng.uniform() < 0.5: out[y0:y0 + bh, x0:x0 + bw] = 1 - out[y0:y0 + bh, x0:x0 + bw]
            else: out[y0:y0 + bh, x0:x0 + bw] = np.float32([[0, 1, 0.9], [1, 0.1, 0.6], [1, 1, 1]][rng.integers(0, 3)])
        out = np.clip((out - 0.5) * 1.35 + 0.5, 0, 1)
        jx = int(rng.uniform(-1, 1) * 30 * u); out = np.roll(out, jx, axis=1)
    step = max(2, int(4 * u))
    out[::step] *= 1 - (0.08 + 0.22 * g)
    if g > 0.05:
        noise = rng.uniform(-1, 1, (H // 4 + 1, W // 4 + 1, 1)).astype(np.float32)
        noise = cv2.resize(noise, (W, H), interpolation=cv2.INTER_NEAREST)[..., None] if noise.ndim == 3 else noise
        out = np.clip(out + noise.reshape(H, W, 1) * (0.1 if hard else 0.035) * g, 0, 1)
    return out


@effect('uninstall', 'Claude At Work', 'a macOS "Uninstalling CapCut" dialog fills while the frame glitches apart')
def uninstall(s, p):
    u = s.u; W, H = s.W, s.H
    app = p.text('CapCut')
    dw, dh = int(min(W - 80 * u, 880 * u)), int(318 * u)
    t_done = max(0.8, s.dur * 0.8)
    q = kit.clamp((s.t - 0.35) / (t_done - 0.35))
    # progress with little stalls
    pr = _io(q) * 0.35 + kit.cubic_out(q) * 0.15 + q * 0.5
    pr = pr - 0.03 * math.sin(pr * 18) * (1 - pr) * pr
    pr = kit.clamp(pr)
    hard = s.t >= t_done + 0.06
    k = kit.expo_out(s.t / 0.4); env = s.env(0.01, 0.12)
    cx = W / 2; cy = max(H * 0.57, s.face.chin + dh / 2 + 70 * u)
    x0, y0 = cx - dw / 2, cy - dh / 2
    panel, pp = _glass_panel(s, 'dlg', x0, y0, dw, dh, 26 * u, tint=(238, 238, 240), tint_a=0.84, blur=30, border_a=0.5,
                             sh_blur=34, sh_alpha=0.45, sh_dy=20)

    def static():
        im = Image.new('RGBA', panel.size, (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
            ccx = pp + 30 * u + i * 30 * u; ccy = pp + 30 * u; rr = 9 * u
            d.ellipse((ccx - rr, ccy - rr, ccx + rr, ccy + rr), fill=c + (255,))
        ic = int(116 * u)
        icon = kit.gradient(ic, ic, (58, 58, 64), (14, 14, 18))
        icon.putalpha(_rr(ic, ic, ic * 0.23).getchannel('A'))
        lt = kit.text_layer(app[:1].upper(), _F(s, 64, 800), fill=WHITE)
        icon.alpha_composite(lt, ((ic - lt.width) // 2, (ic - lt.height) // 2))
        icon, _ = kit.shadow(icon, blur=max(2, int(8 * u)), alpha=0.35, offset=(0, int(4 * u)))
        ipad = (icon.width - ic) // 2
        im.alpha_composite(icon, (int(pp + 40 * u - ipad), int(pp + 82 * u - ipad)))
        tf = _fit(f'Uninstalling {app}…', 'sans', dw - 240 * u, 36 * u, 700)
        d.text((pp + 186 * u, pp + 92 * u), f'Uninstalling {app}…', font=tf, fill=(22, 22, 24, 255))
        return im
    st = s.once('static', static)
    panel.alpha_composite(st)
    d = ImageDraw.Draw(panel)
    fs = _F(s, 26, 500)
    tx = pp + 186 * u
    sub = f'Moving {app} to Trash…' if pr < 1 else f'{app} was removed.'
    d.text((tx, pp + 146 * u), sub, font=fs, fill=(108, 108, 114, 255))
    bx0, by0, bx1 = tx, pp + 206 * u, pp + dw - 120 * u
    d.rounded_rectangle((bx0, by0, bx1, by0 + 12 * u), 6 * u, fill=(214, 214, 220, 255))
    fw = (bx1 - bx0) * pr
    if fw > 12 * u:
        d.rounded_rectangle((bx0, by0, bx0 + fw, by0 + 12 * u), 6 * u, fill=(10, 122, 255, 255))
    d.text((pp + dw - 40 * u, by0 + 6 * u), f'{int(round(pr * 100))}%', font=_F(s, 26, 600), fill=(80, 80, 86, 255), anchor='rm')
    left = max(0, int(math.ceil((1 - pr) * 9)))
    d.text((tx, by0 + 32 * u), f'About {left} seconds remaining' if pr < 1 else 'Done', font=_F(s, 22, 500), fill=(140, 140, 146, 255))
    _put(s, panel, cx, cy, 'c', sc=0.92 + 0.08 * k, alpha=k * env)
    # the glitch grows with the progress bar, then a hard glitch at the end
    g = kit.clamp(pr ** 2.6) * 0.75
    seed = s.i // 2
    spike = np.random.default_rng(seed + 991).uniform() < 0.1 + 0.35 * g
    gg = min(1.0, g * 1.5 + 0.08 if spike else g)
    if gg > 0.02 or hard:
        s.rgb[:] = _glitch(s.rgb, gg if not hard else 1.0, seed, u, hard=hard)


# =====================================================================================
# 8. liquid-glass-graph
# =====================================================================================
def _lg_maps(w, h, r, band, u):
    """Refraction maps for a liquid-glass rounded rect: edge pixels sample further inside (lens bend)."""
    m = np.asarray(_rr(w, h, r).getchannel('A'), np.float32) / 255
    dist = cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    tt = np.clip(1 - dist / band, 0, 1)
    sm = cv2.GaussianBlur(dist, (0, 0), max(1.0, band * 0.25))
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    nrm = np.sqrt(gx ** 2 + gy ** 2) + 1e-6; nx, ny = gx / nrm, gy / nrm          # points inward
    disp = (tt ** 1.6) * band * 1.4
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # rim highlight: top-left lit edge, softer bottom-right counter-light
    rim = np.clip(1 - dist / max(1.5, 3.0 * u), 0, 1)
    light = np.clip(nx * 0.55 + ny * 0.83, 0, 1) + 0.45 * np.clip(-(nx * 0.55 + ny * 0.83), 0, 1)
    return dict(m=m, tt=tt, mx=xx + nx * disp, my=yy + ny * disp, nx=nx, ny=ny, disp=disp, rim=rim * (0.25 + 0.75 * light))


def _liquid_glass(s, x, y, w, h, r):
    u = s.u
    band = 46 * u
    mp = s.once(('lgmaps', w, h), lambda: _lg_maps(w, h, r, band, u))
    H, W = s.rgb.shape[:2]
    pad = int(band * 2)
    X0, Y0 = int(x) - pad, int(y) - pad
    big = cv2.copyMakeBorder(s.rgb, pad * 2, pad * 2, pad * 2, pad * 2, cv2.BORDER_REFLECT)
    reg = big[Y0 + pad * 2 - pad:Y0 + pad * 2 + h + pad, X0 + pad * 2 - pad:X0 + pad * 2 + w + pad]
    reg = np.ascontiguousarray(reg)
    soft = cv2.GaussianBlur(reg, (0, 0), 3 * u)
    heavy = cv2.GaussianBlur(reg, (0, 0), 16 * u)
    out = np.empty((h, w, 3), np.float32)
    for c, extra in ((0, 1.0), (1, 1.08), (2, 1.16)):         # slight chromatic split at the edge
        mx = (np.mgrid[0:h, 0:w][1].astype(np.float32) + mp['nx'] * mp['disp'] * extra) + pad
        my = (np.mgrid[0:h, 0:w][0].astype(np.float32) + mp['ny'] * mp['disp'] * extra) + pad
        out[..., c] = cv2.remap(soft[..., c], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    inner = heavy[pad:pad + h, pad:pad + w]
    tt = mp['tt'][..., None]
    g = inner * (1 - tt) + out * tt
    g = g * 0.74 + 0.02                                       # a touch darker so white type reads
    yy = np.linspace(1, 0, h, dtype=np.float32)[:, None, None]
    g = g + 0.10 * yy ** 2.5                                  # top sheen
    g = g + (tt ** 2.2) * 0.22                                # bright inner edge
    g = np.clip(g + mp['rim'][..., None] * 0.9, 0, 1)
    im = Image.fromarray((g * 255).astype(np.uint8)).convert('RGBA')
    im.putalpha(Image.fromarray((mp['m'] * 255).astype(np.uint8)))
    return im


@effect('liquid-glass-graph', 'Numbers', 'a liquid-glass card under your chin draws a green graph and counts up')
def liquid_glass_graph(s, p):
    u = s.u; W, H = s.W, s.H
    label = p.text('retention')
    to_s = str(p.get('to', '53%')); fr_s = str(p.get('from', '0'))
    v1, v0 = kit.parse_number(to_s), kit.parse_number(fr_s)
    col = p.color('color', (48, 222, 120))
    cw, ch = int(min(W - 80 * u, 880 * u)), int(430 * u); r = 54 * u
    k = kit.expo_out(s.t / 0.55); env = s.env(0.01, 0.3)
    x = W / 2 - cw / 2
    y = kit.clamp(s.face.chin + max(50 * u, s.face.h * 0.3), 0, H - ch - 50 * u) + (1 - k) * 60 * u + math.sin(s.t * 1.7) * 5 * u
    glass = _liquid_glass(s, x, y, cw, ch, r)
    sh, pad = _shadow_base(s, 'lg', cw, ch, r, 36, 0.42, 22)
    card = sh.copy(); card.alpha_composite(glass, (pad, pad))

    # data: a seeded, smooth, rising curve
    def data():
        rng = np.random.default_rng(53)
        n = 48; xs = np.linspace(0, 1, n)
        base = 0.12 + 0.78 * (xs ** 1.3)
        noise = np.convolve(rng.normal(0, 0.09, n + 8), np.ones(7) / 7, mode='same')[4:4 + n]
        ys = np.clip(base + noise * (0.4 + xs), 0.04, 0.97); ys[-1] = 0.94
        return xs, ys
    xs, ys = s.once('data', data)
    q = _io(kit.clamp((s.t - 0.35) / max(0.6, min(2.0, s.dur * 0.55))))
    qn = kit.cubic_out(kit.clamp((s.t - 0.35) / max(0.6, min(2.0, s.dur * 0.55))))
    val = v0 + (v1 - v0) * qn

    SS = 2
    gx0, gx1 = 44 * u, cw - 44 * u; gy0, gy1 = 196 * u, ch - 40 * u
    lay = Image.new('RGBA', (cw * SS, ch * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for gl in range(4):
        yy = (gy0 + (gy1 - gy0) * gl / 3) * SS
        d.line((gx0 * SS, yy, gx1 * SS, yy), fill=(255, 255, 255, 34), width=max(1, int(1.5 * u * SS)))
    npts = len(xs); upto = q * (npts - 1)
    pts = []
    for i in range(int(upto) + 1):
        pts.append((gx0 + xs[i] * (gx1 - gx0), gy1 - ys[i] * (gy1 - gy0)))
    if upto < npts - 1 and q > 0:
        i = int(upto); fr = upto - i
        xa = xs[i] + (xs[i + 1] - xs[i]) * fr; ya = ys[i] + (ys[i + 1] - ys[i]) * fr
        pts.append((gx0 + xa * (gx1 - gx0), gy1 - ya * (gy1 - gy0)))
    if len(pts) >= 2 and q > 0:
        # gradient fill under the line
        poly = [(px * SS, py * SS) for px, py in pts] + [(pts[-1][0] * SS, gy1 * SS), (pts[0][0] * SS, gy1 * SS)]
        fm = Image.new('L', lay.size, 0); ImageDraw.Draw(fm).polygon(poly, fill=255)
        grad = s.once('fillgrad', lambda: kit.gradient(cw * SS, ch * SS, col, col, alpha=(0, 0)))
        ga = np.zeros((ch * SS, cw * SS), np.float32)
        tcol = np.clip((np.arange(ch * SS, dtype=np.float32) / SS - gy0 + 20 * u) / (gy1 - gy0 + 20 * u), 0, 1)
        ga[:] = (0.7 * (1 - tcol) ** 1.2 * 255)[:, None]
        ga = np.minimum(ga, np.asarray(fm, np.float32))
        fill = Image.new('RGBA', lay.size, col + (0,)); fill.putalpha(Image.fromarray(ga.astype(np.uint8)))
        lay.alpha_composite(fill)
        d.line([(px * SS, py * SS) for px, py in pts], fill=col + (255,), width=int(7 * u * SS), joint='curve')
    lay = lay.resize((cw, ch), Image.LANCZOS)
    glw, go = kit.glow(lay, radius=max(2, int(9 * u)), strength=0.8, color=col)
    card.alpha_composite(glw.crop((-go[0], -go[1], -go[0] + cw, -go[1] + ch)), (pad, pad))
    # head dot
    if len(pts) >= 1 and q > 0:
        hx, hy = pts[-1]; pulse = 0.5 + 0.5 * math.sin(s.t * 7)
        halo = _ss(60 * u, 60 * u, lambda dd, kk: dd.ellipse((0, 0, 60 * u * kk - 1, 60 * u * kk - 1), fill=col + (int(70 + 60 * pulse),)), 2)
        card.alpha_composite(halo, (int(pad + hx - halo.width / 2), int(pad + hy - halo.height / 2)))
        dot = _ss(22 * u, 22 * u, lambda dd, kk: dd.ellipse((0, 0, 22 * u * kk - 1, 22 * u * kk - 1), fill=WHITE, outline=col + (255,), width=int(5 * u * kk)), 3)
        card.alpha_composite(dot, (int(pad + hx - dot.width / 2), int(pad + hy - dot.height / 2)))
    # type
    dc = ImageDraw.Draw(card)
    lab = s.once('lab', lambda: kit.text_layer(label.upper(), _F(s, 25, 650), fill=(255, 255, 255, 200), tracking=4 * u))
    card.alpha_composite(lab, (int(pad + 46 * u), int(pad + 44 * u)))
    num = kit.number_text(val, to_s) if not to_s.replace('.', '').isdigit() or True else str(val)
    fnum = _F(s, 112, 800)
    nl = kit.text_layer(num, fnum, fill=WHITE)
    nl, no = kit.shadow(nl, blur=max(2, int(10 * u)), alpha=0.35, offset=(0, int(4 * u)))
    card.alpha_composite(nl, (int(pad + 42 * u + no[0]), int(pad + 84 * u + no[1])))
    # live pill
    fpl = _F(s, 22, 650); txt = 'Last 7 days'
    pw = fpl.getlength(txt) + 56 * u
    px1 = pad + cw - 44 * u; py0 = pad + 40 * u
    dc.rounded_rectangle((px1 - pw, py0, px1, py0 + 44 * u), 22 * u, fill=(255, 255, 255, 36), outline=(255, 255, 255, 70), width=max(1, int(1.5 * u)))
    blink = 0.6 + 0.4 * math.sin(s.t * 5)
    dc.ellipse((px1 - pw + 18 * u, py0 + 16 * u, px1 - pw + 30 * u, py0 + 28 * u), fill=col + (int(255 * blink),))
    dc.text((px1 - pw + 40 * u, py0 + 22 * u), txt, font=fpl, fill=(255, 255, 255, 220), anchor='lm')
    _put(s, card, W / 2, y + ch / 2, 'c', sc=0.94 + 0.06 * k, alpha=k * env)
