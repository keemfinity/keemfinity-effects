"""Claude At Work (glass widgets, checklist, render split, timeline, toggles, the bill) and Your Reels (reel frame, b-roll split)."""
import math
from functools import lru_cache
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from . import kit
from .engine import effect, Face

ORANGE = (217, 119, 87)
CREAM = (244, 238, 228)
INK = (31, 30, 29)
GREEN = (52, 199, 89)
MINT = (110, 231, 160)
WHITE = (255, 255, 255)
_SS = 3


# ---------- private helpers ----------
@lru_cache(maxsize=256)
def _font(role, size, weight=None):
    """kit.font, but sets the right variation axis (Inter has opsz before wght; Montserrat defaults to Thin)."""
    f = ImageFont.truetype(kit.os.path.join(kit.FONT_DIR, kit.FONTS.get(role, kit.FONTS['sans'])), max(1, int(round(size))))
    try:
        if role == 'sans': f.set_variation_by_axes([kit.clamp(size, 14, 32), int(weight or 400)])
        elif role == 'geo': f.set_variation_by_axes([int(weight or 500)])
        elif role in ('serif',) and weight: f.set_variation_by_axes([int(weight)])
    except Exception:
        pass
    return f


def _aa(w, h, fn, k=_SS):
    big = Image.new('RGBA', (max(1, int(round(w * k))), max(1, int(round(h * k)))), (0, 0, 0, 0))
    fn(ImageDraw.Draw(big), k)
    return big.resize((max(1, int(round(w))), max(1, int(round(h)))), Image.LANCZOS)


@lru_cache(maxsize=256)
def _rrect(w, h, r, fill=None, outline=None, width=0.0):
    """Anti-aliased rounded rectangle (read-only: copy before drawing on it)."""
    w, h = max(1, int(round(w))), max(1, int(round(h)))
    def f(d, k):
        d.rounded_rectangle((0, 0, w * k - 1, h * k - 1), max(0, r * k), fill=fill, outline=outline,
                            width=max(0, int(round(width * k))))
    return _aa(w, h, f)


@lru_cache(maxsize=64)
def _mask(w, h, r):
    return _rrect(w, h, r, fill=(255, 255, 255, 255)).getchannel('A')


@lru_cache(maxsize=64)
def _shadow(w, h, r, blur, alpha=0.45, dy=0):
    """Soft shadow of a rounded rect; returns (image, pad) where the rect sits at (pad, pad + dy)."""
    pad = int(blur * 3) + abs(int(dy))
    im = Image.new('RGBA', (int(w) + 2 * pad, int(h) + 2 * pad), (0, 0, 0, 0))
    a = Image.new('L', im.size, 0); a.paste(_mask(w, h, r).point(lambda v: int(v * alpha)), (pad, pad + int(dy)))
    sm = a.resize((max(1, a.width // 4), max(1, a.height // 4)), Image.BILINEAR)
    sm = Image.fromarray(cv2.GaussianBlur(np.asarray(sm), (0, 0), max(0.5, blur / 4)))
    im.putalpha(sm.resize(im.size, Image.BILINEAR))
    return im, pad


def _glass(bg, box, r, tint=(40, 40, 38), tint_a=0.3, blur=26, border=0.32, sheen=0.08):
    """Frosted glass panel (anti-aliased edge) sampled from the frame `bg` under `box`."""
    x, y, w, h = [int(round(v)) for v in box]; H, W = bg.shape[:2]
    w, h = max(2, w), max(2, h)
    pad = int(blur * 1.5)
    X0, Y0, X1, Y1 = x - pad, y - pad, x + w + pad, y + h + pad
    x0, y0, x1, y1 = max(0, X0), max(0, Y0), min(W, X1), min(H, Y1)
    patch = np.zeros((Y1 - Y0, X1 - X0, 3), np.float32)
    if x1 > x0 and y1 > y0:
        patch[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = bg[y0:y1, x0:x1]
    f = 4
    small = cv2.resize(patch, (max(1, patch.shape[1] // f), max(1, patch.shape[0] // f)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.6, blur / f))
    patch = cv2.resize(small, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_LINEAR)[pad:pad + h, pad:pad + w]
    patch = patch * (1 - tint_a) + np.float32(tint) / 255 * tint_a
    if sheen:
        patch = patch + np.linspace(sheen, 0, h, dtype=np.float32)[:, None, None]
    im = Image.fromarray((np.clip(patch, 0, 1) * 255).astype(np.uint8)).convert('RGBA')
    im.putalpha(_mask(w, h, r))
    if border > 0:
        im.alpha_composite(_rrect(w, h, r, outline=(255, 255, 255, int(255 * border)), width=max(1.2, w / 380)))
    return im


@lru_cache(maxsize=32)
def _spark(size, color=ORANGE):
    """Claude-style starburst."""
    lens = [1.0, 0.84, 0.97, 0.8, 1.0, 0.9, 0.83, 0.98, 0.82, 0.95, 0.87, 0.92]
    def f(d, k):
        c = size * k / 2; wd = size * k * 0.085
        for i, L in enumerate(lens):
            a = 2 * math.pi * i / len(lens) + 0.13
            ux, uy = math.cos(a), math.sin(a); px, py = -uy, ux
            r0 = c * 0.1; tip = c * 0.98 * L - wd / 2
            pts = [(c + ux * r0 + px * wd * 0.3, c + uy * r0 + py * wd * 0.3), (c + ux * tip + px * wd / 2, c + uy * tip + py * wd / 2),
                   (c + ux * tip - px * wd / 2, c + uy * tip - py * wd / 2), (c + ux * r0 - px * wd * 0.3, c + uy * r0 - py * wd * 0.3)]
            d.polygon(pts, fill=color + (255,))
            tx, ty = c + ux * tip, c + uy * tip
            d.ellipse((tx - wd / 2, ty - wd / 2, tx + wd / 2, ty + wd / 2), fill=color + (255,))
        d.ellipse((c - c * 0.17, c - c * 0.17, c + c * 0.17, c + c * 0.17), fill=color + (255,))
    return _aa(size, size, f, 4)


@lru_cache(maxsize=32)
def _check_dot(size, fill=GREEN, tick=WHITE):
    def f(d, k):
        s = size * k; d.ellipse((0, 0, s - 1, s - 1), fill=fill + (255,))
        d.line([(s * 0.27, s * 0.52), (s * 0.44, s * 0.68), (s * 0.74, s * 0.34)], fill=tick + (255,), width=int(s * 0.11), joint='curve')
        for (px, py) in [(s * 0.27, s * 0.52), (s * 0.74, s * 0.34)]:
            rr = s * 0.055; d.ellipse((px - rr, py - rr, px + rr, py + rr), fill=tick + (255,))
    return _aa(size, size, f)


def _spinner(size, angle, ring=(255, 255, 255, 60), arc=ORANGE + (255,), width=0.11):
    def f(d, k):
        s = size * k; w = int(s * width)
        d.ellipse((w / 2, w / 2, s - w / 2, s - w / 2), outline=ring, width=w)
        d.arc((w / 2, w / 2, s - w / 2, s - w / 2), angle, angle + 110, fill=arc, width=w)
    return _aa(size, size, f, 2)


def _circle_crop(rgb, cx, cy, side, size):
    """Round avatar cut from a float frame."""
    H, W = rgb.shape[:2]; h = side / 2; size = max(1, int(round(size)))
    x0, y0 = int(max(0, cx - h)), int(max(0, cy - h)); x1, y1 = int(min(W, cx + h)), int(min(H, cy + h))
    im = kit.to_pil(rgb[y0:y1, x0:x1]).resize((int(size), int(size)), Image.LANCZOS).convert('RGBA')
    im.putalpha(_aa(size, size, lambda d, k: d.ellipse((0, 0, size * k - 1, size * k - 1), fill=(255, 255, 255, 255))).getchannel('A'))
    return im


def _tx(img, xy, txt, fnt, fill, anchor='lm'):
    """Text with any alpha, composited (ImageDraw would punch a hole through an opaque panel)."""
    if len(fill) == 3 or fill[3] >= 255:
        ImageDraw.Draw(img).text(xy, txt, font=fnt, fill=tuple(fill[:3]) + (255,), anchor=anchor); return
    if fill[3] <= 0: return
    lay = Image.new('RGBA', img.size, tuple(fill[:3]) + (0,))
    ImageDraw.Draw(lay).text(xy, txt, font=fnt, fill=tuple(fill[:3]) + (255,), anchor=anchor)
    lay.putalpha(lay.getchannel('A').point(lambda v: v * fill[3] // 255)); img.alpha_composite(lay)


def _ln(img, pts, fill, width):
    lay = Image.new('RGBA', img.size, tuple(fill[:3]) + (0,))
    ImageDraw.Draw(lay).line(pts, fill=tuple(fill[:3]) + (255,), width=max(1, int(width)))
    lay.putalpha(lay.getchannel('A').point(lambda v: v * fill[3] // 255)); img.alpha_composite(lay)


def _put(s, im, cx, cy, scale=1.0, alpha=1.0, behind=False, rot=0.0):
    if im is None or alpha <= 0.003 or scale <= 0.02: return
    im = kit.scaled(im, scale)
    if abs(rot) > 0.05: im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
    (s.behind if behind else s.over)(im, cx, cy, 'c', kit.clamp(alpha))


def _pop(t, delay, fade=0.18):
    """Spring pop-in: (scale, alpha)."""
    tt = t - delay
    if tt <= 0: return 0.6, 0.0
    return 0.6 + 0.4 * kit.spring(tt, 2.2, 7.0), kit.clamp(tt / fade)


def _io(s, tin=0.55, tout=0.4):
    """Lands fast (expo) and leaves accelerating (cubic in): 0..1."""
    a = kit.expo_out(s.t / tin)
    b = 1 - kit.cubic_in(kit.clamp((s.t - (s.dur - tout)) / tout))
    return min(a, b)


def _warp(s, M, out_w=None, out_h=None):
    """Apply affine M to frame, mask and face (new arrays)."""
    W, H = out_w or s.W, out_h or s.H
    rgb = cv2.warpAffine(s.rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    al = cv2.warpAffine(s.alpha, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    f = s.face; z = M[0, 0]
    face = Face(z * f.cx + M[0, 2], z * f.cy + M[1, 2], f.w * z, f.h * z)
    return rgb, al, face


def _round_mask_full(W, H, rect, r):
    """Float HxW anti-aliased-ish rounded-rect mask."""
    x0, y0, x1, y1 = rect
    m = Image.new('L', (W, H), 0)
    ImageDraw.Draw(m).rounded_rectangle((x0, y0, x1, y1), max(0, r), fill=255)
    a = np.asarray(m, np.float32) / 255
    return cv2.GaussianBlur(a, (0, 0), 0.7)


def _soft_shadow_full(mask, blur, dy, strength):
    H, W = mask.shape
    sm = cv2.resize(mask, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), max(0.5, blur / 4))
    sh = cv2.resize(sm, (W, H), interpolation=cv2.INTER_LINEAR)
    sh = np.roll(sh, int(dy), axis=0)
    return np.clip(sh * strength, 0, 1)


def _range_faces(s, p):
    i0 = int(p.start * s.fps); i1 = int(p.end * s.fps)
    fs = s.ctx.faces[max(0, i0):max(i0 + 1, i1)]
    return fs or [s.face]


# =====================================================================
# 1. glass-widgets
# =====================================================================
@effect('glass-widgets', 'Claude At Work', 'frosted glass widgets pop in around your head: stats, render bar, now editing')
def glass_widgets(s, p):
    u, f, t = s.u, s.face, s.t
    bg = s.rgb.copy()
    big = p.text('48'); name = p.get('name', 'Keem'); handle = p.get('handle', '@keemfinity')
    simple = p.get('look') == 'simple'
    sp = min(1.0, s.dur / 4.0)
    out = 1 - kit.cubic_in(kit.clamp((t - (s.dur - 0.35)) / 0.35))

    avatar = s.once('avatar', lambda: _circle_crop(s.rgb, f.cx, f.cy - f.h * 0.05, f.w * 1.35, 84 * u))

    fx0, fx1 = s.once('fx', lambda: (f.x, f.x + f.w))
    fx0, fx1 = fx0 * 0.3 + f.x * 0.7, fx1 * 0.3 + (f.x + f.w) * 0.7   # steady columns, still follow big moves
    wl = kit.clamp(fx0 - 0.15 * f.w - 32 * u, 200 * u, 300 * u)
    wr = kit.clamp(s.W - 30 * u - (fx1 + 0.22 * f.w), 300 * u, 400 * u)
    xl = max(30 * u, fx0 - 0.15 * f.w - wl)                        # left column ends just at the head
    xr = min(s.W - 30 * u - wr, fx1 + 0.22 * f.w)
    top = max(96 * u, f.top - 0.32 * f.h)

    W = []   # (key, box, delay)
    W.append(('num', (xl, top, wl, 290 * u), 0.15))
    W.append(('render', (xr, top, wr, 140 * u), 0.3))
    if simple:
        W.append(('edit', (xr, top + 164 * u, wr, 190 * u), 0.45))
    else:
        W.append(('profile', (xr, top + 164 * u, wr, 130 * u), 0.45))
        W.append(('edit', (xr, top + 318 * u, wr, 190 * u), 0.6))
        W.append(('chart', (xl, top + 314 * u, wl, 270 * u), 0.75))

    white = WHITE + (255,)
    soft = WHITE + (205,)
    for n, (key, box, delay) in enumerate(W):
        delay *= sp
        sc, a = _pop(t, delay)
        a *= out; sc *= 0.92 + 0.08 * out
        if a <= 0.003: continue
        x, y, w, h = box
        if key in ('render', 'profile', 'edit'):          # keep clear of the head/beard edge (mask-measured)
            band = s.alpha[int(max(0, y)):int(max(1, y + 0.6 * h)), int(f.cx):]
            cols = np.nonzero((band > 0.5).any(0))[0]
            edge = f.cx + (cols.max() if len(cols) else 0)
            edge = max(edge, s.state.setdefault('edge_' + key, edge) * 0.6 + edge * 0.4)
            s.state['edge_' + key] = edge
            x = max(x, edge + 14 * u)
            w = max(300 * u, min(w, s.W - 30 * u - x)); w = round(w / (8 * u)) * 8 * u
            x = min(x, s.W - 30 * u - w)
        y += math.sin(2 * math.pi * (0.32 * t + n * 0.27)) * 7 * u
        x += math.sin(2 * math.pi * (0.21 * t + n * 0.41)) * 3 * u
        r = 30 * u
        panel = _glass(bg, (x, y, w, h), r, tint=(52, 50, 48), tint_a=0.34, blur=26 * u)
        d = ImageDraw.Draw(panel)
        lt = t - delay
        if key == 'num':
            d.ellipse((26 * u, 30 * u, 40 * u, 44 * u), fill=ORANGE + (255,))
            _tx(panel, (50 * u, 37 * u), 'Claude', _font('sans', 26 * u, 500), soft)
            v = kit.parse_number(big); k = kit.expo_out(lt / 0.9)
            txt = str(int(round(v * k))) if big.isdigit() else big
            d.text((24 * u, 168 * u), txt, font=_font('sans', 150 * u, 800), fill=white, anchor='ls')
            _tx(panel, (28 * u, 214 * u), 'effects', _font('sans', 28 * u, 400), soft)
            _tx(panel, (28 * u, 250 * u), 'one clip', _font('sans', 28 * u, 400), soft)
        elif key == 'render':
            pulse = 0.55 + 0.45 * (0.5 + 0.5 * math.cos(t * 2 * math.pi * 1.2))
            d.ellipse((26 * u, 30 * u, 40 * u, 44 * u), fill=tuple(int(kit.lerp(70, c, pulse)) for c in (255, 92, 72)) + (255,))
            d.text((50 * u, 37 * u), 'Rendering', font=_font('sans', 28 * u, 500), fill=white, anchor='lm')
            k = 0.99 * kit.cubic_out(kit.clamp((lt - 0.15) / max(0.5, s.dur - delay - 0.9)))
            bx, by, bw, bh = 26 * u, 66 * u, w - 52 * u, 14 * u
            panel.alpha_composite(_rrect(bw, bh, bh / 2, fill=(255, 255, 255, 60)), (int(bx), int(by)))
            fw = max(bh, bw * k)
            panel.alpha_composite(_rrect(fw, bh, bh / 2, fill=(255, 255, 255, 245)), (int(bx), int(by)))
            _tx(panel, (26 * u, 110 * u), f'4K render  ·  {int(round(k * 100))}%', _font('sans', 23 * u, 400), soft)
        elif key == 'profile':
            d.text((28 * u, 50 * u), name, font=_font('sans', 36 * u, 700), fill=white, anchor='lm')
            _tx(panel, (28 * u, 90 * u), handle, _font('sans', 24 * u, 400), soft)
            panel.alpha_composite(avatar, (int(w - 28 * u - avatar.width), int((h - avatar.height) / 2)))
        elif key == 'edit':
            ic = 84 * u
            icon = _rrect(ic, ic, 24 * u, fill=ORANGE + (255,)).copy()
            spk = _spark(int(56 * u), (252, 242, 232)); spin = math.sin(t * 1.3) * 10
            spk = spk.rotate(spin, resample=Image.BICUBIC)
            icon.alpha_composite(spk, (int((ic - spk.width) / 2), int((ic - spk.height) / 2)))
            panel.alpha_composite(icon, (int(24 * u), int(30 * u)))
            tx = 126 * u
            _tx(panel, (tx, 40 * u), 'NOW EDITING', _font('mono', 17 * u), WHITE + (180,))
            d.text((tx, 76 * u), 'soft UI pops', font=_font('sans', _fit_size(['soft UI pops'], 'sans', 32 * u, w - tx - 22 * u, 700), 700), fill=white, anchor='lm')
            _tx(panel, (tx, 110 * u), 'sfx  ·  25.7s', _font('sans', 22 * u, 400), WHITE + (185,))
            k = kit.clamp(0.25 + 0.7 * lt / max(1, s.dur - delay))
            lw = w - 48 * u - 70 * u
            panel.alpha_composite(_rrect(lw, 6 * u, 3 * u, fill=(255, 255, 255, 60)), (int(24 * u), int(152 * u)))
            panel.alpha_composite(_rrect(max(6 * u, lw * k), 6 * u, 3 * u, fill=(255, 255, 255, 235)), (int(24 * u), int(152 * u)))
            fx0, fy = w - 70 * u, 155 * u
            for o in (0, 18 * u):
                d.polygon([(fx0 + o, fy - 11 * u), (fx0 + o, fy + 11 * u), (fx0 + o + 18 * u, fy)], fill=white)
        elif key == 'chart':
            _tx(panel, (22 * u, 30 * u), 'artifact · views.html', _font('mono', 16 * u), WHITE + (190,))
            cx0, cy0, cw, ch = 16 * u, 52 * u, w - 32 * u, h - 68 * u
            panel.alpha_composite(_rrect(cw, ch, 18 * u, fill=(250, 247, 242, 250)), (int(cx0), int(cy0)))
            for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
                ox = cx0 + 22 * u + i * 20 * u
                d.ellipse((ox - 6 * u, cy0 + 16 * u, ox + 6 * u, cy0 + 28 * u), fill=c + (255,))
            d.text((cx0 + 18 * u, cy0 + 56 * u), 'Top reels', font=_font('sans', 25 * u, 700), fill=INK + (255,), anchor='lm')
            v = kit.expo_out((lt - 0.1) / 1.0)
            d.text((cx0 + 18 * u, cy0 + 96 * u), kit.number_text(197000 * v, '197K'), font=_font('sans', 42 * u, 800),
                   fill=ORANGE + (255,), anchor='lm')
            hs = [0.5, 0.78, 0.42, 1.0, 0.66]; cols = [(233, 184, 114), ORANGE, (233, 184, 114), (196, 92, 62), ORANGE]
            bw_ = (cw - 36 * u) / len(hs); base = cy0 + ch - 14 * u; maxh = ch - 136 * u
            for i, (hh, c) in enumerate(zip(hs, cols)):
                g = kit.back_out(kit.stagger(lt, i, 0.2, 0.08, 0.5), 1.2)
                bh_ = max(1, maxh * hh * g)
                bx = cx0 + 18 * u + i * bw_
                d.rounded_rectangle((bx, base - bh_, bx + bw_ * 0.72, base), radius=int(5 * u), fill=c + (255,))
        sh, pad = _shadow(int(w), int(h), int(r), int(22 * u), 0.38, int(10 * u))
        full = sh.copy(); full.alpha_composite(panel, (pad, pad))
        _put(s, full, x + w / 2, y + h / 2 + 0, sc, a, behind=(key != 'chart'))

    # bottom pill
    sc, a = _pop(t, 0.9 * sp); a *= out
    if a > 0.003:
        f1 = _font('sans', 40 * u, 500); f2 = _font('sans', 40 * u, 700)
        t1, t2 = 'Edited by ', 'Claude'
        tw = f1.getlength(t1) + f2.getlength(t2)
        pw, ph = tw + 92 * u, 98 * u
        px, py = s.W / 2 - pw / 2, s.H - 250 * u + math.sin(t * 2.1) * 4 * u
        pill = _glass(bg, (px, py, pw, ph), ph / 2, tint=(40, 38, 36), tint_a=0.42, blur=24 * u)
        d = ImageDraw.Draw(pill)
        d.text((46 * u, ph / 2), t1, font=f1, fill=WHITE + (255,), anchor='lm')
        d.text((46 * u + f1.getlength(t1), ph / 2), t2, font=f2, fill=(240, 150, 115, 255), anchor='lm')
        sh, pad = _shadow(int(pw), int(ph), int(ph / 2), int(20 * u), 0.4, int(10 * u))
        full = sh.copy(); full.alpha_composite(pill, (pad, pad))
        _put(s, full, s.W / 2, py + ph / 2, sc, a)


# =====================================================================
# 2. checklist
# =====================================================================
_CHECK_DEFAULT = ('Found your face::every frame|Cut you out of the room::AI mask|Put the text behind you::depth|'
                  'Graded the colour::warm|Rendered in 4K::2160p')


@effect('checklist', 'Claude At Work', 'Claude ticks off its edit list on a dark glass card under your chin')
def checklist(s, p):
    u, f, t = s.u, s.face, s.t
    steps = p.pairs('text', _CHECK_DEFAULT) or [('Done', '')]
    title = p.get('title', 'Claude is editing your clip')
    n = len(steps)
    cw = min(s.W - 80 * u, 900 * u); hh, rh = 108 * u, 90 * u
    ch = hh + n * rh + 24 * u
    k = _io(s, 0.55, 0.35)
    if k <= 0.003: return
    x = s.W / 2 - cw / 2
    y = kit.clamp(f.chin + 85 * u, 0, s.H - ch - 70 * u) + (1 - k) * 90 * u
    t0 = 0.5; per = max(0.25, (s.dur - t0 - 0.55) / n)
    done = sum(1 for i in range(n) if t >= t0 + i * per + 0.62 * per)

    card = _glass(s.rgb, (x, y, cw, ch), 34 * u, tint=(22, 22, 21), tint_a=0.8, blur=30 * u, border=0.16, sheen=0.04)
    d = ImageDraw.Draw(card)
    spk = _spark(int(40 * u)).rotate(-t * 40, resample=Image.BICUBIC)
    card.alpha_composite(spk, (int(34 * u), int(hh / 2 - spk.height / 2)))
    d.text((92 * u, hh / 2), title, font=_font('sans', 30 * u, 600), fill=(250, 249, 246, 255), anchor='lm')
    d.text((cw - 34 * u, hh / 2), f'{done} of {n}', font=_font('mono', 22 * u), fill=(170, 168, 162, 255), anchor='rm')
    lw = cw - 68 * u
    card.alpha_composite(_rrect(lw, 3 * u, 1.5 * u, fill=(255, 255, 255, 30)), (int(34 * u), int(hh - 2 * u)))
    prog = kit.clamp((done + (kit.clamp((t - t0 - done * per) / (0.62 * per)) if done < n else 0) * 0.999) / n)
    if prog > 0:
        card.alpha_composite(_rrect(max(3 * u, lw * prog), 3 * u, 1.5 * u, fill=ORANGE + (255,)), (int(34 * u), int(hh - 2 * u)))
    cs = 44 * u
    for i, (label, meta) in enumerate(steps):
        ry = hh + 12 * u + i * rh + rh / 2
        ra = kit.cubic_out(kit.stagger(t, i, 0.15, 0.07, 0.35))
        if ra <= 0: continue
        start = t0 + i * per; tick = start + 0.62 * per
        cx = 34 * u
        if t >= tick:
            g = kit.back_out((t - tick) / 0.3, 2.2)
            dot = kit.scaled(_check_dot(int(cs)), max(0.05, 0.4 + 0.6 * g))
            card.alpha_composite(dot, (int(cx + cs / 2 - dot.width / 2), int(ry - dot.height / 2)))
            la = 1.0
        elif t >= start:
            sp_ = _spinner(int(cs), (t - start) * 420 - 90)
            card.alpha_composite(sp_, (int(cx), int(ry - cs / 2)))
            la = 1.0
        else:
            card.alpha_composite(_spinner(int(cs), 0, ring=(255, 255, 255, 50), arc=(0, 0, 0, 0)), (int(cx), int(ry - cs / 2)))
            la = 0.42
        _tx(card, (cx + cs + 24 * u, ry), label, _font('sans', 31 * u, 500), (250, 249, 246, int(255 * la * ra)))
        if meta and t >= tick:
            m = kit.cubic_out((t - tick) / 0.35)
            _tx(card, (cw - 34 * u + (1 - m) * 20 * u, ry), meta, _font('mono', 22 * u), MINT + (int(255 * m),), 'rm')
        if i < n - 1:
            _ln(card, [(cx + cs + 24 * u, ry + rh / 2), (cw - 34 * u, ry + rh / 2)], (255, 255, 255, 22), max(1, u))
    sh, pad = _shadow(int(cw), int(ch), int(34 * u), int(30 * u), 0.5, int(16 * u))
    full = sh.copy(); full.alpha_composite(card, (pad, pad))
    s.over(full, x + cw / 2, y + ch / 2, 'c', alpha=k)


# =====================================================================
# 3. render-split
# =====================================================================
_RENDER_ROWS = 'Cut::12 jump cuts|Captions::synced|Grade::warm film|Effects::48 ready|Sound::-14 LUFS'
PURPLE = (59, 31, 74)
LAV = (214, 196, 232)


@effect('render-split', 'Claude At Work', 'top half: Claude renders it in a purple render window; you in a card below', aliases=('render',))
def render_split(s, p):
    u, t = s.u, s.t
    W, H = s.W, s.H
    k = _io(s, 0.65, 0.45)
    title = p.get('title', 'Claude'); sub = p.get('sub', 'renders it')
    rows = p.pairs('text', _RENDER_ROWS)
    caps = p.list('captions', 'one clip|48 effects|zero apps')
    wtitle = p.get('window', 'KEEMFINITY · RENDER')

    # --- video into its card ---
    cx0, cy0, cw, chh = 56 * u, H * 0.5 + 6 * u, W - 112 * u, H * 0.5 - 70 * u
    fs = _range_faces(s, p); fcy = float(np.mean([q.cy for q in fs]))
    z1 = cw / W * 1.0
    tx1 = cx0 + cw / 2 - z1 * W / 2; ty1 = cy0 + chh * 0.36 - z1 * fcy
    ty1 = min(ty1, cy0); ty1 = max(ty1, cy0 + chh - z1 * H)
    z = kit.lerp(1, z1, k); tx = kit.lerp(0, tx1, k); ty = kit.lerp(0, ty1, k)
    M = np.float32([[z, 0, tx], [0, z, ty]])
    rgb, al, face = _warp(s, M)
    rect = (kit.lerp(0, cx0, k), kit.lerp(0, cy0, k), kit.lerp(W, cx0 + cw, k), kit.lerp(H, cy0 + chh, k))
    m = _round_mask_full(W, H, tuple(int(round(v)) for v in rect), int(44 * u * k))
    canvas = np.empty_like(rgb); canvas[:] = np.float32(CREAM) / 255
    sh = _soft_shadow_full(m, 30 * u, 18 * u, 0.32 * k)
    canvas *= (1 - sh[..., None])
    s.rgb[:] = canvas * (1 - m[..., None]) + rgb * m[..., None]
    s.alpha = al * m; s.face = face
    if k <= 0.01: return

    # --- headline + window (slide down from the top) ---
    dy = -(1 - k) * H * 0.45
    head = s.once('head', lambda: _render_headline(title, sub, W - 120 * u, u))
    s.over(head, W / 2, 58 * u + dy, 'tc', alpha=k)
    wx, wy, ww, wh = 56 * u, 58 * u + head.height + 34 * u, W - 112 * u, 0
    wh = cy0 - 40 * u - wy
    win = _render_window(s, ww, wh, rows, caps, wtitle, u, t)
    sh2, pad = _shadow(int(ww), int(wh), int(30 * u), int(26 * u), 0.42, int(16 * u))
    full = sh2.copy(); full.alpha_composite(win, (pad, pad))
    s.over(full, wx + ww / 2, wy + wh / 2 + dy, 'c', alpha=k)


def _render_headline(title, sub, maxw, u):
    size = 118 * u
    while True:
        f1 = _font('serif', size, 700); f2 = _font('serif-italic', size)
        w = f1.getlength(title + ' ') + f2.getlength(sub)
        if w <= maxw or size < 30 * u: break
        size *= 0.95
    asc, desc = f1.getmetrics()
    im = Image.new('RGBA', (int(w + 20 * u), int((asc + desc) * 1.1)), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.text((10 * u, asc), title + ' ', font=f1, fill=INK + (255,), anchor='ls')
    d.text((10 * u + f1.getlength(title + ' '), asc), sub, font=f2, fill=ORANGE + (255,), anchor='ls')
    return im.crop(im.getbbox())


def _render_window(s, ww, wh, rows, caps, wtitle, u, t):
    st = s.once('win_static', lambda: _render_window_static(s, ww, wh, rows, caps, wtitle, u))
    win = st['img'].copy(); d = ImageDraw.Draw(win)
    n = len(rows)
    t0 = 0.55; per = max(0.2, (s.dur * 0.62 - t0) / max(1, n))
    for i, (lab, res) in enumerate(rows):
        ry = st['rows_y'] + i * st['rh'] + st['rh'] / 2
        start = t0 + i * per; tick = start + 0.7 * per
        cs = 34 * u
        if t >= tick:
            g = kit.back_out((t - tick) / 0.3, 2.2)
            dot = kit.scaled(_check_dot(int(cs), ORANGE), max(0.05, 0.4 + 0.6 * g))
            win.alpha_composite(dot, (int(36 * u + cs / 2 - dot.width / 2), int(ry - dot.height / 2)))
            mm = kit.cubic_out((t - tick) / 0.35)
            _tx(win, (ww - 36 * u, ry), res, _font('mono', 21 * u), LAV + (int(255 * mm),), 'rm')
        elif t >= start:
            win.alpha_composite(_spinner(int(cs), (t - start) * 420, ring=(255, 255, 255, 50)), (int(36 * u), int(ry - cs / 2)))
        else:
            win.alpha_composite(_spinner(int(cs), 0, ring=(255, 255, 255, 45), arc=(0, 0, 0, 0)), (int(36 * u), int(ry - cs / 2)))
        _tx(win, (36 * u + cs + 20 * u, ry), lab, _font('sans', 27 * u, 500), (255, 255, 255, 255 if t >= start else 120))
    # playhead over the timeline
    tl = st['tl']; x0, y0, x1, y1 = tl
    pr = kit.smooth(kit.clamp((t - 0.4) / max(0.5, s.dur - 0.9)))
    px = x0 + (x1 - x0) * pr
    # rendered part brighter: dim the unrendered region
    over = Image.new('RGBA', (int(x1 - px) + 1, int(y1 - y0)), (30, 14, 40, 120))
    if px < x1: win.alpha_composite(over, (int(px), int(y0)))
    d.line([(px, y0 - 14 * u), (px, y1)], fill=(255, 255, 255, 255), width=max(2, int(3 * u)))
    d.polygon([(px - 10 * u, y0 - 26 * u), (px + 10 * u, y0 - 26 * u), (px + 10 * u, y0 - 14 * u), (px, y0 - 4 * u), (px - 10 * u, y0 - 14 * u)],
              fill=ORANGE + (255,))
    # footer: progress + %
    fy = st['foot_y']; bw = ww - 72 * u - 120 * u
    win.alpha_composite(_rrect(bw, 10 * u, 5 * u, fill=(255, 255, 255, 40)), (int(36 * u), int(fy - 5 * u)))
    win.alpha_composite(_rrect(max(10 * u, bw * pr), 10 * u, 5 * u, fill=ORANGE + (255,)), (int(36 * u), int(fy - 5 * u)))
    d.text((ww - 36 * u, fy), f'{int(round(pr * 100))}%', font=_font('mono', 30 * u), fill=WHITE + (255,), anchor='rm')
    return win


def _render_window_static(s, ww, wh, rows, caps, wtitle, u):
    win = _rrect(ww, wh, 30 * u, fill=PURPLE + (255,)).copy(); d = ImageDraw.Draw(win)
    # subtle top light
    g = kit.gradient(ww, wh * 0.5, (255, 255, 255), (255, 255, 255), (16, 0))
    gm = Image.new('RGBA', win.size, (0, 0, 0, 0)); gm.paste(g, (0, 0))
    gm.putalpha(Image.fromarray(np.minimum(np.asarray(gm.getchannel("A")), np.asarray(_mask(gm.width, gm.height, int(30 * u))))))
    win.alpha_composite(gm)
    tb = 66 * u
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        ox = 36 * u + i * 24 * u
        d.ellipse((ox - 7 * u, tb / 2 - 7 * u, ox + 7 * u, tb / 2 + 7 * u), fill=c + (255,))
    d.text((120 * u, tb / 2), wtitle, font=_font('mono', 20 * u), fill=LAV + (255,), anchor='lm')
    _tx(win, (ww - 36 * u, tb / 2), '4K · 30FPS', _font('mono', 20 * u), LAV + (200,), 'rm')
    d.line([(0, tb), (ww, tb)], fill=(255, 255, 255, 30), width=max(1, int(u)))
    n = len(rows)
    foot_h = 64 * u
    avail = wh - tb - foot_h
    rh = min(52 * u, avail * 0.45 / max(1, n))
    rows_y = tb + 12 * u
    # timeline box
    ty0 = rows_y + n * rh + 20 * u; ty1 = wh - foot_h
    tlx0, tlx1 = 28 * u, ww - 28 * u
    win.alpha_composite(_rrect(tlx1 - tlx0, ty1 - ty0, 20 * u, fill=(38, 18, 50, 255)), (int(tlx0), int(ty0)))
    # ruler
    ix0, ix1 = tlx0 + 22 * u, tlx1 - 22 * u
    ry = ty0 + 34 * u
    for j in range(0, 41):
        xx = ix0 + (ix1 - ix0) * j / 40
        L = 12 * u if j % 5 == 0 else 6 * u
        d.line([(xx, ry - L), (xx, ry)], fill=(255, 255, 255, 90 if j % 5 == 0 else 45), width=max(1, int(u)))
        if j % 10 == 0:
            d.text((xx + 4 * u, ry - 20 * u), f'00:0{j // 10 * 2}', font=_font('mono', 14 * u), fill=LAV + (150,), anchor='lm')
    inner_h = ty1 - ry - 16 * u
    tr = inner_h / 4.4; gap = tr * 0.1
    ys = [ry + 10 * u + i * (tr + gap) for i in range(4)]
    # track 1: thumbnails
    thumbs = s.ctx.thumbs or []
    th = tr; ncols = max(1, int(round((ix1 - ix0) / (th * 0.62 + 3 * u))))
    tw = (ix1 - ix0 + 3 * u) / ncols - 3 * u
    for j in range(ncols):
        if not thumbs: break
        im = kit.card(thumbs[j % len(thumbs)], int(tw), int(th), r=6 * u)
        win.alpha_composite(im, (int(ix0 + j * (tw + 3 * u)), int(ys[0])))
    # track 2: caption pills
    ph = tr * 0.78; seg = (ix1 - ix0) / max(1, len(caps))
    for j, c in enumerate(caps):
        fnt = _font('sans', ph * 0.5, 600)
        pw = min(seg - 10 * u, fnt.getlength(c) + 30 * u)
        xx = ix0 + j * seg + 4 * u
        win.alpha_composite(_rrect(pw, ph, ph / 2, fill=(240, 150, 115, 255) if j % 2 == 0 else (196, 132, 236, 255)),
                            (int(xx), int(ys[1] + (tr - ph) / 2)))
        d.text((xx + pw / 2, ys[1] + tr / 2), c, font=fnt, fill=(30, 12, 36, 255), anchor='mm')
    # track 3: grade bar
    gh = tr * 0.5
    gb = np.zeros((int(gh), int(ix1 - ix0), 4), np.float32)
    xs = np.linspace(0, 1, gb.shape[1])[None, :, None]
    c1, c2, c3 = np.float32((70, 160, 190)), np.float32(ORANGE), np.float32((250, 214, 150))
    col = np.where(xs < 0.5, c1 + (c2 - c1) * (xs / 0.5), c2 + (c3 - c2) * ((xs - 0.5) / 0.5))
    gb[..., :3] = col; gb[..., 3] = 255
    gim = Image.fromarray(gb.astype(np.uint8), 'RGBA'); gim.putalpha(_mask(gim.width, gim.height, gh / 2))
    win.alpha_composite(gim, (int(ix0), int(ys[2] + (tr - gh) / 2)))
    # track 4: waveform
    rng = np.random.default_rng(11); nb = int((ix1 - ix0) / (6 * u))
    amp = np.abs(np.sin(np.linspace(0, 9, nb))) * 0.6 + rng.random(nb) * 0.4
    mid = ys[3] + tr / 2
    for j, a in enumerate(amp):
        xx = ix0 + j * 6 * u; hh_ = max(2 * u, a * tr * 0.45)
        d.rounded_rectangle((xx, mid - hh_, xx + 3 * u, mid + hh_), radius=int(1.5 * u), fill=LAV + (210,))
    return dict(img=win, rows_y=rows_y, rh=rh, tl=(ix0, ry + 4 * u, ix1, ty1 - 8 * u), foot_y=wh - foot_h / 2)


# =====================================================================
# 4. timeline
# =====================================================================
@effect('timeline', 'Claude At Work', 'red cinematic grade, an editing timeline with a glowing playhead, REC timecode')
def timeline(s, p):
    u, f, t = s.u, s.face, s.t
    W, H = s.W, s.H
    k = s.env(0.3, 0.3)
    s.tint(lambda rgb: kit.vignette(kit.tone(rgb, 'red') * 0.95, 0.5), alpha=0.88 * k)
    title = p.text('AUTO EDIT').upper(); sub = p.get('sub', 'made by Claude')

    # title behind the head
    big = s.once('title', lambda: _glow_text(title, kit.fit_font(title, 'display', W * 0.92, W * 0.34), (255, 236, 232), (255, 40, 60), 26 * u))
    kin = kit.expo_out(t / 0.5)
    ty = max(70 * u, f.crown - big.height * 0.05) - (1 - kin) * 60 * u
    s.behind(kit.scaled(big, 1.04 - 0.04 * kin + 0.02 * s.p), W / 2, ty, 'tc', alpha=k * kin)
    sb = s.once('sub', lambda: _glow_text(sub, _fit_font_px(sub, 'script', W - 54 * u - (f.x + f.w * 1.45), 66 * u), (255, 214, 214), (255, 60, 80), 12 * u))
    sa = kit.cubic_out((t - 0.35) / 0.5) * k
    s.behind(sb, W - 46 * u, ty + big.height * 0.74 + (1 - sa) * 20 * u, 'tr', alpha=sa)

    # REC + timecode
    tc0 = p.get('tc', '00:00:00:00')
    try:
        hh, mm, ss_, ff = [int(v) for v in str(tc0).split(':')]
    except ValueError:
        hh = mm = ss_ = ff = 0
    fr = int(round((hh * 3600 + mm * 60 + ss_) * 30 + ff + t * 30))
    tc = f'{fr // 108000:02d}:{fr // 1800 % 60:02d}:{fr // 30 % 60:02d}:{fr % 30:02d}'
    hud = Image.new('RGBA', (int(560 * u), int(70 * u)), (0, 0, 0, 0)); d = ImageDraw.Draw(hud)
    if (t % 1.0) < 0.62:
        d.ellipse((6 * u, 21 * u, 34 * u, 49 * u), fill=(255, 48, 64, 255))
    d.text((48 * u, 35 * u), 'REC', font=_font('mono', 30 * u), fill=(255, 255, 255, 255), anchor='lm')
    d.text((150 * u, 35 * u), tc, font=_font('mono', 30 * u), fill=(255, 225, 225, 255), anchor='lm')
    hs, off = kit.shadow(hud, int(8 * u), 0.5, (0, int(2 * u)))
    s.over(hs, 44 * u + off[0], 46 * u + off[1], 'tl', alpha=k)
    s.over(s.once('fps', lambda: kit.text_layer('4K · 30FPS', _font('mono', 22 * u), (255, 225, 225, 220))), W - 50 * u, 66 * u, 'tr', alpha=k)

    # timeline panel
    pw, ph = W - 80 * u, 360 * u
    px, py = 40 * u, H - ph - 60 * u + (1 - kit.expo_out(t / 0.6)) * 120 * u
    st = s.once('tl', lambda: _timeline_static(pw, ph, u))
    panel = _glass(s.rgb, (px, py, pw, ph), 28 * u, tint=(26, 6, 9), tint_a=0.62, blur=24 * u, border=0.18, sheen=0.03)
    panel.alpha_composite(st['ghost'])
    x0, x1 = st['x0'], st['x1']
    pr = kit.clamp((t - 0.25) / max(0.5, s.dur - 0.6))
    phx = x0 + (x1 - x0) * pr
    if phx > x0:
        sol = st['solid'].crop((0, 0, int(phx), st['solid'].height))
        panel.alpha_composite(sol)
    # glowing playhead
    gl = Image.new('RGBA', panel.size, (0, 0, 0, 0)); gd = ImageDraw.Draw(gl)
    gd.line([(phx, st['ry'] - 22 * u), (phx, ph - 18 * u)], fill=(255, 120, 140, 255), width=int(10 * u))
    gl = kit.blurred(gl, 9 * u)
    panel.alpha_composite(gl); panel.alpha_composite(gl)
    d = ImageDraw.Draw(panel)
    d.line([(phx, st['ry'] - 22 * u), (phx, ph - 18 * u)], fill=(255, 240, 242, 255), width=max(2, int(3 * u)))
    d.polygon([(phx - 12 * u, st['ry'] - 34 * u), (phx + 12 * u, st['ry'] - 34 * u), (phx + 12 * u, st['ry'] - 20 * u),
               (phx, st['ry'] - 8 * u), (phx - 12 * u, st['ry'] - 20 * u)], fill=(255, 240, 242, 255))
    s.over(panel, px, py, 'tl', alpha=k)


def _fit_font_px(txt, role, maxw, start):
    size = start
    while size > 20 and _font(role, size).getlength(txt) > maxw: size *= 0.95
    return _font(role, size)


def _glow_text(txt, fnt, fill, glow_col, radius):
    im = kit.text_layer(txt, fnt, fill=tuple(fill) + (255,))
    g, off = kit.glow(im, int(max(2, radius)), 1.1, glow_col)
    return g


def _timeline_static(pw, ph, u):
    ghost = Image.new('RGBA', (int(pw), int(ph)), (0, 0, 0, 0)); solid = ghost.copy()
    dg, ds = ImageDraw.Draw(ghost), ImageDraw.Draw(solid)
    lab = 70 * u; x0, x1 = 24 * u + lab, pw - 24 * u
    ry = 60 * u
    for j in range(0, 49):
        xx = x0 + (x1 - x0) * j / 48
        L = 14 * u if j % 6 == 0 else 7 * u
        dg.line([(xx, ry - L), (xx, ry)], fill=(255, 220, 225, 110 if j % 6 == 0 else 55), width=max(1, int(u)))
        if j % 12 == 0:
            dg.text((xx + 5 * u, ry - 26 * u), f'00:{j // 12 * 2:02d}', font=_font('mono', 15 * u), fill=(255, 200, 205, 150), anchor='lm')
    rng = np.random.default_rng(7)
    tr, gap = 70 * u, 14 * u
    cols = [((232, 40, 58), (180, 20, 40)), ((255, 110, 140), (220, 60, 100)), ((140, 24, 36), (100, 14, 26))]
    for r_, nm in enumerate(['V1', 'V2', 'A1']):
        y0 = ry + 22 * u + r_ * (tr + gap)
        dg.text((24 * u + lab / 2 - 6 * u, y0 + tr / 2), nm, font=_font('mono', 20 * u), fill=(255, 200, 205, 170), anchor='mm')
        xx = x0
        while xx < x1 - 20 * u:
            w = float(rng.uniform(90, 250)) * u
            w = min(w, x1 - xx)
            if r_ == 1 and rng.random() < 0.35: xx += w * 0.6; continue
            c_top, c_bot = cols[r_]
            g = kit.gradient(w, tr, c_top, c_bot); g.putalpha(_mask(int(w), int(tr), int(12 * u)))
            dsh = ImageDraw.Draw(g)
            if r_ == 2:     # waveform inside audio clips
                m_ = tr / 2
                for q in range(int(w / (5 * u))):
                    a = float(rng.uniform(0.15, 0.9)) * tr * 0.38
                    dsh.line([(q * 5 * u + 4 * u, m_ - a), (q * 5 * u + 4 * u, m_ + a)], fill=(255, 170, 180, 200), width=max(1, int(2 * u)))
            else:
                dsh.rounded_rectangle((6 * u, 6 * u, min(w - 6 * u, 6 * u + w * 0.3), 14 * u), radius=int(4 * u), fill=(255, 255, 255, 120))
            solid.alpha_composite(g, (int(xx), int(y0)))
            gh = _rrect(w, tr, 12 * u, fill=(255, 90, 110, 26), outline=(255, 120, 140, 90), width=1.5 * u)
            ghost.alpha_composite(gh, (int(xx), int(y0)))
            xx += w + 6 * u
    return dict(ghost=ghost, solid=solid, x0=x0, x1=x1, ry=ry)


# =====================================================================
# 5. apple-toggles
# =====================================================================
@effect('apple-toggles', 'Claude At Work', 'an iOS settings card behind your head: toggles flip green one by one')
def apple_toggles(s, p):
    u, t = s.u, s.t
    s.camera(1 + 0.06 * kit.smooth(s.p))
    f = s.face
    rows = p.list('text', 'One clip|Type /effects|Zero editing apps')
    n = len(rows)
    on = p.color('color', GREEN)
    below = p.get('pos') == 'below'
    ticks = []
    tk = p.list('ticks')
    if tk:
        for v in tk:
            try: ticks.append(float(v) - p.start)
            except ValueError: pass
    step = min(0.65, max(0.25, (s.dur - 1.1) / max(1, n)))
    while len(ticks) < n: ticks.append(0.6 + len(ticks) * step)

    k = _io(s, 0.6, 0.35)
    if k <= 0.003: return
    rh = 108 * u
    if below:
        cw = min(s.W - 100 * u, 820 * u); cx = s.W / 2 - cw / 2
        ch = n * rh + 16 * u
        cy = kit.clamp(f.chin + 60 * u, 0, s.H - ch - 80 * u)
        label_max = cw - 240 * u
    else:
        cw = s.W - 80 * u; cx = 40 * u
        ch = n * rh + 16 * u
        cy = max(56 * u, f.crown - 40 * u)
        label_max = max(150 * u, f.x - 0.14 * f.w - 16 * u - (cx + 108 * u))
    fsz = s.once('fsz', lambda: _fit_size(rows, 'sans', 34 * u, label_max, 600))
    cy += -(1 - k) * 110 * u
    card = _glass(s.rgb, (cx, cy, cw, ch), 38 * u, tint=(250, 250, 252), tint_a=0.84, blur=30 * u, border=0.6, sheen=0.0)
    d = ImageDraw.Draw(card)
    icons = [((255, 149, 0), '1'), ((0, 122, 255), '/'), ((255, 59, 48), '0'), ((88, 86, 214), '+'), ((52, 199, 89), '*')]
    tw_, th_ = 104 * u, 62 * u
    for i, lab in enumerate(rows):
        ry = 8 * u + i * rh + rh / 2
        col, g = icons[i % len(icons)]
        ic = 54 * u
        icon = _rrect(ic, ic, 13 * u, fill=col + (255,)).copy()
        ImageDraw.Draw(icon).text((ic / 2, ic / 2), g, font=_font('sans', 32 * u, 700), fill=(255, 255, 255, 255), anchor='mm')
        card.alpha_composite(icon, (int(30 * u), int(ry - ic / 2)))
        d.text((30 * u + ic + 24 * u, ry), lab, font=_font('sans', fsz, 600), fill=(17, 17, 19, 255), anchor='lm')
        if i < n - 1:
            d.line([(30 * u + ic + 24 * u, ry + rh / 2), (cw, ry + rh / 2)], fill=(60, 60, 67, 45), width=max(1, int(1.2 * u)))
        # toggle
        g_ = kit.expo_out((t - ticks[i]) / 0.32) if t >= ticks[i] else 0.0
        trk_col = tuple(int(kit.lerp(a, b, g_)) for a, b in zip((226, 226, 230), on))
        tx = cw - 34 * u - tw_; ty = ry - th_ / 2
        card.alpha_composite(_rrect(tw_, th_, th_ / 2, fill=trk_col + (255,)), (int(tx), int(ty)))
        kn = th_ - 8 * u
        stretch = 1 + 0.18 * math.sin(math.pi * kit.clamp((t - ticks[i]) / 0.28)) if t >= ticks[i] else 1
        kw = kn * stretch
        kx = tx + 4 * u + (tw_ - 8 * u - kw) * g_
        ksh, kpad = _shadow(int(kw), int(kn), int(kn / 2), int(5 * u), 0.3, int(3 * u))
        card.alpha_composite(ksh, (int(kx - kpad), int(ty + 4 * u - kpad)))
        card.alpha_composite(_rrect(kw, kn, kn / 2, fill=(255, 255, 255, 255)), (int(kx), int(ty + 4 * u)))
    sh, pad = _shadow(int(cw), int(ch), int(38 * u), int(30 * u), 0.35, int(14 * u))
    full = sh.copy(); full.alpha_composite(card, (pad, pad))
    (s.over if below else s.behind)(full, cx + cw / 2, cy + ch / 2, 'c', alpha=k)


def _fit_size(lines, role, start, maxw, weight=None):
    size = start
    while size > 12 and max(_font(role, size, weight).getlength(x) for x in lines) > maxw: size *= 0.95
    return size


# =====================================================================
# 6. the-bill
# =====================================================================
_BILL = ('KEEMFINITY EFFECTS|ORDER #048 · 1 CLIP|---|Cut::$0|Captions::$0|Colour grade::$0|48 effects::$0|'
         'Editing apps::$0|---|TOTAL::$0.00|---|THANK YOU')
PAPER = (248, 246, 240)


@effect('the-bill', 'Claude At Work', 'a thermal receipt prints out behind your head: everything costs $0', aliases=('bill',))
def the_bill(s, p):
    u, f, t = s.u, s.face, s.t
    lines = p.list('text', _BILL)
    side = -1 if p.get('side') == 'left' else 1
    hx0, hx1 = s.once('hx', lambda: (f.x - 0.2 * f.w, f.x + f.w * 1.2))
    room = (s.W - 26 * u - hx1) if side > 0 else (hx0 - 26 * u)
    rw = s.once('rw', lambda: kit.clamp(room, 330 * u, 440 * u))
    st = s.once('bill', lambda: _bill_static(lines, rw, u))
    full = st['img']; FH = full.height
    out = 1 - kit.cubic_in(kit.clamp((t - (s.dur - 0.35)) / 0.35))
    # printing: the strip grows in line-steps (feeds fast, pauses briefly)
    pr = kit.clamp((t - 0.15) / max(0.6, s.dur * 0.62))
    v = pr * FH
    step = st['lh']; base = math.floor(v / step) * step; fr = (v - base) / step
    v = min(FH, base + step * kit.smooth(min(1, fr * 1.6)))
    v = max(2, int(v))
    strip = full.crop((0, 0, full.width, v)).copy()
    # jagged tear edge at the current bottom
    z = st['zig']
    a = np.asarray(strip.getchannel('A')).copy()
    zh = z.shape[0]
    if v > zh:
        a[v - zh:v] = np.minimum(a[v - zh:v], z)
    strip.putalpha(Image.fromarray(a))
    # shadow + sway
    sh, off = kit.shadow(strip, int(14 * u), 0.35, (int(6 * u), int(10 * u)))
    rot = -2.5 * side + math.sin(t * 1.7) * 0.7
    sh = sh.rotate(rot, resample=Image.BICUBIC, expand=True)
    cx = (hx1 + 6 * u + rw * 0.5) if side > 0 else (hx0 - 6 * u - rw * 0.5)
    cx += (f.cx - s.once('cx0', lambda: f.cx)) * 0.6
    cx = kit.clamp(cx, rw * 0.5 + 16 * u, s.W - rw * 0.5 - 16 * u)
    top = -24 * u; th = math.radians(rot)
    s.behind(sh, cx + v / 2 * math.sin(th), top + v / 2 * math.cos(th), 'c', alpha=out)
    # printer slot
    slot = s.once('slot', lambda: _bill_slot(rw, u))
    s.behind(slot, cx, 0, 'tc', alpha=out)


def _bill_slot(rw, u):
    w, h = rw + 60 * u, 30 * u
    im = kit.gradient(w, h, (40, 40, 42), (14, 14, 15)); im.putalpha(_mask(int(w), int(h), int(10 * u)))
    sh, off = kit.shadow(im, int(10 * u), 0.5, (0, int(6 * u)))
    return sh.crop((0, -off[1], sh.width, sh.height))


def _bill_static(lines, rw, u):
    fm = _font('mono', 25 * u); fh = _font('mono', 31 * u); ft = _font('mono', 34 * u)
    lh = 46 * u; padx = 34 * u
    rows = []
    for i, ln in enumerate(lines):
        if ln.strip('-') == '' and ln: rows.append(('rule', None)); continue
        if '::' in ln:
            a, b = ln.split('::', 1)
            rows.append(('total' if a.strip().upper().startswith('TOTAL') else 'item', (a.strip(), b.strip())))
        else:
            rows.append(('head' if i == 0 else 'center', ln))
    H = int(84 * u + lh * len(rows) + 50 * u)
    im = Image.new('RGBA', (int(rw), H), PAPER + (255,)); d = ImageDraw.Draw(im)
    # faint paper grain
    rng = np.random.default_rng(3)
    grain = (rng.random((H, int(rw))) * 10 - 5).astype(np.float32)
    arr = np.asarray(im, np.float32).copy(); arr[..., :3] += grain[..., None]; im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    ink = (34, 32, 30, 255)
    y = 74 * u
    for kind, val in rows:
        cy = y + lh / 2
        if kind == 'rule':
            xx = padx
            while xx < rw - padx:
                d.line([(xx, cy), (xx + 10 * u, cy)], fill=(34, 32, 30, 170), width=max(1, int(2 * u))); xx += 18 * u
        elif kind == 'head':
            d.text((rw / 2, cy), val, font=kit.fit_font(val, 'mono', rw - 2 * padx, fh.size) if fh.getlength(val) > rw - 2 * padx else fh,
                   fill=ink, anchor='mm')
        elif kind == 'center':
            d.text((rw / 2, cy), val, font=fm, fill=ink, anchor='mm')
        else:
            fnt = ft if kind == 'total' else fm
            a, b = val
            wa, wb = fnt.getlength(a), fnt.getlength(b)
            d.text((padx, cy), a, font=fnt, fill=ink, anchor='lm')
            d.text((rw - padx, cy), b, font=fnt, fill=ink, anchor='rm')
            dots_x0, dots_x1 = padx + wa + 10 * u, rw - padx - wb - 10 * u
            xx = dots_x0
            while xx < dots_x1:
                d.ellipse((xx, cy + 6 * u, xx + 3 * u, cy + 9 * u), fill=(34, 32, 30, 150)); xx += 11 * u
        y += lh
    # zigzag tear mask
    zh = max(4, int(16 * u)); tooth = 22 * u
    zm = Image.new('L', (int(rw) * 3, zh * 3), 255); zd = ImageDraw.Draw(zm)
    pts = [(0, zh * 3)]
    xx = 0.0
    while xx <= rw * 3:
        pts += [(xx, zh * 3), (xx + tooth * 1.5, 0)]; xx += tooth * 3
    pts += [(rw * 3, zh * 3)]
    zd.polygon(pts + [(rw * 3, zh * 3 + 5)], fill=0)
    zm = zm.resize((int(rw), zh), Image.LANCZOS)
    return dict(img=im, lh=lh, zig=np.asarray(zm))


# =====================================================================
# 7. reel-frame
# =====================================================================
@effect('reel-frame', 'Your Reels', 'your video shrinks into an Instagram Reel card with likes, comments and Follow')
def reel_frame(s, p):
    u, t = s.u, s.t
    W, H = s.W, s.H
    handle = p.text('@keemfinity')
    if not handle.startswith('@'): handle = '@' + handle
    k = _io(s, 0.6, 0.4)
    avatar = s.once('avatar', lambda: _circle_crop(s.rgb, s.face.cx, s.face.cy - s.face.h * 0.05, s.face.w * 1.35, 64 * u))
    # blurred dark backdrop of itself
    small = cv2.resize(s.rgb, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 6)
    bgimg = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR) * 0.5
    S = 0.8
    z = kit.lerp(1, S, k); cyc = H / 2 - 20 * u * k
    M = np.float32([[z, 0, W / 2 - z * W / 2], [0, z, cyc - z * H / 2]])
    rgb, al, face = _warp(s, M)
    rect = (W / 2 - z * W / 2, cyc - z * H / 2, W / 2 + z * W / 2, cyc + z * H / 2)
    m = _round_mask_full(W, H, tuple(int(round(v)) for v in rect), int(46 * u * k))
    sh = _soft_shadow_full(m, 36 * u, 20 * u, 0.6 * k)
    canvas = bgimg * (1 - sh[..., None])
    canvas = kit.lerp(rgb, canvas, k) if k < 1 else canvas
    s.rgb[:] = canvas * (1 - m[..., None]) + rgb * m[..., None]
    s.alpha = al * m; s.face = face
    ua = kit.clamp((k - 0.5) / 0.5)
    if ua <= 0.01: return
    cw, ch = W * S, H * S
    ui = s.once('ui', lambda: _reel_ui(cw, ch, handle, avatar, u))
    lay = ui.copy(); d = ImageDraw.Draw(lay)
    # heart tap: fills red with a pop
    th = min(1.3, s.dur * 0.4)
    hx, hy = cw - 62 * u, ch - 590 * u
    if t >= th:
        g = kit.spring(t - th, 2.2, 6.5)
        hs = 1 + 0.45 * (1 - g) if t - th > 0.02 else 1.4
        heart = kit.scaled(_heart(int(58 * u), (255, 48, 64), filled=True), max(0.1, hs * min(1, (t - th) / 0.08 + 0.3)))
        lay.alpha_composite(heart, (int(hx - heart.width / 2), int(hy - heart.height / 2)))
    # scrubber
    d.rectangle((0, ch - 5 * u, cw, ch), fill=(255, 255, 255, 60))
    d.rectangle((0, ch - 5 * u, cw * (0.08 + 0.9 * s.p), ch), fill=(255, 255, 255, 230))
    lay.putalpha(Image.fromarray(np.minimum(np.asarray(lay.getchannel('A')), np.asarray(_mask(int(cw), int(ch), int(46 * u))))))
    lay = kit.scaled(lay, z / S)
    s.over(lay, W / 2, cyc, 'c', alpha=ua)


@lru_cache(maxsize=8)
def _heart(size, color=(255, 255, 255), filled=False, width=0.09):
    def f(d, k):
        s_ = size * k
        pts = []
        for i in range(200):
            a = 2 * math.pi * i / 200
            x = 16 * math.sin(a) ** 3
            y = -(13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a))
            pts.append((s_ / 2 + x * s_ / 36, s_ * 0.47 + y * s_ / 36))
        if filled: d.polygon(pts, fill=color + (255,))
        else: d.line(pts + [pts[0]], fill=color + (255,), width=int(s_ * width), joint='curve')
    return _aa(size, size, f)


def _reel_ui(cw, ch, handle, avatar, u):
    lay = Image.new('RGBA', (int(cw), int(ch)), (0, 0, 0, 0))
    # bottom + top gradients for legibility
    gb = kit.gradient(cw, 520 * u, (0, 0, 0), (0, 0, 0), (0, 170)); lay.alpha_composite(gb, (0, int(ch - 520 * u)))
    gt = kit.gradient(cw, 180 * u, (0, 0, 0), (0, 0, 0), (110, 0)); lay.alpha_composite(gt, (0, 0))
    fg = Image.new('RGBA', lay.size, (0, 0, 0, 0)); d = ImageDraw.Draw(fg)
    white = (255, 255, 255, 255)
    d.text((32 * u, 56 * u), 'Reels', font=_font('sans', 40 * u, 700), fill=white, anchor='lm')
    cxp = 32 * u + _font('sans', 40 * u, 700).getlength('Reels') + 22 * u
    d.line([(cxp, 50 * u), (cxp + 9 * u, 60 * u), (cxp + 18 * u, 50 * u)], fill=white, width=int(4 * u), joint='curve')
    # camera icon top right
    cx_, cy_ = cw - 58 * u, 56 * u
    d.rounded_rectangle((cx_ - 22 * u, cy_ - 16 * u, cx_ + 22 * u, cy_ + 18 * u), radius=int(8 * u), outline=white, width=int(3.5 * u))
    d.ellipse((cx_ - 9 * u, cy_ - 7 * u, cx_ + 9 * u, cy_ + 11 * u), outline=white, width=int(3.5 * u))
    # bottom-left: avatar, handle, follow
    ay = ch - 200 * u
    ring = _aa(74 * u, 74 * u, lambda dd, k: dd.ellipse((0, 0, 74 * u * k - 1, 74 * u * k - 1), outline=(255, 255, 255, 255), width=int(3 * u * k)))
    fg.alpha_composite(avatar, (int(32 * u), int(ay - avatar.height / 2)))
    fg.alpha_composite(ring, (int(32 * u - 5 * u), int(ay - 37 * u)))
    fb = _font('sans', 30 * u, 700)
    hx = 32 * u + avatar.width + 20 * u
    d.text((hx, ay), handle, font=fb, fill=white, anchor='lm')
    fx = hx + fb.getlength(handle) + 22 * u; ff = _font('sans', 26 * u, 600)
    fw = ff.getlength('Follow') + 36 * u
    fg.alpha_composite(_rrect(fw, 50 * u, 12 * u, outline=(255, 255, 255, 235), width=2.5 * u), (int(fx), int(ay - 25 * u)))
    d.text((fx + fw / 2, ay), 'Follow', font=ff, fill=white, anchor='mm')
    d.text((32 * u, ch - 128 * u), 'one clip · 48 effects · zero apps', font=_font('sans', 27 * u, 400), fill=white, anchor='lm')
    # music line
    my = ch - 76 * u
    d.ellipse((32 * u, my + 2 * u, 44 * u, my + 12 * u), fill=white)
    d.line([(43 * u, my + 7 * u), (43 * u, my - 12 * u), (52 * u, my - 8 * u)], fill=white, width=int(3 * u))
    d.text((62 * u, my), f'Original audio · {handle.lstrip("@")}', font=_font('sans', 24 * u, 500), fill=(255, 255, 255, 225), anchor='lm')
    # right rail
    rx = cw - 62 * u; fs = _font('sans', 22 * u, 600)
    hy = ch - 590 * u
    fg.alpha_composite(_heart(int(58 * u)), (int(rx - 29 * u), int(hy - 29 * u)))
    d.text((rx, hy + 50 * u), '197K', font=fs, fill=white, anchor='mm')
    cy2 = ch - 460 * u
    bub = _aa(56 * u, 56 * u, lambda dd, k: (dd.ellipse((4 * u * k, 4 * u * k, 52 * u * k, 50 * u * k), outline=(255, 255, 255, 255), width=int(5 * u * k)),
                                             dd.polygon([(6 * u * k, 54 * u * k), (14 * u * k, 38 * u * k), (24 * u * k, 46 * u * k)], fill=(255, 255, 255, 255))))
    fg.alpha_composite(bub, (int(rx - 28 * u), int(cy2 - 28 * u)))
    d.text((rx, cy2 + 50 * u), '2,431', font=fs, fill=white, anchor='mm')
    cy3 = ch - 330 * u
    plane = _aa(56 * u, 56 * u, lambda dd, k: (dd.line([(4 * u * k, 26 * u * k), (52 * u * k, 6 * u * k), (34 * u * k, 52 * u * k), (24 * u * k, 32 * u * k),
                                                         (4 * u * k, 26 * u * k)], fill=(255, 255, 255, 255), width=int(5 * u * k), joint='curve'),
                                               dd.line([(24 * u * k, 32 * u * k), (52 * u * k, 6 * u * k)], fill=(255, 255, 255, 255), width=int(4 * u * k))))
    fg.alpha_composite(plane, (int(rx - 28 * u), int(cy3 - 28 * u)))
    d.text((rx, cy3 + 50 * u), '18.4K', font=fs, fill=white, anchor='mm')
    for j in range(3):
        d.ellipse((rx - 4 * u, ch - 210 * u + j * 14 * u - 4 * u, rx + 4 * u, ch - 210 * u + j * 14 * u + 4 * u), fill=white)
    sh, off = kit.shadow(fg, int(6 * u), 0.45, (0, int(2 * u)))
    lay.alpha_composite(sh.crop((-off[0], -off[1], -off[0] + lay.width, -off[1] + lay.height)))
    return lay


# =====================================================================
# 8. broll-split
# =====================================================================
@effect('broll-split', 'Your Reels', 'split screen: b-roll on top with a price-tag sticker, you centred below')
def broll_split(s, p):
    u, t = s.u, s.t
    W, H = s.W, s.H
    k = _io(s, 0.55, 0.4)
    if k <= 0.003: return
    tag = p.text('4 hrs → 4 min').replace('->', '→')
    img = p.images(1)[0]
    half = H / 2
    # bottom: you, face centred in the bottom half
    fs = _range_faces(s, p)
    min_cy = min(q.cy for q in fs)
    z1 = max(1.0, (half / 2 + 10 * u) / max(1.0, min_cy))
    f = s.face
    tx1 = W / 2 - z1 * f.cx; ty1 = half + half / 2 - z1 * f.cy
    tx1 = kit.clamp(tx1, W - z1 * W, 0)
    z = kit.lerp(1, z1, k); tx = kit.lerp(0, tx1, k); ty = kit.lerp(0, ty1, k)
    M = np.float32([[z, 0, tx], [0, z, ty]])
    rgb, al, face = _warp(s, M)
    # top: b-roll with a Ken Burns push
    hs = int(round(half))
    kb = 1.06 + 0.1 * s.p
    base = s.once('broll', lambda: kit.to_np(kit.card(img, W, hs, r=0)))
    Mk = cv2.getRotationMatrix2D((W * 0.5 + 30 * u * (s.p - 0.5), hs * 0.5), 0, kb)
    top = cv2.warpAffine(base, Mk, (W, hs), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    yoff = int(round(-(1 - k) * hs))
    out = rgb.copy()
    y0 = max(0, yoff); y1 = max(0, min(H, yoff + hs))
    if y1 > y0:
        out[y0:y1] = top[y0 - yoff:y1 - yoff]
        al = al.copy(); al[y0:y1] = 0
    # seam line
    sy = yoff + hs
    if 0 < sy < H:
        th = max(1, int(3 * u))
        out[max(0, sy - th):sy + th] = out[max(0, sy - th):sy + th] * 0.2 + 0.8
    s.rgb[:] = out; s.alpha = al; s.face = face
    # sticker
    st = s.once('tag', lambda: _price_tag(tag, u))
    sc, a = _pop(t, 0.45 * min(1, s.dur / 3))
    a *= kit.clamp(k)
    rot = -8 + (1 - kit.clamp((t - 0.45) / 0.5)) * 14 + math.sin(t * 1.6) * 1.2
    _put(s, st, W * 0.64, half - 120 * u + yoff + hs - hs, sc, a, rot=rot)


def _price_tag(tag, u):
    parts = tag.split('→') if '→' in tag else [tag]
    fa = _font('sans', 54 * u, 800)
    if len(parts) == 2:
        a, b = parts[0].strip(), parts[1].strip()
        segs = [(a, True), ('→', False), (b, False)]
    else:
        segs = [(tag, False)]
    gap = 18 * u
    tw = sum(fa.getlength(x) for x, _ in segs) + gap * (len(segs) - 1)
    hole = 70 * u
    w, h = tw + hole + 60 * u, 110 * u
    def body(d, k):
        pts = [(0, h * k / 2), (hole * 0.7 * k, 0), (w * k, 0), (w * k, h * k), (hole * 0.7 * k, h * k)]
        d.polygon(pts, fill=(255, 214, 64, 255))
        r = 12 * u * k; cx, cy = hole * 0.62 * k, h * k / 2
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 0, 0, 0))
    im = _aa(w, h, body)
    # rounded corners on the body
    m = np.asarray(im.getchannel('A')).copy()
    rr = _mask(int(w), int(h), int(16 * u)); m = np.minimum(m, np.asarray(rr))
    im.putalpha(Image.fromarray(m))
    d = ImageDraw.Draw(im)
    x = hole + 26 * u; cy = h / 2
    for txt, strike in segs:
        col = (120, 92, 20, 255) if strike else (24, 20, 10, 255)
        d.text((x, cy), txt, font=fa, fill=col, anchor='lm')
        tw_ = fa.getlength(txt)
        if strike:
            d.line([(x - 4 * u, cy + 2 * u), (x + tw_ + 4 * u, cy - 4 * u)], fill=(226, 52, 40, 255), width=int(6 * u))
        x += tw_ + gap
    # string through the hole
    sh, off = kit.shadow(im, int(12 * u), 0.45, (int(4 * u), int(10 * u)))
    return sh
