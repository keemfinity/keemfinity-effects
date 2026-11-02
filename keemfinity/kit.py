"""Keemfinity toolkit: motion curves, type, shapes, glass and image helpers shared by every effect.

Images are PIL RGBA unless a name says otherwise. Sizes are in output pixels; use `Shot.u` (output width / 1080)
to scale anything designed at 1080 wide.
"""
import math, os
from functools import lru_cache
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, 'fonts')


# ---------- motion ----------
def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    return a + (b - a) * t


def expo_out(t):
    t = clamp(t); return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


def cubic_out(t):
    t = clamp(t); return 1 - (1 - t) ** 3


def cubic_in(t):
    t = clamp(t); return t ** 3


def smooth(t):
    t = clamp(t); return t * t * (3 - 2 * t)


def back_out(t, k=1.6):
    t = clamp(t) - 1; return 1 + t * t * ((k + 1) * t + k)


def spring(t, freq=4.5, decay=6.0):
    """0 -> 1 with a damped overshoot."""
    if t <= 0: return 0.0
    return 1 - math.exp(-decay * t) * math.cos(freq * 2 * math.pi * t * 0.5)


def window(t, start, length):
    """Progress 0..1 of a sub-animation that starts at `start` and lasts `length` seconds."""
    return clamp((t - start) / max(1e-6, length))


def stagger(t, i, start, each, length):
    return window(t, start + i * each, length)


# ---------- colour ----------
def hex_rgb(h, default=(255, 255, 255)):
    if not h: return default
    if isinstance(h, (tuple, list)): return tuple(int(v) for v in h[:3])
    h = str(h).lstrip('#')
    if len(h) == 3: h = ''.join(c * 2 for c in h)
    try: return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError: return default


def with_alpha(rgb, a):
    return tuple(rgb[:3]) + (int(clamp(a) * 255),)


# ---------- type ----------
FONTS = {
    'display': 'Anton-Regular.ttf', 'heavy': 'ArchivoBlack-Regular.ttf', 'sans': 'Inter.ttf', 'geo': 'Montserrat.ttf',
    'serif': 'PlayfairDisplay.ttf', 'serif-italic': 'PlayfairDisplay-Italic.ttf', 'script': 'InstrumentSerif-Italic.ttf',
    'mono': 'SpaceMono-Bold.ttf', 'marker': 'PermanentMarker-Regular.ttf',
}


@lru_cache(maxsize=256)
def font(role='sans', size=40, weight=None):
    """role: display, heavy, sans, geo, serif, serif-italic, script, mono, marker. weight (100-900) for sans/geo/serif."""
    f = ImageFont.truetype(os.path.join(FONT_DIR, FONTS.get(role, FONTS['sans'])), max(1, int(size)))
    if role in ('sans', 'geo', 'serif'):
        weight = weight or 400
        try:
            axes = f.get_variation_axes()
            vals = []
            for ax in axes:
                tag = ax.get('name', b'')
                tag = tag.decode() if isinstance(tag, bytes) else str(tag)
                if 'eight' in tag or tag.lower().startswith('wght'): vals.append(clamp(int(weight), ax['minimum'], ax['maximum']))
                elif 'ptical' in tag or tag.lower().startswith('opsz'): vals.append(clamp(size * 0.5, ax['minimum'], ax['maximum']))
                else: vals.append(ax.get('default', ax['minimum']))
            f.set_variation_by_axes(vals)
        except Exception: pass
    return f


def text_size(txt, f):
    l, t, r, b = f.getbbox(txt); return r - l, b - t


def fit_font(txt, role, width, start, weight=None, floor=10):
    """Largest font size <= start whose text fits in `width`."""
    s = start
    while s > floor and text_size(txt, font(role, s, weight))[0] > width: s *= 0.94
    return font(role, s, weight)


def text_layer(txt, f, fill=(255, 255, 255, 255), stroke=0, stroke_fill=(0, 0, 0, 255), pad=None, tracking=0):
    """Tight RGBA image of a line of text. tracking = extra px between letters."""
    pad = int(pad if pad is not None else f.size * 0.25 + stroke)
    if tracking:
        widths = [f.getlength(ch) for ch in txt]; w = int(sum(widths) + tracking * max(0, len(txt) - 1))
    else:
        w = int(f.getlength(txt))
    asc, desc = f.getmetrics()
    im = Image.new('RGBA', (w + 2 * pad, asc + desc + 2 * pad), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    if tracking:
        x = pad
        for ch, cw in zip(txt, widths):
            d.text((x, pad), ch, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill); x += cw + tracking
    else:
        d.text((pad, pad), txt, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    bb = im.getbbox()
    return im.crop(bb) if bb else im


def wrap(txt, f, width):
    words, lines, cur = txt.split(), [], ''
    for w in words:
        nxt = (cur + ' ' + w).strip()
        if cur and f.getlength(nxt) > width: lines.append(cur); cur = w
        else: cur = nxt
    if cur: lines.append(cur)
    return lines


# ---------- shapes ----------
def rounded(w, h, r, fill=(255, 255, 255, 255), outline=None, width=0):
    w, h, k = max(1, int(w)), max(1, int(h)), 4                      # drawn 4x and scaled down: smooth corners
    im = Image.new('RGBA', (w * k, h * k), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w * k - 1, h * k - 1), int(r * k), fill=fill, outline=outline, width=int(width * k))
    return im.resize((w, h), Image.LANCZOS)


def _round_mask(w, h, r, k=4):
    m = Image.new('L', (w * k, h * k), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, w * k - 1, h * k - 1), int(r * k), fill=255)
    return m.resize((w, h), Image.LANCZOS)


def mask_rounded(im, r):
    m = _round_mask(im.width, im.height, r)
    out = im.convert('RGBA'); a = np.minimum(np.asarray(out.getchannel('A')), np.asarray(m)); out.putalpha(Image.fromarray(a)); return out


def shadow(im, blur=24, alpha=0.45, offset=(0, 12), color=(0, 0, 0)):
    """Returns (image with soft drop shadow, (dx, dy) to shift paste position by)."""
    pad = int(blur * 2.5) + max(abs(offset[0]), abs(offset[1]))
    a = im.getchannel('A').point(lambda v: int(v * alpha))
    sh = Image.new('RGBA', (im.width + 2 * pad, im.height + 2 * pad), color + (0,))
    sh.putalpha(Image.new('L', sh.size, 0)); al = Image.new('L', sh.size, 0); al.paste(a, (pad + offset[0], pad + offset[1]))
    sh.putalpha(al.filter(ImageFilter.GaussianBlur(blur))); sh.alpha_composite(im, (pad, pad))
    return sh, (-pad, -pad)


def glow(im, radius=18, strength=1.6, color=None):
    """Soft bloom around the opaque parts of `im` (returns a bigger image + offset)."""
    pad = radius * 3
    big = Image.new('RGBA', (im.width + 2 * pad, im.height + 2 * pad), (0, 0, 0, 0)); big.alpha_composite(im, (pad, pad))
    g = big.filter(ImageFilter.GaussianBlur(radius))
    if color is not None:
        g = Image.merge('RGBA', [Image.new('L', g.size, c) for c in color] + [g.getchannel('A')])
    ga = np.asarray(g, np.float32); ga[..., 3] = np.clip(ga[..., 3] * strength, 0, 255)
    out = Image.fromarray(ga.astype(np.uint8)); out.alpha_composite(big)
    return out, (-pad, -pad)


def scaled(im, s):
    if abs(s - 1) < 1e-3: return im
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS if s < 1 else Image.BICUBIC)


def faded(im, a):
    if a >= 0.999: return im
    out = im.copy(); out.putalpha(im.getchannel('A').point(lambda v: int(v * clamp(a)))); return out


def blurred(im, r):
    return im if r < 0.3 else im.filter(ImageFilter.GaussianBlur(r))


def rotated(im, deg):
    return im if abs(deg) < 0.05 else im.rotate(deg, resample=Image.BICUBIC, expand=True)


def gradient(w, h, top, bottom, alpha=(255, 255)):
    t = np.linspace(0, 1, int(h), dtype=np.float32)[:, None, None]
    c = (1 - t) * np.float32(list(top) + [alpha[0]]) + t * np.float32(list(bottom) + [alpha[1]])
    return Image.fromarray(np.repeat(c, int(w), 1).astype(np.uint8), 'RGBA')


def radial(w, h, cx, cy, r, color, alpha=1.0, power=2.0):
    yy, xx = np.mgrid[0:int(h), 0:int(w)].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(1, r)
    a = np.clip(1 - d, 0, 1) ** power * alpha
    out = np.zeros((int(h), int(w), 4), np.float32); out[..., :3] = color; out[..., 3] = a * 255
    return Image.fromarray(out.astype(np.uint8), 'RGBA')


def glass(bg_rgb, box, r, tint=(255, 255, 255), tint_a=0.16, blur=22, border_a=0.35, sheen=True):
    """Frosted glass panel. bg_rgb: HxWx3 float frame it sits on; box=(x, y, w, h). Returns an RGBA panel image."""
    x, y, w, h = [int(v) for v in box]; H, W = bg_rgb.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    patch = np.zeros((h, w, 3), np.float32)
    if x1 > x0 and y1 > y0: patch[y0 - y:y1 - y, x0 - x:x1 - x] = bg_rgb[y0:y1, x0:x1]
    patch = cv2.GaussianBlur(patch, (0, 0), blur)
    patch = patch * (1 - tint_a) + np.float32(tint) / 255 * tint_a
    if sheen:
        g = np.linspace(0.10, 0.0, h, dtype=np.float32)[:, None, None]; patch = np.clip(patch + g, 0, 1)
    im = Image.fromarray((np.clip(patch, 0, 1) * 255).astype(np.uint8)).convert('RGBA')
    im = mask_rounded(im, r)
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w - 1, h - 1), int(r), outline=(255, 255, 255, int(255 * border_a)), width=max(1, int(w / 300)))
    return im


def card(img, w, h=None, r=None, fit='cover'):
    """Image -> rounded card of width w (h defaults to 16:9 portrait ratio of the image)."""
    img = img.convert('RGB'); h = h or int(w * img.height / img.width); r = w * 0.08 if r is None else r
    s = max(w / img.width, h / img.height) if fit == 'cover' else min(w / img.width, h / img.height)
    im = img.resize((max(1, int(img.width * s + 0.5)), max(1, int(img.height * s + 0.5))), Image.LANCZOS)
    im = im.crop(((im.width - w) // 2, (im.height - h) // 2, (im.width - w) // 2 + w, (im.height - h) // 2 + h))
    return mask_rounded(im, r)


def cursor(size=48, color=(255, 255, 255), outline=(20, 20, 20)):
    s = size / 48; pts = [(0, 0), (0, 38), (10, 29), (17, 45), (24, 42), (17, 26), (30, 26)]
    im = Image.new('RGBA', (int(34 * s) + 4, int(48 * s) + 4), (0, 0, 0, 0))
    ImageDraw.Draw(im).polygon([(2 + x * s, 2 + y * s) for x, y in pts], fill=color + (255,), outline=outline + (255,), width=max(1, int(2 * s)))
    return im


def check_mark(size, color=(255, 255, 255), width=None):
    im = Image.new('RGBA', (int(size), int(size)), (0, 0, 0, 0)); s = size
    ImageDraw.Draw(im).line([(s * 0.22, s * 0.52), (s * 0.43, s * 0.72), (s * 0.78, s * 0.30)], fill=color + (255,), width=int(width or s * 0.12), joint='curve')
    return im


def number_text(v, like='0'):
    """Format v like the example string: '197K', '59%', '1.2M', '3,400'."""
    like = str(like)
    if like.endswith('%'): return f'{v:.0f}%'
    if like.upper().endswith('M'): return f'{v / 1e6:.1f}M'
    if like.upper().endswith('K'): return f'{v / 1e3:.0f}K'
    return f'{v:,.0f}'


def parse_number(s):
    s = str(s).strip().replace(',', ''); mul = 1
    if s.endswith('%'): s = s[:-1]
    elif s[-1:].upper() == 'K': mul, s = 1e3, s[:-1]
    elif s[-1:].upper() == 'M': mul, s = 1e6, s[:-1]
    try: return float(s) * mul
    except ValueError: return 0.0


# ---------- frame-level (numpy float RGB 0..1) ----------
def to_np(im):
    return np.asarray(im.convert('RGB'), np.float32) / 255


def to_pil(rgb):
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def luma(rgb):
    return rgb @ np.float32([0.299, 0.587, 0.114])


def tone(rgb, look):
    """Colour looks: moody, bw, warm, cool, pop, film, red."""
    if look == 'bw':
        y = luma(rgb)[..., None]; return np.clip((y - 0.5) * 1.15 + 0.5, 0, 1).repeat(3, 2)
    if look == 'moody':
        y = luma(rgb)[..., None]; out = rgb * 0.75 + y * 0.25
        out = out * np.float32([0.92, 0.98, 1.04]) - np.float32([0.03, 0.01, 0.0]) * (1 - y)
        return np.clip((out - 0.5) * 1.08 + 0.46, 0, 1)
    if look == 'warm':
        return np.clip(rgb * np.float32([1.08, 1.0, 0.86]) + 0.015, 0, 1)
    if look == 'cool':
        return np.clip(rgb * np.float32([0.9, 0.99, 1.1]), 0, 1)
    if look == 'pop':
        y = luma(rgb)[..., None]; return np.clip(y + (rgb - y) * 1.35, 0, 1) ** 0.95
    if look == 'film':
        y = luma(rgb)[..., None]; out = y + (rgb - y) * 0.8
        return np.clip(out * 0.9 + 0.06, 0, 1) * np.float32([1.03, 1.0, 0.94])
    if look == 'red':
        y = luma(rgb)[..., None]; return np.clip(np.concatenate([y * 1.25 + 0.03, y * 0.55, y * 0.5], 2), 0, 1)
    return rgb


def vignette(rgb, amount=0.4):
    H, W = rgb.shape[:2]; yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = ((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2 * 0.6) * 2.4
    return rgb * (1 - amount * np.clip(d, 0, 1))[..., None]
