"""Words behind you, part 3 + Numbers: match-cut, headline-wall, scribble, pause-point, reject-stamp,
countdown-flip, tier-list, chart-split."""
import math, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

from . import kit
from .engine import effect


# ---------- private helpers ----------
def _face0(s, p):
    """Face at the effect's first frame (stable layouts, same in video and still mode)."""
    fs = s.ctx.faces
    return fs[min(len(fs) - 1, max(0, int(round(p.start * s.fps))))]


def _partial(pts, f):
    """First fraction f (by arc length) of a polyline."""
    if f <= 0 or len(pts) < 2: return []
    pts = np.asarray(pts, np.float32)
    if f >= 1: return [tuple(q) for q in pts]
    seg = np.hypot(*np.diff(pts, axis=0).T); L = np.concatenate([[0], np.cumsum(seg)]); tgt = f * L[-1]
    k = int(np.searchsorted(L, tgt))
    out = [tuple(q) for q in pts[:k]]
    if 0 < k < len(pts):
        a = (tgt - L[k - 1]) / max(1e-6, L[k] - L[k - 1]); out.append(tuple(pts[k - 1] + (pts[k] - pts[k - 1]) * a))
    return out


def _arrow_head(path, size):
    """Two short strokes at the end of a path."""
    if len(path) < 2: return []
    pts = np.asarray(path, np.float32); end = pts[-1]
    back = pts[max(0, len(pts) - 6)]
    d = end - back; n = np.hypot(*d)
    if n < 1e-3: return []
    d /= n; out = []
    for ang in (2.6, -2.6):
        c, sn = math.cos(ang), math.sin(ang)
        v = np.float32([d[0] * c - d[1] * sn, d[0] * sn + d[1] * c])
        out.append([tuple(end), tuple(end + v * size)])
    return out


def _strokes(paths, width, color, ss=3):
    """Anti-aliased marker strokes (round caps) -> (RGBA image, x0, y0) in frame coords, or None."""
    paths = [q for q in paths if len(q) >= 2]
    if not paths: return None
    allp = np.float32([pt for q in paths for pt in q])
    pad = int(width * 2 + 4)
    x0, y0 = int(allp[:, 0].min()) - pad, int(allp[:, 1].min()) - pad
    x1, y1 = int(allp[:, 0].max()) + pad, int(allp[:, 1].max()) + pad
    w, h = x1 - x0, y1 - y0
    im = Image.new('RGBA', (w * ss, h * ss), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    r = width * ss / 2; fill = tuple(color[:3]) + (255,)
    for q in paths:
        qq = [((x - x0) * ss, (y - y0) * ss) for x, y in q]
        d.line(qq, fill=fill, width=max(1, int(width * ss)), joint='curve')
        for x, y in (qq[0], qq[-1]): d.ellipse((x - r, y - r, x + r, y + r), fill=fill)
    return im.resize((w, h), Image.LANCZOS), x0, y0


def _over_strokes(s, paths, width, color, shadow_a=0.45):
    L = _strokes(paths, width, color)
    if L is None: return
    im, x0, y0 = L
    sh, (dx, dy) = kit.shadow(im, blur=max(2, int(width * 0.8)), alpha=shadow_a, offset=(0, int(width * 0.35)))
    s.over(sh, x0 + dx, y0 + dy)


def _wipe(im, f, soft=0.08):
    """Reveal an RGBA image left to right (handwriting-ish)."""
    if f >= 1: return im
    w = im.width; edge = max(1.0, w * soft)
    ramp = np.clip(((f * (w + edge)) - np.arange(w, dtype=np.float32)) / edge, 0, 1)
    a = np.asarray(im.getchannel('A'), np.float32) * ramp[None, :]
    out = im.copy(); out.putalpha(Image.fromarray(a.astype(np.uint8))); return out


def _frame_at(s, sec):
    """Decode one RGB frame of the source at `sec` (float 0..1), scaled to the output size."""
    try:
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{max(0.0, sec):.3f}', '-i', s.ctx.src, '-frames:v', '1',
                              '-vf', f'scale={s.W}:{s.H}:flags=bicubic', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                             capture_output=True, check=True).stdout
        if len(raw) >= s.W * s.H * 3:
            return np.frombuffer(raw[:s.W * s.H * 3], np.uint8).reshape(s.H, s.W, 3).astype(np.float32) / 255
    except Exception:
        pass
    return None


def _text_center(d, xy, txt, f, fill):
    l, t, r, b = f.getbbox(txt)
    d.text((xy[0] - (l + r) / 2, xy[1] - (t + b) / 2), txt, font=f, fill=fill)


# =====================================================================
# 1. match-cut
# =====================================================================
_MASTHEADS = ['THE DAILY EDIT', 'The Reel Times', 'MORNING CUT', 'The Creator Post', 'WEEKLY FRAME', 'The Timeline']
_HEADLINES = ['Creator ditches editing apps for good', 'One clip. 48 effects. Zero apps.', 'Is this the end of the timeline?',
              'The reel everyone is talking about', 'AI editor goes viral overnight', 'How @keemfinity edits in seconds']
_PAPER = (246, 242, 233)
_INK = (28, 26, 24)


def _grey_lines(d, x, y, w, y_end, lh, rng, col=(170, 166, 158), u=1.0):
    while y + lh * 0.5 < y_end:
        ln = w * (rng.uniform(0.55, 0.98) if rng.random() < 0.2 else rng.uniform(0.86, 1.0))
        d.rounded_rectangle((x, y, x + ln, y + lh * 0.42), radius=lh * 0.2, fill=col + (255,))
        y += lh


def _photo(d, im, thumb, box, look):
    x0, y0, x1, y1 = [int(v) for v in box]
    if x1 - x0 < 4 or y1 - y0 < 4: return
    t = kit.to_pil(kit.tone(kit.to_np(thumb), look))
    c = kit.card(t, x1 - x0, y1 - y0, r=0)
    im.alpha_composite(c, (x0, y0))


def _news_page(w, h, v, u, thumbs, rng):
    im = Image.new('RGBA', (w, h), _PAPER + (255,)); d = ImageDraw.Draw(im)
    m = int(44 * u); cw = w - 2 * m
    head = _HEADLINES[v % len(_HEADLINES)]; mast = _MASTHEADS[v % len(_MASTHEADS)]
    th = thumbs[v % len(thumbs)]
    lh = 30 * u
    if v == 0:      # classic broadsheet
        f = kit.font('serif', 70 * u, 900); _text_center(d, (w / 2, m + 40 * u), mast, f, _INK + (255,))
        y = m + 90 * u; d.line((m, y, w - m, y), fill=_INK + (255,), width=int(4 * u)); d.line((m, y + 9 * u, w - m, y + 9 * u), fill=_INK + (255,), width=int(2 * u))
        fh = kit.font('serif', 64 * u, 800); yy = y + 30 * u
        for ln in kit.wrap(head, fh, cw)[:2]: d.text((m, yy), ln, font=fh, fill=_INK + (255,)); yy += 74 * u
        yy += 16 * u; _photo(d, im, th, (m + cw * 0.5, yy, w - m, yy + h * 0.36), 'bw')
        _grey_lines(d, m, yy, cw * 0.46, h - m, lh, rng)
        _grey_lines(d, m + cw * 0.5, yy + h * 0.36 + 24 * u, cw * 0.5, h - m, lh, rng)
    elif v == 1:    # photo-led front page
        f = kit.font('heavy', 54 * u); d.text((m, m - 6 * u), mast.upper(), font=f, fill=(190, 30, 36, 255))
        _photo(d, im, th, (m, m + 80 * u, w - m, m + 80 * u + h * 0.34), 'film')
        y = m + 100 * u + h * 0.34
        fh = kit.font('serif', 58 * u, 700)
        for ln in kit.wrap(head, fh, cw)[:2]: d.text((m, y), ln, font=fh, fill=_INK + (255,)); y += 66 * u
        y += 20 * u; col = (cw - 2 * 26 * u) / 3
        for k in range(3): _grey_lines(d, m + k * (col + 26 * u), y, col, h - m, lh, rng)
    elif v == 2:    # tall headline, small photo
        f = kit.font('serif', 34 * u, 600); d.text((m, m), mast + '  |  No. 48', font=f, fill=(110, 104, 96, 255))
        d.line((m, m + 52 * u, w - m, m + 52 * u), fill=_INK + (255,), width=int(2 * u))
        fh = kit.font('serif', 92 * u, 900); y = m + 70 * u
        for ln in kit.wrap(head, fh, cw)[:3]: d.text((m, y), ln, font=fh, fill=_INK + (255,)); y += 100 * u
        y += 26 * u; _photo(d, im, th, (m, y, m + cw * 0.42, y + h * 0.24), 'warm')
        _grey_lines(d, m + cw * 0.46, y, cw * 0.54, h - m, lh, rng)
        _grey_lines(d, m, y + h * 0.24 + 24 * u, cw * 0.42, h - m, lh, rng)
    elif v == 3:    # tabloid banner
        d.rectangle((0, 0, w, 150 * u), fill=(200, 24, 32, 255))
        f = kit.fit_font(mast.upper(), 'display', cw, 120 * u); _text_center(d, (w / 2, 75 * u), mast.upper(), f, (255, 255, 255, 255))
        fh = kit.font('display', 96 * u); y = 180 * u
        for ln in kit.wrap(head.upper(), fh, cw)[:2]: d.text((m, y), ln, font=fh, fill=_INK + (255,)); y += 112 * u
        y += 16 * u; _photo(d, im, th, (m, y, w - m, y + h * 0.28), 'pop')
        _grey_lines(d, m, y + h * 0.28 + 24 * u, cw, h - m, lh, rng)
    elif v == 4:    # magazine
        d.rectangle((0, 0, w, h), fill=(240, 232, 218, 255))
        d.rectangle((m, m, m + 12 * u, h * 0.42), fill=(200, 85, 61, 255))
        f = kit.font('serif-italic', 40 * u); d.text((m + 34 * u, m - 6 * u), mast, font=f, fill=(200, 85, 61, 255))
        fh = kit.font('serif-italic', 78 * u); y = m + 60 * u
        for ln in kit.wrap(head, fh, cw - 40 * u)[:3]: d.text((m + 34 * u, y), ln, font=fh, fill=_INK + (255,)); y += 88 * u
        _photo(d, im, th, (m, h * 0.46, w - m, h * 0.70), 'moody')
        col = (cw - 26 * u) / 2
        for k in range(2): _grey_lines(d, m + k * (col + 26 * u), h * 0.72 + 10 * u, col, h - m, lh, rng)
    else:           # grid of stories
        f = kit.font('geo', 46 * u, 900); _text_center(d, (w / 2, m + 24 * u), mast.upper(), f, _INK + (255,))
        d.line((m, m + 64 * u, w - m, m + 64 * u), fill=_INK + (255,), width=int(3 * u))
        cell = (cw - 26 * u) / 2; top = m + 90 * u; ch = (h - top - m) / 2
        heads = [head, 'Zero apps, one prompt', '48 effects explained', 'Inside the edit']
        for k in range(4):
            cx = m + (k % 2) * (cell + 26 * u); cy = top + (k // 2) * ch
            _photo(d, im, thumbs[(v + k) % len(thumbs)], (cx, cy, cx + cell, cy + ch * 0.42), ['bw', 'warm', 'cool', 'film'][k])
            fh = kit.font('serif', 38 * u, 800); yy = cy + ch * 0.42 + 14 * u
            for ln in kit.wrap(heads[k], fh, cell)[:2]: d.text((cx, yy), ln, font=fh, fill=_INK + (255,)); yy += 44 * u
            _grey_lines(d, cx, yy + 8 * u, cell, cy + ch - 20 * u, lh * 0.9, rng)
    # subtle paper grain + edge
    g = np.random.default_rng(100 + v).normal(0, 4, (h, w, 1)).astype(np.float32)
    a = np.asarray(im, np.float32); a[..., :3] = np.clip(a[..., :3] + g, 0, 255)
    im = Image.fromarray(a.astype(np.uint8), 'RGBA')
    ImageDraw.Draw(im).rectangle((0, 0, w - 1, h - 1), outline=(205, 198, 186, 255), width=max(1, int(2 * u)))
    return im


@effect('match-cut', 'Words Behind You', 'news pages flash behind you while one word stays locked in a red box')
def match_cut(s, p):
    word = p.text('Claude'); red = p.color('color', (222, 32, 40)); u = s.u

    def build():
        f0 = _face0(s, p)
        px, py = int(0.04 * s.W), int(0.03 * s.H); pw, ph = int(0.92 * s.W), int(0.60 * s.H)
        # the word sits beside the head on the roomier side, at eye level
        left = (0.05 * s.W, f0.cx - f0.w * 0.85); right = (f0.cx + f0.w * 0.85, 0.95 * s.W)
        lo, hi = left if (left[1] - left[0]) >= (right[1] - right[0]) else right
        fw = kit.fit_font(word, 'serif', (hi - lo) * 0.80, 92 * u, 900)
        txt = kit.text_layer(word, fw, fill=(255, 255, 255, 255), pad=0)
        bw, bh = int(txt.width + 34 * u), int(txt.height + 30 * u)
        wx, wy = (lo + hi) / 2, f0.cy - f0.h * 0.08
        bx, by = int(wx - bw / 2 - px), int(wy - bh / 2 - py)
        box = Image.new('RGBA', (bw, bh), red + (255,)); box.alpha_composite(txt, ((bw - txt.width) // 2, (bh - txt.height) // 2))
        thumbs = p.images(6)
        pages = []
        for v in range(6):
            pg = _news_page(pw, ph, v, u, thumbs, np.random.default_rng(7 + v))
            clr = int(14 * u)
            ImageDraw.Draw(pg).rectangle((bx - clr, by - clr, bx + bw + clr, by + bh + clr), fill=_PAPER + (255,))
            pg.alpha_composite(box, (bx, by))
            sh, (dx, dy) = kit.shadow(pg, blur=int(26 * u), alpha=0.5, offset=(0, int(14 * u)))
            pages.append((sh, px + dx, py + dy))
        return pages, (px + bx + bw / 2, py + by + bh / 2)

    pages, (wx, wy) = s.once('pages', build)
    hold = min(0.45, s.dur * 0.18)
    cuts = int(max(0, s.t) / 0.12)
    v = 0 if s.t > s.dur - hold else (cuts % len(pages))
    im, x, y = pages[v]
    k = 1 + 0.035 * s.p + 0.06 * (1 - kit.expo_out(s.t / 0.35))       # punch in, then slow push about the word
    lay = kit.scaled(im, k)
    s.behind(lay, wx - (wx - x) * k, wy - (wy - y) * k, alpha=s.env(0.08, 0.22))


# =====================================================================
# 2. headline-wall
# =====================================================================
_HW_DEFAULT = ('I let Claude edit my reels::1.2M views|One clip, 48 effects::847K views|Zero apps. Just prompts.::612K views|'
               'The edit that broke my DMs::503K views|Claude made my thumbnail::421K views|48 effects in one reel::318K views|'
               'How I edit in 60 seconds::209K views')


def _headline_card(head, meta, w, u, i, mirror=False):
    pad = 24 * u; av = 58 * u
    fh = kit.font('sans', 34 * u, 700); fm = kit.font('sans', 26 * u, 500)
    lines = kit.wrap(head, fh, w - 2 * pad - av - 18 * u)[:2]
    h = int(pad * 2 + len(lines) * 42 * u + 38 * u)
    im = kit.rounded(w, h, 24 * u, fill=(255, 255, 255, 255)); d = ImageDraw.Draw(im)
    hues = [(255, 106, 61), (99, 102, 241), (16, 185, 129), (236, 72, 153), (245, 158, 11), (14, 165, 233), (217, 119, 87)]
    c = hues[i % len(hues)]
    ax = w - pad - av if mirror else pad          # avatar sits on the inner side (behind the head), text stays readable
    d.ellipse((ax, pad, ax + av, pad + av), fill=c + (255,))
    _text_center(d, (ax + av / 2, pad + av / 2), 'K', kit.font('heavy', 28 * u), (255, 255, 255, 255))
    x = pad if mirror else pad + av + 18 * u; y = pad - 4 * u
    for ln in lines: d.text((x, y), ln, font=fh, fill=(20, 20, 24, 255)); y += 42 * u
    y += 6 * u
    tri = [(x, y + 6 * u), (x, y + 28 * u), (x + 18 * u, y + 17 * u)]
    d.polygon(tri, fill=(120, 120, 128, 255))
    d.text((x + 28 * u, y), meta, font=fm, fill=(110, 110, 120, 255))
    return im


@effect('headline-wall', 'Words Behind You', 'headline cards pile up behind you, then the total views pop in front')
def headline_wall(s, p):
    u = s.u
    total = str(p.get('total', '4.1M')); label = str(p.get('label', 'total views')).upper()
    items = [tuple((x.split('::', 1) + [''])[:2]) for x in p.text(_HW_DEFAULT).split('|') if x.strip()][:8]

    def build():
        f0 = _face0(s, p); rng = np.random.default_rng(11); cards = []
        n = len(items); cw = int(0.44 * s.W)
        top, bot = 0.035 * s.H, max(0.27 * s.H, f0.chin - 0.01 * s.H)
        for i, (h, m) in enumerate(items):
            side = -1 if i % 2 == 0 else 1
            im = _headline_card(h, m, cw, u, i, mirror=side < 0)
            cx = s.W / 2 + side * (0.27 * s.W + rng.uniform(-0.01, 0.01) * s.W)
            cy = top + (bot - top) * (i / max(1, n - 1)) + im.height * 0.5
            rot = side * rng.uniform(1.5, 4.5) * (1 if rng.random() < 0.7 else -1)
            sh, _ = kit.shadow(kit.rotated(im, rot), blur=int(20 * u), alpha=0.42, offset=(0, int(12 * u)))
            cards.append((sh, cx, cy))
        return cards

    cards = s.once('cards', build)
    n = len(cards); env = s.env(0.0, 0.25)
    t_tot = min(s.dur * 0.62, 0.35 + n * 0.26)
    each = (t_tot - 0.25) / max(1, n)
    for i, (im, cx, cy) in enumerate(cards):
        k = kit.stagger(s.t, i, 0.0, each, 0.4)
        if k <= 0: continue
        e = kit.expo_out(k)
        sc = (1.22 - 0.22 * e) * (1 + 0.012 * s.p)
        drift = (0.004 * s.H) * math.sin(s.t * 1.3 + i)
        s.behind(kit.scaled(im, sc), cx, cy - (1 - e) * 0.05 * s.H + drift, 'c', alpha=min(1, k * 3) * env)
    # the total, in front, below the chin
    k = kit.window(s.t, t_tot, 0.55)
    if k > 0:
        f = _face0(s, p)
        val = kit.parse_number(total) * kit.expo_out(kit.window(s.t, t_tot, 0.6))
        num = total if k >= 1 else kit.number_text(val, total)
        big = kit.text_layer(num, kit.font('display', 210 * u), fill=(255, 255, 255, 255))
        lab = kit.text_layer(label, kit.font('sans', 40 * u, 800), fill=(255, 214, 102, 255), tracking=int(8 * u))
        blk = Image.new('RGBA', (max(big.width, lab.width) + 20, big.height + lab.height + int(26 * u)), (0, 0, 0, 0))
        blk.alpha_composite(big, ((blk.width - big.width) // 2, 0)); blk.alpha_composite(lab, ((blk.width - lab.width) // 2, big.height + int(26 * u)))
        sc = 0.35 + 0.65 * kit.back_out(k, 2.2) + 0.03 * s.p
        cy = s.face.chin + 0.15 * s.H
        pw, ph = int((blk.width + 90 * u) * min(1.0, sc)), int((blk.height + 60 * u) * min(1.0, sc))
        panel = kit.glass(s.rgb, (s.W / 2 - pw / 2, cy - ph / 2, pw, ph), 36 * u, tint=(10, 10, 14), tint_a=0.6, blur=16 * u, border_a=0.3)
        psh, (pdx, pdy) = kit.shadow(panel, blur=int(24 * u), alpha=0.5, offset=(0, int(12 * u)))
        s.over(psh, s.W / 2, cy, 'c', alpha=min(1, k * 4) * env)
        g, (dx, dy) = kit.glow(blk, radius=int(22 * u), strength=0.9, color=(255, 170, 60))
        s.over(kit.scaled(g, sc), s.W / 2, cy, 'c', alpha=min(1, k * 4) * env)


# =====================================================================
# 3. scribble
# =====================================================================
@effect('scribble', 'Words Behind You', 'marker doodles draw themselves around you: circle, arrow, label')
def scribble(s, p):
    u = s.u; col = p.color('color', (255, 210, 63)); word = p.text('ME!')
    f = s.face; rng = s.once('rng', lambda: np.random.default_rng(5).uniform(0, 6.28, 4))
    hx, hy = f.cx, (f.crown + f.chin) / 2 + f.h * 0.02
    rx, ry = f.w * 1.05, (f.chin - f.crown) / 2 * 1.2
    a0 = math.radians(200); turns = 1.12; N = 140
    circ = []
    for i in range(N + 1):
        q = i / N; th = a0 + q * turns * 2 * math.pi
        wob = 1 + 0.035 * math.sin(3 * th + rng[0]) + 0.02 * math.sin(5 * th + rng[1]) + 0.05 * q
        circ.append((hx + rx * wob * math.cos(th), hy + ry * wob * math.sin(th)))
    # label up-right, arrow curving from it to the circle
    side = 1 if f.cx < s.W * 0.56 else -1
    lab = s.once('lab', lambda: kit.text_layer(word, kit.font('marker', 104 * u), fill=col + (255,)))
    lx = kit.clamp(f.cx + side * (f.w * 1.75), lab.width / 2 + 30 * u, s.W - lab.width / 2 - 30 * u)
    ly = max(lab.height / 2 + 30 * u, f.top - f.h * 0.05)
    start = (lx - side * lab.width * 0.15, ly + lab.height * 0.62)
    end = (hx + side * rx * 1.12, hy + ry * 0.05)
    ctrl = (start[0] + (end[0] - start[0]) * 0.05, end[1] + (start[1] - end[1]) * 0.05)
    arr = [((1 - t) ** 2 * start[0] + 2 * (1 - t) * t * ctrl[0] + t * t * end[0],
            (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * ctrl[1] + t * t * end[1]) for t in np.linspace(0, 1, 40)]
    t = s.t * max(1.0, 2.4 / s.dur); out = kit.cubic_out(kit.window(s.t, s.dur - 0.25, 0.25))   # compress timing for short clips
    pc = kit.smooth(kit.window(t, 0.05, 0.55))
    pa = kit.cubic_out(kit.window(t, 0.55, 0.35))
    pl = kit.window(t, 0.85, 0.45)
    wid = 13 * u
    paths = [_partial(circ, pc), _partial(arr, pa)]
    if pa >= 1: paths += _arrow_head(arr, 40 * u * min(1, kit.window(t, 0.9, 0.12) * 1.0 + 0.001))
    if out < 1:
        L = _strokes(paths, wid, col)
        if L:
            sh, (dx, dy) = kit.shadow(L[0], blur=max(2, int(wid * 0.8)), alpha=0.45, offset=(0, int(wid * 0.35)))
            s.over(sh, L[1] + dx, L[2] + dy, alpha=1 - out)
        if pl > 0:
            im = _wipe(lab, kit.smooth(pl))
            im = kit.rotated(im, -8 * side + 1.5 * math.sin(t * 2))
            sh, (dx, dy) = kit.shadow(im, blur=int(8 * u), alpha=0.5, offset=(0, int(4 * u)))
            s.over(sh, lx, ly, 'c', alpha=1 - out)


# =====================================================================
# 4. pause-point
# =====================================================================
def _pause_badge(u):
    h = int(96 * u); f = kit.font('mono', 38 * u)
    txt = kit.text_layer('PAUSED', f, fill=(255, 255, 255, 255))
    w = int(h + txt.width + 34 * u)
    im = kit.rounded(w, h, h / 2, fill=(14, 14, 18, 190), outline=(255, 255, 255, 90), width=max(1, int(2 * u)))
    d = ImageDraw.Draw(im); bx = h * 0.36; bw = h * 0.11
    for k in range(2):
        x = bx + k * bw * 2.1
        d.rounded_rectangle((x, h * 0.3, x + bw, h * 0.7), radius=bw * 0.3, fill=(255, 255, 255, 255))
    im.alpha_composite(txt, (int(h * 0.92), (h - txt.height) // 2))
    return im


@effect('pause-point', 'Words Behind You', 'freeze frame with a pause badge and a hand-drawn arrow to you', aliases=('pause',))
def pause_point(s, p):
    u = s.u; col = p.color('color', (255, 255, 255)); word = p.text('this guy')

    def grab():
        if s.t <= 1.5 / s.fps: return s.rgb.copy(), s.alpha.copy() if s.alpha is not None else None
        fr = _frame_at(s, p.start)
        return (fr if fr is not None else s.rgb.copy()), None
    frz, falpha = s.once('freeze', grab)
    out = 1 - s.env(0.0, 0.15)
    live = s.rgb.copy() if out > 0 else None
    f0 = _face0(s, p)
    zoom = 1 + 0.035 * kit.cubic_out(s.p)
    M = np.float32([[zoom, 0, f0.cx * (1 - zoom)], [0, zoom, f0.cy * (1 - zoom)]])
    img = cv2.warpAffine(frz, M, (s.W, s.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    k_in = kit.cubic_out(s.t / 0.2)
    y = kit.luma(img)[..., None]
    img = (y + (img - y) * (1 - 0.45 * k_in)) * (1 - 0.08 * k_in)
    img = kit.vignette(img, 0.35 * k_in)
    fl = max(0.0, 1 - s.t / 0.12) * 0.55                                       # shutter flash
    img = img * (1 - fl) + fl
    s.rgb[:] = img
    if falpha is not None: s.alpha = cv2.warpAffine(falpha, M, (s.W, s.H))
    fcx, fcy, fw, fh = f0.cx, f0.cy, f0.w * zoom, f0.h * zoom
    # pause badge
    badge = s.once('badge', lambda: _pause_badge(u))
    kb = kit.expo_out(kit.window(s.t, 0.05, 0.4))
    blink = 0.75 + 0.25 * (math.sin(s.t * 6) > -0.2)
    s.over(kit.scaled(badge, 0.8 + 0.2 * kb), 44 * u, 60 * u, 'tl', alpha=kb * (1 - out) * blink)
    # label + arrow
    lab = s.once('lab', lambda: kit.text_layer(word, kit.font('marker', 92 * u), fill=col + (255,)))
    lx = p.num('label_x', 0.80) * s.W
    lx = kit.clamp(lx, lab.width / 2 + 24 * u, s.W - lab.width / 2 - 24 * u)
    ly = fcy + fh * 0.62
    side = 1 if lx > fcx else -1
    start = (lx - side * lab.width * 0.15, ly - lab.height * 0.75)
    end = (fcx + side * fw * 0.62, fcy - fh * 0.08)
    ctrl = (start[0] + side * fw * 0.05, end[1] - fh * 0.1)
    arr = [((1 - t) ** 2 * start[0] + 2 * (1 - t) * t * ctrl[0] + t * t * end[0],
            (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * ctrl[1] + t * t * end[1]) for t in np.linspace(0, 1, 40)]
    pa = kit.cubic_out(kit.window(s.t, 0.45, 0.4)); pl = kit.window(s.t, 0.2, 0.35)
    paths = [_partial(arr, pa)]
    if pa >= 1: paths += _arrow_head(arr, 38 * u)
    L = _strokes(paths, 12 * u, col)
    if L:
        sh, (dx, dy) = kit.shadow(L[0], blur=int(7 * u), alpha=0.6, offset=(0, int(3 * u)))
        s.over(sh, L[1] + dx, L[2] + dy, alpha=1 - out)
    if pl > 0:
        im = kit.rotated(_wipe(lab, kit.smooth(pl)), 6 * side)
        sh, _ = kit.shadow(im, blur=int(8 * u), alpha=0.6, offset=(0, int(4 * u)))
        s.over(sh, lx, ly, 'c', alpha=1 - out)
    if live is not None:                                                     # unfreeze: blend back to live video
        s.rgb[:] = s.rgb * (1 - out) + live * out


# =====================================================================
# 5. reject-stamp
# =====================================================================
def _stamp(word, col, u, seed=3):
    f = kit.font('heavy', 132 * u)
    txt = kit.text_layer(word, f, fill=col + (255,), tracking=int(6 * u))
    pad = int(46 * u); bw = int(16 * u)
    w, h = txt.width + 2 * pad + 2 * bw, txt.height + 2 * pad + 2 * bw
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), int(26 * u), outline=col + (255,), width=bw)
    d.rounded_rectangle((bw + 12 * u, bw + 12 * u, w - bw - 12 * u, h - bw - 12 * u), int(14 * u), outline=col + (255,), width=int(5 * u))
    im.alpha_composite(txt, ((w - txt.width) // 2, (h - txt.height) // 2))
    # ink texture: rough edges + speckle holes + uneven pressure
    rng = np.random.default_rng(seed)
    a = np.asarray(im.getchannel('A'), np.float32) / 255
    n1 = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.2 * u + 0.5)
    n2 = cv2.resize(rng.random((max(2, h // 40), max(2, w // 40))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    edge = cv2.GaussianBlur(a, (0, 0), 2.2 * u + 0.5)
    rough = np.clip((edge - 0.5 + (n1 - 0.5) * 0.9) * 6 + 0.5, 0, 1)
    holes = np.clip((n1 - 0.70) * 9, 0, 1)
    press = 0.72 + 0.28 * np.clip(n2 * 1.3, 0, 1)
    a = a * rough * (1 - holes * 0.85) * press
    im.putalpha(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)))
    return im


@effect('reject-stamp', 'Words Behind You', 'a red rubber stamp slams down on you', aliases=('stamp',))
def reject_stamp(s, p):
    u = s.u; col = p.color('color', (214, 34, 42)); word = p.text('REJECTED').upper()
    ang = p.num('angle', -12)
    base = s.once('stamp', lambda: kit.rotated(_stamp(word, col, u), -ang))
    if base.width > s.W * 0.9: base = s.once('stamp_fit', lambda: kit.scaled(base, s.W * 0.9 / base.width))
    cy = p.num('y', 0.64) * s.H; cx = s.W / 2
    T = 0.2; t = s.t
    out = 1 - s.env(0.0, 0.25)
    if t < T:                                                                # falling: big, soft, translucent
        k = kit.cubic_in(t / T)
        sc = 2.5 - 1.5 * k
        im = kit.blurred(kit.scaled(base, sc), (1 - k) * 6 * u)
        sh = kit.blurred(kit.faded(kit.scaled(base, 1 + 0.4 * (1 - k)), 0.25 * k), 18 * u)
        shadow = Image.merge('RGBA', [Image.new('L', sh.size, 0)] * 3 + [sh.getchannel('A')])
        s.over(shadow, cx + 20 * u, cy + 30 * u, 'c', alpha=0.8)
        s.over(im, cx, cy, 'c', alpha=0.35 + 0.65 * k)
        return
    ti = t - T
    sc = 1 + 0.035 * math.exp(-ti * 18) * math.cos(ti * 40) + 0.012 * s.p
    s.over(kit.scaled(base, sc), cx, cy, 'c', alpha=0.93 * (1 - out))
    # impact: quick camera shake + a white-ish flash ring
    sh = math.exp(-ti * 14)
    if sh > 0.02:
        s.camera(1 + 0.02 * sh, cx=cx, cy=cy, dx=math.sin(ti * 90) * 14 * u * sh, dy=math.cos(ti * 77) * 10 * u * sh)


# =====================================================================
# 6. countdown-flip
# =====================================================================
def _flip_card(ch, w, h, u):
    r = int(w * 0.09)
    im = kit.mask_rounded(kit.gradient(w, h, (46, 46, 52), (20, 20, 24)), r)
    d = ImageDraw.Draw(im)
    f = kit.fit_font(ch, 'geo', w * 0.80, h * 0.80, 800)
    l, t, rr, b = f.getbbox(ch)
    d.text((w / 2 - (l + rr) / 2, h / 2 - (t + b) / 2), ch, font=f, fill=(250, 250, 248, 255))
    # top half a touch lighter
    a = np.asarray(im, np.float32); a[:h // 2, :, :3] = np.clip(a[:h // 2, :, :3] * 1.08 + 4, 0, 255)
    im = Image.fromarray(a.astype(np.uint8), 'RGBA'); d = ImageDraw.Draw(im)
    gap = max(2, int(5 * u))
    d.rectangle((0, h // 2 - gap // 2, w, h // 2 + gap // 2), fill=(6, 6, 8, 255))
    pin = int(w * 0.035)
    for x in (int(w * 0.035), w - int(w * 0.035) - pin):
        d.rectangle((x, h // 2 - pin, x + pin, h // 2 + pin), fill=(70, 70, 78, 255))
    d.rounded_rectangle((0, 0, w - 1, h - 1), r, outline=(255, 255, 255, 30), width=max(1, int(2 * u)))
    return im


def _fold(half, a, top, widen=0.10, ox=0):
    """Fold a card half around the hinge. a=0 flat, a=pi/2 edge-on. top: the half above the hinge."""
    w, hh = half.size
    c = math.cos(a); k = 1 + widen * math.sin(a)
    if c < 0.01: return None
    W2 = int(w * (1 + widen)) + 2; cx = W2 / 2
    src = np.float32([(0, 0), (w, 0), (w, hh), (0, hh)])
    if top:
        yf = hh - hh * c
        dst = np.float32([(cx - w / 2 * k, yf), (cx + w / 2 * k, yf), (cx + w / 2, hh), (cx - w / 2, hh)])
    else:
        yf = hh * c
        dst = np.float32([(cx - w / 2, 0), (cx + w / 2, 0), (cx + w / 2 * k, yf), (cx - w / 2 * k, yf)])
    M = cv2.getPerspectiveTransform(src, dst)
    arr = cv2.warpPerspective(np.asarray(half), M, (W2, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    arr = arr.astype(np.float32); arr[..., :3] *= (1 - 0.55 * math.sin(a))
    return Image.fromarray(arr.astype(np.uint8), 'RGBA')


@effect('countdown-flip', 'Numbers', 'split-flap flip clock counts 3, 2, 1 behind you', aliases=('countdown',))
def countdown_flip(s, p):
    u = s.u; n0 = max(1, int(p.num('from', 3)))
    nums = list(range(n0, 0, -1))
    digits = max(2, len(str(n0)))

    def layout():
        f0 = _face0(s, p)
        room_above = f0.crown - 0.02 * s.H
        if room_above > 0.17 * s.H:                                          # enough sky: clock above the head
            ch = min(room_above * 0.9, 0.16 * s.H); cw = ch / 1.32
            return dict(mode='above', cw=int(cw), ch=int(ch), y=f0.crown - ch / 2 - 0.02 * s.H, gap=cw * 0.08)
        # otherwise flank the head like the hours:minutes of a flip clock (the head is the colon)
        gap = f0.w * 1.0
        cw = min(0.32 * s.W, (s.W - gap) / 2 - 0.04 * s.W)
        ch = cw * 1.3
        return dict(mode='flank', cw=int(cw), ch=int(ch), y=f0.cy - f0.h * 0.05, gap=gap)
    L = s.once('layout', layout)
    cw, ch = L['cw'], L['ch']
    cards = s.state.setdefault('cards', {})

    def card(c):
        if c not in cards: cards[c] = _flip_card(c, cw, ch, u)
        return cards[c]

    lead = 0.3; seg = max(0.3, (s.dur - lead - 0.45) / len(nums))
    flipd = min(0.38, seg * 0.6)
    idx = int(kit.clamp((s.t - lead) / seg, 0, len(nums) - 1 + 0.999))
    loc = (s.t - lead) - idx * seg
    cur = str(nums[idx]).zfill(digits)
    prev = str(nums[idx - 1]).zfill(digits) if idx > 0 else cur
    fp = kit.clamp(loc / flipd) if idx > 0 else 1.0
    hh = ch // 2
    ox = int(cw * 0.06) + 2
    tiles = []
    for d_i in range(digits):
        a_new, a_old = cur[d_i], prev[d_i]
        tile = Image.new('RGBA', (cw + 2 * ox, ch), (0, 0, 0, 0))
        new, old = card(a_new), card(a_old)
        if a_new == a_old or fp >= 1:
            tile.alpha_composite(new, (ox, 0))
        else:
            tile.alpha_composite(new.crop((0, 0, cw, hh)), (ox, 0))
            tile.alpha_composite(old.crop((0, hh, cw, ch)), (ox, hh))
            e = kit.smooth(fp)
            if e < 0.5:
                fl = _fold(old.crop((0, 0, cw, hh)), e * math.pi, True)
                if fl: tile.alpha_composite(fl, (ox + (cw - fl.width) // 2, 0))
            else:
                fl = _fold(new.crop((0, hh, cw, ch)), (1 - e) * math.pi, False)
                if fl: tile.alpha_composite(fl, (ox + (cw - fl.width) // 2, hh))
        tiles.append(tile)
    env = s.env(0.0, 0.0)
    kin = kit.expo_out(s.t / 0.5); kout = kit.cubic_in(kit.window(s.t, s.dur - 0.3, 0.3))
    f = s.face
    for d_i, tile in enumerate(tiles):
        sh, (dx, dy) = kit.shadow(tile, blur=int(22 * u), alpha=0.55, offset=(0, int(16 * u)))
        if L['mode'] == 'above':
            x = s.W / 2 + (d_i - (digits - 1) / 2) * (cw + L['gap'])
        else:
            half = (L['gap'] + cw) / 2
            x = f.cx + (d_i - (digits - 1) / 2) * 2 * half
        y = L['y'] + (f.cy - _face0(s, p).cy) * 0.6 - (1 - kin) * 0.12 * s.H - kout * 0.1 * s.H
        sc = (0.9 + 0.1 * kin) * (1 + 0.02 * s.p)
        s.behind(kit.scaled(sh, sc), x, y, 'c', alpha=min(1, s.t / 0.12) * (1 - kout))


# =====================================================================
# 7. tier-list
# =====================================================================
_TIERS = [('S', (255, 107, 107)), ('A', (255, 169, 77)), ('B', (255, 212, 59)), ('C', (169, 227, 75)), ('D', (105, 219, 124)), ('F', (77, 171, 247))]


def _item_tile(txt, h, u, fill=(42, 42, 48), fg=(255, 255, 255), star=False):
    f = kit.font('sans', h * 0.40, 700)
    tw = f.getlength(txt); w = int(tw + h * 0.7 + (h * 0.5 if star else 0))
    im = kit.rounded(w, h, h * 0.22, fill=fill + (255,), outline=(255, 255, 255, 40), width=max(1, int(2 * u)))
    d = ImageDraw.Draw(im); x = h * 0.35
    if star:
        c = (x + h * 0.18, h / 2); r = h * 0.2
        pts = []
        for k in range(8):
            rr = r if k % 2 == 0 else r * 0.32; a = k * math.pi / 4 - math.pi / 2
            pts.append((c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)))
        d.polygon(pts, fill=fg + (255,)); x += h * 0.5
    l, t, r_, b = f.getbbox(txt)
    d.text((x - l, h / 2 - (t + b) / 2), txt, font=f, fill=fg + (255,))
    return im


@effect('tier-list', 'Numbers', 'a tier list drops in and your pick flies into S tier')
def tier_list(s, p):
    u = s.u
    title = str(p.get('title', 'best way to edit a reel?')); item = str(p.get('item', 'Claude'))
    others = p.list('others', 'templates|timelines|apps|doing it by hand')

    def layout():
        f0 = _face0(s, p)
        bx, bw = 0.045 * s.W, 0.91 * s.W
        tf = kit.fit_font(title, 'serif-italic', bw, 72 * u)
        tl = kit.text_layer(title, tf, fill=(240, 200, 105, 255))
        ty = max(f0.chin + 0.035 * s.H, 0.27 * s.H)
        by = ty + tl.height + 0.018 * s.H
        bh = min(0.965 * s.H - by, 0.62 * s.H); rh = bh / 6
        # static board (letters + tracks), drawn over a glass panel each frame
        board = Image.new('RGBA', (int(bw), int(bh)), (0, 0, 0, 0)); d = ImageDraw.Draw(board)
        m = 12 * u; lw = rh * 1.05
        lf = kit.font('heavy', rh * 0.46)
        for i, (L, c) in enumerate(_TIERS):
            y0 = m + i * (bh - 2 * m) / 6; y1 = y0 + (bh - 2 * m) / 6 - 8 * u
            d.rounded_rectangle((m, y0, m + lw, y1), int(14 * u), fill=c + (255,))
            _text_center(d, (m + lw / 2, (y0 + y1) / 2), L, lf, (24, 22, 26, 255))
            d.rounded_rectangle((m + lw + 10 * u, y0, bw - m, y1), int(14 * u), fill=(255, 255, 255, 18))
        slot_h = (bh - 2 * m) / 6 - 8 * u
        ih = int(slot_h * 0.66)
        tiers = {}
        for k, o in enumerate(others):
            tiers.setdefault(2 + k % 4, []).append(o)
        items = []
        for row, names in tiers.items():
            x = m + lw + 26 * u
            for nm in names:
                im = _item_tile(nm, ih, u)
                items.append((im, x, row)); x += im.width + 14 * u
        hero = _item_tile(item, ih, u, fill=(217, 119, 87), star=True)
        return dict(tl=tl, ty=ty, bx=bx, by=by, bw=bw, bh=bh, board=board, m=m, lw=lw, slot_h=slot_h,
                    items=items, hero=hero, rows=[(m + i * (bh - 2 * m) / 6) for i in range(6)])
    L = s.once('layout', layout)
    env = s.env(0.0, 0.3)
    kin = kit.expo_out(kit.window(s.t, 0.0, 0.6))
    dy = (1 - kin) * 0.55 * s.H + kit.cubic_in(kit.window(s.t, s.dur - 0.3, 0.3)) * 0.4 * s.H
    t_fly = kit.clamp(0.40 * s.dur, 0.5, max(0.5, s.dur - 1.0)); fly = 0.6
    land = kit.window(s.t, t_fly + fly, 0.5)
    shake = math.exp(-max(0, s.t - t_fly - fly) * 12) * (s.t > t_fly + fly) * 8 * u
    bx, by = L['bx'], L['by'] + dy + math.sin(s.t * 50) * shake
    bw, bh = int(L['bw']), int(L['bh'])
    # title
    kt = kit.expo_out(kit.window(s.t, 0.15, 0.5))
    tl, (tdx, tdy) = kit.shadow(L['tl'], blur=int(12 * u), alpha=0.6, offset=(0, int(4 * u)))
    s.over(kit.scaled(tl, 0.92 + 0.08 * kt + 0.015 * s.p), s.W / 2, L['ty'] + L['tl'].height / 2 + dy * 0.6, 'c', alpha=kt * env ** 3)
    # glass + board
    panel = kit.glass(s.rgb, (bx, by, bw, bh), 26 * u, tint=(12, 12, 16), tint_a=0.62, blur=18 * u, border_a=0.25)
    psh, (pdx, pdy) = kit.shadow(panel, blur=int(30 * u), alpha=0.5, offset=(0, int(18 * u)))
    s.over(psh, bx + pdx, by + pdy, alpha=env)
    s.over(L['board'], bx, by, alpha=env)
    # S row flash on landing
    if land > 0:
        y0 = L['rows'][0]
        fl = kit.rounded(bw - L['m'] * 2, L['slot_h'], 14 * u, fill=(255, 140, 90, 255))
        s.over(fl, bx + L['m'], by + y0, alpha=0.35 * (1 - land) * env)
    for i, (im, x, row) in enumerate(L['items']):
        k = kit.stagger(s.t, i, 0.45, 0.08, 0.35)
        if k <= 0: continue
        sc = 0.6 + 0.4 * kit.back_out(k)
        cy = by + L['rows'][row] + L['slot_h'] / 2
        s.over(kit.scaled(im, sc), bx + x + im.width / 2, cy, 'c', alpha=min(1, k * 3) * env)
    # the hero flies into S tier
    k = kit.window(s.t, t_fly, fly)
    if k > 0:
        hero = L['hero']
        ex = bx + L['m'] + L['lw'] + 26 * u + hero.width / 2; ey = by + L['rows'][0] + L['slot_h'] / 2
        sx, sy = s.W * 1.05, s.H * 1.02
        cx_, cy_ = s.W * 0.75, ey - 0.12 * s.H
        e = kit.cubic_out(k)
        x = (1 - e) ** 2 * sx + 2 * (1 - e) * e * cx_ + e * e * ex
        y = (1 - e) ** 2 * sy + 2 * (1 - e) * e * cy_ + e * e * ey
        rot = 28 * (1 - e)
        sc = 1.35 - 0.35 * e
        if k >= 1:
            tl_ = max(0, s.t - t_fly - fly)
            sc = 1 + 0.18 * math.exp(-tl_ * 7) * math.cos(tl_ * 26) + 0.01 * s.p
        g, (gx, gy) = kit.glow(hero, radius=int(16 * u), strength=1.2 if k >= 1 else 0.6, color=(255, 140, 90))
        s.over(kit.scaled(kit.rotated(g, rot), sc), x, y, 'c', alpha=env)


# =====================================================================
# 8. chart-split
# =====================================================================
_CREAM = (243, 237, 226); _RUST = (200, 85, 61); _HONEY = (242, 193, 78); _DARK = (43, 33, 24)


def _ink(a):
    return tuple(int(c * a + k * (1 - a)) for c, k in zip(_DARK, _CREAM)) + (255,)


@effect('chart-split', 'Numbers', 'a cream bar-chart card drops over the top half, you sit below in a card', aliases=('chart',))
def chart_split(s, p):
    u = s.u
    title = str(p.get('title', 'My top 7 reels')); by_ = str(p.get('by', 'by views'))
    bars = [(a, b) for a, b in p.pairs('bars', 'Glass::197K|Orbit::140K|Spot::124K|Docu::83K|Split::50K|Tasks::40K|Render::38K')][:10]
    vals = [kit.parse_number(b) for _, b in bars]
    k_in = kit.expo_out(kit.window(s.t, 0.0, 0.6)); k_out = kit.cubic_in(kit.window(s.t, s.dur - 0.32, 0.32))
    k = k_in * (1 - k_out)
    # 1) person shifts down into a rounded video card
    f0 = _face0(s, p)
    split = 0.47 * s.H
    vy0 = split + 0.012 * s.H
    target = vy0 + (s.H - vy0) * 0.30
    s.camera(1.0, dy=(target - f0.cy) * k)
    m = 0.035 * s.W * k
    vx0, vx1 = m, s.W - m; vy = vy0 * k + 0 * (1 - k); vy1 = s.H - m
    small = cv2.resize(s.rgb, (s.W // 6, s.H // 6), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 6), (s.W, s.H), interpolation=cv2.INTER_LINEAR) * 0.35
    mask = Image.new('L', (s.W, s.H), 0)
    ImageDraw.Draw(mask).rounded_rectangle((int(vx0), int(vy), int(vx1), int(vy1)), int(44 * u * k), fill=255)
    mk = np.asarray(mask.filter(ImageFilter.GaussianBlur(0.8)), np.float32)[..., None] / 255
    s.rgb[:] = s.rgb * mk + bg * (1 - mk)
    # 2) cream card
    cw, chh = int(s.W - 2 * 0.035 * s.W), int(split - 0.03 * s.H)
    cx0, cy0 = 0.035 * s.W, 0.022 * s.H

    def static():
        im = kit.rounded(cw, chh, 40 * u, fill=_CREAM + (255,)); d = ImageDraw.Draw(im)
        pad = 52 * u
        ft = kit.fit_font(title + '  ' + by_, 'serif', cw - 2 * pad, 68 * u, 800)
        fb = kit.font('serif-italic', ft.size)
        tx, ty = pad, pad * 0.75
        d.text((tx, ty), title, font=ft, fill=_DARK + (255,))
        bx = tx + ft.getlength(title + '  ')
        d.text((bx, ty), by_, font=fb, fill=_RUST + (255,))
        asc, _ = ft.getmetrics()
        ul = (bx, ty + asc + 10 * u, bx + fb.getlength(by_), ft.size)
        return im, pad, ty + asc + 34 * u, ul
    base, pad, top, ul = s.once('card', static)
    im = base.copy(); d = ImageDraw.Draw(im)
    # underline draws itself
    ku = kit.cubic_out(kit.window(s.t, 0.35, 0.4))
    if ku > 0:
        x0, y0, x1, _ = ul
        d.rounded_rectangle((x0, y0, x0 + (x1 - x0) * ku, y0 + 5 * u), radius=3 * u, fill=_RUST + (255,))
    # chart area
    lab_f = kit.font('sans', 28 * u, 600); val_f = kit.font('sans', 34 * u, 800)
    gx0, gx1 = pad, cw - pad
    gy1 = chh - pad * 0.55 - 40 * u; gy0 = top + 70 * u
    vmax = max(vals) if vals else 1
    n = max(1, len(bars)); slot = (gx1 - gx0) / n; bwid = slot * 0.62
    for g in range(1, 4):
        yy = gy1 - (gy1 - gy0) * g / 4
        d.line((gx0, yy, gx1, yy), fill=_ink(0.10), width=max(1, int(2 * u)))
    d.line((gx0, gy1, gx1, gy1), fill=_ink(0.4), width=max(1, int(3 * u)))
    for i, ((lab, vs), v) in enumerate(zip(bars, vals)):
        kb = kit.expo_out(kit.stagger(s.t, i, 0.45, 0.09, 0.7))
        x = gx0 + slot * i + (slot - bwid) / 2
        hgt = (gy1 - gy0) * (v / vmax) * kb
        col = _HONEY if i == 0 else _RUST
        if hgt > 2:
            d.rounded_rectangle((x, gy1 - hgt, x + bwid, gy1), radius=min(12 * u, hgt / 2), fill=col + (255,))
            d.rectangle((x, gy1 - min(hgt, 12 * u), x + bwid, gy1), fill=col + (255,))
        _text_center(d, (x + bwid / 2, gy1 + 26 * u), lab, lab_f, _ink(0.75))
        kv = kit.stagger(s.t, i, 0.75, 0.09, 0.3)
        if kv > 0:
            vt = kit.text_layer(vs, val_f, fill=_DARK + (255,))
            vt = kit.scaled(vt, 0.5 + 0.5 * kit.back_out(kv, 2.0))
            vt = kit.faded(vt, min(1, kv * 2.5))
            im.alpha_composite(vt, (int(x + bwid / 2 - vt.width / 2), int(gy1 - hgt - vt.height - 12 * u)))
    # dashed average line
    ka = kit.cubic_out(kit.window(s.t, min(1.2, 0.45 * s.dur), 0.5))
    if ka > 0 and vals:
        avg = sum(vals) / len(vals); ya = gy1 - (gy1 - gy0) * avg / vmax
        xe = gx0 + (gx1 - gx0) * ka; dash = 18 * u; x = gx0
        while x < xe:
            d.line((x, ya, min(x + dash * 0.6, xe), ya), fill=_ink(0.7), width=max(1, int(3 * u))); x += dash
        if ka > 0.6:
            at = kit.text_layer('avg ' + kit.number_text(avg, bars[0][1]), kit.font('serif-italic', 32 * u), fill=_ink(0.85))
            at = kit.faded(at, (ka - 0.6) / 0.4)
            im.alpha_composite(at, (int(gx1 - at.width), int(ya - at.height - 10 * u)))
    sh, (dx, dy) = kit.shadow(im, blur=int(30 * u), alpha=0.5, offset=(0, int(16 * u)))
    ydrop = -(1 - k) * (chh + 0.08 * s.H)
    s.over(sh, cx0 + dx, cy0 + dy + ydrop)
