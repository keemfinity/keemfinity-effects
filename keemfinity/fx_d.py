"""Your Reels: the user's reels orbit, fan, pile up and fill a profile grid around them."""
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw

from . import kit
from .engine import effect

G = 'Your Reels'


# ---------- private helpers ----------
def _reel_card(img, w, h, r, border, border_col=(255, 255, 255)):
    """Rounded reel card with a thin border."""
    w, h, b = int(w), int(h), max(1, int(border))
    base = kit.rounded(w, h, r, fill=border_col + (255,))
    base.alpha_composite(kit.card(img, w - 2 * b, h - 2 * b, max(1, r - b)), (b, b))
    return base


def _shadowed(im, u, blur=18, alpha=0.5, dy=10):
    sh, _ = kit.shadow(im, blur=max(2, int(blur * u)), alpha=alpha, offset=(0, int(dy * u)))
    return sh


def _shade(im, k):
    if k >= 0.995: return im
    a = np.asarray(im).astype(np.float32); a[..., :3] *= k
    return Image.fromarray(a.clip(0, 255).astype(np.uint8), 'RGBA')


def _xf(im, scale=1.0, deg=0.0):
    return kit.rotated(kit.scaled(im, max(0.02, scale)), deg)


def _put(s, im, x, y, front, alpha=1.0):
    """front 0 = behind the person, 1 = in front; values between crossfade so nothing pops."""
    front = kit.clamp(front)
    if front < 1: s.behind(im, x, y, 'c', alpha)
    if front > 0: s.over(im, x, y, 'c', alpha * front)


def _shadow_text(txt, f, fill=(255, 255, 255, 255), u=1.0, a=0.6):
    t = kit.text_layer(txt, f, fill=fill)
    sh, _ = kit.shadow(t, blur=max(1, int(4 * u)), alpha=a, offset=(0, max(1, int(2 * u))))
    return sh


def _eye_icon(sz, col=(255, 255, 255)):
    sz = int(sz); im = Image.new('RGBA', (sz, sz), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    lw = max(1, int(sz * 0.11))
    d.ellipse((lw, sz * 0.2, sz - lw, sz * 0.8), outline=col + (255,), width=lw)
    r = sz * 0.17; d.ellipse((sz / 2 - r, sz / 2 - r, sz / 2 + r, sz / 2 + r), fill=col + (255,))
    return im


def _ring_layer(pts, w, col, alpha):
    """Polyline through pts (frame coords) on a tight layer. Returns (layer, x0, y0) or None."""
    if len(pts) < 2: return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]; m = w * 3 + 4
    x0, y0 = int(min(xs) - m), int(min(ys) - m)
    im = Image.new('RGBA', (int(max(xs) - x0 + m), int(max(ys) - y0 + m)), (0, 0, 0, 0))
    ImageDraw.Draw(im).line([(x - x0, y - y0) for x, y in pts], fill=col + (int(255 * alpha),), width=max(1, int(w)), joint='curve')
    g, (dx, dy) = kit.glow(im, radius=max(2, int(w * 3)), strength=1.2)
    return g, x0 + dx, y0 + dy


def _phone(img, w, u):
    """A phone: black bezel, thin metal rim, dynamic island, reel on screen."""
    w = int(w); h = int(w * 2.05); r = int(w * 0.17); b = max(2, int(w * 0.045))
    ph = kit.rounded(w, h, r, fill=(12, 12, 14, 255), outline=(120, 120, 128, 255), width=max(1, int(2 * u)))
    ph.alpha_composite(kit.card(img, w - 2 * b, h - 2 * b, r - b), (b, b))
    iw, ih = int(w * 0.28), int(w * 0.08)
    ph.alpha_composite(kit.rounded(iw, ih, ih / 2, fill=(0, 0, 0, 255)), ((w - iw) // 2, b + int(w * 0.04)))
    # glass sheen across the screen
    sheen = kit.gradient(w, h, (255, 255, 255), (255, 255, 255), alpha=(34, 0))
    sheen = kit.mask_rounded(sheen, r); ph.alpha_composite(sheen)
    return _shadowed(ph, u, blur=26, alpha=0.6, dy=16)


# ---------- 1. orbit ----------
@effect('orbit', G, 'your reels orbit your head: in front at the bottom, behind at the top', aliases=('reel-orbit', 'orbit-reels'))
def orbit(s, p):
    n = int(kit.clamp(p.num('count', 8), 3, 16)); speed = p.num('speed', 0.16); size = p.num('size', 1.0)
    u, W, f = s.u, s.W, s.face
    cw = 0.2 * W * size; ch = cw * 16 / 9

    def build():
        return [_shadowed(_reel_card(im, cw, ch, cw * 0.09, 3.2 * u), u, 16, 0.55, 10) for im in p.images(n)]
    cards = s.once('cards', build)

    env = s.env(0.01, 0.35)
    rx, ry = 0.40 * W, 1.0 * f.h
    cx, cy = f.cx, f.cy + 0.55 * f.h
    tilt = math.radians(-7 + 3 * math.sin(0.7 * s.t))
    ct, st_ = math.cos(tilt), math.sin(tilt)
    open_k = kit.lerp(0.55, 1.0, kit.expo_out(s.t / 0.9))          # ring unfurls from the head
    spin = 2 * math.pi * speed * s.t - 2.2 * (1 - kit.expo_out(s.t / 1.3))   # whoosh in, then steady
    exit_k = kit.cubic_out((s.dur - s.t) / 0.35)

    def pos(th, k=1.0):
        ex, ey = rx * math.cos(th) * k, ry * math.sin(th) * k
        return cx + ex * ct - ey * st_, cy + ex * st_ + ey * ct

    # faint orbit line: back half behind the head, front half in front
    ra = 0.32 * kit.cubic_out(s.t / 0.6) * env
    for lo, front in ((math.pi, 0.0), (0.0, 1.0)):
        pts = [pos(lo + math.pi * j / 64, open_k) for j in range(65)]
        L = _ring_layer(pts, 2 * u, (255, 255, 255), 0.9)
        if L:
            im, x0, y0 = L
            (s.over if front else s.behind)(im, x0, y0, 'tl', ra)

    items = []
    for i in range(n):
        th = spin + 2 * math.pi * i / n
        d = math.sin(th)                                    # +1 front (bottom), -1 back (top)
        items.append((d, i, th))
    for d, i, th in sorted(items):
        x, y = pos(th, open_k)
        k01 = (d + 1) / 2
        pop = kit.spring(s.t - 0.05 - 0.045 * i, freq=3.2, decay=5.5)
        sc = kit.lerp(0.55, 1.08, k01) * max(0.0, pop) * kit.lerp(0.6, 1, exit_k)
        if sc < 0.03: continue
        deg = -math.degrees(tilt) - 9 * math.cos(th)
        im = _shade(_xf(cards[i], sc, deg), kit.lerp(0.58, 1.0, k01))
        _put(s, im, x, y, kit.smooth((d + 0.2) / 0.4), env * kit.clamp(pop * 3))


# ---------- 2. phone-fan ----------
@effect('phone-fan', G, 'a hand of phones playing your reels fans out in front of you, headline behind your head', aliases=('phones', 'fan'))
def phone_fan(s, p):
    u, W, H = s.u, s.W, s.H
    lines = [x.strip() for x in p.text('48 effects.|zero editing apps.').split('|') if x.strip()][:2]
    n = 5; pw = 0.235 * W

    def build_phones():
        return [_phone(im, pw, u) for im in p.images(n)]
    phones = s.once('phones', build_phones)

    def build_head():
        out = []
        f1 = kit.fit_font(lines[0], 'heavy', W * 0.84, W * 0.105)
        out.append(_shadow_text(lines[0], f1, u=u, a=0.5))
        if len(lines) > 1:
            f2 = kit.fit_font(lines[1], 'heavy', W * 0.78, f1.size * 0.62)
            t = kit.text_layer(lines[1], f2)
            px, py = int(f2.size * 0.32), int(f2.size * 0.22)
            box = kit.rounded(t.width + 2 * px, t.height + 2 * py, 6 * u, fill=(0, 0, 0, 255))
            out.append((box, t, px, py))
        return out
    head = s.once('head', build_head)
    env = s.env(0.01, 0.3)

    # headline: behind the head
    y = H * 0.022
    k1 = kit.expo_out(s.t / 0.5)
    l1 = kit.scaled(head[0], 1 + 0.025 * s.p)
    s.behind(l1, W / 2, y - (1 - k1) * 0.06 * H, 'tc', env * k1)
    y += l1.height * 0.82
    if len(head) > 1:
        box, t, px, py = head[1]
        k2 = kit.expo_out(kit.window(s.t, 0.18, 0.5))
        if k2 > 0.01:
            bw = max(2, int(box.width * k2))
            layer = Image.new('RGBA', box.size, (0, 0, 0, 0))
            layer.paste(box.crop((0, 0, bw, box.height)), (0, 0))
            kt = kit.cubic_out(kit.window(s.t, 0.3, 0.35))
            if kt > 0: layer.alpha_composite(kit.faded(t, kt), (px, py + int((1 - kt) * t.height * 0.4)))
            s.behind(kit.scaled(layer, 1 + 0.025 * s.p), W / 2, y, 'tc', env)

    # phones: fan out from below, in front of the lap
    R = 0.8 * W; pvx, pvy = W / 2, H * 0.79 + R
    breathe = 1 + 0.04 * math.sin(1.6 * s.t)
    order = sorted(range(n), key=lambda i: abs(i - (n - 1) / 2), reverse=True)
    for i in order:
        a_tgt = kit.lerp(-20, 20, i / (n - 1))
        delay = 0.12 + 0.07 * abs(i - (n - 1) / 2) * 2
        e = kit.expo_out(kit.window(s.t, delay, 0.6))
        if e <= 0.001: continue
        a = a_tgt * max(0.0, kit.spring(s.t - delay, freq=2.6, decay=5)) * breathe
        rise = (1 - e) * H * 0.55 + (1 - kit.cubic_out((s.dur - s.t) / 0.3)) * H * 0.5
        x = pvx + R * math.sin(math.radians(a)); y = pvy - R * math.cos(math.radians(a)) + rise
        s.over(_xf(phones[i], 1.0, -a), x, y, 'c', min(1.0, e * 2))


# ---------- 3. place-in-grid ----------
@effect('place-in-grid', G, 'your video shrinks into the middle of your profile grid as your reels pop in around it', aliases=('grid', 'in-grid'))
def place_in_grid(s, p):
    u, W, H = s.u, s.W, s.H
    views = p.list('views', '2.4M|847K|1.2M|312K|3.4M|596K|268K|1.8M|3.1M')
    views = (views * 9)[:9] if views else [''] * 9
    gap = 0.006 * W; gw = 0.94 * W
    tw = (gw - 2 * gap) / 3; th = tw * 4 / 3
    gx0 = (W - gw) / 2; gy0 = H / 2 - 1.5 * th - gap
    tiles_xy = [(gx0 + c * (tw + gap), gy0 + r * (th + gap)) for r in range(3) for c in range(3)]
    cr = 0.03 * W

    def label(v):
        f = kit.font('sans', 0.03 * W, 700)
        t = kit.text_layer(v, f); ey = _eye_icon(f.size * 1.05)
        im = Image.new('RGBA', (int(ey.width + 0.012 * W + t.width), max(ey.height, t.height) + 4), (0, 0, 0, 0))
        im.alpha_composite(ey, (0, (im.height - ey.height) // 2)); im.alpha_composite(t, (int(ey.width + 0.012 * W), (im.height - t.height) // 2))
        sh, _ = kit.shadow(im, blur=max(1, int(5 * u)), alpha=0.75, offset=(0, int(2 * u)))
        return sh

    def build_tiles():
        ims = p.images(8); out = []
        for k, im in enumerate(ims):
            c = kit.card(im, int(tw), int(th), cr)
            c.alpha_composite(kit.gradient(int(tw), int(th), (0, 0, 0), (0, 0, 0), alpha=(0, 110)), (0, 0))
            c = kit.mask_rounded(c, cr)
            lb = label(views[k]); c.alpha_composite(lb, (int(0.02 * W) - (lb.width - lb.width) , int(th - lb.height - 0.012 * W)))
            out.append(_shadowed(c, u, 14, 0.45, 6))
        return out
    tiles = s.once('tiles', build_tiles)
    center_label = s.once('clabel', lambda: label(views[8]))

    def build_header():
        f1 = kit.font('sans', 0.042 * W, 800); f2 = kit.font('sans', 0.026 * W, 500)
        handle = p.get('label', 'keemfinity').lstrip('@')
        a = kit.text_layer(handle, f1); b = kit.text_layer('48 posts  ·  195K followers', f2, fill=(255, 255, 255, 200))
        im = Image.new('RGBA', (max(a.width, b.width), a.height + b.height + int(0.016 * W)), (0, 0, 0, 0))
        im.alpha_composite(a, (0, 0)); im.alpha_composite(b, (0, a.height + int(0.016 * W)))
        return im
    header = s.once('header', build_header)

    frame = s.rgb.copy()
    # frosted backdrop of the live frame
    e0 = kit.expo_out(s.t / 0.75)
    sm = cv2.resize(frame, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), 3.0)
    bg = cv2.resize(sm, (W, H), interpolation=cv2.INTER_LINEAR)
    bg = bg * 0.62 + 0.05
    drift = 1 + 0.025 * s.p
    out_k = kit.cubic_out((s.dur - s.t) / 0.3)
    mix = e0 * out_k
    s.rgb[:] = frame * (1 - mix) + bg * mix

    # avatar + handle header
    av = s.once('avatar', lambda: kit.mask_rounded(kit.card(kit.to_pil(frame).crop((int(s.face.cx - s.face.w * 0.75), int(s.face.cy - s.face.h * 0.8),
                                                                        int(s.face.cx + s.face.w * 0.75), int(s.face.cy + s.face.h * 0.7))), int(0.13 * W), int(0.13 * W), 0), int(0.065 * W)))
    kh = kit.expo_out(kit.window(s.t, 0.35, 0.5)) * out_k
    if kh > 0.01:
        ring = kit.rounded(av.width + int(8 * u), av.height + int(8 * u), (av.width + 8 * u) / 2, fill=(255, 255, 255, 255))
        ring.alpha_composite(av, (int(4 * u), int(4 * u)))
        hy = gy0 - 0.2 * W + (1 - kh) * 0.03 * W
        s.over(ring, gx0 + 0.01 * W, hy, 'tl', kh)
        s.over(header, gx0 + 0.01 * W + ring.width + 0.035 * W, hy + ring.height / 2, 'cl', kh)

    # other tiles pop in, clockwise
    order = [0, 1, 2, 5, 8, 7, 6, 3]
    for j, k in enumerate(order):
        e = kit.window(s.t, 0.55 + 0.045 * j, 0.4)
        if e <= 0: continue
        sc = kit.back_out(e, 1.4) * drift
        ti = k if k < 4 else k - 1
        x, y = tiles_xy[k]
        gx = W / 2 + (x + tw / 2 - W / 2) * drift; gy = H / 2 + (y + th / 2 - H / 2) * drift
        s.over(kit.scaled(tiles[ti], max(0.05, sc)), gx, gy, 'c', min(1.0, e * 3) * out_k)

    # the live frame shrinks into the centre tile (cropped to 3:4 around the face)
    e = kit.expo_out(s.t / 0.7) * out_k
    ch_ = W * 4 / 3; cy0 = kit.clamp(s.face.cy - 0.32 * ch_, 0, H - ch_)
    sx0, sy0, sx1, sy1 = 0, cy0 * e, W, kit.lerp(H, cy0 + ch_, e)
    cx_, cy_ = tiles_xy[4][0] + tw / 2, tiles_xy[4][1] + th / 2
    tw_d = kit.lerp(W, tw * drift, e); th_d = kit.lerp(H, th * drift, e)
    dx = kit.lerp(W / 2, W / 2 + (cx_ - W / 2) * drift, e); dy = kit.lerp(H / 2, H / 2 + (cy_ - H / 2) * drift, e)
    crop = frame[int(sy0):int(sy1), int(sx0):int(sx1)]
    tile = cv2.resize(crop, (max(2, int(tw_d)), max(2, int(th_d))), interpolation=cv2.INTER_AREA)
    timg = kit.mask_rounded(kit.to_pil(tile), kit.lerp(0, cr, e))
    if e > 0.5:
        lb = center_label; timg.alpha_composite(kit.faded(lb, (e - 0.5) * 2), (int(0.02 * W), int(timg.height - lb.height - 0.012 * W)))
    if e > 0.3:
        ImageDraw.Draw(timg).rounded_rectangle((0, 0, timg.width - 1, timg.height - 1), int(kit.lerp(0, cr, e)),
                                               outline=(255, 255, 255, int(230 * (e - 0.3) / 0.7)), width=max(2, int(4 * u)))
    if e > 0.02:
        timg = _shadowed(timg, u, 20, 0.55 * e, 8)
    s.over(timg, dx, dy, 'c')


# ---------- 4. carousel-spin ----------
@effect('carousel-spin', G, 'a 3D ring of your reels spins around you and slows to a stop', aliases=('carousel', 'reel-ring'))
def carousel_spin(s, p):
    n = int(kit.clamp(p.num('count', 10), 4, 16)); size = p.num('size', 1.0); turns = p.num('turns', 1.5)
    u, W, H, f = s.u, s.W, s.H, s.face
    cw = 0.23 * W * size; ch = cw * 16 / 9

    def build():
        out = []
        for im in p.images(n):
            c = _reel_card(im, cw, ch, cw * 0.08, 2.5 * u)
            out.append((_shadowed(c, u, 14, 0.5, 8), _shadowed(c.transpose(Image.FLIP_LEFT_RIGHT), u, 14, 0.5, 8)))
        return out
    cards = s.once('cards', build)
    env = s.env(0.01, 0.35)
    R = 0.40 * W; tilt_y = 0.25 * W; focal = 3.5 * R
    rcx, rcy = f.cx, f.chin + 0.2 * W
    spin_len = max(1.0, min(s.dur * 0.8, s.dur - 0.5))
    sp = kit.cubic_out(s.t / spin_len) ** 0.9
    base = -2 * math.pi * turns * (1 - sp) + 0.08 * s.t          # fast, decelerates, then a slow creep
    vel = turns * 3.0 * (1 - kit.clamp(s.t / spin_len)) ** 2 / spin_len   # rough angular speed (turns/s)
    ghosts = 2 if vel > 0.4 else 0
    items = []
    for i in range(n):
        for gk in range(ghosts, -1, -1):
            th = base + 2 * math.pi * i / n - gk * 0.06 * vel
            items.append((math.cos(th), i, th, gk))
    for z, i, th, gk in sorted(items):
        x3 = R * math.sin(th)
        persp = focal / (focal - R * z)
        x = rcx + x3 * persp; y = rcy + tilt_y * z
        face_w = abs(math.cos(th))                       # cards face outward: edge-on at the sides
        pop = kit.spring(s.t - 0.04 * i, freq=3.0, decay=6)
        sc = 0.8 * persp * max(0.0, pop)
        if sc < 0.03: continue
        im = cards[i][0 if z >= 0 else 1]
        im = kit.scaled(im, sc)
        im = im.resize((max(2, int(im.width * (0.18 + 0.82 * face_w))), im.height), Image.BILINEAR)
        im = _shade(im, kit.lerp(0.5, 1.0, (z + 1) / 2) * (0.85 + 0.15 * face_w))
        a = env * kit.clamp(pop * 3) * (1.0 if gk == 0 else 0.22 / gk)
        _put(s, im, x, y, kit.smooth((z + 0.15) / 0.3), a)


# ---------- 5. profile-wall ----------
@effect('profile-wall', G, 'a wall of your posts fills in behind you while a follower counter ticks up', aliases=('wall', 'profile'))
def profile_wall(s, p):
    u, W, H, f = s.u, s.W, s.H, s.face
    gap = 0.008 * W; tw = (W - 2 * gap) / 3; th = tw * 4 / 3
    rows = int(math.ceil(H / (th + gap))) + 1
    n = rows * 3

    def build():
        return [_shade(kit.card(im, int(tw), int(th), 0.012 * W), 0.62) for im in p.images(n)]
    tiles = s.once('tiles', build)
    env = s.env(0.2, 0.3)
    s.backdrop((10, 10, 12), alpha=env)
    scroll = -0.06 * W * s.t                               # the wall slowly drifts up
    for k in range(n):
        r, c = divmod(k, 3)
        e = kit.window(s.t, 0.1 + 0.06 * k, 0.35)
        if e <= 0: continue
        x = c * (tw + gap) + tw / 2; y = r * (th + gap) + th / 2 + scroll
        if y - th > H or y + th < 0: continue
        sc = kit.lerp(0.82, 1.0, kit.expo_out(e))
        s.behind(kit.scaled(tiles[k], sc), x, y, 'c', min(1.0, e * 2.5) * env)

    # glass follower counter under the chin
    target = p.get('count', '195,218'); label = p.get('label', '@keemfinity')
    v = kit.parse_number(target) * kit.cubic_out(kit.window(s.t, 0.35, max(0.8, min(2.2, s.dur - 1.0))))
    num = kit.number_text(v, target)
    fN = kit.font('display', 0.095 * W); fL = kit.font('sans', 0.03 * W, 600)
    nl = kit.text_layer(num, fN); ll = kit.text_layer(label, fL, fill=(255, 255, 255, 215))
    sub = s.once('sub', lambda: kit.text_layer('FOLLOWERS', kit.font('sans', 0.022 * W, 700), fill=(255, 255, 255, 170), tracking=0.006 * W))
    bw = max(0.48 * W, kit.text_size(kit.number_text(kit.parse_number(target), target), fN)[0] + 0.14 * W); bh = 0.235 * W
    k = kit.expo_out(kit.window(s.t, 0.25, 0.5)) * kit.cubic_out((s.dur - s.t) / 0.3)
    if k > 0.01:
        bx = f.cx - bw / 2; by = f.chin + 0.05 * W + (1 - k) * 0.05 * W
        bx = kit.clamp(bx, 0.03 * W, W * 0.97 - bw)
        panel = kit.glass(s.rgb, (bx, by, bw, bh), 0.05 * W, tint=(16, 16, 20), tint_a=0.5, blur=18 * u, border_a=0.45)
        panel.alpha_composite(ll, ((panel.width - ll.width) // 2, int(0.025 * W)))
        panel.alpha_composite(nl, ((panel.width - nl.width) // 2, int(0.025 * W + ll.height + 0.02 * W)))
        panel.alpha_composite(sub, ((panel.width - sub.width) // 2, int(bh - sub.height - 0.02 * W)))
        s.over(_shadowed(panel, u, 22, 0.45, 10), bx + bw / 2, by + bh / 2, 'c', k)


# ---------- 6. noise ----------
@effect('noise', G, 'your reels slam down behind your head into a growing messy pile', aliases=('pile', 'reel-pile'))
def noise(s, p):
    u, W, H = s.u, s.W, s.H
    size = p.num('size', 1.0); every = 0.25
    n = int(math.ceil(s.dur / every)) + 1
    cw = 0.3 * W * size; ch = cw * 16 / 9

    def layout():
        rng = np.random.default_rng(48)
        ax, ay = s.face.cx, s.face.cy + 0.06 * W
        out = []
        for i in range(n):
            spread = min(1.0, 0.55 + i * 0.05)                  # the pile grows outward
            side = -1 if i % 2 else 1
            dx = side * float(rng.uniform(0.12, 0.36)) * W * spread if i % 5 != 4 else float(rng.uniform(-0.06, 0.06)) * W
            dy = float(rng.uniform(-0.1, 0.12)) * W * spread
            deg = float(rng.uniform(-10, 22)) * -side
            out.append((kit.clamp(ax + dx, 0.2 * W, 0.8 * W), ay + dy, deg))
        return out
    pos = s.once('layout', layout)
    ims = p.images(n)
    cards = s.once('cards', lambda: {})
    settled = s.once('settled', lambda: {})
    env = s.env(0.01, 0.3)

    landed = [i for i in range(n) if s.t >= i * every]
    if not landed: return
    last = landed[-1]
    kick = 1 - kit.expo_out((s.t - last * every) / 0.18)
    s.camera(zoom=1 + 0.012 * kick * env)
    s.tint(lambda x: x * 0.78, alpha=0.8 * env * kit.cubic_out(s.t / 0.5), where='background')
    for i in landed:
        if i not in cards:
            cards[i] = _shadowed(_reel_card(ims[i], cw, ch, cw * 0.05, 0.016 * W), u, 22, 0.6, 14)
        x, y, deg = pos[i]
        e = kit.expo_out((s.t - i * every) / 0.22)
        depth = len(landed) - 1 - i
        shade = round(max(0.62, 1 - 0.06 * depth), 2)
        if e >= 0.999:
            key = (i, shade)
            if key not in settled:
                settled.pop((i, round(shade + 0.06, 2)), None)
                settled[key] = _shade(_xf(cards[i], 1.0, deg), shade)
            im = settled[key]
        else:
            im = _shade(_xf(cards[i], kit.lerp(1.45, 1.0, e), deg + (1 - e) * 14), shade)
        drift = 1 + 0.02 * s.p
        s.behind(im, W / 2 + (x - W / 2) * drift, y * drift - (1 - e) * 0.05 * W, 'c', env * min(1.0, e * 2.2))


# ---------- 7. lineup ----------
@effect('lineup', G, 'numbered tiles of your workflow fan out in an arc above your head', aliases=('steps-arc', 'tiles'))
def lineup(s, p):
    u, W, f = s.u, s.W, s.face
    names = p.list('text', 'Hook|Cut|Caption|Grade|Music|Post')[:8] or ['Hook']
    n = len(names); ts = 0.13 * W if n <= 6 else 0.11 * W

    def build():
        out = []
        fN = kit.font('display', ts * 0.46); fT = kit.font('sans', ts * 0.17, 700)
        for i, nm in enumerate(names):
            t = kit.rounded(ts, ts, ts * 0.18, fill=(250, 250, 248, 255))
            d = ImageDraw.Draw(t)
            nl = kit.text_layer(str(i + 1), fN, fill=(14, 14, 16, 255))
            tl = kit.text_layer(kit.wrap(nm, fT, ts * 0.9)[0] if fT.getlength(nm) > ts * 0.9 else nm,
                                kit.fit_font(nm, 'sans', ts * 0.86, ts * 0.17, 700), fill=(37, 99, 235, 255))
            t.alpha_composite(nl, (int((ts - nl.width) / 2), int(ts * 0.13)))
            t.alpha_composite(tl, (int((ts - tl.width) / 2), int(ts * 0.86 - tl.height)))
            out.append(_shadowed(t, u, 16, 0.5, 8))
        return out
    tiles = s.once('tiles', build)
    env = s.env(0.01, 0.3)
    # an arch: ends beside the head, middle just above the cap
    y_side = f.cy + 0.02 * W
    y_top = max(ts * 0.62, f.crown - 0.065 * W - ts / 2)
    rx, ry = 0.40 * W, y_side - y_top
    span = 80 if n >= 5 else 60
    for i in range(n):
        a = math.radians(kit.lerp(-span, span, i / max(1, n - 1)))
        tx, ty = f.cx + rx * math.sin(a), y_side - ry * math.cos(a)
        e = kit.window(s.t, 0.12 + 0.09 * i, 0.55)
        if e <= 0: continue
        ee = kit.expo_out(e)
        x = kit.lerp(f.cx, tx, ee); y = kit.lerp(f.cy - 0.05 * W, ty, ee)
        y += math.sin(1.8 * s.t + i * 0.9) * 0.005 * W
        sc = kit.back_out(kit.clamp(e * 1.2), 1.8) * (1 + 0.02 * s.p)
        out_e = kit.cubic_out((s.dur - s.t - 0.04 * (n - 1 - i)) / 0.3)
        im = _xf(tiles[i], max(0.05, sc * kit.lerp(0.5, 1, out_e)), -math.degrees(a) * 0.25)
        s.behind(im, x, y, 'c', env * min(1.0, e * 3) * out_e)


# ---------- 8. check-pills ----------
def _grid_paper(W, H, u):
    bg = np.empty((H, W, 3), np.float32); bg[:] = np.float32([247, 246, 241]) / 255
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    light = np.clip(1 - np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H * 0.35) / H) ** 2) * 0.55, 0, 1)
    bg *= (0.9 + 0.1 * light)[..., None]
    step = 0.045 * W; lw = max(1, int(round(1.4 * u)))
    minor = np.float32([196, 214, 244]) / 255; major = np.float32([150, 182, 236]) / 255
    for k, x in enumerate(np.arange(step / 2, W, step)):
        c = major if k % 5 == 2 else minor; bg[:, int(x):int(x) + lw] = c
    for k, y in enumerate(np.arange(step / 2, H, step)):
        c = major if k % 5 == 2 else minor; bg[int(y):int(y) + lw, :] = c
    # floor line: the room
    fy = int(H * 0.8); bg[fy:] *= np.linspace(0.97, 0.86, H - fy, dtype=np.float32)[:, None, None]
    return kit.vignette(bg, 0.18)


def _pin(sz, col=(230, 57, 70)):
    sz = int(sz); im = Image.new('RGBA', (sz, sz), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.ellipse((0, 0, sz - 1, sz - 1), fill=col + (255,))
    d.ellipse((sz * 0.22, sz * 0.18, sz * 0.48, sz * 0.44), fill=(255, 255, 255, 170))
    return im


@effect('check-pills', G, 'grid-paper room, your reels pinned up behind you, a cursor ticks off blue pills', aliases=('pills', 'checklist-pills'))
def check_pills(s, p):
    u, W, H, f = s.u, s.W, s.H, s.face
    names = p.list('text', 'Hook|Captions|Cuts|B-roll')[:6] or ['Hook']
    n = len(names)
    BLUE = (37, 99, 235); LIGHT = (120, 170, 255)
    paper = s.once('paper', lambda: _grid_paper(W, H, u))
    env = s.env(0.25, 0.3)

    # contact shadow of the person on the wall, then the paper
    sm = cv2.resize(s.alpha, (W // 6, H // 6), interpolation=cv2.INTER_AREA)
    sh = cv2.GaussianBlur(sm, (0, 0), 6)
    sh = cv2.resize(sh, (W, H), interpolation=cv2.INTER_LINEAR)
    M = np.float32([[1, 0, 0.02 * W], [0, 1, 0.015 * W]])
    sh = cv2.warpAffine(sh, M, (W, H))
    bg = paper * (1 - 0.22 * sh[..., None])
    s.backdrop(bg, alpha=env)

    # reel boards pinned behind the head
    boards = s.once('boards', lambda: [kit.rotated(_shadowed(_with_pin(_reel_card(im, 0.22 * W, 0.22 * W * 16 / 9, 0.012 * W, 0.012 * W), u), u, 14, 0.35, 8), d)
                                       for im, d in zip(p.images(4), (6, -5, -4, 7))])
    bpos = [(-0.33, -0.1), (0.33, -0.06), (-0.4, 0.36), (0.4, 0.4)]
    for i, (bxr, byr) in enumerate(bpos):
        e = kit.window(s.t, 0.1 + 0.08 * i, 0.45)
        if e <= 0: continue
        sc = kit.back_out(e, 1.3)
        s.behind(kit.scaled(boards[i], max(0.05, sc)), f.cx + bxr * W, f.cy + byr * W, 'c', env * min(1.0, e * 3))

    # pills
    fT = kit.font('sans', 0.036 * W, 700)

    def make_pill(txt, col, checked):
        t = kit.text_layer(txt, fT); ph = 0.082 * W; cs = ph * 0.62
        pw = t.width + 0.075 * W + (cs + 0.018 * W if checked else 0)
        im = kit.rounded(pw, ph, ph / 2, fill=col + (255,))
        x = int(0.0375 * W)
        if checked:
            c = kit.rounded(cs, cs, cs / 2, fill=(255, 255, 255, 255)); c.alpha_composite(kit.check_mark(cs, BLUE, cs * 0.16))
            im.alpha_composite(c, (int(ph * 0.19), int((ph - cs) / 2))); x = int(ph * 0.19 + cs + 0.018 * W)
        im.alpha_composite(t, (x, int((ph - t.height) / 2)))
        return _shadowed(im, u, 14, 0.35, 7)
    pills = s.once('pills', lambda: [(make_pill(t, BLUE, False), make_pill(t, BLUE, True), make_pill(t, LIGHT, False)) for t in names])
    side = [(-1, 0.07), (1, 0.2), (-1, 0.42), (1, 0.56), (-1, 0.7), (1, 0.84)]
    ppos = [(f.cx + sd * 0.29 * W, f.chin + yr * W) for sd, yr in side[:n]]
    ppos = [(kit.clamp(x, 0.2 * W, 0.8 * W), y) for x, y in ppos]

    t0 = min(0.95, 0.4 * s.dur); step = kit.clamp((s.dur - t0 - 0.4) / n, 0.15, 0.7)
    clicks = [t0 + k * step for k in range(n)]
    for k in range(n):
        e = kit.window(s.t, 0.35 + 0.09 * k, 0.4)
        if e <= 0: continue
        base, done, flash = pills[k]
        dt = s.t - clicks[k]
        im = done if dt >= 0.06 else base
        pulse = 1 + (0.12 * math.exp(-dt * 9) * math.sin(dt * 30) if dt >= 0 else 0) - (0.06 if -0.06 < dt < 0.06 else 0)
        sc = kit.back_out(e, 1.6) * pulse
        x, y = ppos[k]
        s.over(kit.scaled(im, max(0.05, sc)), x, y, 'c', env * min(1.0, e * 3))
        if 0 <= dt < 0.35:
            s.over(kit.scaled(flash, sc * (1 + dt * 0.5)), x, y, 'c', env * 0.7 * (1 - dt / 0.35))

    # cursor glides pill to pill and clicks
    cur = s.once('cursor', lambda: _shadowed(kit.cursor(0.06 * W, (255, 255, 255), (20, 20, 24)), u, 6, 0.4, 4))
    start = (W * 0.9, H * 0.95)
    targets = [(x + 0.02 * W, y + 0.01 * W) for x, y in ppos]
    if s.t > t0 - 0.55:
        prev, nxt, k_ = start, targets[0], 0
        for k in range(n):
            if s.t >= clicks[k]: prev, k_ = targets[k], k + 1
        nxt = targets[k_] if k_ < n else (prev[0] + 0.1 * W, prev[1] + 0.15 * W)
        m0 = (clicks[k_ - 1] + 0.12) if k_ > 0 else t0 - 0.55
        m1 = clicks[k_] - 0.06 if k_ < n else m0 + 0.6
        m = kit.smooth(kit.window(s.t, m0, max(0.05, m1 - m0)))
        cx = kit.lerp(prev[0], nxt[0], m); cy = kit.lerp(prev[1], nxt[1], m) - math.sin(m * math.pi) * 0.04 * W
        press = min((abs(s.t - c) for c in clicks), default=1)
        sc = 0.86 if press < 0.06 else 1.0
        s.over(kit.scaled(cur, sc), cx, cy, 'tl', env * kit.clamp((s.t - t0 + 0.55) / 0.2))


def _with_pin(card, u):
    sz = int(card.width * 0.11)
    out = Image.new('RGBA', (card.width, card.height + sz // 2), (0, 0, 0, 0))
    out.alpha_composite(card, (0, sz // 2)); out.alpha_composite(_pin(sz), ((card.width - sz) // 2, 0))
    return out
