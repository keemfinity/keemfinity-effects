"""Words behind you, part 1."""
from . import kit
from .engine import effect


@effect('backdrop-word', 'Words Behind You', 'one giant word drops in behind your head', aliases=('backdrop', 'word'))
def backdrop_word(s, p):
    word = p.text('KEEM').upper()
    col = p.color('color', (245, 242, 235))
    role = 'serif' if p.get('font') == 'serif' else 'display'
    big = s.once('word', lambda: kit.text_layer(word, kit.fit_font(word, role, s.W * 0.94, s.W * 0.42), fill=col + (255,)))
    k = kit.expo_out(s.t / 0.45)                           # drops in fast, long settle
    y = s.face.crown - big.height * 0.15 - (1 - k) * s.H * 0.18
    layer = kit.scaled(big, 1 + 0.03 * s.p)                # slow push: never dead still
    s.behind(layer, s.W / 2, y, 'tc', alpha=s.env(0.15, 0.2) * k)


# =====================================================================================================
# shared private helpers
# =====================================================================================================
import math
import os
from functools import lru_cache
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont


@lru_cache(maxsize=128)
def _font(role='sans', size=40, weight=None):
    """kit.font, but sets Inter's weight on the right axis (Inter's axes are [optical size, weight])."""
    if role == 'sans' and weight:
        f = ImageFont.truetype(os.path.join(kit.FONT_DIR, kit.FONTS['sans']), max(1, int(size)))
        try: f.set_variation_by_axes([kit.clamp(size * 0.5, 14, 32), int(weight)])
        except Exception: pass
        return f
    return kit.font(role, size, weight)


def _fit(txt, role, width, start, weight=None, floor=10):
    sz = start
    while sz > floor and kit.text_size(txt, _font(role, int(sz), weight))[0] > width: sz *= 0.94
    return _font(role, int(sz), weight)



def _pad(im, p):
    p = int(max(0, p))
    if p == 0: return im
    out = Image.new('RGBA', (im.width + 2 * p, im.height + 2 * p), (0, 0, 0, 0)); out.alpha_composite(im, (p, p)); return out


def _slam(im, k, big=1.45, blur=14.0):
    """k 0 -> 1: from bigger + blurred + transparent to sharp. Returns a layer to place by its centre/anchor."""
    if k >= 0.999: return im
    e = kit.cubic_out(k)
    out = kit.scaled(im, kit.lerp(big, 1.0, e))
    r = blur * (1 - e)
    if r > 0.3: out = _pad(out, r * 2.5).filter(ImageFilter.GaussianBlur(r))
    return kit.faded(out, kit.clamp(k * 2.2))


def _with_glow(im, radius, strength, color):
    g, off = kit.glow(im, radius=max(2, int(radius)), strength=strength, color=color)
    return g


def _fill_gradient(mask_im, top, bottom):
    """Recolour an RGBA text layer with a vertical gradient, keeping its alpha."""
    g = kit.gradient(mask_im.width, mask_im.height, top, bottom)
    g.putalpha(mask_im.getchannel('A')); return g


def _text_shadow(im, u, blur=10, alpha=0.55, dy=6):
    sh, off = kit.shadow(im, blur=max(1, int(blur * u)), alpha=alpha, offset=(0, int(dy * u)))
    return sh


def _head_top(s, x0=None, x1=None, thr=0.5):
    """Top row of the person around the face (uses the cut-out mask)."""
    f = s.face
    a = int(max(0, f.cx - f.w * 0.45)) if x0 is None else int(x0)
    b = int(min(s.W, f.cx + f.w * 0.45)) if x1 is None else int(x1)
    col = s.alpha[:, a:b].max(axis=1) if b > a else s.alpha.max(axis=1)
    rows = np.nonzero(col > thr)[0]
    return float(rows[0]) if len(rows) else f.crown


def _circle_mask(d):
    ss = 4; m = Image.new('L', (d * ss, d * ss), 0); ImageDraw.Draw(m).ellipse((0, 0, d * ss - 1, d * ss - 1), fill=255)
    return m.resize((d, d), Image.LANCZOS)


# =====================================================================================================
# 1. red-collage: the reel opener
# =====================================================================================================
_RED = (232, 36, 47)


def _bw_punch(rgb):
    y = kit.luma(rgb)
    y = np.clip((y - 0.47) * 1.55 + 0.5, 0, 1)
    y = y * y * (3 - 2 * y) * 0.35 + y * 0.65                # S-curve: deep blacks, bright highlights
    return np.repeat(y[..., None], 3, 2)


@effect('red-collage', 'Words Behind You', 'black-and-white punch, a giant red word behind you and three lines slamming in',
        aliases=('collage', 'opener'))
def red_collage(s, p):
    u = s.u
    word = p.text('KEEM').upper()
    lines = p.list('lines', 'ONE CLIP.|48 EFFECTS.|ZERO APPS.')
    hl = int(p.num('hl', 1))
    tag = str(p.get('tag', 'REC · KEEMFINITY'))
    red = p.color('color', _RED)
    a_all = s.env(0.08, 0.22)

    # slam impacts: tiny camera kick each time something lands
    t_word = 0.0
    t_lines = [0.32 + i * 0.3 for i in range(len(lines))]
    kick = 0.0
    for t0 in [t_word + 0.12] + [t + 0.15 for t in t_lines]:
        if s.t >= t0: kick = max(kick, math.exp(-(s.t - t0) * 14))
    s.camera(1.0 + 0.018 * s.p + 0.025 * kick)

    # the black and white punch with grain + vignette
    s.tint(_bw_punch, alpha=a_all)
    s.tint(lambda r: kit.vignette(r, 0.45), alpha=a_all)

    # giant red word, stretched to fill the top quarter, glowing, behind the head
    def make_word():
        f = _fit(word, 'display', s.W * 0.95, s.W * 0.75)
        im = kit.text_layer(word, f, fill=red + (255,))
        target_h = s.H * 0.25
        if im.height < target_h: im = im.resize((im.width, int(target_h)), Image.BICUBIC)
        return _with_glow(im, 26 * u, 1.1, red), im.size
    big, (bw, bh) = s.once('word', make_word)
    k = kit.expo_out(s.t / 0.4)
    sc = kit.lerp(1.25, 1.0, k) * (1 + 0.035 * s.p)
    layer = kit.scaled(big, sc)
    s.behind(layer, s.W / 2, s.H * 0.02 + bh * 0.5, 'c', alpha=a_all * kit.clamp(s.t / 0.1))

    # three lines slamming in, lower left
    def make_lines():
        size = min(_fit(l, 'heavy', s.W * 0.8, s.W * 0.14).size for l in lines)
        f = _font('heavy', size)
        out = []
        for i, l in enumerate(lines):
            col = red if i == hl else (255, 255, 255)
            im = kit.text_layer(l, f, fill=col + (255,))
            sh = _text_shadow(im, u, blur=14, alpha=0.7, dy=8)
            if i == hl: sh = _with_glow(sh, 16 * u, 0.6, red)
            out.append((sh, im.size))
        return out, size
    L, size = s.once('lines', make_lines)
    x0 = s.W * 0.06
    gap = size * 1.12
    y_last = s.H * 0.855
    for i, (im, (w, h)) in enumerate(L):
        kk = kit.window(s.t, t_lines[i], 0.15)
        if kk <= 0: continue
        lay = _slam(im, kk, big=1.5, blur=16 * u)
        cy = y_last - (len(L) - 1 - i) * gap
        s.over(lay, x0 + w / 2, cy, 'c', alpha=a_all)

    # tiny red mono tag with a blinking dot
    if s.t > 0.25:
        def make_tag():
            f = _font('mono', s.W * 0.03)
            im = kit.text_layer(tag, f, fill=red + (255,))
            d = int(f.size * 0.62); dot = Image.new('RGBA', (d, d), (0, 0, 0, 0))
            ImageDraw.Draw(dot).ellipse((0, 0, d - 1, d - 1), fill=red + (255,))
            return im, dot
        tim, dot = s.once('tag', make_tag)
        ta = a_all * kit.cubic_out((s.t - 0.25) / 0.3)
        ty = s.H * 0.94
        blink = 1.0 if (s.t % 0.9) < 0.6 else 0.15
        s.over(dot, x0 + dot.width / 2, ty, 'c', alpha=ta * blink)
        s.over(tim, x0 + dot.width * 1.9, ty, 'cl', alpha=ta)

    # film grain on top keeps it alive
    rng = np.random.default_rng(1000 + s.i)
    g = rng.normal(0, 0.03, (s.H // 2, s.W // 2)).astype(np.float32)
    g = cv2.resize(g, (s.W, s.H), interpolation=cv2.INTER_LINEAR)
    s.rgb[:] = np.clip(s.rgb + g[..., None] * a_all, 0, 1)


# =====================================================================================================
# 2. pastel-word: lavender room, ice word, profile card with a follow click
# =====================================================================================================
def _pastel_room(W, H, base, face):
    b = np.float32(base) / 255
    top = np.clip(b * 1.08 + 0.03, 0, 1); wall = b; floor_t = b * 0.93; floor_b = b * 0.84
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    hz = 0.7
    wallc = top[None, None] * (1 - y[..., None] / hz) + wall[None, None] * (y[..., None] / hz)
    fl = (y[..., None] - hz) / (1 - hz)
    floorc = floor_t[None, None] * (1 - fl) + floor_b[None, None] * fl
    m = np.clip((y - hz) / 0.012 + 0.5, 0, 1)[..., None]
    room = wallc * (1 - m) + floorc * m
    room = np.broadcast_to(room, (H, W, 3)).copy()
    # soft cove line + a big window of light behind the head
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - face.cx) / (W * 0.7)) ** 2 + ((yy - face.cy) / (H * 0.42)) ** 2)
    room += (np.clip(1 - d, 0, 1) ** 1.8 * 0.22)[..., None]
    room -= (np.exp(-((yy - hz * H) / (H * 0.006)) ** 2) * 0.035)[..., None]
    return np.clip(kit.vignette(room, 0.18), 0, 1)


@effect('pastel-word', 'Words Behind You', 'soft lavender room, a giant ice word behind you and a profile card you follow',
        aliases=('pastel', 'follow-card'))
def pastel_word(s, p):
    u = s.u
    word = p.text('KEEM').upper()
    name = str(p.get('name', 'Keem')); handle = str(p.get('handle', '@keemfinity'))
    base = p.color('board', (201, 194, 245))
    a = s.env(0.25, 0.25)

    room = s.once('room', lambda: _pastel_room(s.W, s.H, base, s.face))
    s.backdrop(room, alpha=a)
    # the person casts a soft shadow onto the wall
    sh = cv2.GaussianBlur(s.alpha, (0, 0), 18 * u)
    M = np.float32([[1, 0, 26 * u], [0, 1, 14 * u]])
    sh = cv2.warpAffine(sh, M, (s.W, s.H))
    k = (sh * (1 - s.alpha) * 0.28 * a)[..., None]
    s.rgb[:] = s.rgb * (1 - k) + np.float32([0.42, 0.36, 0.72]) * k
    # lift the person's colour into the room
    s.tint(lambda r: np.clip(r * 0.94 + np.float32(base) / 255 * 0.08, 0, 1), alpha=a * 0.8, where='person')

    # giant ice word behind the head
    def make_word():
        f = _fit(word, 'display', s.W * 0.94, s.W * 0.48)
        im = _fill_gradient(kit.text_layer(word, f), (255, 255, 255), (214, 226, 255))
        sh, _ = kit.shadow(im, blur=int(30 * u), alpha=0.28, offset=(0, int(18 * u)), color=(90, 70, 180))
        return sh
    big = s.once('word', make_word)
    kw = kit.expo_out(s.t / 0.5)
    y = s.face.crown - big.height * 0.12 - (1 - kw) * s.H * 0.06
    s.behind(kit.scaled(big, kit.lerp(0.92, 1.0, kw) * (1 + 0.03 * s.p)), s.W / 2, y, 'tc', alpha=a * kw)

    # profile card
    cw, ch = int(s.W * 0.86), int(s.W * 0.22)
    av = int(ch * 0.62); padx = int(ch * 0.19)
    bw, bh = int(s.W * 0.26), int(ch * 0.4)
    t_click = kit.clamp(s.dur * 0.5, 1.1, 2.1)

    def make_card():
        c = kit.rounded(cw, ch, ch * 0.26, fill=(255, 255, 255, 255))
        d = ImageDraw.Draw(c)
        tx = padx + av + int(ch * 0.15)
        fn = _font('sans', ch * 0.21, 800); fh = _font('sans', ch * 0.15, 500)
        d.text((tx, ch * 0.47), name, font=fn, fill=(22, 20, 38, 255), anchor='ls')
        nw = fn.getlength(name)
        # verified badge
        r = int(ch * 0.075); bx = int(tx + nw + ch * 0.06 + r); by = int(ch * 0.47 - ch * 0.075)
        d.ellipse((bx - r, by - r, bx + r, by + r), fill=(108, 92, 231, 255))
        ck = kit.check_mark(int(r * 1.6), (255, 255, 255), width=max(2, int(r * 0.3)))
        c.alpha_composite(ck, (bx - ck.width // 2, by - ck.height // 2))
        d.text((tx, ch * 0.72), handle, font=fh, fill=(126, 120, 150, 255), anchor='ls')
        # avatar ring
        ring = int(av + ch * 0.07)
        g = kit.gradient(ring, ring, (160, 120, 255), (255, 140, 190)); g.putalpha(_circle_mask(ring))
        c.alpha_composite(g, (padx - (ring - av) // 2, (ch - ring) // 2))
        wd = int(av + ch * 0.03); wh = Image.new('RGBA', (wd, wd), (255, 255, 255, 255)); wh.putalpha(_circle_mask(wd))
        c.alpha_composite(wh, (padx - (wd - av) // 2, (ch - wd) // 2))
        return c
    card = s.once('card', make_card)

    def make_buttons():
        f = _font('sans', bh * 0.42, 700)
        b1 = kit.rounded(bw, bh, bh / 2, fill=(108, 92, 231, 255))
        ImageDraw.Draw(b1).text((bw / 2, bh / 2), 'Follow', font=f, fill=(255, 255, 255, 255), anchor='mm')
        b2 = kit.rounded(bw, bh, bh / 2, fill=(238, 235, 248, 255))
        ImageDraw.Draw(b2).text((bw / 2, bh / 2), 'Following', font=f, fill=(40, 36, 64, 255), anchor='mm')
        return b1, b2
    b1, b2 = s.once('buttons', make_buttons)

    kc = kit.expo_out(kit.window(s.t, 0.3, 0.6))
    if kc > 0:
        frame = card.copy()
        # live avatar: a crop of the face from the frame
        f = s.face; side = f.h * 1.9; x0 = int(f.cx - side / 2); y0 = int(f.cy - side * 0.48)
        crop = s.image().crop((x0, y0, x0 + int(side), y0 + int(side))).resize((av, av), Image.LANCZOS).convert('RGBA')
        crop.putalpha(_circle_mask(av)); frame.alpha_composite(crop, (padx, (ch - av) // 2))
        # follow button: press + swap
        press = 1 - 0.08 * math.exp(-((s.t - t_click) / 0.06) ** 2)
        swap = kit.window(s.t, t_click + 0.04, 0.12)
        btn = b1 if swap <= 0 else (b2 if swap >= 1 else Image.blend(b1, b2, swap))
        btn = kit.scaled(btn, press)
        bx = cw - padx - bw // 2; by = ch // 2
        frame.alpha_composite(btn, (int(bx - btn.width / 2), int(by - btn.height / 2)))
        shd = kit.shadow(frame, blur=int(30 * u), alpha=0.3, offset=(0, int(16 * u)), color=(60, 40, 140))[0]
        cy = s.H * 0.86 + (1 - kc) * s.H * 0.2
        s.over(shd, s.W / 2, cy, 'c', alpha=a * kit.clamp(kc * 1.5))

        # cursor glides in, clicks, slips away
        card_x = s.W / 2 - cw / 2; card_y = cy - ch / 2
        tx, ty = card_x + bx + bw * 0.12, card_y + by + bh * 0.1
        km = kit.smooth(kit.window(s.t, t_click - 0.6, 0.5))
        sx, sy = s.W * 0.98, s.H * 1.02
        cx_, cy_ = kit.lerp(sx, tx, km), kit.lerp(sy, ty, km)
        after = kit.window(s.t, t_click + 0.25, 0.4)
        cx_ += after * s.W * 0.05; cy_ += after * s.H * 0.03
        ca = kit.clamp((s.t - (t_click - 0.6)) / 0.15) * (1 - after)
        if ca > 0:
            cur = s.once('cursor', lambda: kit.cursor(int(72 * u), (255, 255, 255), (30, 26, 50)))
            cur = kit.scaled(cur, press)
            # click ripple
            rp = kit.window(s.t, t_click, 0.35)
            if 0 < rp < 1:
                R = int((10 + 50 * kit.cubic_out(rp)) * u) + 2
                ring = Image.new('RGBA', (2 * R + 4, 2 * R + 4), (0, 0, 0, 0))
                ImageDraw.Draw(ring).ellipse((2, 2, 2 * R + 2, 2 * R + 2), outline=(255, 255, 255, int(220 * (1 - rp))), width=max(2, int(4 * u)))
                s.over(ring, tx, ty, 'c', alpha=a)
            s.over(cur, cx_, cy_, 'tl', alpha=a * ca)



# =====================================================================================================
# 3. cover-star: magazine cover
# =====================================================================================================
@effect('cover-star', 'Words Behind You', 'you become a magazine cover: masthead behind your head, cover lines, barcode',
        aliases=('cover', 'magazine'))
def cover_star(s, p):
    u = s.u
    word = p.text('KEEM').upper()
    lines = p.list('lines', '48 effects|you can steal|no editor|needed')
    issue = str(p.get('issue', 'THE AI ISSUE · 2026'))
    red = p.color('color', (226, 30, 44))
    a = s.env(0.15, 0.25)

    s.camera(1.0 + 0.04 * kit.smooth(s.p))                       # slow glossy push
    s.tint(lambda r: kit.vignette(kit.tone(r, 'pop'), 0.3), alpha=a * 0.85)
    flash = math.exp(-s.t / 0.06) * 0.6 if s.t < 0.3 else 0      # camera-flash snap on entry
    if flash > 0.01: s.rgb[:] = s.rgb * (1 - flash) + flash

    # masthead
    def make_mast():
        f = _fit(word, 'serif', s.W * 0.93, s.W * 0.6, weight=900)
        im = kit.text_layer(word, f, fill=red + (255,))
        return kit.shadow(im, blur=int(18 * u), alpha=0.35, offset=(0, int(8 * u)))[0], im.size
    mast, (mw, mh) = s.once('mast', make_mast)
    km = kit.expo_out(s.t / 0.55)
    top = s.H * 0.06
    s.behind(kit.scaled(mast, kit.lerp(1.12, 1.0, km)), s.W / 2, top + mh / 2, 'c', alpha=a * kit.clamp(km * 1.6))

    # issue line above the masthead
    def make_issue():
        f = _font('sans', s.W * 0.028, 700)
        return _text_shadow(kit.text_layer(issue, f, tracking=s.W * 0.006), u, blur=6, alpha=0.5, dy=2)
    iss = s.once('issue', make_issue)
    ki = kit.cubic_out(kit.window(s.t, 0.2, 0.4))
    s.over(iss, s.W / 2, s.H * 0.038 + (1 - ki) * 10 * u, 'c', alpha=a * ki)

    # cover lines: serif italic / heavy sans mix, left column
    def make_lines():
        out = []
        for i, l in enumerate(lines):
            if i % 2 == 0:
                f = _fit(l, 'serif-italic', s.W * 0.44, s.W * 0.1)
                im = kit.text_layer(l, f, fill=(255, 255, 255, 255))
            else:
                f = _fit(l.upper(), 'heavy', s.W * 0.42, s.W * 0.05)
                im = kit.text_layer(l.upper(), f, fill=(255, 255, 255, 255), tracking=s.W * 0.002)
            out.append(_text_shadow(im, u, blur=12, alpha=0.7, dy=5))
        return out
    L = s.once('lines', make_lines)
    x0 = s.W * 0.055; y = s.H * 0.37
    # soft dark scrim behind the cover lines so they read over the jacket
    scrim = s.once('scrim', lambda: kit.radial(s.W, s.H, s.W * 0.18, s.H * 0.47, s.W * 0.62, (0, 0, 0), 0.6, 1.4))
    s.over(scrim, 0, 0, 'tl', alpha=a * kit.cubic_out(kit.window(s.t, 0.25, 0.5)))
    kb = kit.expo_out(kit.window(s.t, 0.3, 0.4))
    if kb > 0:
        bar = kit.rounded(int(s.W * 0.11 * kb) + 1, int(7 * u), 3 * u, fill=red + (255,))
        s.over(bar, x0 + 30 * u, y + 24 * u, 'bl', alpha=a)
    for i, im in enumerate(L):
        k = kit.expo_out(kit.stagger(s.t, i, 0.35, 0.12, 0.5))
        s.over(im, x0 + (1 - k) * -s.W * 0.1, y, 'tl', alpha=a * kit.clamp(k * 1.5))
        y += im.height * (0.8 if i % 2 == 0 else 0.62) + (14 * u if i % 2 == 1 else 0)

    # barcode block, bottom right
    def make_barcode():
        bw, bh = int(s.W * 0.2), int(s.W * 0.13)
        im = Image.new('RGBA', (bw, bh), (255, 255, 255, 255)); d = ImageDraw.Draw(im)
        rng = np.random.default_rng(48); x = bw * 0.08; pad = bw * 0.08; unit = max(1.0, bw / 110)
        while x < bw - pad:
            w = rng.choice([1, 1, 2, 3]) * unit
            d.rectangle((x, bh * 0.1, x + w - 1, bh * 0.72), fill=(10, 10, 10, 255)); x += w + rng.choice([1, 2, 2, 3]) * unit
        f = _font('mono', bh * 0.14)
        d.text((bw / 2, bh * 0.86), '0 48226 02026 4', font=f, fill=(10, 10, 10, 255), anchor='mm')
        return kit.shadow(im, blur=int(10 * u), alpha=0.4, offset=(0, int(4 * u)))[0]
    bc = s.once('barcode', make_barcode)
    kk = kit.expo_out(kit.window(s.t, 0.75, 0.45))
    s.over(bc, s.W * 0.96, s.H * 0.965 + (1 - kk) * s.H * 0.05, 'br', alpha=a * kk)


# =====================================================================================================
# 4. headline-box: punch in + two-line headline above the head
# =====================================================================================================
@effect('headline-box', 'Words Behind You', 'punch in on your face with a two-line headline, the second line boxed',
        aliases=('headline',))
def headline_box(s, p):
    u = s.u
    lines = p.list('text', 'How I edit with|Claude in 1 line')
    l1 = lines[0] if lines else ''; l2 = lines[1] if len(lines) > 1 else ''
    zoom = p.num('zoom', 1.25)
    a = s.env(0.15, 0.25)
    kin = kit.expo_out(s.t / 0.4); kout = 1 - kit.cubic_in(kit.window(s.t, s.dur - 0.3, 0.3))
    z = 1 + (zoom - 1) * kin * kout
    s.camera(z, cx=s.face.cx, cy=0)                      # zoom from the top edge: pushes the head down, no borders
    s.tint(lambda r: kit.vignette(r, 0.35), alpha=a)

    def make():
        # head position at full punch (layout is done once, then follows the face)
        zz = zoom / max(1e-3, z)
        ht = _head_top(s) * zz
        crown = s.face.crown * zz
        room = ht - s.H * 0.016
        size = s.W * 0.095
        while True:
            f1 = _fit(l1 or 'x', 'sans', s.W * 0.92, size, weight=800)
            f2 = _fit(l2 or 'x', 'sans', s.W * 0.84, size, weight=800)
            h1 = kit.text_layer(l1, f1, fill=(255, 255, 255, 255)) if l1 else None
            t2 = kit.text_layer(l2, f2, fill=(12, 12, 12, 255)) if l2 else None
            box = None
            if t2:
                px, py = int(f2.size * 0.32), int(f2.size * 0.2)
                box = kit.rounded(t2.width + 2 * px, t2.height + 2 * py, f2.size * 0.12, fill=(255, 255, 255, 255))
                box.alpha_composite(t2, (px, py))
            gap = f1.size * 0.22
            total = (h1.height if h1 else 0) + gap + (box.height if box else 0)
            if total <= room - s.H * 0.012 or size < s.W * 0.045: break
            size *= 0.93
        h1s = _text_shadow(h1, u, blur=12, alpha=0.75, dy=6) if h1 else None
        boxs = kit.shadow(box, blur=int(18 * u), alpha=0.45, offset=(0, int(8 * u)))[0] if box else None
        return h1s, (h1.size if h1 else (0, 0)), boxs, (box.size if box else (0, 0)), gap, ht - crown
    h1s, h1sz, boxs, bxsz, gap, off = s.once('layout', make)

    head = s.face.crown + off                              # follows the head
    bottom = head - s.H * 0.012
    yb = bottom - bxsz[1] / 2
    y1 = bottom - bxsz[1] - gap - h1sz[1] / 2
    y1 = max(y1, s.H * 0.014 + h1sz[1] / 2); yb = max(yb, y1 + h1sz[1] / 2 + gap + bxsz[1] / 2)
    for i, (im, y) in enumerate(((h1s, y1), (boxs, yb))):
        if im is None: continue
        k = kit.window(s.t, 0.12 + i * 0.14, 0.35)
        if k <= 0: continue
        sc = kit.back_out(k, 2.2) if k < 1 else 1.0
        lay = kit.scaled(im, max(0.05, kit.lerp(0.4, 1.0, sc)) * (1 + 0.02 * s.p))
        s.behind(lay, s.W / 2, y, 'c', alpha=a * kit.clamp(k * 3))


# =====================================================================================================
# 5. split-headline: white headline card pushes the video down
# =====================================================================================================
def _norm(w):
    return ''.join(c for c in w.lower() if c.isalnum())


@effect('split-headline', 'Words Behind You', 'a white headline card slides down and pushes you down the frame',
        aliases=('split',))
def split_headline(s, p):
    u = s.u
    lines = p.list('text', 'Claude|edits my|reels.')
    hl = _norm(p.get('hl', 'edits'))
    pill_txt = str(p.get('pill', 'in seconds ↗'))
    green = p.color('color', (46, 220, 120))
    CH = int(s.H * 0.38)
    kin = kit.expo_out(s.t / 0.5); kout = 1 - kit.cubic_in(kit.window(s.t, s.dur - 0.35, 0.35))
    off = CH * kin * kout
    s.camera(1.0, dy=off)

    def layout():
        padx = s.W * 0.075
        fp = _font('sans', s.W * 0.042, 650)
        pw = int(fp.getlength(pill_txt) + s.W * 0.07); ph = int(s.W * 0.085)
        size = min(_fit(l, 'heavy', s.W - 2 * padx, s.W * 0.16).size for l in lines)
        size = int(min(size, (CH * 0.8 - ph - s.W * 0.04) / (len(lines) * 1.06)))
        f = _font('heavy', size); lh = int(size * 1.06)
        capb = f.getbbox('H', anchor='ls')
        block = lh * len(lines) + ph + s.W * 0.04
        top = (CH - block) / 2 + s.H * 0.01
        out = []
        for i, l in enumerate(lines):
            im = Image.new('RGBA', (s.W, lh), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
            base = int(lh * 0.5 - capb[1] * 0.5)
            d.text((padx, base), l, font=f, fill=(14, 14, 18, 255), anchor='ls')
            hlbox = None; x = padx
            for w in l.split(' '):
                if _norm(w) == hl:
                    bb = f.getbbox(w, anchor='ls')
                    hlbox = (x - size * 0.14, base + capb[1] - size * 0.14, x + bb[2] + size * 0.14, base + size * 0.14)
                x += f.getlength(w + ' ')
            out.append((im, top + i * lh, hlbox))
        pill = kit.rounded(pw, ph, ph / 2, fill=(14, 14, 18, 255))
        ImageDraw.Draw(pill).text((pw / 2, ph / 2), pill_txt, font=fp, fill=(255, 255, 255, 255), anchor='mm')
        return out, pill, (padx, top + lh * len(lines) + s.W * 0.03), lh
    L, pill, ppos, lh = s.once('layout', layout)

    if off < 1: return
    R = int(60 * u)
    card = kit.rounded(s.W, CH + R, R, fill=(250, 250, 247, 255)).crop((0, R, s.W, CH + R))
    d = ImageDraw.Draw(card)
    for i, (im, y, hb) in enumerate(L):
        if hb:
            kh = kit.expo_out(kit.window(s.t, 0.55, 0.4))
            if kh > 0:
                x0, y0, x1, y1 = hb; x1 = x0 + (x1 - x0) * kh
                d.rounded_rectangle((x0, y + y0, x1, y + y1), radius=(y1 - y0) / 2, fill=green + (255,))
        kl = kit.expo_out(kit.stagger(s.t, i, 0.18, 0.09, 0.5))
        if kl > 0:
            slot = Image.new('RGBA', (s.W, lh), (0, 0, 0, 0)); slot.paste(im, (0, int(round((1 - kl) * lh))))
            card.alpha_composite(slot, (0, int(y)))
    kp = kit.window(s.t, 0.75, 0.35)
    if kp > 0:
        pl = kit.scaled(pill, max(0.05, kit.back_out(kp, 2.0)))
        card.alpha_composite(kit.faded(pl, kit.clamp(kp * 3)),
                             (int(ppos[0] + (pill.width - pl.width) / 2), int(ppos[1] + (pill.height - pl.height) / 2)))
    im, (dx, dy) = kit.shadow(card, blur=int(26 * u), alpha=0.35, offset=(0, int(10 * u)))
    s.over(im, dx, off - CH + dy, 'tl')


# =====================================================================================================
# 6. dot-reveal: the shot builds itself out of glowing halftone dots
# =====================================================================================================
@effect('dot-reveal', 'Camera & Light', 'your shot builds itself out of glowing dots, then snaps into the picture',
        aliases=('dots', 'halftone'))
def dot_reveal(s, p):
    c = max(4, int(round(p.num('cell', 18) * s.u)))
    tint = np.float32(p.color('color', (255, 196, 140))) / 255
    tint = tint / max(1e-3, float(tint.max()))
    W, H = s.W, s.H
    gw, gh = -(-W // c), -(-H // c)

    def grid():
        yy, xx = np.mgrid[0:gh * c, 0:gw * c].astype(np.float32)
        d = np.sqrt((xx % c - (c - 1) / 2) ** 2 + (yy % c - (c - 1) / 2) ** 2)[:H, :W]
        r = np.random.default_rng(7).random((gh, gw)).astype(np.float32)
        return d, r
    dist, order = s.once('grid', grid)

    build = max(0.4, s.dur * 0.7)
    q = s.t / build
    src = s.rgb.copy()
    pad = np.pad(src, ((0, gh * c - H), (0, gw * c - W), (0, 0)), mode='edge')
    small = cv2.resize(pad, (gw, gh), interpolation=cv2.INTER_AREA)
    lum = kit.luma(small)
    col = np.clip(small * tint * 1.6 + 0.09 * tint, 0, 1)
    on = np.clip((min(q, 1.0) * 1.12 - order) / 0.12, 0, 1)
    on = on * on * (3 - 2 * on)
    R = c * 0.5 * (0.42 + 0.56 * np.sqrt(lum)) * (0.4 + 0.6 * on) * (on > 0)
    Rm = np.repeat(np.repeat(R, c, 0), c, 1)[:H, :W]
    Cm = np.repeat(np.repeat(col * on[..., None], c, 0), c, 1)[:H, :W]
    mask = np.clip(Rm - dist + 0.5, 0, 1)
    dots = Cm * mask[..., None]
    sm = cv2.resize(dots, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    glow = cv2.resize(cv2.GaussianBlur(sm, (0, 0), c * 0.9), (W, H))
    bg = np.float32([0.05, 0.03, 0.02])
    field = np.clip(bg + dots * 1.05 + glow * 0.9, 0, 1)
    zk = 1.0 + 0.04 * (1 - min(q, 1))                    # the dot field drifts in
    if zk > 1.001:
        M = np.float32([[zk, 0, (1 - zk) * W / 2], [0, zk, (1 - zk) * H / 2]])
        field = cv2.warpAffine(field, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    snap = kit.window(s.t, build, 0.1)                     # snap into the real picture
    mix = kit.cubic_out(s.t / 0.15) * (1 - snap)
    out = src * (1 - mix) + field * mix
    if s.t >= build:
        fl = math.exp(-(s.t - build) / 0.09) * 0.45
        out = out + np.float32([1.0, 0.85, 0.65]) * fl
        s.rgb[:] = np.clip(out, 0, 1)
        s.camera(1 + 0.05 * math.exp(-(s.t - build) / 0.18))
        return
    s.rgb[:] = np.clip(out, 0, 1)


# =====================================================================================================
# 7. rabbit-hole: the frame tunnels into itself
# =====================================================================================================
@effect('rabbit-hole', 'Camera & Light', 'the frame tunnels into itself behind you, nested copies spinning inward',
        aliases=('tunnel', 'droste'))
def rabbit_hole(s, p):
    u = s.u; W, H = s.W, s.H
    ratio = kit.clamp(p.num('ratio', 0.74), 0.4, 0.92)
    speed = p.num('speed', 0.6); spin = p.num('spin', 1.6)
    kin = kit.expo_out(s.t / 0.7); kout = 1 - kit.cubic_in(kit.window(s.t, s.dur - 0.4, 0.4))
    op = kin * kout
    if op < 0.002: return
    r_eff = kit.lerp(0.985, ratio, op)
    src = s.rgb.copy(); alpha = s.alpha
    cx, cy = s.face.cx, min(H * 0.62, s.face.cy + H * 0.24)

    def masks():
        m = Image.new('L', (W, H), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, W - 1, H - 1), int(W * 0.06), fill=255)
        return np.asarray(m, np.float32) / 255
    rr = s.once('mask', masks)
    ker = np.ones((max(2, int(3 * u)),) * 2, np.uint8)

    phi = (s.t * speed) % 1.0
    n = int(math.ceil(math.log(0.1) / math.log(ratio))) + 1
    comp = src.copy()
    for kk in range(1, n + 1):
        e = kk - phi
        sc = r_eff ** e
        if sc < 0.06: break
        fa = kit.smooth(e / 0.6) * op                  # fades as it grows to fill the frame
        if fa < 0.01: continue
        M = cv2.getRotationMatrix2D((cx, cy), spin * e * op, sc)
        A = cv2.warpAffine(rr, M, (W, H), flags=cv2.INTER_LINEAR)
        img = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR) * (0.82 ** e)
        hs = cv2.resize(A, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        shd = cv2.resize(cv2.GaussianBlur(hs, (0, 0), 6 * u + 1), (W, H))
        comp *= (1 - 0.7 * fa * shd * (1 - A))[..., None]
        edge = np.clip(A - cv2.erode(A, ker), 0, 1)
        Af = (A * fa)[..., None]
        comp = comp * (1 - Af) + img * Af
        ea = (edge * fa * 0.95)[..., None]
        comp = comp * (1 - ea) + ea
    s.rgb[:] = comp * (1 - alpha[..., None]) + src * alpha[..., None]
