"""Keemfinity engine: reads a talking-head video, tracks the face, cuts the person out, and runs effects frame by frame.

An effect is a function `fn(s: Shot, p: Params)` registered with `@effect(...)`. It edits `s.rgb` in place
through the Shot helpers (over, behind, backdrop, camera, ...). See README for the plan format.
"""
import glob, importlib, json, math, os, pkgutil, subprocess
import numpy as np
import cv2
from PIL import Image

from . import kit

ROOT = kit.ROOT
MODELS = os.path.join(ROOT, 'models')
EFFECTS, ALIASES = {}, {}


def effect(name, group, desc, aliases=()):
    def reg(fn):
        EFFECTS[name] = dict(fn=fn, group=group, desc=desc, name=name)
        for a in aliases: ALIASES[a] = name
        return fn
    return reg


def load_effects():
    pkg = os.path.dirname(os.path.abspath(__file__))
    for m in sorted(pkgutil.iter_modules([pkg])):
        if m.name.startswith('fx_'): importlib.import_module(f'{__package__}.{m.name}')


def resolve(name):
    n = str(name).lower().lstrip('/').strip()
    return ALIASES.get(n, n)


# ---------- options ----------
class Params:
    """Options for one effect instance, with friendly getters."""
    def __init__(self, opts, start, end, ctx):
        self.o, self.start, self.end, self.ctx = opts, start, end, ctx

    def get(self, key, default=None):
        v = self.o.get(key); return default if v is None or v == '' else v

    def text(self, default=''):
        return str(self.get('text', default))

    def list(self, key, default='', sep='|'):
        v = self.get(key, default)
        return [x.strip() for x in (v if isinstance(v, list) else str(v).split(sep)) if str(x).strip() != '']

    def pairs(self, key, default=''):
        """'a::b|c::d' -> [('a', 'b'), ('c', 'd')]"""
        return [tuple((x.split('::', 1) + [''])[:2]) for x in self.list(key, default)]

    def num(self, key, default=0.0):
        try: return float(self.get(key, default))
        except (TypeError, ValueError): return float(default)

    def color(self, key='color', default=(255, 255, 255)):
        return kit.hex_rgb(self.get(key), default)

    def time(self, key, default):
        """A moment given in video seconds (option) -> seconds since the effect started; `default` is already relative."""
        v = self.get(key)
        return float(v) - self.start if v is not None else default

    def images(self, count=12):
        """The user's reels (option `images`: folder or comma list of files), else stylised stills from this video."""
        src = self.get('images')
        key = ('images', src, count)
        if key in self.ctx.cache: return self.ctx.cache[key]
        ims = []
        if src:
            paths = []
            for part in str(src).split(','):
                part = part.strip()
                cand = [part, os.path.join(self.ctx.base, part)]
                for c in cand:
                    if os.path.isdir(c):
                        paths += sorted(f for f in glob.glob(os.path.join(c, '*')) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.mp4', '.mov')))
                        break
                    if os.path.isfile(c): paths.append(c); break
            for f in paths:
                if f.lower().endswith(('.mp4', '.mov')):
                    cap = cv2.VideoCapture(f); cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.3)); ok, fr = cap.read()
                    if ok: ims.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
                else:
                    try: ims.append(Image.open(f).convert('RGB'))
                    except OSError: pass
        if not ims:
            looks = ['pop', 'warm', 'bw', 'cool', 'film', 'moody', 'red', None]
            ims = [kit.to_pil(kit.tone(kit.to_np(t), looks[i % len(looks)])) for i, t in enumerate(self.ctx.thumbs)]
        ims = (ims * (count // max(1, len(ims)) + 1))[:count]
        self.ctx.cache[key] = ims
        return ims


# ---------- the per-frame canvas ----------
class Face:
    def __init__(self, cx, cy, w, h):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        self.x, self.y = cx - w / 2, cy - h / 2
        self.top, self.bottom = cy - h / 2, cy + h / 2
        self.chin = cy + h * 0.62          # just under the chin
        self.crown = cy - h * 0.85         # top of the head (hair / cap)


class Shot:
    """One output frame inside one effect. Times: T (video seconds), t (since effect start), dur, p (= t / dur)."""
    def __init__(self, ctx, i, rgb, alpha, face, start, end, state):
        self.ctx, self.i = ctx, i
        self.W, self.H, self.fps = ctx.W, ctx.H, ctx.fps
        self.u = ctx.W / 1080
        self.T = i / ctx.fps; self.t = self.T - start; self.dur = max(1e-6, end - start); self.p = kit.clamp(self.t / self.dur)
        self.rgb, self.alpha, self.face, self.state = rgb, alpha, face, state

    # timing
    def env(self, fade_in=0.25, fade_out=0.25):
        """0 -> 1 -> 0 over the effect: eases in for fade_in seconds, out over the last fade_out seconds."""
        a = kit.cubic_out(self.t / fade_in) if fade_in > 0 else 1.0
        b = kit.cubic_out((self.dur - self.t) / fade_out) if fade_out > 0 else 1.0
        return min(a, b)

    def once(self, key, make):
        """Cache anything expensive (layouts, rendered cards) for this effect instance."""
        if key not in self.state: self.state[key] = make()
        return self.state[key]

    # pixels
    def image(self):
        return kit.to_pil(self.rgb)

    def person(self):
        """The person cut out of the current frame as RGBA."""
        a = (np.clip(self.alpha, 0, 1) * 255).astype(np.uint8)
        im = self.image().convert('RGBA'); im.putalpha(Image.fromarray(a)); return im

    def _place(self, im, x, y, anchor):
        w, h = im.size
        ax = {'l': 0, 'c': 0.5, 'r': 1}[anchor[1] if len(anchor) > 1 else 'c']
        ay = {'t': 0, 'c': 0.5, 'b': 1}[anchor[0]]
        return int(round(x - w * ax)), int(round(y - h * ay))

    def _layer(self, im, x, y, anchor, alpha):
        """RGBA image -> (premultiplied rgb, alpha) arrays covering the frame, or None if off-screen."""
        if im is None or alpha <= 0.002: return None
        x, y = self._place(im, x, y, anchor)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(self.W, x + im.width), min(self.H, y + im.height)
        if x1 <= x0 or y1 <= y0: return None
        a = np.asarray(im, np.float32)[y0 - y:y1 - y, x0 - x:x1 - x] / 255
        return (x0, y0, x1, y1), a[..., :3], a[..., 3:4] * alpha

    def over(self, im, x, y, anchor='tl', alpha=1.0):
        """Draw an RGBA image on top of everything. anchor: 'tl', 'c' (centre), 'tc', 'bc', 'cl', 'br', ..."""
        L = self._layer(im, x, y, 'cc' if anchor == 'c' else anchor, alpha)
        if L is None: return
        (x0, y0, x1, y1), rgb, a = L
        r = self.rgb[y0:y1, x0:x1]; r[:] = r * (1 - a) + rgb * a

    def behind(self, im, x, y, anchor='tl', alpha=1.0):
        """Draw an RGBA image behind the person (it disappears behind their head and body)."""
        L = self._layer(im, x, y, 'cc' if anchor == 'c' else anchor, alpha)
        if L is None: return
        (x0, y0, x1, y1), rgb, a = L
        a = a * (1 - self.alpha[y0:y1, x0:x1, None])
        r = self.rgb[y0:y1, x0:x1]; r[:] = r * (1 - a) + rgb * a

    def backdrop(self, bg, alpha=1.0):
        """Replace everything behind the person. bg: HxWx3 float array, RGB tuple, or PIL image."""
        if isinstance(bg, tuple): bg = np.broadcast_to(np.float32(bg) / 255, self.rgb.shape)
        elif isinstance(bg, Image.Image): bg = kit.to_np(bg.resize((self.W, self.H)))
        a = (1 - self.alpha[..., None]) * alpha
        self.rgb[:] = self.rgb * (1 - a) + bg * a

    def tint(self, fn, alpha=1.0, where='all'):
        """Apply fn(rgb)->rgb to the frame. where: 'all', 'person' or 'background'."""
        new = fn(self.rgb)
        if where == 'person': a = self.alpha[..., None] * alpha
        elif where == 'background': a = (1 - self.alpha[..., None]) * alpha
        else: a = alpha
        self.rgb[:] = self.rgb * (1 - a) + new * a

    def camera(self, zoom=1.0, cx=None, cy=None, dx=0.0, dy=0.0):
        """Zoom about (cx, cy) (default: the face) and pan. Moves the person mask and face with the picture."""
        cx = self.face.cx if cx is None else cx; cy = self.face.cy if cy is None else cy
        if abs(zoom - 1) < 1e-4 and abs(dx) < 0.01 and abs(dy) < 0.01: return
        zoom = max(zoom, 1e-3)
        M = np.float32([[zoom, 0, cx - zoom * cx + dx], [0, zoom, cy - zoom * cy + dy]])
        self.rgb[:] = cv2.warpAffine(self.rgb, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        self.alpha = cv2.warpAffine(self.alpha, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        f = self.face
        self.face = Face(zoom * f.cx + M[0, 2], zoom * f.cy + M[1, 2], f.w * zoom, f.h * zoom)

    def keep_face_clear(self, box, margin=0.15):
        """True if box (x, y, w, h) overlaps the face (plus margin): use it to nudge text away from the face."""
        f = self.face; m = f.w * margin
        x, y, w, h = box
        return not (x + w < f.x - m or x > f.x + f.w + m or y + h < f.top - m or y > f.bottom + m)


# ---------- video io ----------
def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate,nb_frames:format=duration',
                          '-of', 'json', path], capture_output=True, text=True, check=True).stdout
    j = json.loads(out); st = j['streams'][0]; num, den = st['r_frame_rate'].split('/')
    fps = float(num) / float(den); dur = float(j['format']['duration'])
    return int(st['width']), int(st['height']), fps, dur


def has_audio(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', path], capture_output=True, text=True).stdout
    return bool(out.strip())


def frames(path, w, h, gray=False):
    pix = 'gray' if gray else 'rgb24'; n = w * h * (1 if gray else 3)
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', path, '-vf', f'scale={w}:{h}:flags=bicubic', '-f', 'rawvideo', '-pix_fmt', pix, '-'],
                         stdout=subprocess.PIPE)
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n: break
            a = np.frombuffer(b, np.uint8)
            yield a.reshape(h, w) if gray else a.reshape(h, w, 3)
    finally:
        p.stdout.close(); p.wait()


# ---------- analysis ----------
class Context:
    def __init__(self, src, W, H, fps, n, base):
        self.src, self.W, self.H, self.fps, self.n, self.base = src, W, H, fps, n, base
        self.faces, self.thumbs, self.cache = [], [], {}


def track_faces(src, n, W, H, log=print):
    """Face box per frame (output coords), smoothed; gaps filled from neighbours."""
    aw = 360; ah = int(round(aw * H / W / 2)) * 2
    det = cv2.FaceDetectorYN.create(os.path.join(MODELS, 'yunet.onnx'), '', (aw, ah), 0.6)
    raw, thumbs = [], []
    every = max(1, n // 12)
    for i, fr in enumerate(frames(src, aw, ah)):
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        _, found = det.detect(bgr)
        if found is not None and len(found):
            x, y, w, h = max(found, key=lambda f: f[2] * f[3])[:4]
            raw.append((x + w / 2, y + h / 2, w, h))
        else:
            raw.append(None)
        if i % every == 0 and len(thumbs) < 12: thumbs.append(Image.fromarray(fr))
    n = len(raw)
    if not any(raw):
        log('  no face found: effects will assume one near the top third')
        raw = [(aw / 2, ah * 0.33, aw * 0.28, aw * 0.36)] * n
    idx = [i for i, r in enumerate(raw) if r]
    arr = np.zeros((n, 4), np.float32)
    for k in range(4):
        arr[:, k] = np.interp(np.arange(n), idx, [raw[i][k] for i in idx])
    # gentle centred smoothing: kills jitter without lag
    k = np.hanning(9)[1:-1]; k /= k.sum()
    sm = np.stack([np.convolve(np.pad(arr[:, j], (3, 3), mode='edge'), k, mode='valid') for j in range(4)], 1)
    s = W / aw
    return [Face(cx * s, cy * s, w * s, h * s * 1.0) for cx, cy, w, h in sm], thumbs


class Cutout:
    """Person mask per frame: from a matte video when given, else the on-device selfie segmenter."""
    def __init__(self, W, H, matte=None):
        self.W, self.H, self.prev = W, H, None
        self.matte = frames(matte, W, H, gray=True) if matte else None
        if not matte:
            import mediapipe as mp
            from mediapipe.tasks.python import vision, BaseOptions
            self.mp = mp
            self.seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=os.path.join(MODELS, 'selfie_segmenter.tflite')), output_confidence_masks=True))

    def next(self, rgb_u8):
        if self.matte is not None:
            m = next(self.matte, None)
            if m is not None: return m.astype(np.float32) / 255
        sw = 384; sh = int(round(sw * self.H / self.W))
        small = np.ascontiguousarray(cv2.resize(rgb_u8, (sw, sh), interpolation=cv2.INTER_AREA))
        m = self.seg.segment(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=small)).confidence_masks[0].numpy_view()
        m = cv2.resize(np.squeeze(m).astype(np.float32), (self.W, self.H), interpolation=cv2.INTER_CUBIC)
        m = np.clip((m - 0.3) / 0.4, 0, 1); m = m * m * (3 - 2 * m)        # firmer edge
        m = cv2.GaussianBlur(m, (0, 0), self.W / 900)
        if self.prev is not None: m = 0.65 * m + 0.35 * self.prev           # steadier edges between frames
        self.prev = m
        return m


# ---------- render ----------
def run(src, out, plan, width=1080, stills=None, matte=None, log=print):
    """plan: list of dicts {fx, start, end, ...options}. stills: list of seconds -> PNGs instead of a video."""
    load_effects()
    sw, sh, fps, dur = probe(src)
    W = int(width) // 2 * 2; H = int(round(W * sh / sw / 2)) * 2
    n = int(round(dur * fps))
    items = []
    for k, d in enumerate(plan):
        d = dict(d); name = resolve(d.pop('fx'))
        if name not in EFFECTS: raise SystemExit(f"Unknown effect '{name}'. Run --list to see all {len(EFFECTS)}.")
        start = float(d.pop('start', 0)); end = float(d.pop('end', dur))
        items.append(dict(name=name, start=start, end=min(end, dur), opts=d, state={}))
    ctx = Context(src, W, H, fps, n, os.path.dirname(os.path.abspath(out)))
    log('Tracking your face...')
    ctx.faces, ctx.thumbs = track_faces(src, n, W, H, log)
    n = len(ctx.faces)
    for it in items: it['params'] = Params(it['opts'], it['start'], it['end'], ctx)
    cut = Cutout(W, H, matte)
    want = sorted({min(n - 1, int(round(float(t) * fps))) for t in stills}) if stills else None

    writer = None
    if not want:
        cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', f'{fps}', '-i', '-']
        if has_audio(src): cmd += ['-i', src, '-map', '0:v', '-map', '1:a', '-c:a', 'aac', '-b:a', '192k', '-shortest']
        cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', '17' if W >= 2000 else '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out]
        writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    log('Adding the effects...')
    saved = []
    for i, fr in enumerate(frames(src, W, H)):
        if i >= n: break
        T = i / fps
        live = [it for it in items if it['start'] <= T < it['end']]
        need_mask = bool(live) or matte
        alpha = cut.next(fr) if need_mask else None
        if want is not None and i not in want: continue
        rgb = fr.astype(np.float32) / 255
        face = ctx.faces[i]
        for it in live:
            s = Shot(ctx, i, rgb, alpha, face, it['start'], it['end'], it['state'])
            EFFECTS[it['name']]['fn'](s, it['params'])
            rgb, alpha, face = s.rgb, s.alpha, s.face
        img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        if writer: writer.stdin.write(img.tobytes())
        else:
            root, _ = os.path.splitext(out); path = f'{root}_{T:.2f}.png'
            Image.fromarray(img).save(path); saved.append(path); log(f'saved {path}')
        if writer and i % int(fps * 5) == 0: log(f'  {T:5.1f}s / {dur:.1f}s')
    if writer:
        writer.stdin.close(); writer.wait(); log(f'Done: {out}')
    return saved
