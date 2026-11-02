#!/usr/bin/env python3
"""Make a clean, full-resolution matte video of the person (BiRefNet portrait, GPU if available).

usage: python matte.py input.mp4 matte.mkv
The matte is a lossless greyscale video, same size and frame rate as the input. Pass it to effects.py with --matte.
"""
import subprocess, sys
import numpy as np
import cv2
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
try:
    import onnxruntime as ort
    ort.preload_dlls()
except Exception:
    pass
from rembg import new_session, remove

sess = new_session('birefnet-portrait', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                        '-c:v', 'ffv1', '-pix_fmt', 'gray', dst], stdin=subprocess.PIPE)
prev = None; i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    m = np.asarray(remove(Image.fromarray(rgb), session=sess, only_mask=True, post_process_mask=False), np.float32)
    if prev is not None: m = 0.75 * m + 0.25 * prev     # a touch of temporal smoothing, kills edge flicker
    prev = m
    enc.stdin.write(np.clip(m, 0, 255).astype(np.uint8).tobytes()); i += 1
    if i % 24 == 0: print(f'  {i}/{n}', flush=True)
enc.stdin.close(); enc.wait(); print('Done:', dst)
