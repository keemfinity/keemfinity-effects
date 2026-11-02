#!/usr/bin/env bash
# Keemfinity Effects: one-time setup. Re-running is safe; prints OK when everything is ready.
set -e
here="$(cd "$(dirname "$0")/.." && pwd)"
cd "$here"

need() { command -v "$1" >/dev/null 2>&1; }

if ! need ffmpeg || ! need ffprobe; then
  echo "MISSING ffmpeg: Mac 'brew install ffmpeg', Windows 'winget install ffmpeg', Linux 'sudo apt install ffmpeg'"
  exit 1
fi

if [ -d .venv/Scripts ]; then py=.venv/Scripts/python.exe; else py=.venv/bin/python; fi
if [ -x "$py" ] && "$py" -c "import cv2, mediapipe, numpy, PIL" >/dev/null 2>&1; then echo OK; exit 0; fi

base=""
for cand in python3.12 python3.11 python3.13 python3.10 python3 python; do
  if need "$cand" && "$cand" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' 2>/dev/null; then base="$cand"; break; fi
done
if [ -z "$base" ]; then
  echo "MISSING python: install Python 3.10+ from python.org, then run this again"
  exit 1
fi

"$base" -m venv .venv
if [ -d .venv/Scripts ]; then py=.venv/Scripts/python.exe; else py=.venv/bin/python; fi
"$py" -m pip install --quiet --upgrade pip
"$py" -m pip install --quiet -r requirements.txt
"$py" -c "import cv2, mediapipe, numpy, PIL; print('OK')"
