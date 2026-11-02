#!/usr/bin/env python3
"""Word timings for a video, so an effect can start on a spoken word.

usage: python words.py clip.mp4            prints "start  end  word" per word and writes clip.words.json
Needs faster-whisper (pip install faster-whisper). The first run downloads a small speech model.
"""
import json, os, sys


def main():
    if len(sys.argv) < 2: raise SystemExit(__doc__)
    src = sys.argv[1]
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise SystemExit('Install it first:  python -m pip install faster-whisper')
    model = WhisperModel('small', device='auto', compute_type='int8')
    segments, _ = model.transcribe(src, word_timestamps=True, vad_filter=True)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({'word': w.word.strip(), 'start': round(w.start, 2), 'end': round(w.end, 2)})
            print(f"{w.start:7.2f} {w.end:7.2f}  {w.word.strip()}")
    out = os.path.splitext(src)[0] + '.words.json'
    with open(out, 'w', encoding='utf-8') as f: json.dump(words, f, indent=1)
    print(f'\n{len(words)} words -> {out}')


if __name__ == '__main__':
    main()
