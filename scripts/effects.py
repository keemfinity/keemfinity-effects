#!/usr/bin/env python3
"""Keemfinity Effects: 48 video effects for Claude, by @keemfinity.

One effect:    python effects.py led-sign in.mp4 out.mp4 --start 2 --end 4.5 --text KEEM
Many effects:  python effects.py --plan plan.json in.mp4 out.mp4
Check frames:  python effects.py --plan plan.json in.mp4 check.png --still 2.5,7
All effects:   python effects.py --list
Options:       --width 2160 (4K), --matte matte.mkv (pro cut-out), --opt key=value (repeatable)
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keemfinity import engine  # noqa: E402

GROUPS = ['Camera & Light', 'Words Behind You', 'Your Reels', 'Claude At Work', 'Social', 'Numbers']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('effect', nargs='?', help='effect name (or omit and use --plan)')
    ap.add_argument('src', nargs='?'); ap.add_argument('out', nargs='?')
    ap.add_argument('--plan', help='JSON file: {"effects": [{"fx": "orbit", "start": 3, "end": 6, ...options}]}')
    ap.add_argument('--start', type=float, default=0); ap.add_argument('--end', type=float)
    ap.add_argument('--text'); ap.add_argument('--opt', action='append', default=[], help='key=value')
    ap.add_argument('--still', help='comma-separated seconds: save PNG check frames instead of a video')
    ap.add_argument('--width', type=int, default=1080, help='output width: 1080 (default) or 2160 for 4K')
    ap.add_argument('--matte', help='greyscale matte video from matte.py, for the pro cut-out')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()

    if a.list:
        engine.load_effects(); k = 0
        for g in GROUPS:
            print(f'\n{g}')
            for e in [e for e in engine.EFFECTS.values() if e['group'] == g]:
                k += 1; print(f'{k:3d}. {e["name"]:<24}{e["desc"]}')
        rest = [e for e in engine.EFFECTS.values() if e['group'] not in GROUPS]
        for e in rest: k += 1; print(f'{k:3d}. {e["name"]:<24}{e["desc"]}')
        print(f'\n{k} effects'); return

    if a.plan:
        src, out = a.effect, a.src
        with open(a.plan, encoding='utf-8') as f: plan = json.load(f)
        plan = plan['effects'] if isinstance(plan, dict) else plan
    else:
        if not (a.effect and a.src and a.out): ap.error('give an effect, an input and an output (or --plan)')
        src, out = a.src, a.out
        d = {'fx': a.effect, 'start': a.start}
        if a.end is not None: d['end'] = a.end
        if a.text is not None: d['text'] = a.text
        for kv in a.opt:
            k, _, v = kv.partition('='); d[k.strip()] = v
        plan = [d]
    if not (src and out): ap.error('give an input video and an output path')
    stills = [float(x) for x in a.still.split(',')] if a.still else None
    engine.run(src, out, plan, width=a.width, stills=stills, matte=a.matte)


if __name__ == '__main__':
    main()
