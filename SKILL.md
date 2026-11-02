---
name: effects
description: Keemfinity Effects. 48 motion-graphics effects for talking-head videos, from camera moves and light (spotlight, LED sign, slow push) to type behind the head (red collage, cover star, backdrop word), reel showcases (orbit, phone fan, place-in-grid), Claude-at-work UI (glass widgets, checklist, render split), social moments (documentary, comment box, iMessage pop-ups) and charts. Effects follow the speaker's face and sit behind their head. Use when the user types /effects, names an effect, or asks to add an effect, zoom, grade, sticker, caption card or animation to a video.
---

# Keemfinity Effects (by @keemfinity)

Whoever is asking is probably a creator, not a developer. Do the work for them, keep messages short, and don't hand them commands to run.

Paths are relative to this skill's folder. Python lives in `.venv` (`.venv/bin/python`, or `.venv/Scripts/python.exe` on Windows); `PY` below means that.

## Step 1 · Setup (first time only)
Run `bash scripts/setup.sh`. `OK` means ready (first run: 1-3 minutes).
- `MISSING ffmpeg`: install it for them (`brew install ffmpeg` on a Mac, `winget install ffmpeg` on Windows), then rerun.
- `MISSING python`: install Python 3.12 (`brew install python@3.12` / `winget install Python.Python.3.12`), then rerun.

## Step 2 · Video and effect
- The video is a file they drop in or a path they give. Vertical, person in frame, works best.
- Map their words to an effect name from the tables below. If you can't tell, show the group names and ask once.

## Step 3 · Timing
- "When I say X": run `PY scripts/words.py <video>` (installs nothing; if faster-whisper is missing, install it into the venv first) and use the word's start time.
- No timing given: 2-4 seconds is the sweet spot per effect. Ask which part only if it matters.

## Step 4 · Plan, check, render
1. Write `<video-name>.effects.json` next to the video:
   ```json
   {"effects": [
     {"fx": "red-collage", "start": 0.0, "end": 2.4, "text": "KEEM"},
     {"fx": "orbit", "start": 4.0, "end": 7.0, "images": "screenshots"}
   ]}
   ```
   Effects stack in list order; put camera moves (`locked-on`, `lean-in`, `slow-push`, `spotlight`) first.
2. Check before rendering: `PY scripts/effects.py --plan <plan> <video> <dir>/check.png --still 1.2,5.5` (mid-point of each effect). Open the PNGs and look. Text over the face, clipped edges or unreadable type? Adjust options and check again.
3. Render: `PY scripts/effects.py --plan <plan> <video> <video-name>_effects.mp4`. Audio is kept. Add `--width 2160` for 4K.
4. Pro cut-out (cleaner hair/cap edges, or when they ask): `PY -m pip install rembg onnxruntime` (`onnxruntime-gpu` with an NVIDIA card), then `PY scripts/matte.py <video> <video-name>_matte.mkv`, and add `--matte <video-name>_matte.mkv` to the check and render commands. It's slow on CPU, so warn them first.
5. Tell them where the file is, and that it's a draft: watch it before posting.

One effect without a plan: `PY scripts/effects.py led-sign <video> <out.mp4> --start 2 --end 4.5 --text KEEM --opt color=ffcc66`.
`PY scripts/effects.py --list` prints every effect.

## The 48 effects

### Camera & Light
| `fx` name | What it does | Options (all optional) |
|---|---|---|
| `locked-on` | Corner brackets snap onto the face, the camera follows | `zoom` (default 1.35), `text` (tag) |
| `lean-in` | A fast punch-in on one word | `zoom` (default 1.45, max 2), `speed` |
| `slow-push` | Slow push-in to the face, room blurs out | `zoom` (default 1.6), `blur` |
| `over-the-shoulder-zoom` | Push into one top corner with a liquid glass line graph beside the head, pull back, then the other corner with a related bar graph; the head stays in front | `text` (two labels with `\|`), `to` (two numbers with `\|`, e.g. `197K\|672K`), `zoom` (default 2.0), `color` |
| `spotlight` | Kills the lights, keeps the person lit | `text` (glowing word behind), `color`, `push` |
| `led-sign` | Lights drop, an LED sign types itself out behind the head | `text` (lines with `\|`), `color`, `push` |
| `moody` | A colour grade | `grade` (`moody`, `bw`, `warm`, `cool`, `pop`) |
| `swipe` | A before and after slider sweeps across the frame | `grade` (the "after" look) |
| `dot-reveal` | The shot builds out of glowing dots, then snaps to the full picture | `color`, `cell` |
| `rabbit-hole` | The frame tunnels into itself, spinning | `ratio`, `speed`, `spin` |

### Words Behind You
| `fx` name | What it does | Options (all optional) |
|---|---|---|
| `backdrop-word` | A giant word drops in behind the head | `text`, `color` (hex), `font` (`bold` or `serif`) |
| `red-collage` | B&W person, giant red word behind, kinetic lines stacked beside | `text` (red word), `lines` (with `\|`), `tag` |
| `pastel-word` | Lavender room, giant word behind, profile card gets followed | `text`, `name`, `handle`, `board` (room colour, hex) |
| `cover-star` | Magazine cover with the title behind the head | `text` (title), `lines` (with `\|`), `issue`, `color`, `look` |
| `headline-box` | Punch-in with a two-line headline, the second line in a white box | `text` (lines with `\|`), `zoom` |
| `split-headline` | Headline card slides down and pushes the person to the bottom half | `text` (lines with `\|`), `hl` (highlighted word), `pill` |
| `match-cut` | News pages cut on one word, locked in a box behind the head | `text` (the word, default Claude), `color` |
| `headline-wall` | Headlines pile up behind the head, then a total pops in front | `text` (`headline::meta\|...`), `total`, `label` |
| `scribble` | Marker doodles draw themselves around the person | `text` (label), `color` |
| `pause-point` | Freezes the frame and points with a hand-drawn arrow | `text` (label), `color`, `label_x` |
| `reject-stamp` | A big red stamp slams onto the screen | `text`, `color`, `angle`, `y` |

### Your Reels
| `fx` name | What it does | Options (all optional) |
|---|---|---|
| `orbit` | Reels circle the head: in front at the bottom, behind at the top | `images` (folder), `count`, `speed`, `size` |
| `phone-fan` | Reels fan out like a hand of phones under a headline | `images`, `text` (lines with `\|`) |
| `carousel-spin` | Reels spin around the person in a 3D carousel | `images`, `count`, `size`, `turns` |
| `place-in-grid` | Person shrinks into the centre tile of a 3x3 profile grid, 8 reels pop in with view counts | `images`, `views` (9 counts with `\|`, the 9th is theirs) |
| `profile-wall` | Profile grid fills the wall behind, a counter ticks up | `images`, `count`, `label` |
| `noise` | Reels pile up behind the head, one every quarter second | `images`, `size` |
| `lineup` | Numbered tiles fan out in an arc above the head | `text` (tile names with `\|`) |
| `check-pills` | Grid-paper room, reel boards behind, a cursor clicks blue pills | `text` (pills with `\|`), `images` |
| `reel-frame` | The video shrinks into an Instagram reel card | `text` (handle) |
| `broll-split` | B-roll on top with a price-tag sticker, person below | `images`, `text` (tag) |

### Claude At Work
| `fx` name | What it does | Options (all optional) |
|---|---|---|
| `glass-widgets` | Frosted glass widgets and mini artifacts float around the head, plus an Edited by Claude pill | `text` (big number), `name`, `handle`, `look` (`simple`) |
| `checklist` | A Claude task list under the chin: each step spins, then ticks green | `text` (`step::meta\|...`), `title` |
| `render-split` | Purple render window with checks, takes, caption pills, a grade bar and a playhead, person below | `title`, `sub`, `text` (`row::result\|...`), `captions`, `window` |
| `timeline` | Dark red grade with an editing timeline, REC timecode and glowing playhead | `text` (title), `sub`, `tc` |
| `apple-toggles` | An iPhone settings card flips its switches on, one line at a time | `text` (rows with `\|`), `ticks` (seconds with `\|`), `color`, `pos` (`below`) |
| `the-bill` | A thermal receipt prints out beside the head | `text` (lines with `\|`; `a::b` = dotted item, `---` = rule), `side` (`left`) |
| `todo` | A checklist card beside the head ticks off | `text` (items with `\|`), `ticks` (seconds with `\|`), `title` |
| `shortcut-rain` | Keyboard keys rain down behind the person | `text` (keys with `\|`), `count` |
| `uninstall` | A glitchy uninstall bar breaks the frame apart | `text` |

### Social
| `fx` name | What it does | Options (all optional) |
|---|---|---|
| `documentary` | B&W person on a dark set with a pink halo, series tag, three bold lines, a name bar and a Follow button that gets clicked | `text` (lines with `\|`), `hl` (pink word), `tag`, `name`, `handle`, `at` |
| `comment-box` | An Instagram comment bar types a word and posts it | `text` (default EFFECTS), `reply`, `chip` |
| `messages` | iMessage notifications pop in one after another, creamy glass with a soft glow | `text` (messages with `\|`), `title`, `y` |
| `tap` | A Follow button pops up and flips to Following | `at` (video second of the flip), `x`, `y` (0 to 1), `color` |

### Numbers
| `fx` name | What it does | Options (all optional) |
|---|---|---|
| `liquid-glass-graph` | Liquid glass card under the chin, a graph climbs while the number counts up | `text` (label), `from`, `to` (e.g. `197K`, `59%`), `color` |
| `chart-split` | Cream chart card over the top half (bars grow, numbers pop), person in a rounded card below | `title`, `by`, `bars` (`label::value\|...`) |
| `tier-list` | A tier list drops in and an item flies into S tier | `title`, `item`, `others` |
| `countdown-flip` | A flip clock counts down above the head | `from` |

## Rules
- The face stays clear. If a check frame shows anything over the face, move or shorten it.
- Real numbers only: charts and counters should show the creator's actual stats. Ask for them.
- Visuals only. Music and captions are theirs to add.
