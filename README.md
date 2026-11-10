# BetaSpotter

Takes a video of a bouldering attempt and writes a folder of annotated screenshots,
one per technique error, each with a suggested fix drawn on the same frame.

Detected in v1:

| Error | How it is measured |
| --- | --- |
| Bent arms | Elbow angle under 150° on a hanging arm while resting (lock-offs before a reach are ignored) |
| Wrong direction of pull | Forearm more than 45° off the hold's usable direction |
| Off balance | Centre of mass sideways outside the span of the hands and feet |
| Hips off the wall | Torso looks much shorter than usual (it is tilting towards the camera) |
| Feet cut loose | Both feet off while hanging on the hands |
| Uncontrolled swing | Hips moving fast with all four limbs placed |
| Sloppy foot placement | The same foot re-placed on the same spot several times |

It uses MediaPipe pose estimation plus hand-written rules, so it needs no training
data and runs on a laptop CPU.

## Install

You need Python 3.10 or newer and git.

```bash
git clone https://github.com/yzyun/BoulderAI.git
```

```bash
cd BoulderAI
```

```bash
python -m venv .venv
```

```bash
.venv/bin/pip install -e ".[dev]"
```

On Windows the last command is `.venv\Scripts\pip install -e ".[dev]"`, and the
program is then at `.venv\Scripts\betaspotter`. On macOS and Linux either call
`.venv/bin/betaspotter` or run `source .venv/bin/activate` first so that plain
`betaspotter` works, as in the examples below.

The pose model (about 9 MB) is downloaded to `~/.cache/betaspotter` the first time
you run it, so that run needs an internet connection. On a minimal Linux install
MediaPipe also needs the system package `libegl1`.

## Quick start

1. Film the climb from behind with the phone held still (propped on the floor or on
   a tripod), the whole body in frame, and only one person on the wall.
2. Run it, giving the colour of the holds on that problem:

   ```bash
   betaspotter analyse climb.mp4 --hold-colour yellow
   ```

3. Open the new folder under `out/` and go through the screenshots in order.

A 30-second clip takes roughly a minute on a laptop.

## Options

Hold colours: red, orange, yellow, green, blue, purple, pink, black, white. For
anything else pass an OpenCV HSV range with `--hsv h_lo,s_lo,v_lo,h_hi,s_hi,v_hi`.

| Option | What it does |
| --- | --- |
| `--hold-colour yellow` | Colour of the problem's holds |
| `--hsv 20,90,100,34,255,255` | Custom colour range instead of a named colour |
| `--out results` | Where to create the results folder (default `out`) |
| `--layout overlay` | Draw the fix over you in one panel instead of side by side |
| `--stride 2` | Analyse every 2nd frame: faster on long clips |
| `--model lite` | Pose model: `lite` (fastest), `full` (default) or `heavy` (most accurate) |
| `--holds holds.yaml` | Use a corrected hold list (see below) |
| `--config my.yaml` | Override rule thresholds (see Tuning) |

`betaspotter analyse --help` lists them all.

### Output

```
out/climb_20261110-143000/
├── 01_00m04.2s_bent-arms_left.png
├── 02_00m07.8s_hips-off-wall.png
├── 03_00m11.5s_wrong-pull-direction_right.png
└── findings.json
```

Each screenshot has up to three panels built from the same video frame:

- **Your climb**: the error marked in red with the measurement.
- **Suggested fix**: a green figure in the corrected position on the same holds,
  with arrows from where you were.
- **Alternative beta** (only when a different hold is clearly better): a blue figure
  using that hold.

A caption underneath gives the error, the measurement against its threshold, and the
fix in words. If nothing is flagged the folder holds a single `00_no-errors.png`.

### Correcting hold directions

Pull direction is guessed from each hold's outline: long flat holds are pulled down,
long upright ones sideways towards the route, round ones down (low confidence). A
camera cannot see which side of a hold is positive, so check the guess:

```bash
betaspotter holds climb.mp4 --hold-colour yellow --out holds.yaml
```

This writes `holds.yaml` and a numbered picture `holds.png`. Edit
`pull_direction_deg` (0 = right, 90 = down, 180 = left, 270 = up), set
`confident: true`, then:

```bash
betaspotter analyse climb.mp4 --holds holds.yaml
```

### Tuning

All thresholds live in `betaspotter/rules/config.yaml`. Copy it, change what you need
and pass it with `--config`; only the keys you include are overridden.

## Limitations

- One camera from behind gives no depth. "Hips off the wall" is inferred from how
  short the torso looks, which a low camera angle also affects.
- Hold directions are guesses until you correct them in `holds.yaml`.
- The fix and beta figures are 2D geometry (your bone lengths, hands and feet on
  holds), not physics. They ignore friction, hold quality, strength and flexibility,
  and can look contorted when no comfortable position exists.
- Fast dynamic moves can lose pose tracking; those frames are skipped.
- The thresholds are starting values and have not yet been tuned on real climbing
  footage.
- The plan's straight-legged-reach and high-heel checks are not in v1.

## Tests

```bash
.venv/bin/python -m pytest
```

The tests use synthetic climbers and walls (`tests/synth.py`), so no footage is needed.

## Layout

```
betaspotter/
  cli.py, pipeline.py      commands and the end-to-end flow
  pose/                    MediaPipe wrapper, smoothing
  holds/                   colour segmentation, pull-direction guess, holds.yaml
  analysis/                geometry, limb-hold contacts, static/move phases
  rules/                   one module per error, thresholds in config.yaml
  fixes/                   pose solver, per-rule corrections, alternative beta
  report/                  frame selection, drawing, output folder
```
