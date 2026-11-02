# Debug Movie Tool

`render_debug.py` makes a video that replays the match from above (2D pitch, drawn from the tracking
data) and shows your model's possession predictions next to the ground truth, so you can see **where
and why** a model gets it wrong. It does not need the match video.

It is a visual debugger, not a scorer. For numbers use `python evaluation/run_evaluation.py`.

---

## Setup

You need the match data in `data/` (see `DATA.md`), plus:

```bash
pip install numpy pandas matplotlib
```

and `ffmpeg` on your PATH (check with `ffmpeg -version`).

---

## Quick start

Run from the repo root:

```bash
python tools/render_debug.py
```

This renders the first 5 minutes against `baseline_smooth50` and writes `debug_movie.mp4` in the repo
root. Rendering takes a couple of minutes. The `.mp4` is gitignored.

---

## Testing your own model

1. Make your model write a normal submission CSV (`frame,label`, 199,319 rows, labels `H` `A` `D` `X`).
2. Point the tool at it:

```bash
python tools/render_debug.py --pred path/to/my_model.csv
```

To compare several submissions, list them all. Each gets its own row:

```bash
python tools/render_debug.py --pred evaluation/results/baseline_smooth50.csv my_model.csv
```

---

## Options

| Option | Default | What it does |
|---|---|---|
| `--pred A.csv B.csv ...` | `baseline_smooth50.csv` | Prediction file(s) to show |
| `--gt file.csv` | public 5-minute ground truth | Truth to compare against |
| `--start N` / `--end N` | 11185 / 18684 | Frame range to render (video frames, 25 per second) |
| `--step N` | 2 | Draw every Nth frame. Higher is faster to render, lower is smoother |
| `--fps N` | 25 | Playback speed of the video |
| `--errors-only` | off | Only render short clips around frames where a prediction is wrong |
| `--pad N` | 75 | Frames of context before and after each error (with `--errors-only`) |
| `--out file.mp4` | `debug_movie.mp4` | Output file |

### Playback speed

Playback time is `frames drawn / fps`. With `--step 3` and the default `--fps 25` the video plays 3x
faster than real time. For real time, set `--fps` to `25 / step`:

```bash
python tools/render_debug.py --step 3 --fps 8.33     # real time
python tools/render_debug.py --step 1 --fps 10       # slow motion, smooth
```

### Useful recipes

```bash
# Only the mistakes (fastest way to find problems)
python tools/render_debug.py --errors-only --pred my_model.csv

# Slow-motion look at one moment
python tools/render_debug.py --start 11300 --end 11600 --step 1 --fps 10

# Look at part of the match that has no ground truth (labels show, errors cannot)
python tools/render_debug.py --start 30000 --end 31500 --pred my_model.csv
```

---

## How to read the video

**Pitch (middle)**
- Blue dots are the home team (Barcelona), red dots are the away team (Athletic).
- White dot with a yellow tail is the ball and its last second of movement.
- Dashed line from the ball to the nearest player (what the baseline uses to decide possession).
- `BALL MISSING` appears when the tracker has lost the ball (about 40% of graded frames).
- Text above the pitch lists the active Smart Tagging phases, e.g. `Progression (Atl)`.

**Top left**
- Frame number and match clock.
- `GT:` is the true label: `H` home has the ball, `A` away, `D` dead ball, `X` no ground truth here.
- One line per prediction, marked `ok` or `<-- WRONG`.

**Timeline (bottom)**
- Time runs left to right. Blue = home, red = away, grey = dead ball.
- Rows: `GT`, one row per prediction, `ERR`, `TAG team`.
- `ERR` is bright red wherever any prediction disagrees with the truth.
- The white vertical line is the current moment.

Watch what happens when `ERR` turns red: is the ball missing, is it a handover between teams, or a
stoppage? That tells you what your model needs to learn.

---

## Limits

- **Ground truth only covers frames 11185-18684** (the first 5 minutes). Outside that the `GT` row is
  `X` and no errors are marked.
- **It uses the tracking, not the video.** To watch the real footage, use the labeler app (needs the MP4s).
- **Beware of flattering results.** If your model was trained on the same 5 minutes it is judged on, it
  will look great here without being any good on the full match. Evaluate on frames the model has not
  seen.
- Prediction files must have exactly 199,319 rows. Validate with
  `python evaluation/validate_submission.py my_model.csv`.
