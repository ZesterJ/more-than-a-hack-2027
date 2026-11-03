# Barca Vision Analyzer -- possession model vs ground truth

A browser tool for inspecting a possession model. Load a model's submission CSV and see, frame by
frame, where it agrees and disagrees with the ground truth, next to the match video (optional) and
a 2D radar drawn from the tracking data.

It does **not** label anything by hand. Every label shown is either the public ground truth or a
model you loaded. (The folder is still called `labeler/` so the launchers keep working.)

---

## Setup

```bash
pip install -r labeler/requirements.txt        # flask, numpy, pandas
```

Data (see `DATA.md`):
- Tracking: `data/tactical/BAR-ATH_tactical_atd.txt` and `.xml` (required)
- Video: `BAR-ATH_broadcast.mp4` or `BAR-ATH_tactical.mp4` in the repo root, `data/`, or
  `data/broadcast/` (optional; or use **Select Video** in the app)

Launch:
- Windows: double-click `run_labeler.bat`
- Mac / Linux: `./run_labeler.sh`, or `python labeler/run_labeler.py`

The app opens at `http://127.0.0.1:5000`. `baseline_smooth50` is loaded on start so there is
something to look at.

---

## Using it

1. **Load a model.** Click **Load model CSV** and pick a `frame,label` file (199,319 rows, labels
   `H` `A` `D` `X`), or pick one from `evaluation/results/` in the dropdown. Load several to compare.
   Files are checked with the official validator, and a bad file shows the reason.
2. **Timeline (bottom).** One row for the ground truth, then one row per model with a thin red row
   beneath it marking every frame where that model is wrong. Click to jump; zoom with 15m / 5m / Full.
3. **Truth vs Predictions (sidebar).** At the current frame: the true label, each model's label, and
   whether it is correct.
4. **Model Analysis (sidebar).** Pick a model to see:
   - spellF1, macroF1, accuracy, possession count and mean length versus the truth, per-class F1
     (all from the official `possession_metrics.py`)
   - **Where the errors are:** ball visible vs missing, near a handover (within 2 s of a real change
     of possession) vs mid-possession
   - **What it confuses:** e.g. "truth D, said H"
   - **Longest wrong stretches:** click one to jump to it
5. **Radar and video.** The radar shows both teams, the ball and its trail, and the nearest players.
   With a video loaded it plays in sync with the radar.

Press **E** / **Shift+E** to jump to the next / previous error of the analysed model.

---

## Important limits

- **Ground truth only exists for frames 11,185-18,684** (the first 5 minutes, 17 possessions). Errors
  can only be marked there. Elsewhere you still see the model's labels, but no verdict.
- **Use it to diagnose, not to rank.** 17 possessions is too few to rank models; one possession moves
  `spellF1` by about 6 points. A model trained on those same 5 minutes will look better here than
  it really is.
- **Video sync.** The tracking is indexed to the tactical camera; the broadcast video is offset
  (default 4,013 frames, adjustable with the -1f / +1f buttons or Alt + arrow keys).

---

## Keyboard shortcuts

| Key | Action |
| :--- | :--- |
| `Space` | Play / pause (video + radar) |
| `Left` / `Right` | Step 1 frame |
| `Shift` + `Left` / `Right` | Step 1 second |
| `Ctrl` + `Left` / `Right` | Step 5 seconds |
| `E` / `Shift+E` | Next / previous error of the analysed model |
| `Alt` + `Left` / `Right` | Nudge video offset by 1 frame (`Shift` = 10) |

---

## Under the hood

`labeler/app.py` serves the tracking frames and a small JSON API:

| Endpoint | Purpose |
| :--- | :--- |
| `GET /api/frame/<n>` | Players, ball, ground-truth label and each model's label for frame `n` |
| `GET /api/models`, `POST /api/models/load`, `POST /api/models/upload`, `POST /api/models/remove` | Manage loaded models |
| `GET /api/ribbon` | Run-length encoded truth, models and error stretches for the timeline |
| `GET /api/analysis?name=<model>` | Official metrics plus the error breakdown and wrong-stretch list |

Which slots belong to which team is read from the ATD metadata XML (slots 1-24 are home, 25-46 away
for this match), not hard-coded.
