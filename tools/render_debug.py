"""Render a debug movie: tracking on a 2D pitch, with predictions and ground truth beside it.

No video needed -- it draws from the tracking file alone, so it works before the MP4s arrive.

    python tools/render_debug.py                                  # first 5 min vs baseline_smooth50
    python tools/render_debug.py --start 12000 --end 13500        # a specific window (video frames)
    python tools/render_debug.py --pred evaluation/results/baseline_smooth50.csv mine.csv
    python tools/render_debug.py --errors-only                    # only clips around wrong frames

What you see
  top    pitch: home (blue) / away (red) tracks, ball + 1 s trail, line to the nearest player,
         "BALL MISSING" banner when the ball is NaN (41% of graded frames)
  title  frame, clock, ground-truth label, each prediction's label, active Smart Tagging phases
  below  timeline: GT row, one row per prediction, an ERR row (red where a prediction disagrees
         with GT), and a Smart Tagging team row. White cursor = current frame.

Ground truth only exists for frames 11185..18684 (the public 5 min); outside it GT shows as X.
Needs: numpy, matplotlib, and ffmpeg on PATH.
"""

import argparse
import itertools
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
import load_smart_tagging  # noqa: E402

HOME_C, AWAY_C, DEAD_C, X_C = "#2f6fdb", "#e0443e", "#8a8a8a", "#2b2b2b"
LABEL_C = {"H": HOME_C, "A": AWAY_C, "D": DEAD_C, "X": X_C}
DEFAULT_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
DEFAULT_PRED = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")


def read_labels(path):
    import pandas as pd
    df = pd.read_csv(path)
    return df["label"].astype(str).str.strip().str.upper().to_numpy()


def read_tracking(raw_path, start, end, n_slots):
    """players[f, slot, 2] and ball[f, 2] (0-1 coords, NaN where missing) for video frames start..end."""
    n = end - start + 1
    players = np.full((n, n_slots, 2), np.nan)
    ball = np.full((n, 2), np.nan)
    with open(raw_path) as handle:
        for i, line in enumerate(itertools.islice(handle, start, end + 1)):
            parts = line.split(":")
            if len(parts) != 3:
                continue
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                v = group.split(",")
                players[i, slot] = (float(v[0]), float(v[1]))
            v = parts[2].split(",")
            ball[i] = (float(v[0]), float(v[1]))
    return players, ball


def error_windows(gt, preds, start, end, pad):
    """Merged [a, b] frame ranges around frames where any prediction disagrees with a scored GT."""
    sl = slice(start, end + 1)
    bad = np.zeros(end - start + 1, bool)
    for p in preds:
        bad |= (gt[sl] != "X") & (p[sl] != gt[sl])
    idx = np.flatnonzero(bad)
    out = []
    for i in idx:
        a, b = max(0, i - pad), min(len(bad) - 1, i + pad)
        if out and a <= out[-1][1]:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(start + a, start + b) for a, b in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=int, default=11185)
    ap.add_argument("--end", type=int, default=18684)
    ap.add_argument("--pred", nargs="+", default=[DEFAULT_PRED], help="prediction CSV(s)")
    ap.add_argument("--gt", default=DEFAULT_GT)
    ap.add_argument("--step", type=int, default=2, help="render every Nth frame (default 2 = 12.5 fps of data)")
    ap.add_argument("--fps", type=float, default=25.0, help="playback fps of the movie")
    ap.add_argument("--errors-only", action="store_true", help="only render +/- --pad frames around errors")
    ap.add_argument("--pad", type=int, default=75)
    ap.add_argument("--out", default=os.path.join(ROOT, "debug_movie.mp4"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    meta, raw = require(["atd_meta", "atd_raw"], GAME)
    L, W = GAMES[GAME]["pitch"]
    team = team_of_each_track(meta)
    gt = read_labels(args.gt)
    preds = [read_labels(p) for p in args.pred]
    names = [os.path.splitext(os.path.basename(p))[0] for p in args.pred]
    for arr, p in zip([gt] + preds, [args.gt] + args.pred):
        if len(arr) != GAMES[GAME]["n_video_frames"]:
            sys.exit("{}: {} rows, expected {}".format(p, len(arr), GAMES[GAME]["n_video_frames"]))

    if args.errors_only:
        spans = error_windows(gt, preds, args.start, args.end, args.pad)
        if not spans:
            sys.exit("no errors in range -- nothing to render")
        frames = [f for a, b in spans for f in range(a, b + 1, args.step)]
        print("{} error windows, {} frames".format(len(spans), len(frames)))
    else:
        frames = list(range(args.start, args.end + 1, args.step))
    lo, hi = min(frames), max(frames)

    players, ball = read_tracking(raw, lo, hi, len(team))
    tags = load_smart_tagging.load()

    # --- static layout -------------------------------------------------------------------------
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#1b1b1b")
    ax = fig.add_axes([0.03, 0.30, 0.94, 0.56])
    ax.set_xlim(-2, L + 2); ax.set_ylim(W + 2, -2); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((-2, -2), L + 4, W + 4, fc="#2e7d46", ec="none", zorder=0))
    kw = dict(fill=False, ec="white", lw=1.2)
    ax.add_patch(Rectangle((0, 0), L, W, **kw))
    ax.plot([L / 2] * 2, [0, W], c="white", lw=1.2)
    ax.add_patch(Circle((L / 2, W / 2), 9.15, **kw))
    for x0, sgn in ((0, 1), (L, -1)):
        ax.add_patch(Rectangle((x0 if sgn > 0 else L - 16.5, W / 2 - 20.15), 16.5, 40.3, **kw))
        ax.add_patch(Rectangle((x0 if sgn > 0 else L - 5.5, W / 2 - 9.16), 5.5, 18.32, **kw))

    n_rows = 2 + len(preds) + 1
    tl = fig.add_axes([0.12, 0.05, 0.85, 0.20], facecolor="#1b1b1b")
    tl.set_xlim(lo, hi); tl.set_ylim(n_rows, 0)
    tl.set_yticks([i + 0.5 for i in range(n_rows)])
    row_names = ["GT"] + names + ["ERR", "TAG team"]
    tl.set_yticklabels(row_names, color="white", fontsize=8)
    tl.tick_params(colors="white", labelsize=8)
    for s in tl.spines.values():
        s.set_visible(False)

    def runs(arr):
        a, i = arr, 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[j + 1] == a[i]:
                j += 1
            yield i, j, a[i]
            i = j + 1

    def draw_row(row, arr, colours):
        for i, j, v in runs(arr):
            c = colours.get(v)
            if c:
                tl.add_patch(Rectangle((lo + i, row + 0.1), j - i + 1, 0.8, fc=c, ec="none"))

    sl = slice(lo, hi + 1)
    draw_row(0, gt[sl], LABEL_C)
    err_any = np.zeros(hi - lo + 1, bool)
    for k, p in enumerate(preds):
        draw_row(1 + k, p[sl], LABEL_C)
        err_any |= (gt[sl] != "X") & (p[sl] != gt[sl])
    draw_row(1 + len(preds), np.where(err_any, "E", "-"), {"E": "#ff2d2d"})
    tag_row = np.full(hi - lo + 1, "-", dtype="<U1")
    for t in tags:
        tm = {"Barcelona": "H", "Atletic Club": "A"}.get(t["tags"].get("Team"))
        if tm:
            tag_row[max(t["start_frame"], lo) - lo:max(0, min(t["end_frame"], hi + 1) - lo)] = tm
    draw_row(2 + len(preds), tag_row, {"H": HOME_C, "A": AWAY_C})
    cursor = tl.axvline(lo, c="white", lw=1.5)

    # --- dynamic artists -----------------------------------------------------------------------
    hs = ax.scatter([], [], s=140, c=HOME_C, ec="white", lw=1, zorder=3)
    as_ = ax.scatter([], [], s=140, c=AWAY_C, ec="white", lw=1, zorder=3)
    trail, = ax.plot([], [], c="yellow", lw=1.5, alpha=0.6, zorder=4)
    bd = ax.scatter([], [], s=90, c="white", ec="black", lw=1.5, zorder=5)
    near, = ax.plot([], [], c="white", lw=1, ls="--", zorder=2)
    banner = ax.text(L / 2, 4, "", color="yellow", ha="center", fontsize=14, weight="bold", zorder=6)
    title = fig.text(0.03, 0.985, "", color="white", fontsize=12, family="monospace", va="top")

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
         "-s", "1280x720", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "20", args.out], stdin=subprocess.PIPE)

    def phases(f):
        return ", ".join("{}({})".format(t["code"].title(), (t["tags"].get("Team") or "-")[:3])
                         for t in tags if t["start_frame"] <= f < t["end_frame"])

    for n, f in enumerate(frames):
        i = f - lo
        px, py = players[i, :, 0] * L, players[i, :, 1] * W
        ok = ~np.isnan(px)
        hm, am = ok & (team == 0), ok & (team == 1)
        hs.set_offsets(np.c_[px[hm], py[hm]]); as_.set_offsets(np.c_[px[am], py[am]])
        b = ball[i] * [L, W]
        if np.isnan(b[0]):
            bd.set_offsets(np.empty((0, 2))); near.set_data([], []); trail.set_data([], [])
            banner.set_text("BALL MISSING")
        else:
            banner.set_text("")
            bd.set_offsets([b])
            tr = ball[max(0, i - 25):i + 1] * [L, W]
            tr = tr[~np.isnan(tr[:, 0])]
            trail.set_data(tr[:, 0], tr[:, 1])
            if ok.any():
                d = (px - b[0]) ** 2 + (py - b[1]) ** 2
                d[~ok] = np.inf
                k = int(np.argmin(d))
                near.set_data([b[0], px[k]], [b[1], py[k]])
        secs = f / 25.0
        lines = ["frame {:>6}  {:02d}:{:05.2f}  GT:{}".format(f, int(secs // 60), secs % 60, gt[f])]
        for nm, p in zip(names, preds):
            flag = "" if gt[f] == "X" else ("  ok" if p[f] == gt[f] else "  <-- WRONG")
            lines.append("  {} : {}{}".format(nm[:22], p[f], flag))
        title.set_text("\n".join(lines))
        ax.set_title(phases(f), color="#ddd", fontsize=9, pad=4)
        cursor.set_xdata([f, f])
        fig.canvas.draw()
        ffmpeg.stdin.write(fig.canvas.buffer_rgba())
        if n % 200 == 0:
            print("  {}/{}".format(n, len(frames)), flush=True)

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
