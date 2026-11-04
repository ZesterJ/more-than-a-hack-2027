"""Load a window of tracking and turn it into per-frame, permutation-invariant features.

The 46 tracking slots are tracklets, not players (no re-identification, see 02_tracking_atd.ipynb),
and the ball is missing on ~43% of graded frames. So every feature here is collective -- team
centroids, spreads, speeds, nearest-player margins -- and degrades to a neutral value, never to NaN,
when the ball is not tracked.

    from features import load_tracking, build_features, load_gt
    trk = load_tracking(n_frames=33685)              # video frames 0..33684, cached to data/cache
    X = build_features(trk)                          # DataFrame, one row per video frame, 22 columns

numpy and pandas only, like the scorer.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, HERE)
from games import GAME, GAMES, DATA_DIR, require  # noqa: E402
import possession_metrics as pm  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

# Params
FPS = 25
N_SLOTS = 46
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
# Real labels exist for video frames [11185, 18685) only: 5 minutes at 25 fps.
LABEL_START, LABEL_STOP = 11185, 18685
# The second half kicks off here (video frame). Used only to mirror x so "attacking right" means the
# same thing in both halves when features are built for the whole match.
SECOND_HALF_START = 113742
NEAR_M = 3.0  # "in a duel" radius around the ball, metres
D_MARGIN_CLIP = 30.0

FEATURES = [
    # ball proximity
    "d_margin", "ball_missing", "ball_speed", "home_within_3m", "away_within_3m",
    # collective geometry
    "home_cx", "home_cy", "away_cx", "away_cy", "centroid_diff_x",
    "home_spread_x", "home_spread_y", "away_spread_x", "away_spread_y", "spread_ratio",
    # kinematics
    "mean_speed_home", "mean_speed_away", "overall_speed",
    # temporal context
    "d_margin_roll_1s", "d_margin_roll_3s", "speed_all_roll_2s", "missing_ratio_roll_2s",
]


def cache_path(n_frames):
    return os.path.join(DATA_DIR, "cache", "atd_first_{}.npz".format(n_frames))


def load_tracking(n_frames=33685, game=GAME, data_dir=None, cache=True):
    """First `n_frames` video frames of tracking: {'players': (n, 46, 3), 'ball': (n, 3)}.

    The last axis is (x, y, speed): x and y normalised 0-1, speed in m/s, NaN where unobserved.
    Row i is video frame i (the raw counter starts at 1, so frame_id - 1). The 15-minute window the
    benchmark uses is ~34k lines and parses in a few seconds; the full match takes about a minute,
    which is why the result is cached under data/cache (git-ignored like everything in data/).
    """
    path = cache_path(n_frames)
    if cache and os.path.isfile(path):
        z = np.load(path)
        return {"players": z["players"], "ball": z["ball"]}

    raw_path = require(["atd_raw"], game, data_dir)[0]
    players = np.full((n_frames, N_SLOTS, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if i >= n_frames:
                break
            players[i] = np.array(parts[1].replace(";", ",").split(","),
                                  dtype=np.float32).reshape(N_SLOTS, 3)
            ball[i] = np.array(parts[2].split(","), dtype=np.float32)

    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, players=players, ball=ball)
    return {"players": players, "ball": ball}


def _rolling_mean(values, width):
    """Centred, NaN-aware moving average; NaN only where nothing in the window was observed."""
    half = width // 2
    ok = ~np.isnan(values)
    csum = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0))))
    ccnt = np.concatenate(([0], np.cumsum(ok)))
    idx = np.arange(len(values))
    lo, hi = np.maximum(0, idx - half), np.minimum(len(values), idx + half + 1)
    cnt = ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (csum[hi] - csum[lo]) / np.maximum(cnt, 1), np.nan)


def _nan_stat(fn, a, axis):
    """nanmean / nanstd without the all-NaN RuntimeWarning noise."""
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return fn(a, axis=axis)


def build_features(trk, game=GAME, data_dir=None, flip_second_half=False, is_home=None):
    """The 22 collective features, one row per frame, all finite.

    `flip_second_half` mirrors x after SECOND_HALF_START; irrelevant inside the first 15 minutes but
    needed before scoring the whole match, because the teams swap ends at half time.
    """
    length, width = GAMES[game]["pitch"]
    if is_home is None:  # the ATD slots; external data passes its own team mask
        is_home = team_of_each_track(require(["atd_meta"], game, data_dir)[0]) == 0

    P, B = trk["players"], trk["ball"]
    n = len(B)
    px, py, ps = P[:, :, 0] * length, P[:, :, 1] * width, P[:, :, 2]
    bx, by, bs = B[:, 0] * length, B[:, 1] * width, B[:, 2]
    if flip_second_half:
        late = np.arange(n) >= SECOND_HALF_START
        px = np.where(late[:, None], length - px, px)
        bx = np.where(late, length - bx, bx)

    missing = np.isnan(bx)
    dist = np.hypot(px - bx[:, None], py - by[:, None])  # NaN if either side is unobserved
    dist_home = np.where(is_home[None, :], dist, np.nan)
    dist_away = np.where(~is_home[None, :], dist, np.nan)
    d_home = _nan_stat(np.nanmin, np.where(np.isnan(dist_home), np.inf, dist_home), 1)
    d_away = _nan_stat(np.nanmin, np.where(np.isnan(dist_away), np.inf, dist_away), 1)
    d_margin = d_away - d_home
    d_margin = np.where(np.isfinite(d_margin), np.clip(d_margin, -D_MARGIN_CLIP, D_MARGIN_CLIP), np.nan)

    f = pd.DataFrame(index=np.arange(n))
    f["d_margin"] = np.nan_to_num(d_margin)
    f["ball_missing"] = missing.astype(np.float32)
    f["ball_speed"] = np.nan_to_num(bs)
    with np.errstate(invalid="ignore"):
        f["home_within_3m"] = np.nansum(dist_home < NEAR_M, axis=1)
        f["away_within_3m"] = np.nansum(dist_away < NEAR_M, axis=1)

    for name, mask in (("home", is_home), ("away", ~is_home)):
        x, y, s = px[:, mask], py[:, mask], ps[:, mask]
        f[name + "_cx"] = _nan_stat(np.nanmean, x, 1)
        f[name + "_cy"] = _nan_stat(np.nanmean, y, 1)
        f[name + "_spread_x"] = _nan_stat(np.nanstd, x, 1)
        f[name + "_spread_y"] = _nan_stat(np.nanstd, y, 1)
        f["mean_speed_" + name] = _nan_stat(np.nanmean, s, 1)
    f["centroid_diff_x"] = f["home_cx"] - f["away_cx"]
    f["spread_ratio"] = (f["home_spread_x"] * f["home_spread_y"]) / \
        (f["away_spread_x"] * f["away_spread_y"]).replace(0, np.nan)
    f["overall_speed"] = _nan_stat(np.nanmean, ps, 1)

    f["d_margin_roll_1s"] = _rolling_mean(d_margin, FPS)
    f["d_margin_roll_3s"] = _rolling_mean(d_margin, 3 * FPS)
    f["speed_all_roll_2s"] = _rolling_mean(f["overall_speed"].values, 2 * FPS)
    f["missing_ratio_roll_2s"] = _rolling_mean(missing.astype(float), 2 * FPS)

    f = f[FEATURES].replace([np.inf, -np.inf], np.nan)
    return f.ffill().bfill().fillna(0.0).astype(np.float32)


EXTRA_FEATURES = [
    "d_min", "ball_accel", "ball_dwell_s", "ball_touchline", "ball_goalline",
    "ball_x", "ball_y", "speed_diff", "n_within_10m", "ball_missing_run_s",
]


def _run_length_since_change(flag):
    """Frames since `flag` last changed value (0 on the change frame)."""
    idx = np.arange(len(flag))
    change = np.concatenate(([True], flag[1:] != flag[:-1]))
    return idx - np.maximum.accumulate(np.where(change, idx, 0))


def build_extra(trk, X, game=GAME, data_dir=None, flip_second_half=False):
    """Ten kinematic / ball-state features on top of the 22 (the teammates' report lists ~10).

    d_min is the distance from the ball to the nearest player of either team (30 m when the ball is
    unseen, so "no contact" never looks like contact); ball_accel is |d speed / dt| of the tracked
    ball speed; ball_dwell_s is how long the nearest team has stayed the same; ball_missing_run_s is
    how long the ball has been continuously unseen, which separates a lofted pass from a stoppage.
    Ball position is forward-filled through gaps for the pitch-position features.
    """
    length, width = GAMES[game]["pitch"]
    P, B = trk["players"], trk["ball"]
    n = len(B)
    px, py = P[:, :, 0] * length, P[:, :, 1] * width
    bx, by, bs = B[:, 0] * length, B[:, 1] * width, B[:, 2]
    if flip_second_half:
        late = np.arange(n) >= SECOND_HALF_START
        px = np.where(late[:, None], length - px, px)
        bx = np.where(late, length - bx, bx)
    missing = np.isnan(bx)
    dist = np.hypot(px - bx[:, None], py - by[:, None])

    f = pd.DataFrame(index=np.arange(n))
    dmin = _nan_stat(np.nanmin, np.where(np.isnan(dist), np.inf, dist), 1)
    f["d_min"] = np.where(np.isfinite(dmin), np.minimum(dmin, D_MARGIN_CLIP), D_MARGIN_CLIP)
    acc = np.abs(np.diff(np.nan_to_num(bs), prepend=0.0)) * FPS
    f["ball_accel"] = np.minimum(_rolling_mean(acc, 3), 100.0)
    nearest_home = (X["d_margin"].values > 0).astype(np.int8)
    team_seen = np.where(missing, -1, nearest_home)
    held = pd.Series(np.where(team_seen < 0, np.nan, team_seen)).ffill().fillna(0).values.astype(np.int8)
    f["ball_dwell_s"] = np.minimum(_run_length_since_change(held) / FPS, 10.0)
    bxf = pd.Series(bx).ffill().bfill().fillna(length / 2).values
    byf = pd.Series(by).ffill().bfill().fillna(width / 2).values
    f["ball_touchline"] = np.minimum(byf, width - byf)
    f["ball_goalline"] = np.minimum(bxf, length - bxf)
    f["ball_x"], f["ball_y"] = bxf / length, byf / width
    f["speed_diff"] = X["mean_speed_home"].values - X["mean_speed_away"].values
    with np.errstate(invalid="ignore"):
        f["n_within_10m"] = np.nansum(dist < 10.0, axis=1)
    f["ball_missing_run_s"] = np.minimum(
        np.where(missing, _run_length_since_change(missing.astype(np.int8)), 0) / FPS, 10.0)
    return f[EXTRA_FEATURES].astype(np.float32)


TAG_FEATURES = ["tag_home", "tag_away", "tag_set_piece"]


def smart_tag_features(n_frames, game=GAME, data_dir=None):
    """Smart Tagging as per-frame *priors*: how many tagged phases of each team cover the frame,
    and whether a set-piece restart does. It is automatic, ~1 s accurate and contains false
    positives (see load_smart_tagging.py), so it goes in as a feature a model may or may not trust,
    never as a label. Home is Barcelona, the first team in the ATD metadata.
    """
    import load_smart_tagging as st
    out = np.zeros((n_frames, 3), dtype=np.float32)
    for phase in st.load(game, data_dir):
        a, b = phase["start_frame"], min(phase["end_frame"], n_frames)
        if a >= n_frames:
            continue
        team = str(dict(phase["tags"]).get("Team", ""))
        if phase["code"].lower() == "set pieces":
            out[a:b, 2] = 1
        elif team == "Barcelona":
            out[a:b, 0] += 1
        elif team.startswith("At"):
            out[a:b, 1] += 1
    return pd.DataFrame(out, columns=TAG_FEATURES)


def nearest_player_codes(X):
    """Baseline signal from the features: 0 = home closer, 1 = away closer, -1 = ball unseen."""
    codes = np.where(X["d_margin"].values > 0, 0, 1)
    return np.where(X["ball_missing"].values > 0, -1, codes).astype(np.int8)


def load_gt(path=PUBLIC_GT):
    """Full-length ground-truth codes (0 H, 1 A, 2 D, 3 X) as the scorer reads them."""
    return pm.label_codes(pm.read_label_csv(path))
