"""Per-frame features for the whole match, built from the tracking data alone.

One row per video frame (199,319 rows, index = video frame), so every method sees the same table
and the same alignment. The table is cached in data/ (gitignored) because parsing the 187 MB
tracking file takes about a minute.

    from experiments.features import load_features
    feats = load_features()

Distances are in metres. Missing observations stay NaN: the ball is missing on about 43% of
frames, and what to do about that is the method's job, not this file's.

Rolling features use centred windows, so a value at frame t looks at frames before and after it.
That is fine here -- the task is offline and only the output file is scored -- but a method that
had to run live could not use them.
"""

import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAME, GAMES, require, DATA_DIR  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

N_SLOTS = 46
FPS = 25
CACHE_VERSION = 1
CACHE_PATH = os.path.join(DATA_DIR, "experiments_features_v{}.pkl".format(CACHE_VERSION))
ROLL_WINDOWS = (25, 50, 75)  # 1 s, 2 s, 3 s


def _read_tracking(raw_path):
    """The raw ATD file as a float array: frame_id, 46 x (x, y, speed), ball (x, y, third value)."""
    with open(raw_path, "r") as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, na_values=["NaN"], dtype="float64")
    expected = 1 + N_SLOTS * 3 + 3
    if table.shape[1] != expected:
        raise ValueError("expected {} columns per tracking line, got {}".format(expected, table.shape[1]))
    return table.to_numpy()


def _min_distance(dist, mask):
    """Row-wise minimum over the slots in mask; NaN where no slot has a distance."""
    masked = np.where(mask[None, :] & ~np.isnan(dist), dist, np.inf)
    out = masked.min(axis=1)
    out[np.isinf(out)] = np.nan
    return out


def _team_stats(px, py, speed, mask):
    """Centroid, spread and mean speed of one team over the slots in mask, ignoring NaN."""
    valid = mask[None, :] & ~np.isnan(px)
    n = valid.sum(axis=1).astype(float)
    safe = np.where(n > 0, n, np.nan)
    x = np.where(valid, px, 0.0)
    y = np.where(valid, py, 0.0)
    cx, cy = x.sum(axis=1) / safe, y.sum(axis=1) / safe
    sx = np.sqrt(np.maximum((x ** 2).sum(axis=1) / safe - cx ** 2, 0.0))
    sy = np.sqrt(np.maximum((y ** 2).sum(axis=1) / safe - cy ** 2, 0.0))
    s_valid = mask[None, :] & ~np.isnan(speed)
    s_n = s_valid.sum(axis=1).astype(float)
    mean_speed = np.where(s_valid, speed, 0.0).sum(axis=1) / np.where(s_n > 0, s_n, np.nan)
    return cx, cy, sx, sy, mean_speed, n


def build_features(game=GAME):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    length, width = GAMES[game]["pitch"]
    n_frames = GAMES[game]["n_video_frames"]

    arr = _read_tracking(raw_path)
    if len(arr) != n_frames or not np.array_equal(arr[:, 0], np.arange(1, n_frames + 1)):
        raise ValueError("tracking file is not frames 1..{} in order".format(n_frames))

    players = arr[:, 1:1 + N_SLOTS * 3].reshape(n_frames, N_SLOTS, 3)
    px, py, speed = players[:, :, 0] * length, players[:, :, 1] * width, players[:, :, 2]
    bx, by = arr[:, 1 + N_SLOTS * 3] * length, arr[:, 2 + N_SLOTS * 3] * width

    team = team_of_each_track(meta_path)
    home, away = team == 0, team == 1

    dist = np.hypot(px - bx[:, None], py - by[:, None])  # NaN where the ball or the player is missing
    d_home, d_away = _min_distance(dist, home), _min_distance(dist, away)

    feats = pd.DataFrame(index=pd.RangeIndex(n_frames, name="video_frame"))
    feats["ball_x"], feats["ball_y"] = bx, by
    feats["ball_missing"] = np.isnan(bx).astype(float)
    step = np.hypot(np.diff(bx, prepend=np.nan), np.diff(by, prepend=np.nan))
    feats["ball_speed"] = step * FPS  # NaN unless the ball is seen on two consecutive frames
    feats["d_home"], feats["d_away"] = d_home, d_away
    feats["margin"] = d_away - d_home  # positive: the home side is closer to the ball
    for radius in (3, 5):
        near = dist < radius
        feats["home_within_{}m".format(radius)] = (near & home[None, :]).sum(axis=1)
        feats["away_within_{}m".format(radius)] = (near & away[None, :]).sum(axis=1)

    for name, mask in (("home", home), ("away", away)):
        cx, cy, sx, sy, mean_speed, n = _team_stats(px, py, speed, mask)
        feats[name + "_cx"], feats[name + "_cy"] = cx, cy
        feats[name + "_spread_x"], feats[name + "_spread_y"] = sx, sy
        feats[name + "_speed"], feats[name + "_tracked"] = mean_speed, n
    with np.errstate(all="ignore"):
        s_ok = ~np.isnan(speed)
        feats["speed_all"] = np.where(s_ok, speed, 0.0).sum(axis=1) / np.where(s_ok.sum(axis=1) > 0, s_ok.sum(axis=1), np.nan)

    for column in ("margin", "speed_all", "ball_missing"):
        for window in ROLL_WINDOWS:
            feats["{}_roll{}".format(column, window)] = (
                feats[column].rolling(window, center=True, min_periods=1).mean())
    return feats.astype("float32")


def load_features(game=GAME, rebuild=False):
    """The feature table, from the cache if it is newer than the tracking file."""
    raw_path = require(["atd_raw"], game)[0]
    fresh = (os.path.isfile(CACHE_PATH) and not rebuild
             and os.path.getmtime(CACHE_PATH) >= os.path.getmtime(raw_path))
    if fresh:
        return pd.read_pickle(CACHE_PATH)
    print("building features from {} (about a minute, cached afterwards) ...".format(
        os.path.basename(raw_path)), flush=True)
    feats = build_features(game)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    feats.to_pickle(CACHE_PATH)
    return feats
