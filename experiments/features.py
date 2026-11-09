"""Per-frame features, built from tracking positions alone, shared by every data source.

`compute_features` takes plain position arrays, so the Barcelona match (this repo's ATD) and any
external match produce *identical* columns. That is what lets a model trained on one source be
applied to another.

    from experiments.features import load_features
    feats = load_features()            # the Barcelona match, one row per video frame

Conventions
-----------
* Metres, origin at the pitch centre, 25 fps.
* Positions are expressed from the HOME team's point of view: the home side always attacks towards
  +x. That removes the "which way are they kicking" difference between sources and between halves.
  The direction is read from where the two teams stand at each kickoff (see attack_direction).
* Speeds are recomputed from positions (central difference over +/-2 frames), never taken from the
  file, so the two sources agree on what "speed" means.
* Missing observations stay NaN: the ball is missing on about 43% of the Barcelona frames. Some
  features forward-fill the last seen ball position (`*_ff`) and say how stale it is (`ball_gap`).
* Rolling features are centred, so they look at frames after the current one. That is fine here --
  the task is offline and only the output file is scored.
"""

import io
import os
import sys

import warnings

import numpy as np
import pandas as pd

# Columns are added one at a time on purpose (readability); the fragmentation warning is noise here.
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAME, GAMES, require, DATA_DIR, half_video_blocks  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

N_SLOTS = 46
FPS = 25
CACHE_VERSION = 5
CACHE_PATH = os.path.join(DATA_DIR, "experiments_features_v{}.pkl".format(CACHE_VERSION))
ROLL_WINDOWS = (25, 50, 75)   # 1 s, 2 s, 3 s
MAX_SPEED = 12.0              # m/s; above this a "speed" is a track swap, not a player
FF_LIMIT = 75                 # frames a seen ball position is carried forward (3 s)
CONTACT_M = 1.5               # a player this close to the ball is treated as having touched it
PITCH_L, PITCH_W = 105.0, 68.0

# Feature groups that improved leave-one-match-out scores on the external matches but made the
# Barcelona scores WORSE (see FINDINGS.md), so models leave them out unless extras=True.
DYNAMICS_PREFIX = "dyn_"      # ball-independent team-movement features; opt-in via dynamics=True
# The deepest-player series separate Barcelona from every external match almost perfectly (AUC 0.96 on their
# own, against 0.60 for one external match vs the rest): a tracking artefact, not football. Never used.
DYNAMICS_EXCLUDE = ["dyn_home_back", "dyn_away_back"]
LONG_CONTEXT_PREFIX = "long_"   # 6 s and 10 s windows; opt-in via long_context=True
EXPERIMENTAL = ["last_contact", "next_contact", "since_contact", "until_contact", "contact_bal",
                "home_contact", "away_contact", "ball_line"]
GAP_CAP = 250                 # frames; cap for the ball_gap features (10 s)
SECOND_HALF_KICKOFF = 113742  # DATA.md / FAQ.md
FIRST_HALF_KICKOFF = 11185


# ----------------------------------------------------------------------------- core

def player_speeds(px, py):
    """m/s per player slot from a +/-2 frame central difference; NaN where undefined or absurd."""
    vx = np.full(px.shape, np.nan)
    vy = np.full(px.shape, np.nan)
    vx[2:-2] = px[4:] - px[:-4]
    vy[2:-2] = py[4:] - py[:-4]
    v = np.hypot(vx, vy) * FPS / 4.0
    v[v > MAX_SPEED] = np.nan
    return v


def attack_direction(px, home, away, halves):
    """Per-frame +1/-1: +1 where the home side attacks towards +x.

    halves = [(first_frame, last_frame, kickoff_frame)]. At a kickoff the home side stands on its
    own half, so if its centroid is at lower x than the away side's it attacks towards +x.
    """
    n = len(px)
    direction = np.zeros(n, dtype=np.float32)
    for first, last, kick in halves:
        window = slice(max(kick - 5, 0), kick + 6)
        with np.errstate(all="ignore"):
            home_cx = np.nanmean(px[window][:, home])
            away_cx = np.nanmean(px[window][:, away])
        direction[first:last + 1] = 1.0 if home_cx < away_cx else -1.0
    # frames outside any half (warm-up, halftime) take the nearest half's direction
    filled = np.where(direction == 0, np.nan, direction)
    return pd.Series(filled).ffill().bfill().to_numpy(dtype=np.float32)


def _since_seen(seen):
    """Frames since the ball was last seen (0 where seen), capped; and until it is next seen."""
    idx = np.arange(len(seen), dtype=float)
    last = pd.Series(np.where(seen, idx, np.nan)).ffill().to_numpy()
    nxt = pd.Series(np.where(seen, idx, np.nan)).bfill().to_numpy()
    back = np.where(np.isnan(last), GAP_CAP, np.minimum(idx - last, GAP_CAP))
    fwd = np.where(np.isnan(nxt), GAP_CAP, np.minimum(nxt - idx, GAP_CAP))
    return back, fwd


def _min_distance(dist, mask):
    """Row-wise minimum over the slots in mask; NaN where no slot has a distance."""
    masked = np.where(mask[None, :] & ~np.isnan(dist), dist, np.inf)
    out = masked.min(axis=1)
    out[np.isinf(out)] = np.nan
    return out


def _team_stats(x, y, v, mask):
    """Centroid, spread and mean speed of one team over the slots in mask, ignoring NaN."""
    valid = mask[None, :] & ~np.isnan(x)
    n = valid.sum(axis=1).astype(float)
    safe = np.where(n > 0, n, np.nan)
    xs, ys = np.where(valid, x, 0.0), np.where(valid, y, 0.0)
    cx, cy = xs.sum(axis=1) / safe, ys.sum(axis=1) / safe
    sx = np.sqrt(np.maximum((xs ** 2).sum(axis=1) / safe - cx ** 2, 0.0))
    sy = np.sqrt(np.maximum((ys ** 2).sum(axis=1) / safe - cy ** 2, 0.0))
    vv = mask[None, :] & ~np.isnan(v)
    vn = vv.sum(axis=1).astype(float)
    speed = np.where(vv, v, 0.0).sum(axis=1) / np.where(vn > 0, vn, np.nan)
    return cx, cy, sx, sy, speed, n


def compute_features(px, py, bx, by, home, away, direction):
    """The feature table for one match.

    px, py     (n, slots) player positions in metres, centred, NaN where unseen
    bx, by     (n,) ball position in metres, NaN where unseen
    home, away (slots,) boolean masks
    direction  (n,) +1/-1, +1 where the home side attacks towards +x
    """
    n = len(px)
    with np.errstate(all="ignore"):
        d = direction[:, None]
        x, y = px * d, py                      # home-attacks-right frame
        bxr, byr = bx * direction, by
        v = player_speeds(px, py)
        dist = np.hypot(px - bx[:, None], py - by[:, None])

        feats = pd.DataFrame(index=pd.RangeIndex(n, name="video_frame"))
        seen = ~np.isnan(bx)
        feats["ball_missing"] = (~seen).astype(float)
        feats["ball_x"] = bxr                   # + = towards the away goal
        feats["ball_abs_y"] = np.abs(byr)       # distance from the centre line, i.e. towards a touchline
        step = np.hypot(np.diff(bx, prepend=np.nan), np.diff(by, prepend=np.nan))
        feats["ball_speed"] = np.where(step * FPS > 45, np.nan, step * FPS)
        back, fwd = _since_seen(seen)
        feats["ball_gap"], feats["ball_next_gap"] = back, fwd

        d_home, d_away = _min_distance(dist, home), _min_distance(dist, away)
        feats["d_home"], feats["d_away"] = d_home, d_away
        feats["margin"] = d_away - d_home       # + = the home side is closer to the ball
        feats["d_min"] = np.fmin(d_home, d_away)
        for radius in (3, 5, 10):
            near = dist < radius
            feats["home_within_{}m".format(radius)] = (near & home[None, :]).sum(axis=1)
            feats["away_within_{}m".format(radius)] = (near & away[None, :]).sum(axis=1)

        # the same distances to the last seen ball position, for frames where the ball is missing
        bxf = pd.Series(bx).ffill(limit=FF_LIMIT).to_numpy()
        byf = pd.Series(by).ffill(limit=FF_LIMIT).to_numpy()
        dist_ff = np.hypot(px - bxf[:, None], py - byf[:, None])
        dh_ff, da_ff = _min_distance(dist_ff, home), _min_distance(dist_ff, away)
        feats["margin_ff"] = da_ff - dh_ff

        # ---- who touched the ball last / next: ownership between touches
        home_c = (d_home < CONTACT_M) & (np.isnan(d_away) | (d_home <= d_away))
        away_c = (d_away < CONTACT_M) & (np.isnan(d_home) | (d_away < d_home))
        contact = np.where(home_c, 1.0, np.where(away_c, -1.0, np.nan))
        touched = ~np.isnan(contact)
        idx = np.arange(n, dtype=float)
        prev_i = pd.Series(np.where(touched, idx, np.nan)).ffill().to_numpy()
        next_i = pd.Series(np.where(touched, idx, np.nan)).bfill().to_numpy()
        prev_team = pd.Series(contact).ffill().to_numpy()
        next_team = pd.Series(contact).bfill().to_numpy()
        feats["last_contact"], feats["next_contact"] = prev_team, next_team   # +1 home, -1 away, NaN none yet
        feats["since_contact"] = np.where(np.isnan(prev_i), GAP_CAP, np.minimum(idx - prev_i, GAP_CAP))
        feats["until_contact"] = np.where(np.isnan(next_i), GAP_CAP, np.minimum(next_i - idx, GAP_CAP))
        signed = np.where(touched, contact, 0.0)
        for window in (25, 75):
            feats["contact_bal_roll{}".format(window)] = (
                pd.Series(signed).rolling(window, center=True, min_periods=1).mean().to_numpy())
        feats["home_contact_roll50"] = pd.Series((contact == 1).astype(float)).rolling(50, center=True, min_periods=1).mean().to_numpy()
        feats["away_contact_roll50"] = pd.Series((contact == -1).astype(float)).rolling(50, center=True, min_periods=1).mean().to_numpy()

        # ---- is the ball at, or past, a line? (throw-ins, goal kicks, corners all start there)
        margin_line = np.minimum(PITCH_L / 2 - np.abs(bx), PITCH_W / 2 - np.abs(by))   # < 0 means outside
        feats["ball_line_margin"] = margin_line
        for window in (25, 75):
            feats["ball_line_min_roll{}".format(window)] = (
                pd.Series(margin_line).rolling(window, center=True, min_periods=1).min().to_numpy())
        feats["ball_line_ff"] = pd.Series(margin_line).ffill(limit=FF_LIMIT).to_numpy()

        for name, mask in (("home", home), ("away", away)):
            cx, cy, sx, sy, speed, tracked = _team_stats(x, y, v, mask)
            feats[name + "_cx"], feats[name + "_cy"] = cx, np.abs(cy)
            feats[name + "_spread_x"], feats[name + "_spread_y"] = sx, sy
            feats[name + "_speed"], feats[name + "_tracked"] = speed, tracked
        feats["cx_diff"] = feats["home_cx"] - feats["away_cx"]
        ok = ~np.isnan(v)
        feats["speed_all"] = np.where(ok, v, 0.0).sum(axis=1) / np.where(ok.sum(axis=1) > 0, ok.sum(axis=1), np.nan)
        feats["frac_moving"] = np.where(ok, v > 1.5, False).sum(axis=1) / np.where(ok.sum(axis=1) > 0, ok.sum(axis=1), np.nan)

        for column in ("margin", "margin_ff", "speed_all", "ball_missing", "ball_speed", "d_min"):
            for window in ROLL_WINDOWS:
                feats["{}_roll{}".format(column, window)] = (
                    feats[column].rolling(window, center=True, min_periods=1).mean())
        # ---- team dynamics, independent of the ball: after a change of possession the side that wins the
        # ball pushes up and the other drops back. Only CHANGES over the previous / next 2 s and 6 s are used
        # (not levels), because levels such as team width differ between teams and sources.
        def _edge(mask, pick):
            vals = np.where(mask[None, :] & ~np.isnan(x), x, np.nan if pick == "max" else np.nan)
            filled = np.where(np.isnan(vals), -np.inf if pick == "max" else np.inf, vals)
            out = filled.max(axis=1) if pick == "max" else filled.min(axis=1)
            out[np.isinf(out)] = np.nan
            return out

        series = {
            "home_cx": feats["home_cx"].to_numpy(), "away_cx": feats["away_cx"].to_numpy(),
            "cx_diff": feats["cx_diff"].to_numpy(),
            "home_front": _edge(home, "max"), "home_back": _edge(home, "min"),
            "away_front": _edge(away, "min"), "away_back": _edge(away, "max"),
            "home_spread_x": feats["home_spread_x"].to_numpy(), "away_spread_x": feats["away_spread_x"].to_numpy(),
            "home_speed": feats["home_speed"].to_numpy(), "away_speed": feats["away_speed"].to_numpy(),
        }
        for name, values in series.items():
            smooth = pd.Series(values).rolling(25, center=True, min_periods=1).mean()
            for window in (50, 150):
                feats["dyn_{}_fwd{}".format(name, window)] = (smooth.shift(-window) - smooth).to_numpy()
                feats["dyn_{}_back{}".format(name, window)] = (smooth - smooth.shift(window)).to_numpy()

        # ---- long-range context: what has been going on over the last/next 3 s on either side
        for column in ("margin", "margin_ff", "d_min", "speed_all", "ball_missing", "contact_bal_roll75"):
            for window in (150, 250):
                feats["long_{}_{}".format(column, window)] = (
                    feats[column].rolling(window, center=True, min_periods=1).mean())
    return feats.astype("float32")


# ----------------------------------------------------------------------------- Barcelona match

def _read_tracking(raw_path):
    """The raw ATD file as a float array: frame_id, 46 x (x, y, speed), ball (x, y, third value)."""
    with open(raw_path, "r") as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, na_values=["NaN"], dtype="float64")
    expected = 1 + N_SLOTS * 3 + 3
    if table.shape[1] != expected:
        raise ValueError("expected {} columns per tracking line, got {}".format(expected, table.shape[1]))
    return table.to_numpy()


def barca_arrays(game=GAME):
    """(px, py, bx, by, home, away) in centred metres for the Barcelona match."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    length, width = GAMES[game]["pitch"]
    n_frames = GAMES[game]["n_video_frames"]
    arr = _read_tracking(raw_path)
    if len(arr) != n_frames or not np.array_equal(arr[:, 0], np.arange(1, n_frames + 1)):
        raise ValueError("tracking file is not frames 1..{} in order".format(n_frames))
    players = arr[:, 1:1 + N_SLOTS * 3].reshape(n_frames, N_SLOTS, 3)
    px, py = (players[:, :, 0] - 0.5) * length, (players[:, :, 1] - 0.5) * width
    bx, by = (arr[:, 1 + N_SLOTS * 3] - 0.5) * length, (arr[:, 2 + N_SLOTS * 3] - 0.5) * width
    team = team_of_each_track(meta_path)
    return px, py, bx, by, team == 0, team == 1


def build_features(game=GAME):
    px, py, bx, by, home, away = barca_arrays(game)
    blocks = half_video_blocks(game)  # [(1, first, last), (2, first, last)]
    halves = [(blocks[0][1], blocks[0][2], FIRST_HALF_KICKOFF),
              (blocks[1][1], blocks[1][2], SECOND_HALF_KICKOFF)]
    direction = attack_direction(px, home, away, halves)
    return compute_features(px, py, bx, by, home, away, direction)


def load_features(game=GAME, rebuild=False):
    """The Barcelona feature table, from the cache if it is newer than the tracking file."""
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
