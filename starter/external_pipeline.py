"""The competition-compliant pipeline: train and tune on external data only.

Rules (organisers' clarification): if you train or retrain you must use only external data; every parameter that
needs tuning must be tuned on external data, never on the ground-truth possession data or the provided video.
So nothing in this module reads the public ground truth. Training and tuning use the seven IDSSE matches
(Sportec/DFL, CC-BY 4.0) with leave-one-match-out validation; the provided match contributes only its *tracking*
as model input (and, for the ball-unseen pattern, the tracking's own gaps).

    sets = load_idsse_sets(ball_unseen_pattern())
    posts = lomo_posteriors("lightgbm", sets)            # each match predicted by a model that never saw it
    params, table = tune_decoder(posts, sets)            # decoder settings, scored on external matches only
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import external as ext  # noqa: E402
from benchmark import Hierarchical, _full_proba  # noqa: E402
from features import (FPS, LABEL_START, _rolling_mean, _run_length_since_change, build_extra,  # noqa: E402
                      build_features, load_tracking)
from games import GAME, GAMES  # noqa: E402
from tactical import controlled_possession, decode_pipeline  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

STEP = 6  # train on every 6th frame (25 fps frames are near-duplicates)

# This match's tracking has no ball speed (the column is always empty), so ball_speed / ball_accel are
# constant 0 here but informative in IDSSE. Training on them would teach a signal that can never be seen at
# prediction time, so they are dropped from every feature matrix.
UNOBSERVED = ("ball_speed", "ball_accel")

GRID = [dict(w=w, min_s=m, contact_m=c, inflight_thr=None)
        for w, m, c in itertools.product([1, 25, 50, 100], [0, 1, 2, 3], [None, 1.1])
        if not (m == 0 and c)]  # the contact filter only matters together with a duration filter


ADVANCED = [
    "ball_rel_home_cx", "ball_rel_home_cy", "ball_rel_away_cx", "ball_rel_away_cy",
    "team_area_diff", "compactness_diff",
    "d_margin_roll_13", "d_margin_roll_100", "d_margin_velocity", "d_min_roll_1s",
    "speed_all_roll_1s", "speed_all_roll_4s", "missing_roll_13", "missing_roll_4s",
    "dwell_home_1m", "dwell_away_1m", "dwell_home_2m", "dwell_away_2m",
    "pressure_diff", "stoppage_slow_energy",
]


def advanced_block(X22, E):
    """Teammates' multi-scale / dwell / shape ideas (starter/advanced_features.py on their branch), rebuilt from
    the columns we already have so it needs no extra tracking pass. Nothing here reads labels."""
    length, width = GAMES[GAME]["pitch"]
    bx, by = E.ball_x.values * length, E.ball_y.values * width
    dm, dmin = X22.d_margin.values, E.d_min.values
    d_home = np.where(dm >= 0, dmin, dmin - dm)  # d_margin = d_away - d_home, d_min = their minimum
    d_away = np.where(dm >= 0, dmin + dm, dmin)

    def dwell(flag):  # seconds the condition has held continuously, capped at 10 s
        return np.where(flag, np.minimum((_run_length_since_change(flag.astype(np.int8)) + 1) / FPS, 10.0), 0.0)

    a = {}
    a["ball_rel_home_cx"], a["ball_rel_home_cy"] = bx - X22.home_cx.values, by - X22.home_cy.values
    a["ball_rel_away_cx"], a["ball_rel_away_cy"] = bx - X22.away_cx.values, by - X22.away_cy.values
    area_h, area_a = X22.home_spread_x.values * X22.home_spread_y.values, X22.away_spread_x.values * X22.away_spread_y.values
    a["team_area_diff"] = area_h - area_a
    a["compactness_diff"] = (X22.home_spread_x.values + X22.home_spread_y.values) - (
        X22.away_spread_x.values + X22.away_spread_y.values)
    a["d_margin_roll_13"], a["d_margin_roll_100"] = _rolling_mean(dm, 13), _rolling_mean(dm, 100)
    a["d_margin_velocity"] = np.diff(dm, prepend=dm[:1]) * FPS
    a["d_min_roll_1s"] = _rolling_mean(dmin, FPS)
    spd, miss = X22.overall_speed.values, X22.ball_missing.values
    a["speed_all_roll_1s"], a["speed_all_roll_4s"] = _rolling_mean(spd, FPS), _rolling_mean(spd, 4 * FPS)
    a["missing_roll_13"], a["missing_roll_4s"] = _rolling_mean(miss, 13), _rolling_mean(miss, 4 * FPS)
    for name, d, r in (("home_1m", d_home, 1.0), ("away_1m", d_away, 1.0), ("home_2m", d_home, 2.0), ("away_2m", d_away, 2.0)):
        a["dwell_" + name] = dwell(d < r)
    a["pressure_diff"] = X22.home_within_3m.values - X22.away_within_3m.values
    a["stoppage_slow_energy"] = a["speed_all_roll_1s"] * (1.0 - a["missing_roll_4s"])
    return np.column_stack([a[k] for k in ADVANCED]).astype(np.float32)


BOUNDARY = [
    "ball_spd_pos", "ball_spd_chg", "ball_dir_chg", "d_home_closing", "d_away_closing",
    "d3_home", "d3_away", "d3_diff", "nearest_switches_2s", "opp_pressure_4m",
    "d_margin_lag1s", "d_margin_lead1s", "d_margin_lag3s", "d_margin_lead3s", "d_min_lag1s", "d_min_lead1s",
]


def _shift(a, k):
    """a[t - k] with the array's edge value repeated (k > 0 looks back, k < 0 looks ahead)."""
    out = np.empty_like(a)
    if k > 0:
        out[k:], out[:k] = a[:-k], a[0]
    elif k < 0:
        out[:k], out[k:] = a[-k:], a[-1]
    else:
        out[:] = a
    return out


def boundary_block(trk, is_home, X22, E):
    """Features for *when control changes hands*: ball velocity / direction change rebuilt from positions (this
    match has no tracked ball speed), how fast each team is closing on the ball, the 3-nearest-player picture,
    nearest-player churn, local pressure, and the margin 1 s / 3 s before and after (a cheap look at context).
    Nothing here reads labels."""
    length, width = GAMES[GAME]["pitch"]
    P, B = trk["players"], trk["ball"]
    n = len(B)
    px, py = P[:, :, 0] * length, P[:, :, 1] * width
    bx, by = B[:, 0] * length, B[:, 1] * width
    missing = np.isnan(bx)
    bxf = pd.Series(bx).ffill().bfill().fillna(length / 2).values
    byf = pd.Series(by).ffill().bfill().fillna(width / 2).values
    a = {}
    # ball kinematics from positions (zero while the ball is unseen)
    vx = (_shift(bxf, -3) - _shift(bxf, 3)) / 6 * FPS
    vy = (_shift(byf, -3) - _shift(byf, 3)) / 6 * FPS
    spd = np.where(missing, 0.0, np.minimum(np.hypot(vx, vy), 40.0))
    a["ball_spd_pos"] = spd
    a["ball_spd_chg"] = np.abs(_shift(spd, -12) - _shift(spd, 12))
    pvx, pvy = bxf - _shift(bxf, 12), byf - _shift(byf, 12)
    fvx, fvy = _shift(bxf, -12) - bxf, _shift(byf, -12) - byf
    den = np.hypot(pvx, pvy) * np.hypot(fvx, fvy)
    cos = np.where(den > 1e-6, (pvx * fvx + pvy * fvy) / np.maximum(den, 1e-6), 1.0)
    a["ball_dir_chg"] = np.where(missing, 0.0, np.arccos(np.clip(cos, -1, 1)))
    # nearest players of each team
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    inf = np.where(np.isnan(dist), np.inf, dist)
    dh, da = np.where(is_home[None, :], inf, np.inf), np.where(~is_home[None, :], inf, np.inf)
    d_home, d_away = np.minimum(dh.min(1), 30.0), np.minimum(da.min(1), 30.0)
    a["d_home_closing"], a["d_away_closing"] = d_home - _shift(d_home, 12), d_away - _shift(d_away, 12)
    k3 = lambda d: np.minimum(np.sort(d, 1)[:, :3], 30.0).mean(1)
    a["d3_home"], a["d3_away"] = k3(dh), k3(da)
    a["d3_diff"] = a["d3_away"] - a["d3_home"]
    near = np.where(missing, -1, inf.argmin(1))
    changed = (near != _shift(near, 1)) & (near >= 0) & (_shift(near, 1) >= 0)
    c = np.concatenate(([0], np.cumsum(changed)))
    idx = np.arange(n)
    a["nearest_switches_2s"] = c[np.minimum(idx + 26, n)] - c[np.maximum(idx - 25, 0)]
    safe = np.maximum(near, 0)
    nx, ny = px[idx, safe], py[idx, safe]
    d_np = np.hypot(px - nx[:, None], py - ny[:, None])
    opp = np.where(is_home[safe][:, None], ~is_home[None, :], is_home[None, :])
    with np.errstate(invalid="ignore"):
        a["opp_pressure_4m"] = np.where(near >= 0, np.nansum((d_np < 4.0) & opp, 1), 0)
    dm, dmin = X22.d_margin.values, E.d_min.values
    for name, arr, k in (("d_margin_lag1s", dm, FPS), ("d_margin_lead1s", dm, -FPS),
                         ("d_margin_lag3s", dm, 3 * FPS), ("d_margin_lead3s", dm, -3 * FPS),
                         ("d_min_lag1s", dmin, FPS), ("d_min_lead1s", dmin, -FPS)):
        a[name] = _shift(np.asarray(arr), k)
    return np.nan_to_num(np.column_stack([a[k] for k in BOUNDARY]).astype(np.float32))


MOTION = [
    "ball_vx", "ball_vy", "rel_speed_nearest", "carry_cos", "nearest_toward_ball", "nearest_speed",
    "d_min_rate", "d_min_min_1s", "d_min_std_1s", "home_vx", "away_vx", "vx_diff",
]


def motion_block(trk, is_home, X22, E):
    """Motion cues for control: player velocity *directions* (tracking gives only scalar speeds, so they are rebuilt
    from positions), whether the ball moves with the nearest player, whether that player runs toward the ball, and
    each team's forward momentum. Nothing here reads labels."""
    length, width = GAMES[GAME]["pitch"]
    P, B = trk["players"], trk["ball"]
    n = len(B)
    px, py = P[:, :, 0] * length, P[:, :, 1] * width
    bx, by = B[:, 0] * length, B[:, 1] * width
    missing = np.isnan(bx)
    bxf = pd.Series(bx).ffill().bfill().fillna(length / 2).values
    byf = pd.Series(by).ffill().bfill().fillna(width / 2).values
    vbx = (_shift(bxf, -3) - _shift(bxf, 3)) / 6 * FPS
    vby = (_shift(byf, -3) - _shift(byf, 3)) / 6 * FPS
    vbx, vby = np.where(missing, 0.0, vbx), np.where(missing, 0.0, vby)
    pvx = (_shift(px, -3) - _shift(px, 3)) / 6 * FPS
    pvy = (_shift(py, -3) - _shift(py, 3)) / 6 * FPS
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    near = np.where(missing, -1, np.where(np.isnan(dist), np.inf, dist).argmin(1))
    idx, safe = np.arange(n), np.maximum(near, 0)
    ok = near >= 0
    nvx = np.where(ok, np.nan_to_num(pvx[idx, safe]), 0.0)
    nvy = np.where(ok, np.nan_to_num(pvy[idx, safe]), 0.0)
    nx, ny = px[idx, safe], py[idx, safe]
    a = {"ball_vx": vbx, "ball_vy": vby}
    a["rel_speed_nearest"] = np.where(ok, np.hypot(vbx - nvx, vby - nvy), 0.0)
    sb, sp = np.hypot(vbx, vby), np.hypot(nvx, nvy)
    a["carry_cos"] = np.where((sb > 0.5) & (sp > 0.5), (vbx * nvx + vby * nvy) / np.maximum(sb * sp, 1e-6), 0.0)
    dx, dy = bxf - nx, byf - ny
    dn = np.maximum(np.hypot(dx, dy), 1e-6)
    a["nearest_toward_ball"] = np.where(ok, (nvx * dx + nvy * dy) / dn, 0.0)
    a["nearest_speed"] = sp
    dmin = E.d_min.values
    a["d_min_rate"] = (_shift(dmin, -6) - _shift(dmin, 6)) / 12 * FPS
    ser = pd.Series(dmin)
    a["d_min_min_1s"] = ser.rolling(FPS, center=True, min_periods=1).min().values
    a["d_min_std_1s"] = ser.rolling(FPS, center=True, min_periods=1).std().fillna(0).values
    with np.errstate(all="ignore"):
        hv, av = np.nanmean(pvx[:, is_home], 1), np.nanmean(pvx[:, ~is_home], 1)
    a["home_vx"], a["away_vx"] = np.nan_to_num(hv), np.nan_to_num(av)
    a["vx_diff"] = a["home_vx"] - a["away_vx"]
    return np.nan_to_num(np.column_stack([a[k] for k in MOTION]).astype(np.float32))


TEMPORAL = [
    "nhf_past_1s", "nhf_past_3s", "nhf_past_5s", "nhf_fut_1s", "nhf_fut_3s", "nhf_fut_5s", "nhf_c10s",
    "team_flips_3s", "team_flips_10s",
    "since_contact_home", "since_contact_away", "until_contact_home", "until_contact_away",
    "ball_path_1s", "ball_path_3s", "ball_straight_3s",
    "still_frac_2s", "still_frac_5s", "speed_trend_3s",
    "d_margin_lag05s", "d_margin_lead05s", "d_margin_lag2s", "d_margin_lead2s", "d_margin_lag5s", "d_margin_lead5s",
    "d_min_lag3s", "d_min_lead3s",
    "ball_spd_lag1s", "ball_spd_lead1s", "ball_spd_lag3s", "ball_spd_lead3s",
]


def _window_mean(a, lo, hi):
    """NaN-aware mean of a[t+lo .. t+hi] (lo <= hi, in frames; negative = past), edges truncated."""
    n = len(a)
    ok = ~np.isnan(a)
    csum = np.concatenate(([0.0], np.cumsum(np.where(ok, a, 0.0))))
    ccnt = np.concatenate(([0], np.cumsum(ok)))
    idx = np.arange(n)
    a0, b0 = np.clip(idx + lo, 0, n), np.clip(idx + hi + 1, 0, n)
    cnt = ccnt[b0] - ccnt[a0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (csum[b0] - csum[a0]) / np.maximum(cnt, 1), np.nan)


def temporal_block(trk, is_home, X22, E):
    """Longer-range temporal context: how the nearest team, contact history, ball path and team stillness evolve over
    0.5 - 10 s before and after each frame (the match is processed offline, so looking ahead is allowed). Includes the
    share of recent / upcoming frames in which each team was nearest to the ball (a distance-based possession proxy).
    Nothing here reads labels."""
    length, width = GAMES[GAME]["pitch"]
    P, B = trk["players"], trk["ball"]
    n = len(B)
    idx = np.arange(n)
    px, py, ps = P[:, :, 0] * length, P[:, :, 1] * width, P[:, :, 2]
    bx, by = B[:, 0] * length, B[:, 1] * width
    missing = np.isnan(bx)
    bxf = pd.Series(bx).ffill().bfill().fillna(length / 2).values
    byf = pd.Series(by).ffill().bfill().fillna(width / 2).values
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    inf = np.where(np.isnan(dist), np.inf, dist)
    d_home = np.minimum(np.where(is_home[None, :], inf, np.inf).min(1), 30.0)
    d_away = np.minimum(np.where(~is_home[None, :], inf, np.inf).min(1), 30.0)
    dm, dmin = X22.d_margin.values, E.d_min.values
    a = {}
    # distance-based possession proxy: share of frames in a window in which the home side was nearest
    nh = np.where(missing, np.nan, (dm > 0).astype(float))
    for name, lo, hi in (("nhf_past_1s", -FPS, 0), ("nhf_past_3s", -3 * FPS, 0), ("nhf_past_5s", -5 * FPS, 0),
                         ("nhf_fut_1s", 0, FPS), ("nhf_fut_3s", 0, 3 * FPS), ("nhf_fut_5s", 0, 5 * FPS),
                         ("nhf_c10s", -5 * FPS, 5 * FPS)):
        a[name] = np.nan_to_num(_window_mean(nh, lo, hi), nan=0.5)
    held = pd.Series(nh).ffill().bfill().fillna(0.5).values
    flips = (np.abs(np.diff(held, prepend=held[:1])) > 0).astype(float)
    a["team_flips_3s"] = _window_mean(flips, -3 * FPS // 2, 3 * FPS // 2) * 3 * FPS
    a["team_flips_10s"] = _window_mean(flips, -5 * FPS, 5 * FPS) * 10 * FPS
    cap = 10 * FPS
    for team, d in (("home", d_home), ("away", d_away)):
        near = d < 1.5
        last = np.maximum.accumulate(np.where(near, idx, -10 ** 9))
        a["since_contact_" + team] = np.minimum(idx - last, cap) / FPS
        nxt = np.minimum.accumulate(np.where(near, idx, 10 ** 9)[::-1])[::-1]
        a["until_contact_" + team] = np.minimum(nxt - idx, cap) / FPS
    step = np.hypot(np.diff(bxf, prepend=bxf[:1]), np.diff(byf, prepend=byf[:1]))
    step = np.where(missing, 0.0, np.minimum(step, 3.0))
    a["ball_path_1s"] = _window_mean(step, -FPS // 2, FPS // 2) * FPS
    a["ball_path_3s"] = _window_mean(step, -3 * FPS // 2, 3 * FPS // 2) * 3 * FPS
    half = int(1.5 * FPS)
    net = np.hypot(_shift(bxf, -half) - _shift(bxf, half), _shift(byf, -half) - _shift(byf, half))
    a["ball_straight_3s"] = np.minimum(net / np.maximum(a["ball_path_3s"], 1e-3), 1.5)
    with np.errstate(all="ignore"):
        still = np.nanmean(np.where(np.isnan(ps), np.nan, (ps < 0.5).astype(float)), axis=1)
    still = np.nan_to_num(still, nan=0.0)
    a["still_frac_2s"], a["still_frac_5s"] = _window_mean(still, -FPS, FPS), _window_mean(still, -5 * FPS // 2, 5 * FPS // 2)
    spd = X22.overall_speed.values
    a["speed_trend_3s"] = _window_mean(spd, -FPS, 0) - _window_mean(_shift(spd, 2 * FPS), -FPS, 0)
    for name, k in (("d_margin_lag05s", FPS // 2), ("d_margin_lead05s", -(FPS // 2)), ("d_margin_lag2s", 2 * FPS),
                    ("d_margin_lead2s", -2 * FPS), ("d_margin_lag5s", 5 * FPS), ("d_margin_lead5s", -5 * FPS)):
        a[name] = _shift(dm, k)
    a["d_min_lag3s"], a["d_min_lead3s"] = _shift(dmin, 3 * FPS), _shift(dmin, -3 * FPS)
    bspd = np.where(missing, 0.0, np.minimum(step * FPS, 40.0))
    for name, k in (("ball_spd_lag1s", FPS), ("ball_spd_lead1s", -FPS), ("ball_spd_lag3s", 3 * FPS), ("ball_spd_lead3s", -3 * FPS)):
        a[name] = _shift(bspd, k)
    return np.nan_to_num(np.column_stack([a[k] for k in TEMPORAL]).astype(np.float32))


# Ball-independent TEAM-MOVEMENT features (temporal). After a change of possession the side that wins the ball pushes up
# and the other drops back, and that is visible even when the ball is not tracked. Only CHANGES over the previous / next
# 2 s and 6 s are used, never levels, because levels such as team width differ between teams and data sources. The
# deepest-player series are deliberately left out: on their own they separate the provided match from every external
# match (source-audit AUC 0.96 against 0.60 for one external match vs the rest), a tracking artefact.
DYNAMICS_SERIES = ("home_cx", "away_cx", "cx_diff", "home_front", "away_front",
                   "home_spread_x", "away_spread_x", "home_speed", "away_speed")
DYNAMICS = ["dyn_{}_{}{}".format(s, d, w) for s in DYNAMICS_SERIES for w in (50, 150) for d in ("fwd", "back")]
USE_DYNAMICS = True    # module default for featureset(); on the evidence of starter/ablation_integration.py


def _segments(n, period):
    """[(start, stop)] runs of constant period, so no window reaches across the halftime join."""
    if period is None:
        return [(0, n)]
    period = np.asarray(period)
    cuts = np.flatnonzero(np.diff(period)) + 1
    edges = np.r_[0, cuts, n]
    return list(zip(edges[:-1], edges[1:]))


def dynamics_block(trk, is_home, X22, E):
    """How each team's centroid, front line, width and speed move over the previous / next 2 s and 6 s."""
    length = GAMES[GAME]["pitch"][0]
    px = trk["players"][:, :, 0] * length
    n = len(px)

    def front(mask, pick):
        vals = px[:, mask]
        filled = np.where(np.isnan(vals), -np.inf if pick == "max" else np.inf, vals)
        out = filled.max(axis=1) if pick == "max" else filled.min(axis=1)
        out[np.isinf(out)] = np.nan
        return out

    series = {
        "home_cx": X22.home_cx.values, "away_cx": X22.away_cx.values, "cx_diff": X22.centroid_diff_x.values,
        "home_front": front(is_home, "max"), "away_front": front(~is_home, "min"),   # home attacks +x
        "home_spread_x": X22.home_spread_x.values, "away_spread_x": X22.away_spread_x.values,
        "home_speed": X22.mean_speed_home.values, "away_speed": X22.mean_speed_away.values,
    }
    cols = []
    for name in DYNAMICS_SERIES:
        values = np.asarray(series[name], dtype=float)
        out = {key: np.full(n, np.nan) for key in ("fwd50", "back50", "fwd150", "back150")}
        for a, b in _segments(n, trk.get("period")):
            smooth = pd.Series(values[a:b]).rolling(25, center=True, min_periods=1).mean()
            for w in (50, 150):
                out["fwd%d" % w][a:b] = (smooth.shift(-w) - smooth).to_numpy()
                out["back%d" % w][a:b] = (smooth - smooth.shift(w)).to_numpy()
        for w in (50, 150):
            cols += [out["fwd%d" % w], out["back%d" % w]]
    return np.nan_to_num(np.column_stack(cols).astype(np.float32))


# Features derived from how the BALL MOVES (computed from its positions). In the IDSSE training data the ball is frozen
# for most dead-ball frames, so these teach "ball not moving => dead"; in the provided tracking the ball keeps moving
# while dead, so the shortcut fails (the source audit in experiments/FINDINGS.md: the temporal ball-speed / ball-path
# features are the most source-specific group, AUC 0.62 each against 0.51-0.52 for one external match vs the rest).
# Everything here is judged on external data and tracking only. See starter/ablation_ball_motion.py.
BALL_MOTION = (
    "ball_spd_pos", "ball_spd_chg", "ball_dir_chg",                                  # boundary block
    "ball_vx", "ball_vy", "rel_speed_nearest", "carry_cos",                          # motion block
    "ball_path_1s", "ball_path_3s", "ball_straight_3s",                              # temporal block
    "ball_spd_lag1s", "ball_spd_lead1s", "ball_spd_lag3s", "ball_spd_lead3s",
)
DROP_BALL_MOTION = True    # module default for featureset(); on the evidence of starter/ablation_ball_motion.py and ablation_integration.py


def featureset(trk, is_home, advanced=False, boundary=False, motion=False, temporal=False, drop_ball_motion=None,
               dynamics=None):
    """(X, aux): 30 model features (+20 advanced ones if asked) and the per-frame arrays the decoder rules need.

    drop_ball_motion removes the BALL_MOTION columns (None = the module default DROP_BALL_MOTION); dynamics appends the
    ball-independent team-movement block (None = the module default USE_DYNAMICS)."""
    drop_ball_motion = DROP_BALL_MOTION if drop_ball_motion is None else drop_ball_motion
    dynamics = USE_DYNAMICS if dynamics is None else dynamics
    X22 = build_features(trk, is_home=is_home)
    E = build_extra(trk, X22)
    aux = dict(d_min=E.d_min.values, ball_missing=X22.ball_missing.values,
               speed_all_roll_2s=X22.speed_all_roll_2s.values, ball_accel=E.ball_accel.values)
    names = list(X22.columns) + list(E.columns)
    keep = [i for i, n in enumerate(names) if n not in UNOBSERVED]
    X = np.hstack([X22.values, E.values])[:, keep]
    out_names = [names[i] for i in keep]
    if advanced:
        X = np.nan_to_num(np.hstack([X, advanced_block(X22, E)]))
        out_names += ADVANCED
    if boundary:
        X = np.hstack([X, boundary_block(trk, is_home, X22, E)])
        out_names += BOUNDARY
    if motion:
        X = np.hstack([X, motion_block(trk, is_home, X22, E)])
        out_names += MOTION
    if temporal:
        X = np.hstack([X, temporal_block(trk, is_home, X22, E)])
        out_names += TEMPORAL
    if dynamics:
        X = np.hstack([X, dynamics_block(trk, is_home, X22, E)])
        out_names += DYNAMICS
    if drop_ball_motion:
        X = X[:, [i for i, n in enumerate(out_names) if n not in BALL_MOTION]]
    return X, aux


def ball_unseen_pattern(first_minutes=None):
    """Where this match's tracking has no ball, from the tracking alone (no labels): tiled onto IDSSE's
    always-present ball so the model sees the same kind of gaps at training and prediction time.

    By default the pattern is the two halves of the whole match (the ball is unseen ~41% of the time there,
    against ~18% in the first 15 minutes), because the submission covers the whole match. `first_minutes`
    restricts it to the labelled start of the match, which is what earlier versions used."""
    if first_minutes:
        n = first_minutes * 60 * FPS
        trk = load_tracking(LABEL_START + n)
        return np.isnan(trk["ball"][LABEL_START:LABEL_START + n, 0])
    n_frames = GAMES[GAME]["n_video_frames"]
    offsets, halves = GAMES[GAME]["offsets"], GAMES[GAME]["halves"]
    trk = load_tracking(n_frames)
    lost = np.isnan(trk["ball"][:, 0])
    return np.concatenate([lost[halves[h][0] - offsets[h]:halves[h][1] - offsets[h] + 1] for h in (1, 2)])


def mirror_tracking(trk, mode):
    """Symmetry augmentation, applied after the pitch is oriented so the home team attacks +x.
    "y": flip the pitch top to bottom. "xswap": flip it left to right and swap the roles of the two teams (so the
    home team still attacks +x), exchanging the H and A labels."""
    P, B = trk["players"].copy(), trk["ball"].copy()
    out = dict(trk)
    if mode == "y":
        P[:, :, 1], B[:, 1] = 1.0 - P[:, :, 1], 1.0 - B[:, 1]
    elif mode == "xswap":
        P[:, :, 0], B[:, 0] = 1.0 - P[:, :, 0], 1.0 - B[:, 0]
        out["is_home"] = ~trk["is_home"]
        y = trk["labels"]
        out["labels"] = np.where(y == 0, 1, np.where(y == 1, 0, y))
    else:
        raise ValueError(mode)
    out["players"], out["ball"] = P, B
    return out


def load_idsse_sets(pattern=None, matches=ext.IDSSE_MATCHES, advanced=False, control_s=None, boundary=False,
                   motion=False, mirror=None, augment=False, temporal=False, loader=None, drift=False, turnover=False):
    """{match: dict(X, aux, y, period, aug)}, oriented so the home team attacks +x, with `pattern` applied.

    `augment` also stores `aug`: (X, y) pairs of the same match mirrored top-to-bottom and left-to-right (with the
    teams swapped), which `fit` adds to the *training* matches only. `mirror` builds a single mirrored version.
    `drift` lets the ball keep moving while it is dead (see external.dead_ball_drift), as it does in the provided
    tracking, before `pattern` is applied. `turnover` additionally hides the ball around possession changes (see
    external.turnover_gaps), after `pattern`."""
    def build(trk, mode):
        if mode:
            trk = mirror_tracking(trk, mode)
        X, aux = featureset(trk, trk["is_home"], advanced, boundary, motion, temporal)
        y = trk["labels"]
        if control_s:  # relabel to "controlled possession": runs under control_s seconds join the previous team
            y = controlled_possession(y, int(control_s * FPS))
        return X, aux, y

    sets = {}
    for m in matches:
        trk = (loader or ext.load_idsse)(m)
        trk = ext.orient_home_attacks_right(trk, trk["is_home"], ext.period_segments(trk))
        if drift:
            trk = ext.dead_ball_drift(trk)
        if pattern is not None:
            trk = ext.apply_occlusion(trk, pattern)
        if turnover:
            trk = ext.turnover_gaps(trk)
        X, aux, y = build(trk, mirror)
        sets[m] = dict(X=X, aux=aux, y=y, period=trk["period"])
        if augment:
            sets[m]["aug"] = [build(trk, mode)[::2] for mode in ("y", "xswap")]  # (X, y) pairs
    return sets


def make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def make_lightgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=150, learning_rate=0.06, num_leaves=15, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=0, verbose=-1, deterministic=True, force_row_wise=True)


MODELS = {
    "logreg": make_logreg,
    "hierarchical": lambda: Hierarchical(make_logreg, make_logreg),
    "lightgbm": make_lightgbm,
}


def target_features(advanced=True, boundary=True, motion=True, temporal=True, step=6):
    """Features of the *provided* match (both halves, inputs only, no labels): the target domain for weighting."""
    from baseline import team_of_each_track
    from features import load_tracking
    from games import asset_paths
    n_frames = GAMES[GAME]["n_video_frames"]
    offsets, halves = GAMES[GAME]["offsets"], GAMES[GAME]["halves"]
    spans = [(halves[h][0] - offsets[h], halves[h][1] - offsets[h] + 1) for h in (1, 2)]
    trk = load_tracking(n_frames)
    is_home = team_of_each_track(asset_paths()["atd_meta"]) == 0
    trk = ext.orient_home_attacks_right(trk, is_home, spans)
    X, _ = featureset(trk, is_home, advanced, boundary, motion, temporal)
    return np.vstack([X[a:b][::step] for a, b in spans])


def domain_weights(sets, X_target, clip=(0.1, 10.0), step=6):
    """Importance weights that make the external matches look like the provided match, from inputs only.

    A small classifier learns to tell external frames from provided-match frames; a frame's weight is its odds of
    being "provided". Cross-fitted by match (a match's weights come from a classifier that never saw it), clipped,
    and normalised to mean 1 per match. No labels are involved."""
    from lightgbm import LGBMClassifier
    weights = {}
    for m in sets:
        src = np.vstack([sets[o]["X"][::step] for o in sets if o != m])
        X = np.vstack([src, X_target])
        d = np.concatenate([np.zeros(len(src)), np.ones(len(X_target))])
        clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=7, min_child_samples=300, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, class_weight="balanced", random_state=0,
                             verbose=-1, deterministic=True, force_row_wise=True).fit(X, d)
        p = np.clip(clf.predict_proba(sets[m]["X"])[:, 1], 1e-3, 1 - 1e-3)
        w = np.clip(p / (1 - p), *clip)
        weights[m] = w / w.mean()
    return weights


def fit(model_name, sets, matches=None, weights=None):
    """Fit `model_name` on the given external matches (default all)."""
    matches = list(sets) if matches is None else matches
    parts = [(sets[m]["X"], sets[m]["y"]) for m in matches] + [a for m in matches for a in sets[m].get("aug", [])]
    X = np.vstack([a[::STEP] for a, _ in parts])
    y = np.concatenate([b[::STEP] for _, b in parts])
    if weights is None:
        return MODELS[model_name]().fit(X, y)
    w = np.concatenate([weights[m][::STEP] for m in matches] + [weights[m][::STEP] for m in matches for _ in sets[m].get("aug", [])])
    return MODELS[model_name]().fit(X, y, sample_weight=w)


def lomo_posteriors(model_name, sets, weights=None):
    """Leave-one-match-out: {held_out_match: (n_frames, 3) posteriors from a model trained on the others}."""
    return {held: _full_proba(fit(model_name, sets, [m for m in sets if m != held], weights), sets[held]["X"])
            for held in sets}


def grid_scores(P, match, grid=GRID):
    """spellF1 on one external match for every decoder setting in `grid`."""
    return np.array([pm.evaluate(match["y"], decode_pipeline(P, match["aux"], **p))["spell_f1_mean"] for p in grid])


def tune_decoder(posts, sets, grid=GRID):
    """Decoder settings chosen on external matches only.

    Returns (joint_params, table): `joint_params` maximises the mean spellF1 over every held-out match (what the
    final submission uses); `table` has, per held-out match, the score of the settings tuned on the *other*
    matches (an honest estimate of how decoder settings transfer) and of a plain smoothing decoder.
    """
    scores = {m: grid_scores(posts[m], sets[m], grid) for m in sets}
    stack = np.vstack(list(scores.values()))
    joint = grid[int(stack.mean(0).argmax())]
    plain = next(i for i, p in enumerate(grid) if p["w"] == 25 and p["min_s"] == 0)
    table = []
    for m in sets:
        others = np.vstack([scores[o] for o in sets if o != m]).mean(0)
        best = int(others.argmax())
        table.append(dict(match=m, tuned_on_others=scores[m][best], plain_w25=scores[m][plain],
                          params=str({k: v for k, v in grid[best].items() if k != "inflight_thr"})))
    return joint, table
