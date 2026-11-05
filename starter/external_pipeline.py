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


def featureset(trk, is_home, advanced=False, boundary=False, motion=False):
    """(X, aux): 30 model features (+20 advanced ones if asked) and the per-frame arrays the decoder rules need."""
    X22 = build_features(trk, is_home=is_home)
    E = build_extra(trk, X22)
    aux = dict(d_min=E.d_min.values, ball_missing=X22.ball_missing.values,
               speed_all_roll_2s=X22.speed_all_roll_2s.values, ball_accel=E.ball_accel.values)
    names = list(X22.columns) + list(E.columns)
    keep = [i for i, n in enumerate(names) if n not in UNOBSERVED]
    X = np.hstack([X22.values, E.values])[:, keep]
    if advanced:
        X = np.nan_to_num(np.hstack([X, advanced_block(X22, E)]))
    if boundary:
        X = np.hstack([X, boundary_block(trk, is_home, X22, E)])
    if motion:
        X = np.hstack([X, motion_block(trk, is_home, X22, E)])
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
                   motion=False, mirror=None, augment=False):
    """{match: dict(X, aux, y, period, aug)}, oriented so the home team attacks +x, with `pattern` applied.

    `augment` also stores `aug`: (X, y) pairs of the same match mirrored top-to-bottom and left-to-right (with the
    teams swapped), which `fit` adds to the *training* matches only. `mirror` builds a single mirrored version."""
    def build(trk, mode):
        if mode:
            trk = mirror_tracking(trk, mode)
        X, aux = featureset(trk, trk["is_home"], advanced, boundary, motion)
        y = trk["labels"]
        if control_s:  # relabel to "controlled possession": runs under control_s seconds join the previous team
            y = controlled_possession(y, int(control_s * FPS))
        return X, aux, y

    sets = {}
    for m in matches:
        trk = ext.load_idsse(m)
        trk = ext.orient_home_attacks_right(trk, trk["is_home"], ext.period_segments(trk))
        if pattern is not None:
            trk = ext.apply_occlusion(trk, pattern)
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


def target_features(advanced=True, boundary=True, motion=True, step=6):
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
    X, _ = featureset(trk, is_home, advanced, boundary, motion)
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
