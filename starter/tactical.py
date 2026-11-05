"""Football-physics rules that sit between posteriors and labels.

Reimplemented from the teammates' report (TACTICAL_ML_BALL_POSSESSION_REPORT.pdf) as separate,
individually switchable steps so each one can be ablated under the nested protocol in benchmark.py.
Nothing here is tuned: every threshold is a parameter of `decode_pipeline`, to be chosen by inner CV.

Codes follow the scorer: 0 = H, 1 = A, 2 = D.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmark import smooth_proba, viterbi  # noqa: E402

FPS = 25
H, A, D = 0, 1, 2


def _runs(codes):
    change = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(codes)]))
    return starts, stops


def duration_filter(codes, min_frames, d_min=None, contact_m=None):
    """Absorb runs shorter than `min_frames` into their longer neighbour, shortest first.

    With `d_min` and `contact_m` the filter is *contact-conditioned*: a short H/A run survives if a
    player actually got within `contact_m` metres of the ball during it. That is the teammates'
    explanation for real 0.6-1.4 s interceptions that a blunt filter erases.
    """
    codes = np.asarray(codes).copy()
    pinned = set()
    while True:
        starts, stops = _runs(codes)
        lengths = stops - starts
        cand = [i for i in range(len(lengths)) if lengths[i] < min_frames and (starts[i], stops[i]) not in pinned]
        if len(lengths) < 2 or not cand:
            return codes
        i = min(cand, key=lambda j: lengths[j])
        a, b = starts[i], stops[i]
        if d_min is not None and contact_m is not None and codes[a] != D and d_min[a:b].min() < contact_m:
            pinned.add((a, b))
            continue
        left = lengths[i - 1] if i > 0 else -1
        right = lengths[i + 1] if i + 1 < len(lengths) else -1
        codes[a:b] = codes[starts[i - 1]] if left >= right else codes[starts[i + 1]]


def controlled_possession(codes, min_frames):
    """Relabel provider data to the competition's notion of *controlled* possession (CHALLENGE.md):
    a touch, deflection or loose ball does not change possession, which "stays with the last team that
    had it". A live H/A run shorter than `min_frames` is therefore handed to the team that held the ball
    just before (or, at the start of a live stretch, to the team that gets it next). A short run fenced in
    by dead ball on both sides is a real, if brief, possession and is kept. Dead frames never change."""
    codes = np.asarray(codes).copy()
    skipped = set()
    while True:
        starts, stops = _runs(codes)
        lengths = stops - starts
        cand = [i for i in range(len(lengths))
                if codes[starts[i]] != D and lengths[i] < min_frames and i not in skipped]
        if not cand:
            return codes
        i = min(cand, key=lambda j: lengths[j])
        a, b = starts[i], stops[i]
        if i > 0 and codes[starts[i - 1]] != D:
            codes[a:b] = codes[starts[i - 1]]
        elif i + 1 < len(lengths) and codes[starts[i + 1]] != D:
            codes[a:b] = codes[starts[i + 1]]
        else:
            skipped.add(i)
            continue
        skipped = set()  # run indices shifted after the merge


def inflight_inertia(codes, ball_missing, speed_roll, speed_thr, max_frames):
    """A short 'dead' stretch while the ball is unseen and everyone is still running is a lofted
    pass, not a stoppage: hand it back to the team that had the ball just before."""
    codes = np.asarray(codes).copy()
    starts, stops = _runs(codes)
    for a, b in zip(starts, stops):
        if codes[a] != D or a == 0 or codes[a - 1] == D:
            continue
        if (b - a) <= max_frames and ball_missing[a:b].mean() > 0.7 and speed_roll[a:b].mean() > speed_thr:
            codes[a:b] = codes[a - 1]
    return codes


def snap_to_acceleration(codes, ball_accel, radius):
    """Move each transition to the ball-acceleration peak within +-`radius` frames (a kick or tackle
    is a physical impulse). Never moves a boundary past its neighbours."""
    codes = np.asarray(codes).copy()
    if radius <= 0:
        return codes
    starts, _ = _runs(codes)
    n = len(codes)
    for t in starts[1:]:
        lo, hi = max(1, t - radius), min(n - 1, t + radius)
        new = lo + int(np.argmax(ball_accel[lo:hi + 1]))
        if new > t:
            codes[t:new] = codes[t - 1]
        elif new < t:
            codes[new:t] = codes[t]
    return codes


def decode_pipeline(P, aux, w=50, viterbi_stay=None, inflight_thr=None, inflight_s=6.0,
                    min_s=0.0, contact_m=None, snap=0):
    """posteriors -> smoothing (or Viterbi) -> in-flight inertia -> duration/contact filter -> snapping.

    Every stage is off by default; switch one on by giving its parameter. `aux` holds per-frame arrays
    for the same frames as P: d_min, ball_missing, speed_all_roll_2s, ball_accel.
    """
    if viterbi_stay:
        codes = viterbi(smooth_proba(P, w) if w > 1 else P, p_stay=viterbi_stay, direct_switch=0.02)
    else:
        codes = smooth_proba(P, w).argmax(1)
    codes = codes.astype(np.int8)
    if inflight_thr is not None:
        codes = inflight_inertia(codes, aux["ball_missing"], aux["speed_all_roll_2s"],
                                 inflight_thr, int(inflight_s * FPS))
    if min_s > 0:
        codes = duration_filter(codes, int(min_s * FPS), aux["d_min"] if contact_m else None, contact_m)
    if snap:
        codes = snap_to_acceleration(codes, aux["ball_accel"], snap)
    return codes
