"""Small building blocks that turn per-frame guesses into possessions.

Codes follow evaluation/possession_metrics.py: 0 = H, 1 = A, 2 = D, 3 = X. All functions take and
return arrays over the whole match.
"""

import numpy as np

H, A, D, X = 0, 1, 2, 3


def vote_smooth(codes, window):
    """Centred majority vote over +/- window frames, H against A.

    D where nobody in the window voted for a team. Ties go to home. Same rule as the repository
    baseline, written so a window of 0 returns the input unchanged.
    """
    codes = np.asarray(codes)
    if window <= 0:
        return codes.copy()
    n = len(codes)
    home = np.concatenate(([0], np.cumsum(codes == H)))
    away = np.concatenate(([0], np.cumsum(codes == A)))
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - window), np.minimum(n, idx + window + 1)
    h, a = home[hi] - home[lo], away[hi] - away[lo]
    out = np.full(n, D, dtype=np.int8)
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], H, A)
    return out


def smooth_probs(probs, window):
    """Centred moving average of an (n, k) probability array, then argmax. Returns codes 0..k-1."""
    probs = np.asarray(probs, dtype=float)
    if window <= 0:
        return probs.argmax(axis=1).astype(np.int8)
    n = len(probs)
    csum = np.vstack([np.zeros((1, probs.shape[1])), np.cumsum(probs, axis=0)])
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - window), np.minimum(n, idx + window + 1)
    return ((csum[hi] - csum[lo]) / (hi - lo)[:, None]).argmax(axis=1).astype(np.int8)


def viterbi(log_probs, switch_team, switch_dead, temperature=1.0, switch_restart=None):
    """Most likely H/A/D path under per-frame log-probabilities and a cost for changing state.

    log_probs      (n, 3) log P(state | features) for H, A, D
    switch_team    cost of H <-> A (a change of possession)
    switch_dead    cost of live -> D (the ball going dead)
    switch_restart cost of D -> live (play restarting); defaults to switch_dead
    temperature    divides the emission term; > 1 trusts the frame classifier less, which is the
                   same as making the switch costs bigger. Frames are heavily autocorrelated, so
                   the raw probabilities are overconfident and the costs need to be sized to match.

    This is a Viterbi pass over a 3-state chain. It replaces "average the probabilities over a
    window" with "pay a fixed price per change", so it removes flicker without blurring genuine
    handovers by half a window.
    """
    emit = np.asarray(log_probs, dtype=float) / temperature
    n = len(emit)
    restart = switch_dead if switch_restart is None else switch_restart
    cost = np.array([[0.0, switch_team, switch_dead],      # cost[i, j]: from state i to state j
                     [switch_team, 0.0, switch_dead],
                     [restart, restart, 0.0]])
    score = emit[0].copy()
    back = np.empty((n, 3), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] - cost          # cand[i, j]: be in i at t-1, move to j at t
        best = cand.argmax(axis=0)
        back[t] = best
        score = cand[best, np.arange(3)] + emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def run_lengths(codes):
    """[(start, stop, code)] constant runs, stop exclusive."""
    codes = np.asarray(codes)
    if not len(codes):
        return []
    change = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(codes)]))
    return [(int(s), int(e), int(codes[s])) for s, e in zip(starts, stops)]


def min_duration_filter(codes, min_frames):
    """Merge runs shorter than min_frames into a neighbour, shortest first.

    A short run is given to the longer of its two neighbours (the only one at the edges). This is
    the blunt version of 'a possession cannot last 0.2 s'. It also removes real quick turnovers,
    so try it with and without and look at the possession count it leaves you.
    """
    runs = [list(r) for r in run_lengths(codes)]
    if min_frames <= 1:
        return np.asarray(codes).copy()
    while len(runs) > 1:
        lengths = [r[1] - r[0] for r in runs]
        i = int(np.argmin(lengths))
        if lengths[i] >= min_frames:
            break
        if i == 0:
            j = 1
        elif i == len(runs) - 1:
            j = i - 1
        else:
            j = i - 1 if lengths[i - 1] >= lengths[i + 1] else i + 1
        runs[j][0] = min(runs[j][0], runs[i][0])
        runs[j][1] = max(runs[j][1], runs[i][1])
        runs[j][2] = runs[j][2]
        del runs[i]
        # neighbours of the same label are now adjacent: fuse them
        k = 0
        while k < len(runs) - 1:
            if runs[k][2] == runs[k + 1][2]:
                runs[k][1] = runs[k + 1][1]
                del runs[k + 1]
            else:
                k += 1
    out = np.empty(len(codes), dtype=np.int8)
    for s, e, c in runs:
        out[s:e] = c
    return out
