"""External training matches: the open DFL / IDSSE Bundesliga dataset.

Source:  Bassek et al. 2025, "An integrated dataset of spatiotemporal and event data in elite
         soccer", Scientific Data. https://doi.org/10.6084/m9.figshare.28196177  (CC BY 4.0)
         25 Hz TRACAB tracking in metres (pitch centre = origin), with a per-frame ball record
         that carries BallPossession (1 home, 2 away) and BallStatus (0 dead, 1 alive).

The labels are built the way this challenge's ground truth is described: D where the ball is
dead, otherwise the team the tracking says is in possession.

Two things make a model trained on this data transfer badly unless handled, and this module
handles both:

1. The tracking here is far cleaner than the Barcelona ATD (the ball is almost always seen, the
   Barcelona ball is missing on 41% of graded frames). `corrupt` replays the REAL missingness
   patterns of the Barcelona tracking onto these matches, so a model learns to cope with gaps.
2. Coordinates and attack direction differ. experiments/features.py puts every source into the
   same frame (home attacks +x), using the kickoff positions of each half.

Download the files into data/external/idsse/ (see experiments/README.md); this module reads them,
caches parsed arrays next to them, and never writes outside data/.
"""

import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import DATA_DIR, half_video_blocks, GAME  # noqa: E402
from experiments.features import compute_features, attack_direction, barca_arrays  # noqa: E402

EXT_DIR = os.path.join(DATA_DIR, "external", "idsse")
H, A, D, X = 0, 1, 2, 3

_FRAMESET = re.compile(r'<FrameSet [^>]*>')
_HEAD = re.compile(r'GameSection="(\w+)"[^>]*TeamId="([^"]+)"[^>]*PersonId="([^"]+)"')
_PLAYER_FRAME = re.compile(r'<Frame N="(\d+)" T="[^"]*" X="([^"]*)" Y="([^"]*)"')
_BALL_FRAME = re.compile(
    r'<Frame N="(\d+)"[^>]*? X="([^"]*)" Y="([^"]*)" Z="([^"]*)"[^>]*? '
    r'BallPossession="(\d)" BallStatus="(\d)"')


def match_ids():
    """Match ids that have both a positions file and a match-information file on disk."""
    if not os.path.isdir(EXT_DIR):
        return []
    files = os.listdir(EXT_DIR)
    ids = set()
    for f in files:
        hit = re.search(r"positions_raw_observed_.*DFL-MAT-(\w+)\.xml$", f)
        if hit:
            ids.add(hit.group(1))
    return sorted(m for m in ids if _find(files, "matchinformation", m))


def _find(files, kind, mid):
    hits = [f for f in files if kind in f and mid in f and f.endswith(".xml")]
    return os.path.join(EXT_DIR, hits[0]) if hits else None


def _teams(info_path):
    text = open(info_path, encoding="utf-8").read()
    home = re.search(r'HomeTeamId="([^"]+)"', text).group(1)
    guest = re.search(r'GuestTeamId="([^"]+)"', text).group(1)
    return home, guest


def _parse_positions(path, home_id, guest_id):
    """{section: {"players": {(team, person): (N, x, y)}, "ball": (N, x, y, z, poss, status)}}"""
    text = open(path, encoding="utf-8").read()
    marks = [(m.start(), m.group(0)) for m in _FRAMESET.finditer(text)]
    sections = {}
    for i, (start, header) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        head = _HEAD.search(header)
        if not head:
            continue
        section, team, person = head.groups()
        body = text[start:end]
        slot = sections.setdefault(section, {"players": {}, "ball": None})
        if team == "BALL":
            rows = _BALL_FRAME.findall(body)
            if rows and slot["ball"] is None:
                slot["ball"] = np.array(rows, dtype=np.float64)
        elif team in (home_id, guest_id):
            rows = _PLAYER_FRAME.findall(body)
            if rows:
                slot["players"][(team, person)] = np.array(rows, dtype=np.float64)
    return sections


def _build_half(section, home_id):
    ball = section["ball"]
    if ball is None:
        raise ValueError("no ball record in this half")
    n0, n1 = int(ball[:, 0].min()), int(ball[:, 0].max())
    n = n1 - n0 + 1
    keys = sorted(section["players"], key=lambda k: (k[0] != home_id, k[1]))
    px = np.full((n, len(keys)), np.nan, dtype=np.float32)
    py = np.full_like(px, np.nan)
    for j, key in enumerate(keys):
        rows = section["players"][key]
        idx = rows[:, 0].astype(int) - n0
        ok = (idx >= 0) & (idx < n)
        px[idx[ok], j], py[idx[ok], j] = rows[ok, 1], rows[ok, 2]
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full_like(bx, np.nan)
    status = np.full(n, -1, dtype=np.int8)
    poss = np.full(n, -1, dtype=np.int8)
    idx = ball[:, 0].astype(int) - n0
    bx[idx], by[idx] = ball[:, 1], ball[:, 2]
    poss[idx], status[idx] = ball[:, 4], ball[:, 5]
    home = np.array([k[0] == home_id for k in keys])
    return dict(px=px, py=py, bx=bx, by=by, status=status, poss=poss, home=home, away=~home)


def load_match(mid, rebuild=False):
    """{1: half, 2: half} of parsed arrays for one match, cached as .npz next to the XML."""
    cache = os.path.join(EXT_DIR, "cache_{}.npz".format(mid))
    if os.path.isfile(cache) and not rebuild:
        z = np.load(cache)
        return {h: {k: z["{}_{}".format(k, h)] for k in
                    ("px", "py", "bx", "by", "status", "poss", "home", "away")} for h in (1, 2)}
    files = os.listdir(EXT_DIR)
    info, pos = _find(files, "matchinformation", mid), _find(files, "positions_raw_observed", mid)
    home_id, guest_id = _teams(info)
    print("parsing {} (a minute or two, cached afterwards) ...".format(mid), flush=True)
    sections = _parse_positions(pos, home_id, guest_id)
    halves = {1: _build_half(sections["firstHalf"], home_id), 2: _build_half(sections["secondHalf"], home_id)}
    np.savez_compressed(cache, **{"{}_{}".format(k, h): v for h, half in halves.items() for k, v in half.items()})
    return halves


def labels_of(half):
    """Codes 0..3 per frame: D if the ball is dead, else the possessing team, else X."""
    out = np.full(len(half["status"]), X, dtype=np.int8)
    alive = half["status"] == 1
    out[alive & (half["poss"] == 1)] = H
    out[alive & (half["poss"] == 2)] = A
    out[half["status"] == 0] = D
    return out


# ----------------------------------------------------------------------------- corruption

def barca_missingness():
    """Real missingness templates from the Barcelona tracking, over the graded halves only.

    Returns (ball_missing: bool (m,), home_tracked: (m,), away_tracked: (m,)): when the ball is
    unseen, and how many of each team's players are tracked at each frame (capped at 11, since the
    tracker sometimes duplicates a player). Per-slot gaps are deliberately NOT replayed: slots are
    tracklets that another slot picks up again, so a slot's gap overstates how often a player is
    really unseen (31% per slot against about 9% per player).
    """
    px, py, bx, by, home, away = barca_arrays()
    graded = np.zeros(len(bx), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    seen = ~np.isnan(px[graded])
    return (np.isnan(bx[graded]),
            np.minimum(seen[:, home].sum(axis=1), 11),
            np.minimum(seen[:, away].sum(axis=1), 11))


def _smooth_scores(scores, window=75):
    """Moving average down the time axis, so a player's drop-out score changes slowly."""
    csum = np.vstack([np.zeros((1, scores.shape[1])), np.cumsum(scores, axis=0)])
    n = len(scores)
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - window), np.minimum(n, idx + window + 1)
    return (csum[hi] - csum[lo]) / (hi - lo)[:, None]


def corrupt(half, templates, rng):
    """A copy of a half with Barcelona-style gaps.

    The ball is hidden wherever a randomly offset stretch of the real Barcelona ball track was
    missing. For players, at each frame as many per team are hidden as the Barcelona tracker was
    missing at a randomly offset frame, and WHICH players are hidden follows slowly varying random
    scores, so the same player disappears for a run of frames, as in real tracking.

    The gaps are independent of what is happening on the pitch. Real ones are not (they cluster
    around aerial balls and crowded boxes), which is a known limit of this approximation.
    """
    ball_missing, home_tracked, away_tracked = templates
    n = len(half["bx"])
    m = len(ball_missing)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in half.items()}
    mask = ball_missing[(int(rng.integers(m)) + np.arange(n)) % m]
    out["bx"][mask], out["by"][mask] = np.nan, np.nan
    for team_mask, counts in ((half["home"], home_tracked), (half["away"], away_tracked)):
        cols = np.flatnonzero(team_mask)
        on_pitch = ~np.isnan(half["px"][:, cols])
        scores = _smooth_scores(rng.normal(size=(n, len(cols))))
        scores = np.where(on_pitch, scores, -np.inf)          # never "drop" someone already absent
        missing = np.maximum(0, 11 - counts[(int(rng.integers(m)) + np.arange(n)) % m])
        order = np.argsort(-scores, axis=1)                   # highest score = first to vanish
        rank = np.empty_like(order)
        rank[np.arange(n)[:, None], order] = np.arange(len(cols))[None, :]
        drop = (rank < missing[:, None]) & on_pitch
        sub_x, sub_y = out["px"][:, cols], out["py"][:, cols]
        sub_x[drop], sub_y[drop] = np.nan, np.nan
        out["px"][:, cols], out["py"][:, cols] = sub_x, sub_y
    return out


# ----------------------------------------------------------------------------- feature tables

def _dead_drift(half, rng):
    """Let the ball keep moving while the ball is dead, instead of sitting frozen.

    In the Barcelona tracking the ball is never still (median step about 5 m/s) and its path is smooth;
    in this dataset it is frozen for most dead-ball frames. Each dead stretch gets a smooth random path
    that starts at the frozen position: heading follows a random walk, speed hovers around 4 m/s.
    """
    out = half["bx"].copy(), half["by"].copy()
    bx, by = out
    dead = half["status"] == 0
    edges = np.flatnonzero(np.diff(np.r_[0, dead.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if np.isnan(bx[s]):
            continue
        x, y = float(bx[s]), float(by[s])
        theta, speed = rng.uniform(0, 2 * np.pi), 4.0
        for i in range(s, e):
            theta += rng.normal(0, 0.15)
            speed = float(np.clip(speed + 0.2 * (4.0 - speed) + rng.normal(0, 0.6), 0.5, 9.0))
            x = float(np.clip(x + speed / 25 * np.cos(theta), -52.5, 52.5))
            y = float(np.clip(y + speed / 25 * np.sin(theta), -34.0, 34.0))
            bx[i], by[i] = x, y
    return bx, by


def _turnover_gaps(half, rng, prob=0.6):
    """Hide the ball around real changes of possession: the tracker tends to lose it exactly there
    (tackles, aerial balls, crowded areas). Returns a boolean mask over the half."""
    lab = labels_of(half)
    change = np.flatnonzero(np.diff(lab) != 0) + 1
    mask = np.zeros(len(lab), dtype=bool)
    for c in change:
        if rng.random() < prob:
            mask[max(0, c - int(rng.integers(25, 100))): c + int(rng.integers(25, 100))] = True
    return mask


def match_table(mid, templates=None, seed=0, stress="none"):
    """(features, labels, half) for one match, both halves, optionally with Barcelona-style gaps.

    stress adds a suspected difference between the sources: "drift" (the ball moves while dead),
    "turnover" (extra gaps around possession changes), "both", or "none".
    """
    rng = np.random.default_rng(seed)
    halves = load_match(mid)
    frames, labels, which = [], [], []
    for h in (1, 2):
        half = halves[h]
        if stress in ("drift", "both"):                      # 1. the ball keeps moving while dead
            half = dict(half)
            half["bx"], half["by"] = _dead_drift(halves[h], rng)
        if templates is not None:                            # 2. Barcelona-style gaps, independent of play
            half = corrupt(half, templates, rng)
        if stress in ("turnover", "both"):                   # 3. extra gaps around real possession changes
            half = dict(half)
            mask = _turnover_gaps(halves[h], rng)
            half["bx"], half["by"] = half["bx"].copy(), half["by"].copy()
            half["bx"][mask], half["by"][mask] = np.nan, np.nan
        n = len(half["bx"])
        # kickoff = first frame with the ball alive; read the sides from the ORIGINAL, uncorrupted half
        orig = halves[h]
        kick = int(np.argmax(orig["status"] == 1))
        direction = attack_direction(orig["px"], orig["home"], orig["away"], [(0, n - 1, kick)])
        feats = compute_features(half["px"], half["py"], half["bx"], half["by"],
                                 half["home"], half["away"], direction)
        frames.append(feats.reset_index(drop=True))
        labels.append(labels_of(halves[h]))
        which.append(np.full(n, h, dtype=np.int8))
    return pd.concat(frames, ignore_index=True), np.concatenate(labels), np.concatenate(which)
