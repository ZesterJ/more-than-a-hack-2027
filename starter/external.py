"""External training/validation data. The submission pipeline uses IDSSE only (CC-BY 4.0).

Metrica Sports sample games 1 and 2 are also loadable here but are NOT used by the pipeline: their README states
no licence and the rules say to check a dataset's licence before using it. Metrica (full matches, 25 fps):

    https://github.com/metrica-sports/sample-data   (README: no licence text, "be responsible,
    acknowledge the source") -- downloaded by `python starter/external.py --fetch` into the
    git-ignored data/external/metrica/. Nothing here is needed to run the scorer or notebooks 01-05.

Metrica ships tracking and events but **no possession labels**, so `derive_labels` builds H/A/D from the
event stream with explicit rules. They are heuristic: use them to compare decoders and to pretrain,
never as ground truth. Two differences from this match's tracking are handled on purpose:

* the ball is almost never missing in Metrica but is unseen on ~43% of frames here, so
  `apply_occlusion` copies this match's ball-missing pattern onto the external ball;
* the teams swap ends at half time and the track order differs, so `orient_home_attacks_right`
  flips x until the home team defends x < 0.5, for every dataset alike.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from features import DATA_DIR, FPS  # noqa: E402

METRICA_DIR = os.path.join(DATA_DIR, "external", "metrica")
IDSSE_DIR = os.path.join(DATA_DIR, "external", "idsse")
IDSSE_HF = "https://huggingface.co/datasets/PySport/idsse-data/resolve/main"
IDSSE_MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
IDSSE_DIR = os.path.join(DATA_DIR, "external", "idsse")
IDSSE_HF = "https://huggingface.co/datasets/PySport/idsse-data/resolve/main"
IDSSE_MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
PITCH = (105.0, 68.0)
H, A, D = 0, 1, 2
KMH_PER_MS = 3.6   # Sportec reports S in km/h (mean player speed 6.6 vs 1.8 m/s in the other sets)


def metrica_files(game):
    stem = "Sample_Game_{}".format(game)
    return {
        "events": os.path.join(METRICA_DIR, stem, stem + "_RawEventsData.csv"),
        "home": os.path.join(METRICA_DIR, stem, stem + "_RawTrackingData_Home_Team.csv"),
        "away": os.path.join(METRICA_DIR, stem, stem + "_RawTrackingData_Away_Team.csv"),
    }


def fetch(games=(1, 2)):
    """Download the CSVs for the given games (about 65 MB each)."""
    import urllib.request
    for g in games:
        for path in metrica_files(g).values():
            if os.path.isfile(path):
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            url = "{}/Sample_Game_{}/{}".format(RAW, g, os.path.basename(path))
            print("downloading", url)
            urllib.request.urlretrieve(url, path)


def idsse_files(match):
    com = "DFL-COM-000001" if match in ("J03WMX", "J03WN1") else "DFL-COM-000002"
    tag = "{}_DFL-MAT-{}.xml".format(com, match)
    return {"positions": os.path.join(IDSSE_DIR, "DFL_04_03_positions_raw_observed_" + tag),
            "info": os.path.join(IDSSE_DIR, "DFL_02_01_matchinformation_" + tag)}


def fetch_idsse(matches=IDSSE_MATCHES):
    """Download IDSSE (Sportec, CC-BY 4.0) positions + match info: ~370 MB per match, 2.6 GB for all 7."""
    import urllib.request
    for m in matches:
        for path in idsse_files(m).values():
            if not os.path.isfile(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                url = "{}/{}".format(IDSSE_HF, os.path.basename(path))
                print("downloading", url)
                urllib.request.urlretrieve(url, path)


def load_idsse(match, cache=True):
    """Same dict as load_metrica, plus the *provider's* per-frame labels in 'labels' (0 H, 1 A, 2 D).

    Sportec's raw positions carry BallStatus (0 = dead) and BallPossession (1 = home, 2 = guest) on every
    ball frame, so no event rules are needed. Coordinates are metres from the pitch centre, rescaled to
    0-1. Both halves are concatenated; 'period' marks them. The parse (~30 s) is cached next to the data.
    """
    import re
    cpath = os.path.join(IDSSE_DIR, "cache", match + ".npz")
    if cache and os.path.isfile(cpath):
        z = np.load(cpath)
        return {k: z[k] for k in z.files}

    frame_re = re.compile(r'N="(\d+)".*? X="(-?[\d.]+)" Y="(-?[\d.]+)"(?: Z="-?[\d.]+")? D="[^"]*" S="([\d.]+)"'
                          r'(?:.*BallPossession="(\d)" BallStatus="(\d)")?')
    files = idsse_files(match)
    info = open(files["info"], encoding="utf-8").read()
    home_id = re.search(r'<Team TeamId="([^"]+)"[^>]*Role="home"', info).group(1)

    rows, sections, cur = {}, {}, None
    with open(files["positions"], encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<FrameSet"):
                sec = re.search(r'GameSection="(\w+)"', line).group(1)
                team = re.search(r'TeamId="([^"]+)"', line).group(1)
                pid = re.search(r'PersonId="([^"]+)"', line).group(1)
                cur = None if team == "referee" else (sec, team, pid)
                if cur:
                    sections.setdefault(sec, []).append(cur)
                    rows[cur] = []
                continue
            if cur and line.startswith("<Frame "):
                m = frame_re.search(line)
                rows[cur].append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)),
                                  int(m.group(5) or 0), int(m.group(6) if m.group(6) else -1)))

    order = [s for s in ("firstHalf", "secondHalf") if s in sections]
    ball_key = {s: next(k for k in sections[s] if k[1] == "BALL") for s in order}
    people = sorted({(t == home_id, pid) for s in order for (_, t, pid) in sections[s] if t != "BALL"}, reverse=True)
    slot = {pid: i for i, (_, pid) in enumerate(people)}
    is_home = np.array([h for h, _ in people])

    parts = []
    for ordinal, sec in enumerate(order, start=1):
        b = np.array(rows[ball_key[sec]], dtype=np.float64)
        n0, n = int(b[0, 0]), len(b)
        players = np.full((n, len(people), 3), np.nan, dtype=np.float32)
        for key in sections[sec]:
            if key[1] == "BALL":
                continue
            r = np.array(rows[key], dtype=np.float64)
            idx = (r[:, 0] - n0).astype(int)
            ok = (idx >= 0) & (idx < n)
            players[idx[ok], slot[key[2]]] = np.column_stack(
                [(r[ok, 1] + 52.5) / 105.0, (r[ok, 2] + 34.0) / 68.0, r[ok, 3] / KMH_PER_MS])
        ball = np.column_stack([(b[:, 1] + 52.5) / 105.0, (b[:, 2] + 34.0) / 68.0, b[:, 3] / KMH_PER_MS]).astype(np.float32)
        labels = np.where(b[:, 5] == 0, D, np.where(b[:, 4] == 1, H, A)).astype(np.int8)
        parts.append((players, ball, labels, np.full(n, ordinal, dtype=np.int8)))
    out = {"players": np.concatenate([p[0] for p in parts]), "ball": np.concatenate([p[1] for p in parts]),
           "labels": np.concatenate([p[2] for p in parts]), "period": np.concatenate([p[3] for p in parts]),
           "is_home": is_home}
    out["frame"] = np.arange(len(out["ball"]))
    if cache:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        np.savez_compressed(cpath, **out)
    return out


def _read_team(path):
    header = pd.read_csv(path, header=None, skiprows=2, nrows=1).iloc[0].tolist()
    n_players = sum(isinstance(v, str) and v.startswith("Player") for v in header)
    t = pd.read_csv(path, header=None, skiprows=3)
    xy = t.iloc[:, 3:3 + 2 * n_players].to_numpy(dtype=np.float32).reshape(len(t), n_players, 2)
    ball = t.iloc[:, 3 + 2 * n_players:5 + 2 * n_players].to_numpy(dtype=np.float32)
    return t[0].to_numpy(), t[1].to_numpy(), xy, ball


def _speed(xy):
    """Per-slot speed in m/s from positions (the CSVs have none), 5-frame smoothed."""
    d = np.diff(xy, axis=0, prepend=xy[:1]) * np.array(PITCH, dtype=np.float32)
    v = np.hypot(d[..., 0], d[..., 1]) * FPS
    v = np.where(np.isnan(v), np.nan, np.minimum(v, 12.0))
    out = pd.DataFrame(v.reshape(len(v), -1)).rolling(5, min_periods=1, center=True).mean().to_numpy()
    return out.reshape(v.shape).astype(np.float32)


def load_metrica(game, n_frames=None):
    """{'players': (n, P, 3), 'ball': (n, 3), 'is_home': (P,) bool, 'period': (n,), 'frame': (n,)}."""
    files = metrica_files(game)
    period, frame, hxy, ball = _read_team(files["home"])
    _, _, axy, _ = _read_team(files["away"])
    if n_frames:
        period, frame, hxy, axy, ball = (a[:n_frames] for a in (period, frame, hxy, axy, ball))
    xy = np.concatenate([hxy, axy], axis=1)
    players = np.concatenate([xy, _speed(xy)[..., None]], axis=-1)
    ball_speed = _speed(ball[:, None, :])[:, :1]
    return {"players": players, "ball": np.concatenate([ball, ball_speed], axis=-1),
            "is_home": np.arange(xy.shape[1]) < hxy.shape[1], "period": period, "frame": frame}


def derive_labels(game, trk):
    """H/A/D per tracking row from the event stream (rules in the module docstring and below).

    Owner = team of the latest PASS / SHOT / SET PIECE / RECOVERY. Dead ball starts at a BALL OUT,
    a FAULT RECEIVED, an offside / hand-ball, a goal, or the end of a half, and lasts until the next
    restart (a SET PIECE or a goal-kick pass). Everything else keeps the current owner.
    """
    ev = pd.read_csv(metrica_files(game)["events"]).sort_values(["Start Frame", "End Frame"], kind="stable")
    frame_to_row = {f: i for i, f in enumerate(trk["frame"])}
    labels = np.full(len(trk["frame"]), D, dtype=np.int8)
    owner, dead, last_row = None, True, 0

    def row_of(frame):
        return frame_to_row.get(frame, int(np.searchsorted(trk["frame"], frame)))

    def paint(upto):
        nonlocal last_row
        labels[last_row:upto] = D if (dead or owner is None) else owner
        last_row = max(last_row, upto)

    for _, e in ev.iterrows():
        typ, sub = e["Type"], str(e["Subtype"]) if pd.notna(e["Subtype"]) else ""
        team = H if e["Team"] == "Home" else A
        start = row_of(e["Start Frame"])
        restart = typ == "SET PIECE" or (typ == "PASS" and sub.startswith("GOAL KICK"))
        if dead:
            if not restart:
                continue
            paint(start)
            dead = False
        else:
            paint(start)
        if typ in ("PASS", "SHOT", "SET PIECE", "RECOVERY"):
            owner = team
        if typ == "BALL OUT" or typ == "FAULT RECEIVED" or (typ == "BALL LOST" and sub in ("OFFSIDE", "HAND BALL", "END HALF")):
            dead = True
        elif typ == "SHOT" and sub.endswith("GOAL"):
            paint(row_of(e["End Frame"]))
            dead = True
    paint(len(labels))
    return labels


def orient_home_attacks_right(trk, is_home, segments=None):
    """Mirror x within each segment so the home team's goalkeeper side is x < 0.5 (home attacks +x).

    The goalkeeper is the home player closest to its own goal line, so compare how close the leftmost
    and the rightmost home player sit to their respective goal lines (average over the segment).
    """
    P, B = trk["players"].copy(), trk["ball"].copy()
    segments = segments or [(0, len(B))]
    for a, b in segments:
        hx = P[a:b][:, is_home, 0]
        with np.errstate(all="ignore"):
            left, right = np.nanmean(np.nanmin(hx, axis=1)), 1.0 - np.nanmean(np.nanmax(hx, axis=1))
        if left > right:          # home's keeper is on the right-hand side: mirror
            P[a:b, :, 0] = 1.0 - P[a:b, :, 0]
            B[a:b, 0] = 1.0 - B[a:b, 0]
    return {**trk, "players": P, "ball": B}


def apply_occlusion(trk, missing_pattern):
    """Blank the ball wherever `missing_pattern` (bool, tiled cyclically) says it was unseen."""
    reps = int(np.ceil(len(trk["ball"]) / len(missing_pattern)))
    mask = np.tile(missing_pattern, reps)[:len(trk["ball"])]
    ball = trk["ball"].copy()
    ball[mask] = np.nan
    return {**trk, "ball": ball}


def period_segments(trk):
    p = trk["period"]
    cuts = np.flatnonzero(np.diff(p)) + 1
    edges = np.concatenate(([0], cuts, [len(p)]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetch-idsse", action="store_true", help="download the 7 IDSSE matches (2.6 GB, CC-BY 4.0)")
    parser.add_argument("--fetch-metrica", action="store_true",
                        help="download Metrica games 1-2 (~130 MB). NOT used by the submission pipeline: "
                             "Metrica's README states no licence, so confirm with the organisers first")
    args = parser.parse_args()
    if args.fetch_idsse:
        fetch_idsse()
    if args.fetch_metrica:
        fetch()
