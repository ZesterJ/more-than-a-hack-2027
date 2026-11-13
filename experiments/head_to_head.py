"""Head-to-head: the starter pipeline (starter/external_pipeline.py) against the experiments pipeline
(experiments/make_submission.py), on identical inputs and identical truth. External data only.

For each of the seven IDSSE matches held out in turn, both pipelines are trained on the other six exactly as they
ship, and both predict the SAME corrupted copy of the held-out match in each of four test conditions:

  frozen    the provided match's ball-gap pattern applied (the ball is frozen while dead, as IDSSE records it)
  moving    the ball keeps moving while dead, as in the provided tracking        (external.dead_ball_drift)
  turnover  extra ball gaps around possession changes                           (external.turnover_gaps)
  both      moving + turnover

The experiments pipeline computes its own features from those same corrupted positions. Predictions are scored with
the official scorer against two versions of the held-out match's provider labels, each pipeline's own convention,
so neither wins because of how the truth is defined:

  starter truth       live runs under 3 s join the previous team  (tactical.controlled_possession)
  experiments truth   runs under 3 s merged into the longer neighbour  (experiments.methods.align_labels)

Shipped settings. Starter: LightGBM on 109 + dynamics + set-piece features with ball-motion features removed,
mirrored training copies, no domain weighting, decoder chosen on the other six matches. Experiments: touch/line
and team-dynamics features, 3 bagged models trained on standard + stressed gapped copies, Viterbi 25/40.

    python -m experiments.head_to_head                 # all seven matches, about 1.5 hours the first time
    python -m experiments.head_to_head --matches J03WMX J03WN1
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    sys.path.insert(0, p)

import possession_metrics as pm  # noqa: E402
import external as sext  # noqa: E402  (starter/external.py)
import external_pipeline as ep  # noqa: E402
from benchmark import _full_proba  # noqa: E402
from tactical import controlled_possession, decode_pipeline  # noqa: E402
from experiments import external as jx  # noqa: E402
from experiments.features import DYNAMICS_EXCLUDE, LONG_CONTEXT_PREFIX, compute_features  # noqa: E402
from experiments.lomo import decode as viterbi_decode, join  # noqa: E402
from experiments.methods import BaggedProbModel, ProbModel, _training_tables, align_labels  # noqa: E402

CONDITIONS = {"frozen": {}, "moving": dict(drift=True), "turnover": dict(turnover=True),
              "both": dict(drift=True, turnover=True)}
ARGS = dict(advanced=True, control_s=3.0, boundary=True, motion=True, temporal=True)
MIN_DUR = 75                                     # both pipelines' 3 s convention


def corrupted_trk(match, pattern, drift=False, turnover=False):
    """The held-out match exactly as ep.load_idsse_sets builds it (same steps, same seeds)."""
    trk = sext.load_idsse(match)
    trk = sext.orient_home_attacks_right(trk, trk["is_home"], sext.period_segments(trk))
    if drift:
        trk = sext.dead_ball_drift(trk)
    if pattern is not None:
        trk = sext.apply_occlusion(trk, pattern)
    if turnover:
        trk = sext.turnover_gaps(trk)
    return trk


def experiments_features(trk):
    """The experiments pipeline's feature table for a starter-format (already oriented) tracking dict."""
    import pandas as pd
    P, B = trk["players"], trk["ball"]
    home = np.asarray(trk["is_home"])
    parts = []
    for h in (1, 2):
        sel = trk["period"] == h
        px, py = P[sel, :, 0] * 105.0 - 52.5, P[sel, :, 1] * 68.0 - 34.0
        bx, by = B[sel, 0] * 105.0 - 52.5, B[sel, 1] * 68.0 - 34.0
        parts.append(compute_features(px, py, bx, by, home, ~home, np.ones(sel.sum(), dtype=np.float32))
                     .reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)


def experiments_model(train):
    drop = ["ball_speed", LONG_CONTEXT_PREFIX] + list(DYNAMICS_EXCLUDE)   # extras=True, dynamics=True
    members, cols = [], None
    for b in range(3):                                                     # bags=3, copies=2
        x, y = _training_tables(train, 2, MIN_DUR, 5, ["none", "both"], b * 2)
        cols = [c for c in x.columns if not any(c.startswith(p) for p in drop)]
        members.append(ProbModel(False).fit(x[cols], y))
    return BaggedProbModel(members), cols


def metrics(truth, pred, half):
    r = pm.evaluate(join(truth, half, pm.X), join(pred, half, pm.X), bootstrap_n=0)
    return dict(spell=r["spell_f1_mean"], macro=r["macro_f1"], turn=r["transitions"][25]["f1"],
                dead=r["per_class"]["D"]["recall"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matches", nargs="+", default=list(sext.IDSSE_MATCHES))
    matches = ap.parse_args().matches
    pattern = ep.ball_unseen_pattern()
    ep.USE_RESTART = False

    print("building starter features ...", flush=True)
    train_sets = ep.load_idsse_sets(pattern, matches=matches, augment=True, **ARGS)
    test_sets = {c: ep.load_idsse_sets(pattern, matches=matches, **kw, **ARGS) for c, kw in CONDITIONS.items()}

    truths, halves, posts, jw_codes = {}, {}, {c: {} for c in CONDITIONS}, {c: {} for c in CONDITIONS}
    for m in matches:
        raw = sext.load_idsse(m)
        half = raw["period"].astype(np.int8)
        jw_raw = np.concatenate([jx.labels_of(jx.load_match(m)[h]) for h in (1, 2)])
        truths[m] = {"starter truth": controlled_possession(raw["labels"], MIN_DUR),
                     "experiments truth": align_labels(jw_raw, half, MIN_DUR)}
        halves[m] = half

    for held in matches:
        others = [m for m in matches if m != held]
        ours = ep.fit("lightgbm", train_sets, others)
        theirs, cols = experiments_model(others)
        for c, kw in CONDITIONS.items():
            posts[c][held] = _full_proba(ours, test_sets[c][held]["X"])
            feats = experiments_features(corrupted_trk(held, pattern, **kw))
            jw_codes[c][held] = viterbi_decode(theirs.predict_proba(feats[cols]), halves[held], 25, 40, 40)
        print("  held out", held, flush=True)

    # starter decoder for each held-out match: the grid setting best on the OTHER matches' frozen-condition posteriors
    grid = ep.GRID
    scores = {m: ep.grid_scores(posts["frozen"][m], test_sets["frozen"][m], grid) for m in matches}
    choice = {m: grid[int(np.mean([scores[o] for o in matches if o != m], axis=0).argmax())] for m in matches}

    rows = {}
    for c in CONDITIONS:
        for m in matches:
            ours_pred = decode_pipeline(posts[c][m], test_sets[c][m]["aux"], **choice[m])
            for tname, truth in truths[m].items():
                rows.setdefault(("starter", tname, c), []).append(metrics(truth, ours_pred, halves[m]))
                rows.setdefault(("experiments", tname, c), []).append(metrics(truth, jw_codes[c][m], halves[m]))

    for key, title in (("spell", "spellF1"), ("turn", "turnover detection F1 (within 1 s)"),
                       ("dead", "dead-ball recall"), ("macro", "macro-F1")):
        print("\n{}   (mean over {} held-out external matches)".format(title, len(matches)))
        print("{:<34}".format("") + "".join("{:>10}".format(c) for c in CONDITIONS) + "{:>10}".format("average"))
        for pipe in ("starter", "experiments"):
            for tname in ("starter truth", "experiments truth"):
                vals = [float(np.mean([r[key] for r in rows[(pipe, tname, c)]])) for c in CONDITIONS]
                print("{:<34}".format("{} on {}".format(pipe, tname))
                      + "".join("{:>10.3f}".format(v) for v in vals) + "{:>10.3f}".format(float(np.mean(vals))))
    wins = {t: sum(np.mean([rows[("starter", t, c)][i]["spell"] for c in CONDITIONS])
                   > np.mean([rows[("experiments", t, c)][i]["spell"] for c in CONDITIONS]) for i in range(len(matches)))
            for t in ("starter truth", "experiments truth")}
    print("\nmatches where the starter pipeline has the higher average spellF1:",
          ", ".join("{} {}/{}".format(t, w, len(matches)) for t, w in wins.items()))


if __name__ == "__main__":
    main()
