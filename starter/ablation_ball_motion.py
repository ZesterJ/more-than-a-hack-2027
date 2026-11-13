"""Does the frozen-ball shortcut hurt, and what removes it? External data only.

In IDSSE the ball is frozen for most dead-ball frames; in the provided tracking it never is. Features built
from how the ball moves (external_pipeline.BALL_MOTION) therefore teach "ball not moving => dead", a cue that
does not exist in the provided tracking. This script measures the damage and the cures with leave-one-match-out
on the seven IDSSE matches, never touching the provided match's labels:

  A  all features, trained on frozen-ball data            (current default)
  B  BALL_MOTION features removed, trained on frozen-ball data
  C  all features, trained on frozen-ball + moving-ball-when-dead data

each tested on the held-out match as recorded (frozen ball) and with the ball made to keep moving while dead
(`external.dead_ball_drift`), both with the provided match's ball-gap pattern applied. Frame metrics use the
argmax of the posteriors; spellF1 uses a plain 2 s smoothing decoder.

    python starter/ablation_ball_motion.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import external as ext  # noqa: E402
import external_pipeline as ep  # noqa: E402
import possession_metrics as pm  # noqa: E402
from benchmark import _full_proba  # noqa: E402
from features import build_extra, build_features  # noqa: E402
from sklearn.metrics import f1_score, recall_score  # noqa: E402
from tactical import decode_pipeline  # noqa: E402

ARGS = dict(advanced=True, control_s=3.0, boundary=True, motion=True, temporal=True)


def column_names():
    """Names of the columns of the full feature matrix, in order (built on a short slice of one match)."""
    t = ext.load_idsse(ext.IDSSE_MATCHES[0])
    small = {**t, "players": t["players"][:3000], "ball": t["ball"][:3000]}
    X22 = build_features(small, is_home=t["is_home"])
    E = build_extra(small, X22)
    base = [n for n in list(X22.columns) + list(E.columns) if n not in ep.UNOBSERVED]
    return base + ep.ADVANCED + ep.BOUNDARY + ep.MOTION + ep.TEMPORAL


def without_ball_motion(sets, names):
    keep = [i for i, n in enumerate(names) if n not in ep.BALL_MOTION]
    return {m: {**s, "X": s["X"][:, keep]} for m, s in sets.items()}


def score(P, match):
    y = match["y"]
    pred = P.argmax(1)
    codes = decode_pipeline(P, match["aux"], w=50)
    return dict(macro=f1_score(y, pred, average="macro"), dead_f1=f1_score(y == 2, pred == 2),
                dead_rec=recall_score(y == 2, pred == 2), spell=pm.evaluate(y, codes)["spell_f1_mean"])


def main():
    pattern = ep.ball_unseen_pattern()          # the provided match's ball-gap pattern (tracking only)
    names = column_names()
    print("building features (frozen ball, then moving-when-dead) ...", flush=True)
    clean = ep.load_idsse_sets(pattern, **ARGS)
    drift = ep.load_idsse_sets(pattern, drift=True, **ARGS)
    assert next(iter(clean.values()))["X"].shape[1] == len(names), "column names out of step with the matrix"
    drop_clean, drop_drift = without_ball_motion(clean, names), without_ball_motion(drift, names)
    print("{} features, {} of them ball-motion".format(len(names), len(ep.BALL_MOTION)), flush=True)

    variants = {
        "A all features, frozen-ball training": (clean, clean, drift),
        "B ball-motion removed": (drop_clean, drop_clean, drop_drift),
        "C all features, + moving-ball training": ({**clean, **{m + "+drift": s for m, s in drift.items()}}, clean, drift),
    }
    results = {(v, t): [] for v in variants for t in ("frozen", "moving")}
    for held in clean:
        for v, (train, test_frozen, test_moving) in variants.items():
            pool = [k for k in train if k.split("+")[0] != held]
            model = ep.fit("lightgbm", train, pool)
            for t, tests in (("frozen", test_frozen), ("moving", test_moving)):
                results[(v, t)].append(score(_full_proba(model, tests[held]["X"]), tests[held]))
        print("  done", held, flush=True)

    print("\nmean over {} held-out external matches".format(len(clean)))
    print("{:<42} {:<8} {:>9} {:>9} {:>9} {:>9}".format("trained as", "tested", "macroF1", "dead F1", "dead rec", "spellF1"))
    for (v, t), rows in results.items():
        mean = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
        print("{:<42} {:<8} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}".format(
            v, t + " ball", mean("macro"), mean("dead_f1"), mean("dead_rec"), mean("spell")))


if __name__ == "__main__":
    main()
