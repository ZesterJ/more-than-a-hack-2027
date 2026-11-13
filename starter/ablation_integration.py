"""Physical + temporal features, together: which combination holds up when the tracking is hard? External data only.

Crosses the feature choices with the training choices on leave-one-match-out over the seven IDSSE matches, never
touching the provided match's labels. Each variant is tested on four versions of the held-out match, all with the
provided match's ball-gap pattern applied:

  frozen    the match as recorded (the ball is frozen while dead)
  moving    the ball keeps moving while dead, as in the provided tracking      (external.dead_ball_drift)
  turnover  extra ball gaps around possession changes                          (external.turnover_gaps)
  both      moving + turnover

Variants (features x training):
  V0  Jordan's 109 features, trained on frozen-ball data                         (the pipeline as merged)
  V1  V0 with the 14 ball-motion features removed
  V2  V0 + 36 team-movement features (dynamics)
  V3  V1 + dynamics
  V4  V3, trained on frozen + (moving + turnover) copies of the matches

Metrics: macro-F1 and dead-ball recall from the posteriors' argmax; turnover detection F1 within 1 s and spellF1
from a plain 2 s smoothing decoder. The extra features are only kept if they win here, not on the provided match.

    python starter/ablation_integration.py
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
TESTS = ("frozen", "moving", "turnover", "both")


def column_names():
    t = ext.load_idsse(ext.IDSSE_MATCHES[0])
    small = {**t, "players": t["players"][:3000], "ball": t["ball"][:3000], "period": t["period"][:3000]}
    X22 = build_features(small, is_home=t["is_home"])
    E = build_extra(small, X22)
    base = [n for n in list(X22.columns) + list(E.columns) if n not in ep.UNOBSERVED]
    return base + ep.ADVANCED + ep.BOUNDARY + ep.MOTION + ep.TEMPORAL + ep.DYNAMICS


def select(sets, names, drop_ball_motion, dynamics):
    keep = [i for i, n in enumerate(names)
            if not (drop_ball_motion and n in ep.BALL_MOTION) and (dynamics or n not in ep.DYNAMICS)]
    return {m: {**s, "X": s["X"][:, keep]} for m, s in sets.items()}


def score(P, match):
    y = match["y"]
    pred = P.argmax(1)
    r = pm.evaluate(y, decode_pipeline(P, match["aux"], w=50))
    return dict(macro=f1_score(y, pred, average="macro"), dead_rec=recall_score(y == 2, pred == 2),
                turn=r["transitions"][25]["f1"], spell=r["spell_f1_mean"])


def main():
    ep.USE_DYNAMICS, ep.DROP_BALL_MOTION = True, False       # build every column once, then select subsets
    ep.USE_SETPIECE, ep.USE_RESTART = False, False          # the feature set as it was when this ablation was run
    pattern = ep.ball_unseen_pattern()
    names = column_names()
    conditions = {"frozen": {}, "moving": dict(drift=True), "turnover": dict(turnover=True),
                  "both": dict(drift=True, turnover=True)}
    built = {}
    for name, kw in conditions.items():
        print("building features: {} ...".format(name), flush=True)
        built[name] = ep.load_idsse_sets(pattern, **kw, **ARGS)
    assert next(iter(built["frozen"].values()))["X"].shape[1] == len(names), "names out of step with the matrix"

    def feats(drop_bm, dyn):
        return {c: select(s, names, drop_bm, dyn) for c, s in built.items()}

    f0, f1, f2, f3 = feats(False, False), feats(True, False), feats(False, True), feats(True, True)
    mix3 = {**f3["frozen"], **{m + "+hard": s for m, s in f3["both"].items()}}
    variants = {
        "V0 Jordan as merged": (f0, f0["frozen"]),
        "V1 ball-motion removed": (f1, f1["frozen"]),
        "V2 + team dynamics": (f2, f2["frozen"]),
        "V3 removed + dynamics": (f3, f3["frozen"]),
        "V4 V3 + hard-condition training": (f3, mix3),
    }
    results = {(v, t): [] for v in variants for t in TESTS}
    for held in built["frozen"]:
        for v, (test_sets, train) in variants.items():
            pool = [k for k in train if k.split("+")[0] != held]
            model = ep.fit("lightgbm", train, pool)
            for t in TESTS:
                results[(v, t)].append(score(_full_proba(model, test_sets[t][held]["X"]), test_sets[t][held]))
        print("  done", held, flush=True)

    for key, title in (("spell", "spellF1"), ("turn", "turnover detection F1 (within 1 s)"),
                       ("dead_rec", "dead-ball recall"), ("macro", "macro-F1")):
        print("\n{}   (mean over {} held-out external matches)".format(title, len(built["frozen"])))
        print("{:<34}".format("") + "".join("{:>10}".format(t) for t in TESTS) + "{:>10}".format("average"))
        for v in variants:
            vals = [float(np.mean([r[key] for r in results[(v, t)]])) for t in TESTS]
            print("{:<34}".format(v) + "".join("{:>10.3f}".format(x) for x in vals) + "{:>10.3f}".format(float(np.mean(vals))))


if __name__ == "__main__":
    main()
