"""Which suspected source-to-source difference hurts a model most, and does training on it fix it?

Barcelona labels cannot be used to find out (the rules forbid tuning on them), so the suspected
differences are simulated on the external matches:

  drift     the ball keeps moving while dead (in the Barcelona tracking it is never still; here it is
            frozen for most dead frames)
  turnover  extra ball gaps around real possession changes (the tracker tends to lose the ball there)

For every held-out external match, a model trained on standard data and a model trained on data that
includes both shifts are each scored on the four versions of the held-out match. The drop from the
"none" column is how much a shift hurts; the second row shows how much training on it recovers.

    python -m experiments.robustness
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from experiments import external as ext  # noqa: E402
from experiments.features import DYNAMICS_EXCLUDE, DYNAMICS_PREFIX, LONG_CONTEXT_PREFIX  # noqa: E402
from experiments.run import parse_params  # noqa: E402
from experiments.lomo import decode, join  # noqa: E402
from experiments.methods import ProbModel, _external_tables, align_labels, external_match  # noqa: E402

DROP = ["ball_speed", LONG_CONTEXT_PREFIX]
TEAM, DEAD = 25, 40
LABEL_MIN_DUR = 75
TEST = ("none", "drift", "turnover", "both")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("params", nargs="*")
    ap.add_argument("--train", nargs="+", default=["none", "both"],
                    help="each entry is one model; join stress variants with + to mix them, e.g. none+both")
    args = ap.parse_args()
    params = parse_params(args.params)
    dynamics = params.pop("dynamics", False)
    drop = DROP + (DYNAMICS_EXCLUDE if dynamics else [DYNAMICS_PREFIX])
    models = args.train
    matches = ext.match_ids()
    results = {(tr, te): [] for tr in models for te in TEST}
    for held in matches:
        others = [m for m in matches if m != held]
        for tr in models:
            parts = [_external_tables(others, 2, LABEL_MIN_DUR, 5, stress=s) for s in tr.split("+")]
            x, y = pd.concat([p[0] for p in parts], ignore_index=True), np.concatenate([p[1] for p in parts])
            cols = [c for c in x.columns if not any(c.startswith(p) for p in drop)]
            model = ProbModel(False, **params).fit(x[cols], y)
            for te in TEST:
                feats, raw, half = external_match(held, 1234, stress=te)
                labels = align_labels(raw, half, LABEL_MIN_DUR)
                proba = model.predict_proba(feats[cols])
                gt = join(labels, half, pm.X)
                r = pm.evaluate(gt, join(decode(proba, half, TEAM, DEAD), half, pm.X), bootstrap_n=0)
                results[(tr, te)].append((r["spell_f1_mean"], r["transitions"][25]["f1"], r["macro_f1"],
                                          r["n_pred_spells"] / max(1, r["n_gt_spells"])))
        print("  done", held, flush=True)

    def table(idx, title, fmt="{:>9.3f}"):
        print()
        print("{}  (rows: trained on, columns: tested on; mean of {} held-out matches)".format(title, len(matches)))
        print("  {:<14}".format("") + "".join("{:>9}".format(te) for te in TEST) + "{:>9}".format("average"))
        for tr in models:
            vals = [float(np.mean([v[idx] for v in results[(tr, te)]])) for te in TEST]
            print("  {:<14}".format("train " + tr) + "".join(fmt.format(v) for v in vals) + fmt.format(float(np.mean(vals))))

    print("dynamics features: {}".format(dynamics))
    table(0, "spellF1")
    table(1, "turnover detection F1 within 1 s")
    table(2, "macroF1")
    table(3, "predicted / true possessions", "{:>9.2f}")


if __name__ == "__main__":
    main()
