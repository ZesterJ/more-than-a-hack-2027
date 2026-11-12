"""Leave-one-match-out evaluation on the external matches.

Each external match is held out in turn: a model is trained on the others (with Barcelona-style
gaps), predicts the held-out match (with its own differently seeded gaps), is decoded, and is scored
against that match's labels with the official scorer. With about 276 possessions per match this is
the only place in the project where two configurations can be told apart; the Barcelona window has
17 possessions.

Labels are aligned to the Barcelona convention first (blips shorter than LABEL_MIN_DUR frames removed).

    python -m experiments.lomo                                       # defaults: one stage
    python -m experiments.lomo two_stage=True
    python -m experiments.lomo two_stage=True 'drop=["ball_speed","contact"]'
    python -m experiments.lomo --matches J03WMX J03WN1 J03WOH        # fewer matches, faster

Parameters go to ProbModel / the feature selection; the decoder grid is fixed below so that
configurations are compared at their best setting, not at a setting that happened to suit one.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from experiments import external as ext  # noqa: E402
from experiments.features import DYNAMICS_EXCLUDE, DYNAMICS_PREFIX, EXPERIMENTAL, LONG_CONTEXT_PREFIX  # noqa: E402
from experiments.methods import (BaggedProbModel, ProbModel, _training_tables, align_labels,  # noqa: E402
                                 external_match)
from experiments.postprocess import viterbi  # noqa: E402
from experiments.run import parse_params  # noqa: E402

GAP = 50
DECODERS = [(20, 40, 40), (25, 40, 40), (30, 40, 40), (40, 40, 40), (25, 60, 60), (15, 40, 40)]   # (team, live->dead, dead->live)


def join(arr, half, fill):
    parts = []
    for h in (1, 2):
        parts.append(arr[half == h])
        if h == 1:
            parts.append(np.full(GAP, fill, dtype=np.int8))
    return np.concatenate(parts).astype(np.int8)


def decode(proba, half, a, b, c=None):
    out = np.zeros(len(proba), dtype=np.int8)
    for h in (1, 2):
        sel = half == h
        out[sel] = viterbi(np.log(np.clip(proba[sel], 1e-6, 1.0)), a, b, switch_restart=c)
    return out


def held_out(matches, held, drop, label_min_dur, copies, two_stage, hgb, bags=1, stresses=("none",)):
    train = [m for m in matches if m != held]
    members = []
    for b in range(bags):
        x, y = _training_tables(train, copies, label_min_dur, 5, stresses, b * copies)
        cols = [c for c in x.columns if not any(c.startswith(p) for p in drop)]
        members.append(ProbModel(two_stage, **hgb).fit(x[cols], y))
    model = members[0] if bags == 1 else BaggedProbModel(members)
    feats, raw, half = external_match(held, 1234)
    proba = model.predict_proba(feats[cols])
    return proba, align_labels(raw, half, label_min_dur), half


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("params", nargs="*")
    ap.add_argument("--matches", nargs="+")
    args = ap.parse_args()
    params = parse_params(args.params)
    extras = params.pop("extras", False)
    long_context = params.pop("long_context", False)
    dynamics = params.pop("dynamics", False)
    stresses = params.pop("train_stress", ["none"])
    drop = (params.pop("drop", ["ball_speed"]) + ([] if extras else EXPERIMENTAL)
            + ([] if long_context else [LONG_CONTEXT_PREFIX])
            + (DYNAMICS_EXCLUDE if dynamics else [DYNAMICS_PREFIX]))
    label_min_dur = params.pop("label_min_dur", 50)
    copies = params.pop("copies", 2)
    two_stage = params.pop("two_stage", False)
    bags = params.pop("bags", 1)
    matches = args.matches or ext.match_ids()
    print("config: drop={} label_min_dur={} copies={} two_stage={} bags={} dynamics={} train_stress={} {}".format(drop, label_min_dur, copies, two_stage, bags, dynamics, stresses, params))

    rows = {d: [] for d in DECODERS}
    for held in matches:
        proba, labels, half = held_out(matches, held, drop, label_min_dur, copies, two_stage, params, bags, stresses)
        gt = join(labels, half, pm.X)
        for a, b, c in DECODERS:
            rows[(a, b, c)].append(pm.evaluate(gt, join(decode(proba, half, a, b, c), half, pm.X), bootstrap_n=0))
        print("  held out {}".format(held), flush=True)

    print("\nmean over {} held-out matches".format(len(matches)))
    print("  {:<10} {:>8} {:>8} {:>10} {:>9} {:>20}".format(
        "decoder", "spellF1", "macroF1", "transF1@1s", "F1 H/A/D", "possessions pred/true"))
    for (a, b, c), rs in rows.items():
        mean = lambda f: float(np.mean([f(r) for r in rs]))  # noqa: E731
        pc = [mean(lambda r, k=k: r["per_class"][k]["f1"]) for k in ("H", "A", "D")]
        print("  {:<10} {:>8.3f} {:>8.3f} {:>10.3f}  {:.2f}/{:.2f}/{:.2f} {:>8.0f} / {:.0f}".format(
            "{}/{}/{}".format(a, b, c), mean(lambda r: r["spell_f1_mean"]), mean(lambda r: r["macro_f1"]),
            mean(lambda r: r["transitions"][25]["f1"]), pc[0], pc[1], pc[2],
            mean(lambda r: r["n_pred_spells"]), mean(lambda r: r["n_gt_spells"])))


if __name__ == "__main__":
    main()
