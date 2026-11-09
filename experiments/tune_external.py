"""Choose decoding settings on the external matches, where there are hundreds of possessions.

Leave one match out: train the classifier on the other external matches (with Barcelona-style
gaps), predict the held-out match (with its own, differently seeded gaps), decode it with each
candidate setting, and score against that match's labels with the official scorer. The Barcelona
public window has 17 possessions and cannot separate settings like these; four matches of about
250 possessions per half can.

The labels are aligned to the Barcelona convention first (possession blips shorter than
LABEL_MIN_DUR frames removed), because the raw Bundesliga labels flip about twice as often as the
Barcelona truth and would reward a decoder that over-segments.

    python -m experiments.tune_external
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from experiments import external as ext  # noqa: E402
from experiments.methods import ExternalGBDT  # noqa: E402
from experiments.postprocess import min_duration_filter, smooth_probs, viterbi  # noqa: E402

LABEL_MIN_DUR = 50
DROP = ["ball_speed"]
GAP = 50  # X frames inserted between the two halves so no possession spans the join

VOTE_WINDOWS = (0, 25, 50, 75, 100, 150)
VITERBI = [(10, 10), (20, 20), (30, 30), (45, 45), (30, 60), (60, 30)]


def held_out_probs(matches, held, copies=2):
    """P(H, A, D) for the held-out match from a model trained on the others."""
    train = [m for m in matches if m != held]
    model = ExternalGBDT(ext=train, copies=copies, label_min_dur=LABEL_MIN_DUR, drop=DROP)
    from experiments.methods import _external_tables
    from sklearn.ensemble import HistGradientBoostingClassifier
    x, y = _external_tables(train, copies, LABEL_MIN_DUR, model.step)
    cols = [c for c in x.columns if not any(c.startswith(p) for p in DROP)]
    clf = HistGradientBoostingClassifier(max_iter=model.max_iter, learning_rate=model.learning_rate,
                                         max_leaf_nodes=model.max_leaf_nodes, l2_regularization=1.0,
                                         min_samples_leaf=model.min_samples_leaf, random_state=0)
    clf.fit(x[cols], y)
    templates = ext.barca_missingness()
    feats, labels, half = ext.match_table(held, templates, seed=1234)
    proba = np.zeros((len(feats), 3))
    proba[:, clf.classes_] = clf.predict_proba(feats[cols])
    aligned = labels.copy()
    for h in (1, 2):
        sel = half == h
        aligned[sel] = min_duration_filter(labels[sel], LABEL_MIN_DUR)
    return np.clip(proba, 1e-6, 1.0), aligned, half


def join_halves(arrays, half, fill):
    parts = []
    for h in (1, 2):
        parts.append(arrays[half == h])
        if h == 1:
            parts.append(np.full(GAP, fill, dtype=np.int8))
    return np.concatenate(parts).astype(np.int8)


def decode(proba, half, kind, a, b):
    out = np.zeros(len(proba), dtype=np.int8)
    for h in (1, 2):
        sel = half == h
        if kind == "vote":
            out[sel] = smooth_probs(proba[sel], a)
        else:
            out[sel] = viterbi(np.log(proba[sel]), a, b)
    return out


def main():
    matches = ext.match_ids()
    print("external matches:", matches)
    held_results = {}
    for held in matches:
        proba, aligned, half = held_out_probs(matches, held)
        gt = join_halves(aligned, half, pm.X)
        rows = []
        for w in VOTE_WINDOWS:
            pred = join_halves(decode(proba, half, "vote", w, None), half, pm.X)
            rows.append((("vote", w), pm.evaluate(gt, pred, bootstrap_n=0)))
        for a, b in VITERBI:
            pred = join_halves(decode(proba, half, "viterbi", a, b), half, pm.X)
            rows.append((("viterbi", a, b), pm.evaluate(gt, pred, bootstrap_n=0)))
        held_results[held] = rows
        n_true = rows[0][1]["n_gt_spells"]
        print("held out {}: {} true possessions".format(held, n_true), flush=True)

    print("\nmean over held-out matches (spellF1 | macroF1 | possessions predicted / true)")
    names = [r[0] for r in held_results[matches[0]]]
    for i, name in enumerate(names):
        spell = np.mean([held_results[m][i][1]["spell_f1_mean"] for m in matches])
        macro = np.mean([held_results[m][i][1]["macro_f1"] for m in matches])
        npred = np.mean([held_results[m][i][1]["n_pred_spells"] for m in matches])
        ntrue = np.mean([held_results[m][i][1]["n_gt_spells"] for m in matches])
        print("  {:<24} spellF1 {:.3f}  macroF1 {:.3f}  {:>5.0f} / {:.0f}".format(
            " ".join(str(p) for p in name), spell, macro, npred, ntrue))


if __name__ == "__main__":
    main()
