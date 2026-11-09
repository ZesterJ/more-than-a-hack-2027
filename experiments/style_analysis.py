"""Does the model do worse on matches (or halves) with longer, more patient possessions?

The hypothesis: Barcelona keeps the ball with short passes, the external Bundesliga matches are more
direct, and that mismatch hurts turnover detection. Barcelona labels cannot be used to test it (the
rules forbid tuning on them), but the external matches differ from each other in style, so the same
question can be asked there: across held-out matches and halves, does performance fall as true
possessions get longer?

Uses only external data. Same frozen configuration as make_submission.py, one model per held-out match.

    python -m experiments.style_analysis
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
from experiments.features import EXPERIMENTAL, LONG_CONTEXT_PREFIX  # noqa: E402
from experiments.lomo import decode, held_out  # noqa: E402
from experiments.postprocess import run_lengths  # noqa: E402

DROP = ["ball_speed", LONG_CONTEXT_PREFIX]
TEAM, DEAD = 25, 40


def half_row(labels, pred):
    """(mean true possession seconds, true count, predicted count, spellF1, transF1) for one half."""
    runs = pm.valid_runs(labels)
    r = pm.evaluate(labels, pred, bootstrap_n=0)
    live = [(b - a) / 25.0 for a, b, c in run_lengths(labels) if c in (0, 1)]
    return (float(np.mean(live)), r["n_gt_spells"], r["n_pred_spells"], r["spell_f1_mean"],
            r["transitions"][25]["f1"])


def main():
    matches = ext.match_ids()
    rows = []
    for held in matches:
        proba, labels, half = held_out(matches, held, DROP, 75, 2, False, {})
        pred = decode(proba, half, TEAM, DEAD)
        for h in (1, 2):
            sel = half == h
            mean_s, n_true, n_pred, spell, trans = half_row(labels[sel].astype(np.int8), pred[sel])
            rows.append((held, h, mean_s, n_true, n_pred, spell, trans))
        print("  done", held, flush=True)

    print("\n{:<8} {:>4} {:>14} {:>7} {:>7} {:>8} {:>8}".format(
        "match", "half", "true poss (s)", "true n", "pred n", "spellF1", "transF1"))
    for held, h, mean_s, n_true, n_pred, spell, trans in sorted(rows, key=lambda r: r[2]):
        print("{:<8} {:>4} {:>14.1f} {:>7d} {:>7d} {:>8.3f} {:>8.3f}".format(
            held, h, mean_s, n_true, n_pred, spell, trans))

    arr = np.array([[r[2], r[3], r[4], r[5], r[6]] for r in rows])
    dur, n_true, n_pred, spell, trans = arr.T
    ratio = n_pred / n_true
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
    print("\nacross {} held-out halves (correlation with the mean true possession length):".format(len(rows)))
    print("  spellF1                         {:+.2f}".format(corr(dur, spell)))
    print("  turnover-detection F1 (1 s)     {:+.2f}".format(corr(dur, trans)))
    print("  predicted / true possessions    {:+.2f}   (mean ratio {:.2f})".format(corr(dur, ratio), ratio.mean()))
    lo, hi = dur <= np.median(dur), dur > np.median(dur)
    print("\nshortest-possession half of the sample vs longest:")
    print("  mean true possession    {:5.1f} s   vs {:5.1f} s".format(dur[lo].mean(), dur[hi].mean()))
    print("  spellF1                 {:5.3f}     vs {:5.3f}".format(spell[lo].mean(), spell[hi].mean()))
    print("  turnover F1 (1 s)       {:5.3f}     vs {:5.3f}".format(trans[lo].mean(), trans[hi].mean()))
    print("  predicted / true count  {:5.2f}     vs {:5.2f}".format(ratio[lo].mean(), ratio[hi].mean()))


if __name__ == "__main__":
    main()
