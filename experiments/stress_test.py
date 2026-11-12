"""How much does a method degrade when the ball goes missing as often as it does in the full match?

The public ground-truth window is unusually easy: the ball is unseen on about 11% of its frames,
against about 41% over the whole match. A score measured there overstates what the same method
will get on the rest of the match. This script hides the ball inside the window, replaying real
Barcelona ball-gap sequences at random offsets, recomputes the features, and scores the same
method again. Players are left as they are.

    python -m experiments.stress_test gbdt ext=[...] drop=[...] decode=viterbi
    python -m experiments.stress_test nearest window=50 --repeats 5

The extra gaps are independent of play, so this is a pessimistic-ish approximation: the real gaps
cluster around aerial balls and crowded boxes. Read it as "roughly how much worse", not a forecast.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from experiments import harness, external  # noqa: E402
from experiments.features import (barca_arrays, attack_direction, compute_features,  # noqa: E402
                                  load_features, FIRST_HALF_KICKOFF, SECOND_HALF_KICKOFF)
from experiments.methods import REGISTRY  # noqa: E402
from experiments.run import parse_params  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402


def stressed_features(offset_seed):
    px, py, bx, by, home, away = barca_arrays()
    blocks = half_video_blocks(GAME)
    halves = [(blocks[0][1], blocks[0][2], FIRST_HALF_KICKOFF), (blocks[1][1], blocks[1][2], SECOND_HALF_KICKOFF)]
    direction = attack_direction(px, home, away, halves)   # read from the untouched tracking
    gt = harness.load_ground_truth()
    lo, hi = harness.labelled_window(gt)
    rng = np.random.default_rng(offset_seed)
    ball_missing = external.barca_missingness()[0]
    n = hi - lo + 1
    mask = ball_missing[(int(rng.integers(len(ball_missing))) + np.arange(n)) % len(ball_missing)]
    bx, by = bx.copy(), by.copy()
    bx[lo:hi + 1][mask], by[lo:hi + 1][mask] = np.nan, np.nan
    window_missing = float(np.isnan(bx[lo:hi + 1]).mean())
    return compute_features(px, py, bx, by, home, away, direction), window_missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("method")
    ap.add_argument("params", nargs="*")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    params = parse_params(args.params)
    factory = lambda: REGISTRY[args.method](**params)  # noqa: E731

    gt = harness.load_ground_truth()
    train = np.flatnonzero(gt != 3)
    base_feats = load_features()
    lo, hi = harness.labelled_window(gt)
    print("window ball-missing as given: {:.1%}".format(float(base_feats["ball_missing"].iloc[lo:hi + 1].mean())))

    method = factory()
    method.fit(base_feats, gt, train)
    if params.get("barca_weight", 0) or not params.get("ext"):
        raise SystemExit("this test needs a method that never saw the Barcelona labels (ext=[...], barca_weight=0): "
                         "otherwise the score is in-sample and the comparison means nothing")
    plain = harness.score(gt, method.predict(base_feats), bootstrap=0)
    rows = []
    for seed in range(args.repeats):
        feats, missing = stressed_features(seed)
        s = harness.score(gt, method.predict(feats), bootstrap=0)
        rows.append((missing, s))
    print("\n{:<28} {:>9} {:>9} {:>9} {:>7}".format("", "ball miss", "spellF1", "macroF1", "poss"))
    print("{:<28} {:>9.1%} {:>9.3f} {:>9.3f} {:>7}".format(
        "as given", float(base_feats["ball_missing"].iloc[lo:hi + 1].mean()), plain["spell_f1"],
        plain["macro_f1"], plain["n_pred_spells"]))
    for i, (missing, s) in enumerate(rows):
        print("{:<28} {:>9.1%} {:>9.3f} {:>9.3f} {:>7}".format(
            "stressed (offset {})".format(i), missing, s["spell_f1"], s["macro_f1"], s["n_pred_spells"]))
    print("{:<28} {:>9.1%} {:>9.3f} {:>9.3f}".format(
        "stressed mean", np.mean([m for m, _ in rows]), np.mean([s["spell_f1"] for _, s in rows]),
        np.mean([s["macro_f1"] for _, s in rows])))


if __name__ == "__main__":
    main()
