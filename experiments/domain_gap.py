"""Where does the Barcelona tracking differ from the gap-corrupted external matches?

Adversarial validation: train a classifier to tell Barcelona frames from external frames using the
tracking-derived features only (never any possession labels). If it cannot, the two sources look
alike to a model. If it can, the features it relies on are exactly where a model trained on external
data will be fed something it has not seen. Uses the provided tracking, which the rules allow us to
inspect, and public external matches. No ground truth is read.

    python -m experiments.domain_gap
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from experiments import external as ext  # noqa: E402
from experiments.features import LONG_CONTEXT_PREFIX, load_features  # noqa: E402
from experiments.methods import external_match  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

STEP = 25  # one frame per second: neighbouring frames are near-duplicates and would inflate the AUC


def graded_barca(feats):
    parts = []
    for _, first, last in half_video_blocks(GAME):
        parts.append(feats.iloc[first:last + 1:STEP])
    return pd.concat(parts, ignore_index=True)


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    cols = [c for c in load_features().columns if not c.startswith(LONG_CONTEXT_PREFIX)]
    barca = graded_barca(load_features())[cols]
    matches = ext.match_ids()
    external = pd.concat([external_match(m, 0)[0].iloc[::STEP][cols] for m in matches], ignore_index=True)

    x = pd.concat([barca, external], ignore_index=True)
    y = np.r_[np.ones(len(barca)), np.zeros(len(external))]
    # group = contiguous time blocks, so a block's neighbours are never split between train and test
    block = np.r_[np.arange(len(barca)) // 40, 10_000 + np.arange(len(external)) // 40]
    print("Barcelona frames {}, external frames {}, features {}".format(len(barca), len(external), len(cols)))

    aucs, importances = [], []
    for train, test in GroupKFold(n_splits=5).split(x, y, block):
        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0)
        model.fit(x.iloc[train], y[train])
        aucs.append(roc_auc_score(y[test], model.predict_proba(x.iloc[test])[:, 1]))
        sample = test[:: max(1, len(test) // 4000)]
        perm = permutation_importance(model, x.iloc[sample], y[sample], scoring="roc_auc",
                                      n_repeats=3, random_state=0)
        importances.append(perm.importances_mean)
    print("\nAUC for telling Barcelona from corrupted external frames: {:.3f}  (0.5 = indistinguishable)".format(
        float(np.mean(aucs))))
    ranking = pd.Series(np.mean(importances, axis=0), index=cols).sort_values(ascending=False)
    print("\nfeatures that give the two sources away (AUC lost when shuffled):")
    for name, value in ranking.head(14).items():
        b, e = barca[name], external[name]
        print("  {:<24} {:+.3f}   Barcelona mean {:>8.2f}  external mean {:>8.2f}   NaN {:>4.0%} vs {:>4.0%}".format(
            name, value, b.mean(), e.mean(), b.isna().mean(), e.isna().mean()))


if __name__ == "__main__":
    main()
