"""Which features reduce the loss, held out, and are they safe to use?

Two targets, so the work can be split:

    python -m experiments.feature_importance --target team    # home vs away, on live frames only
    python -m experiments.feature_importance --target dead    # live vs dead ball, on all frames

For every external match in turn: train a gradient-boosted model on the other matches (with
Barcelona-style gaps), then measure how much the HELD-OUT log-loss gets worse when one feature
family is shuffled. A family is a feature and its windows (`margin`, `margin_roll25`,
`margin_roll75` are one family), because shuffling one of several near-copies shows almost nothing.

Importance alone is not enough. A feature can matter a lot on the external data purely because of how
that provider records things (the Bundesliga data freezes the ball when it is dead; the deepest-player
series separated Barcelona from every external match with AUC 0.96). So each family also gets a
source audit: can a classifier tell Barcelona frames from external frames using that family alone,
compared with how separable one external match is from the rest? Only tracking features are used; no
possession ground truth is read, and nothing is tuned on it.

A family is worth keeping when it is important in at least 6 of the 7 held-out matches AND its
audit is clean. Everything here uses external matches and the provided tracking only.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from experiments import external as ext  # noqa: E402
from experiments.features import (DYNAMICS_EXCLUDE, EXPERIMENTAL, LONG_CONTEXT_PREFIX,  # noqa: E402
                                  load_features)
from experiments.feature_groups import split as split_groups  # noqa: E402
from experiments.methods import _external_tables, align_labels, external_match  # noqa: E402

LABEL_MIN_DUR = 75
AUDIT_STEP = 25
ARTEFACT_AUC = 0.80   # a family this separable on its own is suspect (one external match vs the rest is ~0.52-0.60)


def family(name):
    """margin_roll25 -> margin; dyn_home_cx_fwd50 -> dyn_home_cx; contact_bal_roll75 -> contact_bal."""
    return re.sub(r"_(roll|fwd|back)\d+$", "", name)


def select_columns(columns, use_extras):
    drop = ["ball_speed", LONG_CONTEXT_PREFIX] + list(DYNAMICS_EXCLUDE) + ([] if use_extras else list(EXPERIMENTAL))
    return [c for c in columns if not any(c.startswith(p) for p in drop)]


def target_rows(labels, which):
    """(row mask, binary target) for the chosen question."""
    if which == "team":
        mask = (labels == 0) | (labels == 1)
        return mask, (labels == 0).astype(int)       # 1 = home
    return labels != 3, (labels == 2).astype(int)    # 1 = dead


def held_out_importance(matches, which, use_extras, repeats, rng):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss
    per_match = []
    for held in matches:
        others = [m for m in matches if m != held]
        x, y = _external_tables(others, 2, LABEL_MIN_DUR, 5)
        cols = select_columns(x.columns, use_extras)
        keep, yb = target_rows(y, which)
        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                               min_samples_leaf=100, l2_regularization=1.0, random_state=0)
        model.fit(x.loc[keep, cols], yb[keep])

        feats, raw, half = external_match(held, 1234)
        labels = align_labels(raw, half, LABEL_MIN_DUR)
        mask, target = target_rows(labels, which)
        idx = np.flatnonzero(mask)[::5]
        xt, yt = feats.iloc[idx][cols].reset_index(drop=True), target[idx]
        base = log_loss(yt, model.predict_proba(xt)[:, 1])

        groups = {}
        for c in cols:
            groups.setdefault(family(c), []).append(c)
        parts = split_groups(cols)               # whole groups too, so the physical / temporal split has a number
        groups["[ALL PHYSICAL]"] = parts["physical"]
        groups["[ALL TEMPORAL]"] = parts["temporal"]
        gain = {}
        for fam, members in groups.items():
            losses = []
            for _ in range(repeats):
                shuffled = xt.copy()
                shuffled[members] = xt[members].sample(frac=1, random_state=int(rng.integers(1 << 30))).to_numpy()
                losses.append(log_loss(yt, model.predict_proba(shuffled)[:, 1]) - base)
            gain[fam] = float(np.mean(losses))
        per_match.append(gain)
        print("  {} done (held-out log-loss {:.3f})".format(held, base), flush=True)
    return pd.DataFrame(per_match, index=matches)


def source_audit(matches, columns):
    """AUC of telling Barcelona from external frames using each family alone, and the one-match control."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from experiments.domain_gap import graded_barca

    barca = graded_barca(load_features())
    tabs = {m: external_match(m, 0)[0].iloc[::AUDIT_STEP].reset_index(drop=True) for m in matches}
    rest_all = pd.concat(tabs.values(), ignore_index=True)

    def auc(target, rest, cols):
        x = pd.concat([target[cols], rest[cols]], ignore_index=True)
        y = np.r_[np.ones(len(target)), np.zeros(len(rest))]
        block = np.r_[np.arange(len(target)) // 40, 10_000 + np.arange(len(rest)) // 40]
        scores = []
        for tr, te in GroupKFold(n_splits=3).split(x, y, block):
            m = HistGradientBoostingClassifier(max_iter=60, random_state=0).fit(x.iloc[tr], y[tr])
            scores.append(roc_auc_score(y[te], m.predict_proba(x.iloc[te])[:, 1]))
        return float(np.mean(scores))

    groups = {}
    for c in columns:
        groups.setdefault(family(c), []).append(c)
    control_match = matches[0]
    control_rest = pd.concat([tabs[m] for m in matches if m != control_match], ignore_index=True)
    out = {}
    for fam, members in groups.items():
        out[fam] = (auc(barca, rest_all, members), auc(tabs[control_match], control_rest, members))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=("team", "dead"), required=True)
    ap.add_argument("--extras", action="store_true", help="include the contact / line features")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--no-audit", action="store_true", help="skip the source audit")
    args = ap.parse_args()

    matches = ext.match_ids()
    print("target: {}   ({} external matches)".format(
        "home vs away on live frames" if args.target == "team" else "live vs dead on all frames", len(matches)))
    table = held_out_importance(matches, args.target, args.extras, args.repeats, np.random.default_rng(0))
    mean, std = table.mean(), table.std()
    consistent = (table > 0).sum()

    audit = {}
    if not args.no_audit:
        print("auditing feature families against the Barcelona tracking ...", flush=True)
        columns = select_columns(external_match(matches[0], 0)[0].columns, args.extras)
        audit = source_audit(matches, columns)

    print("\n{:<28} {:>11} {:>8} {:>10} {:>10} {:>8}  {}".format(
        "feature family", "loss gain", "+/-", "matches>0", "Barca AUC", "control", "verdict"))
    for fam in mean.sort_values(ascending=False).index:
        a, c = audit.get(fam, (float("nan"), float("nan")))
        keep = consistent[fam] >= 6 and mean[fam] > 0
        flag = "ARTEFACT RISK" if (not args.no_audit and a >= ARTEFACT_AUC) else ""
        verdict = flag or ("keep" if keep else ("weak" if mean[fam] > 0 else "no value"))
        print("{:<28} {:>11.4f} {:>8.4f} {:>7d}/{} {:>10.3f} {:>8.3f}  {}".format(
            fam, mean[fam], std[fam], int(consistent[fam]), len(matches), a, c, verdict))
    print("\nloss gain = how much held-out log-loss rises when the family is shuffled (higher = more important).")
    print("Barca AUC / control = can the family alone tell Barcelona (or one external match) from the rest.")


if __name__ == "__main__":
    main()
