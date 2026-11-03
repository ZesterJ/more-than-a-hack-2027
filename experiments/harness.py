"""Cross-validated scoring for possession methods, with a trial log.

Why this exists
---------------
The only labels are 7,500 frames (17 possessions). Scoring a method on frames it was trained on
says nothing, and trying many methods against the same 17 possessions finds noise: one possession
moves spellF1 by about six points, and the best of N tries is optimistic by construction. This
module makes the honest way the easy way:

  * the labelled window is cut into contiguous folds; each fold is predicted by a model that never
    saw it, with an embargo gap either side so neighbouring frames cannot leak across the cut
  * the out-of-fold predictions are assembled into one array and scored once with the OFFICIAL
    scorer (evaluation/possession_metrics.py), with a bootstrap interval
  * every run is appended to experiments/results.csv with a trial number, so how many things you
    tried is on record and printed next to the score

A method is any object with `fit(feats, labels, train_idx)` and `predict(feats) -> codes`, where
`predict` returns one code (0 H, 1 A, 2 D, 3 X) for EVERY frame of the match. See methods.py.
"""

import csv
import datetime
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
LOG_PATH = os.path.join(HERE, "results.csv")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

N_FOLDS = 5
EMBARGO = 150          # frames (6 s) dropped from training on each side of a test fold
BOOTSTRAP = 200
NOISE_BAND = 0.04      # spellF1 differences below this are noise on a full match; far worse here
SECOND_HALF_KICKOFF = 113742  # DATA/FAQ: the 2nd half is graded from 110,808 but kicks off here

LOG_FIELDS = ["trial", "time", "commit", "method", "params", "folds", "embargo", "spell_f1",
              "spell_lo", "spell_hi", "macro_f1", "accuracy", "n_pred_spells", "n_gt_spells"]


def load_ground_truth():
    return pm.label_codes(pm.read_label_csv(GT_PATH))


def labelled_window(gt):
    idx = np.flatnonzero(gt != pm.X)
    return int(idx[0]), int(idx[-1])


def make_folds(gt, n_folds=N_FOLDS, embargo=EMBARGO):
    """[(train_idx, test_idx)] over the labelled frames: contiguous test blocks, embargoed train."""
    lo, hi = labelled_window(gt)
    labelled = np.flatnonzero(gt != pm.X)
    edges = np.linspace(lo, hi + 1, n_folds + 1).astype(int)
    folds = []
    for k in range(n_folds):
        a, b = edges[k], edges[k + 1]
        test = labelled[(labelled >= a) & (labelled < b)]
        train = labelled[(labelled < a - embargo) | (labelled >= b + embargo)]
        folds.append((train, test))
    return folds


def score(gt, pred, bootstrap=BOOTSTRAP):
    res = pm.evaluate(gt, pred, bootstrap_n=bootstrap)
    return {
        "spell_f1": res["spell_f1_mean"],
        "spell_lo": res["spell_bootstrap_low"],
        "spell_hi": res["spell_bootstrap_high"],
        "macro_f1": res["macro_f1"],
        "accuracy": res["accuracy"],
        "n_pred_spells": res["n_pred_spells"],
        "n_gt_spells": res["n_gt_spells"],
        "per_class": {k: v["f1"] for k, v in res["per_class"].items()},
    }


def cross_validate(make_method, feats, n_folds=N_FOLDS, embargo=EMBARGO, bootstrap=BOOTSTRAP):
    """Out-of-fold predictions over the labelled window, and their official score.

    make_method is called once per fold, so no state leaks between folds. The whole-match
    prediction of each fold's model is only read on that fold's test frames.
    """
    gt = load_ground_truth()
    oof = np.full(len(gt), pm.X, dtype=np.int8)
    for train, test in make_folds(gt, n_folds, embargo):
        method = make_method()
        method.fit(feats, gt, train)
        oof[test] = np.asarray(method.predict(feats))[test]
    return oof, score(gt, oof, bootstrap), gt


def apply_known_structure(codes):
    """What the published documents already say about the match, applied to a full-length array.

    * frames outside the two graded halves are X (never scored)
    * the second half is graded from 110,808 but kicks off at 113,742: those frames are all D
    This uses only DATA.md / FAQ.md, never the private ground truth.
    """
    codes = np.asarray(codes, dtype=np.int8).copy()
    graded = np.zeros(len(codes), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    codes[~graded] = pm.X
    first_half2 = half_video_blocks(GAME)[1][1]
    codes[first_half2:SECOND_HALF_KICKOFF] = pm.D
    return codes


def write_submission(codes, path):
    """Write frame,label and run the official validator on the result."""
    letters = np.array(pm.LABELS)[codes]
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, l) for i, l in enumerate(letters))
    frame = pm.read_label_csv(path)
    problem = pm.validate_submission(frame, GAMES[GAME]["n_video_frames"])
    if problem:
        raise ValueError("{} failed validation: {}".format(path, problem))
    return path


# ---------------------------------------------------------------------------- trial log

def _commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "?"
    except Exception:
        return "?"


def read_log():
    if not os.path.isfile(LOG_PATH):
        return []
    with open(LOG_PATH, newline="") as handle:
        return list(csv.DictReader(handle))


def log_trial(method, params, n_folds, embargo, result):
    rows = read_log()
    row = {
        "trial": len(rows) + 1,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": _commit(),
        "method": method,
        "params": json.dumps(params, sort_keys=True),
        "folds": n_folds,
        "embargo": embargo,
        "spell_f1": round(result["spell_f1"], 4),
        "spell_lo": round(result["spell_lo"], 4),
        "spell_hi": round(result["spell_hi"], 4),
        "macro_f1": round(result["macro_f1"], 4),
        "accuracy": round(result["accuracy"], 4),
        "n_pred_spells": result["n_pred_spells"],
        "n_gt_spells": result["n_gt_spells"],
    }
    new_file = not os.path.isfile(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return row["trial"]


def format_log(limit=20):
    rows = read_log()
    if not rows:
        return "no trials logged yet"
    lines = ["trial  spellF1  [90% interval]    macroF1  acc    poss  method(params)"]
    for r in rows[-limit:]:
        lines.append("{:>5}  {:>7}  [{:>5}, {:>5}]    {:>7}  {:>5}  {:>4}  {}({})".format(
            r["trial"], r["spell_f1"], r["spell_lo"], r["spell_hi"], r["macro_f1"],
            r["accuracy"], r["n_pred_spells"], r["method"], r["params"]))
    best = max(rows, key=lambda r: float(r["spell_f1"]))
    lines.append("")
    lines.append("{} trials so far. Best of {} is optimistic: expect the best spellF1 ({}) to shrink "
                 "on new data, and ignore gaps under {}.".format(len(rows), len(rows),
                                                                  best["spell_f1"], NOISE_BAND))
    return "\n".join(lines)
