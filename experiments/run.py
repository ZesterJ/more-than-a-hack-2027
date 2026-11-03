"""Run a method through the cross-validated harness.

    python -m experiments.run --list
    python -m experiments.run nearest window=50                  # reproduces the repo baseline
    python -m experiments.run logistic window=50 min_dur=25
    python -m experiments.run logistic window=50 --save-oof      # out-of-fold CSV -> view in the analyzer
    python -m experiments.run logistic window=50 --write-submission my_team.csv
    python -m experiments.run --show-log

Run from the repository root. Parameters are key=value and are recorded in the trial log.
"""

import argparse
import ast
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from experiments import harness  # noqa: E402
from experiments.features import load_features  # noqa: E402
from experiments.methods import REGISTRY  # noqa: E402

LETTERS = np.array(["H", "A", "D", "X"])


def parse_params(items):
    params = {}
    for item in items:
        if "=" not in item:
            raise SystemExit("parameter {!r} should look like key=value".format(item))
        key, value = item.split("=", 1)
        try:
            params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            params[key] = value
    return params


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("method", nargs="?", help="name from the registry (see --list)")
    ap.add_argument("params", nargs="*", help="key=value method parameters")
    ap.add_argument("--list", action="store_true", help="list methods and exit")
    ap.add_argument("--show-log", action="store_true", help="print the trial log and exit")
    ap.add_argument("--folds", type=int, default=harness.N_FOLDS)
    ap.add_argument("--embargo", type=int, default=harness.EMBARGO)
    ap.add_argument("--bootstrap", type=int, default=harness.BOOTSTRAP)
    ap.add_argument("--no-log", action="store_true", help="do not record this run as a trial")
    ap.add_argument("--save-oof", action="store_true",
                    help="write the out-of-fold predictions to evaluation/results/oof_<method>.csv")
    ap.add_argument("--write-submission", metavar="PATH",
                    help="fit on ALL labelled frames, predict the whole match, write + validate a CSV")
    args = ap.parse_args()

    if args.list:
        for name, cls in REGISTRY.items():
            print("{:<10} {}".format(name, (cls.__doc__ or "").strip().splitlines()[0]))
        return
    if args.show_log:
        print(harness.format_log())
        return
    if args.method not in REGISTRY:
        raise SystemExit("unknown method {!r}; try --list".format(args.method))

    params = parse_params(args.params)
    factory = lambda: REGISTRY[args.method](**params)  # noqa: E731
    feats = load_features()

    oof, result, gt = harness.cross_validate(factory, feats, args.folds, args.embargo, args.bootstrap)

    pc = result["per_class"]
    print("{}({})  {}-fold, embargo {} frames, out-of-fold on the labelled window".format(
        args.method, params, args.folds, args.embargo))
    print("  spellF1  {:.4f}   90% interval [{:.4f}, {:.4f}]".format(
        result["spell_f1"], result["spell_lo"], result["spell_hi"]))
    print("  macroF1  {:.4f}   accuracy {:.1%}   F1 H {:.3f}  A {:.3f}  D {:.3f}".format(
        result["macro_f1"], result["accuracy"], pc["H"], pc["A"], pc["D"]))
    print("  possessions: {} predicted vs {} true".format(
        result["n_pred_spells"], result["n_gt_spells"]))

    if not args.no_log:
        trial = harness.log_trial(args.method, params, args.folds, args.embargo, result)
        n = len(harness.read_log())
        print("  logged as trial #{}. {} trial(s) so far -- the best of many is optimistic and "
              "differences under {} are noise.".format(trial, n, harness.NOISE_BAND))

    if args.save_oof:
        path = os.path.join(harness.RESULTS_DIR, "oof_{}.csv".format(args.method))
        with open(path, "w") as handle:
            handle.write("frame,label\n")
            handle.writelines("{},{}\n".format(i, l) for i, l in enumerate(LETTERS[oof]))
        print("  wrote", os.path.relpath(path, ROOT))

    if args.write_submission:
        method = factory()
        train = np.flatnonzero(gt != 3)
        method.fit(feats, gt, train)
        codes = harness.apply_known_structure(method.predict(feats))
        path = harness.write_submission(codes, args.write_submission)
        print("  wrote and validated", path)
        print("  NOTE: this model was fit on all 7,500 labelled frames, so its score on them is "
              "in-sample.\n        The cross-validated score above is the honest estimate.")


if __name__ == "__main__":
    main()
