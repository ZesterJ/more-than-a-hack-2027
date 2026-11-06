"""Experiment A: raw external-only H/A/D LightGBM evaluated as LIVE/DEAD."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm

MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
TOLERANCES = (12, 25)
CONFIG = dict(model="lightgbm", advanced=True, boundary=True, motion=True,
              control_s=3, augment=True, pattern=None, weights=None,
              dead_threshold=0.5, decoder=None, smoothing=None, search=False,
              train_stride=6, fps=25, matches=list(MATCHES))


def binary_probabilities(P):
    P = np.asarray(P)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("Expected H/A/D probabilities with shape (n, 3)")
    if not np.isfinite(P).all() or np.any(P < 0) or np.any(P > 1):
        raise ValueError("Invalid probability values")
    if not np.allclose(P.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Probabilities must sum to one")
    return np.column_stack((P[:, 0] + P[:, 1], P[:, 2]))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0,
                tp=int(tp), fp=int(fp), fn=int(fn))


def frame_metrics(cm):
    cm = np.asarray(cm, dtype=np.int64)
    live = prf(int(cm[0, 0]), int(cm[1, 0]), int(cm[0, 1]))
    dead = prf(int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0]))
    return dict(live=live, dead=dead, accuracy=float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0,
                macro_f1=(live["f1"] + dead["f1"]) / 2,
                confusion=cm.tolist(), confusion_order=["LIVE", "DEAD"],
                support=dict(live=int(cm[0].sum()), dead=int(cm[1].sum())),
                n_scored=int(cm.sum()))


def valid_spans(y, period):
    """Split on every period change and every invalid reference frame."""
    spans, start = [], None
    for i in range(len(y)):
        valid = y[i] in (0, 1, 2)
        if start is not None and (not valid or period[i] != period[i - 1]):
            spans.append((start, i))
            start = None
        if valid and start is None:
            start = i
    if start is not None:
        spans.append((start, len(y)))
    return spans


def duration_distribution(durations):
    a = np.asarray(durations, dtype=float)
    edges = [0, 1, 2, 5, 10, 30, 60, float("inf")]
    return dict(count=len(a), total_s=float(a.sum()), mean_s=float(a.mean()) if len(a) else None,
                std_s=float(a.std()) if len(a) else None,
                quantiles_s={k: float(np.quantile(a, q)) if len(a) else None
                             for k, q in (("min", 0), ("p25", .25), ("median", .5),
                                          ("p75", .75), ("p90", .9), ("max", 1))},
                histogram_edges_s=[0, 1, 2, 5, 10, 30, 60, None],
                histogram_counts=np.histogram(a, bins=edges)[0].tolist(),
                durations_s=a.tolist())


def boundary_summary(tp, fp, fn, errors):
    a = np.asarray(errors, dtype=float) / 25
    return dict(**prf(tp, fp, fn), signed_errors_s=a.tolist(),
                mean_signed_error_s=float(a.mean()) if len(a) else None,
                median_signed_error_s=float(np.median(a)) if len(a) else None,
                mean_absolute_error_s=float(np.abs(a).mean()) if len(a) else None,
                p90_absolute_error_s=float(np.quantile(np.abs(a), .9)) if len(a) else None)


def evaluate_match(y, P, period):
    Q = binary_probabilities(P)
    return evaluate_labels(y, Q[:, 1] >= .5, period)


def evaluate_labels(y, pred, period):
    """Score hard binary decisions without treating them as calibrated probabilities."""
    y, pred, period = np.asarray(y), np.asarray(pred), np.asarray(period)
    if y.ndim != 1 or pred.shape != y.shape or period.shape != y.shape:
        raise ValueError("Labels, periods and predictions must align")
    if not np.isin(pred, (False, True)).all():
        raise ValueError("Expected binary predictions")
    pred = pred.astype(bool)
    valid = np.isin(y, (0, 1, 2))
    cm = np.bincount((y[valid] == 2).astype(int) * 2 + pred[valid], minlength=4).reshape(2, 2)
    result = frame_metrics(cm)
    result["n_invalid_reference"] = int((~valid).sum())
    # Retain scorer codes: LIVE=H=0, DEAD=D=2. H/A changes disappear.
    truth = np.where(y == 2, 2, 0)
    predicted = np.where(pred, 2, 0)
    counts = {str(t): np.zeros(3, dtype=int) for t in IOUS}
    boundaries = {name: {str(t): [0, 0, 0, []] for t in TOLERANCES}
                  for name in ("stoppage_onset", "restart")}
    gt_durations, pred_durations = [], []
    for a, b in valid_spans(y, period):
        runs = [(a, b)]
        gs = [s for s in pm.segments(truth, runs) if s[2] == 2]
        ps = [s for s in pm.segments(predicted, runs) if s[2] == 2]
        gt_durations.extend((s[1] - s[0]) / 25 for s in gs)
        pred_durations.extend((s[1] - s[0]) / 25 for s in ps)
        overlaps = pm.spell_overlaps(gs, ps)
        for t in IOUS:
            counts[str(t)] += pm.match_spells(overlaps, len(gs), len(ps), t)
        gt_trans, pred_trans = pm.transitions(truth, runs), pm.transitions(predicted, runs)
        for name, destination in (("stoppage_onset", 2), ("restart", 0)):
            g = [s for s in gt_trans if s[2] == destination]
            p = [s for s in pred_trans if s[2] == destination]
            for t in TOLERANCES:
                pairs, missed, spurious = pm.match_transitions(g, p, t, typed="pair")
                c = boundaries[name][str(t)]
                c[0] += len(pairs); c[1] += len(spurious); c[2] += len(missed)
                c[3].extend(p[0] - g[0] for g, p in pairs)
    result["dead_intervals"] = {t: prf(*map(int, c)) for t, c in counts.items()}
    result["dead_interval_mean_f1"] = float(np.mean([v["f1"] for v in result["dead_intervals"].values()]))
    result["boundaries"] = {name: {t: boundary_summary(*c) for t, c in values.items()}
                            for name, values in boundaries.items()}
    result["dead_duration"] = dict(reference=duration_distribution(gt_durations),
                                   predicted=duration_distribution(pred_durations))
    result["dead_interval_count_error"] = len(pred_durations) - len(gt_durations)
    return result


def flatten(value, prefix=""):
    """Scalar numeric leaves for CSV and equal-weight match summaries."""
    out = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            out.update(flatten(item, name + "."))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            out[name] = item
    return out


def aggregate(results):
    rows = [flatten(r) for r in results.values()]
    keys = sorted(set().union(*(r.keys() for r in rows)))
    equal = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        equal[k] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        n_matches=len(vals))
    pooled = frame_metrics(np.sum([r["confusion"] for r in results.values()], axis=0))
    # Match locally, then pool counts/errors; never match across matches or spans.
    intervals = {}
    for t in map(str, IOUS):
        c = [sum(r["dead_intervals"][t][k] for r in results.values()) for k in ("tp", "fp", "fn")]
        intervals[t] = prf(*c)
    boundaries = {}
    for name in ("stoppage_onset", "restart"):
        boundaries[name] = {}
        for t in map(str, TOLERANCES):
            items = [r["boundaries"][name][t] for r in results.values()]
            c = [sum(r[k] for r in items) for k in ("tp", "fp", "fn")]
            errors = [e * 25 for r in items for e in r["signed_errors_s"]]
            boundaries[name][t] = boundary_summary(*c, errors)
    durations = {name: duration_distribution([d for r in results.values()
                 for d in r["dead_duration"][name]["durations_s"]]) for name in ("reference", "predicted")}
    return dict(equal_weight_matches=equal, pooled_frames=pooled,
                pooled_dead_intervals=intervals,
                pooled_dead_interval_mean_f1=float(np.mean([v["f1"] for v in intervals.values()])),
                pooled_boundaries=boundaries, pooled_dead_duration=durations)


def load_sets(pipeline):
    return pipeline.load_idsse_sets(pattern=None, matches=MATCHES, advanced=True, control_s=3,
                                   boundary=True, motion=True, augment=True)


def external_posteriors(sets, pipeline):
    """Delegate training to PR #11; held-out match and its mirrors never enter fit."""
    from benchmark import _full_proba
    for held in sets:
        train = [m for m in sets if m != held]
        print("Training fold: " + held, flush=True)
        model = pipeline.fit("lightgbm", sets, matches=train, weights=None)
        yield held, _full_proba(model, sets[held]["X"])


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def source_manifest():
    paths = ["starter/" + p for p in ("live_dead_baseline.py", "features.py", "external.py",
             "external_pipeline.py", "tactical.py", "benchmark.py", "baseline.py")]
    paths += ["games.py", "evaluation/possession_metrics.py", "requirements.txt", "requirements-starter.txt",
              "tests/test_live_dead_baseline.py", "experiments/live_dead_baseline.md"]
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "lightgbm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), source_commit=git("rev-parse", "HEAD"),
                branch=git("branch", "--show-current"), git_status=git("status", "--porcelain"),
                source_sha256={p: sha256(ROOT / p) for p in paths}, python=sys.version,
                platform=platform.platform(), dependencies=versions, config=CONFIG,
                training_params=None, folds={m: [o for o in MATCHES if o != m] for m in MATCHES},
                probability_columns=["H", "A", "D"], binary_columns=["LIVE", "DEAD"],
                status="preflight", data_inputs={})


def check_inputs():
    # Only check explicitly constructed external IDSSE paths; never traverse the data root.
    base = Path(os.environ.get("MTAH_DATA_DIR", str(ROOT / "data"))) / "external" / "idsse"
    inputs, missing = {}, []
    for m in MATCHES:
        cache = base / "cache" / (m + ".npz")
        com = "DFL-COM-000001" if m in MATCHES[:2] else "DFL-COM-000002"
        tag = com + "_DFL-MAT-" + m + ".xml"
        paths = [cache] if cache.is_file() else [base / (stem + tag) for stem in
                  ("DFL_04_03_positions_raw_observed_", "DFL_02_01_matchinformation_")]
        inputs[m] = [str(p) for p in paths]
        missing.extend(str(p) for p in paths if not p.is_file())
    return inputs, missing


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_artifact_checksums(output):
    """Seal completed local artifacts for downstream verification and replay."""
    output = Path(output)
    checksums = {str(path.relative_to(output)): sha256(path)
                 for path in sorted(output.rglob('*'))
                 if path.is_file() and path.name != 'artifact_checksums.json'}
    write_json(output / 'artifact_checksums.json', checksums)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = source_manifest()
    write_json(output / "manifest.json", manifest)
    try:
        inputs, missing = check_inputs()
        manifest["data_inputs"] = inputs
        absent = [p for p, v in manifest["dependencies"].items() if v is None]
        if missing or absent:
            raise RuntimeError("Missing IDSSE files: %d; missing dependencies: %s. No downloads attempted."
                               % (len(missing), ", ".join(absent) or "none"))
        import external_pipeline as ep
        manifest["training_params"] = ep.make_lightgbm().get_params()
        manifest["data_sha256"] = {p: sha256(p) for paths in inputs.values() for p in paths}
        sets = load_sets(ep)
        for m, d in sets.items():
            if d["X"].shape != (len(d["y"]), 78) or not np.isfinite(d["X"]).all():
                raise ValueError("Invalid 78-feature matrix: " + m)
            if not np.isin(d["y"], (0, 1, 2)).all():
                raise ValueError("Invalid external training labels: " + m)
        (output / "posteriors").mkdir()
        results = {}
        for m, P in external_posteriors(sets, ep):
            results[m] = evaluate_match(sets[m]["y"], P, sets[m]["period"])
            np.savez_compressed(output / "posteriors" / (m + ".npz"),
                                P_had=P, y=sets[m]["y"], period=sets[m]["period"])
            write_json(output / "metrics.json", dict(status="running", per_match=results))
        write_json(output / "metrics.json", dict(status="complete", per_match=results, aggregate=aggregate(results)))
        rows = [dict(match=m, **flatten(r)) for m, r in results.items()]
        fields = ["match"] + sorted(set().union(*(r.keys() for r in rows)) - {"match"})
        with open(output / "per_match.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        manifest["status"] = "complete"
        return 0
    except Exception as exc:
        manifest.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        write_json(output / "failure.json", dict(error=manifest["error"]))
        print(manifest["error"], file=sys.stderr)
        return 1
    finally:
        write_json(output / "manifest.json", manifest)
        if manifest['status'] == 'complete':
            write_artifact_checksums(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "data" / "experiment_a" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    args = parser.parse_args()
    return run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
