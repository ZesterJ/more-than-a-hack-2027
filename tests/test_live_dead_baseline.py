"""Synthetic Experiment A tests; runnable without pytest or ML dependencies."""
import builtins
import importlib.util
import json
import io
import ast
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "starter"), str(ROOT / "evaluation")]
import live_dead_baseline as lb
from benchmark import _full_proba


def probabilities(dead):
    dead = np.asarray(dead, dtype=float)
    return np.column_stack(((1 - dead) / 2, (1 - dead) / 2, dead))


def evaluate(y, dead, periods=None):
    return lb.evaluate_match(np.array(y), probabilities(dead),
                             np.ones(len(y)) if periods is None else periods)


class BaselineTests(unittest.TestCase):
    def test_probability_collapse_and_argmax_counterexample(self):
        q = lb.binary_probabilities([[.3, .3, .4], [.1, .1, .8]])
        np.testing.assert_allclose(q, [[.6, .4], [.2, .8]])
        np.testing.assert_allclose(q.sum(1), 1)

    def test_probability_alignment_missing_class(self):
        model = types.SimpleNamespace(classes_=np.array([2, 0]),
                                      predict_proba=lambda X: np.tile([.4, .6], (len(X), 1)))
        np.testing.assert_allclose(_full_proba(model, np.zeros((2, 1))), [[.6, 0, .4]] * 2)

    def test_bad_probabilities_and_lengths(self):
        for p in ([[.1, .1]], [[.1, .1, .1]], [[-1, 1, 1]], [[np.nan, 0, 1]], [[2, 0, -1]]):
            with self.assertRaises(ValueError):
                lb.binary_probabilities(p)
        with self.assertRaises(ValueError):
            lb.evaluate_match([0], probabilities([0, 1]), [1])

    def test_frame_metrics_and_tie(self):
        r = evaluate([0, 1, 2, 2], [.1, .5, .8, .1])
        self.assertEqual(r["confusion"], [[1, 1], [1, 1]])
        self.assertEqual(r["support"], dict(live=2, dead=2))
        self.assertEqual(r["dead"]["f1"], .5)
        self.assertEqual(r["live"]["precision"], .5)
        self.assertEqual(r["accuracy"], .5)
        self.assertEqual(r["macro_f1"], .5)

    def test_absent_class_and_empty_input(self):
        self.assertEqual(evaluate([0, 0], [0, 0])["dead"]["f1"], 0)
        r = lb.evaluate_match(np.array([], dtype=int), np.empty((0, 3)), np.array([]))
        self.assertEqual(r["n_scored"], 0)
        self.assertIsNone(r["dead_duration"]["predicted"]["mean_s"])

    def test_perfect_intervals_and_binary_team_changes(self):
        r = evaluate([0, 1, 2, 2, 0], [0, 0, 1, 1, 0])
        self.assertEqual(r["dead_intervals"]["0.5"]["f1"], 1)
        self.assertEqual(r["boundaries"]["stoppage_onset"]["12"]["tp"], 1)
        self.assertEqual(r["boundaries"]["restart"]["12"]["tp"], 1)
        self.assertEqual(r["dead_duration"]["reference"]["durations_s"], [.08])

    def test_split_interval_is_one_to_one(self):
        r = evaluate([0] + [2] * 10 + [0], [0] + [1] * 4 + [0] * 2 + [1] * 4 + [0])
        self.assertEqual(r["dead_intervals"]["0.3"]["tp"], 1)
        self.assertEqual(r["dead_intervals"]["0.3"]["fp"], 1)
        self.assertEqual(r["dead_intervals"]["0.5"]["tp"], 0)
        self.assertEqual(r["dead_interval_count_error"], 1)

    def test_merged_and_missed_intervals(self):
        r = evaluate([2] * 4 + [0] * 2 + [2] * 4, [1] * 10)
        self.assertEqual(r["dead_intervals"]["0.3"]["tp"], 1)
        self.assertEqual(r["dead_intervals"]["0.3"]["fn"], 1)
        self.assertEqual(evaluate([2, 2], [0, 0])["dead_intervals"]["0.5"]["fn"], 1)

    def test_period_and_invalid_spans(self):
        r = evaluate([2, 2, 3, 2, 2], [1] * 5, [1, 2, 2, 2, 2])
        self.assertEqual(r["n_scored"], 4)
        self.assertEqual(r["dead_duration"]["reference"]["count"], 3)
        self.assertEqual(r["boundaries"]["stoppage_onset"]["25"]["tp"], 0)
        # Opposite boundary errors in adjacent periods must not cancel by matching.
        r = evaluate([0, 2, 0, 0], [0, 0, 0, 1], [1, 1, 2, 2])
        b = r["boundaries"]["stoppage_onset"]["25"]
        self.assertEqual((b["tp"], b["fp"], b["fn"]), (0, 1, 1))
        r = evaluate([0, 2, 3, 0, 0], [0, 0, 0, 0, 1])
        self.assertEqual(r["boundaries"]["stoppage_onset"]["25"]["tp"], 0)

    def test_timing_sign_tolerance_and_unmatched_errors(self):
        y = [0] * 30 + [2] * 40 + [0] * 30
        r = evaluate(y, [0] * 43 + [1] * 40 + [0] * 17)
        onset = r["boundaries"]["stoppage_onset"]
        self.assertEqual(onset["12"]["tp"], 0)
        self.assertIsNone(onset["12"]["mean_signed_error_s"])
        self.assertEqual(onset["25"]["mean_signed_error_s"], 13 / 25)
        early = evaluate(y, [0] * 28 + [1] * 40 + [0] * 32)
        self.assertEqual(early["boundaries"]["restart"]["12"]["mean_signed_error_s"], -2 / 25)
        self.assertEqual(early["boundaries"]["restart"]["12"]["mean_absolute_error_s"], 2 / 25)

    def test_equal_weight_vs_pooled(self):
        a = evaluate([2], [1])
        b = evaluate([2] * 9, [0] * 9)
        r = lb.aggregate(dict(a=a, b=b))
        self.assertEqual(r["equal_weight_matches"]["accuracy"]["mean"], .5)
        self.assertEqual(r["pooled_frames"]["accuracy"], .1)
        self.assertEqual(r["pooled_dead_intervals"]["0.5"]["fn"], 1)
        self.assertEqual(r["pooled_dead_duration"]["reference"]["count"], 2)

    def test_exact_loader_configuration(self):
        from unittest.mock import Mock
        pipeline = types.SimpleNamespace(load_idsse_sets=Mock(return_value={}))
        lb.load_sets(pipeline)
        pipeline.load_idsse_sets.assert_called_once_with(pattern=None, matches=lb.MATCHES,
            advanced=True, control_s=3, boundary=True, motion=True, augment=True)

    def test_fold_isolation_and_no_weights(self):
        sets = {m: dict(X=np.full((12, 78), i), y=np.zeros(12),
                        aug=[(np.full((12, 78), i + 10), np.zeros(12))])
                for i, m in enumerate(("a", "b", "c"))}
        calls = []
        def fit(name, received, matches, weights):
            self.assertEqual(name, "lightgbm"); self.assertIsNone(weights)
            held = next(m for m in sets if m not in matches)
            self.assertEqual(len(matches), 2)
            used = [received[m]["X"] for m in matches] + [x for m in matches for x, _ in received[m]["aug"]]
            self.assertTrue(all(not np.array_equal(x, received[held]["X"]) for x in used))
            self.assertTrue(all(not np.array_equal(x, received[held]["aug"][0][0]) for x in used))
            calls.append((held, matches))
            return types.SimpleNamespace(classes_=np.arange(3), predict_proba=lambda X: probabilities(np.zeros(len(X))))
        posts = dict(lb.external_posteriors(sets, types.SimpleNamespace(fit=fit)))
        self.assertEqual(set(posts), set(sets)); self.assertEqual(len(calls), 3)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn unavailable; real feature-module import requires it")
    def test_real_feature_matrix_78_finite_and_control_preserves_dead(self):
        import external_pipeline as ep
        rng = np.random.RandomState(0)
        trk = dict(players=rng.rand(200, 6, 3).astype(np.float32), ball=rng.rand(200, 3).astype(np.float32))
        home = np.array([True] * 3 + [False] * 3)
        X, _ = ep.featureset(trk, home, advanced=True, boundary=True, motion=True)
        self.assertEqual(X.shape, (200, 78)); self.assertTrue(np.isfinite(X).all())
        from tactical import controlled_possession
        y = np.array([0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(controlled_possession(y, 75) == 2, y == 2)

    def test_competition_access_guard_and_artifacts(self):
        # Exercise the entire orchestration against synthetic external inputs.
        # All opens except external/output/code paths are denied, including challenge assets.
        from unittest.mock import Mock
        n = 200
        fake = {m: dict(X=np.zeros((n, 78)), y=np.tile([0, 1, 2, 2], 50),
                        period=np.repeat([1, 2], 100), aug=[]) for m in lb.MATCHES}
        pipeline = types.SimpleNamespace(load_idsse_sets=Mock(return_value=fake),
            make_lightgbm=lambda: types.SimpleNamespace(get_params=lambda: {}),
            fit=lambda *a, **k: types.SimpleNamespace(classes_=np.arange(3),
                predict_proba=lambda X: probabilities(np.tile([0, 0, 1, 1], 50))))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            original = builtins.open
            original_io = io.open
            def guarded(path, *args, **kwargs):
                if isinstance(path, (str, Path)):
                    p = Path(path).resolve()
                    if p.suffix not in (".py", ".pyc") and not p.is_relative_to(Path(tmp).resolve()):
                        raise AssertionError("Non-external access: " + str(p))
                return original(path, *args, **kwargs)
            def guarded_io(path, *args, **kwargs):
                guarded(path, "rb").close() if isinstance(path, (str, Path)) and Path(path).exists() else None
                if isinstance(path, (str, Path)) and not Path(path).resolve().is_relative_to(Path(tmp).resolve()):
                    raise AssertionError("Unexpected IO path: " + str(path))
                return original_io(path, *args, **kwargs)
            with patch.dict(sys.modules, external_pipeline=pipeline), patch.object(lb, "source_manifest", return_value={"dependencies": {}}), patch.object(lb, "check_inputs", return_value=({}, [])), patch("builtins.open", side_effect=guarded), patch("io.open", side_effect=guarded_io):
                self.assertEqual(lb.run(output), 0)
            self.assertEqual(json.loads((output / "metrics.json").read_text())["status"], "complete")
            self.assertEqual(len(list((output / "posteriors").glob("*.npz"))), 7)
            self.assertTrue((output / "per_match.csv").is_file())
            checks = json.loads((output / "artifact_checksums.json").read_text())
            for rel in ("manifest.json", "metrics.json", "posteriors/J03WMX.npz"):
                self.assertEqual(checks[rel], lb.sha256(output / rel))

    def test_entrypoint_has_no_competition_or_decoder_calls(self):
        tree = ast.parse(Path(lb.__file__).read_text())
        calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
                 for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, (ast.Name, ast.Attribute))}
        forbidden = {"load_tracking", "load_gt", "read_label_csv", "require", "asset_paths",
                     "ball_unseen_pattern", "target_features", "domain_weights", "decode_pipeline",
                     "tune_decoder", "smooth_proba", "fetch_idsse", "predict_and_write"}
        self.assertFalse(calls & forbidden)

    def test_controlled_labels_preserve_binary_reference(self):
        from tactical import controlled_possession
        y = np.array([0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(controlled_possession(y, 75) == 2, y == 2)

    def test_preflight_missing_dataset_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"MTAH_DATA_DIR": tmp}):
                inputs, missing = lb.check_inputs()
            self.assertEqual(len(inputs), 7); self.assertEqual(len(missing), 14)
            output = Path(tmp) / "failed"
            with patch.object(lb, "source_manifest", return_value={"dependencies": {}}), patch.object(lb, "check_inputs", return_value=(inputs, missing)):
                self.assertEqual(lb.run(output), 1)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "failed")
            self.assertFalse((output / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
