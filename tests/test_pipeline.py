"""The external-only pipeline: logic on synthetic matches, plus a guard that it never touches the ground truth."""
import os
import re

import numpy as np

import external_pipeline as ep

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fake_match(seed, n=6000):
    """A match whose label follows the first feature, in long runs, so any sane model can learn it."""
    rng = np.random.RandomState(seed)
    y = np.repeat(rng.randint(0, 3, n // 250), 250).astype(np.int8)
    X = rng.randn(n, 30).astype(np.float32); X[:, 0] += 3.0 * (y - 1)
    aux = dict(d_min=np.full(n, 20.0), ball_missing=np.zeros(n), speed_all_roll_2s=np.full(n, 2.0), ball_accel=np.zeros(n))
    return dict(X=X, aux=aux, y=y, period=np.ones(n, dtype=np.int8))


def _sets(k=3):
    return {"m%d" % i: _fake_match(i) for i in range(k)}


def test_lomo_posteriors_cover_every_match_and_sum_to_one():
    sets = _sets()
    posts = ep.lomo_posteriors("logreg", sets)
    assert set(posts) == set(sets)
    for m, P in posts.items():
        assert P.shape == (len(sets[m]["y"]), 3) and np.allclose(P.sum(1), 1)


def test_fit_with_a_subset_ignores_the_other_matches():
    sets = _sets()
    model = ep.fit("logreg", sets, matches=["m0"])
    other = ep.fit("logreg", {"m0": sets["m0"]})
    assert np.allclose(model.predict_proba(sets["m2"]["X"]), other.predict_proba(sets["m2"]["X"]))


def test_tune_decoder_returns_grid_member_and_a_row_per_match():
    sets = _sets()
    posts = ep.lomo_posteriors("logreg", sets)
    small = [dict(w=25, min_s=0, contact_m=None, inflight_thr=None), dict(w=25, min_s=2, contact_m=None, inflight_thr=None)]
    joint, table = ep.tune_decoder(posts, sets, grid=small)
    assert joint in small and [r["match"] for r in table] == list(sets)
    assert all(0.0 <= r["tuned_on_others"] <= 1.0 for r in table)


def test_default_grid_never_pairs_contact_filter_with_zero_duration():
    assert all(not (p["min_s"] == 0 and p["contact_m"]) for p in ep.GRID)


def test_unobservable_ball_features_are_dropped():
    assert set(ep.UNOBSERVED) == {"ball_speed", "ball_accel"}


GT_PATTERNS = re.compile(r"load_gt|possession_gt|gt_public|PUBLIC_GT|LABEL_STOP|read_label_csv")


def test_pipeline_files_and_notebook_do_not_touch_the_ground_truth():
    """Rules: never train, fit or tune on the ground-truth possession data."""
    guarded = ["starter/external_pipeline.py", "starter/generate_full_submission.py", "starter/external.py",
               "starter/tactical.py", "starter/04_external_benchmark.ipynb"]
    for rel in guarded:
        text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        hits = [m.group(0) for m in GT_PATTERNS.finditer(text)]
        assert not hits, "%s references the ground truth: %s" % (rel, hits)


def test_shift_looks_back_and_ahead_with_edge_fill():
    import numpy as np
    import external_pipeline as ep
    a = np.arange(6.0)
    assert ep._shift(a, 2).tolist() == [0, 0, 0, 1, 2, 3]     # a[t-2]
    assert ep._shift(a, -2).tolist() == [2, 3, 4, 5, 5, 5]    # a[t+2]
    assert ep._shift(a, 0).tolist() == a.tolist()


def _toy_tracking(n=40):
    import numpy as np
    rng = np.random.RandomState(0)
    return dict(players=rng.rand(n, 6, 3).astype(np.float32), ball=rng.rand(n, 3).astype(np.float32),
                is_home=np.array([True, True, True, False, False, False]),
                labels=rng.randint(0, 3, n).astype(np.int8))


def test_mirroring_twice_is_the_identity_and_swaps_team_labels_once():
    import numpy as np
    import external_pipeline as ep
    t = _toy_tracking()
    for mode in ("y", "xswap"):
        back = ep.mirror_tracking(ep.mirror_tracking(t, mode), mode)
        assert np.allclose(back["players"], t["players"]) and np.allclose(back["ball"], t["ball"])
        assert (back["labels"] == t["labels"]).all() and (back["is_home"] == t["is_home"]).all()
    once = ep.mirror_tracking(t, "xswap")
    assert ((once["labels"] == 1) == (t["labels"] == 0)).all()      # H <-> A
    assert ((once["labels"] == 2) == (t["labels"] == 2)).all()      # dead ball stays dead
    assert (once["is_home"] == ~t["is_home"]).all()


def test_domain_weights_are_clipped_and_average_one_without_labels():
    import numpy as np
    import external_pipeline as ep
    rng = np.random.RandomState(0)
    sets = {m: dict(X=rng.randn(900, 5) + 0.2 * i) for i, m in enumerate("abc")}   # no labels anywhere
    w = ep.domain_weights(sets, rng.randn(600, 5) + 0.5)
    for m in sets:
        assert abs(w[m].mean() - 1.0) < 1e-6 and w[m].min() > 0


def test_window_mean_and_temporal_block_are_finite_and_causal_where_named():
    import numpy as np
    import external_pipeline as ep
    a = np.arange(10.0)
    assert ep._window_mean(a, -2, 0)[5] == 4.0            # mean of a[3..5]
    assert ep._window_mean(a, 0, 2)[5] == 6.0             # mean of a[5..7]
    assert ep._window_mean(np.array([np.nan, 2.0, 4.0]), -1, 1)[0] == 2.0   # NaN-aware, edge truncated
    assert len(ep.TEMPORAL) == 31 and len(set(ep.TEMPORAL)) == 31


def test_setpiece_block_reads_players_only():
    import numpy as np
    import external_pipeline as ep
    n = 60
    P = np.full((n, 4, 3), np.nan, dtype=np.float32)
    P[:, 0] = (0.30, 0.50, 0.2)            # home, own half, standing still
    P[:, 1] = (0.40, 0.001, 0.1)           # home, on the touchline
    P[:, 2] = (0.70, 0.50, 6.0)            # away, own half, sprinting
    trk = dict(players=P, ball=np.full((n, 3), np.nan, dtype=np.float32))   # the ball is never seen
    B = ep.setpiece_block(trk, np.array([True, True, False, False]), None, None)
    col = {k: B[:, i] for i, k in enumerate(ep.SETPIECE)}
    assert B.shape == (n, len(ep.SETPIECE)) and np.isfinite(B).all()
    assert col["sp_touchline_d"][0] < 0.1 and abs(col["sp_touchline_spd"][0] - 0.1) < 1e-6
    assert col["sp_own_half_share"][0] == 1.0 and col["sp_n_slow"][0] == 2.0
    assert abs(col["sp_speed_max"][0] - 6.0) < 1e-6


def test_restart_block_counts_within_each_half():
    import numpy as np
    import external_pipeline as ep
    period = np.r_[np.ones(500), 2 * np.ones(500)]
    B = ep.restart_block(1000, [100, 600], period)
    since, until = B[:, 0], B[:, 1]
    assert abs(since[125] - 1.0) < 1e-6 and until[50] == 2.0
    assert until[400] == ep.RESTART_CAP_S            # the half-two restart is not visible from half one
    assert abs(until[500] - 4.0) < 1e-6 and since[500] == ep.RESTART_CAP_S
