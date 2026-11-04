import numpy as np

import benchmark as bm


def test_smooth_proba_rows_sum_to_one_and_flatten_flicker():
    P = np.tile([0.9, 0.05, 0.05], (200, 1)); P[100] = [0.05, 0.9, 0.05]
    S = bm.smooth_proba(P, 25)
    assert np.allclose(S.sum(1), 1) and (S.argmax(1) == 0).all()


def test_viterbi_removes_single_frame_flip_but_follows_real_change():
    P = np.tile([0.8, 0.1, 0.1], (400, 1)); P[200] = [0.1, 0.8, 0.1]
    assert (bm.viterbi(P) == 0).all()
    P[200:] = [0.1, 0.8, 0.1]
    path = bm.viterbi(P)
    assert path[0] == 0 and path[-1] == 1


def test_min_duration_function():
    codes = np.array([0] * 60 + [1] * 5 + [0] * 60)
    assert (bm.min_duration(codes, 10) == 0).all()


def test_hierarchical_posteriors_sum_to_one():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(0)
    X = rng.randn(600, 3); y = rng.randint(0, 3, 600)
    m = bm.Hierarchical(LogisticRegression, LogisticRegression).fit(X, y)
    assert np.allclose(m.predict_proba(X).sum(1), 1)


def test_full_proba_pads_missing_classes():
    from sklearn.linear_model import LogisticRegression
    X = np.random.RandomState(0).randn(100, 2); y = (X[:, 0] > 0).astype(int)          # classes 0 and 1 only
    P = bm._full_proba(LogisticRegression().fit(X, y), X)
    assert P.shape == (100, 3) and np.allclose(P[:, 2], 0) and np.allclose(P.sum(1), 1)
