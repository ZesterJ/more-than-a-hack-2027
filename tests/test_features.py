import numpy as np

import features as ft


def _trk(n=200):
    """Home slots 0-1, away slots 2-3; the ball sits on home player 0."""
    P = np.full((n, 4, 3), np.nan, dtype=np.float32)
    P[:, 0] = [0.50, 0.50, 1.0]; P[:, 1] = [0.30, 0.50, 1.0]
    P[:, 2] = [0.60, 0.50, 1.0]; P[:, 3] = [0.80, 0.50, 1.0]
    B = np.tile(np.array([0.505, 0.50, np.nan], dtype=np.float32), (n, 1))
    return {"players": P, "ball": B}, np.array([True, True, False, False])


def test_rolling_mean_is_nan_aware():
    v = np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan])
    r = ft._rolling_mean(v, 3)
    assert r[1] == 2.0 and np.isnan(r[5])


def test_build_features_shape_finite_and_margin_sign():
    trk, is_home = _trk()
    X = ft.build_features(trk, is_home=is_home)
    assert list(X.columns) == ft.FEATURES and np.isfinite(X.values).all()
    assert (X["d_margin"] > 0).all(), "home player is closest, so away-minus-home distance must be positive"
    assert (X["ball_missing"] == 0).all()
    codes = ft.nearest_player_codes(X)
    assert (codes == 0).all()


def test_missing_ball_is_flagged_and_stays_finite():
    trk, is_home = _trk()
    trk["ball"][:] = np.nan
    X = ft.build_features(trk, is_home=is_home)
    assert (X["ball_missing"] == 1).all() and np.isfinite(X.values).all()
    assert (ft.nearest_player_codes(X) == -1).all()


def test_build_extra_columns_and_contact_distance():
    trk, is_home = _trk()
    X = ft.build_features(trk, is_home=is_home)
    E = ft.build_extra(trk, X)
    assert list(E.columns) == ft.EXTRA_FEATURES and np.isfinite(E.values).all()
    assert (E["d_min"] < 1.0).all()                       # ball is ~0.5 m from home player 0
