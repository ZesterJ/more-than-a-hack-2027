import numpy as np

from tactical import D, decode_pipeline, duration_filter, inflight_inertia, snap_to_acceleration

H, A = 0, 1


def runs(codes):
    out, start = [], 0
    for i in range(1, len(codes) + 1):
        if i == len(codes) or codes[i] != codes[start]:
            out.append((int(codes[start]), i - start)); start = i
    return out


def test_duration_filter_absorbs_short_run_into_longer_neighbour():
    codes = np.array([H] * 100 + [A] * 10 + [H] * 50)
    assert runs(duration_filter(codes, 25)) == [(H, 160)]


def test_duration_filter_keeps_runs_long_enough():
    codes = np.array([H] * 100 + [A] * 30 + [H] * 50)
    assert runs(duration_filter(codes, 25)) == [(H, 100), (A, 30), (H, 50)]


def test_contact_conditioned_filter_keeps_short_run_with_contact():
    codes = np.array([H] * 100 + [A] * 10 + [H] * 50)
    d_min = np.full(len(codes), 20.0); d_min[105] = 0.5          # a player touched the ball during the A run
    assert runs(duration_filter(codes, 25, d_min, 1.1)) == [(H, 100), (A, 10), (H, 50)]
    d_min[105] = 5.0                                              # no contact: unconfirmed noise is merged
    assert runs(duration_filter(codes, 25, d_min, 1.1)) == [(H, 160)]


def test_inflight_inertia_hands_lofted_pass_back_to_passing_team():
    codes = np.array([H] * 50 + [D] * 20 + [H] * 50)
    missing, speed = np.ones(len(codes)), np.full(len(codes), 3.0)
    assert runs(inflight_inertia(codes, missing, speed, 1.2, 150)) == [(H, 120)]


def test_inflight_inertia_leaves_real_stoppage_alone():
    codes = np.array([H] * 50 + [D] * 20 + [A] * 50)
    missing, speed = np.ones(len(codes)), np.full(len(codes), 0.4)     # everyone walking
    assert runs(inflight_inertia(codes, missing, speed, 1.2, 150))[1] == (D, 20)


def test_snap_moves_transition_to_acceleration_peak():
    codes = np.array([H] * 50 + [A] * 50)
    accel = np.zeros(100); accel[53] = 9.0
    assert runs(snap_to_acceleration(codes, accel, 5)) == [(H, 53), (A, 47)]
    assert runs(snap_to_acceleration(codes, accel, 0)) == [(H, 50), (A, 50)]


def test_decode_pipeline_default_is_plain_smoothing():
    rng = np.random.RandomState(0)
    P = rng.dirichlet(np.ones(3), size=300)
    from benchmark import smooth_proba
    assert np.array_equal(decode_pipeline(P, {}, w=25), smooth_proba(P, 25).argmax(1))
