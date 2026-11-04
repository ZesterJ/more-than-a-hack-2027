import numpy as np
import pandas as pd

import external as ext

H, A, D = 0, 1, 2


def _events(rows):
    return pd.DataFrame(rows, columns=["Team", "Type", "Subtype", "Period", "Start Frame", "End Frame"])


def _labels(tmp_path, monkeypatch, rows, n=1000):
    path = tmp_path / "ev.csv"
    _events(rows).to_csv(path, index=False)
    monkeypatch.setattr(ext, "metrica_files", lambda g: {"events": str(path)})
    return ext.derive_labels(1, {"frame": np.arange(1, n + 1)})


def test_owner_follows_latest_event_and_ball_out_is_dead_until_restart(tmp_path, monkeypatch):
    lab = _labels(tmp_path, monkeypatch, [
        ["Home", "SET PIECE", "KICK OFF", 1, 1, 1],
        ["Home", "PASS", None, 1, 100, 120],
        ["Away", "RECOVERY", "INTERCEPTION", 1, 300, 300],
        ["Away", "BALL OUT", None, 1, 500, 500],
        ["Home", "SET PIECE", "THROW IN", 1, 600, 600],
    ])
    assert lab[50] == H and lab[350] == A
    assert lab[550] == D and lab[700] == H


def test_goal_is_dead_until_kick_off(tmp_path, monkeypatch):
    lab = _labels(tmp_path, monkeypatch, [
        ["Home", "SET PIECE", "KICK OFF", 1, 1, 1],
        ["Home", "SHOT", "ON TARGET-GOAL", 1, 200, 220],
        ["Away", "SET PIECE", "KICK OFF", 1, 400, 400],
    ])
    assert lab[100] == H and lab[300] == D and lab[500] == A


def test_orientation_mirrors_when_home_keeper_is_on_the_right():
    n = 50
    P = np.full((n, 4, 3), 0.5, dtype=np.float32)
    P[:, 0, 0] = 0.95; P[:, 1, 0] = 0.55        # home keeper near the right goal line
    P[:, 2, 0] = 0.05; P[:, 3, 0] = 0.45
    trk = {"players": P, "ball": np.full((n, 3), 0.7, dtype=np.float32)}
    out = ext.orient_home_attacks_right(trk, np.array([True, True, False, False]))
    assert np.isclose(out["players"][0, 0, 0], 0.05) and np.isclose(out["ball"][0, 0], 0.3)
    again = ext.orient_home_attacks_right(out, np.array([True, True, False, False]))
    assert np.allclose(again["players"][..., 0], out["players"][..., 0])


def test_apply_occlusion_tiles_pattern():
    trk = {"ball": np.ones((10, 3), dtype=np.float32)}
    out = ext.apply_occlusion(trk, np.array([True, False, False]))
    assert np.isnan(out["ball"][[0, 3, 6, 9], 0]).all() and not np.isnan(out["ball"][[1, 2, 4], 0]).any()
