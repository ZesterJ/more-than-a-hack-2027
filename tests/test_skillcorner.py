"""The SkillCorner loader, checked on a tiny synthetic match (no downloaded data needed)."""
import csv
import json
import os

import numpy as np

import external as ext


def _write_match(root, match="1"):
    d = os.path.join(root, match)
    os.makedirs(d)
    players = [dict(id=pid, team_id=tid) for pid, tid in ((1, 10), (2, 10), (3, 20), (4, 20))]
    json.dump(dict(home_team=dict(id=10), players=players), open(os.path.join(d, match + "_match.json"), "w"))
    x = {1: -10.0, 2: -20.0, 3: 10.0, 4: 20.0}
    with open(os.path.join(d, match + "_tracking_extrapolated.jsonl"), "w") as f:
        for i in range(12):
            pd_ = [dict(player_id=pid, x=x[pid], y=0.0, is_detected=pid != 4) for pid in (3, 1, 4, 2)]  # away listed first
            f.write(json.dumps(dict(frame=i, period=1, ball_data=dict(x=0.0, y=0.0, is_detected=i % 2 == 0),
                                    possession=None, player_data=pd_)) + "\n")
    with open(os.path.join(d, match + "_phases_of_play.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_start", "frame_end", "team_in_possession_id"])
        w.writeheader()
        w.writerow(dict(frame_start=2, frame_end=7, team_in_possession_id=10))


def test_loader_orders_home_first_resamples_to_25fps_and_labels_from_phases(tmp_path, monkeypatch):
    monkeypatch.setattr(ext, "SKILLCORNER_DIR", str(tmp_path))
    _write_match(str(tmp_path))
    t = ext.load_skillcorner("1", cache=False)
    assert t["is_home"].tolist() == [True, True, False, False]
    # slot 0 is home player 1 (x = -10 m -> (x + 52.5) / 105), slot 2 is away player 3 (x = +10 m)
    assert np.allclose(t["players"][2, 0, 0], (-10 + 52.5) / 105, atol=1e-5)
    assert np.allclose(t["players"][2, 2, 0], (10 + 52.5) / 105, atol=1e-5)
    assert np.isnan(t["players"][2, 3, 0])                         # the undetected away player stays missing
    assert not np.isnan(ext.load_skillcorner("1", cache=False, extrapolate=True)["players"][2, 3, 0])
    assert abs(len(t["labels"]) - 11 * 2.5) <= 2                  # 10 fps -> 25 fps
    y = t["labels"]
    assert (y == ext.D).any() and (y == ext.H).any() and not (y == ext.A).any()
    assert y[int(4.5 * 2.5)] == ext.H and y[0] == ext.D            # inside the phase: home; before it: dead ball
