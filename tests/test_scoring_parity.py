"""Anchors against the official scorer; the last one needs the gated match data."""
import numpy as np

import features as ft
from benchmark import score
from conftest import needs_atd


def test_perfect_prediction_scores_one_and_all_abstain_scores_zero():
    gt = ft.load_gt()
    window = gt[ft.LABEL_START:ft.LABEL_STOP]
    perfect = score(window, gt)
    assert perfect["spell_f1_mean"] == 1.0 and perfect["macro_f1"] == 1.0
    assert perfect["n_gt_spells"] == 17
    nothing = score(np.full(len(window), 2, dtype=np.int8), gt)          # constant dead ball
    assert nothing["spell_f1_mean"] == 0.0


@needs_atd
def test_nearest_player_baseline_matches_the_published_raw_row():
    from benchmark import row
    trk = ft.load_tracking(ft.LABEL_START + 7500)
    X = ft.build_features(trk)
    codes = ft.nearest_player_codes(X)[ft.LABEL_START:ft.LABEL_STOP]
    gt = ft.load_gt()
    r = row("raw", score(np.where(codes < 0, 2, codes), gt))
    assert r["spells"] == 131 and abs(r["spellF1"] - 0.0378) < 5e-4
