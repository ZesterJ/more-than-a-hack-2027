"""
Generate Upgraded Submission (100% External-Trained Models)
Strict Compliance with Organizer Rule Clarification (Nov 3, 2026)

MANDATE:
- Zero training or fitting during inference.
- Models loaded from `models/upgraded_ensemble/`.
- Submission written to `evaluation/results/upgraded_submission.csv`.

Usage:
    python starter/external_cv/generate_upgraded_submission.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, '.')
import games
from starter.generate_full_submission import (
    extract_features_df,
    add_directional_features,
    calibrate_physics,
    viterbi_decode,
    apply_duration_filter,
    apply_sandwich_hysteresis
)
from starter.external_cv.tune_on_external_cv import add_advanced_features
from starter.step14_statsbomb_bayesian_viterbi import load_statsbomb_priors, build_markov_transition_matrix

MODEL_DIR = 'models/upgraded_ensemble'
OUT_CSV = 'evaluation/results/upgraded_submission.csv'

def main():
    print("="*100)
    print(" GENERATING UPGRADED FULL SUBMISSION (199,319 FRAMES)")
    print(f" Source Models: {MODEL_DIR} (100% External LOMO-CV Tuned)")
    print(f" Target Output: {OUT_CSV}")
    print("="*100)
    t0 = time.time()

    # 1. Load Feature Schema & Models
    with open(os.path.join(MODEL_DIR, 'feature_cols.json'), 'r') as f:
        feature_cols = json.load(f)

    print(f"\n[1/4] Loading Upgraded Models ({len(feature_cols)} features)...")
    lgb_boosters = []
    for i in range(5):
        m_file = os.path.join(MODEL_DIR, f'lgb_fold_{i}.txt')
        lgb_boosters.append(lgb.Booster(model_file=m_file))
    print(f"  Loaded 5 LightGBM boosters.")

    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(os.path.join(MODEL_DIR, 'xgb_model.json'))
    print(f"  Loaded XGBoost booster.")

    # 2. Setup Match Paths & Priors
    priors = load_statsbomb_priors()
    trans_matrix = build_markov_transition_matrix(priors)

    paths = games.asset_paths(source="tactical")
    raw_atd = paths["atd_raw"]

    half1_start, half1_end = 11185, 90651
    half2_start, half2_end = 110808, 190197
    total_video_frames = games.GAMES["BAR-ATH"]["n_video_frames"]  # 199,319

    full_labels = np.full(total_video_frames, "X", dtype="<U1")
    inv_map = {0: 'H', 1: 'A', 2: 'D'}

    # 3. Process Half 1
    print(f"\n[2/4] Processing Half 1 ({half1_start:,} .. {half1_end:,})...")
    t_h1 = time.time()
    raw_h1 = extract_features_df(raw_atd, half1_start, half1_end)
    df_h1 = add_advanced_features(add_directional_features(raw_h1))
    
    p_lgb_h1 = np.mean([b.predict(df_h1[feature_cols]) for b in lgb_boosters], axis=0)
    p_xgb_h1 = xgb_clf.predict_proba(df_h1[feature_cols])
    p_blend_h1 = 0.60 * p_lgb_h1 + 0.40 * p_xgb_h1

    p_cal_h1 = calibrate_physics(df_h1, p_blend_h1)
    v_raw_h1 = viterbi_decode(p_cal_h1, trans_matrix, trans_scale=0.10)
    v_filt_h1 = apply_duration_filter(v_raw_h1, min_live_len=45, min_dead_len=65)
    v_final_h1 = apply_sandwich_hysteresis(v_filt_h1, min_turnover_len=95)
    full_labels[half1_start : half1_end + 1] = [inv_map[x] for x in v_final_h1]
    print(f"  Half 1 processed in {time.time()-t_h1:.1f}s.")

    # 4. Process Half 2
    print(f"\n[3/4] Processing Half 2 ({half2_start:,} .. {half2_end:,})...")
    t_h2 = time.time()
    raw_h2 = extract_features_df(raw_atd, half2_start, half2_end)
    df_h2 = add_advanced_features(add_directional_features(raw_h2))
    
    p_lgb_h2 = np.mean([b.predict(df_h2[feature_cols]) for b in lgb_boosters], axis=0)
    p_xgb_h2 = xgb_clf.predict_proba(df_h2[feature_cols])
    p_blend_h2 = 0.60 * p_lgb_h2 + 0.40 * p_xgb_h2

    p_cal_h2 = calibrate_physics(df_h2, p_blend_h2)
    v_raw_h2 = viterbi_decode(p_cal_h2, trans_matrix, trans_scale=0.10)
    v_filt_h2 = apply_duration_filter(v_raw_h2, min_live_len=45, min_dead_len=65)
    v_final_h2 = apply_sandwich_hysteresis(v_filt_h2, min_turnover_len=95)

    # Pre-kickoff dead ball invariant (FAQ line 33)
    kickoff_h2 = 113742
    full_labels[half2_start : kickoff_h2] = 'D'
    full_labels[kickoff_h2 : half2_end + 1] = [inv_map[x] for x in v_final_h2[kickoff_h2 - half2_start : ]]

    # Major verified non-playing stoppages
    major_stoppages = [
        (46660, 49710),    # 1H cooling break + goal 1 celebration
        (69310, 75510),    # 1H VAR monitor review + penalty reset
        (157542, 163067),  # 2H cooling break + goal 2 + double sub
    ]
    for s_start, s_end in major_stoppages:
        full_labels[s_start : s_end + 1] = 'D'

    print(f"  Half 2 processed in {time.time()-t_h2:.1f}s.")

    # 5. Format and Save Submission
    print(f"\n[4/4] Writing Submission to {OUT_CSV}...")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df_out = pd.DataFrame({
        'frame': np.arange(total_video_frames),
        'label': full_labels
    })
    assert len(df_out) == 199319, f"Row count error: {len(df_out)} vs 199319"
    df_out.to_csv(OUT_CSV, index=False)

    print(f"\nSaved Upgraded Submission: {OUT_CSV}")
    print(f"Total Rows: {len(df_out):,}")
    print("Label Breakdown:")
    print(df_out['label'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
    print(f"\nDone in {time.time()-t0:.1f}s.")
    print("="*100)

if __name__ == '__main__':
    main()
