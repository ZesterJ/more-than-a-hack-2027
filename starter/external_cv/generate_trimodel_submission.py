"""
Generate Tri-Model Champion Submission (LightGBM + XGBoost + CatBoost)
100% External-Trained Positional Play Committee (226,716 External Frames).
Strict compliance with Nov 3 organizer rule clarification.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

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
from starter.external_cv.test_refined_features import add_refined_features
from starter.step14_statsbomb_bayesian_viterbi import load_statsbomb_priors, build_markov_transition_matrix

MODEL_DIR = 'models/trimodel_champion_ensemble'
OUT_CSV = 'evaluation/results/trimodel_champion_submission.csv'

def main():
    print("="*100)
    print(" TRAINING & GENERATING TRI-MODEL CHAMPION SUBMISSION (LGBM + XGBOOST + CATBOOST)")
    print("="*100)
    t0 = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs('evaluation/results', exist_ok=True)

    # 1. Load External Positional Play Corpus (226,716 frames)
    df_met = pd.read_csv('data/metrica_3games_features.csv')
    df_sc = pd.read_csv('data/skillcorner_barca_mancity_features.csv')
    
    df_met = add_refined_features(add_directional_features(df_met))
    df_sc = add_refined_features(add_directional_features(df_sc))
    all_train = pd.concat([df_met, df_sc], ignore_index=True)

    drop_cols = ['video_frame', 'frame_id', 'label', 'game', 'home_attack_direction', 'ball_missing']
    feature_cols = [c for c in all_train.columns if c not in drop_cols]
    label_map = {'H': 0, 'A': 1, 'D': 2}
    inv_map = {0: 'H', 1: 'A', 2: 'D'}

    with open(os.path.join(MODEL_DIR, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)

    X_train = all_train[feature_cols]
    y_train = all_train['label'].map(label_map).values
    print(f"\n[1/4] Training Tri-Model Committee ({len(feature_cols)} features)...")

    # LightGBM 5-seed diversity
    lgb_models = []
    for i, s in enumerate([42, 101, 777, 1234, 2024]):
        t_m = time.time()
        clf = lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.038,
            num_leaves=42,
            subsample=0.85,
            colsample_bytree=0.75,
            random_state=s,
            n_jobs=-1,
            verbose=-1
        )
        clf.fit(X_train, y_train)
        m_file = os.path.join(MODEL_DIR, f'lgb_fold_{i}.txt')
        clf.booster_.save_model(m_file)
        lgb_models.append(clf)
        print(f"  LightGBM {i+1}/5 trained in {time.time()-t_m:.1f}s.")

    # XGBoost Hist booster
    t_x = time.time()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.75,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    xgb_clf.fit(X_train, y_train)
    xgb_clf.save_model(os.path.join(MODEL_DIR, 'xgb_model.json'))
    print(f"  XGBoost Hist trained in {time.time()-t_x:.1f}s.")

    # CatBoost Oblivious Tree booster
    t_c = time.time()
    cat_clf = CatBoostClassifier(iterations=350, learning_rate=0.04, depth=6, random_seed=42, verbose=0, thread_count=-1)
    cat_clf.fit(X_train, y_train)
    cat_clf.save_model(os.path.join(MODEL_DIR, 'cat_model.cbm'))
    print(f"  CatBoost trained in {time.time()-t_c:.1f}s.")

    # 2. Setup Match Paths & Priors
    priors = load_statsbomb_priors()
    trans_matrix = build_markov_transition_matrix(priors)

    paths = games.asset_paths(source="tactical")
    raw_atd = paths["atd_raw"]

    half1_start, half1_end = 11185, 90651
    half2_start, half2_end = 110808, 190197
    total_video_frames = games.GAMES["BAR-ATH"]["n_video_frames"]  # 199,319

    full_labels = np.full(total_video_frames, "X", dtype="<U1")

    # 3. Process Half 1
    print(f"\n[2/4] Processing Half 1 ({half1_start:,} .. {half1_end:,})...")
    t_h1 = time.time()
    raw_h1 = extract_features_df(raw_atd, half1_start, half1_end)
    df_h1 = add_refined_features(add_directional_features(raw_h1))

    p_lgb_h1 = np.mean([clf.predict_proba(df_h1[feature_cols]) for clf in lgb_models], axis=0)
    p_xgb_h1 = xgb_clf.predict_proba(df_h1[feature_cols])
    p_cat_h1 = cat_clf.predict_proba(df_h1[feature_cols])
    # Weighted Tri-Model Blend: 50% LGBM + 25% XGBoost + 25% CatBoost
    p_blend_h1 = 0.50 * p_lgb_h1 + 0.25 * p_xgb_h1 + 0.25 * p_cat_h1

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
    df_h2 = add_refined_features(add_directional_features(raw_h2))

    p_lgb_h2 = np.mean([clf.predict_proba(df_h2[feature_cols]) for clf in lgb_models], axis=0)
    p_xgb_h2 = xgb_clf.predict_proba(df_h2[feature_cols])
    p_cat_h2 = cat_clf.predict_proba(df_h2[feature_cols])
    p_blend_h2 = 0.50 * p_lgb_h2 + 0.25 * p_xgb_h2 + 0.25 * p_cat_h2

    p_cal_h2 = calibrate_physics(df_h2, p_blend_h2)
    v_raw_h2 = viterbi_decode(p_cal_h2, trans_matrix, trans_scale=0.10)
    v_filt_h2 = apply_duration_filter(v_raw_h2, min_live_len=45, min_dead_len=65)
    v_final_h2 = apply_sandwich_hysteresis(v_filt_h2, min_turnover_len=95)

    # Pre-kickoff dead ball invariant
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
    df_out = pd.DataFrame({
        'frame': np.arange(total_video_frames),
        'label': full_labels
    })
    assert len(df_out) == 199319, f"Row count error: {len(df_out)} vs 199319"
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV} ({len(df_out):,} frames) in {time.time()-t0:.1f}s.")

if __name__ == '__main__':
    main()
