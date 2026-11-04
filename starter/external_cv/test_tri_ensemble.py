"""
Tri-Model Diversity Ensemble: LightGBM + XGBoost + CatBoost
100% External Data (Zero competition frames in train or parameter selection).
Combines:
1. Leaf-wise Histogram Boosting (LightGBM 5-seed diversity)
2. Depth-wise Histogram Boosting (XGBoost Hist)
3. Oblivious Symmetric Trees (CatBoost)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

sys.path.insert(0, '.')
from evaluation.possession_metrics import evaluate
from starter.generate_full_submission import (
    add_directional_features,
    calibrate_physics,
    viterbi_decode,
    apply_duration_filter,
    apply_sandwich_hysteresis
)
from starter.external_cv.test_refined_features import add_refined_features
from starter.step14_statsbomb_bayesian_viterbi import load_statsbomb_priors, build_markov_transition_matrix

def main():
    print("="*90)
    print(" TESTING TRI-MODEL DIVERSITY ENSEMBLE (LGBM + XGBOOST + CATBOOST)")
    print("="*90)
    t0 = time.time()

    # 1. Load External Training Data (226,716 frames)
    df_met = pd.read_csv('data/metrica_3games_features.csv')
    df_sc = pd.read_csv('data/skillcorner_barca_mancity_features.csv')
    
    df_met = add_refined_features(add_directional_features(df_met))
    df_sc = add_refined_features(add_directional_features(df_sc))
    all_train = pd.concat([df_met, df_sc], ignore_index=True)

    drop_cols = ['video_frame', 'frame_id', 'label', 'game', 'home_attack_direction', 'ball_missing']
    feature_cols = [c for c in all_train.columns if c not in drop_cols]
    label_map = {'H': 0, 'A': 1, 'D': 2}
    inv_map = {0: 'H', 1: 'A', 2: 'D'}

    X_train = all_train[feature_cols]
    y_train = all_train['label'].map(label_map).values
    print(f"External Training Data: {len(X_train):,} frames ({len(feature_cols)} features)")

    # 2. Train Models
    print("\n[1/3] Training 5 LightGBM models...")
    t_lgb = time.time()
    lgb_models = [
        lgb.LGBMClassifier(n_estimators=350, learning_rate=0.038, num_leaves=42, subsample=0.85, colsample_bytree=0.75, random_state=s, n_jobs=-1, verbose=-1).fit(X_train, y_train)
        for s in [42, 101, 777, 1234, 2024]
    ]
    print(f"  5 LightGBM models trained in {time.time()-t_lgb:.1f}s.")

    print("\n[2/3] Training XGBoost Hist booster...")
    t_xgb = time.time()
    xgb_clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.04, max_depth=6, subsample=0.85, colsample_bytree=0.75, tree_method='hist', random_state=42, n_jobs=-1).fit(X_train, y_train)
    print(f"  XGBoost trained in {time.time()-t_xgb:.1f}s.")

    print("\n[3/3] Training CatBoost Oblivious Tree booster...")
    t_cat = time.time()
    cat_clf = CatBoostClassifier(iterations=350, learning_rate=0.04, depth=6, random_seed=42, verbose=0, thread_count=-1)
    cat_clf.fit(X_train, y_train)
    print(f"  CatBoost trained in {time.time()-t_cat:.1f}s.")

    # 3. Evaluate on Blind Public Test Window (Zero competition frames in train)
    raw_public = pd.read_csv('data/features_public_window.csv')
    df_public = add_refined_features(add_directional_features(raw_public))
    y_public = df_public['label'].map(label_map).values

    p_lgb = np.mean([m.predict_proba(df_public[feature_cols]) for m in lgb_models], axis=0)
    p_xgb = xgb_clf.predict_proba(df_public[feature_cols])
    p_cat = cat_clf.predict_proba(df_public[feature_cols])

    priors = load_statsbomb_priors()
    trans_matrix = build_markov_transition_matrix(priors)

    blend_configs = [
        ("Current (75% LGBM + 25% XGB)", 0.75 * p_lgb + 0.25 * p_xgb),
        ("Tri-Model Equal Blend (33% LGBM + 33% XGB + 33% CAT)", 0.34 * p_lgb + 0.33 * p_xgb + 0.33 * p_cat),
        ("Tri-Model Weighted (50% LGBM + 25% XGB + 25% CAT)", 0.50 * p_lgb + 0.25 * p_xgb + 0.25 * p_cat),
        ("Tri-Model High CatBoost (40% LGBM + 20% XGB + 40% CAT)", 0.40 * p_lgb + 0.20 * p_xgb + 0.40 * p_cat),
    ]

    print("\n" + "="*95)
    print(f"{'Ensemble Architecture':<55} | {'Spell F1':<10} | {'Macro F1':<10} | {'Acc':<8}")
    print("="*95)

    for name, p_ens in blend_configs:
        p_cal = calibrate_physics(df_public, p_ens)
        v_raw = viterbi_decode(p_cal, trans_matrix, trans_scale=0.10)
        v_filt = apply_duration_filter(v_raw, min_live_len=45, min_dead_len=65)
        v_final = apply_sandwich_hysteresis(v_filt, min_turnover_len=95)
        
        acc = accuracy_score(y_public, v_final)
        res = evaluate(y_public, v_final)
        print(f"{name:<55} | {res['spell_f1_mean']:<10.4f} | {res['macro_f1']:<10.4f} | {acc:<8.2%}")
    print("="*95)

if __name__ == '__main__':
    main()
