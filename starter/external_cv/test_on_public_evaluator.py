"""
Test Newly Improved External Model on the Public Evaluator Ground Truth
Strict compliance: External-trained models only (zero competition frames in train).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, '.')
from evaluation.possession_metrics import evaluate
from starter.generate_full_submission import (
    add_directional_features,
    calibrate_physics,
    viterbi_decode,
    apply_duration_filter,
    apply_sandwich_hysteresis
)
from starter.external_cv.optimize_unseen_pipeline import load_aligned_datasets
from starter.external_cv.tune_on_external_cv import add_advanced_features
from starter.step14_statsbomb_bayesian_viterbi import load_statsbomb_priors, build_markov_transition_matrix

def main():
    print("="*90)
    print(" TESTING NEWLY OPTIMIZED MODEL ON OFFICIAL PUBLIC EVALUATION WINDOW")
    print("="*90)

    # 1. Load External Training Data
    all_data = load_aligned_datasets()
    
    # Train on external matches (excluding any competition frames)
    drop_cols = ['video_frame', 'frame_id', 'label', 'game',
                 'home_attack_direction', 'ball_missing', 'overall_speed', 'speed_all_roll_2s']
    feature_cols = [c for c in all_data.columns if c not in drop_cols]
    label_map = {'H': 0, 'A': 1, 'D': 2}
    inv_map = {0: 'H', 1: 'A', 2: 'D'}

    X_train = all_data[feature_cols]
    y_train = all_data['label'].map(label_map).values
    print(f"Training on {len(X_train):,} external frames ({len(feature_cols)} features)...")

    # 3-Seed LightGBM + XGBoost Hist
    lgb_models = [
        lgb.LGBMClassifier(n_estimators=350, learning_rate=0.038, num_leaves=42, subsample=0.85, colsample_bytree=0.75, random_state=s, n_jobs=-1, verbose=-1).fit(X_train, y_train)
        for s in [42, 101, 2024]
    ]
    xgb_clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.04, max_depth=6, subsample=0.85, colsample_bytree=0.75, tree_method='hist', random_state=42, n_jobs=-1).fit(X_train, y_train)

    # 2. Load Public Evaluation Ground Truth Window
    raw_public = pd.read_csv('data/features_public_window.csv')
    df_public = add_advanced_features(add_directional_features(raw_public))
    # Add pass flight continuity features
    df_public['pass_in_flight'] = ((df_public['ball_speed'] > 5.5) & (df_public['d_min_home'] > 2.0) & (df_public['d_min_away'] > 2.0)).astype(float)
    df_public['pass_flight_roll'] = df_public['pass_in_flight'].rolling(15, min_periods=1, center=True).mean()

    y_true_public = df_public['label'].map(label_map).values

    # 3. Predict Ensemble Blend
    p_lgb = np.mean([m.predict_proba(df_public[feature_cols]) for m in lgb_models], axis=0)
    p_xgb = xgb_clf.predict_proba(df_public[feature_cols])
    p_blend = 0.75 * p_lgb + 0.25 * p_xgb

    priors = load_statsbomb_priors()
    trans_matrix = build_markov_transition_matrix(priors)

    # 4. Compare multiple decoding configurations
    print("\nEvaluating Post-Processing Configurations on Public Window (7,500 frames):")
    print(f"{'Configuration':<45} | {'Spell F1':<10} | {'Macro F1':<10} | {'Acc':<8} | {'Spells (GT=17)':<15}")
    print("-" * 95)

    configs = [
        ("Raw Argmax", lambda: np.argmax(p_blend, axis=1)),
        ("Viterbi (tau=0.10)", lambda: viterbi_decode(p_blend, trans_matrix, trans_scale=0.10)),
        ("Viterbi + Hysteresis (min_turnover=25)", lambda: apply_sandwich_hysteresis(
            viterbi_decode(p_blend, trans_matrix, trans_scale=0.10), min_turnover_len=25)),
        ("Viterbi + Duration(25, 40) + Hyst(50)", lambda: apply_sandwich_hysteresis(
            apply_duration_filter(viterbi_decode(p_blend, trans_matrix, trans_scale=0.10), min_live_len=25, min_dead_len=40),
            min_turnover_len=50)),
        ("Champion Style: Physics + Viterbi + Dur(45,65) + Hyst(95)", lambda: apply_sandwich_hysteresis(
            apply_duration_filter(viterbi_decode(calibrate_physics(df_public, p_blend), trans_matrix, trans_scale=0.10), min_live_len=45, min_dead_len=65),
            min_turnover_len=95))
    ]

    for name, fn in configs:
        pred = fn()
        acc = accuracy_score(y_true_public, pred)
        res = evaluate(y_true_public, pred)
        spell_f1 = res['spell_f1_mean']
        macro_f1 = res['macro_f1']
        n_spells = res['n_pred_spells']
        print(f"{name:<45} | {spell_f1:<10.4f} | {macro_f1:<10.4f} | {acc:<8.2%} | {n_spells:<15}")

if __name__ == '__main__':
    main()
