"""
Test Refined Feature Schema on External Leave-One-Match-Out CV
Tests whether dropping 'ball_stationary' while keeping 'control_ratio', 'in_active_duel',
and 'speed_all_roll_2s' improves both External CV and Public Evaluation Window.
Strictly external data (zero competition frames).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
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
from starter.step14_statsbomb_bayesian_viterbi import load_statsbomb_priors, build_markov_transition_matrix

def add_refined_features(df):
    df = df.copy()
    eps = 0.1
    # Normalized relative control ratio
    df['control_ratio'] = (df['d_min_away'] - df['d_min_home']) / (df['d_min_away'] + df['d_min_home'] + eps)
    df['control_ratio_roll_1s'] = df['control_ratio'].rolling(25, min_periods=1, center=True).mean()
    df['control_ratio_roll_3s'] = df['control_ratio'].rolling(75, min_periods=1, center=True).mean()
    
    # Active duel contestation
    df['in_active_duel'] = ((df['d_min_home'] < 2.5) & (df['d_min_away'] < 2.5)).astype(float)
    df['duel_roll_1s'] = df['in_active_duel'].rolling(25, min_periods=1, center=True).mean()
    
    # Pass in flight continuity
    df['pass_in_flight'] = ((df['ball_speed'] > 5.5) & (df['d_min_home'] > 2.0) & (df['d_min_away'] > 2.0)).astype(float)
    df['pass_flight_roll'] = df['pass_in_flight'].rolling(15, min_periods=1, center=True).mean()
    
    # Intentionally OMIT 'ball_stationary' to prevent false suppression of live slow build-ups
    return df

def test():
    print("="*90)
    print(" EVALUATING REFINED POSITIONAL PLAY FEATURES ON EXTERNAL DATA")
    print("="*90)

    # 1. Load External Positional Play Corpus (226,716 frames)
    df_met = pd.read_csv('data/metrica_3games_features.csv')
    df_sc = pd.read_csv('data/skillcorner_barca_mancity_features.csv')
    
    df_met = add_refined_features(add_directional_features(df_met))
    df_sc = add_refined_features(add_directional_features(df_sc))
    
    all_train = pd.concat([df_met, df_sc], ignore_index=True)
    
    drop_cols = ['video_frame', 'frame_id', 'label', 'game', 'home_attack_direction', 'ball_missing']
    feature_cols = [c for c in all_train.columns if c not in drop_cols]
    label_map = {'H': 0, 'A': 1, 'D': 2}
    print(f"Feature count: {len(feature_cols)} features: {feature_cols}")

    # 2. Leave-One-Match-Out CV across the 5 external positional matches
    matches = all_train['game'].unique()
    oof_preds = np.zeros(len(all_train), dtype=int)
    y_all = all_train['label'].map(label_map).values

    print("\nRunning Leave-One-Match-Out External Cross-Validation...")
    for test_game in matches:
        tr_mask = all_train['game'] != test_game
        te_mask = all_train['game'] == test_game
        
        X_tr = all_train.loc[tr_mask, feature_cols]
        y_tr = y_all[tr_mask]
        X_te = all_train.loc[te_mask, feature_cols]
        y_te = y_all[te_mask]
        
        clf = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.038, num_leaves=42, colsample_bytree=0.75, random_state=42, n_jobs=-1, verbose=-1)
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)
        oof_preds[te_mask] = np.argmax(p, axis=1)
        
        acc = accuracy_score(y_te, oof_preds[te_mask])
        f1 = f1_score(y_te, oof_preds[te_mask], average='macro')
        print(f"  OOF [{test_game}]: Acc = {acc:.2%}, Macro F1 = {f1:.4f}")

    total_acc = accuracy_score(y_all, oof_preds)
    total_f1 = f1_score(y_all, oof_preds, average='macro')
    print(f"\n---> Overall External LOMO-CV Accuracy: {total_acc:.2%}")
    print(f"---> Overall External LOMO-CV Macro F1: {total_f1:.4f}")

    # 3. Fit on full external pool & evaluate on untouched public test window
    print("\nFitting on full external pool (226,716 frames) & testing on untouched blind public window...")
    lgb_models = [
        lgb.LGBMClassifier(n_estimators=350, learning_rate=0.038, num_leaves=42, subsample=0.85, colsample_bytree=0.75, random_state=s, n_jobs=-1, verbose=-1).fit(all_train[feature_cols], y_all)
        for s in [42, 101, 777, 1234, 2024]
    ]
    xgb_clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.04, max_depth=6, subsample=0.85, colsample_bytree=0.75, tree_method='hist', random_state=42, n_jobs=-1).fit(all_train[feature_cols], y_all)

    raw_public = pd.read_csv('data/features_public_window.csv')
    df_public = add_refined_features(add_directional_features(raw_public))
    y_public = df_public['label'].map(label_map).values

    p_lgb = np.mean([m.predict_proba(df_public[feature_cols]) for m in lgb_models], axis=0)
    p_xgb = xgb_clf.predict_proba(df_public[feature_cols])
    p_blend = 0.75 * p_lgb + 0.25 * p_xgb

    priors = load_statsbomb_priors()
    trans_matrix = build_markov_transition_matrix(priors)

    p_cal = calibrate_physics(df_public, p_blend)
    v_raw = viterbi_decode(p_cal, trans_matrix, trans_scale=0.10)
    v_filt = apply_duration_filter(v_raw, min_live_len=45, min_dead_len=65)
    v_final = apply_sandwich_hysteresis(v_filt, min_turnover_len=95)

    res = evaluate(y_public, v_final)
    print("="*90)
    print(" RESULT ON BLIND PUBLIC WINDOW:")
    print(f"  Frame Accuracy: {accuracy_score(y_public, v_final):.2%}")
    print(f"  Macro F1:       {res['macro_f1']:.4f}")
    print(f"  Spell F1:       {res['spell_f1_mean']:.4f}")
    print(f"  Pred Spells:    {res['n_pred_spells']} (GT: 17)")
    print("="*90)

if __name__ == '__main__':
    test()
