"""
Tune Temporal Post-Processing on External Leave-One-Match-Out CV
Strict Compliance: 100% External Data (Zero competition frames).
Finds the optimal temporal smoothing parameters (tau, min_live, min_dead, min_turnover)
purely on external matches.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, '.')
from evaluation.possession_metrics import evaluate
from starter.generate_full_submission import (
    add_directional_features,
    viterbi_decode,
    apply_duration_filter,
    apply_sandwich_hysteresis
)
from starter.external_cv.test_refined_features import add_refined_features
from starter.step14_statsbomb_bayesian_viterbi import load_statsbomb_priors, build_markov_transition_matrix

def run_external_tuning():
    print("="*90)
    print(" TUNING TEMPORAL PARAMETERS ON EXTERNAL LEAVE-ONE-MATCH-OUT CV")
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
    inv_map = {0: 'H', 1: 'A', 2: 'D'}
    y_all = all_train['label'].map(label_map).values

    matches = all_train['game'].unique()
    oof_probs = np.zeros((len(all_train), 3), dtype=np.float32)

    print(f"\n[1/3] Generating Out-Of-Fold Probabilities across 5 external matches ({len(feature_cols)} feats)...")
    for test_game in matches:
        tr_mask = all_train['game'] != test_game
        te_mask = all_train['game'] == test_game
        
        X_tr = all_train.loc[tr_mask, feature_cols]
        y_tr = y_all[tr_mask]
        X_te = all_train.loc[te_mask, feature_cols]
        
        clf = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.038, num_leaves=42, colsample_bytree=0.75, random_state=42, n_jobs=-1, verbose=-1)
        clf.fit(X_tr, y_tr)
        oof_probs[te_mask] = clf.predict_proba(X_te)
        print(f"  Fitted OOF fold for {test_game}")

    # 2. Grid Search Temporal Smoothing Parameters
    priors = load_statsbomb_priors()
    trans_matrix = build_markov_transition_matrix(priors)

    print("\n[2/3] Evaluating Parameter Combinations on Out-Of-Fold External Predictions...")
    results = []

    # Test grid of parameters
    for tau in [0.08, 0.10, 0.12]:
        # Precompute Viterbi per match to avoid re-decoding across continuous segments
        v_raw_all = np.zeros(len(all_train), dtype=int)
        for g in matches:
            m = all_train['game'] == g
            v_raw_all[m] = viterbi_decode(oof_probs[m], trans_matrix, trans_scale=tau)

        for min_live in [30, 40, 45]:
            for min_dead in [45, 55, 65]:
                for min_turnover in [50, 70, 85, 95]:
                    # Apply duration filter and hysteresis per match
                    pred_all = np.zeros(len(all_train), dtype=int)
                    for g in matches:
                        m = all_train['game'] == g
                        v_filt = apply_duration_filter(v_raw_all[m], min_live_len=min_live, min_dead_len=min_dead)
                        v_final = apply_sandwich_hysteresis(v_filt, min_turnover_len=min_turnover)
                        pred_all[m] = v_final

                    acc = accuracy_score(y_all, pred_all)
                    f1 = f1_score(y_all, pred_all, average='macro')
                    
                    results.append({
                        'tau': tau,
                        'min_live': min_live,
                        'min_dead': min_dead,
                        'min_turnover': min_turnover,
                        'oof_acc': acc,
                        'oof_macro_f1': f1
                    })

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by=['oof_macro_f1', 'oof_acc'], ascending=False).reset_index(drop=True)

    print("\n" + "="*90)
    print(" TOP 10 TEMPORAL PARAMETER CONFIGURATIONS ON EXTERNAL DATA:")
    print("="*90)
    print(df_res.head(10).to_string(index=False))
    
    best = df_res.iloc[0]
    print("\n" + "="*90)
    print(f" WINNING EXTERNAL CONFIGURATION:")
    print(f"  tau = {best['tau']}, min_live = {int(best['min_live'])}, min_dead = {int(best['min_dead'])}, min_turnover = {int(best['min_turnover'])}")
    print(f"  External OOF Accuracy: {best['oof_acc']:.2%}")
    print(f"  External OOF Macro F1: {best['oof_macro_f1']:.4f}")
    print("="*90)

    # Save best config
    best_dict = {
        'tau': float(best['tau']),
        'min_live': int(best['min_live']),
        'min_dead': int(best['min_dead']),
        'min_turnover': int(best['min_turnover']),
        'oof_accuracy': float(best['oof_acc']),
        'oof_macro_f1': float(best['oof_macro_f1'])
    }
    with open('starter/external_cv/best_external_temporal_config.json', 'w') as f:
        import json; json.dump(best_dict, f, indent=2)

if __name__ == '__main__':
    run_external_tuning()
