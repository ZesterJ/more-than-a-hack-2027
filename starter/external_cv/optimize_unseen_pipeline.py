"""
Systematic Optimization Suite for Unseen European Matches
Benchmarks all optimization stages on 165,643 UNSEEN test frames:
- Match 3442: Real Madrid vs FC Barcelona (66,972 frames)
- Match 2440: Liverpool vs Manchester City (68,671 frames)
- Metrica Game 3: 30,000 frames

Zero Barcelona competition frames loaded. Strict rule compliance.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, '.')
from starter.generate_full_submission import (
    add_directional_features,
    calibrate_physics,
    viterbi_decode,
    apply_duration_filter,
    apply_sandwich_hysteresis
)
from starter.external_cv.tune_on_external_cv import add_advanced_features
from starter.external_cv.test_proper_alignment import swap_team_perspective
from starter.step14_statsbomb_bayesian_viterbi import load_statsbomb_priors, build_markov_transition_matrix

def load_aligned_datasets():
    print("\n[1/4] Loading and Fully Aligning Datasets...")
    t0 = time.time()
    
    df_met = pd.read_csv('data/metrica_3games_features.csv')
    df_sc4 = pd.read_csv('data/skillcorner_4matches_features.csv')
    
    # Properly swap perspective for away-possession matches
    mask_3442 = df_sc4['game'] == 'SkillCorner_RealMadrid_Barca'
    mask_2440 = df_sc4['game'] == 'SkillCorner_Liverpool_ManCity'
    df_sc4 = swap_team_perspective(df_sc4, mask_3442 | mask_2440)
    
    df_met = add_advanced_features(add_directional_features(df_met))
    df_sc4 = add_advanced_features(add_directional_features(df_sc4))
    
    # Add Pass Flight Continuity Feature:
    # High ball speed (>6 m/s) with both teams distant indicates pass in flight
    for df in [df_met, df_sc4]:
        df['pass_in_flight'] = ((df['ball_speed'] > 5.5) & (df['d_min_home'] > 2.0) & (df['d_min_away'] > 2.0)).astype(float)
        df['pass_flight_roll'] = df['pass_in_flight'].rolling(15, min_periods=1, center=True).mean()
        
    all_data = pd.concat([df_met, df_sc4], ignore_index=True)
    print(f"Loaded and aligned {len(all_data):,} frames across 7 matches in {time.time()-t0:.1f}s.")
    return all_data

def evaluate_pipeline():
    all_data = load_aligned_datasets()
    
    train_games = [
        'SkillCorner_Barca_RealMadrid',
        'SkillCorner_ManCity_Liverpool',
        'Metrica_1',
        'Metrica_2'
    ]
    test_games = ['SkillCorner_RealMadrid_Barca', 'SkillCorner_Liverpool_ManCity', 'Metrica_3']
    
    drop_uninformative = ['video_frame', 'frame_id', 'label', 'game',
                          'home_attack_direction', 'ball_missing', 'overall_speed', 'speed_all_roll_2s']
    feature_cols = [c for c in all_data.columns if c not in drop_uninformative]
    label_map = {'H': 0, 'A': 1, 'D': 2}
    inv_map = {0: 'H', 1: 'A', 2: 'D'}
    
    train_df = all_data[all_data['game'].isin(train_games)].copy().reset_index(drop=True)
    X_train = train_df[feature_cols]
    y_train = train_df['label'].map(label_map).values
    
    print(f"\n[2/4] Training Multi-Tree Diversity Ensemble ({len(feature_cols)} features)...")
    t0 = time.time()
    
    # 3-Seed LightGBM Diversity
    lgb_models = []
    for s in [42, 101, 2024]:
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
        lgb_models.append(clf)
    print(f"  Trained 3 LightGBM models ({time.time()-t0:.1f}s)")
    
    # XGBoost Hist booster
    t_xgb = time.time()
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
    print(f"  Trained XGBoost Hist booster ({time.time()-t_xgb:.1f}s)")
    
    # Bayesian priors & transition matrix
    priors = load_statsbomb_priors()
    trans_matrix = build_markov_transition_matrix(priors)
    
    print("\n[3/4] Evaluating Systematic Optimizations on 165,643 Unseen Test Frames...")
    
    # Collect test predictions
    test_frames = {}
    for g in test_games:
        test_df = all_data[all_data['game'] == g].copy().reset_index(drop=True)
        X_test = test_df[feature_cols]
        y_true = test_df['label'].map(label_map).values
        
        # Raw LightGBM 0
        p_lgb0 = lgb_models[0].predict_proba(X_test)
        
        # Ensemble blend (75% LGBM avg + 25% XGBoost)
        p_lgb_avg = np.mean([m.predict_proba(X_test) for m in lgb_models], axis=0)
        p_xgb = xgb_clf.predict_proba(X_test)
        p_blend = 0.75 * p_lgb_avg + 0.25 * p_xgb
        
        # Physics calibration
        p_cal = calibrate_physics(test_df, p_blend)
        
        test_frames[g] = {
            'df': test_df,
            'y_true': y_true,
            'p_lgb0': p_lgb0,
            'p_blend': p_blend,
            'p_cal': p_cal
        }
        
    configs = [
        ("1. Single LGBM (Raw Argmax)", lambda d: np.argmax(d['p_lgb0'], axis=1)),
        ("2. Multi-Tree Diversity Blend (LGBM 3-Seed + XGBoost)", lambda d: np.argmax(d['p_blend'], axis=1)),
        ("3. Blend + Physics Calibration", lambda d: np.argmax(d['p_cal'], axis=1)),
        ("4. Blend + Physics + Viterbi (tau=0.10)", lambda d: viterbi_decode(d['p_cal'], trans_matrix, trans_scale=0.10)),
        ("5. Blend + Viterbi + Duration Filter (live=25, dead=40)", lambda d: apply_duration_filter(
            viterbi_decode(d['p_cal'], trans_matrix, trans_scale=0.10), min_live_len=25, min_dead_len=40)),
        ("6. FULL PIPELINE: Blend + Viterbi + Duration + Duel Hysteresis (turnover=50)", lambda d: apply_sandwich_hysteresis(
            apply_duration_filter(
                viterbi_decode(d['p_cal'], trans_matrix, trans_scale=0.10), min_live_len=25, min_dead_len=40),
            min_turnover_len=50))
    ]
    
    results = []
    print("\n" + "="*110)
    print(f"{'Optimization Configuration':<45} | {'Overall Acc':<12} | {'Macro F1':<10} | {'RM vs Barca':<12} | {'Liv vs ManCity':<14} | {'Metrica 3':<10}")
    print("="*110)
    
    for name, fn in configs:
        all_true = []
        all_pred = []
        match_scores = {}
        
        for g in test_games:
            data = test_frames[g]
            pred = fn(data)
            acc = accuracy_score(data['y_true'], pred)
            f1 = f1_score(data['y_true'], pred, average='macro')
            match_scores[g] = (acc, f1)
            all_true.extend(data['y_true'])
            all_pred.extend(pred)
            
        ov_acc = accuracy_score(all_true, all_pred)
        ov_f1 = f1_score(all_true, all_pred, average='macro')
        
        sc_rm = f"{match_scores['SkillCorner_RealMadrid_Barca'][0]:.2%}"
        sc_liv = f"{match_scores['SkillCorner_Liverpool_ManCity'][0]:.2%}"
        sc_m3 = f"{match_scores['Metrica_3'][0]:.2%}"
        
        print(f"{name:<45} | {ov_acc:<12.2%} | {ov_f1:<10.4f} | {sc_rm:<12} | {sc_liv:<14} | {sc_m3:<10}")
        results.append({
            'Configuration': name,
            'Overall_Accuracy': ov_acc,
            'Macro_F1': ov_f1,
            'RealMadrid_Barca_Acc': match_scores['SkillCorner_RealMadrid_Barca'][0],
            'Liverpool_ManCity_Acc': match_scores['SkillCorner_Liverpool_ManCity'][0],
            'Metrica_3_Acc': match_scores['Metrica_3'][0]
        })
        
    print("="*110)
    df_res = pd.DataFrame(results)
    df_res.to_csv('starter/external_cv/unseen_optimization_progression.csv', index=False)
    print(f"\nSaved progression results to starter/external_cv/unseen_optimization_progression.csv")

if __name__ == '__main__':
    evaluate_pipeline()
