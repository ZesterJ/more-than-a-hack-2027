"""
External Multi-Match Out-of-Sample Benchmark
Strict Compliance with Organizer Rule Clarification (Nov 3, 2026)

PURPOSE:
- Train models on a subset of external matches.
- Benchmark and optimize models on completely UNSEEN external matches:
    Test Match 1: SkillCorner Match 3442 (Real Madrid vs FC Barcelona, 66,972 frames)
    Test Match 2: SkillCorner Match 2440 (Liverpool vs Manchester City, 68,671 frames)
    Test Match 3: Metrica Game 3 (30,000 frames)
- Zero competition frames loaded.
- Diagnoses error modes and benchmarks iterative improvements purely on external matches.

Usage:
    python starter/external_cv/benchmark_unseen_matches.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
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
from starter.step14_statsbomb_bayesian_viterbi import load_statsbomb_priors, build_markov_transition_matrix

def load_and_prepare_datasets():
    print("\n[1/3] Loading External Tracking Datasets...")
    t0 = time.time()
    
    df_met = pd.read_csv('data/metrica_3games_features.csv')
    df_sc4 = pd.read_csv('data/skillcorner_4matches_features.csv')
    
    df_met = add_advanced_features(add_directional_features(df_met))
    df_sc4 = add_advanced_features(add_directional_features(df_sc4))
    
    # Tactically Align SC4 matches where possession team was Away
    # Match 3442: Real Madrid (H) vs Barca (A) -> Align Barca to Role 0 (H)
    mask_3442 = df_sc4['game'] == 'SkillCorner_RealMadrid_Barca'
    # Match 2440: Liverpool (H) vs Man City (A) -> Align Man City to Role 0 (H)
    mask_2440 = df_sc4['game'] == 'SkillCorner_Liverpool_ManCity'
    
    for mask in [mask_3442, mask_2440]:
        # Swap labels H <-> A
        lbls = df_sc4.loc[mask, 'label'].copy()
        df_sc4.loc[mask & (lbls == 'H'), 'label'] = 'TEMP_A'
        df_sc4.loc[mask & (lbls == 'A'), 'label'] = 'H'
        df_sc4.loc[mask & (df_sc4['label'] == 'TEMP_A'), 'label'] = 'A'
        
        # Swap distance and spatial margins
        temp_dh = df_sc4.loc[mask, 'd_min_home'].copy()
        df_sc4.loc[mask, 'd_min_home'] = df_sc4.loc[mask, 'd_min_away']
        df_sc4.loc[mask, 'd_min_away'] = temp_dh
        
        df_sc4.loc[mask, 'd_margin'] = -df_sc4.loc[mask, 'd_margin']
        df_sc4.loc[mask, 'd_margin_roll_1s'] = -df_sc4.loc[mask, 'd_margin_roll_1s']
        df_sc4.loc[mask, 'd_margin_roll_3s'] = -df_sc4.loc[mask, 'd_margin_roll_3s']
        df_sc4.loc[mask, 'control_ratio'] = -df_sc4.loc[mask, 'control_ratio']
        df_sc4.loc[mask, 'control_ratio_roll_1s'] = -df_sc4.loc[mask, 'control_ratio_roll_1s']
        df_sc4.loc[mask, 'control_ratio_roll_3s'] = -df_sc4.loc[mask, 'control_ratio_roll_3s']
        
        temp_hcx = df_sc4.loc[mask, 'home_centroid_x'].copy()
        df_sc4.loc[mask, 'home_centroid_x'] = df_sc4.loc[mask, 'away_centroid_x']
        df_sc4.loc[mask, 'away_centroid_x'] = temp_hcx
        
        temp_hcy = df_sc4.loc[mask, 'home_centroid_y'].copy()
        df_sc4.loc[mask, 'home_centroid_y'] = df_sc4.loc[mask, 'away_centroid_y']
        df_sc4.loc[mask, 'away_centroid_y'] = temp_hcy
        
        df_sc4.loc[mask, 'centroid_diff_x'] = -df_sc4.loc[mask, 'centroid_diff_x']

    all_data = pd.concat([df_met, df_sc4], ignore_index=True)
    print(f"Loaded and aligned {len(all_data):,} frames across 7 external matches in {time.time()-t0:.1f}s.")
    return all_data

def evaluate_on_unseen_matches(model_type, feature_schema, train_games, test_games, all_data, **kwargs):
    label_map = {'H': 0, 'A': 1, 'D': 2}
    
    train_mask = all_data['game'].isin(train_games)
    train_df = all_data[train_mask].copy().reset_index(drop=True)
    X_train = train_df[feature_schema]
    y_train = train_df['label'].map(label_map).values
    
    t0 = time.time()
    if model_type == 'lgb':
        clf = lgb.LGBMClassifier(
            n_estimators=kwargs.get('n_estimators', 350),
            learning_rate=kwargs.get('lr', 0.038),
            num_leaves=kwargs.get('num_leaves', 42),
            subsample=0.85,
            colsample_bytree=kwargs.get('colsample', 0.75),
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        clf.fit(X_train, y_train)
    elif model_type == 'xgb':
        clf = xgb.XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 350),
            learning_rate=kwargs.get('lr', 0.04),
            max_depth=kwargs.get('max_depth', 6),
            subsample=0.85,
            colsample_bytree=kwargs.get('colsample', 0.75),
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        clf.fit(X_train, y_train)
    elif model_type == 'blend':
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=350, learning_rate=0.038, num_leaves=42,
            subsample=0.85, colsample_bytree=kwargs.get('colsample', 0.75),
            random_state=42, n_jobs=-1, verbose=-1
        )
        clf_lgb.fit(X_train, y_train)
        
        clf_xgb = xgb.XGBClassifier(
            n_estimators=350, learning_rate=0.04, max_depth=6,
            subsample=0.85, colsample_bytree=kwargs.get('colsample', 0.75),
            random_state=42, n_jobs=-1, tree_method='hist'
        )
        clf_xgb.fit(X_train, y_train)
    t_train = time.time() - t0
    
    # Evaluate across each unseen test game
    match_metrics = {}
    total_y_true = []
    total_y_pred = []
    
    for g in test_games:
        test_mask = all_data['game'] == g
        test_df = all_data[test_mask].copy().reset_index(drop=True)
        y_test = test_df['label'].map(label_map).values
        
        if model_type == 'blend':
            p_lgb = clf_lgb.predict_proba(test_df[feature_schema])
            p_xgb = clf_xgb.predict_proba(test_df[feature_schema])
            p_test = 0.60 * p_lgb + 0.40 * p_xgb
        else:
            p_test = clf.predict_proba(test_df[feature_schema])
            
        preds = np.argmax(p_test, axis=1)
        
        # Apply hysteresis / smoothing if requested
        if kwargs.get('apply_hysteresis', False):
            preds = apply_duration_filter(preds, min_live_len=45, min_dead_len=65)
            preds = apply_sandwich_hysteresis(preds, min_turnover_len=kwargs.get('min_turnover_len', 95))
            
        acc = accuracy_score(y_test, preds)
        mf1 = f1_score(y_test, preds, average='macro')
        f1_h, f1_a, f1_d = f1_score(y_test, preds, average=None)
        
        match_metrics[g] = {
            "Accuracy": acc,
            "Macro F1": mf1,
            "H F1": f1_h,
            "A F1": f1_a,
            "D F1": f1_d,
            "Frames": len(y_test)
        }
        total_y_true.extend(y_test)
        total_y_pred.extend(preds)
        
    overall_acc = accuracy_score(total_y_true, total_y_pred)
    overall_mf1 = f1_score(total_y_true, total_y_pred, average='macro')
    overall_f1s = f1_score(total_y_true, total_y_pred, average=None)
    
    return {
        "overall_accuracy": overall_acc,
        "overall_macro_f1": overall_mf1,
        "overall_h_f1": overall_f1s[0],
        "overall_a_f1": overall_f1s[1],
        "overall_d_f1": overall_f1s[2],
        "per_match": match_metrics,
        "train_time": t_train
    }

def main():
    print("="*100)
    print(" EXTERNAL MULTI-MATCH BENCHMARK & SYSTEMATIC OPTIMIZATION SUITE")
    print(" (Zero Competition Data Loaded - 100% External Out-of-Sample Generalization)")
    print("="*100)

    all_data = load_and_prepare_datasets()

    # Split matches into Train Pool and Unseen Test Benchmark
    train_games = [
        'SkillCorner_Barca_RealMadrid',   # Match 2841 (68.4k)
        'SkillCorner_ManCity_Liverpool',  # Match 4039 (68.3k)
        'Metrica_1',                      # Metrica 1 (30k)
        'Metrica_2'                       # Metrica 2 (30k)
    ]
    
    unseen_test_games = [
        'SkillCorner_RealMadrid_Barca',   # Match 3442 (67.0k) -> Pure Unseen El Clásico
        'SkillCorner_Liverpool_ManCity',  # Match 2440 (68.7k) -> Pure Unseen Pep Match
        'Metrica_3'                       # Metrica 3 (30.0k)   -> Pure Unseen Metrica Match
    ]
    
    print("\n[2/3] Partitioning External Datasets:")
    train_frames = all_data['game'].isin(train_games).sum()
    test_frames = all_data['game'].isin(unseen_test_games).sum()
    print(f"  Training Games ({train_frames:,} frames): {train_games}")
    print(f"  UNSEEN Test Benchmark ({test_frames:,} frames): {unseen_test_games}")

    drop_uninformative = ['video_frame', 'frame_id', 'label', 'game',
                          'home_attack_direction', 'ball_missing', 'overall_speed', 'speed_all_roll_2s']
    feature_schema_baseline = [c for c in all_data.columns if c not in ['video_frame', 'frame_id', 'label', 'game', 
                                                                        'control_ratio', 'control_ratio_roll_1s', 'control_ratio_roll_3s',
                                                                        'in_active_duel', 'duel_roll_1s', 'ball_stationary', 'stationary_roll_2s']]
    feature_schema_upgraded = [c for c in all_data.columns if c not in drop_uninformative]

    print(f"\n[3/3] Running Iterative Optimization Experiments on Unseen External Matches...")
    
    experiments = [
        # Iteration 1: Baseline 38 features with LightGBM
        ("1. Baseline LightGBM (38 features, colsample=0.85)", 
         {"model_type": "lgb", "feature_schema": feature_schema_baseline, "colsample": 0.85}),
        
        # Iteration 2: Upgraded 41 features (Control Ratio + Duel Flags)
        ("2. Upgraded Features (41 features, colsample=0.85)", 
         {"model_type": "lgb", "feature_schema": feature_schema_upgraded, "colsample": 0.85}),
         
        # Iteration 3: Feature Weight Optimization (colsample=0.75)
        ("3. Feature Weight Tuning (41 features, colsample=0.75)", 
         {"model_type": "lgb", "feature_schema": feature_schema_upgraded, "colsample": 0.75}),
         
        # Iteration 4: XGBoost Hist Classifier
        ("4. XGBoost Hist (41 features, colsample=0.75)", 
         {"model_type": "xgb", "feature_schema": feature_schema_upgraded, "colsample": 0.75}),
         
        # Iteration 5: Multi-Tree Blend (LightGBM + XGBoost)
        ("5. Multi-Tree Blend (LGBM + XGB, colsample=0.75)", 
         {"model_type": "blend", "feature_schema": feature_schema_upgraded, "colsample": 0.75}),
         
        # Iteration 6: Multi-Tree Blend + Duel Momentum Hysteresis
        ("6. Multi-Tree Blend + Duel Hysteresis (min_len=95)", 
         {"model_type": "blend", "feature_schema": feature_schema_upgraded, "colsample": 0.75, "apply_hysteresis": True, "min_turnover_len": 95})
    ]

    results_table = []
    
    for title, cfg in experiments:
        print(f"\nEvaluating {title}...")
        res = evaluate_on_unseen_matches(
            train_games=train_games,
            test_games=unseen_test_games,
            all_data=all_data,
            **cfg
        )
        
        print(f"  --> Overall Unseen Accuracy: {res['overall_accuracy']:.2%}")
        print(f"  --> Overall Unseen Macro F1: {res['overall_macro_f1']:.4f}")
        for g_name, m_stats in res["per_match"].items():
            short_g = g_name.replace("SkillCorner_", "")
            print(f"      [{short_g}]: Acc={m_stats['Accuracy']:.2%}, Macro F1={m_stats['Macro F1']:.4f}")
            
        results_table.append({
            "Experiment": title,
            "Unseen Accuracy": round(res["overall_accuracy"] * 100, 2),
            "Unseen Macro F1": round(res["overall_macro_f1"], 4),
            "Home F1": round(res["overall_h_f1"], 4),
            "Away F1": round(res["overall_a_f1"], 4),
            "Stoppage F1": round(res["overall_d_f1"], 4),
            "Barca Clásico Acc": round(res["per_match"]['SkillCorner_RealMadrid_Barca']['Accuracy'] * 100, 2),
            "Pep ManCity Acc": round(res["per_match"]['SkillCorner_Liverpool_ManCity']['Accuracy'] * 100, 2),
            "Metrica 3 Acc": round(res["per_match"]['Metrica_3']['Accuracy'] * 100, 2)
        })

    df_res = pd.DataFrame(results_table).sort_values("Unseen Macro F1", ascending=False)
    print("\n" + "="*100)
    print(" EXTERNAL MULTI-MATCH OPTIMIZATION SCOREBOARD (165,643 UNSEEN TEST FRAMES)")
    print("="*100)
    print(df_res.to_string(index=False))
    
    out_csv = 'starter/external_cv/external_unseen_matches_benchmark.csv'
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved scoreboard to {out_csv}.")

if __name__ == '__main__':
    main()
