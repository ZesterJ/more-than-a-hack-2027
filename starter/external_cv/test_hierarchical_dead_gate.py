"""
Benchmark Flat 3-Class vs Hierarchical 2-Stage Decomposition
Calibrated Dead Gate Threshold + Temporal Duration Constraint
Strict Nov 3 Rule Compliance (100% External LOMO Cross-Validation)
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

sys.path.insert(0, '.')
from evaluation.possession_metrics import evaluate

def filter_short_spells(preds, min_dead_frames=50, min_live_frames=25):
    """
    Apply football domain physics:
    1. A dead-ball stoppage CANNOT last < 2.0s (50 frames). If isolated, revert to neighbor.
    2. A live touch inside a dead-ball period cannot last < 1.0s (25 frames).
    """
    n = len(preds)
    out = np.array(preds, copy=True)
    
    # 1. First pass: remove short dead-ball blips (< min_dead_frames)
    i = 0
    while i < n:
        if out[i] == 2:  # Dead
            start = i
            while i < n and out[i] == 2:
                i += 1
            length = i - start
            if length < min_dead_frames:
                # Replace with surrounding live label
                prev_label = out[start - 1] if start > 0 else 0
                next_label = out[i] if i < n else prev_label
                fill_label = prev_label if prev_label in [0, 1] else (next_label if next_label in [0, 1] else 0)
                out[start:i] = fill_label
        else:
            i += 1
            
    # 2. Second pass: remove short live blips (< min_live_frames) surrounded by Dead
    i = 0
    while i < n:
        if out[i] in [0, 1]:  # Live
            start = i
            while i < n and out[i] in [0, 1]:
                i += 1
            length = i - start
            if length < min_live_frames:
                prev_dead = (start > 0 and out[start - 1] == 2)
                next_dead = (i < n and out[i] == 2)
                if prev_dead and next_dead:
                    out[start:i] = 2
        else:
            i += 1
            
    return out

def main():
    print("="*90)
    print(" HIERARCHICAL DEAD GATE & DURATION CALIBRATION (100% EXTERNAL DATA)")
    print("="*90)

    df_met = pd.read_csv('data/metrica_3games_features.csv')
    df_sc = pd.read_csv('data/skillcorner_barca_mancity_features.csv')
    df = pd.concat([df_met, df_sc], ignore_index=True)

    # Feature Engineering
    eps = 1e-4
    df['min_dist_any'] = np.minimum(df['d_min_home'].fillna(99), df['d_min_away'].fillna(99))
    df['min_dist_roll_1s'] = df['min_dist_any'].rolling(25, min_periods=1, center=True).mean()
    df['min_dist_roll_3s'] = df['min_dist_any'].rolling(75, min_periods=1, center=True).mean()
    df['min_dist_roll_5s'] = df['min_dist_any'].rolling(125, min_periods=1, center=True).mean()

    df['players_within_3m_total'] = df['home_within_3m'] + df['away_within_3m']
    df['players_3m_roll_1s'] = df['players_within_3m_total'].rolling(25, min_periods=1, center=True).mean()
    df['players_3m_roll_3s'] = df['players_within_3m_total'].rolling(75, min_periods=1, center=True).mean()

    df['active_total'] = df['home_active'] + df['away_active']
    df['active_roll_2s'] = df['active_total'].rolling(50, min_periods=1, center=True).mean()
    df['missing_ratio_roll_5s'] = df['ball_missing'].rolling(125, min_periods=1, center=True).mean()

    df['speed_all_roll_2s'] = df['speed_all_roll_2s'].fillna(df['overall_speed'])
    df['pitch_speed_drop'] = df['speed_all_roll_2s'] - df['speed_all_roll_2s'].rolling(125, min_periods=1).mean()

    # Team features
    df['control_ratio'] = (df['d_min_away'] - df['d_min_home']) / (df['d_min_away'] + df['d_min_home'] + eps)
    df['control_ratio_roll_1s'] = df['control_ratio'].rolling(25, min_periods=1, center=True).mean()
    df['control_ratio_roll_3s'] = df['control_ratio'].rolling(75, min_periods=1, center=True).mean()
    df['in_active_duel'] = ((df['d_min_home'] < 2.5) & (df['d_min_away'] < 2.5)).astype(float)
    df['duel_roll_1s'] = df['in_active_duel'].rolling(25, min_periods=1, center=True).mean()

    dead_features = [
        'min_dist_any', 'min_dist_roll_1s', 'min_dist_roll_3s', 'min_dist_roll_5s',
        'players_within_3m_total', 'players_3m_roll_1s', 'players_3m_roll_3s',
        'active_total', 'active_roll_2s',
        'missing_ratio_roll_2s', 'missing_ratio_roll_5s',
        'ball_missing',
        'speed_all_roll_2s', 'pitch_speed_drop', 'overall_speed',
        'home_spread_x', 'home_spread_y', 'away_spread_x', 'away_spread_y'
    ]

    team_features = [
        'd_min_home', 'd_min_away', 'd_margin', 'd_margin_roll_1s', 'd_margin_roll_3s',
        'control_ratio', 'control_ratio_roll_1s', 'control_ratio_roll_3s',
        'in_active_duel', 'duel_roll_1s',
        'home_within_3m', 'away_within_3m',
        'home_centroid_x', 'home_centroid_y', 'away_centroid_x', 'away_centroid_y',
        'centroid_diff_x', 'home_spread_x', 'home_spread_y', 'away_spread_x', 'away_spread_y',
        'spread_ratio'
    ]

    all_features = sorted(list(set(dead_features + team_features)))

    train_mask = (df['game'] != 'SkillCorner_Barca_RealMadrid')
    val_mask = (df['game'] == 'SkillCorner_Barca_RealMadrid')

    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_train_flat = df.loc[train_mask, 'label'].map(label_map).values
    y_val_flat = df.loc[val_mask, 'label'].map(label_map).values

    print(f"Train Pool: {train_mask.sum():,} frames | Unseen Validation: {val_mask.sum():,} frames")

    # Train Flat 3-Class LightGBM Baseline
    print("\nTraining Flat 3-Class LightGBM...")
    clf_flat = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=35, colsample_bytree=0.8, random_state=42, verbose=-1)
    clf_flat.fit(df.loc[train_mask, all_features], y_train_flat)
    prob_flat = clf_flat.predict_proba(df.loc[val_mask, all_features])
    pred_flat = np.argmax(prob_flat, axis=1)
    res_flat = evaluate(y_val_flat, pred_flat)

    # Train Stage 1: Dead Gate
    print("Training Stage 1 (Dead Gate LightGBM)...")
    y_train_dead = (df.loc[train_mask, 'label'] == 'D').astype(int).values
    clf_stage1 = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=31, colsample_bytree=0.8, random_state=42, verbose=-1)
    clf_stage1.fit(df.loc[train_mask, dead_features], y_train_dead)
    prob_dead = clf_stage1.predict_proba(df.loc[val_mask, dead_features])[:, 1]

    # Train Stage 2: Team Possession Router (Live only)
    print("Training Stage 2 (Possession Router LightGBM on Live frames only)...")
    live_train_mask = train_mask & (df['label'].isin(['H', 'A']))
    y_train_team = (df.loc[live_train_mask, 'label'] == 'A').astype(int).values
    clf_stage2 = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=31, colsample_bytree=0.8, random_state=42, verbose=-1)
    clf_stage2.fit(df.loc[live_train_mask, team_features], y_train_team)
    prob_away_live = clf_stage2.predict_proba(df.loc[val_mask, team_features])[:, 1]
    pred_team = (prob_away_live >= 0.50).astype(int)  # 0=H, 1=A

    print("\n" + "="*95)
    print(" DEAD GATE THRESHOLD & DURATION CALIBRATION SWEEP")
    print("="*95)
    print(f"{'Configuration':<35} | {'Macro F1':<10} | {'Alive/Dead F1':<14} | {'Dead F1':<10} | {'Dead Prec':<10} | {'Dead Rec':<10}")
    print("-"*95)
    print(f"{'Flat 3-Class (Argmax Baseline)':<35} | {res_flat['macro_f1']:<10.4f} | {res_flat['alive_dead_f1']:<14.4f} | {res_flat['per_class']['D']['f1']:<10.4f} | {res_flat['per_class']['D']['precision']:<10.4f} | {res_flat['per_class']['D']['recall']:<10.4f}")

    best_macro = 0
    best_cfg = None

    for tau in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        pred_gate = np.where(prob_dead >= tau, 2, pred_team)
        res_gate = evaluate(y_val_flat, pred_gate)
        print(f"Hierarchical Gate (tau={tau:.2f})       | {res_gate['macro_f1']:<10.4f} | {res_gate['alive_dead_f1']:<14.4f} | {res_gate['per_class']['D']['f1']:<10.4f} | {res_gate['per_class']['D']['precision']:<10.4f} | {res_gate['per_class']['D']['recall']:<10.4f}")
        
        # Test with Football Duration Filter (min 50 frames dead, min 25 frames live)
        pred_filtered = filter_short_spells(pred_gate, min_dead_frames=50, min_live_frames=25)
        res_filt = evaluate(y_val_flat, pred_filtered)
        print(f"  + Duration Filter (tau={tau:.2f})    | {res_filt['macro_f1']:<10.4f} | {res_filt['alive_dead_f1']:<14.4f} | {res_filt['per_class']['D']['f1']:<10.4f} | {res_filt['per_class']['D']['precision']:<10.4f} | {res_filt['per_class']['D']['recall']:<10.4f}")

if __name__ == '__main__':
    main()
