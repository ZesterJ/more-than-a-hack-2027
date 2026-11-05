"""
Test Specialized Physics & Boundary Features for Stage 1 Dead Detector
Strict Nov 3 Rule Compliance (100% External Cross-Validation)
"""

import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, f1_score

def add_dead_physics_features(df):
    eps = 1e-4
    # 1. Pitch boundary distance & out-of-bounds indicator
    # Pitch bounds: x in [0, 105], y in [0, 68]
    bx = df['ball_x'].fillna(52.5)
    by = df['ball_y'].fillna(34.0)
    
    df['dist_to_left_line'] = bx
    df['dist_to_right_line'] = 105.0 - bx
    df['dist_to_bottom_line'] = by
    df['dist_to_top_line'] = 68.0 - by
    df['dist_to_boundary'] = np.minimum(
        np.minimum(df['dist_to_left_line'], df['dist_to_right_line']),
        np.minimum(df['dist_to_bottom_line'], df['dist_to_top_line'])
    )
    df['is_out_of_bounds'] = (df['dist_to_boundary'] < 0.0).astype(float)
    df['near_boundary_1m'] = (df['dist_to_boundary'] < 1.0).astype(float)
    df['boundary_roll_1s'] = df['is_out_of_bounds'].rolling(25, min_periods=1, center=True).mean()

    # 2. Consecutive missing ball count
    missing = df['ball_missing'].values
    n = len(missing)
    consec_missing = np.zeros(n, dtype=float)
    c = 0
    for i in range(n):
        if missing[i] == 1:
            c += 1
        else:
            c = 0
        consec_missing[i] = c
    df['consecutive_missing'] = consec_missing
    df['missing_over_1s'] = (consec_missing >= 25).astype(float)
    df['missing_over_3s'] = (consec_missing >= 75).astype(float)

    # 3. Whole pitch collective deceleration (whistle signature)
    df['speed_all_roll_2s'] = df['speed_all_roll_2s'].fillna(df['overall_speed'])
    pitch_speed = df['speed_all_roll_2s'].values
    df['pitch_decel_2s'] = pd.Series(pitch_speed).diff(50).fillna(0) # change over 2s
    df['is_collective_walking'] = (df['speed_all_roll_2s'] < 1.2).astype(float)
    df['walking_roll_2s'] = df['is_collective_walking'].rolling(50, min_periods=1, center=True).mean()

    # 4. Nearest outfield player
    df['min_dist_any'] = np.minimum(df['d_min_home'].fillna(99), df['d_min_away'].fillna(99))
    df['min_dist_roll_1s'] = df['min_dist_any'].rolling(25, min_periods=1, center=True).mean()
    df['min_dist_roll_3s'] = df['min_dist_any'].rolling(75, min_periods=1, center=True).mean()
    df['ball_isolated_5m'] = (df['min_dist_any'] > 5.0).astype(float)
    df['ball_isolated_10m'] = (df['min_dist_any'] > 10.0).astype(float)

    # 5. Team dispersion expansion during stoppages
    df['total_spread_area'] = (df['home_spread_x'] * df['home_spread_y']) + (df['away_spread_x'] * df['away_spread_y'])
    df['spread_area_roll_2s'] = df['total_spread_area'].rolling(50, min_periods=1, center=True).mean()

    return df

def main():
    print("="*90)
    print(" TESTING SPECIALIZED STAGE 1 DEAD-BALL PHYSICS FEATURES")
    print("="*90)

    df_met = pd.read_csv('data/metrica_3games_features.csv')
    df_sc = pd.read_csv('data/skillcorner_barca_mancity_features.csv')
    df = pd.concat([df_met, df_sc], ignore_index=True)

    df = add_dead_physics_features(df)
    df['is_dead'] = (df['label'] == 'D').astype(int)

    base_dead_features = [
        'min_dist_any', 'min_dist_roll_1s', 'min_dist_roll_3s',
        'home_within_3m', 'away_within_3m',
        'home_active', 'away_active',
        'missing_ratio_roll_2s', 'ball_missing',
        'speed_all_roll_2s', 'overall_speed',
        'home_spread_x', 'home_spread_y', 'away_spread_x', 'away_spread_y'
    ]

    new_dead_features = base_dead_features + [
        'dist_to_boundary', 'is_out_of_bounds', 'near_boundary_1m', 'boundary_roll_1s',
        'consecutive_missing', 'missing_over_1s', 'missing_over_3s',
        'pitch_decel_2s', 'is_collective_walking', 'walking_roll_2s',
        'ball_isolated_5m', 'ball_isolated_10m',
        'total_spread_area', 'spread_area_roll_2s'
    ]

    train_mask = (df['game'] != 'SkillCorner_Barca_RealMadrid')
    val_mask = (df['game'] == 'SkillCorner_Barca_RealMadrid')

    y_train = df.loc[train_mask, 'is_dead'].values
    y_val = df.loc[val_mask, 'is_dead'].values

    # Base features
    clf_base = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=31, colsample_bytree=0.8, random_state=42, verbose=-1)
    clf_base.fit(df.loc[train_mask, base_dead_features], y_train)
    prob_base = clf_base.predict_proba(df.loc[val_mask, base_dead_features])[:, 1]
    pred_base = (prob_base >= 0.5).astype(int)

    # New physics features
    clf_new = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=31, colsample_bytree=0.8, random_state=42, verbose=-1)
    clf_new.fit(df.loc[train_mask, new_dead_features], y_train)
    prob_new = clf_new.predict_proba(df.loc[val_mask, new_dead_features])[:, 1]
    pred_new = (prob_new >= 0.5).astype(int)

    auc_base = roc_auc_score(y_val, prob_base)
    auc_new = roc_auc_score(y_val, prob_new)
    f1_dead_base = f1_score(y_val, pred_base, pos_label=1)
    f1_dead_new = f1_score(y_val, pred_new, pos_label=1)
    f1_macro_base = f1_score(y_val, pred_base, average='macro')
    f1_macro_new = f1_score(y_val, pred_new, average='macro')

    print(f"\n--- STAGE 1 DEAD-BALL GATE COMPARISON ---")
    print(f"Base Features ({len(base_dead_features)} feats):")
    print(f"  ROC-AUC:  {auc_base:.4f} | Dead F1: {f1_dead_base:.4f} | Macro F1: {f1_macro_base:.4f}")
    print(f"New Physics Features ({len(new_dead_features)} feats):")
    print(f"  ROC-AUC:  {auc_new:.4f} | Dead F1: {f1_dead_new:.4f} | Macro F1: {f1_macro_new:.4f}")
    print(f"Delta:")
    print(f"  ROC-AUC:  {auc_new - auc_base:+.4f} | Dead F1: {f1_dead_new - f1_dead_base:+.4f} | Macro F1: {f1_macro_new - f1_macro_base:+.4f}")

    imps = sorted(zip(new_dead_features, clf_new.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Most Important Dead-Ball Features:")
    for f, imp in imps[:10]:
        print(f"  {f:25s}: {imp:4d}")

if __name__ == '__main__':
    main()
