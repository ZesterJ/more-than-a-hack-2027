"""
Step 4: Physical Kinematics & Velocity Alignment Feature Engineering Benchmark

Computes 10 advanced physical and kinematic features:
1. ball_out_of_bounds: Euclidean distance outside standard pitch boundary.
2. is_out_of_bounds: Binary indicator of ball leaving field of play.
3. ball_accel: Second-order acceleration norm of the ball (identifying kicks/strikes).
4. d_margin_velocity: Temporal rate of change of territorial distance margin.
5. dwell_home_2m: Consecutive frames ball has stayed under home control radius.
6. dwell_away_2m: Consecutive frames ball has stayed under away control radius.
7. centroid_push_velocity: Rate of team territorial advance along x-axis.
8. stoppage_energy_index: Interaction between player speed and sensor occlusion.
9. pressure_differential: home_within_3m - away_within_3m.
10. ratio_d_min: d_min_home / (d_min_away + 1e-4).

Evaluates the combined pipeline (Steps 1 + 2 + 3 + 4) on the 30% unseen test slice.
"""

import sys
sys.path.insert(0, '.')
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from evaluation.possession_metrics import evaluate
from starter.step2_temporal_dynamic_programming import merge_short_spells

def augment_kinematic_features(df):
    """Derives second-order physical and kinematic features from trajectory columns."""
    df = df.copy()

    # 1. Pitch Bounds (105.3m x 68.0m -> bounds at 52.65m and 34.0m)
    dist_x = np.maximum(0, np.abs(df['ball_x'].fillna(0)) - 52.65)
    dist_y = np.maximum(0, np.abs(df['ball_y'].fillna(0)) - 34.0)
    df['ball_out_of_bounds'] = np.sqrt(dist_x**2 + dist_y**2)
    df['is_out_of_bounds'] = (df['ball_out_of_bounds'] > 0.5).astype(int)

    # 2. Ball Kinematics
    vx = df['ball_x'].diff().fillna(0) * 25.0
    vy = df['ball_y'].diff().fillna(0) * 25.0
    ax = vx.diff().fillna(0) * 25.0
    ay = vy.diff().fillna(0) * 25.0
    df['ball_accel'] = np.sqrt(ax**2 + ay**2)

    # 3. Territorial Momentum
    df['d_margin_velocity'] = df['d_margin'].diff().fillna(0) * 25.0
    df['centroid_push_velocity'] = df['centroid_diff_x'].diff().fillna(0) * 25.0

    # 4. Control Dwell History
    home_near = (df['d_min_home'] < 2.0).astype(int)
    away_near = (df['d_min_away'] < 2.0).astype(int)
    
    # Calculate consecutive run lengths
    def calc_dwell(series):
        dwell = np.zeros(len(series))
        cnt = 0
        for i, val in enumerate(series):
            if val == 1:
                cnt += 1
            else:
                cnt = 0
            dwell[i] = cnt
        return dwell

    df['dwell_home_2m'] = calc_dwell(home_near)
    df['dwell_away_2m'] = calc_dwell(away_near)

    # 5. Stoppage & Pressure Indices
    df['stoppage_energy_index'] = df['speed_all_roll_2s'] * (1.0 - df['missing_ratio_roll_2s'])
    df['pressure_differential'] = df['home_within_3m'] - df['away_within_3m']
    df['ratio_d_min'] = df['d_min_home'] / (df['d_min_away'] + 1e-4)

    return df

def score(y_true_str, y_pred_str, name="Model", min_dur=75):
    y_pred_post = merge_short_spells(y_pred_str, min_duration=min_dur)
    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_true = np.array([label_map[x] for x in y_true_str], dtype=np.int64)
    y_pred = np.array([label_map[x] for x in y_pred_post], dtype=np.int64)
    res = evaluate(y_true, y_pred)
    acc = (y_true == y_pred).mean()
    n_sp = (y_pred[:-1] != y_pred[1:]).sum() + 1
    print(f"| {name:42s} | {n_sp:6d} | {res['spell_f1_mean']:.4f} | {res['macro_f1']:.4f} | {acc:.2%} | {res['per_class']['H']['f1']:.4f} | {res['per_class']['A']['f1']:.4f} | {res['per_class']['D']['f1']:.4f} |")
    return res

def main():
    print("="*105)
    print(" STEP 4: PHYSICAL KINEMATICS & FEATURE AUGMENTATION BENCHMARK")
    print("="*105)

    # 1. Load Data
    raw_barca = pd.read_csv('data/features_public_window.csv')
    raw_met = pd.read_csv('data/metrica_3games_features.csv')

    print("Augmenting kinematic & physical features on Barca and Metrica datasets...")
    df_barca = augment_kinematic_features(raw_barca)
    df_met = augment_kinematic_features(raw_met)

    all_features = [c for c in df_barca.columns if c not in ['video_frame', 'frame_id', 'label']]
    base_features = [c for c in raw_barca.columns if c not in ['video_frame', 'frame_id', 'label']]
    print(f"Base Features: {len(base_features)} | Augmented Features: {len(all_features)}")

    # Chronological Split
    split_idx = int(len(df_barca) * 0.70)
    train_barca = df_barca.iloc[:split_idx].copy()
    test_barca = df_barca.iloc[split_idx:].copy()

    joint_df = pd.concat([df_met[all_features + ['label']], train_barca[all_features + ['label']]], ignore_index=True)
    sample_weights = np.ones(len(joint_df))
    sample_weights[len(df_met):] = 30.0  # Optimal weight from Step 3

    print(f"| {'Model / Feature Configuration':42s} | Spells | SpellF1| MacroF1| Acc    | F1 (H) | F1 (A) | F1 (D) |")
    print("|" + "-"*44 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|")

    # A. Step 3 Champion (Base Features only)
    joint_base = pd.concat([raw_met[base_features + ['label']], raw_barca.iloc[:split_idx][base_features + ['label']]], ignore_index=True)
    clf_base = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=35, random_state=42, verbosity=-1)
    clf_base.fit(joint_base[base_features], joint_base['label'], sample_weight=sample_weights)
    score(test_barca['label'].values, clf_base.predict(test_barca[base_features]), "Step 3 Benchmark (28 Base Features)", min_dur=75)

    # B. Step 4 with Augmented Kinematic Features
    clf_aug = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=35, random_state=42, verbosity=-1)
    clf_aug.fit(joint_df[all_features], joint_df['label'], sample_weight=sample_weights)
    score(test_barca['label'].values, clf_aug.predict(test_barca[all_features]), "Step 4 Augmented (38 Features, dur=75)", min_dur=75)

    # C. Step 4 with Duration Filter Tuning
    for dur in [50, 75, 100, 125]:
        score(test_barca['label'].values, clf_aug.predict(test_barca[all_features]), f"Step 4 Augmented (dur={dur} frames / {dur/25:.1f}s)", min_dur=dur)

    # D. Feature Importance of Augmented Kinematic Signals
    importances = clf_aug.feature_importances_
    feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False)
    print("\nTop 12 Most Predictive Features in the Augmented Model:")
    for feat, imp in feat_imp.head(12).items():
        print(f"  {feat:26s}: {imp:5d}")

if __name__ == '__main__':
    main()
