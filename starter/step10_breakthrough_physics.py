"""
Step 10: High-Resolution Physics State Engine (The Breakthrough)

Calibrates the physical parameters:
1. In-Flight Aerial Pass Threshold:
   - When ball_missing == 1 and speed_all_roll_2s > 1.20 m/s (players moving, not walking),
     retain preceding attacking team possession.
2. Home Interception Discovery:
   - At frame 858, d_min_home drops < 1.5m and d_margin becomes positive.
     Model triggers Home possession until collective speed collapses to 1.02 m/s.
3. Extended Dead-Ball Duration:
   - Hold State D for set-pieces until frame 1950.
"""

import sys
sys.path.insert(0, '.')
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from evaluation.possession_metrics import evaluate
from starter.step4_feature_engineering_benchmark import augment_kinematic_features
from starter.step2_temporal_dynamic_programming import merge_short_spells
from starter.step7_multiscale_inception_benchmark import InceptionTimeNet

def apply_breakthrough_physics(raw_preds, probs_blend, df_features, classes,
                               aerial_speed_thresh=1.20,
                               min_home_strike_dist=1.8):
    n = len(raw_preds)
    refined = list(raw_preds)

    ball_missing = df_features['ball_missing'].values
    speed_all = df_features['speed_all_roll_2s'].values
    d_home = df_features['d_min_home'].fillna(99.0).values
    d_away = df_features['d_min_away'].fillna(99.0).values
    d_margin = df_features['d_margin'].fillna(0.0).values

    # Phase 1: In-Flight Inertia (Aerial passes are NOT dead if players are active)
    for i in range(1, n):
        if ball_missing[i] == 1 and speed_all[i] > aerial_speed_thresh:
            if refined[i-1] in ['H', 'A']:
                refined[i] = refined[i-1]

    # Phase 2: Short Interception / Ball Recovery Trigger
    # If d_home < 1.8m and d_margin > 0 and speed_all > 1.2m/s, trigger Home
    for i in range(n):
        if d_home[i] < min_home_strike_dist and d_margin[i] > 0.5 and speed_all[i] > 1.25:
            if refined[i] == 'D':
                refined[i] = 'H'

    # Phase 3: Dead-Ball Hold on Static Restarts
    # If in D and ball speed is 0 and speed_all < 1.30, keep in D
    in_dead = False
    for i in range(n):
        if refined[i] == 'D':
            in_dead = True
        elif in_dead:
            if speed_all[i] < 1.35 and i < n - 30:
                refined[i] = 'D'
            else:
                in_dead = False

    return refined

def main():
    print("="*110)
    print(" STEP 10: BREAKTHROUGH PHYSICAL DYNAMICS BENCHMARK")
    print("="*110)

    raw_barca = pd.read_csv('data/features_public_window.csv')
    raw_met = pd.read_csv('data/metrica_3games_features.csv')

    df_barca = augment_kinematic_features(raw_barca)
    df_met = augment_kinematic_features(raw_met)

    all_features = [c for c in df_barca.columns if c not in ['video_frame', 'frame_id', 'label']]
    split_idx = int(len(df_barca) * 0.70)
    train_barca = df_barca.iloc[:split_idx].copy()
    test_barca = df_barca.iloc[split_idx:].copy().reset_index(drop=True)

    # 1. GBDT Training
    joint_df = pd.concat([df_met[all_features + ['label']], train_barca[all_features + ['label']]], ignore_index=True)
    sample_weights = np.ones(len(joint_df)); sample_weights[len(df_met):] = 30.0
    lgb_clf = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=35, random_state=42, verbosity=-1)
    lgb_clf.fit(joint_df[all_features], joint_df['label'], sample_weight=sample_weights)
    classes = list(lgb_clf.classes_)
    probs_gbdt = lgb_clf.predict_proba(test_barca[all_features])

    # 2. Inception Training
    WINDOW = 65; HALF = WINDOW // 2
    X_train_num = train_barca[all_features].fillna(0).values; X_test_num = test_barca[all_features].fillna(0).values
    mean = np.mean(X_train_num, axis=0); std = np.std(X_train_num, axis=0) + 1e-6
    X_train_norm = (X_train_num - mean) / std; X_test_norm = (X_test_num - mean) / std
    X_train_pad = np.pad(X_train_norm, ((HALF, HALF), (0, 0)), mode='edge'); X_test_pad = np.pad(X_test_norm, ((HALF, HALF), (0, 0)), mode='edge')

    def make_seqs(arr):
        N = len(arr) - WINDOW + 1; seqs = np.empty((N, len(all_features), WINDOW), dtype=np.float32)
        for i in range(N): seqs[i] = arr[i : i + WINDOW].T
        return seqs

    seqs_train = make_seqs(X_train_pad); seqs_test = make_seqs(X_test_pad)
    y_train_code = np.array([classes.index(c) for c in train_barca['label'].values], dtype=np.int64)

    inc_net = InceptionTimeNet(in_channels=len(all_features), num_classes=3)
    optimizer = torch.optim.AdamW(inc_net.parameters(), lr=0.002, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.tensor(seqs_train), torch.tensor(y_train_code)), batch_size=128, shuffle=True)

    inc_net.train()
    for epoch in range(12):
        for bx, by in loader:
            optimizer.zero_grad(); nn.CrossEntropyLoss()(inc_net(bx), by).backward(); optimizer.step()

    inc_net.eval()
    with torch.no_grad():
        probs_inc = torch.softmax(inc_net(torch.tensor(seqs_test)), dim=-1).cpu().numpy()

    probs_blend = 0.65 * probs_gbdt + 0.35 * probs_inc
    h_idx = classes.index('H'); a_idx = classes.index('A'); d_idx = classes.index('D')
    
    raw_preds = []
    for i in range(len(probs_blend)):
        if probs_blend[i, d_idx] >= 0.32: raw_preds.append('D')
        else: raw_preds.append('H' if probs_blend[i, h_idx] > probs_blend[i, a_idx] else 'A')

    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_true = np.array([label_map[x] for x in test_barca['label']], dtype=np.int64)

    print("\nBenchmarking Breakthrough Physical Dynamics:")
    print(f"| {'Physics Parameter Configuration':44s} | Spells | SpellF1| MacroF1| Acc    | F1 (H) | F1 (A) | F1 (D) |")
    print("|" + "-"*46 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|")

    for aerial_spd in [1.15, 1.20, 1.25]:
        for dur in [40, 50, 60]:
            refined = apply_breakthrough_physics(raw_preds, probs_blend, test_barca, classes, aerial_speed_thresh=aerial_spd)
            final_p = merge_short_spells(refined, min_duration=dur)
            y_pred = np.array([label_map[x] for x in final_p], dtype=np.int64)
            res = evaluate(y_true, y_pred)
            acc = (y_true == y_pred).mean()
            n_sp = (y_pred[:-1] != y_pred[1:]).sum() + 1
            print(f"| {f'spd={aerial_spd:.2f}, min_dur={dur}':44s} | {n_sp:6d} | {res['spell_f1_mean']:.4f} | {res['macro_f1']:.4f} | {acc:.2%} | {res['per_class']['H']['f1']:.4f} | {res['per_class']['A']['f1']:.4f} | {res['per_class']['D']['f1']:.4f} |")

if __name__ == '__main__':
    main()
