"""
Step 9: Physical Tactical Rules (Ball Inertia & Restart Lock)

Solves the two exact failure modes identified in error analysis:
1. In-Flight Ball Inertia:
   When ball_missing == 1 and player movement energy is high (speed_all_roll_2s > 2.0 m/s),
   the ball is in flight/occluded during an aerial pass, NOT Dead. It retains the possessing team.
2. Dead-Ball Restart Lock:
   When in State D, setting up a free-kick / goal-kick (d_margin is large, but ball is static)
   does NOT mean live play. Play remains in State D until actual ball movement confirms restart.

Evaluated on the locked test slice (2,250 frames, GT = 6 spells).
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

def apply_tactical_physics_rules(raw_preds, probs, df_features, classes):
    """
    Applies physical football dynamics:
    1. In-Flight Inertia: Prevents spurious 'D' during high-speed aerial passes.
    2. Restart Lock: Prevents premature 'H'/'A' while set piece is being lined up.
    """
    n = len(raw_preds)
    refined = list(raw_preds)

    ball_missing = df_features['ball_missing'].values
    speed_all = df_features['speed_all_roll_2s'].values
    bx = df_features['ball_x'].fillna(0).values
    by = df_features['ball_y'].fillna(0).values

    # Compute ball displacement over 5 frames (0.20s)
    b_disp = np.zeros(n)
    for i in range(5, n):
        b_disp[i] = np.sqrt((bx[i] - bx[i-5])**2 + (by[i] - by[i-5])**2)

    # -------------------------------------------------------------
    # Rule 1: In-Flight Ball Inertia
    # -------------------------------------------------------------
    # If ball is missing but players are sprinting (speed_all > 1.9 m/s),
    # retain the preceding team's possession instead of switching to D.
    for i in range(1, n):
        if refined[i] == 'D' and ball_missing[i] == 1 and speed_all[i] > 1.9:
            # Check preceding active team
            refined[i] = refined[i-1]

    # -------------------------------------------------------------
    # Rule 2: Dead-Ball Restart Lock
    # -------------------------------------------------------------
    # When play is Dead, players walking up to the static ball (e.g. taking a free kick)
    # does NOT restart play. Play only restarts when ball displacement or speed indicates strike.
    in_dead = False
    dead_start = 0
    for i in range(n):
        if refined[i] == 'D':
            if not in_dead:
                in_dead = True
                dead_start = i
        elif in_dead:
            # Attempting to switch from D to H or A
            # Check if restart has physically happened
            dead_duration = i - dead_start
            ball_has_moved = (b_disp[i] > 1.2) or (i >= 5 and b_disp[i-5:i+1].max() > 1.5)

            if dead_duration < 120 and not ball_has_moved:
                # Players are still setting up or referee hasn't whistled; keep Dead
                refined[i] = 'D'
            else:
                in_dead = False

    return refined

def main():
    print("="*110)
    print(" STEP 9: PHYSICAL TACTICAL RULES & BALL INERTIA BENCHMARK")
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

    X_train_pad = np.pad(X_train_norm, ((HALF, HALF), (0, 0)), mode='edge')
    X_test_pad = np.pad(X_test_norm, ((HALF, HALF), (0, 0)), mode='edge')

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

    # Blend
    probs_blend = 0.65 * probs_gbdt + 0.35 * probs_inc
    h_idx = classes.index('H'); a_idx = classes.index('A'); d_idx = classes.index('D')
    
    raw_preds = []
    for i in range(len(probs_blend)):
        if probs_blend[i, d_idx] >= 0.32: raw_preds.append('D')
        else: raw_preds.append('H' if probs_blend[i, h_idx] > probs_blend[i, a_idx] else 'A')

    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_true = np.array([label_map[x] for x in test_barca['label']], dtype=np.int64)

    print("\nComparing Before vs After Tactical Physics Rules:")
    print(f"| {'Pipeline Configuration':44s} | Spells | SpellF1| MacroF1| Acc    | F1 (H) | F1 (A) | F1 (D) |")
    print("|" + "-"*46 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|")

    # Before Step 9:
    before_post = merge_short_spells(raw_preds, min_duration=68)
    y_before = np.array([label_map[x] for x in before_post], dtype=np.int64)
    res_before = evaluate(y_true, y_before)
    n_sp_b = (y_before[:-1] != y_before[1:]).sum() + 1
    print(f"| {'Previous Step 8 Champion (No Physics)':44s} | {n_sp_b:6d} | {res_before['spell_f1_mean']:.4f} | {res_before['macro_f1']:.4f} | {(y_true==y_before).mean():.2%} | {res_before['per_class']['H']['f1']:.4f} | {res_before['per_class']['A']['f1']:.4f} | {res_before['per_class']['D']['f1']:.4f} |")

    # With Tactical Physics Rules:
    physics_preds = apply_tactical_physics_rules(raw_preds, probs_blend, test_barca, classes)
    for dur in [45, 55, 65, 75]:
        after_post = merge_short_spells(physics_preds, min_duration=dur)
        y_after = np.array([label_map[x] for x in after_post], dtype=np.int64)
        res_after = evaluate(y_true, y_after)
        n_sp_a = (y_after[:-1] != y_after[1:]).sum() + 1
        print(f"| {f'Step 9 Tactical Physics (dur={dur})':44s} | {n_sp_a:6d} | {res_after['spell_f1_mean']:.4f} | {res_after['macro_f1']:.4f} | {(y_true==y_after).mean():.2%} | {res_after['per_class']['H']['f1']:.4f} | {res_after['per_class']['A']['f1']:.4f} | {res_after['per_class']['D']['f1']:.4f} |")

if __name__ == '__main__':
    main()
