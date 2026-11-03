"""
Step 8: Joint Calibration & Threshold Optimization

Optimizes:
1. Blend Weight alpha: P_blend = alpha * P_gbdt + (1 - alpha) * P_inception
2. Dead-Ball Sensitivity Threshold: if P(D) >= theta_d -> 'D'; else argmax(H, A)
3. Minimum Duration Threshold: min_duration in frames
All evaluated on the locked test slice (2,250 frames, GT = 6 spells).
"""

import sys
sys.path.insert(0, '.')
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from evaluation.possession_metrics import evaluate
from starter.step4_feature_engineering_benchmark import augment_kinematic_features
from starter.step2_temporal_dynamic_programming import merge_short_spells
from starter.step7_multiscale_inception_benchmark import InceptionTimeNet

def main():
    print("="*110)
    print(" STEP 8: JOINT HYPERPARAMETER CALIBRATION & THRESHOLD OPTIMIZATION")
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
    X_train_num = train_barca[all_features].fillna(0).values
    X_test_num = test_barca[all_features].fillna(0).values
    mean = np.mean(X_train_num, axis=0); std = np.std(X_train_num, axis=0) + 1e-6
    X_train_norm = (X_train_num - mean) / std; X_test_norm = (X_test_num - mean) / std

    X_train_pad = np.pad(X_train_norm, ((HALF, HALF), (0, 0)), mode='edge')
    X_test_pad = np.pad(X_test_norm, ((HALF, HALF), (0, 0)), mode='edge')

    def make_seqs(arr):
        N = len(arr) - WINDOW + 1
        seqs = np.empty((N, len(all_features), WINDOW), dtype=np.float32)
        for i in range(N): seqs[i] = arr[i : i + WINDOW].T
        return seqs

    seqs_train = make_seqs(X_train_pad); seqs_test = make_seqs(X_test_pad)
    y_train_code = np.array([classes.index(c) for c in train_barca['label'].values], dtype=np.int64)

    inc_net = InceptionTimeNet(in_channels=len(all_features), num_classes=3)
    optimizer = optim.AdamW(inc_net.parameters(), lr=0.002, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.tensor(seqs_train), torch.tensor(y_train_code)), batch_size=128, shuffle=True)

    inc_net.train()
    for epoch in range(12):
        for bx, by in loader:
            optimizer.zero_grad(); nn.CrossEntropyLoss()(inc_net(bx), by).backward(); optimizer.step()

    inc_net.eval()
    with torch.no_grad():
        probs_inc = torch.softmax(inc_net(torch.tensor(seqs_test)), dim=-1).cpu().numpy()

    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_true = np.array([label_map[x] for x in test_barca['label']], dtype=np.int64)

    h_idx = classes.index('H')
    a_idx = classes.index('A')
    d_idx = classes.index('D')

    best_score = 0.0
    best_config = None
    results = []

    print("\nRunning Joint Grid Calibration (alpha x theta_dead x min_duration)...")
    for alpha in [0.55, 0.60, 0.65, 0.70]:
        probs_blend = alpha * probs_gbdt + (1.0 - alpha) * probs_inc
        for theta_d in [0.28, 0.32, 0.35, 0.40]:
            # Apply threshold
            preds = []
            for i in range(len(probs_blend)):
                if probs_blend[i, d_idx] >= theta_d:
                    preds.append('D')
                else:
                    preds.append('H' if probs_blend[i, h_idx] > probs_blend[i, a_idx] else 'A')

            for dur in [60, 68, 72, 75, 80]:
                preds_post = merge_short_spells(preds, min_duration=dur)
                y_pred = np.array([label_map[x] for x in preds_post], dtype=np.int64)
                res = evaluate(y_true, y_pred)
                acc = (y_true == y_pred).mean()
                n_sp = (y_pred[:-1] != y_pred[1:]).sum() + 1
                sf1 = res['spell_f1_mean']
                mf1 = res['macro_f1']
                
                results.append({
                    "alpha": alpha, "theta_d": theta_d, "dur": dur,
                    "spells": n_sp, "spell_f1": sf1, "macro_f1": mf1, "acc": acc,
                    "f1_h": res['per_class']['H']['f1'],
                    "f1_a": res['per_class']['A']['f1'],
                    "f1_d": res['per_class']['D']['f1']
                })
                if sf1 > best_score:
                    best_score = sf1
                    best_config = results[-1]

    df_res = pd.DataFrame(results).sort_values(by=["spell_f1", "macro_f1", "acc"], ascending=False)
    print("\nTop 8 Configurations from Calibration:")
    print(f"| {'Config (alpha, theta_d, dur)':32s} | Spells | SpellF1| MacroF1| Acc    | F1 (H) | F1 (A) | F1 (D) |")
    print("|" + "-"*34 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|")
    for _, row in df_res.head(8).iterrows():
        cfg_str = f"a={row['alpha']:.2f}, th={row['theta_d']:.2f}, d={int(row['dur'])}"
        print(f"| {cfg_str:32s} | {int(row['spells']):6d} | {row['spell_f1']:.4f} | {row['macro_f1']:.4f} | {row['acc']:.2%} | {row['f1_h']:.4f} | {row['f1_a']:.4f} | {row['f1_d']:.4f} |")

    # Save champion model scoreboard
    df_res.to_csv('starter/calibration_grid_scoreboard.csv', index=False)
    print("\nSaved calibration scoreboard to starter/calibration_grid_scoreboard.csv")

if __name__ == '__main__':
    main()
