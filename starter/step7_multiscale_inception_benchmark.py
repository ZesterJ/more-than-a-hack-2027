"""
Step 7: Multi-Scale Temporal Inception CNN Benchmark

Implements parallel multi-scale temporal convolution kernels:
- k=3 (0.12s): Instant ball strike / tackle impact
- k=9 (0.36s): Passing trajectory
- k=25 (1.00s): Continuous ball carry / sprint
- k=51 (2.04s): Sustained phase of play
Evaluates standalone InceptionTime vs ResNet-1D, and blends with LightGBM on the locked test set.
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

torch.manual_seed(42)
np.random.seed(42)

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels=24):
        super().__init__()
        bottleneck = max(out_channels // 2, 8)
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1)
        
        # Multi-scale parallel branches
        self.conv1 = nn.Conv1d(bottleneck, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(bottleneck, out_channels, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(bottleneck, out_channels, kernel_size=25, padding=12)
        self.conv4 = nn.Conv1d(bottleneck, out_channels, kernel_size=51, padding=25)
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.bn = nn.BatchNorm1d(out_channels * 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        b = self.bottleneck(x)
        c1 = self.conv1(b)
        c2 = self.conv2(b)
        c3 = self.conv3(b)
        c4 = self.conv4(b)
        cp = self.pool_branch(x)
        out = torch.cat([c1, c2, c3, c4, cp], dim=1)
        return self.relu(self.bn(out))

class InceptionTimeNet(nn.Module):
    def __init__(self, in_channels, num_classes=3):
        super().__init__()
        self.block1 = InceptionBlock1D(in_channels, out_channels=20)  # 20 * 5 = 100
        self.shortcut = nn.Conv1d(in_channels, 100, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.shortcut(x)
        x = self.relu(self.block1(x) + res)
        feat = self.pool(x).squeeze(-1)
        return self.fc(feat)

def main():
    print("="*110)
    print(" STEP 7: MULTI-SCALE TEMPORAL INCEPTION BENCHMARK")
    print("="*110)

    raw_barca = pd.read_csv('data/features_public_window.csv')
    raw_met = pd.read_csv('data/metrica_3games_features.csv')

    df_barca = augment_kinematic_features(raw_barca)
    df_met = augment_kinematic_features(raw_met)

    all_features = [c for c in df_barca.columns if c not in ['video_frame', 'frame_id', 'label']]
    split_idx = int(len(df_barca) * 0.70)
    train_barca = df_barca.iloc[:split_idx].copy()
    test_barca = df_barca.iloc[split_idx:].copy().reset_index(drop=True)

    # 1. GBDT Champion
    joint_df = pd.concat([df_met[all_features + ['label']], train_barca[all_features + ['label']]], ignore_index=True)
    sample_weights = np.ones(len(joint_df)); sample_weights[len(df_met):] = 30.0
    lgb_clf = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=35, random_state=42, verbosity=-1)
    lgb_clf.fit(joint_df[all_features], joint_df['label'], sample_weight=sample_weights)
    classes = list(lgb_clf.classes_)
    probs_gbdt = lgb_clf.predict_proba(test_barca[all_features])

    # 2. InceptionTime Training (65-frame temporal window = 2.6s context)
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

    print("Training Multi-Scale InceptionTime Network (k=3, 9, 25, 51)...")
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

    # 3. Evaluate on Locked Test Slice
    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_true = np.array([label_map[x] for x in test_barca['label']], dtype=np.int64)

    print("\nResults on Locked Test Slice (2,250 frames):")
    print(f"| {'Architecture / Blend':42s} | Spells | SpellF1| MacroF1| Acc    | F1 (H) | F1 (A) | F1 (D) |")
    print("|" + "-"*44 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|")

    # Inception alone
    preds_inc_raw = [classes[i] for i in np.argmax(probs_inc, axis=1)]
    p_inc_post = merge_short_spells(preds_inc_raw, min_duration=75)
    y_pred_inc = np.array([label_map[x] for x in p_inc_post], dtype=np.int64)
    res_inc = evaluate(y_true, y_pred_inc)
    n_sp = (y_pred_inc[:-1] != y_pred_inc[1:]).sum() + 1
    print(f"| {'InceptionTime Alone (dur=75)':42s} | {n_sp:6d} | {res_inc['spell_f1_mean']:.4f} | {res_inc['macro_f1']:.4f} | {(y_true==y_pred_inc).mean():.2%} | {res_inc['per_class']['H']['f1']:.4f} | {res_inc['per_class']['A']['f1']:.4f} | {res_inc['per_class']['D']['f1']:.4f} |")

    # Blends with GBDT
    for alpha in [0.65, 0.70, 0.75, 0.80]:
        for dur in [70, 75, 80]:
            probs_blend = alpha * probs_gbdt + (1.0 - alpha) * probs_inc
            preds_blend = [classes[i] for i in np.argmax(probs_blend, axis=1)]
            p_post = merge_short_spells(preds_blend, min_duration=dur)
            y_pred_b = np.array([label_map[x] for x in p_post], dtype=np.int64)
            res_b = evaluate(y_true, y_pred_b)
            n_sp = (y_pred_b[:-1] != y_pred_b[1:]).sum() + 1
            if res_b['spell_f1_mean'] >= 0.70:
                print(f"| {f'GBDT + Inception Blend (a={alpha:.2f}, d={dur})':42s} | {n_sp:6d} | {res_b['spell_f1_mean']:.4f} | {res_b['macro_f1']:.4f} | {(y_true==y_pred_b).mean():.2%} | {res_b['per_class']['H']['f1']:.4f} | {res_b['per_class']['A']['f1']:.4f} | {res_b['per_class']['D']['f1']:.4f} |")

if __name__ == '__main__':
    main()
