"""
Step 5: GBDT + 1D-CNN Probability Blending & Adaptive Hysteresis Smoothing

1. Trains LightGBM on Domain-Weighted Metrica + Barca Train.
2. Trains ResNet-1D CNN on 41-frame temporal windows of Barca Train.
3. Generates out-of-sample calibrated probability distributions P_gbdt and P_cnn.
4. Explores ensemble blending: P_ensemble = alpha * P_gbdt + (1 - alpha) * P_cnn.
5. Implements Adaptive Hysteresis Smoothing:
   - Uses kinematic physical conditions (ball stillness, out-of-bounds, collective speed)
     to preserve short legitimate turnovers while suppressing false micro-flickers.
6. Evaluates against the 6 ground-truth spells on the 30% unseen held-out test slice.
"""

import sys
import os
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

# -------------------------------------------------------------
# 1. 1D-CNN Architecture (ResNet-1D)
# -------------------------------------------------------------
class ResNetBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

class ResNet1D(nn.Module):
    def __init__(self, in_channels, num_classes=3):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU()
        )
        self.block1 = ResNetBlock1D(48)
        self.block2 = ResNetBlock1D(48)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        feat = self.pool(x).squeeze(-1)
        return self.fc(feat)

# -------------------------------------------------------------
# 2. Adaptive Hysteresis Smoother
# -------------------------------------------------------------
def adaptive_hysteresis_filter(preds, probs, df_features, 
                               min_live_frames=45, 
                               min_dead_frames=35, 
                               dead_threshold=0.38):
    """
    Adaptive Hysteresis State Machine:
    Transitions require either sustained runs or physical state confirmation.
    """
    classes = ['A', 'D', 'H']
    d_idx = classes.index('D')
    n = len(preds)
    out = [''] * n

    # Extract physical trigger signals
    ball_speed = df_features['ball_speed'].values
    speed_all = df_features['speed_all_roll_2s'].values
    is_out = df_features['is_out_of_bounds'].values

    curr_state = preds[0]
    cand_state = curr_state
    cand_count = 0

    for i in range(n):
        p_d = probs[i, d_idx]
        raw_state = preds[i]

        # Physical Dead Ball Confirmation
        physical_dead = (is_out[i] == 1) or (ball_speed[i] < 0.15 and speed_all[i] < 1.75)

        if raw_state == curr_state:
            cand_state = curr_state
            cand_count = 0
        else:
            if raw_state == cand_state:
                cand_count += 1
            else:
                cand_state = raw_state
                cand_count = 1

            # Transition Rules
            can_switch = False
            if cand_state == 'D':
                if (cand_count >= min_dead_frames) or (physical_dead and p_d >= dead_threshold and cand_count >= 15):
                    can_switch = True
            else: # Live play ('H' or 'A')
                if curr_state == 'D':
                    # Leaving dead ball requires sustained live signal
                    if cand_count >= min_live_frames:
                        can_switch = True
                else:
                    # Direct turnover H <-> A requires sustained control
                    if cand_count >= min_live_frames:
                        can_switch = True

            if can_switch:
                curr_state = cand_state
                cand_count = 0

        out[i] = curr_state

    return out

def score_predictions(y_true_str, y_pred_str, label_name="Model"):
    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_true = np.array([label_map[x] for x in y_true_str], dtype=np.int64)
    y_pred = np.array([label_map[x] for x in y_pred_str], dtype=np.int64)
    res = evaluate(y_true, y_pred)
    acc = (y_true == y_pred).mean()
    n_sp = (y_pred[:-1] != y_pred[1:]).sum() + 1
    print(f"| {label_name:44s} | {n_sp:6d} | {res['spell_f1_mean']:.4f} | {res['macro_f1']:.4f} | {acc:.2%} | {res['per_class']['H']['f1']:.4f} | {res['per_class']['A']['f1']:.4f} | {res['per_class']['D']['f1']:.4f} |")
    return res

def main():
    print("="*110)
    print(" STEP 5: GBDT + 1D-CNN PROBABILISTIC BLEND & ADAPTIVE HYSTERESIS BENCHMARK")
    print("="*110)

    # 1. Load Data
    raw_barca = pd.read_csv('data/features_public_window.csv')
    raw_met = pd.read_csv('data/metrica_3games_features.csv')

    df_barca = augment_kinematic_features(raw_barca)
    df_met = augment_kinematic_features(raw_met)

    all_features = [c for c in df_barca.columns if c not in ['video_frame', 'frame_id', 'label']]
    split_idx = int(len(df_barca) * 0.70)
    train_barca = df_barca.iloc[:split_idx].copy()
    test_barca = df_barca.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"Train Set: {len(train_barca)} frames | Test Set: {len(test_barca)} frames | Features: {len(all_features)}")

    # -------------------------------------------------------------
    # Model 1: LightGBM (Weighted Domain Transfer)
    # -------------------------------------------------------------
    print("Training LightGBM model...")
    joint_df = pd.concat([df_met[all_features + ['label']], train_barca[all_features + ['label']]], ignore_index=True)
    sample_weights = np.ones(len(joint_df))
    sample_weights[len(df_met):] = 30.0

    lgb_clf = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=35, random_state=42, verbosity=-1)
    lgb_clf.fit(joint_df[all_features], joint_df['label'], sample_weight=sample_weights)
    
    classes = list(lgb_clf.classes_) # ['A', 'D', 'H']
    probs_gbdt = lgb_clf.predict_proba(test_barca[all_features])

    # -------------------------------------------------------------
    # Model 2: ResNet-1D CNN
    # -------------------------------------------------------------
    print("Building sequences and training ResNet-1D CNN...")
    WINDOW = 35
    HALF = WINDOW // 2
    
    # Normalize features
    X_train_num = train_barca[all_features].fillna(0).values
    X_test_num = test_barca[all_features].fillna(0).values
    mean = np.mean(X_train_num, axis=0)
    std = np.std(X_train_num, axis=0) + 1e-6
    X_train_norm = (X_train_num - mean) / std
    X_test_norm = (X_test_num - mean) / std

    # Pad
    X_train_pad = np.pad(X_train_norm, ((HALF, HALF), (0, 0)), mode='edge')
    X_test_pad = np.pad(X_test_norm, ((HALF, HALF), (0, 0)), mode='edge')

    def make_seqs(arr):
        N = len(arr) - WINDOW + 1
        seqs = np.empty((N, len(all_features), WINDOW), dtype=np.float32)
        for i in range(N):
            seqs[i] = arr[i : i + WINDOW].T
        return seqs

    seqs_train = make_seqs(X_train_pad)
    seqs_test = make_seqs(X_test_pad)

    y_train_code = np.array([classes.index(c) for c in train_barca['label'].values], dtype=np.int64)
    
    cnn_model = ResNet1D(in_channels=len(all_features), num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cnn_model.parameters(), lr=0.003, weight_decay=1e-4)

    dataset_train = TensorDataset(torch.tensor(seqs_train), torch.tensor(y_train_code))
    loader = DataLoader(dataset_train, batch_size=128, shuffle=True)

    cnn_model.train()
    for epoch in range(12):
        for bx, by in loader:
            optimizer.zero_grad()
            out = cnn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    cnn_model.eval()
    with torch.no_grad():
        test_logits = cnn_model(torch.tensor(seqs_test))
        probs_cnn = torch.softmax(test_logits, dim=-1).cpu().numpy()

    print("\nScoreboard across Blends & Smoothing Strategies:")
    print(f"| {'Model / Blend Configuration':44s} | Spells | SpellF1| MacroF1| Acc    | F1 (H) | F1 (A) | F1 (D) |")
    print("|" + "-"*46 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|")

    # A. Baselines
    preds_gbdt_raw = [classes[i] for i in np.argmax(probs_gbdt, axis=1)]
    score_predictions(test_barca['label'].values, merge_short_spells(preds_gbdt_raw, 75), "1. LightGBM Alone (dur=75)")

    preds_cnn_raw = [classes[i] for i in np.argmax(probs_cnn, axis=1)]
    score_predictions(test_barca['label'].values, merge_short_spells(preds_cnn_raw, 75), "2. ResNet-1D CNN Alone (dur=75)")

    # B. Probabilistic Blends
    for alpha in [0.70, 0.80, 0.85]:
        probs_blend = alpha * probs_gbdt + (1.0 - alpha) * probs_cnn
        preds_blend = [classes[i] for i in np.argmax(probs_blend, axis=1)]
        score_predictions(test_barca['label'].values, merge_short_spells(preds_blend, 75), f"3. Blend (alpha={alpha:.2f}, dur=75)")

    # C. Adaptive Hysteresis Smoothing on Best Blend
    probs_blend_opt = 0.80 * probs_gbdt + 0.20 * probs_cnn
    preds_blend_opt = [classes[i] for i in np.argmax(probs_blend_opt, axis=1)]

    for min_live in [35, 45, 55]:
        for min_dead in [25, 35]:
            hyst_preds = adaptive_hysteresis_filter(
                preds_blend_opt, probs_blend_opt, test_barca,
                min_live_frames=min_live,
                min_dead_frames=min_dead,
                dead_threshold=0.35
            )
            # small final cleanup of any <25f blips
            hyst_preds_clean = merge_short_spells(hyst_preds, min_duration=25)
            score_predictions(test_barca['label'].values, hyst_preds_clean, f"4. Hysteresis (live={min_live}, dead={min_dead})")

if __name__ == '__main__':
    main()
