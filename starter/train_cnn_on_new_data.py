"""Train Temporal 1D Convolutional Neural Networks on the Unified 650k Dataset.

Evaluates two top-tier 1D CNN architectures:
1. InceptionTime-1D: Multi-scale parallel receptive fields (k=3, 7, 15, 31)
2. ResNet-1D: Deep residual convolutional neural network with skip connections

Trained on the massive multi-match dataset (Bundesliga + Metrica + Barca)
and tested on the official competition test set (Barca vs Athletic Ground Truth).
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm

torch.manual_seed(42)
np.random.seed(42)

FEATURE_COLS = [
    "ball_missing", "ball_speed", "d_margin", "d_margin_roll_1s", "d_margin_roll_3s",
    "home_within_3m", "away_within_3m", "home_centroid_x", "home_centroid_y",
    "away_centroid_x", "away_centroid_y", "centroid_diff_x", "home_spread_x",
    "home_spread_y", "away_spread_x", "away_spread_y", "spread_ratio",
    "mean_speed_home", "mean_speed_away", "overall_speed", "speed_all_roll_2s",
    "missing_ratio_roll_2s"
]

LABEL_TO_CODE = {"H": 0, "A": 1, "D": 2}
CODE_TO_LABEL = {0: "H", 1: "A", 2: "D"}
WINDOW_SIZE = 25  # 1-second temporal context (25 frames @ 25 fps)
HALF_W = WINDOW_SIZE // 2


# ---------------------------------------------------------------------
# 1. InceptionTime 1D CNN Architecture
# ---------------------------------------------------------------------
class InceptionModule1D(nn.Module):
    def __init__(self, in_channels, out_channels=24):
        super().__init__()
        bottleneck = max(out_channels // 4, 8)
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1)
        self.conv1 = nn.Conv1d(bottleneck, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(bottleneck, out_channels, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(bottleneck, out_channels, kernel_size=15, padding=7)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        total_out = out_channels * 4
        self.bn = nn.BatchNorm1d(total_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        b = self.bottleneck(x)
        c1 = self.conv1(b)
        c2 = self.conv2(b)
        c3 = self.conv3(b)
        p = self.conv_pool(self.maxpool(x))
        out = torch.cat([c1, c2, c3, p], dim=1)
        return self.relu(self.bn(out))


class InceptionTimeCNN(nn.Module):
    def __init__(self, in_channels=len(FEATURE_COLS), num_classes=3):
        super().__init__()
        self.inc1 = InceptionModule1D(in_channels, out_channels=20)  # 20 * 4 = 80
        self.shortcut1 = nn.Conv1d(in_channels, 80, kernel_size=1)
        self.inc2 = InceptionModule1D(80, out_channels=24)          # 24 * 4 = 96
        self.shortcut2 = nn.Conv1d(80, 96, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(96, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        res1 = self.shortcut1(x)
        x = self.relu(self.inc1(x) + res1)
        res2 = self.shortcut2(x)
        x = self.relu(self.inc2(x) + res2)
        feat = self.pool(x).squeeze(-1)
        feat = self.dropout(feat)
        return self.fc(feat)


# ---------------------------------------------------------------------
# 2. ResNet 1D Architecture
# ---------------------------------------------------------------------
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
    def __init__(self, in_channels=len(FEATURE_COLS), num_classes=3):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU()
        )
        self.block1 = ResNetBlock1D(48)
        self.block2 = ResNetBlock1D(48)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        feat = self.pool(x).squeeze(-1)
        feat = self.dropout(feat)
        return self.fc(feat)


# Fast Streaming PyTorch Dataset using Memory-Mapped Array
class WindowedArrayDataset(Dataset):
    def __init__(self, X_padded, y, window_size=WINDOW_SIZE):
        self.X_padded = X_padded  # (N + 2*half, C)
        self.y = y
        self.w = window_size

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Return (Channels, TimeSteps)
        window = self.X_padded[idx : idx + self.w].T
        return torch.from_numpy(window), self.y[idx]


def smooth_predictions(preds: list, window: int = 25) -> list:
    n = len(preds)
    smoothed = list(preds)
    half = window // 2
    for i in range(n):
        sub = preds[max(0, i - half):min(n, i + half + 1)]
        vals, counts = np.unique(sub, return_counts=True)
        smoothed[i] = vals[np.argmax(counts)]
    return smoothed


def build_sequences(df, scaler, is_train=False):
    X_raw = df[FEATURE_COLS].fillna(0).values.astype(np.float32)
    if is_train:
        X_norm = scaler.fit_transform(X_raw)
    else:
        X_norm = scaler.transform(X_raw)
    y_code = np.array([LABEL_TO_CODE.get(l, 2) for l in df["label"]], dtype=np.int64)
    X_padded = np.pad(X_norm, ((HALF_W, HALF_W), (0, 0)), mode="edge")
    return X_padded, y_code


def train_and_eval_cnn(model_name: str, model: nn.Module, train_loader, test_loader, y_test_str, epochs=4):
    print(f"\nTraining {model_name}...")
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 2.5]))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t0 = time.time()
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
            if n_batches % 250 == 0:
                print(f"  [Epoch {ep+1}/{epochs}] Step {n_batches}/{len(train_loader)} - Batch Loss: {loss.item():.4f}")
        scheduler.step()
        print(f"  Epoch {ep+1}/{epochs} Complete - Mean Loss: {ep_loss / n_batches:.4f}")

    print(f"Training finished in {time.time()-t0:.1f}s. Evaluating on Barca Ground Truth...")
    model.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in test_loader:
            out = model(bx)
            p = torch.argmax(out, dim=-1).cpu().numpy()
            preds.extend(p)

    preds_raw = [CODE_TO_LABEL[c] for c in preds]
    preds_smooth = smooth_predictions(preds_raw, window=25)

    true_codes = np.array([LABEL_TO_CODE[l] for l in y_test_str], dtype=np.int8)
    pred_smooth_codes = np.array([LABEL_TO_CODE[l] for l in preds_smooth], dtype=np.int8)
    metrics = pm.evaluate(true_codes, pred_smooth_codes)

    return {
        "Model": f"{model_name} + Filter (w=25)",
        "spell_f1": metrics["spell_f1_mean"],
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "f1_home": metrics["per_class"]["H"]["f1"],
        "f1_away": metrics["per_class"]["A"]["f1"],
        "f1_dead": metrics["per_class"]["D"]["f1"],
        "n_spells": metrics["n_pred_spells"],
    }


def main():
    print("=" * 75)
    print("  TRAINING TEMPORAL 1D CNNs ON MULTI-MATCH CORPUS")
    print("=" * 75)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # Load mega dataset
    mega_path = os.path.join(ROOT, "data", "ml_unified_training_dataset.csv")
    print(f"Loading Mega Training Dataset from {mega_path}...")
    df_mega = pd.read_csv(mega_path)
    print(f"Loaded {len(df_mega):,} frames.")

    # Load Barca ground truth
    barca_path = os.path.join(ROOT, "data", "features_first_15min_play.csv")
    df_barca = pd.read_csv(barca_path)
    df_barca_gt = df_barca[df_barca["label"].isin(["H", "A", "D"])].copy().reset_index(drop=True)
    y_test_str = df_barca_gt["label"].values
    print(f"Loaded {len(df_barca_gt):,} test ground truth frames.")

    # Subsample 150,000 frames from the 650k corpus for balanced fast CPU training
    # Keeps all dead ball frames (underrepresented) and stratified samples of active frames
    dead_df = df_mega[df_mega["label"] == "D"]
    alive_df = df_mega[df_mega["label"] != "D"].sample(n=140000, random_state=42)
    df_train_sub = pd.concat([dead_df, alive_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Balanced CNN Training Subset: {len(df_train_sub):,} samples")

    X_train_pad, y_train = build_sequences(df_train_sub, scaler, is_train=True)
    X_test_pad, y_test = build_sequences(df_barca_gt, scaler, is_train=False)

    train_ds = WindowedArrayDataset(X_train_pad, y_train)
    test_ds = WindowedArrayDataset(X_test_pad, y_test)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    results = []

    # 1. InceptionTime-1D
    model_inc = InceptionTimeCNN(in_channels=len(FEATURE_COLS), num_classes=3)
    res_inc = train_and_eval_cnn("InceptionTime-1D CNN", model_inc, train_loader, test_loader, y_test_str, epochs=3)
    results.append(res_inc)
    print(f"  Result: Spell F1 = {res_inc['spell_f1']:.4f}, Macro F1 = {res_inc['macro_f1']:.4f}")

    # 2. ResNet-1D
    model_res = ResNet1D(in_channels=len(FEATURE_COLS), num_classes=3)
    res_res = train_and_eval_cnn("ResNet-1D CNN", model_res, train_loader, test_loader, y_test_str, epochs=3)
    results.append(res_res)
    print(f"  Result: Spell F1 = {res_res['spell_f1']:.4f}, Macro F1 = {res_res['macro_f1']:.4f}")

    # Output scoreboard
    df_res = pd.DataFrame(results)
    out_csv = os.path.join(HERE, "cnn_multi_match_scoreboard.csv")
    df_res.to_csv(out_csv, index=False)
    print("\n" + "=" * 80)
    print("  CNN MULTI-MATCH SCOREBOARD")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print(f"\nSaved scoreboard to {out_csv}")


if __name__ == "__main__":
    main()
