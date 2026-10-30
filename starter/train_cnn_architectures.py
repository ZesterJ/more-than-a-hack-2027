"""Train Dedicated CNN Architectures (InceptionTime 1D CNN & ResNet-1D) for Ball Possession.

Architectures:
  1. InceptionTime 1D CNN: Multi-scale parallel temporal convolution kernels (k=3, 7, 15, 31)
  2. ResNet-1D: Deep Residual 1D Convolutional Network with skip connections
  3. InceptionTime + Temporal Probability Smoothing (Windowed Majority Vote w=25 & w=50)

Evaluated with 5-Fold Contiguous Out-of-Sample Cross-Validation on the 5-minute Ground Truth
using the official competition evaluator (spellF1 and macroF1).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

# Path resolution
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm

torch.manual_seed(42)
np.random.seed(42)

# Load features
DATA_PATH = os.path.join(ROOT, "data", "features_first_15min_play.csv")
df = pd.read_csv(DATA_PATH)
df_labeled = df[df["label"].isin(["H", "A", "D"])].copy().reset_index(drop=True)

FEATURE_COLS = [
    "ball_missing", "ball_speed", "d_margin", "d_margin_roll_1s", "d_margin_roll_3s",
    "home_within_3m", "away_within_3m", "home_centroid_x", "home_centroid_y",
    "away_centroid_x", "away_centroid_y", "centroid_diff_x", "home_spread_x",
    "home_spread_y", "away_spread_x", "away_spread_y", "spread_ratio",
    "mean_speed_home", "mean_speed_away", "overall_speed", "speed_all_roll_2s",
    "missing_ratio_roll_2s"
]

X_raw = df_labeled[FEATURE_COLS].fillna(0).values
y_str = df_labeled["label"].values

LABEL_TO_CODE = {"H": 0, "A": 1, "D": 2}
CODE_TO_LABEL = {0: "H", 1: "A", 2: "D"}
y_code = np.array([LABEL_TO_CODE[lbl] for lbl in y_str], dtype=np.int8)

N_SAMPLES = len(y_code)
WINDOW_SIZE = 41  # +/- 20 frames context (~1.6 seconds centered window)
HALF_WINDOW = WINDOW_SIZE // 2

# Normalization and sequence creation
mean = np.mean(X_raw, axis=0)
std = np.std(X_raw, axis=0) + 1e-6
X_norm = (X_raw - mean) / std

X_padded = np.pad(X_norm, ((HALF_WINDOW, HALF_WINDOW), (0, 0)), mode="edge")
X_seq = np.empty((N_SAMPLES, len(FEATURE_COLS), WINDOW_SIZE), dtype=np.float32)

for i in range(N_SAMPLES):
    # (Channels, TimeSteps) for Conv1D
    X_seq[i] = X_padded[i : i + WINDOW_SIZE].T

print(f"Sequence dataset built for CNN: {X_seq.shape} -> (Samples, Channels, TimeSteps)")


# ---------------------------------------------------------------------
# 1. InceptionTime 1D CNN Architecture
# ---------------------------------------------------------------------
class InceptionModule1D(nn.Module):
    """Inception block combining small, medium, and wide temporal receptive fields."""
    def __init__(self, in_channels, out_channels=32):
        super().__init__()
        bottleneck = max(out_channels // 4, 8)
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1)
        
        # Parallel multi-scale filters:
        # k=3: instant player touch (~0.12s)
        # k=7: short pass / tackle (~0.28s)
        # k=15: medium spatial sequence (~0.60s)
        # k=31: long tactical buildup (~1.24s)
        self.conv1 = nn.Conv1d(bottleneck, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(bottleneck, out_channels, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(bottleneck, out_channels, kernel_size=15, padding=7)
        self.conv4 = nn.Conv1d(bottleneck, out_channels, kernel_size=31, padding=15)
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        total_out = out_channels * 5
        self.bn = nn.BatchNorm1d(total_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        b = self.bottleneck(x)
        c1 = self.conv1(b)
        c2 = self.conv2(b)
        c3 = self.conv3(b)
        c4 = self.conv4(b)
        p = self.conv_pool(self.maxpool(x))
        out = torch.cat([c1, c2, c3, c4, p], dim=1)
        return self.relu(self.bn(out))


class InceptionTimeCNN(nn.Module):
    """Deep InceptionTime Network for Time-Series Tracking."""
    def __init__(self, in_channels=len(FEATURE_COLS), num_classes=3):
        super().__init__()
        self.inc1 = InceptionModule1D(in_channels, out_channels=24)  # 24 * 5 = 120
        self.shortcut1 = nn.Conv1d(in_channels, 120, kernel_size=1)
        
        self.inc2 = InceptionModule1D(120, out_channels=32)         # 32 * 5 = 160
        self.shortcut2 = nn.Conv1d(120, 160, kernel_size=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(160, num_classes)
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
# 2. ResNet-1D Architecture
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
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.block1 = ResNetBlock1D(64)
        self.block2 = ResNetBlock1D(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        feat = self.pool(x).squeeze(-1)
        feat = self.dropout(feat)
        return self.fc(feat)


# PyTorch Dataset
class FootyCNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Class weighted loss to handle Dead Ball balance
class_weights = torch.tensor([1.0, 1.2, 1.8], dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights)


# ---------------------------------------------------------------------
# 5-Fold Cross Validation Training
# ---------------------------------------------------------------------
print("\n" + "="*70)
print("TRAINING CNN ARCHITECTURES (5-FOLD CONTIGUOUS CV)...")
print("="*70)

kf = KFold(n_splits=5, shuffle=False)

oof_preds_inception = np.zeros((N_SAMPLES, 3))
oof_preds_resnet = np.zeros((N_SAMPLES, 3))

BATCH_SIZE = 64
EPOCHS = 16

for fold, (train_idx, val_idx) in enumerate(kf.split(X_seq)):
    print(f"\n--- Fold {fold+1}/5 ---")
    train_loader = DataLoader(FootyCNNDataset(X_seq[train_idx], y_code[train_idx]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FootyCNNDataset(X_seq[val_idx], y_code[val_idx]), batch_size=BATCH_SIZE, shuffle=False)

    # 1. Train InceptionTime CNN
    model_inc = InceptionTimeCNN()
    optimizer_inc = optim.AdamW(model_inc.parameters(), lr=0.003, weight_decay=1e-4)

    model_inc.train()
    for epoch in range(EPOCHS):
        for bx, by in train_loader:
            optimizer_inc.zero_grad()
            out = model_inc(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer_inc.step()

    model_inc.eval()
    val_probs_inc = []
    with torch.no_grad():
        for bx, _ in val_loader:
            val_probs_inc.append(torch.softmax(model_inc(bx), dim=-1).cpu().numpy())
    oof_preds_inception[val_idx] = np.concatenate(val_probs_inc, axis=0)

    # 2. Train ResNet-1D
    model_res = ResNet1D()
    optimizer_res = optim.AdamW(model_res.parameters(), lr=0.003, weight_decay=1e-4)

    model_res.train()
    for epoch in range(EPOCHS):
        for bx, by in train_loader:
            optimizer_res.zero_grad()
            out = model_res(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer_res.step()

    model_res.eval()
    val_probs_res = []
    with torch.no_grad():
        for bx, _ in val_loader:
            val_probs_res.append(torch.softmax(model_res(bx), dim=-1).cpu().numpy())
    oof_preds_resnet[val_idx] = np.concatenate(val_probs_res, axis=0)

    print(f"Fold {fold+1} complete.")


# ---------------------------------------------------------------------
# Evaluation & Metric Comparison
# ---------------------------------------------------------------------
def smooth_probs(probs, window=25):
    kernel = np.ones(window) / float(window)
    smoothed = np.zeros_like(probs)
    for c in range(3):
        smoothed[:, c] = np.convolve(probs[:, c], kernel, mode="same")
    return np.argmax(smoothed, axis=1)

results = {
    "1. InceptionTime 1D CNN (Smooth w=50 / 2.0s)": smooth_probs(oof_preds_inception, window=50),
    "2. InceptionTime 1D CNN (Smooth w=25 / 1.0s)": smooth_probs(oof_preds_inception, window=25),
    "3. InceptionTime 1D CNN (Raw)": np.argmax(oof_preds_inception, axis=1),
    "4. ResNet-1D CNN (Smooth w=50 / 2.0s)": smooth_probs(oof_preds_resnet, window=50),
    "5. ResNet-1D CNN (Smooth w=25 / 1.0s)": smooth_probs(oof_preds_resnet, window=25),
    "6. ResNet-1D CNN (Raw)": np.argmax(oof_preds_resnet, axis=1),
    "7. CNN Blend (InceptionTime + ResNet, w=35)": smooth_probs(0.5 * oof_preds_inception + 0.5 * oof_preds_resnet, window=35)
}

scoreboard = []
for name, p in results.items():
    m = pm.evaluate(y_code, p.astype(np.int8))
    scoreboard.append({
        "Model": name,
        "spellF1 (Ranked)": round(m["spell_f1_mean"], 4),
        "macroF1 (Tie-break)": round(m["macro_f1"], 4),
        "Accuracy": round(m["accuracy"], 4),
        "F1 (H)": round(m["per_class"]["H"]["f1"], 4),
        "F1 (A)": round(m["per_class"]["A"]["f1"], 4),
        "F1 (D)": round(m["per_class"]["D"]["f1"], 4),
        "Spells": m["n_pred_spells"],
        "Mean Dur (s)": round(m["mean_spell_duration_s"], 1)
    })

score_df = pd.DataFrame(scoreboard).sort_values("spellF1 (Ranked)", ascending=False).reset_index(drop=True)
score_df.index = score_df.index + 1

print("\n" + "="*95)
print("DEDICATED CNN OFFICIAL BENCHMARK SCOREBOARD (GT: 17 Spells, 14.2s Mean Dur)")
print("="*95)
print(score_df.to_string())

score_csv_path = os.path.join(HERE, "cnn_scoreboard.csv")
score_df.to_csv(score_csv_path, index=True)
print(f"\nScoreboard saved to: {score_csv_path}")

# Plot comparison
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35
indices = np.arange(len(score_df))

ax.barh(indices - bar_width/2, score_df["spellF1 (Ranked)"], bar_width, label="spellF1 (Headline Ranking Metric)", color="#2ca02c")
ax.barh(indices + bar_width/2, score_df["macroF1 (Tie-break)"], bar_width, label="macroF1 (Per-frame Frame Metric)", color="#1f77b4")

ax.set_yticks(indices)
ax.set_yticklabels(score_df["Model"])
ax.invert_yaxis()
ax.set_xlabel("F1 Score")
ax.set_title("Dedicated CNN Architectures on 5-Minute Labeled Ground Truth")
ax.set_xlim(0, 1.0)
ax.grid(axis="x", alpha=0.3)
ax.legend(loc="lower right")

for i, row in score_df.iterrows():
    idx = i - 1
    ax.text(row["spellF1 (Ranked)"] + 0.01, idx - bar_width/2, f"{row['spellF1 (Ranked)']:.3f}", va="center", fontsize=9, fontweight="bold", color="#2ca02c")
    ax.text(row["macroF1 (Tie-break)"] + 0.01, idx + bar_width/2, f"{row['macroF1 (Tie-break)']:.3f}", va="center", fontsize=9, color="#1f77b4")

plt.tight_layout()
comp_plot_path = os.path.join(HERE, "cnn_comparison_chart.png")
plt.savefig(comp_plot_path, dpi=120)
plt.close()
print(f"Chart saved to: {comp_plot_path}")
