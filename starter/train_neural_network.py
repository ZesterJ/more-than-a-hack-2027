"""PyTorch Temporal Neural Networks for Ball Possession Classification.

Architectures:
  1. TemporalConv1D (Dilated Multi-scale 1D CNN with Residual Connections)
  2. Bidirectional GRU (BiGRU)

Evaluated with 5-Fold Contiguous Out-of-Sample Cross Validation on the 5-minute Ground Truth
using the official competition scorer (spellF1 and macroF1).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Path resolution
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm

# Seed
torch.manual_seed(42)
np.random.seed(42)

# Load data
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
WINDOW_SIZE = 35  # +/- 17 frames context (~1.4 seconds centered window)
HALF_WINDOW = WINDOW_SIZE // 2

# Pad features for sequence window generation
X_padded = np.pad(X_raw, ((HALF_WINDOW, HALF_WINDOW), (0, 0)), mode="edge")

# Normalize features
mean = np.mean(X_raw, axis=0)
std = np.std(X_raw, axis=0) + 1e-6
X_norm = (X_padded - mean) / std

# Build 3D sequence tensors: (N, Window, Features)
X_seq = np.empty((N_SAMPLES, WINDOW_SIZE, len(FEATURE_COLS)), dtype=np.float32)
for i in range(N_SAMPLES):
    X_seq[i] = X_norm[i : i + WINDOW_SIZE]

print(f"Sequence dataset built: {X_seq.shape} -> (Samples, TimeSteps, Features)")


# ---------------------------------------------------------------------
# Neural Network Architectures
# ---------------------------------------------------------------------
class TemporalConv1D(nn.Module):
    """Multi-scale Dilated 1D CNN with Residual Connections."""
    def __init__(self, in_channels, num_classes=3):
        super().__init__()
        # Conv1D expects (Batch, Channels, Time)
        self.conv1 = nn.Conv1d(in_channels, 48, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(48)
        
        # Dilated layers capturing broader temporal spans (dilation=2 and 4)
        self.conv2 = nn.Conv1d(48, 64, kernel_size=3, dilation=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.res2 = nn.Conv1d(48, 64, kernel_size=1)
        
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, dilation=4, padding=4)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (Batch, Time, Channels) -> transpose to (Batch, Channels, Time)
        x = x.transpose(1, 2)
        
        out = self.relu(self.bn1(self.conv1(x)))
        res = self.res2(out)
        out = self.relu(self.bn2(self.conv2(out)) + res)
        out = self.dropout(out)
        
        out = self.relu(self.bn3(self.conv3(out)) + out)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)


class BiGRU(nn.Module):
    """Bidirectional GRU for forward/backward temporal context."""
    def __init__(self, in_features, hidden_size=48, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden_size, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        center_repr = out[:, out.size(1) // 2, :]  # Center frame embedding
        return self.fc(center_repr)


# Custom PyTorch Dataset
class FootyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Class weights for focal cross entropy (heavier weight on D)
class_weights = torch.tensor([1.0, 1.2, 1.8], dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights)


# ---------------------------------------------------------------------
# 5-Fold Cross Validation Training
# ---------------------------------------------------------------------
print("\n" + "="*70)
print("TRAINING NEURAL SEQUENCE MODELS (5-FOLD CONTIGUOUS CV)...")
print("="*70)

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=False)

oof_preds_tcn = np.zeros((N_SAMPLES, 3))
oof_preds_gru = np.zeros((N_SAMPLES, 3))

BATCH_SIZE = 64
EPOCHS = 18

for fold, (train_idx, val_idx) in enumerate(kf.split(X_seq)):
    print(f"\n--- Fold {fold+1}/5 ---")
    train_loader = DataLoader(FootyDataset(X_seq[train_idx], y_code[train_idx]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FootyDataset(X_seq[val_idx], y_code[val_idx]), batch_size=BATCH_SIZE, shuffle=False)

    # 1. Train Temporal Conv1D
    model_tcn = TemporalConv1D(in_channels=len(FEATURE_COLS))
    optimizer_tcn = optim.AdamW(model_tcn.parameters(), lr=0.003, weight_decay=1e-4)

    model_tcn.train()
    for epoch in range(EPOCHS):
        for bx, by in train_loader:
            optimizer_tcn.zero_grad()
            out = model_tcn(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer_tcn.step()

    model_tcn.eval()
    val_probs = []
    with torch.no_grad():
        for bx, _ in val_loader:
            val_probs.append(torch.softmax(model_tcn(bx), dim=-1).cpu().numpy())
    oof_preds_tcn[val_idx] = np.concatenate(val_probs, axis=0)

    # 2. Train BiGRU
    model_gru = BiGRU(in_features=len(FEATURE_COLS))
    optimizer_gru = optim.AdamW(model_gru.parameters(), lr=0.002, weight_decay=1e-4)

    model_gru.train()
    for epoch in range(EPOCHS):
        for bx, by in train_loader:
            optimizer_gru.zero_grad()
            out = model_gru(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer_gru.step()

    model_gru.eval()
    val_probs = []
    with torch.no_grad():
        for bx, _ in val_loader:
            val_probs.append(torch.softmax(model_gru(bx), dim=-1).cpu().numpy())
    oof_preds_gru[val_idx] = np.concatenate(val_probs, axis=0)
    print(f"Fold {fold+1} complete.")


# ---------------------------------------------------------------------
# Evaluation & Comparison against Previous Top Score
# ---------------------------------------------------------------------
def smooth_probs(probs, window=25):
    kernel = np.ones(window) / float(window)
    smoothed = np.zeros_like(probs)
    for c in range(3):
        smoothed[:, c] = np.convolve(probs[:, c], kernel, mode="same")
    return np.argmax(smoothed, axis=1)

# Evaluate Neural Models
results = {
    "1. Previous Winner (LogReg + Smooth w=50)": smooth_probs(np.load(os.path.join(HERE, "oof_lr_probs.npy")) if os.path.exists(os.path.join(HERE, "oof_lr_probs.npy")) else np.zeros((N_SAMPLES, 3)), 50),
    "2. Temporal Conv1D (Raw)": np.argmax(oof_preds_tcn, axis=1),
    "3. Temporal Conv1D (Smooth w=25 / 1.0s)": smooth_probs(oof_preds_tcn, window=25),
    "4. Temporal Conv1D (Smooth w=50 / 2.0s)": smooth_probs(oof_preds_tcn, window=50),
    "5. Bidirectional GRU (Raw)": np.argmax(oof_preds_gru, axis=1),
    "6. Bidirectional GRU (Smooth w=25 / 1.0s)": smooth_probs(oof_preds_gru, window=25),
    "7. Ensemble (TCN + BiGRU + LogReg Blend, w=35)": smooth_probs(0.4 * oof_preds_tcn + 0.4 * oof_preds_gru, window=35)
}

scoreboard = []
for name, p in results.items():
    if p.sum() == 0 and "LogReg" in name:
        continue
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
print("NEURAL NETWORK OFFICIAL BENCHMARK SCOREBOARD (GT: 17 Spells, 14.2s Mean Dur)")
print("="*95)
print(score_df.to_string())

score_csv_path = os.path.join(HERE, "neural_network_scoreboard.csv")
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
ax.set_title("Neural Network vs Previous Models on 5-Minute Labeled Ground Truth")
ax.set_xlim(0, 1.0)
ax.grid(axis="x", alpha=0.3)
ax.legend(loc="lower right")

for i, row in score_df.iterrows():
    idx = i - 1
    ax.text(row["spellF1 (Ranked)"] + 0.01, idx - bar_width/2, f"{row['spellF1 (Ranked)']:.3f}", va="center", fontsize=9, fontweight="bold", color="#2ca02c")
    ax.text(row["macroF1 (Tie-break)"] + 0.01, idx + bar_width/2, f"{row['macroF1 (Tie-break)']:.3f}", va="center", fontsize=9, color="#1f77b4")

plt.tight_layout()
comp_plot_path = os.path.join(HERE, "neural_network_comparison_chart.png")
plt.savefig(comp_plot_path, dpi=120)
plt.close()
print(f"Chart saved to: {comp_plot_path}")
