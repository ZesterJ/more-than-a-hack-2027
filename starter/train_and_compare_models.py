"""Train multiple ML models on the labeled ground truth window (first 5 minutes)
and compare them using official competition metrics (macroF1, spellF1, per-class F1).

Models evaluated:
  1. Nearest Player Heuristic (Official Repo Baseline)
  2. Nearest Player Heuristic (Smoothed w=25)
  3. Multinomial Logistic Regression (Linear)
  4. Decision Tree Classifier (Non-linear Rules)
  5. K-Nearest Neighbors (Spatial Instance-based)
  6. LightGBM Gradient Boosted Trees (GBDT)
  7. LightGBM + Temporal Smoothing (Windowed Majority Vote w=25 & w=50)

Outputs:
  - Official competition scoreboard
  - Feature importance ranking & plot
  - Visual comparison bar charts
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path resolution
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import lightgbm as lgb

# Load features
DATA_PATH = os.path.join(ROOT, "data", "features_first_15min_play.csv")
df = pd.read_csv(DATA_PATH)

# Filter for the 7,500 labeled ground truth frames (first 5 minutes of play)
labeled_mask = df["label"].isin(["H", "A", "D"])
df_labeled = df[labeled_mask].copy().reset_index(drop=True)

print(f"Loaded {len(df_labeled):,} labeled ground truth frames.")
print(f"Class distribution:\n{df_labeled['label'].value_counts(normalize=True).round(3)}")

FEATURE_COLS = [
    "ball_missing",
    "ball_speed",
    "d_margin",
    "d_margin_roll_1s",
    "d_margin_roll_3s",
    "home_within_3m",
    "away_within_3m",
    "home_centroid_x",
    "home_centroid_y",
    "away_centroid_x",
    "away_centroid_y",
    "centroid_diff_x",
    "home_spread_x",
    "home_spread_y",
    "away_spread_x",
    "away_spread_y",
    "spread_ratio",
    "mean_speed_home",
    "mean_speed_away",
    "overall_speed",
    "speed_all_roll_2s",
    "missing_ratio_roll_2s",
]

X = df_labeled[FEATURE_COLS].fillna(0).values
y_str = df_labeled["label"].values

LABEL_TO_CODE = {"H": 0, "A": 1, "D": 2}
CODE_TO_LABEL = {0: "H", 1: "A", 2: "D"}
y_code = np.array([LABEL_TO_CODE[lbl] for lbl in y_str], dtype=np.int8)


def smooth_predictions(preds_str, window=25):
    """Majority vote smoothing over a window (e.g. 25 frames = 1 sec)."""
    codes = np.array([LABEL_TO_CODE[l] for l in preds_str])
    n = len(codes)
    out = np.empty(n, dtype="object")
    
    counts_h = np.concatenate(([0], np.cumsum(codes == 0)))
    counts_a = np.concatenate(([0], np.cumsum(codes == 1)))
    counts_d = np.concatenate(([0], np.cumsum(codes == 2)))
    
    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        h = counts_h[end] - counts_h[start]
        a = counts_a[end] - counts_a[start]
        d = counts_d[end] - counts_d[start]
        
        best_count = max(h, a, d)
        if h == best_count:
            out[i] = "H"
        elif a == best_count:
            out[i] = "A"
        else:
            out[i] = "D"
    return out


print("\n" + "="*70)
print("RUNNING 5-FOLD OUT-OF-FOLD CROSS VALIDATION ACROSS MODELS...")
print("="*70)

kf = KFold(n_splits=5, shuffle=False)

preds_baseline = np.empty(len(y_str), dtype="object")
preds_lr = np.empty(len(y_str), dtype="object")
preds_dt = np.empty(len(y_str), dtype="object")
preds_knn = np.empty(len(y_str), dtype="object")
preds_lgb = np.empty(len(y_str), dtype="object")

# 1. Baseline heuristic
for i in range(len(df_labeled)):
    if df_labeled.loc[i, "ball_missing"] == 1:
        preds_baseline[i] = "D"
    else:
        margin = df_labeled.loc[i, "d_margin"]
        preds_baseline[i] = "H" if margin > 0 else "A"

# 2. Train ML Models in 5-fold CV
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Training Fold {fold + 1}/5...")
    X_train, y_train = X[train_idx], y_code[train_idx]
    X_val, y_val = X[val_idx], y_code[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_train_scaled, y_train)
    preds_lr[val_idx] = [CODE_TO_LABEL[c] for c in lr.predict(X_val_scaled)]

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=42)
    dt.fit(X_train, y_train)
    preds_dt[val_idx] = [CODE_TO_LABEL[c] for c in dt.predict(X_val)]

    # KNN
    knn = KNeighborsClassifier(n_neighbors=15, weights="distance")
    knn.fit(X_train_scaled, y_train)
    preds_knn[val_idx] = [CODE_TO_LABEL[c] for c in knn.predict(X_val_scaled)]

    # LightGBM
    model_lgb = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.08,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    model_lgb.fit(X_train, y_train)
    preds_lgb[val_idx] = [CODE_TO_LABEL[c] for c in model_lgb.predict(X_val)]

# 3. Post-processed versions with temporal smoothing
preds_baseline_smooth25 = smooth_predictions(preds_baseline, window=25)
preds_lgb_smooth25 = smooth_predictions(preds_lgb, window=25)
preds_lgb_smooth50 = smooth_predictions(preds_lgb, window=50)

# ---------------------------------------------------------------------
# Official Competition Evaluation
# ---------------------------------------------------------------------
models_dict = {
    "1. Nearest Player (Raw Heuristic)": preds_baseline,
    "2. Nearest Player (Smooth w=25)": preds_baseline_smooth25,
    "3. Multinomial Logistic Regression": preds_lr,
    "4. Decision Tree Classifier": preds_dt,
    "5. K-Nearest Neighbors (k=15)": preds_knn,
    "6. LightGBM Classifier (Raw)": preds_lgb,
    "7. LightGBM (Smooth w=25 / 1.0s)": preds_lgb_smooth25,
    "8. LightGBM (Smooth w=50 / 2.0s)": preds_lgb_smooth50,
}

scoreboard = []

for name, pred_arr in models_dict.items():
    pred_codes = np.array([LABEL_TO_CODE[l] for l in pred_arr], dtype=np.int8)
    metrics = pm.evaluate(y_code, pred_codes)
    
    f1_h = metrics["per_class"]["H"]["f1"]
    f1_a = metrics["per_class"]["A"]["f1"]
    f1_d = metrics["per_class"]["D"]["f1"]
    
    scoreboard.append({
        "Model": name,
        "spellF1 (Ranked)": round(metrics["spell_f1_mean"], 4),
        "macroF1 (Tie-break)": round(metrics["macro_f1"], 4),
        "Accuracy": round(metrics["accuracy"], 4),
        "F1 (H)": round(f1_h, 4),
        "F1 (A)": round(f1_a, 4),
        "F1 (D)": round(f1_d, 4),
        "Possessions": metrics["n_pred_spells"],
        "Mean Dur (s)": round(metrics["mean_spell_duration_s"], 1)
    })

score_df = pd.DataFrame(scoreboard).sort_values("spellF1 (Ranked)", ascending=False).reset_index(drop=True)
score_df.index = score_df.index + 1

print("\n" + "="*95)
print("OFFICIAL COMPETITION SCOREBOARD ON 5-MINUTE GROUND TRUTH (17 Real Possessions, 14.2s Mean Dur)")
print("="*95)
print(score_df.to_string())

score_csv_path = os.path.join(HERE, "models_comparison_scoreboard.csv")
score_df.to_csv(score_csv_path, index=True)
print(f"\nScoreboard saved to: {score_csv_path}")

# Feature Importance from Full LightGBM Model
full_lgb = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.08, num_leaves=31, random_state=42, verbosity=-1)
full_lgb.fit(X, y_code)
importances = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Importance": full_lgb.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print("\n=== TOP 10 MOST INFLUENTIAL FEATURES (LightGBM) ===")
print(importances.head(10).to_string(index=False))

# Plot Comparison Charts
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35
indices = np.arange(len(score_df))

ax.barh(indices - bar_width/2, score_df["spellF1 (Ranked)"], bar_width, label="spellF1 (Headline Ranking Metric)", color="#2ca02c")
ax.barh(indices + bar_width/2, score_df["macroF1 (Tie-break)"], bar_width, label="macroF1 (Per-frame Frame Metric)", color="#1f77b4")

ax.set_yticks(indices)
ax.set_yticklabels(score_df["Model"])
ax.invert_yaxis()
ax.set_xlabel("F1 Score")
ax.set_title("Model Comparison on 5-Minute Labeled Ground Truth")
ax.set_xlim(0, 1.0)
ax.grid(axis="x", alpha=0.3)
ax.legend(loc="lower right")

for i, row in score_df.iterrows():
    idx = i - 1
    ax.text(row["spellF1 (Ranked)"] + 0.01, idx - bar_width/2, f"{row['spellF1 (Ranked)']:.3f}", va="center", fontsize=9, fontweight="bold", color="#2ca02c")
    ax.text(row["macroF1 (Tie-break)"] + 0.01, idx + bar_width/2, f"{row['macroF1 (Tie-break)']:.3f}", va="center", fontsize=9, color="#1f77b4")

plt.tight_layout()
comp_plot_path = os.path.join(HERE, "models_comparison_chart.png")
plt.savefig(comp_plot_path, dpi=120)
plt.close()
print(f"Model comparison chart saved to: {comp_plot_path}")

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(importances["Feature"][:12][::-1], importances["Importance"][:12][::-1], color="#1f77b4")
plt.xlabel("Feature Importance (Split Count in LightGBM)")
plt.title("Top 12 Most Influential Features for Ball Possession")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
imp_path = os.path.join(HERE, "feature_importance_lgb.png")
plt.savefig(imp_path, dpi=120)
plt.close()
print(f"Feature importance plot saved to: {imp_path}")
