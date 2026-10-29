"""Generate comprehensive feature visualizations for the 15-minute tracking dataset.

Outputs:
    - feature_timeline_15min.png: Multi-signal timeline comparing features against ground truth
    - feature_distributions_by_class.png: Boxplots/violin plots showing class separability (H vs A vs D)
    - feature_correlations.png: Correlation heatmap of engineered features
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(HERE), "data", "features_first_15min_play.csv")
OUT_DIR = HERE

df = pd.read_csv(DATA_PATH)
fps = 25.0
time_minutes = (df["video_frame"] - df["video_frame"].iloc[0]) / (fps * 60.0)

# Colors
COLOR_H = "#d62728"  # Barcelona Red
COLOR_A = "#1f77b4"  # Athletic Blue
COLOR_D = "#7f7f7f"  # Dead Ball Grey

# -------------------------------------------------------------
# 1. Timeline Visualization over 15 Minutes
# -------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

# A. Ground Truth Background Shading on first 5 minutes
labeled_mask = df["label"].isin(["H", "A", "D"])
labeled_df = df[labeled_mask].copy()

# Subplot 1: Ball Proximity Margin
ax1 = axes[0]
ax1.plot(time_minutes, df["d_margin_roll_1s"], label="Distance Margin (1s smooth)", color="#2ca02c", lw=1.2)
ax1.plot(time_minutes, df["d_margin_roll_3s"], label="Distance Margin (3s smooth)", color="#1b5e20", lw=2.0)
ax1.axhline(0, color="black", linestyle="--", alpha=0.6)
ax1.set_ylabel("Distance Margin (m)\n(>0 Home closer, <0 Away closer)")
ax1.set_title("15-Minute Feature Dynamics vs Ground Truth (First 5 min scored)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right")

# Subplot 2: Player Movement Speed (Stoppage Indicator)
ax2 = axes[1]
ax2.plot(time_minutes, df["overall_speed"], label="Raw Player Speed", color="#aec7e8", alpha=0.5, lw=0.8)
ax2.plot(time_minutes, df["speed_all_roll_2s"], label="Smoothed Speed (2s roll)", color="#ff7f0e", lw=1.8)
ax2.axhline(1.0, color="red", linestyle=":", label="Stoppage Threshold (~1.0 m/s)")
ax2.set_ylabel("Mean Player Speed (m/s)")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")

# Subplot 3: Team Longitudinal Centroids
ax3 = axes[2]
ax3.plot(time_minutes, df["home_centroid_x"], label="Barcelona Centroid X", color=COLOR_H, lw=1.5)
ax3.plot(time_minutes, df["away_centroid_x"], label="Athletic Centroid X", color=COLOR_A, lw=1.5)
ax3.axhline(52.65, color="grey", linestyle="--", alpha=0.5, label="Halfway Line (52.65m)")
ax3.set_ylabel("Pitch X Coordinate (m)")
ax3.grid(True, alpha=0.3)
ax3.legend(loc="upper right")

# Subplot 4: Team Spread Ratio (Spatial Expansion vs Compression)
ax4 = axes[3]
ax4.plot(time_minutes, df["spread_ratio"], label="Spread Ratio (Home Spread / Away Spread)", color="#9467bd", lw=1.5)
ax4.axhline(1.0, color="black", linestyle="--", alpha=0.5)
ax4.set_ylabel("Spread Ratio\n(>1 Home expansive)")
ax4.set_xlabel("Elapsed Match Time from Kickoff (Minutes)")
ax4.grid(True, alpha=0.3)
ax4.legend(loc="upper right")

# Highlight GT spans on all subplots for the first 5 minutes
curr_label = None
start_t = None
for t, lbl in zip(time_minutes[labeled_mask], df["label"][labeled_mask]):
    if lbl != curr_label:
        if curr_label in ["H", "A", "D"]:
            c = COLOR_H if curr_label == "H" else (COLOR_A if curr_label == "A" else COLOR_D)
            for ax in axes:
                ax.axvspan(start_t, t, color=c, alpha=0.15)
        curr_label = lbl
        start_t = t
if curr_label in ["H", "A", "D"]:
    c = COLOR_H if curr_label == "H" else (COLOR_A if curr_label == "A" else COLOR_D)
    for ax in axes:
        ax.axvspan(start_t, time_minutes[labeled_mask].iloc[-1], color=c, alpha=0.15)

plt.tight_layout()
timeline_path = os.path.join(OUT_DIR, "feature_timeline_15min.png")
plt.savefig(timeline_path, dpi=120)
plt.close()
print(f"Saved: {timeline_path}")

# -------------------------------------------------------------
# 2. Feature Distributions by Class (Public 5 Minutes)
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
features_to_plot = [
    ("d_margin_roll_1s", "Smoothed Ball Distance Margin (m)", axes[0, 0]),
    ("speed_all_roll_2s", "Smoothed Outfield Player Speed (m/s)", axes[0, 1]),
    ("centroid_diff_x", "Centroid Difference (Home X - Away X) (m)", axes[1, 0]),
    ("spread_ratio", "Team Spread Ratio (Home/Away)", axes[1, 1])
]

for col, title, ax in features_to_plot:
    data_by_class = [labeled_df[labeled_df["label"] == c][col].dropna() for c in ["H", "A", "D"]]
    box = ax.boxplot(data_by_class, tick_labels=["H (Barcelona)", "A (Athletic)", "D (Dead Ball)"],
                     patch_artist=True, medianprops=dict(color="black", lw=1.5))
    colors = [COLOR_H, COLOR_A, COLOR_D]
    for patch, c in zip(box["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)

plt.suptitle("Feature Separation by Class on Labeled 5-Minute Ground Truth", fontsize=14, y=1.01)
plt.tight_layout()
dist_path = os.path.join(OUT_DIR, "feature_distributions_by_class.png")
plt.savefig(dist_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {dist_path}")

# -------------------------------------------------------------
# 3. Correlation Heatmap
# -------------------------------------------------------------
plt.figure(figsize=(10, 8))
selected_cols = [
    "d_margin", "d_margin_roll_1s", "d_margin_roll_3s",
    "overall_speed", "speed_all_roll_2s",
    "home_centroid_x", "away_centroid_x", "centroid_diff_x",
    "spread_ratio", "ball_missing", "missing_ratio_roll_2s"
]
corr = labeled_df[selected_cols].corr()

plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Pearson Correlation")
plt.xticks(range(len(selected_cols)), selected_cols, rotation=45, ha="right")
plt.yticks(range(len(selected_cols)), selected_cols)
plt.title("Correlation Matrix of Engineered Features", fontsize=13)

for i in range(len(selected_cols)):
    for j in range(len(selected_cols)):
        val = corr.iloc[i, j]
        plt.text(j, i, f"{val:.2f}", ha="center", va="center",
                 color="white" if abs(val) > 0.5 else "black", fontsize=8)

plt.tight_layout()
corr_path = os.path.join(OUT_DIR, "feature_correlations.png")
plt.savefig(corr_path, dpi=120)
plt.close()
print(f"Saved: {corr_path}")
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
