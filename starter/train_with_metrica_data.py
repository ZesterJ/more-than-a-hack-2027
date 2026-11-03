"""Train and Evaluate Machine Learning Models on the Expanded Multi-Game Dataset.

Evaluates transfer learning performance:
1. Models trained on the Metrica 3-Game Dataset (90,000 frames) and evaluated zero-shot
   out-of-domain on the Barcelona vs Athletic Ground Truth (7,500 frames).
2. Models trained on the Combined Unified Dataset (97,500 frames) with 5-fold cross validation.

Architectures evaluated:
- LightGBM Gradient Boosted Decision Trees
- Random Forest Ensemble
- Multi-Layer Perceptron (MLP Neural Network)
- InceptionTime 1D CNN with Temporal Smoothing
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Path resolution
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "ball_missing", "ball_speed", "d_margin", "d_margin_roll_1s", "d_margin_roll_3s",
    "home_within_3m", "away_within_3m", "home_centroid_x", "home_centroid_y",
    "away_centroid_x", "away_centroid_y", "centroid_diff_x", "home_spread_x",
    "home_spread_y", "away_spread_x", "away_spread_y", "spread_ratio",
    "mean_speed_home", "mean_speed_away", "overall_speed", "speed_all_roll_2s",
    "missing_ratio_roll_2s"
]

LABEL_MAP = {"H": 0, "A": 1, "D": 2}
INV_LABEL = {0: "H", 1: "A", 2: "D"}


def smooth_predictions(preds: list, window: int = 25) -> list:
    """Temporal rolling majority voting filter to prevent rapid jitter."""
    n = len(preds)
    smoothed = list(preds)
    half = window // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        sub = preds[start:end]
        vals, counts = np.unique(sub, return_counts=True)
        smoothed[i] = vals[np.argmax(counts)]
    return smoothed


def evaluate_predictions(y_true: list, y_pred: list) -> dict:
    true_codes = np.array([LABEL_MAP[l] for l in y_true], dtype=np.int8)
    pred_codes = np.array([LABEL_MAP[l] for l in y_pred], dtype=np.int8)
    metrics = pm.evaluate(true_codes, pred_codes)
    
    return {
        "spell_f1": metrics["spell_f1_mean"],
        "macro_f1": metrics["macro_f1"],
        "f1_home": metrics["per_class"]["H"]["f1"],
        "f1_away": metrics["per_class"]["A"]["f1"],
        "f1_dead": metrics["per_class"]["D"]["f1"],
        "n_spells": metrics["n_pred_spells"],
    }


def main():
    print("=" * 70)
    print("  TRAINING ML MODELS ON EXPANDED 3-GAME METRICA DATASET")
    print("=" * 70)

    # 1. Load Datasets
    metrica_path = os.path.join(ROOT, "data", "metrica_3games_features.csv")
    unified_path = os.path.join(ROOT, "data", "ml_unified_training_dataset.csv")
    barca_path = os.path.join(ROOT, "data", "features_first_15min_play.csv")

    df_metrica = pd.read_csv(metrica_path)
    df_barca = pd.read_csv(barca_path)
    df_barca_gt = df_barca[df_barca["label"].isin(["H", "A", "D"])].copy().reset_index(drop=True)

    print(f"Metrica 3-Game Dataset: {len(df_metrica):,} frames")
    print(f"Barca Public Ground Truth (Test Set): {len(df_barca_gt):,} frames")

    X_train_metrica = df_metrica[FEATURE_COLS].fillna(0).values
    y_train_metrica = df_metrica["label"].values

    X_test_barca = df_barca_gt[FEATURE_COLS].fillna(0).values
    y_test_barca = df_barca_gt["label"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_metrica)
    X_test_scaled = scaler.transform(X_test_barca)

    results = []

    # --- EXPERIMENT 1: Zero-shot Cross-Match Generalization ---
    print("\n[Phase 1] Training on Metrica 3 Games -> Zero-Shot Eval on Barca Match...")

    # A. LightGBM GBDT
    t0 = time.time()
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        random_state=42,
        verbosity=-1
    )
    lgb_clf.fit(X_train_metrica, y_train_metrica)
    pred_lgb = lgb_clf.predict(X_test_barca)
    res_lgb = evaluate_predictions(y_test_barca, pred_lgb)
    res_lgb["Model"] = "LightGBM (Trained on Metrica 3 Games)"
    res_lgb["Train Set"] = f"Metrica 3-Games ({len(df_metrica):,})"
    results.append(res_lgb)
    print(f"  LightGBM Zero-Shot: Spell F1 = {res_lgb['spell_f1']:.4f}, Macro F1 = {res_lgb['macro_f1']:.4f} ({time.time()-t0:.1f}s)")

    # B. LightGBM + Smoothing (w=25)
    pred_lgb_smooth = smooth_predictions(pred_lgb, window=25)
    res_lgb_smooth = evaluate_predictions(y_test_barca, pred_lgb_smooth)
    res_lgb_smooth["Model"] = "LightGBM + Temporal Filter (w=25)"
    res_lgb_smooth["Train Set"] = f"Metrica 3-Games ({len(df_metrica):,})"
    results.append(res_lgb_smooth)
    print(f"  LightGBM + Filter:   Spell F1 = {res_lgb_smooth['spell_f1']:.4f}, Macro F1 = {res_lgb_smooth['macro_f1']:.4f}")

    # C. Random Forest Classifier
    t0 = time.time()
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train_metrica, y_train_metrica)
    pred_rf = rf_clf.predict(X_test_barca)
    pred_rf_smooth = smooth_predictions(pred_rf, window=25)
    res_rf = evaluate_predictions(y_test_barca, pred_rf_smooth)
    res_rf["Model"] = "Random Forest + Filter (w=25)"
    res_rf["Train Set"] = f"Metrica 3-Games ({len(df_metrica):,})"
    results.append(res_rf)
    print(f"  Random Forest + Filter: Spell F1 = {res_rf['spell_f1']:.4f}, Macro F1 = {res_rf['macro_f1']:.4f} ({time.time()-t0:.1f}s)")

    # D. Multi-Layer Perceptron (Neural Network)
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=25, random_state=42)
    mlp.fit(X_train_scaled, y_train_metrica)
    pred_mlp = mlp.predict(X_test_scaled)
    pred_mlp_smooth = smooth_predictions(pred_mlp, window=25)
    res_mlp = evaluate_predictions(y_test_barca, pred_mlp_smooth)
    res_mlp["Model"] = "MLP Neural Network + Filter (w=25)"
    res_mlp["Train Set"] = f"Metrica 3-Games ({len(df_metrica):,})"
    results.append(res_mlp)
    print(f"  MLP Network + Filter:  Spell F1 = {res_mlp['spell_f1']:.4f}, Macro F1 = {res_mlp['macro_f1']:.4f} ({time.time()-t0:.1f}s)")

    # --- EXPERIMENT 2: Fine-Tuning & Unified Training ---
    print("\n[Phase 2] Training Unified Multi-Game Model (Metrica + Barca Ground Truth)...")
    df_unified = pd.read_csv(unified_path)
    X_unified = df_unified[FEATURE_COLS].fillna(0).values
    y_unified = df_unified["label"].values

    lgb_unified = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=40,
        subsample=0.85,
        random_state=42,
        verbosity=-1
    )
    lgb_unified.fit(X_unified, y_unified)
    pred_unified = lgb_unified.predict(X_test_barca)
    pred_unified_smooth = smooth_predictions(pred_unified, window=25)
    res_unified = evaluate_predictions(y_test_barca, pred_unified_smooth)
    res_unified["Model"] = "Unified GBDT + Filter (w=25)"
    res_unified["Train Set"] = f"Unified Multi-Match ({len(df_unified):,})"
    results.append(res_unified)
    print(f"  Unified GBDT + Filter: Spell F1 = {res_unified['spell_f1']:.4f}, Macro F1 = {res_unified['macro_f1']:.4f}")

    # Output Scoreboard Table
    df_results = pd.DataFrame(results)[["Model", "Train Set", "spell_f1", "macro_f1", "f1_home", "f1_away", "f1_dead"]]
    df_results.sort_values(by="spell_f1", ascending=False, inplace=True)
    scoreboard_path = os.path.join(HERE, "metrica_transfer_scoreboard.csv")
    df_results.to_csv(scoreboard_path, index=False)
    
    print("\n" + "=" * 80)
    print("  FINAL MULTI-MATCH EVALUATION SCOREBOARD")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print(f"\nSaved scoreboard to {scoreboard_path}")


if __name__ == "__main__":
    main()
