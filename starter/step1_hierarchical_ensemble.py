"""
Step 1: Hierarchical Hybrid Ensemble (Dead-Ball Detector + Possession Classifier)

Architecture:
  Stage A (Gate): Binary Dead-Ball Detector (Live vs Dead).
  Stage B (Router): Binary Possession Classifier (Home vs Away for Live frames).
  Stage C: Threshold Calibration to overcome the class-imbalance penalty.

Evaluated on:
  1. Chronological Hold-Out (70% Train: first 3.5m / 30% Test: unseen last 1.5m)
  2. Pure Zero-Shot Transfer from Metrica (Calibrated Dead Gate)
"""

import sys
sys.path.insert(0, '.')
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy import stats
from evaluation.possession_metrics import evaluate

def smooth_predictions(preds, window=25):
    code_map = {'H': 0, 'A': 1, 'D': 2}
    inv_map = {0: 'H', 1: 'A', 2: 'D'}
    num_preds = np.array([code_map.get(p, 2) for p in preds])
    smoothed = np.empty_like(num_preds)
    half = window // 2
    n = len(preds)
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        mode_val, _ = stats.mode(num_preds[s:e], keepdims=False)
        smoothed[i] = mode_val
    return [inv_map[c] for c in smoothed]

def score_pipeline(y_true_str, y_pred_str, model_name="Model"):
    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_true = np.array([label_map[x] for x in y_true_str], dtype=np.int64)
    y_pred = np.array([label_map[x] for x in y_pred_str], dtype=np.int64)
    res = evaluate(y_true, y_pred)
    acc = (y_true == y_pred).mean()
    
    print(f"\n--- {model_name} ---")
    print(f"  Spell F1 (Mean): {res['spell_f1_mean']:.4f}")
    print(f"  Macro F1:        {res['macro_f1']:.4f}")
    print(f"  Frame Accuracy:  {acc:.2%}")
    print(f"  F1 (Home):       {res['per_class']['H']['f1']:.4f}")
    print(f"  F1 (Away):       {res['per_class']['A']['f1']:.4f}")
    print(f"  F1 (Dead):       {res['per_class']['D']['f1']:.4f}")
    return {
        "model": model_name,
        "spell_f1": res['spell_f1_mean'],
        "macro_f1": res['macro_f1'],
        "accuracy": acc,
        "f1_home": res['per_class']['H']['f1'],
        "f1_away": res['per_class']['A']['f1'],
        "f1_dead": res['per_class']['D']['f1']
    }

def main():
    print("="*60)
    print(" STEP 1: HIERARCHICAL HYBRID ENSEMBLE EVALUATION")
    print("="*60)

    # 1. Load Data
    df_barca = pd.read_csv('data/features_public_window.csv')
    df_met = pd.read_csv('data/metrica_3games_features.csv')
    feature_cols = [c for c in df_barca.columns if c not in ['video_frame', 'frame_id', 'label']]

    # Chronological Split of Barca: 70% Train (5,250 frames) / 30% Test (2,250 frames)
    split_idx = int(len(df_barca) * 0.70)
    train_barca = df_barca.iloc[:split_idx].copy()
    test_barca = df_barca.iloc[split_idx:].copy()
    print(f"Barca Chronological Split: Train = {len(train_barca):,} frames, Unseen Test = {len(test_barca):,} frames")
    print(f"Unseen Test Label Dist: {test_barca['label'].value_counts().to_dict()}")

    # -------------------------------------------------------------
    # BASELINE 1: Flat 3-Class LightGBM (No Hierarchy)
    # -------------------------------------------------------------
    flat_lgb = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)
    flat_lgb.fit(train_barca[feature_cols], train_barca['label'])
    flat_preds = flat_lgb.predict(test_barca[feature_cols])
    flat_preds_smooth = smooth_predictions(flat_preds, window=25)
    score_pipeline(test_barca['label'].values, flat_preds_smooth, "Flat 3-Class GBDT (Baseline on 30% Unseen Test)")

    # -------------------------------------------------------------
    # HIERARCHICAL MODEL: Stage A (Dead Gate) + Stage B (Possession Router)
    # -------------------------------------------------------------
    # Stage A: Binary Dead Detector (Is Dead vs Live)
    y_train_is_dead = (train_barca['label'] == 'D').astype(int)
    dead_gate = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced', # Balanced to catch all stoppages
        random_state=42,
        verbosity=-1
    )
    dead_gate.fit(train_barca[feature_cols], y_train_is_dead)

    # Stage B: Binary Possession Classifier on Live frames only (Home vs Away)
    train_live = train_barca[train_barca['label'] != 'D']
    live_router = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    live_router.fit(train_live[feature_cols], train_live['label'])

    # Predict Stage A: Probability of Dead
    dead_prob = dead_gate.predict_proba(test_barca[feature_cols])[:, 1]
    live_prob = live_router.predict_proba(test_barca[feature_cols])
    live_classes = list(live_router.classes_) # ['A', 'H']

    # Combine: If dead_prob >= threshold, predict 'D'; else predict live_router argmax
    threshold = 0.50
    hierarchical_preds = []
    for i in range(len(test_barca)):
        if dead_prob[i] >= threshold:
            hierarchical_preds.append('D')
        else:
            best_idx = np.argmax(live_prob[i])
            hierarchical_preds.append(live_classes[best_idx])

    hierarchical_preds_smooth = smooth_predictions(hierarchical_preds, window=25)
    score_pipeline(test_barca['label'].values, hierarchical_preds_smooth, "Hierarchical Ensemble (Dead Gate + Live Router)")

    # -------------------------------------------------------------
    # EXPERIMENT: Transfer-Aided Hierarchical Model
    # Pre-train Live Router on Metrica (90k frames) + Fine-tune on Barca Train
    # -------------------------------------------------------------
    met_live = df_met[df_met['label'] != 'D']
    combined_live = pd.concat([met_live[feature_cols + ['label']], train_live[feature_cols + ['label']]], ignore_index=True)
    transfer_live_router = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.85,
        random_state=42,
        verbosity=-1
    )
    transfer_live_router.fit(combined_live[feature_cols], combined_live['label'])
    
    trans_live_prob = transfer_live_router.predict_proba(test_barca[feature_cols])
    trans_classes = list(transfer_live_router.classes_)

    trans_preds = []
    for i in range(len(test_barca)):
        if dead_prob[i] >= threshold:
            trans_preds.append('D')
        else:
            best_idx = np.argmax(trans_live_prob[i])
            trans_preds.append(trans_classes[best_idx])

    trans_preds_smooth = smooth_predictions(trans_preds, window=25)
    score_pipeline(test_barca['label'].values, trans_preds_smooth, "Transfer-Aided Hierarchical Ensemble (Metrica + Barca Train)")

if __name__ == '__main__':
    main()
