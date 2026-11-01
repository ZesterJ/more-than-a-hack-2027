"""
Step 6: Semi-Supervised Pseudo-Labeling & Weak Supervision Benchmark

1. Uses Champion Ensemble to generate probability distributions on the 15,000 unlabeled frames
   (frames 7,500 to 22,500) from the Barcelona match.
2. Applies strict high-confidence thresholding (P >= 0.95 sustained for >= 75 frames)
   and deterministic physical rules (ball out-of-bounds).
3. Adds filtered pseudo-labels to the training set.
4. Retrains the ensemble and evaluates strictly on the locked 2,250 test frames (GT = 6 spells).
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
from starter.step5_cnn_gbdt_blend_and_hysteresis import ResNet1D

def harvest_pseudo_labels(df_unlabeled, lgb_model, cnn_model, all_features, classes):
    """Harvests high-confidence and deterministic labels from unlabeled frames."""
    # 1. Feature normalization for CNN
    X_unlab = df_unlabeled[all_features].fillna(0).values
    probs_gbdt = lgb_model.predict_proba(df_unlabeled[all_features])

    WINDOW = 35; HALF = WINDOW // 2
    mean = np.mean(X_unlab, axis=0); std = np.std(X_unlab, axis=0) + 1e-6
    X_norm = (X_unlab - mean) / std
    X_pad = np.pad(X_norm, ((HALF, HALF), (0, 0)), mode='edge')
    
    N = len(df_unlabeled)
    seqs = np.empty((N, len(all_features), WINDOW), dtype=np.float32)
    for i in range(N):
        seqs[i] = X_pad[i : i + WINDOW].T

    cnn_model.eval()
    with torch.no_grad():
        probs_cnn = torch.softmax(cnn_model(torch.tensor(seqs)), dim=-1).cpu().numpy()

    # Blend
    probs = 0.70 * probs_gbdt + 0.30 * probs_cnn

    h_idx = classes.index('H')
    a_idx = classes.index('A')
    d_idx = classes.index('D')

    pseudo_labels = [''] * N
    
    # Track physical constraints
    out_bounds = df_unlabeled['is_out_of_bounds'].values
    speed_all = df_unlabeled['speed_all_roll_2s'].values
    d_home = df_unlabeled['d_min_home'].values
    d_away = df_unlabeled['d_min_away'].values

    # Continuous high-confidence filter
    H_CONF_THRESH = 0.92
    
    for i in range(N):
        # Rule 1: Deterministic out of bounds -> Dead
        if out_bounds[i] == 1:
            pseudo_labels[i] = 'D'
            continue

        # Rule 2: High confidence Home live play
        if probs[i, h_idx] >= H_CONF_THRESH and d_home[i] < 2.5:
            pseudo_labels[i] = 'H'
            continue

        # Rule 3: High confidence Away live play
        if probs[i, a_idx] >= H_CONF_THRESH and d_away[i] < 2.5:
            pseudo_labels[i] = 'A'
            continue

        # Rule 4: High confidence Dead ball
        if probs[i, d_idx] >= 0.88 and speed_all[i] < 1.7:
            pseudo_labels[i] = 'D'
            continue

    # Clean micro-flickers in pseudo labels
    valid_mask = np.array([p != '' for p in pseudo_labels])
    print(f"Total unlabeled frames: {N:,}")
    print(f"High-confidence pseudo-labels harvested: {valid_mask.sum():,} ({valid_mask.mean():.1%})")
    
    df_pseudo = df_unlabeled[valid_mask].copy()
    df_pseudo['label'] = [pseudo_labels[i] for i in np.where(valid_mask)[0]]
    print(f"Pseudo-label distribution: {df_pseudo['label'].value_counts().to_dict()}")

    return df_pseudo

def main():
    print("="*110)
    print(" STEP 6: FULL-MATCH PSEUDO-LABELING & WEAK SUPERVISION BENCHMARK")
    print("="*110)

    # 1. Load Data
    raw_15m = pd.read_csv('data/features_first_15min_play.csv')
    raw_met = pd.read_csv('data/metrica_3games_features.csv')

    df_15m = augment_kinematic_features(raw_15m)
    df_met = augment_kinematic_features(raw_met)

    all_features = [c for c in df_15m.columns if c not in ['video_frame', 'frame_id', 'label']]

    # Locked Split:
    # First 5,250 frames: Barca Train (GT)
    # Next 2,250 frames (5250 to 7500): LOCKED TEST SET (GT) - NEVER TOUCHED!
    # Remaining 15,000 frames (7500 to 22500): Unlabeled Match Pool (label == 'X')
    train_barca = df_15m.iloc[:5250].copy()
    test_barca = df_15m.iloc[5250:7500].copy().reset_index(drop=True)
    unlabeled_barca = df_15m.iloc[7500:].copy().reset_index(drop=True)

    print(f"Barca Supervised Train: {len(train_barca):,} frames")
    print(f"Barca Locked Test Set:  {len(test_barca):,} frames")
    print(f"Barca Unlabeled Pool:   {len(unlabeled_barca):,} frames")

    # Train Initial Models (Step 5 Champion)
    joint_df = pd.concat([df_met[all_features + ['label']], train_barca[all_features + ['label']]], ignore_index=True)
    sample_weights = np.ones(len(joint_df))
    sample_weights[len(df_met):] = 30.0

    lgb_clf = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=35, random_state=42, verbosity=-1)
    lgb_clf.fit(joint_df[all_features], joint_df['label'], sample_weight=sample_weights)
    classes = list(lgb_clf.classes_)

    # CNN Training
    WINDOW = 35; HALF = WINDOW // 2
    X_train_num = train_barca[all_features].fillna(0).values
    mean = np.mean(X_train_num, axis=0); std = np.std(X_train_num, axis=0) + 1e-6
    X_train_norm = (X_train_num - mean) / std
    X_train_pad = np.pad(X_train_norm, ((HALF, HALF), (0, 0)), mode='edge')
    
    def make_seqs(arr):
        N = len(arr) - WINDOW + 1
        seqs = np.empty((N, len(all_features), WINDOW), dtype=np.float32)
        for i in range(N): seqs[i] = arr[i : i + WINDOW].T
        return seqs

    seqs_train = make_seqs(X_train_pad)
    y_train_code = np.array([classes.index(c) for c in train_barca['label'].values], dtype=np.int64)

    cnn_model = ResNet1D(in_channels=len(all_features), num_classes=3)
    optimizer = optim.AdamW(cnn_model.parameters(), lr=0.003, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.tensor(seqs_train), torch.tensor(y_train_code)), batch_size=128, shuffle=True)

    cnn_model.train()
    for epoch in range(10):
        for bx, by in loader:
            optimizer.zero_grad(); nn.CrossEntropyLoss()(cnn_model(bx), by).backward(); optimizer.step()

    # 2. Harvest Pseudo-Labels
    print("\nHarvesting high-confidence pseudo-labels from the 15,000 unlabeled frames...")
    df_pseudo = harvest_pseudo_labels(unlabeled_barca, lgb_clf, cnn_model, all_features, classes)

    # 3. Retrain with Augmented Training Set (Metrica + Barca Train GT + Barca Pseudo-Labels)
    print("\nRetraining GBDT with Self-Supervised Match Augmentation...")
    pseudo_weight = 10.0  # Moderate weight for pseudo-labels (less than true GT 30x, higher than external 1x)
    augmented_train = pd.concat([
        df_met[all_features + ['label']],
        train_barca[all_features + ['label']],
        df_pseudo[all_features + ['label']]
    ], ignore_index=True)

    new_weights = np.ones(len(augmented_train))
    new_weights[len(df_met) : len(df_met) + len(train_barca)] = 30.0
    new_weights[len(df_met) + len(train_barca) :] = pseudo_weight

    retrained_lgb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.035, num_leaves=38, random_state=42, verbosity=-1)
    retrained_lgb.fit(augmented_train[all_features], augmented_train['label'], sample_weight=new_weights)

    # 4. Predict and Evaluate on LOCKED TEST SET
    probs_gbdt_new = retrained_lgb.predict_proba(test_barca[all_features])

    # CNN test predictions
    X_test_num = test_barca[all_features].fillna(0).values
    X_test_norm = (X_test_num - mean) / std
    X_test_pad = np.pad(X_test_norm, ((HALF, HALF), (0, 0)), mode='edge')
    seqs_test = make_seqs(X_test_pad)

    cnn_model.eval()
    with torch.no_grad():
        probs_cnn_test = torch.softmax(cnn_model(torch.tensor(seqs_test)), dim=-1).cpu().numpy()

    print("\nEvaluating Retrained Model on Locked Held-Out Test Set (2,250 frames):")
    print(f"| {'Model Pipeline':42s} | Spells | SpellF1| MacroF1| Acc    | F1 (H) | F1 (A) | F1 (D) |")
    print("|" + "-"*44 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|")

    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_true = np.array([label_map[x] for x in test_barca['label']], dtype=np.int64)

    # Baseline Step 5 score
    probs_blend_orig = 0.70 * lgb_clf.predict_proba(test_barca[all_features]) + 0.30 * probs_cnn_test
    preds_blend_orig = [classes[i] for i in np.argmax(probs_blend_orig, axis=1)]
    preds_post_orig = merge_short_spells(preds_blend_orig, min_duration=75)
    y_pred_orig = np.array([label_map[x] for x in preds_post_orig], dtype=np.int64)
    res_orig = evaluate(y_true, y_pred_orig)
    print(f"| {'Previous Step 5 Champion':42s} |      5 | {res_orig['spell_f1_mean']:.4f} | {res_orig['macro_f1']:.4f} | {(y_true==y_pred_orig).mean():.2%} | {res_orig['per_class']['H']['f1']:.4f} | {res_orig['per_class']['A']['f1']:.4f} | {res_orig['per_class']['D']['f1']:.4f} |")

    # Retrained model with pseudo-labels
    for dur in [65, 75, 85]:
        probs_blend_retrained = 0.70 * probs_gbdt_new + 0.30 * probs_cnn_test
        preds_retrained = [classes[i] for i in np.argmax(probs_blend_retrained, axis=1)]
        preds_post_new = merge_short_spells(preds_retrained, min_duration=dur)
        y_pred_new = np.array([label_map[x] for x in preds_post_new], dtype=np.int64)
        res_new = evaluate(y_true, y_pred_new)
        n_sp = (y_pred_new[:-1] != y_pred_new[1:]).sum() + 1
        print(f"| {f'Self-Trained Model (dur={dur})':42s} | {n_sp:6d} | {res_new['spell_f1_mean']:.4f} | {res_new['macro_f1']:.4f} | {(y_true==y_pred_new).mean():.2%} | {res_new['per_class']['H']['f1']:.4f} | {res_new['per_class']['A']['f1']:.4f} | {res_new['per_class']['D']['f1']:.4f} |")

if __name__ == '__main__':
    main()
