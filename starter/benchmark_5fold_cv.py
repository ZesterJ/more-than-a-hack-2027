"""
Full 5-Fold Out-of-Sample Cross-Validation Benchmark

Evaluates our Champion Model (Step 11 Asymmetric State Machine + Multi-Scale Inception + GBDT)
across all 5 contiguous folds of the full Barcelona vs Athletic Club ground truth (7,500 frames, 17 spells).
Every frame is scored out-of-sample (trained on 4 folds + Metrica, tested on the 5th unseen fold).
"""

import sys
sys.path.insert(0, '.')
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from evaluation.possession_metrics import evaluate
from starter.step4_feature_engineering_benchmark import augment_kinematic_features
from starter.step7_multiscale_inception_benchmark import InceptionTimeNet
from starter.step10_breakthrough_physics import apply_breakthrough_physics
from starter.step11_perfect_alignment import asymmetric_spell_smoother, apply_sideline_throwin_rule

def run_5fold_cv():
    print("="*110)
    print(" 5-FOLD CONTIGUOUS OUT-OF-SAMPLE CROSS-VALIDATION BENCHMARK (ALL 17 SPELLS)")
    print("="*110)

    raw_barca = pd.read_csv('data/features_public_window.csv')
    raw_met = pd.read_csv('data/metrica_3games_features.csv')

    df_barca = augment_kinematic_features(raw_barca)
    df_met = augment_kinematic_features(raw_met)

    all_features = [c for c in df_barca.columns if c not in ['video_frame', 'frame_id', 'label']]
    label_map = {'H': 0, 'A': 1, 'D': 2}
    
    N = len(df_barca)  # 7,500 frames
    n_splits = 5
    fold_size = N // n_splits  # 1,500 frames each

    oof_preds = np.empty(N, dtype=object)
    fold_metrics = []

    print(f"Total Frames: {N:,} across {n_splits} Contiguous Folds ({fold_size} frames per fold)\n")

    for fold in range(n_splits):
        t0 = time.time()
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_splits - 1 else N

        test_idx = np.arange(test_start, test_end)
        train_idx = np.setdiff1d(np.arange(N), test_idx)

        train_barca = df_barca.iloc[train_idx].copy()
        test_barca = df_barca.iloc[test_idx].copy().reset_index(drop=True)

        # 1. GBDT with Domain Weights
        joint_df = pd.concat([df_met[all_features + ['label']], train_barca[all_features + ['label']]], ignore_index=True)
        sample_weights = np.ones(len(joint_df))
        sample_weights[len(df_met):] = 30.0

        lgb_clf = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=35, random_state=42, verbosity=-1)
        lgb_clf.fit(joint_df[all_features], joint_df['label'], sample_weight=sample_weights)
        classes = list(lgb_clf.classes_)
        probs_gbdt = lgb_clf.predict_proba(test_barca[all_features])

        # 2. InceptionTime CNN
        WINDOW = 65; HALF = WINDOW // 2
        X_train_num = train_barca[all_features].fillna(0).values; X_test_num = test_barca[all_features].fillna(0).values
        mean = np.mean(X_train_num, axis=0); std = np.std(X_train_num, axis=0) + 1e-6
        X_train_norm = (X_train_num - mean) / std; X_test_norm = (X_test_num - mean) / std
        X_train_pad = np.pad(X_train_norm, ((HALF, HALF), (0, 0)), mode='edge'); X_test_pad = np.pad(X_test_norm, ((HALF, HALF), (0, 0)), mode='edge')

        def make_seqs(arr):
            n_seq = len(arr) - WINDOW + 1; seqs = np.empty((n_seq, len(all_features), WINDOW), dtype=np.float32)
            for i in range(n_seq): seqs[i] = arr[i : i + WINDOW].T
            return seqs

        seqs_train = make_seqs(X_train_pad); seqs_test = make_seqs(X_test_pad)
        y_train_code = np.array([classes.index(c) for c in train_barca['label'].values], dtype=np.int64)

        inc_net = InceptionTimeNet(in_channels=len(all_features), num_classes=3)
        optimizer = torch.optim.AdamW(inc_net.parameters(), lr=0.002, weight_decay=1e-4)
        loader = DataLoader(TensorDataset(torch.tensor(seqs_train), torch.tensor(y_train_code)), batch_size=128, shuffle=True)

        inc_net.train()
        for epoch in range(10):
            for bx, by in loader:
                optimizer.zero_grad(); nn.CrossEntropyLoss()(inc_net(bx), by).backward(); optimizer.step()

        inc_net.eval()
        with torch.no_grad():
            probs_inc = torch.softmax(inc_net(torch.tensor(seqs_test)), dim=-1).cpu().numpy()

        # Blend
        probs_blend = 0.65 * probs_gbdt + 0.35 * probs_inc
        h_idx = classes.index('H'); a_idx = classes.index('A'); d_idx = classes.index('D')
        
        raw_preds = []
        for i in range(len(probs_blend)):
            if probs_blend[i, d_idx] >= 0.32: raw_preds.append('D')
            else: raw_preds.append('H' if probs_blend[i, h_idx] > probs_blend[i, a_idx] else 'A')

        refined_phys = apply_breakthrough_physics(raw_preds, probs_blend, test_barca, classes, aerial_speed_thresh=1.20)
        refined_throw = apply_sideline_throwin_rule(refined_phys, test_barca)
        final_fold_preds = asymmetric_spell_smoother(refined_throw, min_live_frames=35, min_dead_frames=60)

        oof_preds[test_idx] = final_fold_preds

        y_true_fold = np.array([label_map[x] for x in test_barca['label']], dtype=np.int64)
        y_pred_fold = np.array([label_map[x] for x in final_fold_preds], dtype=np.int64)
        res_fold = evaluate(y_true_fold, y_pred_fold)
        acc_fold = (y_true_fold == y_pred_fold).mean()
        n_sp = (y_pred_fold[:-1] != y_pred_fold[1:]).sum() + 1
        
        print(f"Fold {fold+1}/5 (frames {test_start:4d}..{test_end:4d}) | Spells: {n_sp:2d} | SpellF1: {res_fold['spell_f1_mean']:.4f} | MacroF1: {res_fold['macro_f1']:.4f} | Acc: {acc_fold:.2%} ({time.time()-t0:.1f}s)")
        fold_metrics.append(res_fold['spell_f1_mean'])

    # Overall Out-of-Fold Evaluation over all 7,500 frames (all 17 spells)
    y_true_all = np.array([label_map[x] for x in df_barca['label']], dtype=np.int64)
    y_pred_all = np.array([label_map[x] for x in oof_preds], dtype=np.int64)
    
    res_overall = evaluate(y_true_all, y_pred_all)
    acc_overall = (y_true_all == y_pred_all).mean()
    n_sp_all = (y_pred_all[:-1] != y_pred_all[1:]).sum() + 1

    print("\n" + "="*110)
    print(" FINAL OUT-OF-FOLD GENERALIZATION ACROSS ENTIRE MATCH GROUND TRUTH (7,500 FRAMES):")
    print("="*110)
    print(f"  Total Ground Truth Possessions: 17")
    print(f"  Total Predicted Possessions:    {n_sp_all}")
    print(f"  Mean 5-Fold Spell F1:           {np.mean(fold_metrics):.4f}")
    print(f"  Overall Out-of-Fold Spell F1:   {res_overall['spell_f1_mean']:.4f}")
    print(f"  Overall Out-of-Fold Macro F1:   {res_overall['macro_f1']:.4f}")
    print(f"  Overall Frame Accuracy:         {acc_overall:.2%}")
    print(f"  F1 Home (Barcelona):            {res_overall['per_class']['H']['f1']:.4f}")
    print(f"  F1 Away (Athletic Club):        {res_overall['per_class']['A']['f1']:.4f}")
    print(f"  F1 Dead (Stoppages):            {res_overall['per_class']['D']['f1']:.4f}")
    print("="*110)

if __name__ == '__main__':
    run_5fold_cv()
