"""
5-Fold Chronological Cross-Validation with StatsBomb Bayesian Viterbi Decoder
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, '.')
from evaluation.possession_metrics import evaluate
from starter.step4_feature_engineering_benchmark import augment_kinematic_features
from starter.step14_statsbomb_bayesian_viterbi import load_statsbomb_priors, build_markov_transition_matrix, viterbi_decode

def main():
    print("="*110)
    print(" 5-FOLD CHRONOLOGICAL CROSS-VALIDATION WITH STATSBOMB BAYESIAN VITERBI DECODER")
    print("="*110)

    raw_barca = pd.read_csv('data/features_public_window.csv')
    raw_met = pd.read_csv('data/metrica_3games_features.csv')
    df_barca = augment_kinematic_features(raw_barca)
    df_met = augment_kinematic_features(raw_met)

    all_features = [c for c in df_barca.columns if c not in ['video_frame', 'frame_id', 'label']]
    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_true_all = np.array([label_map[x] for x in df_barca['label']], dtype=np.int64)

    priors = load_statsbomb_priors()
    A = build_markov_transition_matrix(priors)

    n_splits = 5
    fold_size = len(df_barca) // n_splits
    fold_spell_f1s = []
    fold_macro_f1s = []
    fold_accs = []

    for fold in range(n_splits):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_splits - 1 else len(df_barca)

        train_idx = list(range(0, val_start)) + list(range(val_end, len(df_barca)))
        val_idx = list(range(val_start, val_end))

        train_barca = df_barca.iloc[train_idx].copy()
        val_barca = df_barca.iloc[val_idx].copy().reset_index(drop=True)

        joint_df = pd.concat([df_met[all_features + ['label']], train_barca[all_features + ['label']]], ignore_index=True)
        sample_weights = np.ones(len(joint_df)); sample_weights[len(df_met):] = 30.0

        lgb_clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.035, num_leaves=38, random_state=42, verbosity=-1)
        lgb_clf.fit(joint_df[all_features], joint_df['label'], sample_weight=sample_weights)
        classes = list(lgb_clf.classes_)

        probs_val = lgb_clf.predict_proba(val_barca[all_features])
        viterbi_preds_val = viterbi_decode(probs_val, A, classes)

        y_true_val = y_true_all[val_idx]
        y_pred_val = np.array([label_map[x] for x in viterbi_preds_val], dtype=np.int64)
        res_val = evaluate(y_true_val, y_pred_val)
        acc_val = (y_true_val == y_pred_val).mean()

        fold_spell_f1s.append(res_val['spell_f1_mean'])
        fold_macro_f1s.append(res_val['macro_f1'])
        fold_accs.append(acc_val)
        print(f"Fold {fold+1}: Spell F1 = {res_val['spell_f1_mean']:.4f}, Macro F1 = {res_val['macro_f1']:.4f}, Accuracy = {acc_val:.2%}")

    print("\n" + "="*80)
    print("                    5-FOLD CROSS-VALIDATION SUMMARY")
    print("="*80)
    print(f"  Mean Out-Of-Fold Spell F1:  {np.mean(fold_spell_f1s):.4f} +/- {np.std(fold_spell_f1s):.4f}")
    print(f"  Mean Out-Of-Fold Macro F1:  {np.mean(fold_macro_f1s):.4f} +/- {np.std(fold_macro_f1s):.4f}")
    print(f"  Mean Out-Of-Fold Accuracy:  {np.mean(fold_accs):.2%} +/- {np.std(fold_accs):.2%}")
    print("="*80)

if __name__ == '__main__':
    main()
