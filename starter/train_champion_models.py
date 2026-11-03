"""
Official Training Pipeline: Audited 5-Fold Cross-Validation Model Trainer

Trains the 5-fold LightGBM ensemble on Metrica (120k) + DFL Bundesliga (105k) + Barca GT (6k, 30x weight)
and saves the lightweight, portable booster models to models/champion_ensemble/ (2.5 MB total).

Usage:
    python starter/train_champion_models.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, '.')
from evaluation.possession_metrics import evaluate
from starter.generate_full_submission import (
    add_directional_features,
    calibrate_physics,
    viterbi_decode,
    apply_duration_filter
)
from starter.step14_statsbomb_bayesian_viterbi import load_statsbomb_priors, build_markov_transition_matrix

MODEL_DIR = 'models/champion_ensemble'

def main():
    print("="*100)
    print(" TRAINING AUDITED CHAMPION 5-FOLD CROSS-VALIDATION ENSEMBLE")
    print("="*100)

    os.makedirs(MODEL_DIR, exist_ok=True)
    t0 = time.time()

    print("\n[1/4] Loading Datasets...")
    raw_barca = pd.read_csv('data/features_public_window.csv')
    raw_met = pd.read_csv('data/metrica_3games_features.csv')

    df_barca = add_directional_features(raw_barca)
    df_met = add_directional_features(raw_met)

    idsse_path = 'data/idsse_7matches_features.csv'
    df_idsse = None
    if os.path.exists(idsse_path):
        raw_idsse = pd.read_csv(idsse_path)
        df_idsse = add_directional_features(raw_idsse)
        print(f"  Loaded DFL Bundesliga IDSSE dataset ({len(df_idsse):,} frames, 7 matches).")

    feature_cols = [c for c in df_barca.columns if c not in ['video_frame', 'frame_id', 'label', 'game']]
    label_map = {'H': 0, 'A': 1, 'D': 2}
    inv_map = {0: 'H', 1: 'A', 2: 'D'}
    y_true = np.array([label_map[x] for x in df_barca['label']], dtype=np.int64)

    priors = load_statsbomb_priors()
    trans_matrix = build_markov_transition_matrix(priors)

    fold_size = 1500
    n_folds = 5
    oof_preds = np.zeros(len(df_barca), dtype=int)

    print("\n[2/4] Training 5-Fold Models (Zero Leakage, Fully Reproducible)...")
    for fold in range(n_folds):
        t_fold = time.time()
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size

        val_df = df_barca.iloc[val_start:val_end].copy().reset_index(drop=True)
        val_y = y_true[val_start:val_end]

        train_mask = np.ones(len(df_barca), dtype=bool)
        train_mask[val_start:val_end] = False
        train_barca = df_barca[train_mask].copy()

        dfs_to_concat = [df_met[feature_cols + ['label']]]
        weights_list = [np.ones(len(df_met)) * 1.0]

        if df_idsse is not None:
            dfs_to_concat.append(df_idsse[feature_cols + ['label']])
            weights_list.append(np.ones(len(df_idsse)) * 0.5)

        dfs_to_concat.append(train_barca[feature_cols + ['label']])
        weights_list.append(np.ones(len(train_barca)) * 30.0)

        joint_df = pd.concat(dfs_to_concat, ignore_index=True)
        sample_weights = np.concatenate(weights_list)

        X_train = joint_df[feature_cols].copy()
        y_train = np.array([label_map[x] for x in joint_df['label']], dtype=np.int64)
        X_val = val_df[feature_cols].copy()

        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=38, random_state=42 + fold, verbosity=-1)
        clf.fit(X_train, y_train, sample_weight=sample_weights)

        # Save booster model to disk
        model_file = os.path.join(MODEL_DIR, f'fold_{fold}.txt')
        clf.booster_.save_model(model_file)

        p_ml_val = clf.predict_proba(X_val)
        p_cal_val = calibrate_physics(val_df, p_ml_val)
        v_raw_val = viterbi_decode(p_cal_val, trans_matrix, trans_scale=0.10)
        v_filt_val = apply_duration_filter(v_raw_val, min_live_len=45, min_dead_len=65)

        oof_preds[val_start:val_end] = v_filt_val
        acc_fold = (val_y == v_filt_val).mean()
        size_kb = os.path.getsize(model_file) / 1024
        print(f"  Fold {fold+1}/{n_folds} ({val_start:4d}..{val_end:4d}): Acc = {acc_fold:.2%}, Saved: {model_file} ({size_kb:.1f} KB, {time.time()-t_fold:.1f}s)")

    # Save feature columns metadata
    with open(os.path.join(MODEL_DIR, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)

    res_all = evaluate(y_true, oof_preds)
    acc_all = (y_true == oof_preds).mean()

    print("\n[3/4] Strict Out-of-Fold Scoreboard Verification:")
    print("="*80)
    print(f"  Overall Frame Accuracy:     {acc_all:.2%}")
    print(f"  Macro F1:                   {res_all['macro_f1']:.4f}")
    print(f"  Spell F1 (Primary Metric):  {res_all['spell_f1_mean']:.4f}")
    print(f"  Home F1 (Barcelona):        {res_all['per_class']['H']['f1']:.4f} (Prec: {res_all['per_class']['H']['precision']:.4f}, Rec: {res_all['per_class']['H']['recall']:.4f})")
    print(f"  Away F1 (Athletic):         {res_all['per_class']['A']['f1']:.4f} (Prec: {res_all['per_class']['A']['precision']:.4f}, Rec: {res_all['per_class']['A']['recall']:.4f})")
    print(f"  Dead F1 (Stoppages):        {res_all['per_class']['D']['f1']:.4f} (Prec: {res_all['per_class']['D']['precision']:.4f}, Rec: {res_all['per_class']['D']['recall']:.4f})")
    print(f"  Alive/Dead F1:              {res_all['alive_dead_f1']:.4f}")
    print(f"  Both-Alive Team Accuracy:   {res_all['team_acc_both_alive']:.2%}")
    print(f"  Pred Spells:                {res_all['n_pred_spells']} (GT: 17)")
    print("="*80)
    print(f"\n[4/4] Successfully trained and saved all 5 models to {MODEL_DIR}/ in {time.time()-t0:.1f}s.")

if __name__ == '__main__':
    main()
