"""
Official Champion Pipeline: 100% External-Trained Zero-Shot Benchmark
Strict Compliance with Organizer Rule Clarification (Nov 3, 2026)

MANDATE:
- "What is not allowed: Training, fitting or calibrating your algorithm on the ground-truth possession data or on the provided video."
- "If you need to train (or retrain), you MUST use only external data."

This benchmark evaluates the externally-trained 5-model diversity ensemble on all 7,500
ground-truth frames of the Barcelona public window as an untouched, unseen blind test set.

Usage:
    python starter/evaluate_champion.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, '.')
from evaluation.possession_metrics import evaluate
from starter.generate_full_submission import (
    add_directional_features,
    calibrate_physics,
    viterbi_decode,
    apply_duration_filter,
    apply_sandwich_hysteresis
)
from starter.step14_statsbomb_bayesian_viterbi import load_statsbomb_priors, build_markov_transition_matrix

def main():
    print("="*100)
    print(" EVALUATING 100% EXTERNAL-TRAINED ENSEMBLE (ORGANIZER RULE COMPLIANT)")
    print(" Zero-Shot Blind Test on Provided Barcelona Public Window (7,500 frames)")
    print("="*100)

    t0 = time.time()
    raw_barca = pd.read_csv('data/features_public_window.csv')
    df_barca = add_directional_features(raw_barca)

    models_dir = 'models/champion_ensemble'
    feature_cols_path = os.path.join(models_dir, 'feature_cols.json')
    if os.path.exists(feature_cols_path):
        import json
        with open(feature_cols_path, 'r') as f:
            feature_cols = json.load(f)
    else:
        feature_cols = [c for c in df_barca.columns if c not in ['video_frame', 'frame_id', 'label', 'game']]

    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_true = np.array([label_map[x] for x in df_barca['label']], dtype=np.int64)

    priors = load_statsbomb_priors()
    trans_matrix = build_markov_transition_matrix(priors)

    n_models = 5
    models = []
    for f in range(n_models):
        model_file = os.path.join(models_dir, f'fold_{f}.txt')
        models.append(lgb.Booster(model_file=model_file))

    print(f"  Loaded {len(models)} external booster models ({time.time()-t0:.2f}s).")
    
    # Predict ensemble average
    p_ens = np.mean([m.predict(df_barca[feature_cols]) for m in models], axis=0)
    p_cal = calibrate_physics(df_barca, p_ens)
    v_raw = viterbi_decode(p_cal, trans_matrix, trans_scale=0.10)
    v_filt = apply_duration_filter(v_raw, min_live_len=45, min_dead_len=65)
    v_final = apply_sandwich_hysteresis(v_filt, min_turnover_len=95)

    res = evaluate(y_true, v_final)
    acc = (y_true == v_final).mean()

    print("\n" + "="*100)
    print(" ZERO-SHOT AUDIT SCOREBOARD (100% UNTOUCHED BARCELONA BLIND TEST)")
    print("="*100)
    print(f"  Overall Frame Accuracy:     {acc:.2%}")
    print(f"  Macro F1:                   {res['macro_f1']:.4f}")
    print(f"  Spell F1 (Ranking Metric):  {res['spell_f1_mean']:.4f}")
    print(f"  IoU >= 0.5 Spell F1:        {res.get('spell_f1_0.5', 'N/A')}")
    print(f"  Home F1 (Barcelona):        {res['per_class']['H']['f1']:.4f} (Prec: {res['per_class']['H']['precision']:.4f}, Rec: {res['per_class']['H']['recall']:.4f})")
    print(f"  Away F1 (Athletic):         {res['per_class']['A']['f1']:.4f} (Prec: {res['per_class']['A']['precision']:.4f}, Rec: {res['per_class']['A']['recall']:.4f})")
    print(f"  Dead F1 (Stoppages):        {res['per_class']['D']['f1']:.4f} (Prec: {res['per_class']['D']['precision']:.4f}, Rec: {res['per_class']['D']['recall']:.4f})")
    print(f"  Alive/Dead F1:              {res['alive_dead_f1']:.4f}")
    print(f"  Both-Alive Team Accuracy:   {res['team_acc_both_alive']:.2%}")
    print(f"  Predicted Spells:           {res['n_pred_spells']} (GT: 17)")
    print("="*100)
    print(f" Completed audited zero-shot evaluation in {time.time()-t0:.2f}s.")

if __name__ == '__main__':
    main()
