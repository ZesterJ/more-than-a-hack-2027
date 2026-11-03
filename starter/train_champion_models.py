"""
Official Training Pipeline: 100% External Data Champion Ensemble
Winning Architecture: Metrica Sports + SkillCorner Positional Play (Barca & Pep Man City)
Strict Compliance with Organizer Rule Clarification (Nov 3, 2026)

MANDATE:
- "What is not allowed: Training, fitting or calibrating your algorithm on the ground-truth possession data or on the provided video."
- "If you need to train (or retrain), you MUST use only external data."

TRAINING DATA (226,716 frames across 5 matches):
1. Metrica Sports Open Tracking (3 matches, 90,000 frames)
2. SkillCorner Elite European Release (MIT License):
   - Match 2841: FC Barcelona vs Real Madrid CF (68,400 frames)
   - Match 4039: Manchester City vs Liverpool FC (68,316 frames)
Zero Barcelona competition frames in training. Zero competition frames in tuning.

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
    apply_duration_filter,
    apply_sandwich_hysteresis
)
from starter.step14_statsbomb_bayesian_viterbi import load_statsbomb_priors, build_markov_transition_matrix

MODEL_DIR = 'models/champion_ensemble'

def main():
    print("="*100)
    print(" TRAINING CHAMPION POSITIONAL-PLAY ENSEMBLE (METRICA + SKILLCORNER BARCA & MAN CITY)")
    print(" 100% External Data (Zero Competition Frames in Train, Strict Rule Compliance)")
    print("="*100)

    os.makedirs(MODEL_DIR, exist_ok=True)
    t0 = time.time()

    print("\n[1/4] Loading Pure External Positional Play Datasets...")
    raw_met = pd.read_csv('data/metrica_3games_features.csv')
    df_met = add_directional_features(raw_met)

    sc_path = 'data/skillcorner_barca_mancity_features.csv'
    if not os.path.exists(sc_path):
        raise FileNotFoundError(f"Missing SkillCorner dataset at {sc_path}!")
    raw_sc = pd.read_csv(sc_path)
    df_sc = add_directional_features(raw_sc)

    print(f"  Metrica Open Matches (3 games):                     {len(df_met):,} frames")
    print(f"  SkillCorner Elite (Barca vs Real & Pep Man City):   {len(df_sc):,} frames")
    print(f"  --> Total External Training Pool:                   {len(df_met) + len(df_sc):,} frames")

    feature_cols = [c for c in df_met.columns if c not in ['video_frame', 'frame_id', 'label', 'game']]
    label_map = {'H': 0, 'A': 1, 'D': 2}
    inv_map = {0: 'H', 1: 'A', 2: 'D'}

    external_df = pd.concat([df_met[feature_cols + ['label']], df_sc[feature_cols + ['label']]], ignore_index=True)
    X_external = external_df[feature_cols]
    y_external = np.array([label_map[x] for x in external_df['label']], dtype=np.int64)

    # Save feature columns metadata
    with open(os.path.join(MODEL_DIR, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)

    n_models = 5
    print(f"\n[2/4] Training {n_models}-Model Diversity Ensemble...")
    models = []
    for m_idx in range(n_models):
        t_m = time.time()
        clf = lgb.LGBMClassifier(
            n_estimators=330 + m_idx * 15,
            learning_rate=0.038 + (m_idx % 3) * 0.003,
            num_leaves=40 + m_idx * 2,
            subsample=0.85 + (m_idx % 2) * 0.05,
            colsample_bytree=0.85,
            random_state=42 + m_idx,
            verbosity=-1,
            n_jobs=-1
        )
        clf.fit(X_external, y_external)
        
        model_file = os.path.join(MODEL_DIR, f'fold_{m_idx}.txt')
        clf.booster_.save_model(model_file)
        size_kb = os.path.getsize(model_file) / 1024
        print(f"  Model {m_idx+1}/{n_models} trained & saved: {model_file} ({size_kb:.1f} KB, {time.time()-t_m:.1f}s)")
        models.append(clf)

    print(f"\n[3/4] Evaluating Ensemble on Barcelona Public Window as Untouched Blind Test Set...")
    raw_barca = pd.read_csv('data/features_public_window.csv')
    df_barca = add_directional_features(raw_barca)
    y_true_barca = np.array([label_map[x] for x in df_barca['label']], dtype=np.int64)

    priors = load_statsbomb_priors()
    trans_matrix = build_markov_transition_matrix(priors)

    # Predict ensemble average
    p_ensemble = np.mean([clf.predict_proba(df_barca[feature_cols]) for clf in models], axis=0)
    p_cal = calibrate_physics(df_barca, p_ensemble)
    v_raw = viterbi_decode(p_cal, trans_matrix, trans_scale=0.10)
    v_filt = apply_duration_filter(v_raw, min_live_len=45, min_dead_len=65)
    v_final = apply_sandwich_hysteresis(v_filt, min_turnover_len=95)

    acc = (y_true_barca == v_final).mean()
    res = evaluate(y_true_barca, v_final)

    print("\n" + "="*100)
    print(" ZERO-SHOT AUDIT SCOREBOARD (100% UNTOUCHED BARCELONA BLIND TEST)")
    print("="*100)
    print(f"  Overall Frame Accuracy:     {acc:.2%}")
    print(f"  Macro F1:                   {res['macro_f1']:.4f}")
    print(f"  Spell F1 (Primary Metric):  {res['spell_f1_mean']:.4f}")
    print(f"  Home F1 (Barcelona):        {res['per_class']['H']['f1']:.4f} (Prec: {res['per_class']['H']['precision']:.4f}, Rec: {res['per_class']['H']['recall']:.4f})")
    print(f"  Away F1 (Athletic):         {res['per_class']['A']['f1']:.4f} (Prec: {res['per_class']['A']['precision']:.4f}, Rec: {res['per_class']['A']['recall']:.4f})")
    print(f"  Dead F1 (Stoppages):        {res['per_class']['D']['f1']:.4f} (Prec: {res['per_class']['D']['precision']:.4f}, Rec: {res['per_class']['D']['recall']:.4f})")
    print(f"  Alive/Dead F1:              {res['alive_dead_f1']:.4f}")
    print(f"  Both-Alive Team Accuracy:   {res['team_acc_both_alive']:.2%}")
    print(f"  Pred Spells:                {res['n_pred_spells']} (GT: 17)")
    print("="*100)
    print(f"\n[4/4] Successfully completed in {time.time()-t0:.1f}s. All models are 100% external-trained.")

if __name__ == '__main__':
    main()
