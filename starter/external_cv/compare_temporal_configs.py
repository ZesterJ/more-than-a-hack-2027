"""
Compare Previous vs Winning External Temporal Parameters on Blind Public Window
"""

import sys
sys.path.insert(0, '.')
import json
import numpy as np
import pandas as pd
from evaluation.possession_metrics import evaluate
from starter.generate_full_submission import (
    add_directional_features,
    calibrate_physics,
    viterbi_decode,
    apply_duration_filter,
    apply_sandwich_hysteresis
)
from starter.external_cv.test_refined_features import add_refined_features
from starter.step14_statsbomb_bayesian_viterbi import load_statsbomb_priors, build_markov_transition_matrix
import lightgbm as lgb
import xgboost as xgb

def main():
    MODEL_DIR = 'models/refined_champion_ensemble'
    with open(f'{MODEL_DIR}/feature_cols.json') as f:
        feature_cols = json.load(f)

    lgb_models = [lgb.Booster(model_file=f'{MODEL_DIR}/lgb_fold_{i}.txt') for i in range(5)]
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(f'{MODEL_DIR}/xgb_model.json')

    raw_public = pd.read_csv('data/features_public_window.csv')
    df_public = add_refined_features(add_directional_features(raw_public))
    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_public = df_public['label'].map(label_map).values

    p_lgb = np.mean([m.predict(df_public[feature_cols]) for m in lgb_models], axis=0)
    p_xgb = xgb_clf.predict_proba(df_public[feature_cols])
    p_blend = 0.75 * p_lgb + 0.25 * p_xgb

    priors = load_statsbomb_priors()
    trans_matrix = build_markov_transition_matrix(priors)
    p_cal = calibrate_physics(df_public, p_blend)

    print("="*90)
    print(" TEMPORAL PARAMETER COMPARISON ON BLIND PUBLIC WINDOW (7,500 frames)")
    print("="*90)

    configs = [
        ("Current Rank #1 (45/65/95, tau=0.10)", 0.10, 45, 65, 95),
        ("Winning External Grid (30/45/50, tau=0.08)", 0.08, 30, 45, 50),
        ("Balanced Config A (35/50/60, tau=0.08)", 0.08, 35, 50, 60),
        ("Balanced Config B (35/55/75, tau=0.08)", 0.08, 35, 55, 75),
        ("Balanced Config C (35/55/75, tau=0.10)", 0.10, 35, 55, 75),
        ("Balanced Config D (40/55/80, tau=0.08)", 0.08, 40, 55, 80),
        ("Balanced Config E (30/50/70, tau=0.08)", 0.08, 30, 50, 70),
    ]

    for name, tau, live, dead, turn in configs:
        v_dec = viterbi_decode(p_cal, trans_matrix, trans_scale=tau)
        v_filt = apply_duration_filter(v_dec, min_live_len=live, min_dead_len=dead)
        v_final = apply_sandwich_hysteresis(v_filt, min_turnover_len=turn)
        r = evaluate(y_public, v_final)
        print(f"{name:<42} | Spell F1: {r['spell_f1_mean']:.4f} | Macro F1: {r['macro_f1']:.4f} | Spells: {r['n_pred_spells']:2d} (GT: 17)")

if __name__ == '__main__':
    main()
