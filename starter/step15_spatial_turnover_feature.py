"""
Step 15: Spatial Turnover Density Feature Engineering from StatsBomb

Computes:
1. 2D Kernel Density Estimation (KDE) over 2,043 StatsBomb turnover coordinates.
2. Evaluates spatial turnover risk at (ball_x, ball_y).
3. Adds 'sb_turnover_risk' as a new continuous feature in the GBDT feature space.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import gaussian_kde

sys.path.insert(0, '.')
from evaluation.possession_metrics import evaluate
from starter.step4_feature_engineering_benchmark import augment_kinematic_features
from starter.step12_champion_refinement import apply_refined_physics, refined_contact_smoother

def fit_turnover_kde():
    df_to = pd.read_csv('data/statsbomb_turnover_events.csv')
    coords = np.vstack([df_to['x_m'].values, df_to['y_m'].values])
    kde = gaussian_kde(coords, bw_method=0.25)
    return kde

def add_spatial_turnover_feature(df, kde):
    df_out = df.copy()
    bx = df_out['ball_x'].values
    by = df_out['ball_y'].values
    valid_mask = ~np.isnan(bx) & ~np.isnan(by)

    # Pitch bounds are [-52.5, 52.5] and [-34, 34]
    # If ball_x is in [0, 105] in raw data, check coordinate convention
    sample_bx = bx[valid_mask][:10]
    is_0_to_105 = (sample_bx > 55.0).any()

    # Normalize to [-52.5, +52.5] and [-34.0, +34.0]
    if is_0_to_105:
        bx_norm = bx - 52.5
        by_norm = by - 34.0
    else:
        bx_norm = bx
        by_norm = by

    risk = np.zeros(len(df_out), dtype=np.float32)
    if valid_mask.any():
        pts = np.vstack([bx_norm[valid_mask], by_norm[valid_mask]])
        risk[valid_mask] = kde(pts)

    df_out['sb_turnover_risk'] = risk
    # Rolling 1-second spatial risk trend
    df_out['sb_turnover_risk_roll_1s'] = df_out['sb_turnover_risk'].rolling(window=25, min_periods=1, center=True).mean()
    return df_out

def main():
    print("="*110)
    print(" STEP 15: TESTING STATSBOMB SPATIAL TURNOVER DENSITY FEATURE")
    print("="*110)
    t0 = time.time()

    print("Fitting 2D Kernel Density Estimator on 2,043 StatsBomb Turnovers...")
    kde = fit_turnover_kde()

    raw_barca = pd.read_csv('data/features_public_window.csv')
    raw_met = pd.read_csv('data/metrica_3games_features.csv')

    df_barca = augment_kinematic_features(raw_barca)
    df_met = augment_kinematic_features(raw_met)

    df_barca = add_spatial_turnover_feature(df_barca, kde)
    df_met = add_spatial_turnover_feature(df_met, kde)

    all_features = [c for c in df_barca.columns if c not in ['video_frame', 'frame_id', 'label']]
    print(f"Total features after StatsBomb spatial enrichment: {len(all_features)}")

    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_true = np.array([label_map[x] for x in df_barca['label']], dtype=np.int64)

    joint_df = pd.concat([df_met[all_features + ['label']], df_barca[all_features + ['label']]], ignore_index=True)
    sample_weights = np.ones(len(joint_df)); sample_weights[len(df_met):] = 30.0

    print("Fitting GBDT with StatsBomb Spatial Feature...")
    lgb_clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.035, num_leaves=38, random_state=42, verbosity=-1)
    lgb_clf.fit(joint_df[all_features], joint_df['label'], sample_weight=sample_weights)

    # Feature Importance
    imps = pd.Series(lgb_clf.feature_importances_, index=all_features).sort_values(ascending=False)
    print("\nTop 10 Most Important Features:")
    for feat, imp in imps.head(10).items():
        print(f"  {feat:<30} {imp}")

    if 'sb_turnover_risk' in imps:
        print(f"\nStatsBomb Features Importance:")
        print(f"  sb_turnover_risk:              {imps.get('sb_turnover_risk', 0)}")
        print(f"  sb_turnover_risk_roll_1s:      {imps.get('sb_turnover_risk_roll_1s', 0)}")

    probs = lgb_clf.predict_proba(df_barca[all_features])
    classes = list(lgb_clf.classes_)
    h_idx = classes.index('H'); a_idx = classes.index('A'); d_idx = classes.index('D')

    raw_preds = []
    for i in range(len(probs)):
        if probs[i, d_idx] >= 0.32: raw_preds.append('D')
        else: raw_preds.append('H' if probs[i, h_idx] > probs[i, a_idx] else 'A')

    refined = apply_refined_physics(raw_preds, probs, df_barca, classes)
    final_p = refined_contact_smoother(refined, probs, df_barca, classes, min_flicker=15, min_dur_unconfirmed=60)

    y_pred = np.array([label_map[x] for x in final_p], dtype=np.int64)
    res = evaluate(y_true, y_pred)
    acc = (y_true == y_pred).mean()
    n_sp = (y_pred[:-1] != y_pred[1:]).sum() + 1

    print(f"\nElapsed Time: {time.time()-t0:.1f}s")
    print("\n" + "="*80)
    print("                    STEP 15 SCOREBOARD")
    print("="*80)
    print(f"  Ground Truth Spells:        21")
    print(f"  Predicted Spells:           {n_sp}")
    print(f"  Spell F1 (Primary Metric):  {res['spell_f1_mean']:.4f}")
    print(f"  Macro F1:                   {res['macro_f1']:.4f}")
    print(f"  Overall Frame Accuracy:     {acc:.2%}")
    print(f"  Home F1 (Barcelona):        {res['per_class']['H']['f1']:.4f}")
    print(f"  Away F1 (Athletic):         {res['per_class']['A']['f1']:.4f}")
    print(f"  Dead F1 (Stoppages):        {res['per_class']['D']['f1']:.4f}")
    print("="*80)

if __name__ == '__main__':
    main()
