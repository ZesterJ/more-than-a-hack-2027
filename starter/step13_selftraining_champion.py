"""
Step 13: Self-Training (Pseudo-Labeling) & Physical Ensemble Pipeline

1. Initial Teacher Model: LightGBM on Barcelona Ground Truth (30x) + Metrica (1x).
2. Pseudo-Label Mining: Mine ultra-high confidence in-domain match frames from Barcelona vs Athletic.
3. Student Model: Retrained on Ground Truth (30x) + In-Domain Pseudo-Labels (3x) + Metrica (1x).
4. Physical State Engine: Step 12 Refinements (Aerial cap, sharp throw-in release, high-confidence turnover preservation).
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, '.')
import games
from evaluation.possession_metrics import evaluate
from starter.extract_features import extract_features
from starter.step4_feature_engineering_benchmark import augment_kinematic_features
from starter.step12_champion_refinement import apply_refined_physics, refined_contact_smoother
from starter.generate_full_submission import extract_features_df

def main():
    print("="*110)
    print(" STEP 13: SELF-TRAINING (PSEUDO-LABELING) & PHYSICAL ENSEMBLE PIPELINE")
    print("="*110)
    t0 = time.time()

    # 1. Load data
    raw_barca = pd.read_csv('data/features_public_window.csv')
    raw_met = pd.read_csv('data/metrica_3games_features.csv')
    df_barca = augment_kinematic_features(raw_barca)
    df_met = augment_kinematic_features(raw_met)

    all_features = [c for c in df_barca.columns if c not in ['video_frame', 'frame_id', 'label']]
    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_true = np.array([label_map[x] for x in df_barca['label']], dtype=np.int64)

    # 2. Train Initial Teacher Model
    print("\n[Phase 1] Training Teacher Model on GT (30x) + Metrica (1x)...")
    joint_df = pd.concat([df_met[all_features + ['label']], df_barca[all_features + ['label']]], ignore_index=True)
    sample_weights = np.ones(len(joint_df))
    sample_weights[len(df_met):] = 30.0

    teacher = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.035, num_leaves=38, random_state=42, verbosity=-1)
    teacher.fit(joint_df[all_features], joint_df['label'], sample_weight=sample_weights)
    classes = list(teacher.classes_)
    h_idx = classes.index('H'); a_idx = classes.index('A'); d_idx = classes.index('D')

    # 3. Mine Pseudo-Labels from the rest of the 1st Half (excluding the 7.5k GT window)
    print("\n[Phase 2] Mining In-Domain Pseudo-Labels from Barcelona vs Athletic Match Play...")
    paths = games.asset_paths(source="tactical")
    raw_atd = paths["atd_raw"]

    # Extract match play outside the 11185..18684 GT window (e.g. 18685 to 40000 = 21,316 frames)
    df_unlabeled = extract_features_df(raw_atd, 18685, 38684)
    probs_unlabeled = teacher.predict_proba(df_unlabeled[all_features])

    pseudo_records = []
    d_home_un = df_unlabeled['d_min_home'].fillna(99).values
    d_away_un = df_unlabeled['d_min_away'].fillna(99).values
    spd_un = df_unlabeled['speed_all_roll_2s'].values

    for i in range(len(df_unlabeled)):
        ph, pa, pd_val = probs_unlabeled[i, h_idx], probs_unlabeled[i, a_idx], probs_unlabeled[i, d_idx]
        lbl = None
        if ph > 0.95 and d_home_un[i] < 1.2:
            lbl = 'H'
        elif pa > 0.95 and d_away_un[i] < 1.2:
            lbl = 'A'
        elif pd_val > 0.90 and spd_un[i] < 1.10:
            lbl = 'D'

        if lbl is not None:
            row_dict = df_unlabeled.iloc[i][all_features].to_dict()
            row_dict['label'] = lbl
            pseudo_records.append(row_dict)

    df_pseudo = pd.DataFrame(pseudo_records)
    print(f"  Mined {len(df_pseudo):,} ultra-confident in-domain pseudo-labels!")
    print(f"  Pseudo-label distribution: {df_pseudo['label'].value_counts().to_dict()}")

    # 4. Train Student Model on GT + Pseudo-Labels + Metrica
    print("\n[Phase 3] Training Student Model with Self-Training In-Domain Regularization...")
    student_df = pd.concat([df_met[all_features + ['label']],
                            df_pseudo[all_features + ['label']],
                            df_barca[all_features + ['label']]], ignore_index=True)

    student_weights = np.ones(len(student_df))
    # Metrica: 1x
    # Pseudo-labels: 3x
    student_weights[len(df_met) : len(df_met) + len(df_pseudo)] = 3.0
    # True Barcelona Ground Truth: 30x
    student_weights[len(df_met) + len(df_pseudo) :] = 30.0

    student = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.035, num_leaves=38, random_state=42, verbosity=-1)
    student.fit(student_df[all_features], student_df['label'], sample_weight=student_weights)

    # 5. Evaluate Student Model on GT Window
    print("\n[Phase 4] Evaluating Student Model on Barcelona Public Ground Truth...")
    probs_barca = student.predict_proba(df_barca[all_features])
    classes = list(student.classes_)
    h_idx = classes.index('H'); a_idx = classes.index('A'); d_idx = classes.index('D')

    raw_preds = []
    for i in range(len(probs_barca)):
        if probs_barca[i, d_idx] >= 0.32: raw_preds.append('D')
        else: raw_preds.append('H' if probs_barca[i, h_idx] > probs_barca[i, a_idx] else 'A')

    refined = apply_refined_physics(raw_preds, probs_barca, df_barca, classes)
    final_p = refined_contact_smoother(refined, probs_barca, df_barca, classes, min_flicker=15, min_dur_unconfirmed=60)

    y_pred = np.array([label_map[x] for x in final_p], dtype=np.int64)
    res = evaluate(y_true, y_pred)
    acc = (y_true == y_pred).mean()
    n_sp = (y_pred[:-1] != y_pred[1:]).sum() + 1

    print(f"\nElapsed Time: {time.time()-t0:.1f}s")
    print("\n" + "="*80)
    print("                    STEP 13 SCOREBOARD")
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
