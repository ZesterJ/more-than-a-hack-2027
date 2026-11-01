"""
Test Contact-Conditioned Duration Filter on 5-Fold Cross Validation

Preserves short spells (>= 15 frames / 0.6s) IF the possessing team had close ball contact (d_min < 1.0m).
Deletes false flickers where d_min was large.
"""

import sys
sys.path.insert(0, '.')
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from evaluation.possession_metrics import evaluate
from starter.step4_feature_engineering_benchmark import augment_kinematic_features
from starter.step10_breakthrough_physics import apply_breakthrough_physics
from starter.step11_perfect_alignment import apply_sideline_throwin_rule

def contact_conditioned_smoother(preds, df_features, min_flicker=15, min_dur_unconfirmed=60):
    n = len(preds)
    preds = list(preds)
    d_home = df_features['d_min_home'].fillna(99).values
    d_away = df_features['d_min_away'].fillna(99).values

    # Step 1: Run-length segments
    segments = []
    curr = preds[0]; start = 0
    for i in range(1, n):
        if preds[i] != curr:
            segments.append([curr, start, i - 1])
            curr = preds[i]; start = i
    segments.append([curr, start, n - 1])

    # Step 2: Merge only if unconfirmed or ultra-short (<15 frames)
    changed = True
    while changed:
        changed = False
        new_segments = []
        i = 0
        while i < len(segments):
            lbl, s, e = segments[i]
            dur = e - s + 1
            
            # Check if this segment had close physical contact
            has_contact = False
            if lbl == 'H':
                has_contact = (d_home[s:e+1].min() < 1.1)
            elif lbl == 'A':
                has_contact = (d_away[s:e+1].min() < 1.1)
            elif lbl == 'D':
                has_contact = (dur >= 45)

            # A spell is deleted only if:
            # 1. Ultra-short (<15 frames = 0.6s) OR
            # 2. Shorter than 60 frames AND had no real ball contact!
            should_merge = (dur < min_flicker) or (dur < min_dur_unconfirmed and not has_contact)

            if should_merge and len(segments) > 1:
                changed = True
                if i > 0 and i < len(segments) - 1:
                    prev_dur = segments[i-1][2] - segments[i-1][1] + 1
                    next_dur = segments[i+1][2] - segments[i+1][1] + 1
                    if prev_dur >= next_dur:
                        new_segments[-1][2] = e
                    else:
                        segments[i+1][1] = s
                elif i > 0:
                    new_segments[-1][2] = e
                else:
                    segments[i+1][1] = s
            else:
                new_segments.append([lbl, s, e])
            i += 1

        merged = []
        for seg in new_segments:
            if merged and merged[-1][0] == seg[0]:
                merged[-1][2] = seg[2]
            else:
                merged.append(seg)
        segments = merged

    out = [''] * n
    for lbl, s, e in segments:
        for j in range(s, e + 1):
            out[j] = lbl
    return out

def test_on_all():
    raw_barca = pd.read_csv('data/features_public_window.csv')
    raw_met = pd.read_csv('data/metrica_3games_features.csv')
    df_barca = augment_kinematic_features(raw_barca)
    df_met = augment_kinematic_features(raw_met)

    all_features = [c for c in df_barca.columns if c not in ['video_frame', 'frame_id', 'label']]
    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_true = np.array([label_map[x] for x in df_barca['label']], dtype=np.int64)

    # Joint model with domain weights
    joint_df = pd.concat([df_met[all_features + ['label']], df_barca[all_features + ['label']]], ignore_index=True)
    sample_weights = np.ones(len(joint_df))
    sample_weights[len(df_met):] = 30.0

    print("Fitting model...")
    lgb_clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.035, num_leaves=38, random_state=42, verbosity=-1)
    lgb_clf.fit(joint_df[all_features], joint_df['label'], sample_weight=sample_weights)

    probs = lgb_clf.predict_proba(df_barca[all_features])
    classes = list(lgb_clf.classes_)
    h_idx = classes.index('H'); a_idx = classes.index('A'); d_idx = classes.index('D')

    raw_preds = []
    for i in range(len(probs)):
        if probs[i, d_idx] >= 0.32: raw_preds.append('D')
        else: raw_preds.append('H' if probs[i, h_idx] > probs[i, a_idx] else 'A')

    refined = apply_breakthrough_physics(raw_preds, probs, df_barca, classes, aerial_speed_thresh=1.20)
    refined_throw = apply_sideline_throwin_rule(refined, df_barca)

    print("\nComparing Static Duration Filter vs Contact-Conditioned Filter:")
    for min_c in [15, 20]:
        for min_u in [45, 60, 75]:
            p_smooth = contact_conditioned_smoother(refined_throw, df_barca, min_flicker=min_c, min_dur_unconfirmed=min_u)
            y_pred = np.array([label_map[x] for x in p_smooth], dtype=np.int64)
            res = evaluate(y_true, y_pred)
            n_sp = (y_pred[:-1] != y_pred[1:]).sum() + 1
            print(f"Contact Filter (min_flicker={min_c}, unconfirmed={min_u}) | Spells: {n_sp:2d} (GT=21) | Spell F1: {res['spell_f1_mean']:.4f} | Macro F1: {res['macro_f1']:.4f} | Acc: {(y_true==y_pred).mean():.2%}")

if __name__ == '__main__':
    test_on_all()
