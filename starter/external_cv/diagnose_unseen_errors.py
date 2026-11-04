"""
Error Diagnostic on Unseen Matches
Diagnoses misclassifications on SkillCorner Match 3442 and Match 2440
to design physical pass-continuity and duel improvements.
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix

sys.path.insert(0, '.')
from starter.external_cv.benchmark_unseen_matches import load_and_prepare_datasets

def diagnose():
    print("="*90)
    print(" ERROR DIAGNOSTIC: UNSEEN MATCHES (MATCH 3442 & MATCH 2440)")
    print("="*90)
    
    all_data = load_and_prepare_datasets()
    
    train_games = [
        'SkillCorner_Barca_RealMadrid',
        'SkillCorner_ManCity_Liverpool',
        'Metrica_1',
        'Metrica_2'
    ]
    test_games = ['SkillCorner_RealMadrid_Barca', 'SkillCorner_Liverpool_ManCity']
    
    drop_uninformative = ['video_frame', 'frame_id', 'label', 'game',
                          'home_attack_direction', 'ball_missing', 'overall_speed', 'speed_all_roll_2s']
    feats = [c for c in all_data.columns if c not in drop_uninformative]
    label_map = {'H': 0, 'A': 1, 'D': 2}
    inv_map = {0: 'H', 1: 'A', 2: 'D'}
    
    train_df = all_data[all_data['game'].isin(train_games)].copy().reset_index(drop=True)
    clf = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.038, num_leaves=42, colsample_bytree=0.75, random_state=42, n_jobs=-1, verbose=-1)
    clf.fit(train_df[feats], train_df['label'].map(label_map).values)
    
    for g in test_games:
        test_df = all_data[all_data['game'] == g].copy().reset_index(drop=True)
        y_true = test_df['label'].map(label_map).values
        probs = clf.predict_proba(test_df[feats])
        y_pred = np.argmax(probs, axis=1)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        print(f"\n--- Confusion Matrix for {g} ({len(test_df):,} frames) ---")
        print("Rows = True (H, A, D), Cols = Pred (H, A, D):")
        print(cm)
        
        errors = (y_true != y_pred)
        error_df = test_df[errors].copy()
        
        # Analyze error kinematics
        print(f"Total Errors: {errors.sum():,} / {len(test_df):,} ({errors.mean():.2%})")
        print("Error breakdown by True -> Pred:")
        err_types = {}
        for t, p in zip(y_true[errors], y_pred[errors]):
            k = f"{inv_map[t]} -> {inv_map[p]}"
            err_types[k] = err_types.get(k, 0) + 1
        for k, v in sorted(err_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {k}: {v:,} frames ({v/len(test_df):.2%})")
            
        # Check ball speed during Live -> Dead errors
        live_to_dead = (y_true != 2) & (y_pred == 2)
        if live_to_dead.sum() > 0:
            avg_b_spd = test_df.loc[live_to_dead, 'ball_speed'].mean()
            avg_dh = test_df.loc[live_to_dead, 'd_min_home'].mean()
            avg_da = test_df.loc[live_to_dead, 'd_min_away'].mean()
            print(f"  Live -> Dead False Stoppage Profile: Avg Ball Speed = {avg_b_spd:.2f} m/s, Avg d_home = {avg_dh:.2f}m, Avg d_away = {avg_da:.2f}m")

        # Check Dead -> Live False Possession errors
        dead_to_live = (y_true == 2) & (y_pred != 2)
        if dead_to_live.sum() > 0:
            avg_b_spd_d = test_df.loc[dead_to_live, 'ball_speed'].mean()
            print(f"  Dead -> Live False Possession Profile: Avg Ball Speed = {avg_b_spd_d:.2f} m/s")

if __name__ == '__main__':
    diagnose()
