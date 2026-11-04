"""
Experiment: Direction-Invariant & Pitch-Symmetric Features
Tests whether eliminating pitch-end bias (goal-direction dependency)
eliminates team-confusion errors (H <-> A) on unseen external matches.
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, '.')
from starter.external_cv.benchmark_unseen_matches import load_and_prepare_datasets, evaluate_on_unseen_matches

def main():
    print("="*90)
    print(" EXPERIMENT: PITCH-SYMMETRIC & ATTACK-DIRECTION INVARIANT FEATURES")
    print("="*90)

    all_data = load_and_prepare_datasets()

    train_games = [
        'SkillCorner_Barca_RealMadrid',
        'SkillCorner_ManCity_Liverpool',
        'Metrica_1',
        'Metrica_2'
    ]
    unseen_test_games = [
        'SkillCorner_RealMadrid_Barca',
        'SkillCorner_Liverpool_ManCity',
        'Metrica_3'
    ]

    # Current Upgraded Schema (contains asymmetric x features: centroid_diff_x, dist_to_home_goal, dist_to_away_goal, ball_vx)
    drop_uninformative = ['video_frame', 'frame_id', 'label', 'game',
                          'home_attack_direction', 'ball_missing', 'overall_speed', 'speed_all_roll_2s']
    current_schema = [c for c in all_data.columns if c not in drop_uninformative]

    # Direction-Invariant Schema: Remove goal-dependent asymmetric features
    asymmetric_features = [
        'home_centroid_x', 'away_centroid_x', 'centroid_diff_x',
        'dist_to_home_goal', 'dist_to_away_goal', 'ball_vx'
    ]
    symmetric_schema = [c for c in current_schema if c not in asymmetric_features]

    print(f"\nCurrent Schema: {len(current_schema)} features (includes asymmetric x-coordinates)")
    print(f"Symmetric Schema: {len(symmetric_schema)} features (removed {asymmetric_features})")

    # Evaluate Current Schema
    print("\n--- Evaluating Current Asymmetric Schema on Unseen Matches ---")
    res_curr = evaluate_on_unseen_matches(
        model_type='blend',
        feature_schema=current_schema,
        train_games=train_games,
        test_games=unseen_test_games,
        all_data=all_data,
        colsample=0.75,
        apply_hysteresis=True,
        min_turnover_len=95
    )
    print(f"  Current Overall Accuracy: {res_curr['overall_accuracy']:.2%}")
    print(f"  Current Overall Macro F1: {res_curr['overall_macro_f1']:.4f}")
    for g, s in res_curr["per_match"].items():
        print(f"    [{g.replace('SkillCorner_', '')}]: Acc={s['Accuracy']:.2%}, Macro F1={s['Macro F1']:.4f}")

    # Evaluate Symmetric Schema
    print("\n--- Evaluating Pitch-Symmetric Schema on Unseen Matches ---")
    res_sym = evaluate_on_unseen_matches(
        model_type='blend',
        feature_schema=symmetric_schema,
        train_games=train_games,
        test_games=unseen_test_games,
        all_data=all_data,
        colsample=0.75,
        apply_hysteresis=True,
        min_turnover_len=95
    )
    print(f"  Symmetric Overall Accuracy: {res_sym['overall_accuracy']:.2%}")
    print(f"  Symmetric Overall Macro F1: {res_sym['overall_macro_f1']:.4f}")
    for g, s in res_sym["per_match"].items():
        print(f"    [{g.replace('SkillCorner_', '')}]: Acc={s['Accuracy']:.2%}, Macro F1={s['Macro F1']:.4f}")

    # Comparison
    print("\n" + "="*90)
    print(" COMPARISON SUMMARY (165,643 UNSEEN TEST FRAMES):")
    print(f"  Accuracy:  {res_curr['overall_accuracy']:.2%}  -->  {res_sym['overall_accuracy']:.2%} ({res_sym['overall_accuracy'] - res_curr['overall_accuracy']:+.2%})")
    print(f"  Macro F1:  {res_curr['overall_macro_f1']:.4f}  -->  {res_sym['overall_macro_f1']:.4f} ({res_sym['overall_macro_f1'] - res_curr['overall_macro_f1']:+.4f})")
    print("="*90)

if __name__ == '__main__':
    main()
