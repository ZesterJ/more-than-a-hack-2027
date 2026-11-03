"""
Step 3: Two-Stage Transfer Learning Benchmark

Evaluates transfer learning strategies to bridge external 643k data with the 5-minute Barcelona ground truth:
1. Baseline: Barca Train Only (5,250 frames)
2. External Only: Zero-shot Metrica (90,000 frames)
3. Naive Joint Training: External + Barca Train (Equal weights)
4. Domain-Adapted Sample Weighting: High sample weight on Barca Train (10x, 25x, 50x)
5. Continual Booster Adaptation: LightGBM init_model warm-start transfer
All evaluated strictly on the 30% unseen held-out test slice (2,250 frames).
"""

import sys
sys.path.insert(0, '.')
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from evaluation.possession_metrics import evaluate
from starter.step2_temporal_dynamic_programming import merge_short_spells

def score(y_true_str, y_pred_str, name="Model", min_dur=75):
    # Apply Step 2 optimal post-processor
    y_pred_post = merge_short_spells(y_pred_str, min_duration=min_dur)
    
    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_true = np.array([label_map[x] for x in y_true_str], dtype=np.int64)
    y_pred = np.array([label_map[x] for x in y_pred_post], dtype=np.int64)
    res = evaluate(y_true, y_pred)
    acc = (y_true == y_pred).mean()
    n_sp = (y_pred[:-1] != y_pred[1:]).sum() + 1
    print(f"| {name:40s} | {n_sp:6d} | {res['spell_f1_mean']:.4f} | {res['macro_f1']:.4f} | {acc:.2%} | {res['per_class']['H']['f1']:.4f} | {res['per_class']['A']['f1']:.4f} | {res['per_class']['D']['f1']:.4f} |")
    return res

def main():
    print("="*105)
    print(" STEP 3: TWO-STAGE TRANSFER LEARNING ON UNSEEN HELD-OUT TEST SLICE")
    print("="*105)

    # 1. Load Data
    df_barca = pd.read_csv('data/features_public_window.csv')
    df_met = pd.read_csv('data/metrica_3games_features.csv')
    feature_cols = [c for c in df_barca.columns if c not in ['video_frame', 'frame_id', 'label']]

    # Chronological Split of Barca: 70% Train (5,250 frames) / 30% Test (2,250 frames)
    split_idx = int(len(df_barca) * 0.70)
    train_barca = df_barca.iloc[:split_idx].copy()
    test_barca = df_barca.iloc[split_idx:].copy()

    print(f"| {'Transfer Strategy':40s} | Spells | SpellF1| MacroF1| Acc    | F1 (H) | F1 (A) | F1 (D) |")
    print("|" + "-"*42 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|")

    # 1. Baseline: Barca Train Only
    clf_base = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)
    clf_base.fit(train_barca[feature_cols], train_barca['label'])
    score(test_barca['label'].values, clf_base.predict(test_barca[feature_cols]), "1. Baseline (Barca Train Only 5.2k)")

    # 2. Pure External: Metrica Only (Zero Barca)
    clf_ext = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)
    clf_ext.fit(df_met[feature_cols], df_met['label'])
    score(test_barca['label'].values, clf_ext.predict(test_barca[feature_cols]), "2. Pure External Zero-Shot (Metrica 90k)")

    # 3. Naive Joint Concatenation (Equal Weights)
    joint_df = pd.concat([df_met[feature_cols + ['label']], train_barca[feature_cols + ['label']]], ignore_index=True)
    clf_joint = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)
    clf_joint.fit(joint_df[feature_cols], joint_df['label'])
    score(test_barca['label'].values, clf_joint.predict(test_barca[feature_cols]), "3. Naive Joint (Metrica + Barca Equal)")

    # 4. Domain-Adapted Sample Weighting
    for w in [5, 15, 30, 60]:
        sample_weights = np.ones(len(joint_df))
        sample_weights[len(df_met):] = w  # Boost Barca ground truth weight
        clf_weighted = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=35, random_state=42, verbosity=-1)
        clf_weighted.fit(joint_df[feature_cols], joint_df['label'], sample_weight=sample_weights)
        score(test_barca['label'].values, clf_weighted.predict(test_barca[feature_cols]), f"4. Weighted Domain Transfer ({w}x Barca Weight)")

    # 5. Two-Stage Warm Start Continual Transfer
    # Stage 1: Fit 100 trees on External
    booster_stage1 = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)
    booster_stage1.fit(df_met[feature_cols], df_met['label'])
    
    # Stage 2: Continue fitting 100 additional trees strictly on Barca Train with lower LR
    booster_stage2 = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.02, num_leaves=31, random_state=42, verbosity=-1)
    booster_stage2.fit(train_barca[feature_cols], train_barca['label'], init_model=booster_stage1.booster_)
    score(test_barca['label'].values, booster_stage2.predict(test_barca[feature_cols]), "5. Continual Warm-Start Transfer")

if __name__ == '__main__':
    main()
