"""
Train Production Stage 1 Dead-Ball Detector Ensemble
Strict Compliance with Nov 3 Rule Mandate (100% External Data)

Trains a 5-seed diversity LightGBM ensemble on 362,359 external frames
across 7 European matches (SkillCorner 4 matches + Metrica 3 games).
Saves models to `models/stage1_dead_detector/`.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, '.')
from starter.stage1_dead_detector.dead_detector import (
    DEAD_FEATURE_SCHEMA, add_dead_physics_features
)

MODEL_DIR = 'models/stage1_dead_detector'

def main():
    print("="*90)
    print(" TRAINING PRODUCTION STAGE 1 DEAD-BALL DETECTOR ENSEMBLE")
    print(" 100% External Data (Nov 3 Rule Compliance) | 362,359 Training Frames")
    print("="*90)
    t0 = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load All 7 External Matches
    print("\n[1/3] Loading all 7 external European tracking matches...")
    sc_path = 'data/skillcorner_4matches_features.csv'
    met_path = 'data/metrica_3games_features.csv'
    assert os.path.exists(sc_path), f"Missing {sc_path}"
    assert os.path.exists(met_path), f"Missing {met_path}"

    df_sc = pd.read_csv(sc_path)
    df_met = pd.read_csv(met_path)
    common_cols = [c for c in df_sc.columns if c in df_met.columns]
    all_train = pd.concat([df_sc[common_cols], df_met[common_cols]], ignore_index=True)

    # 2. Extract Stage 1 Features
    print("[2/3] Extracting 5 Physical Pillars of Dead-Ball Features...")
    all_train = add_dead_physics_features(all_train)
    all_train['is_dead'] = (all_train['label'] == 'D').astype(int)

    X = all_train[DEAD_FEATURE_SCHEMA]
    y = all_train['is_dead'].values

    print(f"  Training Corpus: {len(X):,} frames across 7 matches")
    print(f"  Features in Schema: {len(DEAD_FEATURE_SCHEMA)}")
    print(f"  Dead Frames: {y.sum():,} ({100*y.mean():.1f}%) | Live Frames: {(1-y).sum():,} ({100*(1-y).mean():.1f}%)")

    # Save metadata
    meta = {
        'training_frames': len(X),
        'matches_used': all_train['game'].unique().tolist(),
        'features': DEAD_FEATURE_SCHEMA,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'rule_compliance': 'Nov 3, 2026 Mandate (100% External Data, 0 Competition Match Frames)'
    }
    with open(os.path.join(MODEL_DIR, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # 3. Train 5-Seed Diversity LightGBM Boosters
    print("\n[3/3] Training 5-Seed Diversity LightGBM Boosters...")
    seeds = [42, 101, 777, 1234, 2026]
    for i, s in enumerate(seeds):
        t_m = time.time()
        clf = lgb.LGBMClassifier(
            n_estimators=320 + i * 10,
            learning_rate=0.035 + (i % 3) * 0.003,
            num_leaves=35 + (i % 2) * 5,
            subsample=0.85,
            colsample_bytree=0.80,
            random_state=s,
            n_jobs=-1,
            verbose=-1
        )
        clf.fit(X, y)
        m_path = os.path.join(MODEL_DIR, f'lgb_stage1_dead_seed_{s}.txt')
        clf.booster_.save_model(m_path)
        print(f"  Booster [{i+1}/5] (seed={s}) trained & saved in {time.time()-t_m:.1f}s")

    print("\n" + "="*90)
    print(f" STAGE 1 ENSEMBLE TRAINING COMPLETE IN {time.time()-t0:.1f}s")
    print(f" Models saved to: {MODEL_DIR}")
    print("="*90)

if __name__ == '__main__':
    main()
