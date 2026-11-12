"""
Train Combined Production Stage 1 Dead-Ball Detector Ensemble
Strict Compliance with Nov 3 Rule Mandate (100% External Data)

Trains a 5-seed diversity LightGBM ensemble on 482,359 external frames across 11 unique matches:
- SkillCorner 4 European Matches (272,359 frames; real broadcast ~52% ball gaps)
- IDSSE 7 Bundesliga Matches Clean (105,000 frames; TRACAB optical tracking)
- IDSSE 7 Bundesliga Matches 41% Ball-Gap Masked (105,000 frames; broadcast dropout augmentation)

(Metrica 3 games excluded from training because Metrica only tags 2.3%-8.5% dead balls
and leaves throw-ins as Live, which suppresses P(Dead) recall.)

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
from starter.stage1_dead_detector.combined_live_dead_benchmark import simulate_ball_gaps

MODEL_DIR = 'models/stage1_dead_detector'


def main():
    print("=" * 90)
    print(" TRAINING COMBINED PRODUCTION STAGE 1 DEAD-BALL DETECTOR ENSEMBLE")
    print(" 100% External Data (Nov 3 Rule Compliance) | 482,359 Training Frames (V3_LeadLag_34f)")
    print("=" * 90)
    t0 = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load All 11 External Matches (SkillCorner + IDSSE; Metrica excluded due to incomplete dead labels)
    print("\n[1/3] Loading SkillCorner (4) and IDSSE (7) external matches...")
    sc_path = 'data/skillcorner_4matches_features.csv'
    idsse_path = 'data/idsse_7matches_features.csv'
    assert os.path.exists(sc_path), f"Missing {sc_path}"
    assert os.path.exists(idsse_path), f"Missing {idsse_path}"

    df_sc = pd.read_csv(sc_path)
    df_idsse = pd.read_csv(idsse_path)

    common_cols = [c for c in df_sc.columns if c in df_idsse.columns]
    df_sc = df_sc[common_cols].copy()
    df_idsse = df_idsse[common_cols].copy()

    # Create 41% ball-gap masked copy of IDSSE so the model learns missing-ball resilience
    df_idsse_masked = simulate_ball_gaps(df_idsse, target_missing_rate=0.41, seed=42)

    # 2. Extract Stage 1 Features per match (preserving rolling boundaries)
    print("[2/3] Extracting V3_LeadLag_34f Physical Pillars of Dead-Ball Features per match...")
    feat_frames = []
    for dframe in [df_sc, df_idsse, df_idsse_masked]:
        for _, gdf in dframe.groupby('game', sort=False):
            gf = add_dead_physics_features(gdf)
            gf['is_dead'] = (gf['label'] == 'D').astype(int)
            feat_frames.append(gf)

    all_train = pd.concat(feat_frames, ignore_index=True)
    X = all_train[DEAD_FEATURE_SCHEMA]
    y = all_train['is_dead'].values

    matches_used = sorted(all_train['game'].unique().tolist())
    print(f"  Training Corpus:    {len(X):,} frames across {len(matches_used)} unique matches (+ gap-masked IDSSE)")
    print(f"  Features in Schema: {len(DEAD_FEATURE_SCHEMA)}")
    print(f"  Dead Frames: {y.sum():,} ({100*y.mean():.1f}%) | Live Frames: {(1-y).sum():,} ({100*(1-y).mean():.1f}%)")

    # Save metadata
    meta = {
        'training_frames': int(len(X)),
        'matches_used': matches_used,
        'datasets': ['SkillCorner_4matches', 'IDSSE_7matches_clean', 'IDSSE_7matches_masked41'],
        'features': DEAD_FEATURE_SCHEMA,
        'default_decoder': {
            'tau': 0.38,
            'smooth_win': 25,
            'hyst': [0.25, 0.45],
            'min_dead_frames': 50,
            'min_live_frames': 25,
        },
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'rule_compliance': 'Nov 3, 2026 Mandate (100% External Data, 0 Competition Match Frames)'
    }
    with open(os.path.join(MODEL_DIR, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # 3. Train 5-Seed Diversity LightGBM Boosters
    print("\n[3/3] Training 5-Seed Diversity LightGBM Boosters...")
    seeds = [42, 101, 777, 1234, 2026]
    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    for i, s in enumerate(seeds):
        t_m = time.time()
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.035 + (i % 3) * 0.003,
            'num_leaves': 35 + (i % 2) * 5,
            'subsample': 0.85,
            'subsample_freq': 1,
            'colsample_bytree': 0.80,
            'seed': s,
            'n_jobs': -1,
            'verbose': -1,
        }
        booster = lgb.train(params, dtrain, num_boost_round=320 + i * 10)
        m_path = os.path.join(MODEL_DIR, f'lgb_stage1_dead_seed_{s}.txt')
        booster.save_model(m_path)
        print(f"  Booster [{i+1}/5] (seed={s}) trained & saved in {time.time()-t_m:.1f}s")

    print("\n" + "=" * 90)
    print(f" COMBINED STAGE 1 ENSEMBLE TRAINING COMPLETE IN {time.time()-t0:.1f}s")
    print(f" Models saved to: {MODEL_DIR}")
    print("=" * 90)


if __name__ == '__main__':
    main()
