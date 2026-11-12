"""
Production Stage 1 Dead-Ball Detector
Author: Machine Learning & Analytics Team
Strict Nov 3 Rule Compliance (100% External Training)

Implements the 5 Physical Pillars of Dead-Ball Detection:
1. Pitch Boundary Violations (105m x 68m limits, dist_to_boundary, is_out_of_bounds)
2. Collective Pitch-Wide Deceleration (Speed collapses below 1.25 m/s during whistle stoppages)
3. Tracking Occlusion Persistence (consecutive_missing, missing_over_2s, missing_ratio_roll_5s)
4. Tactical Team Dispersion Expansion (spread_area_roll_2s, home_spread, away_spread)
5. Player-to-Ball Physical Isolation (min_dist_any, min_dist_roll_3s, ball_isolated_5m)
+ Smoothed Dual-Threshold Hysteresis & Football Stoppage Duration Filtering (50f/25f)
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

DEAD_FEATURE_SCHEMA = [
    # 1. Pitch Boundary Physics
    'dist_to_boundary', 'is_out_of_bounds', 'near_boundary_1m', 'boundary_roll_1s',
    # 2. Whole Pitch Collective Kinetics (Whistle Signature)
    'speed_all_roll_2s', 'pitch_decel_2s', 'is_collective_walking', 'walking_roll_2s',
    # 3. Missing Ball Persistence
    'ball_missing', 'consecutive_missing', 'missing_over_1s', 'missing_over_3s',
    'missing_ratio_roll_2s', 'missing_ratio_roll_5s',
    # 4. Tactical Dispersion Expansion
    'home_spread_x', 'home_spread_y', 'away_spread_x', 'away_spread_y',
    'total_spread_area', 'spread_area_roll_2s',
    # 5. Nearest Player Distance & Player Isolation
    'min_dist_any', 'min_dist_roll_1s', 'min_dist_roll_3s', 'min_dist_roll_5s',
    'ball_isolated_5m', 'ball_isolated_10m',
    'players_within_3m_total', 'players_3m_roll_1s',
    # 6. Active Players (Camera Coverage)
    'active_total', 'active_roll_2s',
    # 7. Directional Lead/Lag (Future 2s minus Past 2s for Stoppage/Restart Edges)
    'spread_delta_fwd_2s', 'dist_delta_fwd_2s', 'missing_delta_fwd_2s', 'players3m_delta_fwd_2s'
]


def add_dead_physics_features(df):
    """Compute the 5 physical pillars of Stage 1 Dead-Ball detection."""
    df = df.copy()

    # Harmonize missing ball representations (replace 999.0 with np.nan)
    if 'd_min_home' in df.columns:
        df['d_min_home'] = df['d_min_home'].replace(999.0, np.nan)
    if 'd_min_away' in df.columns:
        df['d_min_away'] = df['d_min_away'].replace(999.0, np.nan)

    # 1. Pitch Boundary Distance & Out-of-bounds (Pitch 105m x 68m)
    bx = df['ball_x'].fillna(52.5)
    by = df['ball_y'].fillna(34.0)
    dist_left = bx
    dist_right = 105.0 - bx
    dist_bottom = by
    dist_top = 68.0 - by

    df['dist_to_boundary'] = np.minimum(
        np.minimum(dist_left, dist_right),
        np.minimum(dist_bottom, dist_top)
    )
    df['is_out_of_bounds'] = (df['dist_to_boundary'] < 0.0).astype(float)
    df['near_boundary_1m'] = (df['dist_to_boundary'] < 1.0).astype(float)
    df['boundary_roll_1s'] = df['is_out_of_bounds'].rolling(25, min_periods=1, center=True).mean()

    # 2. Whole-pitch Collective Deceleration (Whistle Signature)
    if 'speed_all_roll_2s' not in df.columns or df['speed_all_roll_2s'].isna().all():
        df['speed_all_roll_2s'] = df.get('overall_speed', pd.Series(1.5, index=df.index))
    df['speed_all_roll_2s'] = df['speed_all_roll_2s'].fillna(1.5)

    pitch_speed = df['speed_all_roll_2s'].values
    df['pitch_decel_2s'] = pd.Series(pitch_speed).diff(50).fillna(0).values
    df['is_collective_walking'] = (df['speed_all_roll_2s'] < 1.25).astype(float)
    df['walking_roll_2s'] = df['is_collective_walking'].rolling(50, min_periods=1, center=True).mean()

    # Mask constant dummy speed (e.g. IDSSE 1.40 constant) so trees don't split on fake constants
    if 'overall_speed' in df.columns and df['overall_speed'].std() < 0.05:
        for col in ['speed_all_roll_2s', 'pitch_decel_2s', 'is_collective_walking', 'walking_roll_2s']:
            df[col] = np.nan

    # 3. Missing Ball Persistence
    missing = df['ball_missing'].fillna(0).values
    n = len(missing)
    consec = np.zeros(n, dtype=float)
    c = 0
    for i in range(n):
        if missing[i] == 1:
            c += 1
        else:
            c = 0
        consec[i] = c
    df['consecutive_missing'] = consec
    df['missing_over_1s'] = (consec >= 25).astype(float)
    df['missing_over_3s'] = (consec >= 75).astype(float)
    if 'missing_ratio_roll_2s' not in df.columns:
        df['missing_ratio_roll_2s'] = df['ball_missing'].rolling(50, min_periods=1, center=True).mean()
    df['missing_ratio_roll_5s'] = df['ball_missing'].rolling(125, min_periods=1, center=True).mean()

    # 4. Tactical Dispersion Expansion
    hx = df['home_spread_x'].fillna(15.0)
    hy = df['home_spread_y'].fillna(15.0)
    ax = df['away_spread_x'].fillna(15.0)
    ay = df['away_spread_y'].fillna(15.0)
    df['total_spread_area'] = (hx * hy) + (ax * ay)
    df['spread_area_roll_2s'] = df['total_spread_area'].rolling(50, min_periods=1, center=True).mean()

    # 5. Nearest Player Distance & Player Isolation
    d_h = df['d_min_home'].fillna(50.0)
    d_a = df['d_min_away'].fillna(50.0)
    df['min_dist_any'] = np.minimum(d_h, d_a)
    df['min_dist_roll_1s'] = df['min_dist_any'].rolling(25, min_periods=1, center=True).mean()
    df['min_dist_roll_3s'] = df['min_dist_any'].rolling(75, min_periods=1, center=True).mean()
    df['min_dist_roll_5s'] = df['min_dist_any'].rolling(125, min_periods=1, center=True).mean()
    df['ball_isolated_5m'] = (df['min_dist_any'] > 5.0).astype(float)
    df['ball_isolated_10m'] = (df['min_dist_any'] > 10.0).astype(float)

    h_3m = df['home_within_3m'].fillna(0)
    a_3m = df['away_within_3m'].fillna(0)
    df['players_within_3m_total'] = h_3m + a_3m
    df['players_3m_roll_1s'] = df['players_within_3m_total'].rolling(25, min_periods=1, center=True).mean()

    # 6. Active Players (Broadcast View Occlusion)
    h_act = df['home_active'].fillna(10)
    a_act = df['away_active'].fillna(10)
    df['active_total'] = h_act + a_act
    df['active_roll_2s'] = df['active_total'].rolling(50, min_periods=1, center=True).mean()

    # 7. Directional Lead/Lag (Future 2s minus Past 2s for Stoppage/Restart Edges)
    def _lead_minus_lag(series, win=50):
        lead = series.iloc[::-1].rolling(win, min_periods=1).mean().iloc[::-1]
        lag = series.rolling(win, min_periods=1).mean()
        return (lead - lag).fillna(0.0)

    df['spread_delta_fwd_2s'] = _lead_minus_lag(df['total_spread_area'], 50)
    df['dist_delta_fwd_2s'] = _lead_minus_lag(df['min_dist_any'], 50)
    df['missing_delta_fwd_2s'] = _lead_minus_lag(df['ball_missing'].fillna(0), 50)
    df['players3m_delta_fwd_2s'] = _lead_minus_lag(df['players_within_3m_total'], 50)

    return df


def apply_stoppage_duration_filter(is_dead_preds, min_dead_frames=50, min_live_frames=25):
    """
    Apply football domain physics:
    1. Dead-ball stoppages CANNOT last < 2.0s (50 frames). If isolated, revert to Live.
    2. Live touches inside a dead-ball period cannot last < 1.0s (25 frames).
    """
    n = len(is_dead_preds)
    out = np.array(is_dead_preds, dtype=int, copy=True)

    # Pass 1: Remove short dead blips (< min_dead_frames)
    i = 0
    while i < n:
        if out[i] == 1:
            start = i
            while i < n and out[i] == 1:
                i += 1
            length = i - start
            if length < min_dead_frames:
                out[start:i] = 0
        else:
            i += 1

    # Pass 2: Fill short live blips (< min_live_frames) surrounded by Dead
    i = 0
    while i < n:
        if out[i] == 0:
            start = i
            while i < n and out[i] == 0:
                i += 1
            length = i - start
            if length < min_live_frames:
                prev_dead = (start > 0 and out[start - 1] == 1)
                next_dead = (i < n and out[i] == 1)
                if prev_dead and next_dead:
                    out[start:i] = 1
        else:
            i += 1

    return out


class Stage1DeadDetector:
    """Production Stage 1 Dead-Ball Gate for Hierarchical Possession Models."""

    def __init__(self, model_dir='models/stage1_dead_detector'):
        self.model_dir = model_dir
        self.models = []
        self.features = DEAD_FEATURE_SCHEMA

    def load(self):
        """Load trained ensemble models from disk."""
        meta_file = os.path.join(self.model_dir, 'metadata.json')
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            self.features = meta.get('features', DEAD_FEATURE_SCHEMA)

        self.models = []
        for f in sorted(os.listdir(self.model_dir)):
            if f.endswith('.txt'):
                m_path = os.path.join(self.model_dir, f)
                booster = lgb.Booster(model_file=m_path)
                self.models.append(booster)
        print(f"Loaded {len(self.models)} Stage 1 boosters from {self.model_dir}")

    def predict_proba(self, df):
        """Compute calibrated probability P(Dead) for every frame."""
        df_feat = add_dead_physics_features(df)
        X = df_feat[self.features]
        probs = np.zeros(len(df))
        for booster in self.models:
            probs += booster.predict(X)
        probs /= len(self.models)
        return probs

    def predict_gate(self, df, tau=0.38, min_dead_frames=50, min_live_frames=25, smooth_win=25, hyst=(0.25, 0.45)):
        """
        Return boolean array: True = Dead Ball, False = Live Ball.
        Uses smoothed probabilities + dual-threshold hysteresis + football stoppage duration filtering
        (selected via external LOMO benchmark across SkillCorner + IDSSE Clean & 41%-Masked).
        """
        prob_dead = self.predict_proba(df)
        p = pd.Series(prob_dead).rolling(smooth_win, min_periods=1, center=True).mean().values if smooth_win > 1 else prob_dead

        if hyst is not None:
            lo, hi = hyst
            raw_dead = np.zeros(len(p), dtype=int)
            state = int(p[0] >= tau)
            raw_dead[0] = state
            for i in range(1, len(p)):
                if state == 1 and p[i] <= lo:
                    state = 0
                elif state == 0 and p[i] >= hi:
                    state = 1
                raw_dead[i] = state
        else:
            raw_dead = (p >= tau).astype(int)

        filtered_dead = apply_stoppage_duration_filter(
            raw_dead,
            min_dead_frames=min_dead_frames,
            min_live_frames=min_live_frames
        )
        return filtered_dead == 1
