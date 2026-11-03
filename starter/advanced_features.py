"""
Advanced Spatial-Temporal, Kinematic & Spearman Pitch Control Feature Engineering
"""

import numpy as np
import pandas as pd

PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0
FPS = 25.0

def compute_advanced_features(df):
    """
    Computes a comprehensive 55+ dimensional spatial-temporal feature space.
    """
    df = df.copy()

    # 1. Pitch Boundary Distances (Origin is [0, 0] at corner)
    bx = df['ball_x'].fillna(PITCH_LENGTH / 2.0)
    by = df['ball_y'].fillna(PITCH_WIDTH / 2.0)

    dist_x = np.maximum(0, np.maximum(-bx, bx - PITCH_LENGTH))
    dist_y = np.maximum(0, np.maximum(-by, by - PITCH_WIDTH))
    df['ball_out_of_bounds'] = np.sqrt(dist_x**2 + dist_y**2)
    df['is_out_of_bounds'] = (df['ball_out_of_bounds'] > 0.1).astype(int)

    df['dist_to_center_x'] = np.abs(bx - (PITCH_LENGTH / 2.0))
    df['dist_to_center_y'] = np.abs(by - (PITCH_WIDTH / 2.0))
    df['dist_to_touchline'] = np.minimum(by, PITCH_WIDTH - by)
    df['dist_to_byline_home'] = bx
    df['dist_to_byline_away'] = PITCH_LENGTH - bx

    # 2. Kinematic Ball Velocity & Acceleration
    vx = df['ball_x'].diff().fillna(0) * FPS
    vy = df['ball_y'].diff().fillna(0) * FPS
    ax = vx.diff().fillna(0) * FPS
    ay = vy.diff().fillna(0) * FPS

    df['ball_speed_kin'] = np.sqrt(vx**2 + vy**2)
    df['ball_accel_kin'] = np.sqrt(ax**2 + ay**2)

    # Rolling ball kinematics
    for w in [13, 25, 50]:  # ~0.5s, 1.0s, 2.0s
        s_name = f'ball_speed_roll_{w}'
        a_name = f'ball_accel_roll_{w}'
        df[s_name] = df['ball_speed_kin'].rolling(w, min_periods=1, center=True).mean()
        df[a_name] = df['ball_accel_kin'].rolling(w, min_periods=1, center=True).mean()

    # 3. Spearman-Inspired Pitch Control at Ball Position
    dh = df['d_min_home'].fillna(999.0)
    da = df['d_min_away'].fillna(999.0)

    # Soft exponential decay with scale tau = 2.0m
    exp_h = np.exp(-np.clip(dh, 0, 30.0) / 2.0)
    exp_a = np.exp(-np.clip(da, 0, 30.0) / 2.0)
    denom = exp_h + exp_a + 1e-6

    df['spearman_ctrl_home'] = exp_h / denom
    df['spearman_ctrl_away'] = exp_a / denom
    df['spearman_ctrl_margin'] = df['spearman_ctrl_home'] - df['spearman_ctrl_away']
    df['ratio_d_min'] = dh / (da + 1e-4)

    # 4. Tactical Shape & Team Expansion Dynamics
    hsx = df['home_spread_x'].fillna(0)
    hsy = df['home_spread_y'].fillna(0)
    asx = df['away_spread_x'].fillna(0)
    asy = df['away_spread_y'].fillna(0)

    df['home_team_area'] = hsx * hsy
    df['away_team_area'] = asx * asy
    df['team_area_diff'] = df['home_team_area'] - df['away_team_area']
    df['team_area_ratio'] = df['home_team_area'] / (df['away_team_area'] + 1e-4)
    df['compactness_diff'] = (hsx + hsy) - (asx + asy)

    # Centroid Relatives
    df['ball_rel_home_cx'] = bx - df['home_centroid_x'].fillna(PITCH_LENGTH / 2.0)
    df['ball_rel_home_cy'] = by - df['home_centroid_y'].fillna(PITCH_WIDTH / 2.0)
    df['ball_rel_away_cx'] = bx - df['away_centroid_x'].fillna(PITCH_LENGTH / 2.0)
    df['ball_rel_away_cy'] = by - df['away_centroid_y'].fillna(PITCH_WIDTH / 2.0)

    # 5. Multi-Scale Rolling Temporal Proximities & Momentum
    for w in [13, 25, 50, 100]:  # 0.5s, 1.0s, 2.0s, 4.0s
        df[f'd_margin_roll_{w}'] = df['d_margin'].rolling(w, min_periods=1, center=True).mean()
        df[f'speed_all_roll_{w}'] = df['overall_speed'].rolling(w, min_periods=1, center=True).mean()
        df[f'missing_roll_{w}'] = df['ball_missing'].rolling(w, min_periods=1, center=True).mean()

    # Derivatives / Velocities of Margins
    df['d_margin_velocity'] = df['d_margin'].diff().fillna(0) * FPS
    df['centroid_diff_velocity'] = df['centroid_diff_x'].diff().fillna(0) * FPS
    df['ctrl_margin_velocity'] = df['spearman_ctrl_margin'].diff().fillna(0) * FPS

    # 6. Consecutive Dwell Runs (Control History)
    def calc_dwell(bool_arr):
        dwell = np.zeros(len(bool_arr))
        cnt = 0
        for i, val in enumerate(bool_arr):
            if val:
                cnt += 1
            else:
                cnt = 0
            dwell[i] = cnt
        return dwell

    df['dwell_home_1m'] = calc_dwell((dh < 1.0).values)
    df['dwell_away_1m'] = calc_dwell((da < 1.0).values)
    df['dwell_home_2m'] = calc_dwell((dh < 2.0).values)
    df['dwell_away_2m'] = calc_dwell((da < 2.0).values)

    # 7. Physical Stoppage & Pressure Signatures
    df['pressure_diff'] = df['home_within_3m'] - df['away_within_3m']
    df['stoppage_slow_energy'] = df['speed_all_roll_50'] * (1.0 - df['missing_roll_50'])
    df['is_goal_area'] = ((bx < 5.5) | (bx > PITCH_LENGTH - 5.5)).astype(int)

    return df
