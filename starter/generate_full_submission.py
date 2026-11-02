"""
Generate the Official Full-Match Competition Possession Submission CSV (Ultimate Champion Edition)

Output: evaluation/results/champion_submission.csv
Rows: Exactly 199,319 data rows (frame 0 .. 199318)

Architecture (All 15 Steps Integrated):
1. 40-Dimensional Kinematic & Spatial Feature Space.
2. Step 15: StatsBomb 2D KDE Spatial Turnover Risk Topology (from 2,043 historical events).
3. Step 13: In-Domain Self-Training with High-Confidence Pseudo-Labels (5,210 frames from match play).
4. Multi-Tier Domain Loss Weighting: Barça GT (30x) + Pseudo-Labels (3x) + Metrica (1x).
5. Step 14: StatsBomb Bayesian Viterbi Sequence Decoder (3x3 Markov Transition Matrix from 1,727 historical spells).
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

import games
from starter.extract_features import extract_features, PITCH_LENGTH, PITCH_WIDTH, FPS, HOME_SLOTS, AWAY_SLOTS, FIELD_SEP, GROUP_SEP, MISSING_CHAR
from starter.step4_feature_engineering_benchmark import augment_kinematic_features
from starter.step15_spatial_turnover_feature import fit_turnover_kde, add_spatial_turnover_feature
from starter.step14_statsbomb_bayesian_viterbi import load_statsbomb_priors, build_markov_transition_matrix, viterbi_decode

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "champion_submission.csv")

def extract_features_df(raw_path, start_frame, end_frame):
    """Fast in-memory feature extractor returning an augmented DataFrame."""
    records = []
    
    with open(raw_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(FIELD_SEP)
            if len(parts) != 3:
                continue

            frame_id = int(parts[0])
            video_frame = frame_id - 1

            if video_frame < start_frame:
                continue
            if video_frame > end_frame:
                break

            # 1. Ball Coordinates
            ball_part = parts[2]
            if not ball_part or ball_part[0] == MISSING_CHAR or ball_part.startswith("-1"):
                ball_missing = 1
                ball_x = np.nan
                ball_y = np.nan
                ball_speed = 0.0
            else:
                ball_missing = 0
                b_tokens = ball_part.split(",")
                ball_x = float(b_tokens[0]) * PITCH_LENGTH
                ball_y = float(b_tokens[1]) * PITCH_WIDTH
                ball_speed = float(b_tokens[2]) if len(b_tokens) > 2 and b_tokens[2][0] != MISSING_CHAR else 0.0

            # 2. Player Coordinates & Speeds
            home_coords = []
            home_speeds = []
            away_coords = []
            away_speeds = []

            for slot_idx, group in enumerate(parts[1].split(GROUP_SEP)):
                if not group or group[0] == MISSING_CHAR or group.startswith("-1"):
                    continue
                tokens = group.split(",")
                px = float(tokens[0]) * PITCH_LENGTH
                py = float(tokens[1]) * PITCH_WIDTH
                ps = float(tokens[2]) if len(tokens) > 2 and tokens[2][0] != MISSING_CHAR else 0.0

                if slot_idx in HOME_SLOTS:
                    home_coords.append((px, py))
                    home_speeds.append(ps)
                elif slot_idx in AWAY_SLOTS:
                    away_coords.append((px, py))
                    away_speeds.append(ps)

            # 3. Proximity & Margins
            d_min_home = 999.0
            d_min_away = 999.0
            home_within_3m = 0
            away_within_3m = 0

            if ball_missing == 0:
                for px, py in home_coords:
                    d = math.hypot(px - ball_x, py - ball_y)
                    if d < d_min_home:
                        d_min_home = d
                    if d <= 3.0:
                        home_within_3m += 1

                for px, py in away_coords:
                    d = math.hypot(px - ball_x, py - ball_y)
                    if d < d_min_away:
                        d_min_away = d
                    if d <= 3.0:
                        away_within_3m += 1

                d_margin = d_min_away - d_min_home
            else:
                d_min_home = np.nan
                d_min_away = np.nan
                d_margin = 0.0

            # 4. Collective Geometry
            hx = [c[0] for c in home_coords]
            hy = [c[1] for c in home_coords]
            ax = [c[0] for c in away_coords]
            ay = [c[1] for c in away_coords]

            home_cx = float(np.mean(hx)) if hx else 0.0
            home_cy = float(np.mean(hy)) if hy else 0.0
            away_cx = float(np.mean(ax)) if ax else 0.0
            away_cy = float(np.mean(ay)) if ay else 0.0
            diff_cx = home_cx - away_cx

            home_sx = float(np.std(hx)) if len(hx) > 1 else 0.0
            home_sy = float(np.std(hy)) if len(hy) > 1 else 0.0
            away_sx = float(np.std(ax)) if len(ax) > 1 else 0.0
            away_sy = float(np.std(ay)) if len(ay) > 1 else 0.0
            spread_ratio = (home_sx * home_sy) / ((away_sx * away_sy) + 1e-4)

            # 5. Speeds
            m_spd_home = float(np.mean(home_speeds)) if home_speeds else 0.0
            m_spd_away = float(np.mean(away_speeds)) if away_speeds else 0.0
            all_speeds = home_speeds + away_speeds
            ov_spd = float(np.mean(all_speeds)) if all_speeds else 0.0

            records.append({
                "video_frame": video_frame,
                "frame_id": frame_id,
                "ball_missing": ball_missing,
                "ball_x": ball_x,
                "ball_y": ball_y,
                "ball_speed": ball_speed,
                "d_min_home": d_min_home,
                "d_min_away": d_min_away,
                "d_margin": d_margin,
                "home_within_3m": home_within_3m,
                "away_within_3m": away_within_3m,
                "home_active": len(home_coords),
                "away_active": len(away_coords),
                "home_centroid_x": home_cx,
                "home_centroid_y": home_cy,
                "away_centroid_x": away_cx,
                "away_centroid_y": away_cy,
                "centroid_diff_x": diff_cx,
                "home_spread_x": home_sx,
                "home_spread_y": home_sy,
                "away_spread_x": away_sx,
                "away_spread_y": away_sy,
                "spread_ratio": spread_ratio,
                "mean_speed_home": m_spd_home,
                "mean_speed_away": m_spd_away,
                "overall_speed": ov_spd,
            })

    df = pd.DataFrame(records)
    # Temporal Rolling Features
    df["d_margin_roll_1s"] = df["d_margin"].rolling(window=FPS, min_periods=1, center=True).mean()
    df["d_margin_roll_3s"] = df["d_margin"].rolling(window=FPS * 3, min_periods=1, center=True).mean()
    df["speed_all_roll_2s"] = df["overall_speed"].rolling(window=FPS * 2, min_periods=1, center=True).mean()
    df["missing_ratio_roll_2s"] = df["ball_missing"].rolling(window=FPS * 2, min_periods=1, center=True).mean()
    
    return augment_kinematic_features(df)

def main():
    print("="*100)
    print(" GENERATING FULL-MATCH COMPETITION SUBMISSION CSV (ULTIMATE CHAMPION - 199,319 FRAMES)")
    print("="*100)
    t_start = time.time()

    # 1. Fit StatsBomb 2D KDE & Load Markov Priors
    print("\n[Stage 1] Loading StatsBomb Empirical Priors & Fitting 2D KDE...")
    kde = fit_turnover_kde()
    priors = load_statsbomb_priors()
    trans_matrix = build_markov_transition_matrix(priors)
    print("  Markov Transition Matrix A initialized:")
    print(np.round(trans_matrix, 5))

    # 2. Load Ground Truth & Metrica with StatsBomb Spatial Enrichment
    print("\n[Stage 2] Preparing Augmented Training Datasets...")
    raw_barca = pd.read_csv('data/features_public_window.csv')
    raw_met = pd.read_csv('data/metrica_3games_features.csv')

    df_barca = add_spatial_turnover_feature(augment_kinematic_features(raw_barca), kde)
    df_met = add_spatial_turnover_feature(augment_kinematic_features(raw_met), kde)

    all_features = [c for c in df_barca.columns if c not in ['video_frame', 'frame_id', 'label']]
    joint_df = pd.concat([df_met[all_features + ['label']], df_barca[all_features + ['label']]], ignore_index=True)
    sample_weights = np.ones(len(joint_df))
    sample_weights[len(df_met):] = 30.0

    # 3. Train Teacher Model
    print("\n[Stage 3] Training Teacher Model on GT (30x) + Metrica (1x)...")
    teacher = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.035, num_leaves=38, random_state=42, verbosity=-1)
    teacher.fit(joint_df[all_features], joint_df['label'], sample_weight=sample_weights)
    classes = list(teacher.classes_)
    h_idx = classes.index('H'); a_idx = classes.index('A'); d_idx = classes.index('D')

    # 4. Extract Match Play and Mine High-Confidence Pseudo-Labels
    paths = games.asset_paths(source="tactical")
    raw_atd = paths["atd_raw"]

    half1_start, half1_end = 11185, 90651
    half2_start, half2_end = 110808, 190197
    total_video_frames = games.GAMES["BAR-ATH"]["n_video_frames"]  # 199,319

    full_labels = np.full(total_video_frames, "X", dtype="<U1")

    # Extract Half 1 features
    print(f"\n[Stage 4] Extracting First Half features ({half1_start:,} .. {half1_end:,} = {half1_end-half1_start+1:,} frames)...")
    t0 = time.time()
    df_h1 = add_spatial_turnover_feature(extract_features_df(raw_atd, half1_start, half1_end), kde)
    print(f"  Extracted Half 1 features in {time.time()-t0:.1f}s ({len(df_h1):,} rows, {len(all_features)} features).")

    # Mine pseudo-labels from Half 1 outside GT window (18685 to 38684)
    print("\n[Stage 5] Mining In-Domain Pseudo-Labels from Half 1 match play...")
    mask_pseudo = (df_h1['video_frame'] >= 18685) & (df_h1['video_frame'] <= 38684)
    df_unlabeled = df_h1[mask_pseudo].copy().reset_index(drop=True)
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
    print(f"  Mined {len(df_pseudo):,} pseudo-labels (distribution: {df_pseudo['label'].value_counts().to_dict()})")

    # Train Student Model
    print("\n[Stage 6] Training Student Model with In-Domain Self-Training & StatsBomb Topology...")
    student_df = pd.concat([df_met[all_features + ['label']],
                            df_pseudo[all_features + ['label']],
                            df_barca[all_features + ['label']]], ignore_index=True)

    student_weights = np.ones(len(student_df))
    student_weights[len(df_met) : len(df_met) + len(df_pseudo)] = 3.0
    student_weights[len(df_met) + len(df_pseudo) :] = 30.0

    student = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.035, num_leaves=38, random_state=42, verbosity=-1)
    student.fit(student_df[all_features], student_df['label'], sample_weight=student_weights)
    print("  Student Model successfully trained.")

    # Infer Half 1 with Bayesian Viterbi Sequence Decoder
    print("\n[Stage 7] Decoding Half 1 with StatsBomb Bayesian Viterbi Dynamic Programming...")
    t0 = time.time()
    probs_h1 = student.predict_proba(df_h1[all_features])
    viterbi_preds_h1 = viterbi_decode(probs_h1, trans_matrix, classes)
    full_labels[half1_start : half1_end + 1] = viterbi_preds_h1
    print(f"  First Half decoded in {time.time()-t0:.1f}s.")

    # Half 2 Processing
    print(f"\n[Stage 8] Processing Second Half ({half2_start:,} .. {half2_end:,} = {half2_end-half2_start+1:,} frames)...")
    t0 = time.time()
    df_h2 = add_spatial_turnover_feature(extract_features_df(raw_atd, half2_start, half2_end), kde)
    print(f"  Extracted Half 2 features in {time.time()-t0:.1f}s ({len(df_h2):,} rows).")

    probs_h2 = student.predict_proba(df_h2[all_features])
    viterbi_preds_h2 = viterbi_decode(probs_h2, trans_matrix, classes)
    full_labels[half2_start : half2_end + 1] = viterbi_preds_h2
    print(f"  Second Half decoded in {time.time()-t0:.1f}s.")

    # Write Full Submission CSV
    print(f"\n[Stage 9] Writing submission file to {OUT_PATH}...")
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("frame,label\n")
        f.writelines(f"{frame},{lbl}\n" for frame, lbl in enumerate(full_labels))

    print(f"  Written {len(full_labels):,} rows in {time.time()-t_start:.1f}s total.")

    # Validate Submission
    print("\n[Stage 10] Running Official Competition Validator (validate_submission.py)...")
    val_cmd = f"python evaluation/validate_submission.py {OUT_PATH}"
    os.system(val_cmd)

    # Score Against Public Ground Truth
    print("\n[Stage 11] Scoring Against Public Ground Truth (run_evaluation.py)...")
    os.system("python evaluation/run_evaluation.py")

if __name__ == '__main__':
    main()
