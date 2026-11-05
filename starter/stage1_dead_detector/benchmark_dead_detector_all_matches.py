"""
Stage 1 Dead-Ball Detector: Cross-Match Benchmark Across All 7 Matches
Strict Compliance with Nov 3 Rule Clarification (100% External Data)

Matches Benchmarked (362,359 total frames):
1. SkillCorner 2841: FC Barcelona vs Real Madrid (Camp Nou, 68,400 frames)
2. SkillCorner 4039: Manchester City vs Liverpool FC (Etihad Stadium, 68,316 frames)
3. SkillCorner 3442: Real Madrid vs FC Barcelona (Bernabéu/Di Stéfano, 66,972 frames)
4. SkillCorner 2440: Liverpool FC vs Manchester City (Anfield, 68,671 frames)
5. Metrica Sports Sample Game 1 (30,000 frames)
6. Metrica Sports Sample Game 2 (30,000 frames)
7. Metrica Sports Sample Game 3 (30,000 frames)

Evaluates Leave-One-Match-Out (LOMO) cross-validation for every single match:
- Dead Precision, Recall, F1
- Alive Precision, Recall, F1
- Overall Accuracy
- Official Competition Alive/Dead F1
- ROC-AUC
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

sys.path.insert(0, '.')
from evaluation.possession_metrics import alive_dead_f1, label_codes, confusion, scored_confusion

STAGE1_DEAD_FEATURES = [
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
    'active_total', 'active_roll_2s'
]

def add_stage1_dead_features(df):
    """Compute the 5 physical pillars of Stage 1 Dead-Ball detection."""
    df = df.copy()
    eps = 1e-4

    # 1. Pitch Boundary Distance & Out-of-bounds (Pitch 105 x 68)
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

    return df

def filter_short_dead_spells(is_dead_preds, min_dead_frames=50, min_live_frames=25):
    """Enforce football duration physics on binary is_dead predictions."""
    n = len(is_dead_preds)
    out = np.array(is_dead_preds, dtype=int, copy=True)
    
    # 1. Remove short dead blips (< min_dead_frames)
    i = 0
    while i < n:
        if out[i] == 1:  # Dead
            start = i
            while i < n and out[i] == 1:
                i += 1
            length = i - start
            if length < min_dead_frames:
                out[start:i] = 0  # Revert to Live
        else:
            i += 1

    # 2. Fill short live blips (< min_live_frames) surrounded by Dead
    i = 0
    while i < n:
        if out[i] == 0:  # Live
            start = i
            while i < n and out[i] == 0:
                i += 1
            length = i - start
            if length < min_live_frames:
                prev_dead = (start > 0 and out[start - 1] == 1)
                next_dead = (i < n and out[i] == 1)
                if prev_dead and next_dead:
                    out[start:i] = 1  # Fill with Dead
        else:
            i += 1

    return out

def compute_competition_alive_dead_f1(y_true_str, is_dead_pred):
    """
    Compute official alive_dead_f1 matching evaluation/possession_metrics.py.
    y_true_str: array of 'H', 'A', 'D'
    is_dead_pred: array of 0 (Live) or 1 (Dead)
    """
    pred_str = np.where(is_dead_pred == 1, 'D', np.where(y_true_str == 'A', 'A', 'H'))
    gt_codes = label_codes(y_true_str)
    pred_codes = label_codes(pred_str)
    cm = confusion(gt_codes, pred_codes)
    cm3 = scored_confusion(cm)
    return alive_dead_f1(cm3)

def load_all_external_matches():
    """Load and unify all 7 external matches with Stage 1 features."""
    print("\n[1/3] Loading all 7 external professional matches...")
    t0 = time.time()
    
    sc_path = 'data/skillcorner_4matches_features.csv'
    met_path = 'data/metrica_3games_features.csv'
    assert os.path.exists(sc_path), f"Missing {sc_path}"
    assert os.path.exists(met_path), f"Missing {met_path}"

    df_sc = pd.read_csv(sc_path)
    df_met = pd.read_csv(met_path)

    print(f"  SkillCorner 4 matches: {len(df_sc):,} frames")
    print(f"  Metrica 3 matches:    {len(df_met):,} frames")

    # Combine into unified dataset
    common_cols = [c for c in df_sc.columns if c in df_met.columns]
    df_all = pd.concat([df_sc[common_cols], df_met[common_cols]], ignore_index=True)
    
    print("\n[2/3] Extracting 5 Physical Pillars of Dead-Ball Features...")
    df_all = add_stage1_dead_features(df_all)
    df_all['is_dead'] = (df_all['label'] == 'D').astype(int)

    print(f"  Total External Corpus: {len(df_all):,} frames ({time.time() - t0:.1f}s)")
    print(f"  Total Dead frames:     {(df_all['is_dead'] == 1).sum():,} ({100*(df_all['is_dead'] == 1).mean():.1f}%)")
    print(f"  Total Live frames:     {(df_all['is_dead'] == 0).sum():,} ({100*(df_all['is_dead'] == 0).mean():.1f}%)")
    return df_all

def run_leave_one_match_out_benchmark(df_all):
    """Run Leave-One-Match-Out cross-validation across all 7 matches."""
    matches = df_all['game'].unique()
    print("\n" + "="*110)
    print(" RUNNING LEAVE-ONE-MATCH-OUT (LOMO) BENCHMARK ACROSS ALL 7 MATCHES")
    print("="*110)

    results = []
    
    for mid, test_game in enumerate(matches):
        t_fold = time.time()
        train_mask = (df_all['game'] != test_game)
        test_mask = (df_all['game'] == test_game)

        X_train = df_all.loc[train_mask, STAGE1_DEAD_FEATURES]
        y_train = df_all.loc[train_mask, 'is_dead'].values
        X_test = df_all.loc[test_mask, STAGE1_DEAD_FEATURES]
        y_test = df_all.loc[test_mask, 'is_dead'].values
        y_test_labels = df_all.loc[test_mask, 'label'].values

        # Train Stage 1 LightGBM Classifier
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.038,
            num_leaves=35,
            subsample=0.85,
            colsample_bytree=0.80,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        clf.fit(X_train, y_train)

        # Probabilities
        prob_dead = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob_dead)

        # Raw Thresholded Predictions (tau=0.45)
        raw_pred_dead = (prob_dead >= 0.45).astype(int)

        # Duration Filtered Predictions (min 50 frames dead, min 25 frames live)
        filt_pred_dead = filter_short_dead_spells(raw_pred_dead, min_dead_frames=50, min_live_frames=25)

        # Metrics
        acc_raw = accuracy_score(y_test, raw_pred_dead)
        acc_filt = accuracy_score(y_test, filt_pred_dead)

        prec, rec, f1, _ = precision_recall_fscore_support(y_test, filt_pred_dead, average=None, labels=[0, 1])
        prec_dead, rec_dead, f1_dead = prec[1], rec[1], f1[1]
        prec_live, rec_live, f1_live = prec[0], rec[0], f1[0]
        macro_f1 = (f1_dead + f1_live) / 2.0

        comp_alive_dead_f1 = compute_competition_alive_dead_f1(y_test_labels, filt_pred_dead)

        n_frames = len(y_test)
        dead_share = y_test.mean()

        print(f"\nMatch [{mid+1}/7]: {test_game} ({n_frames:,} frames, {dead_share:.1%} Dead) in {time.time()-t_fold:.1f}s")
        print(f"  ROC-AUC:        {auc:.4f}")
        print(f"  Frame Accuracy: {acc_filt:.2%} (Raw: {acc_raw:.2%})")
        print(f"  Alive/Dead F1:  {comp_alive_dead_f1:.4f}  (Competition Metric)")
        print(f"  Dead Class:     P={prec_dead:.4f} | R={rec_dead:.4f} | F1={f1_dead:.4f}")
        print(f"  Live Class:     P={prec_live:.4f} | R={rec_live:.4f} | F1={f1_live:.4f}")

        results.append({
            'game': test_game,
            'frames': n_frames,
            'dead_rate': dead_share,
            'roc_auc': auc,
            'accuracy': acc_filt,
            'alive_dead_f1': comp_alive_dead_f1,
            'dead_precision': prec_dead,
            'dead_recall': rec_dead,
            'dead_f1': f1_dead,
            'live_f1': f1_live,
            'macro_f1': macro_f1
        })

    df_res = pd.DataFrame(results)
    print("\n" + "="*110)
    print(" SUMMARY SCOREBOARD: STAGE 1 DEAD-BALL DETECTOR ACROSS ALL 7 MATCHES")
    print("="*110)
    print(f"{'Match Identifier':<32} | {'Frames':<8} | {'ROC-AUC':<8} | {'Accuracy':<9} | {'Alive/Dead F1':<14} | {'Dead F1':<8} | {'Dead P':<7} | {'Dead R':<7}")
    print("-"*110)
    for _, r in df_res.iterrows():
        print(f"{r['game']:<32} | {r['frames']:<8,d} | {r['roc_auc']:<8.4f} | {r['accuracy']:<9.2%} | {r['alive_dead_f1']:<14.4f} | {r['dead_f1']:<8.4f} | {r['dead_precision']:<7.4f} | {r['dead_recall']:<7.4f}")
    
    print("-"*110)
    mean_auc = df_res['roc_auc'].mean()
    mean_acc = df_res['accuracy'].mean()
    mean_ad_f1 = df_res['alive_dead_f1'].mean()
    mean_df1 = df_res['dead_f1'].mean()
    mean_dp = df_res['dead_precision'].mean()
    mean_dr = df_res['dead_recall'].mean()
    print(f"{'MEAN OUT-OF-SAMPLE BENCHMARK':<32} | {df_res['frames'].sum():<8,d} | {mean_auc:<8.4f} | {mean_acc:<9.2%} | {mean_ad_f1:<14.4f} | {mean_df1:<8.4f} | {mean_dp:<7.4f} | {mean_dr:<7.4f}")
    print("="*110)

    # Save summary report CSV
    out_csv = 'starter/stage1_dead_detector/stage1_dead_detector_7matches_scoreboard.csv'
    df_res.to_csv(out_csv, index=False)
    print(f"\nWrote full scoreboard to: {out_csv}")
    return df_res

def main():
    print("="*110)
    print(" STAGE 1 DEAD-BALL DETECTOR: 7-MATCH COMPREHENSIVE OUT-OF-SAMPLE BENCHMARK")
    print(" Strict Nov 3 Rule Compliance (100% External LOMO Cross-Validation)")
    print("="*110)
    df_all = load_all_external_matches()
    run_leave_one_match_out_benchmark(df_all)

if __name__ == '__main__':
    main()
