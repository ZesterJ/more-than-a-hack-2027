"""
Combined Stage 1 Live/Dead Detector: Fusing PR #13 (5 Physical Pillars + Missing-Ball Resilience)
with PR #14 / PR #16 (IDSSE 7-Match Corpus, Ball-Gap Masking Augmentation & Interval Temporal Filtering)

Strict Nov 3 Rule Compliance (100% External Data, 0 Competition Match Frames in Training/Selection).

Datasets Unified (14 External Matches, 467,359 Raw Frames):
- SkillCorner 4 European Matches (272,359 frames; real broadcast tracking ~52% ball gaps)
- IDSSE 7 Bundesliga Matches (105,000 frames; clean TRACAB tracking + gap-masked augmentation)
- Metrica 3 Exhibition Matches (90,000 frames; optical tracking)

Evaluates both Frame-Level and PR #14 Interval-Level Metrics:
- Dead Frame F1, Precision, Recall, Binary Macro F1, ROC-AUC
- Dead Interval F1@0.5, Mean Interval F1 (IoU 0.3-0.7), Internal False-DEAD Bursts
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

sys.path.insert(0, '.')
from starter.stage1_dead_detector.dead_detector import (
    DEAD_FEATURE_SCHEMA, add_dead_physics_features, apply_stoppage_duration_filter
)


def simulate_ball_gaps(df, target_missing_rate=0.41, seed=42):
    """
    Simulate realistic contiguous broadcast ball-tracking dropouts (e.g. 41% missing rate)
    on clean optical tracking data (IDSSE) so Stage 1 learns to handle missing-ball spans.
    Only masks ball-dependent inputs; leaves team positions/spreads intact.
    """
    rng = np.random.default_rng(seed)
    df_masked = df.copy()
    n = len(df_masked)
    mask = np.zeros(n, dtype=bool)

    # Generate contiguous dropout spans (5 to 75 frames, i.e., 0.2s to 3.0s)
    total_target = int(n * target_missing_rate)
    masked_count = 0
    attempts = 0
    while masked_count < total_target and attempts < n:
        attempts += 1
        start = rng.integers(0, n)
        dur = int(rng.integers(5, 76))
        end = min(n, start + dur)
        if not mask[start:end].any():
            mask[start:end] = True
            masked_count += (end - start)

    df_masked.loc[mask, 'ball_missing'] = 1
    df_masked.loc[mask, 'ball_x'] = np.nan
    df_masked.loc[mask, 'ball_y'] = np.nan
    df_masked.loc[mask, 'ball_speed'] = np.nan
    df_masked.loc[mask, 'd_min_home'] = np.nan
    df_masked.loc[mask, 'd_min_away'] = np.nan
    df_masked.loc[mask, 'd_margin'] = np.nan
    df_masked.loc[mask, 'home_within_3m'] = 0
    df_masked.loc[mask, 'away_within_3m'] = 0
    return df_masked


def extract_binary_intervals(is_dead_bool):
    """Extract contiguous (start, end) intervals where is_dead_bool is True."""
    arr = np.asarray(is_dead_bool, dtype=bool)
    if len(arr) == 0:
        return []
    padded = np.r_[False, arr, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return list(zip(starts.tolist(), ends.tolist()))


def interval_f1_metrics(y_true_dead, y_pred_dead, iou_thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """
    Compute PR #14 style interval matching metrics:
    - Dead Interval Precision/Recall/F1 @ IoU 0.5
    - Mean Dead Interval F1 across IoU 0.3-0.7
    - Internal false-DEAD bursts (predicted DEAD runs strictly inside a true LIVE run)
    """
    gt_ints = extract_binary_intervals(y_true_dead)
    pred_ints = extract_binary_intervals(y_pred_dead)

    # Count internal false-DEAD bursts (pred dead interval completely inside a true live span)
    false_dead_bursts = 0
    for ps, pe in pred_ints:
        if not y_true_dead[ps:pe].any():
            if ps > 0 and pe < len(y_true_dead):
                false_dead_bursts += 1

    f1_by_iou = {}
    p_05, r_05, f1_05 = 0.0, 0.0, 0.0

    for thr in iou_thresholds:
        if not gt_ints and not pred_ints:
            p, r, f1 = 1.0, 1.0, 1.0
        elif not gt_ints or not pred_ints:
            p, r, f1 = 0.0, 0.0, 0.0
        else:
            # Greedy 1-to-1 matching by highest IoU
            pairs = []
            for gi, (gs, ge) in enumerate(gt_ints):
                for pi, (ps, pe) in enumerate(pred_ints):
                    inter = max(0, min(ge, pe) - max(gs, ps))
                    if inter > 0:
                        union = max(ge, pe) - min(gs, ps)
                        iou = inter / union
                        if iou >= thr:
                            pairs.append((iou, gi, pi))
            pairs.sort(reverse=True)
            used_g, used_p = set(), set()
            matched = 0
            for _, gi, pi in pairs:
                if gi not in used_g and pi not in used_p:
                    used_g.add(gi)
                    used_p.add(pi)
                    matched += 1
            p = matched / len(pred_ints)
            r = matched / len(gt_ints)
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        f1_by_iou[thr] = f1
        if abs(thr - 0.5) < 1e-6:
            p_05, r_05, f1_05 = p, r, f1

    return {
        'gt_intervals': len(gt_ints),
        'pred_intervals': len(pred_ints),
        'false_dead_bursts': false_dead_bursts,
        'interval_p_05': p_05,
        'interval_r_05': r_05,
        'interval_f1_05': f1_05,
        'interval_f1_mean': float(np.mean(list(f1_by_iou.values())))
    }


def apply_temporal_candidate(prob_dead, tau=0.45, dead_min=9, live_min=0, smooth_win=1, hyst=None):
    """Apply optional smoothing/hysteresis + threshold + internal run-length filtering."""
    p = np.asarray(prob_dead, dtype=float)
    if smooth_win > 1:
        p = pd.Series(p).rolling(smooth_win, min_periods=1, center=True).mean().values

    if hyst is not None:
        lo, hi = hyst
        codes = np.zeros(len(p), dtype=int)
        state = int(p[0] >= tau)
        codes[0] = state
        for i in range(1, len(p)):
            if state == 1 and p[i] <= lo:
                state = 0
            elif state == 0 and p[i] >= hi:
                state = 1
            codes[i] = state
    else:
        codes = (p >= tau).astype(int)

    if dead_min > 1 or live_min > 1:
        codes = apply_stoppage_duration_filter(
            codes,
            min_dead_frames=max(1, dead_min),
            min_live_frames=max(1, live_min)
        )
    return codes


def main():
    print("=" * 110)
    print(" COMBINED STAGE 1 LIVE/DEAD DETECTOR BENCHMARK (PR #13 + PR #14/16)")
    print(" 14 External Matches | SkillCorner + IDSSE (Clean & 41% Gap-Masked) + Metrica")
    print("=" * 110)
    t0 = time.time()

    # 1. Load all 3 external sources
    sc_path = 'data/skillcorner_4matches_features.csv'
    idsse_path = 'data/idsse_7matches_features.csv'
    met_path = 'data/metrica_3games_features.csv'

    df_sc = pd.read_csv(sc_path)
    df_idsse = pd.read_csv(idsse_path)
    df_met = pd.read_csv(met_path)

    common_cols = [c for c in df_sc.columns if c in df_idsse.columns and c in df_met.columns]
    df_sc = df_sc[common_cols].copy()
    df_idsse = df_idsse[common_cols].copy()
    df_met = df_met[common_cols].copy()

    df_sc['source'] = 'SkillCorner'
    df_idsse['source'] = 'IDSSE'
    df_met['source'] = 'Metrica'

    # Create gap-masked IDSSE copy for domain-resilient training & missing-ball stress testing
    print("\n[1/4] Building 41% Ball-Gap Masked IDSSE Augmentation...")
    df_idsse_masked = simulate_ball_gaps(df_idsse, target_missing_rate=0.41, seed=42)
    df_idsse_masked['source'] = 'IDSSE_GapMasked'

    # Compute Stage 1 features on each dataset separately (respecting match boundaries for rolling)
    print("[2/4] Extracting 30 Physical-Pillar Dead-Ball Features...")
    parts = []
    for name, dframe in [('SkillCorner', df_sc), ('IDSSE', df_idsse), ('IDSSE_GapMasked', df_idsse_masked), ('Metrica', df_met)]:
        feat_parts = []
        for g, gdf in dframe.groupby('game', sort=False):
            gf = add_dead_physics_features(gdf)
            gf['is_dead'] = (gf['label'] == 'D').astype(int)
            feat_parts.append(gf)
        parts.append(pd.concat(feat_parts, ignore_index=True))

    df_sc_f, df_idsse_f, df_idsse_m_f, df_met_f = parts
    print(f"  SkillCorner:     {len(df_sc_f):,} frames (4 matches)")
    print(f"  IDSSE Clean:     {len(df_idsse_f):,} frames (7 matches)")
    print(f"  IDSSE Masked41%: {len(df_idsse_m_f):,} frames (7 matches)")
    print(f"  Metrica:         {len(df_met_f):,} frames (3 matches)")

    # 2. LOMO Cross-Validation across SkillCorner + IDSSE (Clean & Masked)
    # Note: For every held-out match g, NEVER include g (clean or masked) in training!
    all_games_eval = (
        [(g, 'SkillCorner', df_sc_f) for g in df_sc_f['game'].unique()] +
        [(g, 'IDSSE_Clean', df_idsse_f) for g in df_idsse_f['game'].unique()] +
        [(g, 'IDSSE_Masked41', df_idsse_m_f) for g in df_idsse_m_f['game'].unique()]
    )

    # Combine training pool (excluding Metrica because Metrica lacks full throw-in dead tags)
    train_pool_sc = df_sc_f
    train_pool_id_clean = df_idsse_f
    train_pool_id_mask = df_idsse_m_f

    candidates = [
        ('raw_0.45', 0.45, 1, 1, 1, None),
        ('PR14_d9_l0', 0.45, 9, 1, 1, None),
        ('PR13_d50_l25', 0.45, 50, 25, 1, None),
        ('sm25_d50_l25', 0.45, 50, 25, 25, None),
        ('hyst_d50_l25', 0.45, 50, 25, 15, (0.32, 0.55)),
    ]

    print("\n[3/4] Running Strictly Isolated LOMO Benchmark Across SkillCorner + IDSSE (Clean & 41% Masked)...")
    oof_records = []

    unique_games = list(df_sc_f['game'].unique()) + list(df_idsse_f['game'].unique())
    for idx, held_game in enumerate(unique_games):
        t_f = time.time()
        # Exclude held_game from ALL training pools (both clean and masked!)
        tr_sc = train_pool_sc[train_pool_sc['game'] != held_game]
        tr_ic = train_pool_id_clean[train_pool_id_clean['game'] != held_game]
        tr_im = train_pool_id_mask[train_pool_id_mask['game'] != held_game]
        tr_df = pd.concat([tr_sc, tr_ic, tr_im], ignore_index=True)

        dtrain = lgb.Dataset(tr_df[DEAD_FEATURE_SCHEMA], label=tr_df['is_dead'].values, free_raw_data=True)
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.038,
            'num_leaves': 35,
            'subsample': 0.85,
            'subsample_freq': 1,
            'colsample_bytree': 0.80,
            'seed': 42,
            'n_jobs': -1,
            'verbose': -1,
        }
        booster = lgb.train(lgb_params, dtrain, num_boost_round=300)

        # Evaluate on the held-out match (and for IDSSE, evaluate on BOTH Clean and 41% Gap-Masked!)
        eval_targets = []
        if held_game in df_sc_f['game'].values:
            eval_targets.append(('SkillCorner', df_sc_f[df_sc_f['game'] == held_game]))
        else:
            eval_targets.append(('IDSSE_Clean', df_idsse_f[df_idsse_f['game'] == held_game]))
            eval_targets.append(('IDSSE_Masked41', df_idsse_m_f[df_idsse_m_f['game'] == held_game]))

        for dom, te_df in eval_targets:
            y_true = te_df['is_dead'].values
            probs = booster.predict(te_df[DEAD_FEATURE_SCHEMA])
            auc = roc_auc_score(y_true, probs)

            for cname, tau, dmin, lmin, swin, hyst in candidates:
                preds = apply_temporal_candidate(probs, tau=tau, dead_min=dmin, live_min=lmin, smooth_win=swin, hyst=hyst)
                prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average=None, labels=[0, 1], zero_division=0)
                macro_f1 = (f1[0] + f1[1]) / 2.0
                im = interval_f1_metrics(y_true == 1, preds == 1)

                oof_records.append({
                    'domain': dom,
                    'game': held_game,
                    'filter': cname,
                    'frames': len(y_true),
                    'roc_auc': auc,
                    'dead_p': prec[1],
                    'dead_r': rec[1],
                    'dead_f1': f1[1],
                    'binary_macro_f1': macro_f1,
                    'interval_f1_05': im['interval_f1_05'],
                    'interval_f1_mean': im['interval_f1_mean'],
                    'false_dead_bursts': im['false_dead_bursts'],
                    'gt_intervals': im['gt_intervals'],
                    'pred_intervals': im['pred_intervals']
                })
        print(f"  Fold [{idx+1}/{len(unique_games)}] {held_game:<32} done in {time.time()-t_f:.1f}s")

    df_res = pd.DataFrame(oof_records)

    print("\n" + "=" * 115)
    print(" [4/4] AGGREGATE RESULTS BY DOMAIN & TEMPORAL FILTER (STRICT OUT-OF-SAMPLE LOMO)")
    print("=" * 115)
    summary = df_res.groupby(['domain', 'filter'], sort=False)[
        ['roc_auc', 'dead_p', 'dead_r', 'dead_f1', 'binary_macro_f1', 'interval_f1_05', 'interval_f1_mean', 'false_dead_bursts']
    ].mean().reset_index()

    print(f"{'Domain':<16} | {'Filter':<14} | {'AUC':<7} | {'Dead P':<7} | {'Dead R':<7} | {'Dead F1':<7} | {'Mac F1':<7} | {'IntF1@.5':<8} | {'IntF1Mean':<9} | {'Bursts':<6}")
    print("-" * 115)
    for _, r in summary.iterrows():
        print(f"{r['domain']:<16} | {r['filter']:<14} | {r['roc_auc']:<7.4f} | {r['dead_p']:<7.4f} | {r['dead_r']:<7.4f} | {r['dead_f1']:<7.4f} | {r['binary_macro_f1']:<7.4f} | {r['interval_f1_05']:<8.4f} | {r['interval_f1_mean']:<9.4f} | {r['false_dead_bursts']:<6.1f}")
    print("=" * 115)

    out_csv = 'starter/stage1_dead_detector/combined_stage1_lomo_scoreboard.csv'
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved detailed per-match results to: {out_csv} (Total time: {time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
