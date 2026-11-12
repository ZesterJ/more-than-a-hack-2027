"""
Ablation & Combination Study for Stage 1 Live/Dead Detector Improvements
Strict Nov 3 Rule Compliance (100% External 11-Fold LOMO CV)

Tests all 5 measures & combinations:
- V0: Current PR #17 (30 features)
- V1: + SpeedMask (NaN-out constant 1.40 dummy speed in IDSSE)
- V2: V1 + MatchNorm (match-relative z-scores & ratios for spread, active players, dist)
- V3: V1 + LeadLag (directional lead_2s - lag_2s boundary transition features)
- V4: V1 + V2 + V3 (All features combined)
+ External LOMO sweep of hysteresis thresholds (lo, hi) & smoothing windows.
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
from starter.stage1_dead_detector.combined_live_dead_benchmark import (
    simulate_ball_gaps, interval_f1_metrics, apply_temporal_candidate
)

SPEED_COLS = ['speed_all_roll_2s', 'pitch_decel_2s', 'is_collective_walking', 'walking_roll_2s']
NORM_COLS = ['spread_area_z', 'home_spread_x_z', 'away_spread_x_z', 'active_ratio', 'min_dist_z']
LEADLAG_COLS = ['spread_delta_fwd_2s', 'dist_delta_fwd_2s', 'missing_delta_fwd_2s', 'players3m_delta_fwd_2s']


def add_v4_features_per_match(df, mask_const_speed=True):
    """Compute base 30 features + SpeedMask + MatchNorm + LeadLag per match."""
    gf = add_dead_physics_features(df)

    # V1: Mask constant dummy speed (e.g. IDSSE 1.40 constant) so trees don't split on fake constants
    if mask_const_speed and 'overall_speed' in gf.columns:
        if gf['overall_speed'].std() < 0.05:
            for c in SPEED_COLS:
                if c in gf.columns:
                    gf[c] = np.nan

    # V2: Match-relative Z-scores & ratios (using only the match's own inputs)
    def safe_z(s):
        mu = s.mean()
        sd = s.std()
        return (s - mu) / (sd + 1e-4)

    gf['spread_area_z'] = safe_z(gf['total_spread_area'])
    gf['home_spread_x_z'] = safe_z(gf['home_spread_x'].fillna(15.0))
    gf['away_spread_x_z'] = safe_z(gf['away_spread_x'].fillna(15.0))
    q95_act = max(1.0, float(gf['active_total'].quantile(0.95)))
    gf['active_ratio'] = gf['active_total'] / q95_act
    gf['min_dist_z'] = safe_z(gf['min_dist_any'])

    # V3: Directional Lead/Lag (future 2s minus past 2s) for exact stoppage/restart edges
    # Using shift(-25).rolling(50) ~ lead 2s; shift(25).rolling(50) ~ lag 2s
    def lead_minus_lag(series, win=50):
        lead = series.iloc[::-1].rolling(win, min_periods=1).mean().iloc[::-1]
        lag = series.rolling(win, min_periods=1).mean()
        return (lead - lag).fillna(0.0)

    gf['spread_delta_fwd_2s'] = lead_minus_lag(gf['total_spread_area'], 50)
    gf['dist_delta_fwd_2s'] = lead_minus_lag(gf['min_dist_any'], 50)
    gf['missing_delta_fwd_2s'] = lead_minus_lag(gf['ball_missing'].fillna(0), 50)
    gf['players3m_delta_fwd_2s'] = lead_minus_lag(gf['players_within_3m_total'], 50)

    return gf


def evaluate_feature_set_lomo(feat_cols, df_sc, df_ic, df_im, unique_games, decoder_cfg):
    """Run 11-fold LOMO and return per-domain aggregate metrics + OOF predictions."""
    oof_probs = {}
    records = []

    for held_game in unique_games:
        tr_df = pd.concat([
            df_sc[df_sc['game'] != held_game],
            df_ic[df_ic['game'] != held_game],
            df_im[df_im['game'] != held_game]
        ], ignore_index=True)

        dtrain = lgb.Dataset(tr_df[feat_cols], label=tr_df['is_dead'].values, free_raw_data=True)
        params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'learning_rate': 0.038, 'num_leaves': 35,
            'subsample': 0.85, 'subsample_freq': 1, 'colsample_bytree': 0.80,
            'seed': 42, 'n_jobs': -1, 'verbose': -1,
        }
        booster = lgb.train(params, dtrain, num_boost_round=300)

        targets = []
        if held_game in df_sc['game'].values:
            targets.append(('SkillCorner', df_sc[df_sc['game'] == held_game]))
        else:
            targets.append(('IDSSE_Clean', df_ic[df_ic['game'] == held_game]))
            targets.append(('IDSSE_Masked41', df_im[df_im['game'] == held_game]))

        for dom, te_df in targets:
            y_true = te_df['is_dead'].values
            probs = booster.predict(te_df[feat_cols])
            oof_probs[(dom, held_game)] = (y_true, probs)

            preds = apply_temporal_candidate(
                probs,
                tau=decoder_cfg['tau'],
                dead_min=decoder_cfg['dead_min'],
                live_min=decoder_cfg['live_min'],
                smooth_win=decoder_cfg['smooth_win'],
                hyst=decoder_cfg['hyst']
            )
            auc = roc_auc_score(y_true, probs)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average=None, labels=[0, 1], zero_division=0)
            im = interval_f1_metrics(y_true == 1, preds == 1)
            records.append({
                'domain': dom, 'game': held_game, 'roc_auc': auc,
                'dead_p': prec[1], 'dead_r': rec[1], 'dead_f1': f1[1],
                'macro_f1': (f1[0] + f1[1]) / 2.0,
                'int_f1_05': im['interval_f1_05'], 'int_f1_mean': im['interval_f1_mean'],
                'bursts': im['false_dead_bursts']
            })

    return pd.DataFrame(records), oof_probs


def main():
    print("=" * 115)
    print(" STAGE 1 ABLATION & COMBINATION STUDY (11-FOLD EXTERNAL LOMO CV)")
    print("=" * 115)
    t0 = time.time()

    df_sc = pd.read_csv('data/skillcorner_4matches_features.csv')
    df_id = pd.read_csv('data/idsse_7matches_features.csv')
    common_cols = [c for c in df_sc.columns if c in df_id.columns]
    df_sc = df_sc[common_cols].copy()
    df_id = df_id[common_cols].copy()
    df_id_m = simulate_ball_gaps(df_id, target_missing_rate=0.41, seed=42)

    def build_dataset(dframe, mask_speed):
        out = []
        for _, gdf in dframe.groupby('game', sort=False):
            gf = add_v4_features_per_match(gdf, mask_const_speed=mask_speed)
            gf['is_dead'] = (gf['label'] == 'D').astype(int)
            out.append(gf)
        return pd.concat(out, ignore_index=True)

    print("[1/3] Building feature datasets (raw speed vs speed-masked)...")
    sc_v0 = build_dataset(df_sc, mask_speed=False)
    ic_v0 = build_dataset(df_id, mask_speed=False)
    im_v0 = build_dataset(df_id_m, mask_speed=False)

    sc_v1 = build_dataset(df_sc, mask_speed=True)
    ic_v1 = build_dataset(df_id, mask_speed=True)
    im_v1 = build_dataset(df_id_m, mask_speed=True)

    unique_games = list(df_sc['game'].unique()) + list(df_id['game'].unique())
    default_dec = {'tau': 0.45, 'dead_min': 50, 'live_min': 25, 'smooth_win': 15, 'hyst': (0.32, 0.55)}

    variants = [
        ('V0_Baseline_30f', DEAD_FEATURE_SCHEMA, sc_v0, ic_v0, im_v0),
        ('V1_SpeedMask_30f', DEAD_FEATURE_SCHEMA, sc_v1, ic_v1, im_v1),
        ('V2_MatchNorm_35f', DEAD_FEATURE_SCHEMA + NORM_COLS, sc_v1, ic_v1, im_v1),
        ('V3_LeadLag_34f', DEAD_FEATURE_SCHEMA + LEADLAG_COLS, sc_v1, ic_v1, im_v1),
        ('V4_AllCombined_39f', DEAD_FEATURE_SCHEMA + NORM_COLS + LEADLAG_COLS, sc_v1, ic_v1, im_v1),
    ]

    print("\n[2/3] Running Feature Ablations (with fixed hyst_d50_l25 decoder)...")
    all_summaries = []
    oof_by_variant = {}

    for vname, fcols, d_sc, d_ic, d_im in variants:
        tv = time.time()
        df_r, oofs = evaluate_feature_set_lomo(fcols, d_sc, d_ic, d_im, unique_games, default_dec)
        oof_by_variant[vname] = oofs
        agg = df_r.groupby('domain', sort=False)[['roc_auc', 'dead_p', 'dead_r', 'dead_f1', 'macro_f1', 'int_f1_05', 'int_f1_mean', 'bursts']].mean()
        overall = df_r[['roc_auc', 'dead_f1', 'macro_f1', 'int_f1_05', 'int_f1_mean']].mean()
        print(f"  {vname:<22} ({time.time()-tv:.1f}s) | Mean AUC={overall['roc_auc']:.4f} | DeadF1={overall['dead_f1']:.4f} | IntF1@.5={overall['int_f1_05']:.4f} | IntF1Mean={overall['int_f1_mean']:.4f}")
        for dom, row in agg.iterrows():
            all_summaries.append({'variant': vname, 'domain': dom, **row.to_dict()})

    df_sum = pd.DataFrame(all_summaries)
    print("\n--- Detailed Feature Variant Comparison by Domain ---")
    print(f"{'Variant':<22} | {'Domain':<15} | {'AUC':<7} | {'Dead P':<7} | {'Dead R':<7} | {'Dead F1':<7} | {'IntF1@.5':<8} | {'Bursts':<6}")
    print("-" * 100)
    for _, r in df_sum.iterrows():
        print(f"{r['variant']:<22} | {r['domain']:<15} | {r['roc_auc']:<7.4f} | {r['dead_p']:<7.4f} | {r['dead_r']:<7.4f} | {r['dead_f1']:<7.4f} | {r['int_f1_05']:<8.4f} | {r['bursts']:<6.1f}")

    # Pick best feature variant by mean(Dead F1 + IntF1@.5) across all 3 domains
    rank_df = df_sum.groupby('variant', sort=False)[['roc_auc', 'dead_f1', 'int_f1_05', 'int_f1_mean']].mean()
    rank_df['score'] = 0.5 * rank_df['dead_f1'] + 0.5 * rank_df['int_f1_05']
    best_var = rank_df['score'].idxmax()
    print(f"\n--> Winning Feature Variant on External LOMO: {best_var} (Score={rank_df.loc[best_var, 'score']:.4f})")

    # [3/3] Sweep Hysteresis & Smoothing on the Winning Feature Variant's OOF Probabilities
    print(f"\n[3/3] Sweeping Hysteresis & Smoothing on {best_var} OOF Probabilities...")
    dec_candidates = [
        ('hyst_0.32_0.55_sw15', 0.45, 50, 25, 15, (0.32, 0.55)),
        ('hyst_0.28_0.48_sw15', 0.40, 50, 25, 15, (0.28, 0.48)),
        ('hyst_0.25_0.42_sw15', 0.35, 50, 25, 15, (0.25, 0.42)),
        ('hyst_0.30_0.45_sw25', 0.40, 50, 25, 25, (0.30, 0.45)),
        ('hyst_0.25_0.45_sw25', 0.38, 50, 25, 25, (0.25, 0.45)),
    ]

    best_oofs = oof_by_variant[best_var]
    dec_rows = []
    for dname, tau, dmin, lmin, swin, hyst in dec_candidates:
        for (dom, g), (y_true, probs) in best_oofs.items():
            preds = apply_temporal_candidate(probs, tau=tau, dead_min=dmin, live_min=lmin, smooth_win=swin, hyst=hyst)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average=None, labels=[0, 1], zero_division=0)
            im = interval_f1_metrics(y_true == 1, preds == 1)
            dec_rows.append({
                'decoder': dname, 'domain': dom,
                'dead_p': prec[1], 'dead_r': rec[1], 'dead_f1': f1[1],
                'int_f1_05': im['interval_f1_05'], 'int_f1_mean': im['interval_f1_mean'],
                'bursts': im['false_dead_bursts']
            })

    df_dec = pd.DataFrame(dec_rows)
    dec_agg = df_dec.groupby('decoder', sort=False)[['dead_p', 'dead_r', 'dead_f1', 'int_f1_05', 'int_f1_mean', 'bursts']].mean()
    dec_agg['score'] = 0.5 * dec_agg['dead_f1'] + 0.5 * dec_agg['int_f1_05']
    print(f"\n{'Decoder':<22} | {'Dead P':<7} | {'Dead R':<7} | {'Dead F1':<7} | {'IntF1@.5':<8} | {'IntF1Mean':<9} | {'Bursts':<6} | {'Score':<7}")
    print("-" * 92)
    for dname, r in dec_agg.iterrows():
        print(f"{dname:<22} | {r['dead_p']:<7.4f} | {r['dead_r']:<7.4f} | {r['dead_f1']:<7.4f} | {r['int_f1_05']:<8.4f} | {r['int_f1_mean']:<9.4f} | {r['bursts']:<6.1f} | {r['score']:<7.4f}")

    df_sum.to_csv('starter/stage1_dead_detector/ablation_variants_scoreboard.csv', index=False)
    print(f"\nTotal ablation study completed in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
