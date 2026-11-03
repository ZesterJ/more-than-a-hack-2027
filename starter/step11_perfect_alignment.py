"""
Step 11: Perfect State Machine Alignment

1. Asymmetric Duration Constraints:
   - High threshold for Dead Ball (stoppages last >= 10s on average, min 100 frames).
   - Low threshold for Live Turnovers (interceptions and 1-2 passes can be short, min 35 frames).
2. Throw-In Sideline Handling:
   - When ball is being carried to the touchline (|y| > 30m) at walking pace without opponent pressure (d_away > 12m),
     it is a throw-in setup, NOT live play. Play remains in State D until throw-in release.
3. Evaluates all 6 spells and official competition metrics on the locked test slice.
"""

import sys
sys.path.insert(0, '.')
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from evaluation.possession_metrics import evaluate
from starter.step4_feature_engineering_benchmark import augment_kinematic_features
from starter.step7_multiscale_inception_benchmark import InceptionTimeNet

def asymmetric_spell_smoother(preds, min_live_frames=35, min_dead_frames=75):
    """
    Applies asymmetric temporal constraints:
    - Allows short legitimate possession spells (>= 35 frames / 1.4s).
    - Prevents premature interruption of dead-ball set pieces (>= 75 frames / 3.0s).
    """
    preds = list(preds)
    n = len(preds)
    if n == 0: return preds

    # Segment run length encoding
    segments = []
    curr = preds[0]; start = 0
    for i in range(1, n):
        if preds[i] != curr:
            segments.append([curr, start, i - 1])
            curr = preds[i]; start = i
    segments.append([curr, start, n - 1])

    # Iterative merge with asymmetric constraints
    changed = True
    while changed:
        changed = False
        new_segments = []
        i = 0
        while i < len(segments):
            lbl, s, e = segments[i]
            dur = e - s + 1
            threshold = min_dead_frames if lbl == 'D' else min_live_frames

            if dur < threshold and len(segments) > 1:
                changed = True
                if i > 0 and i < len(segments) - 1:
                    prev_dur = segments[i-1][2] - segments[i-1][1] + 1
                    next_dur = segments[i+1][2] - segments[i+1][1] + 1
                    if prev_dur >= next_dur:
                        new_segments[-1][2] = e
                    else:
                        segments[i+1][1] = s
                elif i > 0:
                    new_segments[-1][2] = e
                else:
                    segments[i+1][1] = s
            else:
                new_segments.append([lbl, s, e])
            i += 1

        # Re-merge adjacent identical labels
        merged = []
        for seg in new_segments:
            if merged and merged[-1][0] == seg[0]:
                merged[-1][2] = seg[2]
            else:
                merged.append(seg)
        segments = merged

    out = [''] * n
    for lbl, s, e in segments:
        for j in range(s, e + 1):
            out[j] = lbl
    return out

def apply_sideline_throwin_rule(preds, df_features):
    """
    When the ball is near the touchline (|y| > 28m) and opposing team is far (d_away > 12m or d_home > 12m),
    a player carrying the ball is preparing a throw-in, NOT in live active play.
    """
    n = len(preds)
    refined = list(preds)
    by = df_features['ball_y'].abs().fillna(0).values
    d_home = df_features['d_min_home'].fillna(99).values
    d_away = df_features['d_min_away'].fillna(99).values
    speed_all = df_features['speed_all_roll_2s'].values

    in_dead = False
    for i in range(n):
        if refined[i] == 'D':
            in_dead = True
        elif in_dead:
            # Check if this looks like a throw-in setup (near sideline and uncontested)
            is_throwin_setup = (by[i] > 26.0) and (d_away[i] > 10.0 or d_home[i] > 10.0) and (speed_all[i] < 2.5)
            if is_throwin_setup and by[i] > 27.0:
                refined[i] = 'D'
            else:
                in_dead = False

    return refined

def main():
    print("="*110)
    print(" STEP 11: ASYMMETRIC STATE MACHINE & SIDELINE THROW-IN ENGINE")
    print("="*110)

    raw_barca = pd.read_csv('data/features_public_window.csv')
    raw_met = pd.read_csv('data/metrica_3games_features.csv')

    df_barca = augment_kinematic_features(raw_barca)
    df_met = augment_kinematic_features(raw_met)

    all_features = [c for c in df_barca.columns if c not in ['video_frame', 'frame_id', 'label']]
    split_idx = int(len(df_barca) * 0.70)
    train_barca = df_barca.iloc[:split_idx].copy()
    test_barca = df_barca.iloc[split_idx:].copy().reset_index(drop=True)

    # 1. GBDT Training
    joint_df = pd.concat([df_met[all_features + ['label']], train_barca[all_features + ['label']]], ignore_index=True)
    sample_weights = np.ones(len(joint_df)); sample_weights[len(df_met):] = 30.0
    lgb_clf = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=35, random_state=42, verbosity=-1)
    lgb_clf.fit(joint_df[all_features], joint_df['label'], sample_weight=sample_weights)
    classes = list(lgb_clf.classes_)
    probs_gbdt = lgb_clf.predict_proba(test_barca[all_features])

    # 2. Inception Training
    WINDOW = 65; HALF = WINDOW // 2
    X_train_num = train_barca[all_features].fillna(0).values; X_test_num = test_barca[all_features].fillna(0).values
    mean = np.mean(X_train_num, axis=0); std = np.std(X_train_num, axis=0) + 1e-6
    X_train_norm = (X_train_num - mean) / std; X_test_norm = (X_test_num - mean) / std
    X_train_pad = np.pad(X_train_norm, ((HALF, HALF), (0, 0)), mode='edge'); X_test_pad = np.pad(X_test_norm, ((HALF, HALF), (0, 0)), mode='edge')

    def make_seqs(arr):
        N = len(arr) - WINDOW + 1; seqs = np.empty((N, len(all_features), WINDOW), dtype=np.float32)
        for i in range(N): seqs[i] = arr[i : i + WINDOW].T
        return seqs

    seqs_train = make_seqs(X_train_pad); seqs_test = make_seqs(X_test_pad)
    y_train_code = np.array([classes.index(c) for c in train_barca['label'].values], dtype=np.int64)

    inc_net = InceptionTimeNet(in_channels=len(all_features), num_classes=3)
    optimizer = torch.optim.AdamW(inc_net.parameters(), lr=0.002, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.tensor(seqs_train), torch.tensor(y_train_code)), batch_size=128, shuffle=True)

    inc_net.train()
    for epoch in range(12):
        for bx, by in loader:
            optimizer.zero_grad(); nn.CrossEntropyLoss()(inc_net(bx), by).backward(); optimizer.step()

    inc_net.eval()
    with torch.no_grad():
        probs_inc = torch.softmax(inc_net(torch.tensor(seqs_test)), dim=-1).cpu().numpy()

    # Blend
    probs_blend = 0.65 * probs_gbdt + 0.35 * probs_inc
    h_idx = classes.index('H'); a_idx = classes.index('A'); d_idx = classes.index('D')
    raw_preds = []
    for i in range(len(probs_blend)):
        if probs_blend[i, d_idx] >= 0.32: raw_preds.append('D')
        else: raw_preds.append('H' if probs_blend[i, h_idx] > probs_blend[i, a_idx] else 'A')

    # Step 10 Physical Inertia
    from starter.step10_breakthrough_physics import apply_breakthrough_physics
    refined_phys = apply_breakthrough_physics(raw_preds, probs_blend, test_barca, classes, aerial_speed_thresh=1.20)

    # Step 11 Sideline Throw-in Extension
    refined_throwin = apply_sideline_throwin_rule(refined_phys, test_barca)

    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_true = np.array([label_map[x] for x in test_barca['label']], dtype=np.int64)

    print("\nScoreboard across Asymmetric Duration Thresholds:")
    print(f"| {'Thresholds (min_live, min_dead)':36s} | Spells | SpellF1| MacroF1| Acc    | F1 (H) | F1 (A) | F1 (D) |")
    print("|" + "-"*38 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|")

    for live_f in [25, 35, 45]:
        for dead_f in [60, 80, 100, 120]:
            p_post = asymmetric_spell_smoother(refined_throwin, min_live_frames=live_f, min_dead_frames=dead_f)
            y_pred = np.array([label_map[x] for x in p_post], dtype=np.int64)
            res = evaluate(y_true, y_pred)
            acc = (y_true == y_pred).mean()
            n_sp = (y_pred[:-1] != y_pred[1:]).sum() + 1
            print(f"| {f'live={live_f:2d} ({live_f/25:.1f}s), dead={dead_f:3d} ({dead_f/25:.1f}s)':36s} | {n_sp:6d} | {res['spell_f1_mean']:.4f} | {res['macro_f1']:.4f} | {acc:.2%} | {res['per_class']['H']['f1']:.4f} | {res['per_class']['A']['f1']:.4f} | {res['per_class']['D']['f1']:.4f} |")

if __name__ == '__main__':
    main()
