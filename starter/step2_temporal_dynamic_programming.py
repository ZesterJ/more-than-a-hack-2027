"""
Step 2: Temporal Dynamic Programming & Minimum Duration Post-Processing

This script implements:
1. Minimum Spell Duration Constraint: Merges transient micro-spells shorter than threshold T.
2. Viterbi Path Decoder with Transition Cost Penalties.
3. Evaluates impact on Spell F1, Macro F1, and Frame Accuracy on the 30% unseen test slice.
"""

import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import lightgbm as lgb
from evaluation.possession_metrics import evaluate

def merge_short_spells(preds, min_duration=50):
    """
    Suppresses transient micro-spells shorter than min_duration frames.
    Merges them into the preceding or dominant surrounding spell.
    """
    preds = list(preds)
    n = len(preds)
    if n == 0:
        return preds

    # Step 1: Identify all contiguous segments
    segments = []
    curr_label = preds[0]
    start = 0
    for i in range(1, n):
        if preds[i] != curr_label:
            segments.append([curr_label, start, i - 1])
            curr_label = preds[i]
            start = i
    segments.append([curr_label, start, n - 1])

    # Step 2: Iterate and merge segments shorter than min_duration
    changed = True
    while changed:
        changed = False
        new_segments = []
        i = 0
        while i < len(segments):
            lbl, s, e = segments[i]
            dur = e - s + 1
            if dur < min_duration and len(segments) > 1:
                changed = True
                # Merge with neighbor
                if i > 0 and i < len(segments) - 1:
                    prev_dur = segments[i-1][2] - segments[i-1][1] + 1
                    next_dur = segments[i+1][2] - segments[i+1][1] + 1
                    if prev_dur >= next_dur:
                        new_segments[-1][2] = e  # merge into previous
                    else:
                        segments[i+1][1] = s     # merge into next
                elif i > 0:
                    new_segments[-1][2] = e      # merge into previous
                else:
                    segments[i+1][1] = s         # merge into next
            else:
                new_segments.append([lbl, s, e])
            i += 1
        
        # Merge adjacent segments with identical label
        merged = []
        for seg in new_segments:
            if merged and merged[-1][0] == seg[0]:
                merged[-1][2] = seg[2]
            else:
                merged.append(seg)
        segments = merged

    # Reconstruct frame array
    out = [''] * n
    for lbl, s, e in segments:
        for j in range(s, e + 1):
            out[j] = lbl
    return out

def viterbi_decode(prob_matrix, classes, stay_prob=0.99, switch_dead_cost=0.005, direct_turnover_cost=0.001):
    """
    Viterbi sequence decoder penalizing rapid transitions between states.
    States: 0: A, 1: D, 2: H (alphabetical)
    """
    T, K = prob_matrix.shape
    # Build transition matrix
    A = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                A[i, j] = stay_prob
            else:
                # Direct turnover H <-> A is more expensive than transitioning through Dead
                c_from, c_to = classes[i], classes[j]
                if (c_from == 'H' and c_to == 'A') or (c_from == 'A' and c_to == 'H'):
                    A[i, j] = direct_turnover_cost
                else:
                    A[i, j] = switch_dead_cost
        A[i, :] /= A[i, :].sum() # normalize

    log_A = np.log(A + 1e-12)
    log_B = np.log(prob_matrix + 1e-12)

    # DP tables
    V = np.zeros((T, K))
    ptr = np.zeros((T, K), dtype=int)

    V[0, :] = log_B[0, :]

    for t in range(1, T):
        for j in range(K):
            scores = V[t-1, :] + log_A[:, j] + log_B[t, j]
            ptr[t, j] = np.argmax(scores)
            V[t, j] = scores[ptr[t, j]]

    # Backtrack
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(V[-1, :])
    for t in range(T - 2, -1, -1):
        path[t] = ptr[t + 1, path[t + 1]]

    return [classes[idx] for idx in path]

def score_predictions(y_true_str, y_pred_str, label_name="Model"):
    label_map = {'H': 0, 'A': 1, 'D': 2}
    y_true = np.array([label_map[x] for x in y_true_str], dtype=np.int64)
    y_pred = np.array([label_map[x] for x in y_pred_str], dtype=np.int64)
    res = evaluate(y_true, y_pred)
    acc = (y_true == y_pred).mean()
    n_sp = (y_pred[:-1] != y_pred[1:]).sum() + 1
    print(f"| {label_name:38s} | {n_sp:6d} | {res['spell_f1_mean']:.4f} | {res['macro_f1']:.4f} | {acc:.2%} | {res['per_class']['H']['f1']:.4f} | {res['per_class']['A']['f1']:.4f} | {res['per_class']['D']['f1']:.4f} |")
    return res

def main():
    print("="*105)
    print(" STEP 2: TEMPORAL DYNAMIC PROGRAMMING & SPELL DURATION BENCHMARK")
    print("="*105)

    df_barca = pd.read_csv('data/features_public_window.csv')
    train_barca = df_barca.iloc[:5250].copy()
    test_barca = df_barca.iloc[5250:].copy()
    feature_cols = [c for c in df_barca.columns if c not in ['video_frame', 'frame_id', 'label']]

    # Train LightGBM model on Barca Train slice
    model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)
    model.fit(train_barca[feature_cols], train_barca['label'])
    
    probs = model.predict_proba(test_barca[feature_cols])
    classes = list(model.classes_)
    raw_preds = [classes[i] for i in np.argmax(probs, axis=1)]

    print(f"| {'Post-Processing Method':38s} | Spells | SpellF1| MacroF1| Acc    | F1 (H) | F1 (A) | F1 (D) |")
    print("|" + "-"*40 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|")

    # 1. Raw Argmax
    score_predictions(test_barca['label'].values, raw_preds, "Raw Model Predictions (No Filter)")

    # 2. Standard Rolling Window baseline (w=25)
    from starter.step1_hierarchical_ensemble import smooth_predictions
    score_predictions(test_barca['label'].values, smooth_predictions(raw_preds, 25), "Rolling Majority Window (w=25 / 1s)")

    # 3. Viterbi Path Decoder
    viterbi_preds = viterbi_decode(probs, classes, stay_prob=0.995, switch_dead_cost=0.003, direct_turnover_cost=0.0005)
    score_predictions(test_barca['label'].values, viterbi_preds, "Viterbi Sequence Decoder (High Stay Prior)")

    # 4. Minimum Duration Spell Filter
    for min_dur in [25, 50, 75, 100, 125, 150]:
        merged = merge_short_spells(raw_preds, min_duration=min_dur)
        score_predictions(test_barca['label'].values, merged, f"Min Duration Filter ({min_dur} frames / {min_dur/25:.1f}s)")

    # 5. Hybrid: Viterbi + Min Duration Filter
    viterbi_merged = merge_short_spells(viterbi_preds, min_duration=75)
    score_predictions(test_barca['label'].values, viterbi_merged, "Viterbi + Min Duration (75 frames / 3s)")

    viterbi_merged_125 = merge_short_spells(viterbi_preds, min_duration=125)
    score_predictions(test_barca['label'].values, viterbi_merged_125, "Viterbi + Min Duration (125 frames / 5s)")

if __name__ == '__main__':
    main()
