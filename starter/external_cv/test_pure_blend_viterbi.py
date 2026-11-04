"""
Test Pure Multi-Tree Blend + Bayesian Viterbi (without brittle heuristics)
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, '.')
from starter.generate_full_submission import viterbi_decode, apply_duration_filter, apply_sandwich_hysteresis
from starter.external_cv.optimize_unseen_pipeline import load_aligned_datasets
from starter.step14_statsbomb_bayesian_viterbi import load_statsbomb_priors, build_markov_transition_matrix

def main():
    all_data = load_aligned_datasets()
    train_games = ['SkillCorner_Barca_RealMadrid', 'SkillCorner_ManCity_Liverpool', 'Metrica_1', 'Metrica_2']
    test_games = ['SkillCorner_RealMadrid_Barca', 'SkillCorner_Liverpool_ManCity', 'Metrica_3']
    
    drop_uninformative = ['video_frame', 'frame_id', 'label', 'game',
                          'home_attack_direction', 'ball_missing', 'overall_speed', 'speed_all_roll_2s']
    feature_cols = [c for c in all_data.columns if c not in drop_uninformative]
    label_map = {'H': 0, 'A': 1, 'D': 2}
    
    train_df = all_data[all_data['game'].isin(train_games)].copy().reset_index(drop=True)
    X_train = train_df[feature_cols]
    y_train = train_df['label'].map(label_map).values
    
    print("Training 3 LGBM + 1 XGBoost...")
    lgb_models = [
        lgb.LGBMClassifier(n_estimators=350, learning_rate=0.038, num_leaves=42, subsample=0.85, colsample_bytree=0.75, random_state=s, n_jobs=-1, verbose=-1).fit(X_train, y_train)
        for s in [42, 101, 2024]
    ]
    xgb_clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.04, max_depth=6, subsample=0.85, colsample_bytree=0.75, tree_method='hist', random_state=42, n_jobs=-1).fit(X_train, y_train)
    
    priors = load_statsbomb_priors()
    trans_matrix = build_markov_transition_matrix(priors)
    
    test_frames = {}
    for g in test_games:
        tdf = all_data[all_data['game'] == g].copy().reset_index(drop=True)
        p_lgb = np.mean([m.predict_proba(tdf[feature_cols]) for m in lgb_models], axis=0)
        p_xgb = xgb_clf.predict_proba(tdf[feature_cols])
        p_blend = 0.75 * p_lgb + 0.25 * p_xgb
        test_frames[g] = {'y_true': tdf['label'].map(label_map).values, 'p_blend': p_blend}
        
    for tau in [0.0, 0.02, 0.05, 0.10]:
        for min_turnover in [0, 25, 45, 65]:
            all_true, all_pred = [], []
            for g in test_games:
                y_true = test_frames[g]['y_true']
                p_blend = test_frames[g]['p_blend']
                if tau == 0.0:
                    pred = np.argmax(p_blend, axis=1)
                else:
                    pred = viterbi_decode(p_blend, trans_matrix, trans_scale=tau)
                if min_turnover > 0:
                    pred = apply_sandwich_hysteresis(pred, min_turnover_len=min_turnover)
                all_true.extend(y_true)
                all_pred.extend(pred)
            acc = accuracy_score(all_true, all_pred)
            f1 = f1_score(all_true, all_pred, average='macro')
            print(f"tau={tau:.2f}, min_turnover={min_turnover:2d} -> Acc={acc:.2%}, Macro F1={f1:.4f}")

if __name__ == '__main__':
    main()
