"""
Test Proper Tactical Alignment on Unseen Matches
Verifies the fix where ALL home/away features are consistently aligned.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import lightgbm as lgb

sys.path.insert(0, '.')
from starter.generate_full_submission import add_directional_features
from starter.external_cv.tune_on_external_cv import add_advanced_features

def swap_team_perspective(df, mask):
    """
    Completely and symmetrically swaps Home and Away features for selected games/rows.
    """
    df = df.copy()
    
    # 1. Labels
    lbls = df.loc[mask, 'label'].copy()
    df.loc[mask & (lbls == 'H'), 'label'] = 'TEMP_A'
    df.loc[mask & (lbls == 'A'), 'label'] = 'H'
    df.loc[mask & (df['label'] == 'TEMP_A'), 'label'] = 'A'
    
    # 2. Distance and proximity pairs
    paired_cols = [
        ('d_min_home', 'd_min_away'),
        ('home_within_3m', 'away_within_3m'),
        ('home_active', 'away_active'),
        ('home_centroid_x', 'away_centroid_x'),
        ('home_centroid_y', 'away_centroid_y'),
        ('home_spread_x', 'away_spread_x'),
        ('home_spread_y', 'away_spread_y'),
        ('mean_speed_home', 'mean_speed_away'),
    ]
    for c_h, c_a in paired_cols:
        if c_h in df.columns and c_a in df.columns:
            temp = df.loc[mask, c_h].copy()
            df.loc[mask, c_h] = df.loc[mask, c_a]
            df.loc[mask, c_a] = temp
            
    # 3. Differentials and ratios
    diff_cols = ['d_margin', 'd_margin_roll_1s', 'd_margin_roll_3s', 'centroid_diff_x']
    for c in diff_cols:
        if c in df.columns:
            df.loc[mask, c] = -df.loc[mask, c]
            
    if 'spread_ratio' in df.columns:
        df.loc[mask, 'spread_ratio'] = 1.0 / np.clip(df.loc[mask, 'spread_ratio'], 1e-4, 1e4)
        
    return df

def run_test():
    print("="*90)
    print(" TESTING PROPER FULL FEATURE ALIGNMENT ON UNSEEN MATCHES")
    print("="*90)
    
    # Load raw features
    df_met = pd.read_csv('data/metrica_3games_features.csv')
    df_sc4 = pd.read_csv('data/skillcorner_4matches_features.csv')
    
    # Swap before adding derived features
    mask_3442 = df_sc4['game'] == 'SkillCorner_RealMadrid_Barca'
    mask_2440 = df_sc4['game'] == 'SkillCorner_Liverpool_ManCity'
    df_sc4 = swap_team_perspective(df_sc4, mask_3442 | mask_2440)
    
    # Now add derived features consistently
    df_met = add_advanced_features(add_directional_features(df_met))
    df_sc4 = add_advanced_features(add_directional_features(df_sc4))
    
    all_data = pd.concat([df_met, df_sc4], ignore_index=True)
    
    train_games = [
        'SkillCorner_Barca_RealMadrid',
        'SkillCorner_ManCity_Liverpool',
        'Metrica_1',
        'Metrica_2'
    ]
    test_games = ['SkillCorner_RealMadrid_Barca', 'SkillCorner_Liverpool_ManCity', 'Metrica_3']
    
    drop_uninformative = ['video_frame', 'frame_id', 'label', 'game',
                          'home_attack_direction', 'ball_missing', 'overall_speed', 'speed_all_roll_2s']
    feats = [c for c in all_data.columns if c not in drop_uninformative]
    label_map = {'H': 0, 'A': 1, 'D': 2}
    
    train_df = all_data[all_data['game'].isin(train_games)].copy().reset_index(drop=True)
    clf = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.038, num_leaves=42, colsample_bytree=0.75, random_state=42, n_jobs=-1, verbose=-1)
    clf.fit(train_df[feats], train_df['label'].map(label_map).values)
    
    overall_y_true = []
    overall_y_pred = []
    
    for g in test_games:
        test_df = all_data[all_data['game'] == g].copy().reset_index(drop=True)
        y_true = test_df['label'].map(label_map).values
        probs = clf.predict_proba(test_df[feats])
        y_pred = np.argmax(probs, axis=1)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"[{g}]: Acc={acc:.2%}, Macro F1={f1:.4f}")
        overall_y_true.extend(y_true)
        overall_y_pred.extend(y_pred)
        
    total_acc = accuracy_score(overall_y_true, overall_y_pred)
    total_f1 = f1_score(overall_y_true, overall_y_pred, average='macro')
    print("="*90)
    print(f"PROPERLY ALIGNED OVERALL ACCURACY (165,643 frames): {total_acc:.2%}")
    print(f"PROPERLY ALIGNED OVERALL MACRO F1: {total_f1:.4f}")
    print("="*90)

if __name__ == '__main__':
    run_test()
