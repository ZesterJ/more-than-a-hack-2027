# Stage 1 Dead-Ball Detector (Combined Hierarchical Gate)

**More Than A Hack 2027: Tactical Ball Possession Prediction**  
*Combining PR #13 (5 Physical Pillars & Missing-Ball Resilience) + PR #14/16 (IDSSE 7-Match Corpus, 41% Ball-Gap Masking & Interval Temporal Filtering)*

---

## 1. Overview & Teammate Interface Contract

This directory contains the production **Stage 1 Dead-Ball Detector** designed for the hierarchical possession prediction pipeline:

```mermaid
flowchart LR
    A["Tracking Frame<br/>(25 FPS)"] --> B["Stage 1 Dead Detector<br/>(5 Physical Pillars + 572k Frames)"]
    B --> C["Smoothed Hysteresis (0.32/0.55)<br/>+ Duration Filter (50f/25f)"]
    C --> D{"is_dead?"}
    D -- "True (Dead)" --> E["Output: 'D'<br/>(Dead Ball)"]
    D -- "False (Live)" --> F["Stage 2 Possession Router<br/>(Teammates' Model)"]
    F --> G["Output: 'H' or 'A'"]
```

### Python Teammate Integration Example

```python
import numpy as np
from starter.stage1_dead_detector.dead_detector import Stage1DeadDetector

# 1. Initialize and load trained 5-booster ensemble (572,359 external frames)
detector = Stage1DeadDetector(model_dir='models/stage1_dead_detector')
detector.load()

# 2. Predict calibrated P(Dead) for every frame
prob_dead = detector.predict_proba(df_match)

# 3. Predict filtered boolean gate (True = Dead, False = Live)
# Defaults: smooth_win=15, hyst=(0.32, 0.55), min_dead_frames=50, min_live_frames=25
is_dead = detector.predict_gate(df_match)

# 4. Hand off to Stage 2 (Live frames only):
live_indices = ~is_dead
team_preds = np.empty(len(df_match), dtype=object)
team_preds[is_dead] = 'D'
team_preds[live_indices] = your_stage2_model.predict(X_team[live_indices])
```

---

## 2. Combined 11-Fold LOMO Benchmark (SkillCorner + IDSSE Clean & 41% Gap-Masked)

Evaluated via strict Leave-One-Match-Out (LOMO) cross-validation where the held-out match (both clean and 41% gap-masked copies) was **never seen during training**:

| Domain | Temporal Filter | ROC-AUC | Dead P | Dead R | Dead F1 | Binary Macro F1 | Dead Int F1@0.5 | Mean Int F1 (0.3–0.7) | False-DEAD Bursts |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SkillCorner** (~52% ball gaps) | `raw_0.45` | 0.9647 | 0.8898 | 0.8972 | 0.8934 | 0.9038 | 0.3575 | 0.3433 | 95.2 |
| **SkillCorner** (~52% ball gaps) | `PR14_d9_l0` | 0.9647 | 0.8942 | 0.8928 | 0.8934 | 0.9042 | 0.5353 | 0.5141 | 37.2 |
| **SkillCorner** (~52% ball gaps) | `PR13_d50_l25` | 0.9647 | 0.9080 | 0.8765 | 0.8918 | 0.9040 | 0.7118 | 0.6762 | **13.0** |
| **SkillCorner** (~52% ball gaps) | **`hyst_d50_l25`** | **0.9647** | 0.8917 | **0.8982** | **0.8947** | **0.9049** | **0.7355** | **0.7053** | 13.5 |
| **IDSSE Clean** (0% ball gaps) | `raw_0.45` | 0.9677 | 0.9698 | 0.7851 | 0.8624 | 0.8942 | 0.3840 | 0.3913 | 13.3 |
| **IDSSE Clean** (0% ball gaps) | `PR14_d9_l0` | 0.9677 | 0.9771 | 0.7833 | 0.8641 | 0.8955 | 0.6023 | 0.6148 | 2.0 |
| **IDSSE Clean** (0% ball gaps) | **`hyst_d50_l25`** | **0.9677** | 0.9821 | **0.7913** | **0.8713** | **0.9017** | **0.7649** | **0.7758** | **0.0** |
| **IDSSE 41%-Masked** (stress test) | `raw_0.45` | 0.9139 | 0.8854 | 0.7109 | 0.7839 | 0.8346 | 0.1165 | 0.1130 | 29.7 |
| **IDSSE 41%-Masked** (stress test) | `PR14_d9_l0` | 0.9139 | 0.8971 | 0.7061 | 0.7854 | 0.8362 | 0.1687 | 0.1642 | 11.1 |
| **IDSSE 41%-Masked** (stress test) | **`hyst_d50_l25`** | **0.9139** | 0.9292 | **0.7316** | **0.8121** | **0.8558** | **0.5228** | **0.5251** | **1.4** |

### Why `hyst_d50_l25` Wins Across Every Domain
- **Eliminates false bursts on noisy broadcast data**: Cuts internal false-DEAD bursts from `95.2` $\to$ `13.5` on SkillCorner (`37.2` with PR #14's 9-frame filter) and from `13.3` $\to$ **`0.0`** on clean IDSSE.
- **4.5x higher Interval F1 under 41% ball dropouts**: On `IDSSE_Masked41`, `hyst_d50_l25` jumps from `0.1165` (raw) and `0.1687` (PR #14) up to **`0.5228` Dead Interval F1@0.5** and **`0.8121` Dead Frame F1**.

---

## 3. Production Training Corpus (572,359 Frames, 100% External)

Trained via `python starter/stage1_dead_detector/train_stage1_detector.py` on:
1. **SkillCorner 4 European Matches** (`272,359` frames; real broadcast ~52% ball gaps)
2. **IDSSE 7 Bundesliga Matches (Clean)** (`105,000` frames; TRACAB optical tracking)
3. **IDSSE 7 Bundesliga Matches (41% Ball-Gap Masked)** (`105,000` frames; simulated broadcast dropouts)
4. **Metrica Sports 3 Matches** (`90,000` frames; optical exhibition tracking)

---

## 4. The 5 Physical Pillars of Dead-Ball Features (30 Features)

1. **Pitch Boundary Violations**: `dist_to_boundary`, `is_out_of_bounds`, `near_boundary_1m`, `boundary_roll_1s`.
2. **Whole-Pitch Collective Kinetics (Whistle Signature)**: `speed_all_roll_2s`, `pitch_decel_2s`, `is_collective_walking`, `walking_roll_2s`.
3. **Missing Ball Tracking Persistence**: `ball_missing`, `consecutive_missing`, `missing_over_1s`, `missing_over_3s`, `missing_ratio_roll_2s`, `missing_ratio_roll_5s`.
4. **Tactical Team Dispersion Expansion**: `home_spread_x/y`, `away_spread_x/y`, `total_spread_area`, `spread_area_roll_2s`.
5. **Player-to-Ball Physical Isolation & Coverage**: `min_dist_any`, `min_dist_roll_1s/3s/5s`, `ball_isolated_5m/10m`, `players_within_3m_total`, `players_3m_roll_1s`, `active_total`, `active_roll_2s`.
