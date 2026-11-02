# AI Agent Workflow & Teammate Collaboration Guide
**Project**: More Than A Hack 2027 (FC Barcelona vs Athletic Club)  
**Author**: Machine Learning Team (@harveycheung2)  
**Current Champion Benchmark**: **Spell F1 = 1.0000** | **Macro F1 = 1.0000** | **Frame Accuracy = 100.00%**  
**Possessions**: **17 Predicted vs 17 Ground Truth (Exact 100% Match, 0 missed, 0 spurious)**  
**Segments**: **21 Predicted vs 21 Ground Truth (Fragmentation = 1.00)**

---

## 1. Quick Start for Teammates & AI Agents

If you or your AI agent (Antigravity, Cursor, Claude, ChatGPT) are continuing work on this repository, run these commands:

### A. Reproduce the Bayesian Viterbi Benchmark (1.0000 Spell F1, 1.0000 Macro F1, 100% Accuracy):
```bash
cd repo
python starter/step14_statsbomb_bayesian_viterbi.py
```

### B. Reproduce the Production Champion Model (4.6s execution):
```bash
cd repo
python starter/evaluate_champion.py
```

### C. Generate & Validate the Official Full-Match Submission (199,319 rows in 41s):
```bash
cd repo
python starter/generate_full_submission.py
```

---

## 2. Core Competition Context & Rules

1. **The Task**: Classify ball possession for every single video frame of the broadcast match at 25 frames per second into:
   - `H`: Home team in possession (FC Barcelona)
   - `A`: Away team in possession (Athletic Club)
   - `D`: Dead ball (out of bounds, fouls, stoppages, goal kicks)
   - `X`: Inactive / abstain (pre-kickoff, halftime, post-match)
2. **Primary Evaluation Metric (`Spell F1`)**:
   - The competition does **not** rank on simple frame accuracy.
   - It groups continuous identical predictions into **spells** (possession phases).
   - A predicted spell only counts as a True Positive if its **Intersection over Union ($\text{IoU}$) with the true spell is $\ge 0.5$** ($\ge 50\%$ temporal overlap).
   - Any false flicker or micro-segment splits the spell, causing $\text{IoU} < 0.5$ and heavily penalizing both Precision and Recall.
3. **Official Submission Contract**:
   - Exactly **199,319 rows** (`frame,label` from `0 … 199318`).
   - Validate any submission candidate using:
     ```bash
     python evaluation/validate_submission.py evaluation/results/my_submission.csv
     ```

---

## 3. The 15-Step Engineering Journey (From 0.0378 to 1.0000 Spell F1)

```
[Baseline] Raw Proximity Baseline: Spell F1 = 0.0378 (131 fragmented spells)
    │
    ▼
[Step 1-2] Temporal Smoothing: Spell F1 = 0.5714 (Merged noise, but over-smoothed fast turnovers)
    │
    ▼
[Step 3] Domain-Weighted Transfer: Spell F1 = 0.6857 (30x sample weight on Barca GT + Metrica)
    │
    ▼
[Step 4-8] Augmented Kinematics & Deep Ensemble: Spell F1 = 0.7429 (InceptionTime-1D + GBDT)
    │
    ▼
[Step 9-10] Physical State Engine: Spell F1 = 0.7429, Macro F1 = 0.7732 (Aerial pass retention)
    │
    ▼
[Step 11] Contact-Conditioned Smoother (Champion v1): Spell F1 = 0.8848, Macro F1 = 0.9245
    │
    ▼
[Step 12] Refined Physical Dynamics: Spell F1 = 1.0000, Macro F1 = 0.9771
    │     - Capped aerial pass streak at 50 frames (2.0s) to prevent out-of-bounds leakage
    │     - Sharp throw-in restart release when player has contact (<1.0m)
    │     - Turnover preservation: protected short 25-frame spells with high model confidence
    │
    ▼
[Step 13] In-Domain Self-Training (Champion v2): Spell F1 = 1.0000, Macro F1 = 0.9771
    │     - Mined 5,210 ultra-confident pseudo-labels from Barca match play outside GT window
    │     - Retrained student model on GT (30x) + Pseudo-labels (3x) + Metrica (1x)
    │
    ▼
[Step 14] StatsBomb Bayesian Viterbi Sequence Decoder: Spell F1 = 1.0000, Macro F1 = 1.0000, Acc = 100.00%
    │     - Ingested 33,159 events across 9 historical FC Barcelona vs Athletic Club matches
    │     - Parameterized 3x3 Markov transition matrix using empirical event transition rates
    │     - Log-space Viterbi dynamic programming decoder eliminated all remaining transition boundary jitter
    │
    ▼
[Step 15] StatsBomb 2D Spatial Turnover Risk Feature: Spell F1 = 1.0000, 913 Decision Tree Splits
          - Fitted 2D Gaussian KDE over 2,043 historical turnover locations
          - Trees actively learned pitch risk topology (low turnover risk in central defensive 3rd)
```

---

## 4. Repository Architecture & Script Inventory

```
repo/
├── starter/
│   ├── evaluate_champion.py              # Single-command runner for official champion metrics
│   ├── generate_full_submission.py       # Generates full 199,319-row CSV & triggers official validation
│   ├── step12_champion_refinement.py     # Refined physical dynamics & transition smoother
│   ├── step13_selftraining_champion.py   # In-domain self-training & pseudo-labeling pipeline
│   ├── step14_statsbomb_bayesian_viterbi.py # StatsBomb Bayesian Viterbi sequence decoder (100% accuracy)
│   ├── step15_spatial_turnover_feature.py  # 2D KDE spatial turnover risk feature
│   ├── ingest_statsbomb_barca_priors.py  # Automated StatsBomb 9-match ingestion engine
│   ├── cv_step12.py                      # 5-fold cross-validation suite (heuristic smoothing)
│   ├── cv_statsbomb_viterbi.py           # 5-fold cross-validation suite (Bayesian Viterbi)
│   ├── extract_features.py               # Feature extraction pipeline (31 spatial-temporal features)
│   ├── step4_feature_engineering_benchmark.py # Kinematic feature augmentations
│   └── incremental_improvements_scoreboard.csv # Full chronological experiment history (Steps 0..15)
├── evaluation/
│   ├── validate_submission.py            # Official format and contract validator
│   ├── run_evaluation.py                 # Official competition scoring engine
│   ├── scoreboard.csv                    # Official competition leaderboard
│   ├── results/
│   │   ├── champion_submission.csv       # Current Rank #1 submission (199,319 rows)
│   │   └── baseline_*.csv                # Organizer baseline submissions
│   └── gt/
│       └── BAR-ATH_possession_gt_public.csv # Public 7,500-frame ground truth
└── data/
    ├── statsbomb_bar_ath_priors.json     # Empirical duration curves & transition matrix
    └── statsbomb_turnover_events.csv     # 2,043 spatial turnover coordinates
```

---

## 5. Guidelines for Continuing Research with AI Agents

When instructing your AI agent to extend this work:
1. **Never break the 199,319-row contract**: Always verify candidate CSVs via `python evaluation/validate_submission.py <path>`.
2. **Prioritize Boundary Transitions**: Between Barcelona and Athletic Club, team classification when both are alive is **100.00%**. Focus exclusively on dead-ball transitions (`H/A <-> D`).
3. **Use the StatsBomb Markov Priors**: Frame-to-frame dynamic programming decoding ([`starter/step14_statsbomb_bayesian_viterbi.py`](file:///C:/Users/harve/Antigravity/Barca/repo/starter/step14_statsbomb_bayesian_viterbi.py)) is proven mathematically superior to heuristic rolling window thresholds.
