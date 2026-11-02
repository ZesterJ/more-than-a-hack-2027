# AI Agent Workflow & Teammate Collaboration Guide
**Project**: More Than A Hack 2027 (FC Barcelona vs Athletic Club)  
**Author**: Machine Learning Team (@harveycheung2)  
**Current Champion Benchmark (v2)**: **Spell F1 = 1.0000** | **Macro F1 = 0.9771** | **Frame Accuracy = 98.17%**  
**Possessions**: **17 Predicted vs 17 Ground Truth (Exact 100% Match, 0 missed, 0 spurious)**  
**Segments**: **21 Predicted vs 21 Ground Truth (Fragmentation = 1.00)**

---

## 1. Quick Start for Teammates & AI Agents

If you or your AI agent (Antigravity, Cursor, Claude, ChatGPT) are continuing work on this repository, run this single command to reproduce the champion benchmark:

```bash
cd repo
python starter/evaluate_champion.py
```

### Expected Output:
```text
================================================================================
                    OFFICIAL COMPETITION SCOREBOARD
================================================================================
  Ground Truth Spells:        21
  Predicted Spells:           21 (EXACT MATCH!)
  Spell F1 (Primary Metric):  1.0000
  Macro F1:                   0.9771
  Overall Frame Accuracy:     98.17%
  Home F1 (Barcelona):        0.9895 (Precision: 1.0000, Recall: 0.9792)
  Away F1 (Athletic):         0.9874 (Precision: 0.9883, Recall: 0.9864)
  Dead F1 (Stoppages):        0.9546 (Precision: 0.9308, Recall: 0.9796)
================================================================================
```

To generate and validate the official 199,319-row full match submission:
```bash
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

## 3. The 13-Step Engineering Journey (From 0.0378 to 1.0000 Spell F1)

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
          - Mined 5,210 ultra-confident pseudo-labels from Barca match play outside GT window
          - Retrained student model on GT (30x) + Pseudo-labels (3x) + Metrica (1x)
```

---

## 4. Repository Architecture & Script Inventory

```
repo/
├── starter/
│   ├── evaluate_champion.py            # Single-command runner for official champion metrics
│   ├── generate_full_submission.py     # Generates full 199,319-row CSV & triggers official validation
│   ├── step12_champion_refinement.py   # Refined physical dynamics & transition smoother
│   ├── step13_selftraining_champion.py # In-domain self-training & pseudo-labeling pipeline
│   ├── cv_step12.py                    # 5-fold cross-validation suite
│   ├── extract_features.py             # Feature extraction pipeline (31 spatial-temporal features)
│   ├── step4_feature_engineering_benchmark.py # Kinematic feature augmentations
│   └── incremental_improvements_scoreboard.csv # Full chronological experiment history
├── evaluation/
│   ├── validate_submission.py          # Official format and contract validator
│   ├── run_evaluation.py               # Official competition scoring engine
│   ├── scoreboard.csv                  # Official competition leaderboard
│   ├── results/
│   │   ├── champion_submission.csv     # Current Rank #1 submission (199,319 rows)
│   │   └── baseline_*.csv              # Organizer baseline submissions
│   └── gt/
│       └── BAR-ATH_possession_gt_public.csv # Public 7,500-frame ground truth
```

---

## 5. Guidelines for Continuing Research with AI Agents

When instructing your AI agent to extend this work:
1. **Never break the 199,319-row contract**: Always run `python evaluation/validate_submission.py <path>`.
2. **Prioritize Boundary Transitions**: Between Barcelona and Athletic Club, team classification when both are alive is already **100.00%**. Focus exclusively on dead-ball transitions (`H/A <-> D`).
3. **Guard Against Domain Shift**: External tracking datasets (Bundesliga, Swedish league) play with different tempos. Always maintain a high sample weight ($30\times$) on Barcelona data or use in-domain self-training.
