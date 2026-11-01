# AI Agent Workflow & Teammate Collaboration Guide
**Project**: More Than A Hack 2027 (FC Barcelona vs Athletic Club)  
**Author**: Machine Learning Team (@harveycheung2)  
**Current Champion Benchmark**: **Spell F1 = 0.8848** | **Macro F1 = 0.9247** | **Frame Accuracy = 93.85%**  
**Spell Count**: **21 Predicted Spells vs 21 Ground Truth Spells (Exact 100% Match)**

---

## 1. Quick Start for Teammates & AI Agents

If you or your AI agent (Antigravity, Cursor, Claude, ChatGPT) are continuing work on this repository, run this single command to reproduce the champion benchmark:

```bash
cd repo
python starter/evaluate_champion.py
```

### Expected Output:
```
================================================================================
                    OFFICIAL COMPETITION SCOREBOARD
================================================================================
  Ground Truth Possessions:   21
  Predicted Possessions:      21 (EXACT MATCH!)
  Spell F1 (Primary Metric):  0.8848
  Macro F1:                   0.9247
  Overall Frame Accuracy:     93.85%
  Home F1 (Barcelona):        0.9694 (Precision: 99.06%, Recall: 94.90%)
  Away F1 (Athletic Club):    0.9445 (Precision: 94.07%, Recall: 94.84%)
  Dead F1 (Stoppages):        0.8602 (Precision: 82.67%, Recall: 89.65%)
================================================================================
```

---

## 2. Core Competition Context & Rules

1. **The Task**: Classify ball possession for every single video frame of the match at 25 frames per second into:
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

## 3. Repository Architecture & Script Inventory

```
repo/
├── evaluation/
│   ├── possession_metrics.py     # Official competition evaluator (spellF1, macroF1, IoU)
│   ├── validate_submission.py    # Official submission format validator (199,319 rows)
│   └── results/                  # Generated submission CSVs
├── data/                         # Gated data (gitignored)
│   ├── features_public_window.csv        # 7,500 frames of Barcelona ground truth
│   ├── features_first_15min_play.csv     # 22,500 frames (7.5k GT + 15k unlabeled)
│   ├── metrica_3games_features.csv       # 90,000 frames from Metrica Games 1, 2, 3
│   └── idsse_7matches_full_features.csv  # 553,105 frames from 7 DFL Bundesliga matches
└── starter/
    ├── evaluate_champion.py              # Official Champion Pipeline (Spell F1 = 0.8848)
    ├── benchmark_5fold_cv.py             # 5-fold contiguous out-of-sample cross validation
    ├── step1_hierarchical_ensemble.py     # Step 1: Dead gate + Live router
    ├── step2_temporal_dynamic_programming.py # Step 2: Minimum duration & Viterbi filter
    ├── step3_two_stage_transfer_learning.py # Step 3: Domain-adapted 30x sample weighting
    ├── step4_feature_engineering_benchmark.py # Step 4: 10 physical kinematics features
    ├── step5_cnn_gbdt_blend_and_hysteresis.py # Step 5: GBDT + 1D-CNN probability blend
    ├── step6_pseudo_labeling_benchmark.py # Step 6: Semi-supervised pseudo-labeling experiment
    ├── step7_multiscale_inception_benchmark.py # Step 7: Multi-scale temporal Inception (k=3,9,25,51)
    ├── step8_hyperparameter_calibration.py # Step 8: Joint grid calibration
    ├── step9_physics_tactical_rules.py    # Step 9: In-flight inertia & restart lock
    ├── step10_breakthrough_physics.py    # Step 10: Calibrated aerial thresholds
    ├── step11_perfect_alignment.py       # Step 11: Asymmetric state smoother
    ├── test_contact_filter.py            # Step 11b: Contact-conditioned duration filter
    └── incremental_improvements_scoreboard.csv # Historical progression tracking
```

---

## 4. The 8 Critical Learnings (What Worked & What Failed)

### What Worked:
1. **Contact-Conditioned Duration Filtering (`step11` / `evaluate_champion.py`)**:
   - Ground truth contains real 15–35 frame ($0.6\text{s}\text{--}1.4\text{s}$) interceptions.
   - Blunt static filters ($d \ge 50$) erased them. The contact filter checks if $d_{\text{min}} < 1.1\text{m}$. If physical contact occurred, it preserves the short turnover; if unconfirmed, it merges it. This jumped Spell F1 from **$0.7429 \to 0.8848$** and gave an exact **21/21 spell match**.
2. **Domain-Adapted Sample Weighting (`step3`)**:
   - Equal concatenation of external data diluted Barcelona’s specific optical camera calibration.
   - Giving **$30\times$ sample weight** to Barcelona ground-truth frames while using 90k Metrica frames for regularization boosted Spell F1 from **$0.51 \to 0.68$**.
3. **In-Flight Aerial Pass Inertia (`step9`, `step10`)**:
   - When the ball is kicked high in the air, optical trackers lose it (`ball_missing = 1`, `d_min = NaN`).
   - If collective player speed is high ($\text{speed} > 1.2\text{m/s}$), players are running to receive the pass. The state is **NOT Dead**; it retains the passing team's possession.
4. **Multi-Scale Inception Temporal CNN (`step7`)**:
   - Parallel temporal kernels ($k=3, 9, 25, 51$) enabled the neural network to perceive instant touches ($0.12$s) and sustained possession phases ($2.04$s) simultaneously.

### What Failed (Do NOT Repeat):
1. **Uncontrolled Pseudo-Labeling (`step6`)**:
   - Generating pseudo-labels on the 15,000 unlabeled Barcelona frames harvested $85\%$ Dead balls ($9,041$ Dead vs $78$ Away). Retraining on this skewed data caused **class collapse**, dropping Spell F1 from $0.74 \to 0.51$.
2. **Zero-Shot Neural Transfer Without Tempo Normalization**:
   - German Bundesliga data has fast vertical counter-attacks. Evaluating a pure Bundesliga-trained CNN on Barcelona's patient tiki-taka caused the CNN to over-segment spells (predicting 31 spells instead of 17).
3. **Nearest-Player Heuristics (`baseline.py`)**:
   - Assigning possession to whoever is closest to the ball produces chaotic flickering, scoring only **$0.0378$ Spell F1**.

---

## 5. How Teammates & AI Agents Can Push Past 0.8848

Here are the highest-probability avenues for further score gains:

### Avenue A: Acceleration Peak Boundary Snapping
- **Hypothesis**: In football, transitions occur on physical ball strikes (kicks or tackles).
- **Implementation**: After the model predicts a transition at frame $t$, search within a window of $\pm 5$ frames for the local maximum of ball acceleration $\max \|\vec{a}_{\text{ball}}\|$. Snap the transition boundary to that exact contact frame.
- **Goal**: Tighten $0.85$ IoUs to $>0.95$ IoUs.

### Avenue B: Full-Match 90-Minute Production Inference
- Run `evaluate_champion.py` across all 199,319 frames of the match video.
- Enforce the official half boundaries from `games.py`:
  - Half 1: frames `11,185` to `90,651`
  - Halftime: frames `90,652` to `110,807` $\to$ set to `X`
  - Half 2: frames `110,808` to `190,197`
  - Pre-match & Post-match: set to `X`
- Validate the generated file with `python evaluation/validate_submission.py`.

---

## 6. Prompt Templates to Give to Your AI Agent

Copy and paste these exact prompts into Antigravity, Cursor, Claude, or ChatGPT:

### Prompt 1: Benchmark Verification
> *"Please inspect `starter/evaluate_champion.py` and run it via terminal. Verify that it reproduces the 0.8848 Spell F1 and 93.85% Accuracy scoreboard on the 7,500 ground-truth frames. Then explain how the contact-conditioned duration filter works."*

### Prompt 2: Testing Boundary Snapping
> *"Read `starter/evaluate_champion.py`. I want to implement peak acceleration boundary snapping: when a possession transition occurs, search $\pm 6$ frames for the peak ball acceleration norm and snap the transition to that exact frame. Evaluate whether this improves Spell F1 beyond 0.8848."*

### Prompt 3: Generating Full Submission
> *"Read `SUBMISSION.md` and `games.py`. Take the champion model from `starter/evaluate_champion.py` and write `starter/generate_full_submission.py` to stream the full 199,319 frames of the tactical video. Output `evaluation/results/ensemble_submission.csv` and validate it using `evaluation/validate_submission.py`."*
