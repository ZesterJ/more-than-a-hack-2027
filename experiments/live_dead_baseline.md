# Experiment A: external-only Live/Dead baseline

Evaluate the existing PR #11 three-class LightGBM as a binary Stage 1 gate.
No binary model is trained. Existing training, notebook and submission code remain unchanged.

## Fixed protocol

- Seven IDSSE matches: J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9.
- Seven leave-one-match-out folds: fit six matches, predict all frames of the seventh.
- 78 features: 22 base + 10 extra - ball_speed/ball_accel + 20 advanced + 16 boundary + 12 motion.
- Explicit advanced=True, boundary=True, motion=True, control_s=3, augment=True.
- Mirror augmentation enters fitting only for the six training matches. The held-out original is predicted once.
- PR #11 fitting stride: every sixth frame of each original/mirrored training array.
- PR #11 LightGBM: 150 estimators, learning_rate=0.06, num_leaves=15,
  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0,
  deterministic=True, force_row_wise=True, verbose=-1. Other library defaults are recorded by get_params().
- pattern=None, weights=None. No competition tracking, occlusion pattern, domain adaptation,
  video, smart tags, ground truth, submission generation or competition scoring is used.
- No smoothing, tactical decoder, threshold selection, calibration or hyperparameter search.
- Class probabilities are aligned to H/A/D using benchmark._full_proba (including missing-class columns).
  P(LIVE)=P(H)+P(A); P(DEAD)=P(D). DEAD is predicted when P(DEAD)>=0.5, including ties.
- Controlled-possession relabelling can change H/A training targets but leaves the binary reference unchanged.

This reuses the PR #11 feature/training recipe without its competition-target adaptation. It is
not an exact reproduction of the target-adapted submission. Tracking is normalized/oriented by the
existing loader. The feature builder retains existing pitch scaling and temporal behavior, including
centered/future context, backfilling, and temporal calculations on concatenated halves. This is an
offline baseline, not a causal streaming evaluation. Those inherited choices are not modified here.

## Reproduce

Use the existing project environment with NumPy, pandas, scikit-learn and LightGBM.
The documented PR #11 environment is Python 3.12.7, NumPy 1.26.4, pandas 2.2.2,
scikit-learn 1.5.1 and LightGBM 4.7.0. See requirements-starter.txt and its OpenMP note.
The runner installs nothing and downloads nothing. Python's unittest runs the synthetic tests;
pytest is optional for this new test file.

```sh
conda run -n football-ml python -m unittest discover -s tests -p test_live_dead_baseline.py -v
conda run -n football-ml python starter/live_dead_baseline.py --output data/experiment_a/run_001
```

The output directory must not already exist. The default output is a UTC-timestamped directory
under data/experiment_a, already ignored by Git. Keep prediction arrays local. Set MTAH_DATA_DIR
if external files are elsewhere; the only checked input location is
$MTAH_DATA_DIR/external/idsse (or data/external/idsse by default). The runner checks seven cache
NPZs or the corresponding position/match-info XML pairs; it never traverses the data root.
The existing external loader may create IDSSE caches when parsing XML.

Outputs on a successful run:

- manifest.json: source commit, branch/status, source hashes, UTC time, Python/platform,
  actual dependency versions, model parameters, fixed settings, fold membership, exact input paths/hashes.
  Source hashes identify uncommitted implementation code beyond the base commit.
- metrics.json: per-match results and separate equal-weight/pooled aggregates.
- per_match.csv: scalar metrics for each match, including counts and supports. Full matrices,
  histograms and duration/error arrays are in JSON.
- posteriors/<match>.npz: P_had (H/A/D columns), y (controlled external labels), period.
  Rows retain the external loader's concatenated frame order. There are no challenge predictions.
- artifact_checksums.json: hashes of all completed artifacts, generated automatically for Experiment B verification.

Preflight failure writes manifest.json and failure.json and exits 1. It does not generate fake
metrics or posteriors. A later failure retains available fold outputs, marks the manifest failed,
and leaves partial metrics marked running. Only status=complete is a complete seven-match run.

## Evaluation definitions

Frame confusion rows are reference and columns prediction, ordered [LIVE, DEAD]. DEAD is positive.
Report precision, recall, F1 for both classes, binary accuracy, macro-F1, confusion and supports.
Zero denominators return zero. Non-H/A/D references are excluded; predictions must be finite
normalized probabilities. The current training runner fails on invalid training labels, rather than
silently inventing labels; the standalone evaluator supports invalid reference gaps.

Dead intervals are maximal D runs in half-open [start, stop) coordinates. Every temporal operation
is performed independently within contiguous valid-reference spans of a single period and match.
Intervals clipped by these spans are evaluated as observed; this includes duration summaries.
Initial/final states of a span do not create artificial boundary events.

At IoU 0.3, 0.4, 0.5, 0.6, 0.7, use the existing scorer's deterministic descending-IoU greedy
one-to-one matching (ties by reference/prediction index). An unmatched prediction is FP; an
unmatched reference is FN. Report PRF and counts at every IoU, plus mean F1 over the five IoUs.
This is a dead-interval metric, distinct from the scorer's H/A possession spellF1.

Collapse H/A to LIVE before extracting boundaries. LIVE->DEAD is stoppage onset; DEAD->LIVE is
restart. Match separately per span and event type using the scorer's one-to-one maximum-cardinality
boundary matching at inclusive +/-12 and +/-25 frames (0.48 and 1 second at 25 fps). This matching
is deterministic, not necessarily minimum-latency. Report PRF, TP/FP/FN, signed errors
(predicted-reference; positive means late), mean/median signed error, mean absolute error and p90
absolute error in seconds. Timing summaries include matched boundaries only; no matches gives null.

Report reference/predicted interval count, count error, durations in seconds, total/mean/population
SD, min/p25/median/p75/p90/max and histogram bins [0,1,2,5,10,30,60,infinity). Durations are retained
locally in JSON so error analysis is reproducible.

Equal-weight aggregates give the mean and sample SD across matches for each scalar metric, with
n_matches indicating available contributors (undefined timing/duration statistics are omitted).
Pooled frame metrics are recomputed from summed confusion matrices, not averaged F1s. Pooled
interval/boundary counts are summed after local matching; timing errors/durations are then pooled.
No matching crosses matches. Do not use frame-independent confidence intervals for correlated frames.

## Validation and leakage controls

Synthetic tests cover probability normalization and class alignment, collapse-before-decision,
threshold ties, known confusion counts, absent classes/empty inputs, split/merged/missed intervals,
one-to-one matching, period/invalid-span isolation, signed timing and tolerance handling,
equal-weight versus pooled statistics, exact loader flags, and held-out original/augmentation exclusion.
A mocked seven-fold run tests artifact serialization under filesystem guards; a source-call guard
rejects competition/decoder entrypoints. A real synthetic 78-finite-feature test runs when
scikit-learn is importable. Its skip is explicit, not a successful feature verification.

All future tuning must be external-only. Reusing global out-of-fold scores to choose settings and
reporting the same scores is selection-biased. If Experiment B tunes thresholds or smoothing,
use nested match-level validation: for each outer held-out match, run inner validation on its six
training matches only. Do not feed the outer match into inner training, weighting or selection.

## Completed execution and measured results

Source base: `0c364e61e10a247832379d4b764f53d660bdc2d0` (PR #11), branch `feat/live-dead-baseline`.
Completed run: `data/experiment_a/football_ml_external_run_001` (local, Git-ignored).
Environment: Python 3.12.16, NumPy 2.5.4, pandas 3.1.1, scikit-learn 1.5.1, LightGBM 4.7.0.
These Python/NumPy/pandas versions differ from the historical submission environment above. Actual
versions, source/input hashes and environment exports are retained with the completed run.
All 14 external IDSSE XML files parsed successfully. All 18 A tests passed without skips in that environment.
Earlier missing-data/dependency failures are superseded by this completed run.

The seven OOF folds cover **1,002,644 frames**. Every saved per-match metric replays exactly from its
saved probabilities. The post-review metric wrapper also preserves these scores exactly.

| Match | Dead precision | Dead recall | Dead F1 | Live F1 | Dead interval F1@0.5 |
|---|---:|---:|---:|---:|---:|
| J03WMX | 0.998368 | 0.999307 | 0.998837 | 0.999174 | 0.891775 |
| J03WN1 | 0.998240 | 0.999617 | 0.998928 | 0.998851 | 0.858333 |
| J03WOH | 0.998194 | 0.999081 | 0.998637 | 0.998762 | 0.857143 |
| J03WOY | 0.998537 | 0.999049 | 0.998793 | 0.998951 | 0.848485 |
| J03WPY | 0.997841 | 0.999444 | 0.998642 | 0.998919 | 0.873874 |
| J03WQQ | 0.996371 | 0.998914 | 0.997641 | 0.998374 | 0.752852 |
| J03WR9 | 0.996836 | 0.999008 | 0.997921 | 0.998499 | 0.702290 |

Pooled DEAD precision/recall/F1: **0.997803 / 0.999215 / 0.998508**.
Pooled LIVE precision/recall/F1: **0.999361 / 0.998212 / 0.998786**.
Binary accuracy: **0.998662**; macro-F1: **0.998647**.
Equal-weight match dead F1: **0.998485 +/- 0.000499**; live F1: **0.998790 +/- 0.000275** (sample SD).
Equal-weight mean dead-interval F1 across IoUs: **0.825917 +/- 0.070225**.

At IoU 0.5, pooled dead-interval precision/recall/F1 is **0.703078 / 0.992987 / 0.823256**:
713 reference intervals, 1,007 predictions, 708 matches, 299 unmatched predictions and 5 missed references.
There are 275 internal false-DEAD bursts and 14 internal false-LIVE gaps; 285 unmatched dead intervals
lie wholly in true LIVE periods. These observations motivated [Experiment B](live_dead_temporal.md).

## Interpretation limits

The protocol does not simulate missing-ball observations and does not copy challenge tracking gaps.
Consequently these very high external frame scores do not demonstrate robustness under substantial
ball missingness or transfer to competition tracking. Provider possession/status definitions, tracking
characteristics and label timing may differ between domains. Existing future-context features make
this an offline evaluation. No competition tracking, video or ground truth was used for these results.

See [the team review summary](live_dead_review.md) for the paired A/B comparison, validation evidence,
and remaining limitations. Only aggregate external results belong in the review; frame-level arrays,
XML inputs and local source/environment snapshots remain under ignored `data/` directories.
