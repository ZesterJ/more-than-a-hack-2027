# Experiment B: external-only temporal consistency

Fixed before execution. No new binary classifier, classifier parameter changes or production inference changes.
Use the seven IDSSE matches and the exact PR #11 recipe from Experiment A: 78 features, control_s=3,
training mirrors, stride 6, no challenge occlusion, no domain weighting. P(DEAD)=P(D); P(LIVE)=P(H)+P(A).

## Frozen candidates and selection

22 candidates in total, with no combined families:

- Raw: P(DEAD)>=0.5.
- 15 non-identity asymmetric run filters from DEAD minima {0,3,5,9} and LIVE minima {0,3,5,10} frames.
  Zero disables a pass. Replace strictly shorter internal runs only. Fill LIVE gaps bracketed by DEAD first,
  then remove DEAD bursts bracketed by LIVE. Each pass reads a snapshot; no iterative convergence.
  Processing-span edge runs are preserved. Identity (0,0) is raw, counted once.
- Centered mean probability windows {3,5,9}, with truncated windows at processing edges; threshold at 0.5.
- Hysteresis (exit,enter)={(0.4,0.6),(0.3,0.7),(0.2,0.8)}. Initialise from raw 0.5 decision per span;
  exit when p<=exit, enter when p>=enter. No debounce or smoothing combined with hysteresis.

Select by equal-weight mean inner-match dead-interval F1, itself averaged over IoUs 0.3–0.7.
Eligibility requires absolute difference in mean inner macro-F1 from raw <=0.001 (inclusive, either direction).
Effective ties are within 1e-6 of the best eligible interval score. Among ties prefer raw, then run filters
ordered by (sum of thresholds, maximum threshold, DEAD threshold, LIVE threshold), then smoothing windows
ascending, then hysteresis bands narrowest first. This total ordering is explicitly saved before training.
Raw is always eligible. Separate family comparisons also select solely on inner folds, with raw fallback.

## Strict nesting and leakage controls

For each outer match M, inner predictions for each other match N come from training on the five matches
excluding both M and N. A pair-excluded model can predict both excluded matches: 21 fits, 42 predictions.
The fit function receives only the five training dictionaries, including only those training mirrors.
Never use Experiment A's other six OOF arrays as inner predictions: their models saw M.
Select all outer configurations before computing outer processed metrics. Outer probabilities are reused
from Experiment A only after its checksums, input/source lineage and exact raw-score parity pass.
All classifier feature/model code must match A; only its metric wrapper is refactored for hard decisions.

Transforms accept probabilities, period IDs and coverage masks, not reference state identities. All current
external frames have coverage. Invalid training labels fail closed rather than silently deriving an inference
policy from labels. Unit tests exercise explicit coverage gaps. Buffers, state and run operations reset at
period/coverage barriers. Scoring and matching reset at invalid-reference spans as well.
Feature construction retains A's offline future context and original concatenated-half behavior; this is
not a streaming evaluation or a new feature representation.

These seven matches informed the design through A diagnostics. Nesting prevents outer-to-inner training/
selection leakage for the frozen search, but new external matches are needed for untouched confirmation.

## Metrics and artifacts

Raw and processed metrics are separate. Reuse A's evaluator with hard binary decisions; do not turn hard
labels into artificial probabilities. Report frame PRF, macro-F1, accuracy, confusion/support, dead interval
PRF at every IoU, counts, duration distributions, onset/restart PRF and signed/absolute timing errors at
12/25 frames. Use equal-weight match statistics and separately pooled locally matched counts.

Diagnostics count internal false-DEAD bursts, internal false-LIVE gaps, split reference runs, completely
removed genuine states and short (<25 frame) genuine states. "Removed" means no correct frame remains
within the reference state; this is distinct from failing interval IoU. Report raw and processed counts and
newly removed short states separately, so pre-existing misses are not attributed to processing.

Locally save frozen_protocol.json before fitting, manifest/source snapshots, 42 inner H/A/D probability
arrays and their excluded/training IDs, inner_splits.json, candidate inner_scores.json, selections.json,
raw/processed and independent family metrics, per-match changes/CSV and processed labels. Keep all output
under data/experiment_b, which is Git-ignored. Snapshot and hash the actual environment and inputs.

## Commands

```sh
/opt/anaconda3/envs/football-ml/bin/python -m pytest tests/test_live_dead_baseline.py tests/test_live_dead_temporal.py -v
/opt/anaconda3/envs/football-ml/bin/python starter/evaluate_live_dead_temporal.py --output data/experiment_b/reproduction_001
```

Use a new output directory. No downloads or installations occur. The runner verifies the completed A
artifacts from data/experiment_a/football_ml_external_run_001 by default. On failure, manifest status and
failure.txt record the exception; partial outputs are not a complete experiment.

Expected failure modes: real brief states may be removed, long false bursts may survive, smoothing may
shift boundaries, hysteresis may latch onto confident spikes. Do not infer challenge-domain performance
from these external results. No dedicated binary-classifier experiment is authorised by this workflow.

## Completed nested validation

Completed run: `data/experiment_b/nested_run_001` (local, Git-ignored). Environment matches the completed
A run: Python 3.12.16, NumPy 2.5.4, pandas 3.1.1, scikit-learn 1.5.1 and LightGBM 4.7.0.
All 21 inner fits completed, producing 42 probability arrays. Every cache has five training matches
disjoint from its two excluded matches. All seven raw score dictionaries exactly match A.

Every outer fold independently selected **DEAD minimum 9 frames, LIVE-gap minimum 0**. This removes
internal DEAD runs of 1–8 frames (strictly under 0.36 seconds at 25 fps), without LIVE-gap filling.
All selected mean inner macro-F1 changes were +0.000334 to +0.000441, within the frozen guardrail.
All 33 tests passed at execution. The current expanded review suite is documented in the team summary.

| Outer match | Selected configuration | A interval F1@0.5 | B interval F1@0.5 |
|---|---|---:|---:|
| J03WMX | DEAD 9 / LIVE 0 | 0.891775 | 1.000000 |
| J03WN1 | DEAD 9 / LIVE 0 | 0.858333 | 0.980952 |
| J03WOH | DEAD 9 / LIVE 0 | 0.857143 | 0.990291 |
| J03WOY | DEAD 9 / LIVE 0 | 0.848485 | 0.982456 |
| J03WPY | DEAD 9 / LIVE 0 | 0.873874 | 0.989796 |
| J03WQQ | DEAD 9 / LIVE 0 | 0.752852 | 0.994975 |
| J03WR9 | DEAD 9 / LIVE 0 | 0.702290 | 0.989247 |

| Pooled measure | Raw A | Nested B |
|---|---:|---:|
| Dead frame F1 | 0.998508 | 0.998983 |
| Live frame F1 | 0.998786 | 0.999173 |
| Binary accuracy | 0.998662 | 0.999087 |
| Binary macro-F1 | 0.998647 | 0.999078 |
| Dead interval precision@0.5 | 0.703078 | 0.986072 |
| Dead interval recall@0.5 | 0.992987 | 0.992987 |
| Dead interval F1@0.5 | 0.823256 | 0.989518 |
| Predicted dead count (reference 713) | 1,007 | 718 |
| Internal false-DEAD bursts | 275 | 3 |
| Internal false-LIVE gaps | 14 | 5 |

Equal-weight match mean interval F1 across IoUs: **0.825917 +/- 0.070225 -> 0.989130 +/- 0.007554**.
The filter corrects 437 false-DEAD frames while losing 10 correct-DEAD frames. It newly completely
removes zero genuine short states; three DEAD and two LIVE short states were already missed by raw A.
The drop in counted LIVE gaps is not explicit gap repair: deleting tiny correct-DEAD islands can absorb
those gaps into boundary errors. At 12-frame onset tolerance, matched onsets decrease from 704 to 703;
at 25 frames, the matched count remains 706. Mean onset absolute error at 25 frames: 44.3 -> 42.1 ms.

Separate nested family comparisons: run-filter interval F1@0.5 **0.989518**, smoothing **0.979253**,
hysteresis **0.899048**. Smoothing slightly improves frame macro-F1 further but has lower interval F1.

## Review and reproducibility notes

The nested runner verifies unchanged classifier/scorer/dependency source hashes, checked A manifests,
metrics and posterior checksums, and exact raw-score replay. Documentation and test maintenance do not
invalidate historical A artifacts. Completed A and B runners now automatically write artifact checksums;
this removes the manual checksum step needed by the original execution. Historical run snapshots remain
immutable; these review-time changes do not alter cached predictions or measured results.

The selected filter is supported as an optional offline external-baseline postprocessor. Production
submission code has not been changed. Missing-ball observations were not simulated, so robustness under
ball missingness and challenge-domain transfer is unmeasured. Existing future context and concatenated-half
feature behavior remain unchanged; only new processing/scoring respects explicit period/coverage barriers.
These seven matches informed the design through A diagnostics, so new external matches are needed for
untouched confirmation. Experiment C and a dedicated binary classifier have not begun.

See [the team review summary](live_dead_review.md) for test scope, production parity, paired results and
artifact handling. Full local metrics retain every IoU, boundary timing statistic and duration distribution.
