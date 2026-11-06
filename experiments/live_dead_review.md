# Team review: external-only Live/Dead Experiments A and B

Experiment B's DEAD-only short-run filter improves pooled dead-interval F1@0.5 from **0.823256 to
0.989518**, with unchanged interval recall and a small improvement in frame F1. Every outer fold
independently selected the same configuration: remove internal predicted DEAD runs **strictly shorter
than 9 frames** (1–8 frames at 25 fps); LIVE-gap filling disabled. This is an experimental offline
postprocessor. Production submission inference remains unchanged.

## Scope and implementation

- `starter/live_dead_baseline.py`: fixed external three-class LightGBM LOMO runner, probability collapse,
  shared hard-label metrics, local manifests/posteriors/checksums.
- `starter/live_dead_temporal.py`: independent snapshot run filters, probability smoothing and hysteresis;
  coverage/period barriers, inner selection and fragmentation diagnostics.
- `starter/evaluate_live_dead_temporal.py`: strict nested runner, pair-excluded prediction caches,
  provenance/raw-parity checks and per-fold selections.
- `tests/test_live_dead_baseline.py`, `tests/test_live_dead_temporal.py`: correctness, fold isolation,
  boundary handling, artifact generation and provenance regressions.
- [Experiment A protocol/results](live_dead_baseline.md) and
  [Experiment B protocol/results](live_dead_temporal.md): full methodology and reproduction commands.

No new binary classifier, model hyperparameter changes, threshold search in A, challenge-domain
adaptation, production inference integration, or Experiment C work is included.

## Evaluation design

A trains the existing PR #11 three-class LightGBM on six of seven external IDSSE matches and predicts
the seventh. Use 78 features, controlled-possession labels at 3 seconds, training-only mirror
augmentation and every-sixth-frame fitting. P(LIVE)=P(H)+P(A); P(DEAD)=P(D); DEAD is positive at a fixed
inclusive 0.5 threshold. No smoothing, decoder, copied ball gaps or domain weights are used in A.

B freezes 22 independent candidates before execution. For each outer match, select only from inner
predictions trained on five matches excluding both the outer and inner validation match. Twenty-one
pair-excluded fits produce 42 predictions. The other six global A OOF arrays are never used for inner
selection. Reuse A's checked six-match OOF probabilities only for the final outer evaluation.

Select by equal-weight mean inner-match interval F1 across IoUs 0.3–0.7, with an absolute 0.001
macro-F1 guardrail and a 1e-6 effective-tie tolerance favoring raw/smaller interventions. All selected
inner macro-F1 changes were positive (+0.000334 to +0.000441). All configurations were saved before
processed outer evaluation. Matching and processing respect explicit period/coverage barriers.

## Measured results

All 1,002,644 frames below are external IDSSE frames. Full raw and processed results remain separate.

| Pooled measure | Raw A | Nested B |
|---|---:|---:|
| Dead precision | 0.997803 | 0.998773 |
| Dead recall | 0.999215 | 0.999193 |
| Dead F1 | 0.998508 | 0.998983 |
| Live precision | 0.999361 | 0.999343 |
| Live recall | 0.998212 | 0.999002 |
| Live F1 | 0.998786 | 0.999173 |
| Binary accuracy | 0.998662 | 0.999087 |
| Binary macro-F1 | 0.998647 | 0.999078 |
| Dead interval precision@0.5 | 0.703078 | 0.986072 |
| Dead interval recall@0.5 | 0.992987 | 0.992987 |
| Dead interval F1@0.5 | 0.823256 | 0.989518 |
| Predicted dead intervals (reference 713) | 1,007 | 718 |
| Internal false-DEAD bursts | 275 | 3 |
| Internal false-LIVE gaps | 14 | 5 |

Equal-weight match mean interval F1 over IoUs 0.3–0.7: **0.825917 +/- 0.070225 ->
0.989130 +/- 0.007554** (sample SD). This is distinct from the pooled interval F1@0.5 above.

| Outer match | Selected DEAD/LIVE minima | A interval F1@0.5 | B interval F1@0.5 |
|---|---|---:|---:|
| J03WMX | 9 / 0 | 0.891775 | 1.000000 |
| J03WN1 | 9 / 0 | 0.858333 | 0.980952 |
| J03WOH | 9 / 0 | 0.857143 | 0.990291 |
| J03WOY | 9 / 0 | 0.848485 | 0.982456 |
| J03WPY | 9 / 0 | 0.873874 | 0.989796 |
| J03WQQ | 9 / 0 | 0.752852 | 0.994975 |
| J03WR9 | 9 / 0 | 0.702290 | 0.989247 |

Separate nested family interval F1@0.5: run filtering **0.989518**, smoothing **0.979253**,
hysteresis **0.899048**. No combined temporal families were searched.

## Trade-offs and limitations

The selected filter corrects 437 false-DEAD frames and loses 10 correct-DEAD frames. No genuine short
state is newly completely removed; raw already missed three short DEAD and two short LIVE states.
Fewer counted LIVE gaps does not imply gap repair: tiny correct-DEAD islands can be absorbed into
boundary errors. At 12-frame tolerance, one fewer onset matches; at 25 frames the onset matched count
is unchanged. Mean onset absolute error at 25 frames improves from 44.3 to 42.1 ms.

The evidence supports an optional offline filter for this external baseline, not a competition-domain
accuracy claim. Missing-ball observations were not simulated and no challenge gaps were copied, so
robustness to substantial ball missingness is unmeasured. Provider labels and tracking may differ by
domain. Inherited centered/future features and concatenated-half feature calculations are unchanged;
this is not a causal streaming benchmark. A's diagnostic inspection informed B's candidate design,
so new external matches are needed for untouched confirmation beyond this frozen nested comparison.

## Verification and review fixes

Review command in the existing `football-ml` environment:

```sh
conda run -n football-ml python -m pytest tests --ignore=tests/test_scoring_parity.py -q
```

**64 passed, 0 failed.** The two tests in `test_scoring_parity.py` are intentionally excluded: they read
competition ground truth and/or tracking. All data-independent feature, external-loader, benchmark,
training-pipeline, tactical and A/B tests are included. Twelve pre-existing numerical/solver warnings
remain nonfatal; production code is unchanged.

Verified all tracked production files against PR #11 merge
`0c364e61e10a247832379d4b764f53d660bdc2d0`, including submission generation, feature engineering,
external training, tactical decoder, notebook, scorer and dependency files. They are byte-identical.
Raw A per-match/aggregate metrics and B selections, decisions and full metrics replay exactly from
saved external arrays. All historical B artifact checksums pass.

Review-time fixes do not change predictions, selection rules or measured results:

1. Replace stale blocked-run documentation with completed A/B results.
2. Check classifier/scorer/dependency source lineage independently of documentation/test hashes,
   so updating results prose does not block reuse of valid historical A artifacts. Exact raw replay
   remains mandatory. A regression test rejects classifier edits while allowing documentation edits.
3. Automatically generate completed-run artifact checksums in A and B; B verifies A manifest/metrics
   as well as OOF array hashes. Previously checksum generation needed a manual post-run step.

Environment used for measurements: Python 3.12.16, NumPy 2.5.4, pandas 3.1.1, scikit-learn 1.5.1,
LightGBM 4.7.0. Python/NumPy/pandas differ from the historical PR #11 submission environment; actual
versions, source snapshots, input hashes and package exports are preserved locally. Historical
snapshots are not rewritten by this review. New provenance/serialization behavior is covered by tests;
no model refit is necessary for these review-only fixes.

## Artifacts and branch status

Local, Git-ignored completed runs:

- `data/experiment_a/football_ml_external_run_001/`
- `data/experiment_b/nested_run_001/`

Each contains full metrics, manifest/provenance, local probability/decision arrays and diagnostics.
B also contains frozen rules, inner split lists, candidate scores and per-outer selections. Share only
aggregate external tables and source code in the pull request. Raw XML, caches, predictions, per-frame
labels, local environment exports and source snapshots stay under ignored `data/`.

The repository already tracks the public scorer reference and published baseline CSVs from before
this branch; those files are unchanged and were not opened for this review. No new restricted data,
credentials or frame-level outputs belong in this change.

Branch: `feat/live-dead-baseline`. Changes remain uncommitted pending user approval. No push or PR
creation has been performed. No blocking review issue remains after the fixes above; domain-shift,
missing-ball and streaming generalization remain explicitly unvalidated. Experiment C has not begun.
