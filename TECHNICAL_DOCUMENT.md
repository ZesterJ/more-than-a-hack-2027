# Technical document: ball possession from tracking, trained on external data only

## Approach in one paragraph
For every frame of the two halves we predict who has the ball (`H`, `A`) or that it is dead (`D`); frames outside the
halves are `X`. A LightGBM classifier reads 78 permutation-invariant features of the tracking (who is near the ball, how
the teams are shaped and moving, how the ball and the nearest player are moving, and the same signals 1 s and 3 s earlier
and later). Training uses mirrored copies of the matches and weights the external frames by how much they resemble this
match's tracking (inputs only). Its
per-frame probabilities are smoothed (25-frame window) and runs shorter than 3 s are absorbed into their neighbours.
**The model is trained, and every setting is chosen, on external data only**: the seven Bundesliga matches of the IDSSE
open dataset (Sportec/DFL, CC-BY 4.0). The provided match contributes only its tracking, as input.

## Assets used, and how
| Asset | Use |
|---|---|
| Provided tracking (ATD) of this match | Input at prediction time; also where the ball is unseen (41% of frames), copied onto the external matches so the model trains on the same kind of gaps. Labels never used. |
| IDSSE open data, 7 matches (CC-BY 4.0) | Training and tuning. Provider ball-status / possession labels. Pitch oriented so the home team attacks +x in every half. |
| Provided ground truth | **Not used for training, fitting, calibration or model selection.** Used only to report scores through the official evaluator (see "Disclosure"). |
| Smart Tagging, video | Not used by the submitted model. |

Tried and dropped: Metrica sample games (loader exists, unused: no stated licence); learned HMM decoder (0.637 against 0.644);
bidirectional GRU (0.580); 5-seed ensembles (no gain); LightGBM hyperparameter search (11 variants, none better than +0.006);
frame weighting near possession changes (inconsistent, 4 or 5 of 7 matches); the teammates' asymmetric duration filter and "sandwich" hysteresis (no gain once tuned externally); the
teammates' hand-calibrated physics override (its thresholds were set on the public ground truth).

## Possession rules implemented
Taken from CHALLENGE.md: possession changes only when a team gains *controlled* possession; touches, deflections and loose
balls do not change it, and possession "stays with the last team that had it". Dead ball is a stoppage, never a possession.
- IDSSE provider labels switch team on every touch (median spell 3.7 s). They are relabelled so that any live run under
  **3 s** joins the previous team (median 8.8 s). 3 s was fixed in advance from the challenge text ("a few seconds"),
  not tuned on any score. A short run fenced in by dead ball on both sides is kept.
- At prediction time the same idea is the minimum-duration filter (3 s, chosen by leave-one-match-out on IDSSE).

## Validation (external data only)
Leave-one-match-out: each IDSSE match is predicted by a model trained on the other six, and decoder settings are chosen on
the other six too. spellF1 against the relabelled provider labels (ball gaps as in this match):

| Step | spellF1 | matches improved |
|---|---|---|
| 30 collective / kinematic features | 0.567 | |
| + 20 multi-scale, dwell, shape | 0.591 | most |
| + 16 boundary (ball direction change, closing speed, pressure, +-1 s / 3 s context) | 0.628 | 7 of 7 (vs 50 features, 15-min gaps) |
| + 12 motion (player velocity direction, ball carried with player, momentum) | 0.658 | 7 of 7 |
| + mirrored training copies (top-bottom, and left-right with teams swapped) | 0.674 | 7 of 7 |
| + input-only domain weighting toward this match's tracking | 0.688 | 7 of 7 |

Per-match noise is about 0.035; each step improved most of the 7 matches. These scores are against provider-derived labels, so
they are for comparing methods, not absolute predictions of the competition score.

## Limitations
- **Dead ball on this match.** On the public window the final model labels only about 7% of frames dead (about 20% are), and
  its dead-ball F1 is about 0.4 against about 0.76 for the 50-feature model. We could not reproduce this on external data
  (including a test where the ball vanishes during stoppages) and did not tune it away on the ground truth. Cause unknown.
- **External gains did not carry to the public window.** The last three external improvements (+0.046 in total) left the
  public-window spellF1 at about 0.57 (and 0.43 for the version without domain weighting). Seventeen possessions cannot
  separate these, but the pattern points to a domain gap we cannot see on external data.
- **Under-segmentation.** On the public window it predicts 12 possessions against 17 true; the 3 s rule may merge too much.
- **Label mismatch.** IDSSE labels come from a provider's own definition, not this competition's.
- **Domain shift.** Different tracking system, league and tempo; the ball is unseen far more often here.
- Public-window scores are from 17 possessions (bootstrap intervals about 0.3 wide) and are not a reliable ranking signal.

## Disclosure
The official evaluator was run on the public window a handful of times to *report* scores for candidate exports. No model,
feature, weight or setting was selected by those results; selection used external leave-one-match-out only. Earlier
experiments in this repository trained on the public ground truth; that work was removed from the branch after the
organisers' clarification and none of it is in the submission. We used an AI coding assistant; small aggregate scores and,
early on, a few printed data extracts were shown to it, which the team is raising with the organisers.

## Reproduction
```bash
pip install -r requirements.txt -r requirements-starter.txt     # macOS: brew install libomp
python starter/external.py --fetch-idsse                        # once, 2.6 GB, CC-BY 4.0
python starter/generate_full_submission.py --out submission.csv # defaults = submitted configuration
python evaluation/validate_submission.py evaluation/results/submission.csv
python -m pytest tests                                          # includes a guard against touching the ground truth
```
Domain weighting needs this match's tracking features (inputs only). Python 3.12.7 (use 3.12: scikit-learn 1.5.1 has no wheels for 3.13), numpy 1.26.4, pandas 2.2.2, scikit-learn 1.5.1,
lightgbm 4.7.0. The run is deterministic: a fresh clone of the repository, with the data caches rebuilt from the raw files,
produced a byte-identical submission.
Notebook `starter/04_external_benchmark.ipynb` shows the validation and export step by step.
