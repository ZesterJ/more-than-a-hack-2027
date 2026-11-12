# Experiment harness

A fixed, honest way to compare possession methods, plus a path to a validated full-match file.

## Why

The only labels are 7,500 frames (17 possessions). Scoring a method on frames it trained on proves
nothing, and trying many variants against the same 17 possessions mostly finds noise: one possession
moves `spellF1` by about 6 points, and the best of N tries is optimistic by construction. The repo's
own baselines show it, since the smoothing window that wins on the public window loses on the full match.

So the harness makes the honest way the easy way:

- The labelled window is cut into **5 contiguous folds**. Each fold is predicted by a model that
  never saw it, with a **150-frame embargo** either side so neighbouring frames cannot leak.
- The out-of-fold predictions are joined and scored **once** with the official scorer
  (`evaluation/possession_metrics.py`), with a **90% bootstrap interval**.
- Every run is appended to `experiments/results.csv` with a **trial number**, so how many things you
  tried is on record, and it is printed next to the score.

Sanity check: `nearest window=50` reproduces the repository baseline exactly
(spellF1 0.3256, macroF1 0.6207), so the folds add no distortion.

## Setup

```bash
pip install numpy pandas scikit-learn     # scikit-learn only for the `logistic` example
```

Needs the match data in `data/` (see `DATA.md`). Run everything from the repository root. The first
run parses the 187 MB tracking file and caches the feature table in `data/` (gitignored).

## Use

```bash
python -m experiments.run --list
python -m experiments.run nearest window=50                  # the repo baseline, cross-validated
python -m experiments.run logistic window=50 min_dur=60      # a learned method
python -m experiments.run logistic window=50 --save-oof      # also writes evaluation/results/oof_logistic.csv,
                                                             #   which the analyzer (labeler/) can load
python -m experiments.run --show-log                         # every trial so far
python -m experiments.run logistic window=50 min_dur=60 --write-submission my_team.csv
```

`--write-submission` fits on **all** labelled frames, predicts all 199,319 frames, sets frames
outside the graded halves to `X`, sets the 2,934 second-half frames before kickoff to `D` (the FAQ
says the truth has them all dead), and runs the official validator on the result. Its score on the
labelled frames is in-sample; the cross-validated score printed above it is the honest estimate.

## Reading the output

```
logistic({'window': 50, 'min_dur': 60})  5-fold, embargo 150 frames, out-of-fold on the labelled window
  spellF1  0.4000   90% interval [0.2476, 0.5931]
  macroF1  0.7229   accuracy 74.5%   F1 H 0.787  A 0.751  D 0.630
  possessions: 12 predicted vs 17 true
```

- **The interval is the point.** Here it is about +/-0.17 wide, so a gap of 0.05 between two methods
  is not evidence of anything. Treat gaps under 0.04 as noise, and prefer changes you can explain.
- **Possession count** next to the truth's 17 shows flicker (too many) or over-smoothing (too few).
- **Trial count:** the more you try, the more the best score flatters you. The log makes that visible.

## Training on external matches

> The organisers' clarification forbids training, fitting or calibrating on the ground-truth possession
> data, and requires tuning on external data. `harness.py` / `run.py` score methods on the public window
> and log trials; **do not use their scores to choose between variants.** Use `lomo.py` (external matches
> only) to choose, `make_submission.py` to build the file, and the official scorer once at the end. See
> the first section of [FINDINGS.md](FINDINGS.md).


Seven open Bundesliga matches (DFL / IDSSE, CC BY 4.0) give about 1M labelled frames, where the
challenge gives 7,500. Their labels are built the way the challenge describes its own (dead where the
ball status is 0, else the team in possession). Results, caveats and what did not work are in
[FINDINGS.md](FINDINGS.md).

```bash
python -m experiments.download_idsse                       # ~2.6 GB into data/external/idsse (gitignored)
python -m experiments.tune_external                        # choose decoding settings, leave-one-match-out
python -m experiments.run gbdt 'ext=["J03WMX","J03WN1","J03WOH","J03WOY","J03WPY","J03WQQ","J03WR9"]' \
    copies=2 'drop=["ball_speed"]' label_min_dur=50 decode=viterbi switch_team=20 switch_dead=20
python -m experiments.stress_test gbdt <same parameters>   # score again with more of the ball hidden
```

Three things the code does that you would otherwise have to rediscover:

- **Barcelona-style gaps are replayed onto the clean external data** (`external.corrupt`), because the
  external ball is almost never missing and the Barcelona ball is missing on 41% of graded frames.
- **Ball-speed features are dropped** (`drop=['ball_speed']`): the Bundesliga data freezes the ball
  when it is dead, which teaches a cue that does not exist in the Barcelona tracking.
- **Positions are put in one frame** (home attacks +x, read from each kickoff) so sources and halves
  are comparable.

The harness also runs a **fold-order leak check** on every run: it reruns the folds in reverse order and
refuses to log a result that changes. It exists because a model cache once let a method train on the
folds it was being tested on.

## Add your own method

A method is any object with:

```python
fit(feats, labels, train_idx)   # learn from labelled frames in train_idx (may do nothing)
predict(feats) -> codes         # one code per frame of the WHOLE match: 0 H, 1 A, 2 D, 3 X
```

`feats` is one row per video frame (`experiments/features.py`: ball position and speed, nearest-player
distance per team, players within 3 m / 5 m, team centroids and spread, team speeds, 1-3 s rolling
means). `labels` is the public ground truth as codes; only read it at `train_idx`. Write a class in
`methods.py` and add it to `REGISTRY`. `postprocess.py` has the smoothing and minimum-duration
building blocks.

Rolling features use centred windows, which look at frames after the current one. That is fine here
because only the output file is scored.

## Limits

- 17 possessions cannot rank methods. The harness stops you fooling yourself; it cannot add data.
- Folds are inside the first five minutes only, so they never test on a different phase of the match
  (the second half, late-game stoppages). The full-match score stays unknown until results day.
- A method that trains on other matches should be tested the same way: held-out Barcelona folds.
