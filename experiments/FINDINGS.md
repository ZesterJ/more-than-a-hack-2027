# Findings: training on external matches

Everything below was measured with the harness (`python -m experiments.run`): 5 contiguous folds
over the 5 labelled minutes, a 150-frame embargo, the official scorer, a 90% bootstrap interval,
and a fold-order leak check on every run. 20 trials were logged; the best of 20 is optimistic.
17 possessions cannot separate results that sit inside each other's intervals.

## Status after the organisers' rules clarification (read this first)

The clarification bans training, fitting or **calibrating** on the ground-truth possession data, and
requires any tuning to be done on external data. The numbers further down this document were produced
by an earlier workflow that **scored variants on the public window and picked among them**, which is
calibrating on the ground truth. Treat the "Barcelona" columns and "trial" results below as history,
not as the basis of the submission. In particular these earlier choices were influenced by the window:
dropping the ball-speed features (diagnosed from the window, though it is also visible in the external
data alone), a 2 s possession minimum (tried on the window), and rejecting the touch/line features
(rejected because Barcelona got worse). Harvey's champion pipeline was trained and tuned on the ground
truth and must not be submitted.

**The current procedure** uses no ground truth at all:

1. Train only on the seven external Bundesliga matches.
2. Choose every setting with `python -m experiments.lomo`, which holds out one external match at a time
   (about 221 possessions per match) and scores it with the official scorer on that match's own labels.
3. Take the minimum possession length from the challenge text ("a player keeps the ball for a few
   seconds"), not from any data: 75 frames (3 s). This is a judgement call.
4. Freeze the configuration in `experiments/make_submission.py`, build the file, and run the official
   scorer **once** afterwards, reporting whatever it says.

**The frozen configuration**, chosen on held-out external matches (spellF1 0.629, 211 predicted
possessions against 221 true): the touch and line features on (`extras=True`), an average of three models
trained on differently gapped copies (`bags=3`), Viterbi decoding with switch costs 25 (team) and 40
(live to dead), ball-speed features dropped, 3 s label alignment.

What was compared on those external matches, so nobody repeats it:

| change | spellF1 | verdict |
|---|---|---|
| touch and line features (vs not) | 0.573 vs 0.541 | kept |
| two-stage model (live/dead, then team) | +0.003 | dropped, within noise |
| bigger, slower model (300 iterations) | 0.610 vs 0.611 | dropped, no gain |
| average of 3 models | 0.620 vs 0.611 | kept (+0.01 at every decoder setting) |
| separate cost for "play restarts" | within 0.004 | dropped |
| 6 s / 10 s window features | 0.628 vs 0.626 at the best decoder | dropped, no gain and fewer changes found |
| team-switch cost 20 -> 25-30 | 0.620 -> 0.629 | kept (25) |

The held-out external score has plateaued at about 0.63; the remaining weakness (finding team
turnovers, about 46% within one second) is limited by the evidence in the tracking, not by these knobs.

Note that this configuration includes the touch and line features, which hurt on the Barcelona window in
the earlier workflow. Under the rules that is not a reason to remove them: the window may not be used to
choose. It is a reason to expect the public-window score to be lower than the earlier 0.63, and it is
reported as such.

```bash
python -m experiments.download_idsse
python -m experiments.make_submission my_team.csv     # trains on external data only, writes and validates
python evaluation/run_evaluation.py                   # run once, after freezing
```

## Round 3: root cause of the missed turnovers, and the fix (supersedes the frozen configuration above)

Everything here uses external matches and the provided TRACKING only; no ground truth was read.

**1. The two sources really differ.** A classifier can tell Barcelona frames from gap-corrupted external
frames from tracking features alone (AUC 0.969). The control matters: one external match against the
other six is only AUC 0.70-0.91 (mean 0.77), so Barcelona is more different from the pool than any
external match is. The ball track is part of it: in Barcelona the ball is never still (median step
about 5 m/s, smooth) while in the Bundesliga data it is frozen for most dead-ball frames, and Barcelona
has 2-3x more teleport-like jumps.

**2. What hurts turnover detection (simulated on external matches).** Each suspected difference was
simulated and the model scored on it, so the cost is measured without Barcelona labels:

| held-out match with... | spellF1 | turnover detection (1 s) | predicted / true possessions |
|---|---|---|---|
| clean tracking | 0.618 | 0.450 | 1.01 |
| ball moving while dead | 0.602 | 0.413 | 0.98 |
| **ball lost around possession changes** | **0.568** | **0.349** | **0.93** |

When the tracker loses the ball exactly where possession changes hands, the turnover signal disappears
and the decoder merges possessions: the same symptom seen on Barcelona (too few possessions). This is a
hypothesis supported by simulation, not proven for Barcelona.

**3. The fix: evidence that does not need the ball.** After a change of possession the side that wins
the ball pushes up and the other drops back. Features that capture how team centroids, front lines,
width and speed MOVE over the previous/next 2 s and 6 s (`dyn_*`) do not depend on the ball.
Leave-one-match-out on the seven external matches:

| | spellF1 clean / turnover-gaps | turnover detection clean / turnover-gaps |
|---|---|---|
| without team dynamics | 0.618 / 0.568 | 0.450 / 0.349 |
| **with team dynamics** | **0.692 / 0.668** | **0.523 / 0.452** |
| with, trained on standard + shifted copies | 0.694 / 0.684 | 0.516 / 0.469 |

**4. The audit that caught an artefact.** Because the touch features had looked great on external data
and then failed on Barcelona, the new features were audited before use. The deepest-player series
(`dyn_*_back*`) separate Barcelona from the external pool with AUC **0.96 on their own** (one external
match against the rest: 0.60): a tracking artefact, probably an extra or stationary track, not football.
They contributed nothing to the gain (0.692 with and without) and are excluded
(`features.DYNAMICS_EXCLUDE`). The kept families (centroids 0.62, front lines 0.61, width 0.62, team
speed 0.53) sit near the external control (0.52-0.53). Run `experiments/domain_gap.py` on any new feature
group before trusting an external gain.

**5. New frozen configuration**, leave-one-match-out spellF1 about 0.69-0.70 across decoder settings
(0.694 at 25/40, 219 predicted vs 221 true possessions): touch/line features, team dynamics (back lines
excluded), training on standard plus shifted copies, an average of 3 models, Viterbi 25 (team) / 40
(live to dead), 3 s label alignment. `experiments/make_submission.py` holds it.

**Not yet known:** the Barcelona score of this configuration. The earlier lesson stands: a gain on
external matches can fail to transfer, and only one official-scorer look may be taken after freezing.
Switch costs 20-40 are within noise of each other; 25 was kept for the closest possession count.

## Integration branch: the frozen-ball shortcut, measured (external data only)

`starter/ablation_ball_motion.py` on the integration branch, leave-one-match-out over the seven IDSSE matches,
with the provided match's ball-gap pattern applied. In IDSSE the ball is frozen for most dead-ball frames; in the
provided tracking it never is, so features built from how the ball moves (`external_pipeline.BALL_MOTION`, 14
columns) teach a cue that does not exist there. The "moving" test lets the ball keep moving while dead
(`external.dead_ball_drift`).

| trained as | tested on frozen ball: dead recall / spellF1 | tested on moving ball: dead recall / spellF1 |
|---|---|---|
| A all features (the default before this) | 0.96 / 0.632 | **0.51 / 0.423** |
| B ball-motion features removed | 0.95 / 0.607 | 0.90 / 0.572 |
| C all features + moving-ball training copies | 0.96 / 0.621 | 0.95 / 0.608 |

The shortcut halves dead-ball recall when the ball keeps moving, which is the weakness the integration branch's
own PR reports on the provided match (recall 27.5%). Removing the features (B) costs about 0.025 spellF1 on a frozen
ball and recovers most of the loss on a moving one. C does best, but its moving-ball test comes from the same
simulator it trained on, so its edge is optimistic; B does not depend on the simulator at all. Untested here: whether the
simulator's ball path resembles the real one. See the next section for the combination with the team-movement features.

## Integration: physical + temporal features together (external data only)

`starter/ablation_integration.py`, leave-one-match-out over the seven IDSSE matches, four test versions of every held-out
match (as recorded; ball keeps moving while dead; ball lost around possession changes; both), plain 2 s smoothing decoder.
Average over the four versions:

| variant | spellF1 | turnover detection (1 s) | dead-ball recall | macro-F1 |
|---|---|---|---|---|
| V0 Jordan's pipeline as merged (109 features) | 0.516 | 0.356 | 0.736 | 0.787 |
| V1 the 14 ball-motion features removed | 0.577 | 0.414 | 0.924 | 0.860 |
| V2 V0 + 36 team-movement features (`dyn_*`) | 0.543 | 0.397 | 0.744 | 0.797 |
| V3 V1 + team-movement features | 0.617 | 0.468 | 0.933 | 0.875 |
| **V4 V3, also trained on moving-ball / turnover-gap copies** | **0.632** | **0.495** | **0.947** | **0.882** |

Reading it: V0 collapses when the ball keeps moving during dead frames (dead-ball recall 0.96 to 0.51; spellF1 0.632 to
0.423). The team-movement features raise clean-data and turnover scores on their own (V2: 0.632 to 0.661) but do not
remove the shortcut. Removing the ball-motion features fixes dead ball but costs clean-data accuracy (V1); with the
team-movement features that loss is recovered (V3), and V3 beats both parts on every hard condition. V4 adds a little more
on every hard condition. Caveats: all numbers are on external matches, the moving-ball and turnover-gap conditions are our
own simulators, and V4 trains on the same simulators, so its extra edge is optimistic; V3 does not depend on them.

**Defaults on the integration branch:** `external_pipeline.DROP_BALL_MOTION = True` and `USE_DYNAMICS = True`.
`generate_full_submission.py` has `--keep-ball-motion`, `--no-dynamics` and `--hard-training` (V4). The executed
notebook 04 and the technical document describe the earlier behaviour and need regenerating.

## Feature importance (loss-based, held-out)

`python -m experiments.feature_importance --target team` (home vs away on live frames) and
`--target dead` (live vs dead on all frames). For each of the seven external matches in turn: train on
the other six with Barcelona-style gaps, shuffle one feature family on the held-out match, and record how
much the held-out log-loss rises. A family is a feature and its windows. No family was flagged as a source
artefact (every Barcelona-vs-external audit AUC was 0.50-0.63; the alarm is 0.80). The contact / line
features are not in this run (`--extras` includes them). Ball speed and the deepest-player series are
excluded for the reasons given above.

| home vs away | loss gain | | live vs dead | loss gain |
|---|---|---|---|---|
| ALL PHYSICAL features | 0.633 | | `speed_all` (+ windows) | 0.752 |
| ALL TEMPORAL features | 0.340 | | ALL TEMPORAL features | 0.747 |
| `home_spread_y` | 0.188 | | ALL PHYSICAL features | 0.469 |
| `away_spread_y` | 0.124 | | `ball_x` | 0.135 |
| `dyn_home_cx` | 0.040 | | `ball_abs_y` | 0.085 |
| `dyn_away_cx` | 0.027 | | `dyn_home_speed` / `dyn_away_speed` | 0.026 / 0.024 |
| `margin` | 0.023 | | `d_min` | 0.018 |
| `ball_x` | 0.012 | | `home_spread_y` | 0.012 |

What it says: for **who has the ball**, team shape (width across the pitch, how the centroids move)
carries far more than nearest-player distance, which is weaker because the ball is hidden on 40% of frames;
for **dead ball**, collective player speed dominates, followed by the ball's position near a line, and
temporal features matter more than physical ones. Families with no value on either question:
`*_within_3m`, `ball_missing`, `ball_gap`, `ball_next_gap`. Caveats: permutation importance shares credit
between correlated features (family grouping reduces this), and these are importances on external matches
under simulated gaps. Whether pruning the no-value families helps is untested.

## The result

A gradient-boosted classifier trained **only on seven open Bundesliga matches** (CC BY 4.0, never on
Barcelona labels), decoded into possessions with a Viterbi pass:

| | spellF1 [90% interval] | macroF1 | accuracy | F1 H / A / D |
|---|---|---|---|---|
| nearest-player baseline, smooth 50 | 0.326 [0.22, 0.46] | 0.621 | 71.3% | 0.78 / 0.76 / 0.32 |
| logistic regression on Barcelona folds | 0.414 [0.25, 0.62] | 0.643 | 70.6% | 0.79 / 0.73 / 0.41 |
| GBDT on Barcelona folds only | 0.331 [0.22, 0.50] | 0.657 | 71.7% | 0.78 / 0.76 / 0.43 |
| **GBDT on 7 external matches, Viterbi** | **0.629 [0.46, 0.83]** | **0.884** | **89.9%** | **0.91 / 0.92 / 0.82** |

The frame-level gain (accuracy 71% to 90%) is large and consistent across the variants tried. The
spellF1 interval is wide and overlaps the others, so the possession-level gain is likely but not
proven on this sample.

## What made the difference (and what did not)

1. **External data teaches who owns the ball.** Trained on Bundesliga matches alone, home/away F1
   rises from about 0.77 to about 0.88 versus training on 5 minutes of Barcelona.
2. **A frozen-ball artefact had to be removed.** In the Bundesliga data `BallStatus = 0` frames have
   ball speed of exactly 0.00 (the provider freezes the ball), so a model learns "ball still means
   dead". In the Barcelona tracking the ball still moves at about 4 m/s during dead frames. Dropping
   the ball-speed features moved dead-ball F1 from 0.40 to 0.81 and macroF1 from 0.70 to 0.85. This is
   verifiable from the Bundesliga data alone.
3. **Barcelona-style gaps are replayed onto the clean external data.** The external ball is almost
   never missing; the Barcelona ball is missing on 41% of graded frames. Real Barcelona gap
   sequences and per-team tracked-player counts are replayed at random offsets
   (`external.corrupt`). A first version replayed per-slot gaps and removed far too many players
   (7.4 tracked per team against the real 10.0); slots are tracklets, so that was wrong.
4. **Viterbi beats a moving-average window.** Tuned on leave-one-match-out over the external matches
   (about 276 possessions per match, so the differences are stable): viterbi 20/20 reached spellF1
   0.543 against 0.519 for the best window (50 frames).
5. **Fine-tuning on Barcelona labels added nothing** (spellF1 0.629 either way, macroF1 slightly
   lower), so the final model never sees them.

## Round 2: trying to find more team changes (negative results)

**Diagnosis.** Of the 20 real changes in the labelled window, the model found 1 of 4 "play goes dead",
3 of 4 "restart", and only **3 of 12 team turnovers**. It reports 11 possessions where the truth has 17.

**What was tried**, each compared on leave-one-match-out over the seven external matches (about 1,900
possessions, so differences there are stable) and then on the Barcelona window (`lomo.py`):

| change | external leave-one-match-out spellF1 / macroF1 | Barcelona macroF1 / accuracy |
|---|---|---|
| baseline (round 1) | 0.541 / 0.845 | **0.884 / 89.9%** |
| + "who touched the ball last / next" and "ball at a line" features | **0.573 / 0.862** | 0.821 / 84.9% |
| ... touch features only | n/a | 0.816 / 84.4% |
| ... line features only | n/a | 0.861 / 88.0% |
| two-stage model (live vs dead, then home vs away on live frames) | 0.576 / 0.861 (no gain over one stage) | not run |
| cheaper team switches in the decoder (15/40, 10/40) | about equal | spellF1 0.587 and 0.550, down from 0.629 |
| more variety of replayed gaps (4 copies per match) | n/a | 0.855 / 87.4% |
| bigger, slower model (300 iterations, learning rate 0.05) | n/a | 0.875 / 89.3% |

**Conclusion: none of it beat round 1 on Barcelona, so round 1 is still the model.**

**The lesson.** The new features clearly helped on the held-out Bundesliga matches and clearly hurt on
Barcelona. Even the touch features' own target, the home/away F1, fell on Barcelona (about 0.91/0.92 to
0.87/0.86). The likely reason is that the Bundesliga possession label is defined as "the player in
control of the ball", so a touch feature nearly reproduces how those labels were made, while the
Barcelona truth uses a different convention. A feature can then look excellent on the external matches
for a reason that does not transfer. **Held-out external matches cannot validate changes that are meant
for Barcelona**; only Barcelona's own frames can, and they are few. The features stay in the code
behind `extras=True` (`features.EXPERIMENTAL`) and are off by default.

**What is left.** Turnover detection is limited by the evidence in the tracking, not by the decoder or
the classifier's size. The remaining honest levers are more Barcelona-like labelled data, and the
other team's contact rule tested on held-out frames, not further tuning against 17 possessions.

## Things I got wrong on the way (kept so nobody repeats them)

- A "fit once" model cache applied to the Barcelona-only method too, so every fold reused fold 1's
  model: macroF1 0.94, dead-ball F1 0.998 for a model that was trained on the test data. The
  fold-order leak check now catches this class of bug (it detects the re-created bug and passes the fix).
- I first suspected the teammate's champion over-called dead ball outside the 5 minutes (37% against
  20% inside). The Bundesliga data shows dead ball is 33-59% of frames in a real match, so that
  suspicion was unfounded.

## Caveats that limit these numbers

- **The public window is easy.** The ball is missing on 11.8% of its frames (live play: 6%) against
  41% over the match. Stress-testing with the ball hidden on about 61% of the window (harsher than the
  match) left macroF1 at 0.866 (from 0.884); spellF1 fell to about 0.53 but is noisy.
- **Label conventions differ.** The Bundesliga possession labels flip about twice as often as the
  Barcelona truth (median live spell 3.1 s against 7.6 s). External labels have blips under 2 s removed
  (`label_min_dur=50`) to align them. This is a guess at the convention, not a fact.
- **Gaps are independent of play** in the replay, but real gaps cluster around aerial balls and
  crowded boxes.
- **Several design choices were informed by looking at the Barcelona window** (dropping the ball-speed
  features after seeing the dead-ball profile; trying `label_min_dur`). The frozen ball is verifiable on
  the external data alone, but the window has still influenced the model.
- **No ground truth outside the 5 minutes.** On the rest of the match this model and the teammate's
  champion disagree on 23% of frames (mostly: external says dead or home where the champion says away).
  Agreement with the Smart Tagging attacking-phase team tags is 0.82 against 0.74 for the champion,
  but the tags are noisy (true labels agree with them only 68% in the window) and come from the same
  automatic process, so this shows similarity to that tagger and not correctness.

## Reproduce

```bash
python -m experiments.download_idsse                      # about 2.6 GB into data/external/idsse
python -m experiments.tune_external                       # decoding settings, leave-one-match-out
python -m experiments.run gbdt 'ext=["J03WMX","J03WN1","J03WOH","J03WOY","J03WPY","J03WQQ","J03WR9"]' \
    copies=2 'drop=["ball_speed"]' label_min_dur=50 decode=viterbi switch_team=20 switch_dead=20
python -m experiments.stress_test gbdt <same parameters>   # needs a method that never saw Barcelona labels
python -m experiments.lomo                                # leave-one-match-out on the external matches
```

Add `--write-submission my_team.csv` to the `run` command to write and validate the full-match file.
