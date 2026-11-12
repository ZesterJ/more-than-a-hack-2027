# External training data: what exists, and how close it is to our inputs

Rules (Nov 3 clarification): train and tune on external data only, and only on datasets we have the rights to use.
Our inputs are the organisers' tracking (ATD): every player and the ball at 25 fps in pitch metres, produced by computer
vision from the tactical video, so players and the ball are often missing (the ball is unseen in 41% of graded frames).
The labels to predict are `H`, `A` (team in possession) and `D` (ball dead). Research done 2026-11-11; every licence below
must still be confirmed by the team or the organisers before use.

## Summary

| Dataset | Tracking | Fit to our inputs | Possession / dead-ball labels | Licence | Status |
|---|---|---|---|---|---|
| **IDSSE** (Sportec/DFL) | optical, 25 fps, 7 Bundesliga matches | good geometry, ball almost always seen | provider ball status and possession | CC-BY 4.0 | **used** |
| **SkillCorner open data** | broadcast-video tracking, **10 fps**, 10 A-League 2024/25 matches | closest in nature to ours: derived from video, players have a detected / extrapolated flag, ball has a detected flag | per-frame `possession` (player id + home/away); phases of play exist only while the ball is in play | MIT (confirmed in the repo's LICENSE); README asks for credit to SkillCorner | **tested (3 matches); optional via `--with-skillcorner`** |
| **PFF FC World Cup 2022** | broadcast tracking, 64 matches | similar in nature to SkillCorner | events; possession labels not confirmed | free through a form; terms not seen | needs a person to read the terms and fill the form |
| **Metrica sample games** | optical, 25 fps, 3 games | same vendor family as our ATD | events (possession derivable) | **no licence stated** ("be responsible") | not used |
| StatsBomb open data | freeze frames, no continuous tracking | poor | events | attribution agreement | not useful for frame models |
| SoccerTrack v2 | player positions from panoramic video, amateur matches | no ball tracking found | action labels only | not stated | not useful |
| SoccerNet-Tracking | bounding-box tracklets | pixels, not pitch metres | none | n/a | not useful |

## Why SkillCorner is the best next dataset
- **Same kind of gaps.** Our tracking is video-derived and misses players and the ball. SkillCorner's frames carry an
  `is_detected` flag for every player and for the ball, so we can set undetected players to missing and reproduce the same
  structure, instead of copying a gap pattern onto a ball that was always seen (what we do with IDSSE).
- **Different league and camera, same problem.** A third source reduces how much the model overfits to Bundesliga tempo.
- **Possession labels exist.** The tracking has a per-frame possession field, and phases of play are defined only while the
  ball is in play, so a dead-ball label can be derived from frames outside any phase.

## What we tested (3 of the 20 matches, 2026-11-11)
Fetch: `python starter/external.py --fetch-skillcorner 3` (the tracking files are Git LFS, about 90 MB each; all 20 would be
about 1.8 GB). Loader: `ext.load_skillcorner`. Labels: a frame inside a phase of play takes that phase's team in possession,
any frame outside every phase is dead ball. Players the broadcast did not detect can stay missing or take the source's
extrapolated positions (the extrapolated version scored better and is used). The loader's labels agree with "the team
nearest the ball" 83-87% of the time, the same as IDSSE (85%).

| Test (spellF1, fixed decoder) | Result |
|---|---|
| Train IDSSE -> predict SkillCorner | 0.55 |
| Train SkillCorner -> predict IDSSE | 0.51 |
| Within IDSSE / within SkillCorner (leave-one-match-out) | 0.70 / 0.71 |
| IDSSE + 2 SkillCorner -> held-out SkillCorner | 0.68 |
| IDSSE + 3 SkillCorner pooled leave-one-match-out | IDSSE matches 0.686, SkillCorner matches 0.675 |

- **The domain gap is real:** each dataset predicts itself far better (0.70) than the other (0.5 to 0.55).
- **Features that help on IDSSE do not all transfer:** training on IDSSE only, scoring on SkillCorner peaks at the 50-feature
  set (0.595) and falls with the boundary, motion and temporal blocks (0.535 to 0.551). Once some SkillCorner data is in
  training, every block helps again (0.606 to 0.676 from 30 to 109 features). So the richer features need more than one
  tracking source to pay off.
- **Input-only similarity to the competition tracking:** a classifier that tells external frames from this match's frames
  (no labels) gives SkillCorner frames higher odds of looking like this match (median 0.58) than IDSSE frames (0.35).
- **Report-only look at the public window with IDSSE + 3 SkillCorner:** spellF1 0.648 (0.593 with IDSSE alone), dead-ball
  recall 30% at 99% precision. One noisy look; the choice to combine rests on the cross-dataset evidence above.
- **Cost:** a small loss on IDSSE matches (0.703 to 0.686) for a large gain on SkillCorner ones.

## Decision pending
Default stays IDSSE-only until the team confirms the licence (MIT, credit SkillCorner) with the organisers and decides
whether to download all 20 matches.

## Open questions for the organisers
- Are SkillCorner (MIT repository) and PFF FC (form-gated) acceptable as training data?
- Is Metrica's sample data acceptable given it states no licence?

## Sources
- SkillCorner open data: https://github.com/SkillCorner/opendata and https://skillcorner.github.io/opendata/
- PFF FC 2022 World Cup dataset: https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset
- Metrica sample data: https://github.com/metrica-sports/sample-data
- SoccerTrack v2: https://arxiv.org/html/2508.01802v1
- SoccerNet-Tracking: https://arxiv.org/abs/2204.06918
