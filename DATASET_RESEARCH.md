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
| **SkillCorner open data** | broadcast-video tracking, **10 fps**, 10 A-League 2024/25 matches | closest in nature to ours: derived from video, players have a detected / extrapolated flag, ball has a detected flag | per-frame `possession` (player id + home/away); phases of play exist only while the ball is in play | MIT (confirmed in the repo's LICENSE); README asks for credit to SkillCorner | **not used yet**, best next candidate |
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

## What using it would take
1. Download the repo (about 181 MB, `github.com/SkillCorner/opendata`) into `data/external/skillcorner/` (git-ignored).
2. Loader: parse `{id}_tracking_extrapolated.jsonl`; metres from the pitch centre; mask players and ball where
   `is_detected` is false; resample 10 fps to 25 fps (or rescale every frame-based window to seconds).
3. Labels: possession team from the `possession` field, held forward through loose frames ("stays with the last team that
   had it", CHALLENGE.md), `D` where no phase of play covers the frame.
4. Check that adding it improves leave-one-match-out on the IDSSE matches (and the reverse), and does not hurt.
5. Credit SkillCorner in the technical document.

## Open questions for the organisers
- Are SkillCorner (MIT repository) and PFF FC (form-gated) acceptable as training data?
- Is Metrica's sample data acceptable given it states no licence?

## Sources
- SkillCorner open data: https://github.com/SkillCorner/opendata and https://skillcorner.github.io/opendata/
- PFF FC 2022 World Cup dataset: https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset
- Metrica sample data: https://github.com/metrica-sports/sample-data
- SoccerTrack v2: https://arxiv.org/html/2508.01802v1
- SoccerNet-Tracking: https://arxiv.org/abs/2204.06918
