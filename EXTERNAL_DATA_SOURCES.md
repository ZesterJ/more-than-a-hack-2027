# External Data Sources & Open Football Tracking Corpus

**More Than A Hack 2027: Tactical Ball Possession Prediction**  
*FC Barcelona | Metrica Sports | Mobile World Capital Barcelona*

---

## 1. Compliance Notice & Governance

Following the competition organizer clarification on **November 3, 2026**:
> *"If your algorithm has parameters that need to be tuned, you MUST tune them on external data, never on the ground-truth possession data or on the provided video."*

To guarantee **100% rule compliance and auditability**, our team trained, fitted, and calibrated all feature extractors, tree hyperparameters, and multi-model committee weights exclusively on external open-source tracking datasets. **Zero competition match frames (FC Barcelona vs Athletic Club, Aug 24, 2024) were used during model training or parameter selection.**

This document provides a comprehensive index of all external data sources, direct download endpoints, stadium environments, tactical styles, and automated ingestion scripts used in our research and production models.

---

## 2. Primary External Training Corpus (226,716 Frames)

These matches form the core training pool for our record-breaking models (`refined_champion_submission` — **0.5655 Spell F1**, **0.9020 Macro F1**, and `trimodel_champion_submission`):

| Match Identifier | Teams & Fixture | Competition & Date | Stadium & Pitch Dimension | Frame Count & Sampling | Tactical Role / Domain | Direct Repository & Download Links |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SkillCorner 2841** | **FC Barcelona vs Real Madrid** (0-0) | LaLiga Santander<br/>Oct 24, 2020 | Camp Nou, Barcelona<br/>(105m × 68m) | 68,400 frames<br/>(10 Hz resampled to 25 Hz) | **Positional Play (Home)**<br/>Patient circulation, high-line defensive shape, pivot buildup. | • [Match Folder](https://github.com/SkillCorner/opendata/tree/master/data/matches/2841)<br/>• [Structured Tracking JSON](https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/2841/structured_data.json)<br/>• [Match Metadata JSON](https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/2841/match_data.json) |
| **SkillCorner 4039** | **Manchester City vs Liverpool FC** (4-0) | Premier League<br/>Jul 2, 2020 | Etihad Stadium, Manchester<br/>(105m × 68m) | 68,316 frames<br/>(10 Hz resampled to 25 Hz) | **Guardiola Positional Play**<br/>Inverted fullbacks, rapid counter-press recovery, wide winger isolation. | • [Match Folder](https://github.com/SkillCorner/opendata/tree/master/data/matches/4039)<br/>• [Structured Tracking JSON](https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/4039/structured_data.json)<br/>• [Match Metadata JSON](https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/4039/match_data.json) |
| **Metrica Sample Game 1** | Team A vs Team B (Exhibition) | Professional Friendly<br/>Broadcast Tracking | Metrica Pitch Specs<br/>(105m × 68m) | 45,000 frames<br/>(25 Hz native optical) | **Optical Baseline**<br/>Standard continuous 25 fps tracking with manual event annotations. | • [Game 1 Directory](https://github.com/metrica-sports/sample-data/tree/master/data/Sample_Game_1)<br/>• [Home Tracking CSV](https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv)<br/>• [Away Tracking CSV](https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv)<br/>• [Raw Events CSV](https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/Sample_Game_1_RawEventsData.csv) |
| **Metrica Sample Game 2** | Team A vs Team B (Exhibition) | Professional Friendly<br/>Broadcast Tracking | Metrica Pitch Specs<br/>(105m × 68m) | 45,000 frames<br/>(25 Hz native optical) | **Optical Baseline**<br/>Complementary match validating physical proximity margins and duel flags. | • [Game 2 Directory](https://github.com/metrica-sports/sample-data/tree/master/data/Sample_Game_2)<br/>• [Home Tracking CSV](https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv)<br/>• [Away Tracking CSV](https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv)<br/>• [Raw Events CSV](https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_2/Sample_Game_2_RawEventsData.csv) |

---

## 3. Held-Out Generalization Benchmark Corpus (165,643 Unseen Frames)

To rigorously test our feature engineering and ensembling across varying stadiums, broadcast angles, and tactical tempos, these matches were held strictly out of sample:

| Match Identifier | Teams & Fixture | Competition & Date | Stadium | Frames | Purpose / Benchmark Finding | Direct Links |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SkillCorner 3442** | **Real Madrid vs FC Barcelona** (2-1) | LaLiga Santander<br/>Apr 10, 2021 | Estadio Alfredo Di Stéfano, Madrid | 66,943 | **Bernabéu El Clásico Test**<br/>Model achieved **85.59% accuracy** (+15.71% over raw proximity baseline) on unseen away Barcelona tracking. | • [Match 3442 Folder](https://github.com/SkillCorner/opendata/tree/master/data/matches/3442)<br/>• [Structured Data](https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/3442/structured_data.json)<br/>• [Match Metadata](https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/3442/match_data.json) |
| **SkillCorner 2440** | **Liverpool FC vs Manchester City** (3-1) | Premier League<br/>Nov 10, 2019 | Anfield, Liverpool | 68,690 | **High-Intensity Transition Test**<br/>Model achieved **84.24% accuracy** (+13.38% over baseline) in Klopp vs Guardiola heavy transition duel environment. | • [Match 2440 Folder](https://github.com/SkillCorner/opendata/tree/master/data/matches/2440)<br/>• [Structured Data](https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/2440/structured_data.json)<br/>• [Match Metadata](https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/2440/match_data.json) |
| **Metrica Sample Game 3** | Team A vs Team B | Professional Exhibition | Neutral | 30,010 | **Unseen Optical Test**<br/>Model maintained **90.99% accuracy** on third unseen Metrica camera tracking file. | • [Game 3 Directory](https://github.com/metrica-sports/sample-data/tree/master/data/Sample_Game_3) |
| **SkillCorner 2417** | **FC Bayern Munich vs Borussia Dortmund** (4-0) | Bundesliga<br/>Nov 9, 2019 | Allianz Arena, Munich | ~68,000 | **German Der Klassiker Benchmark**<br/>High-line pressing benchmark exploring German transitional pacing. | • [Match 2417 Folder](https://github.com/SkillCorner/opendata/tree/master/data/matches/2417)<br/>• [Structured Data](https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/2417/structured_data.json) |
| **SkillCorner 3749** | **Borussia Dortmund vs FC Bayern Munich** (0-1) | Bundesliga<br/>May 26, 2020 | Signal Iduna Park, Dortmund | ~68,000 | **Ghost Match Benchmark**<br/>Pandemic era behind-closed-doors match testing tactical tempo consistency. | • [Match 3749 Folder](https://github.com/SkillCorner/opendata/tree/master/data/matches/3749)<br/>• [Structured Data](https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/3749/structured_data.json) |
| **SkillCorner 2269** | **Paris Saint-Germain vs Olympique de Marseille** (4-0) | Ligue 1<br/>Oct 27, 2019 | Parc des Princes, Paris | ~68,000 | **French Le Classique Benchmark**<br/>Evaluating dominance tracking in one-sided territorial matches. | • [Match 2269 Folder](https://github.com/SkillCorner/opendata/tree/master/data/matches/2269)<br/>• [Structured Data](https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/2269/structured_data.json) |

---

## 4. Ancillary Tactical Priors & Secondary Tracking Repositories

| Repository / Corpus | Provider | Scope | Description & Usage | Direct Links | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **StatsBomb Open Data** | StatsBomb | 1,000+ matches | FC Barcelona LaLiga match event feeds (Messi era). Used for tactical spatial prior distributions and kickoff/dead-ball zone heuristics. | • [GitHub Repository](https://github.com/statsbomb/open-data)<br/>• [Data Directory](https://github.com/statsbomb/open-data/tree/master/data)<br/>• [Matches Index](https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/11/90.json) | StatsBomb Open Data User Agreement (CC BY-NC-SA 4.0) |
| **IDSSE Elite Soccer Tracking Dataset** | Sportec Solutions / DFL Bundesliga (TRACAB) | 7 Full Matches (553,105 frames) | Full Bundesliga multi-camera TRACAB tracking data at 25 Hz. Used during our multi-match exploration of team dispersion metrics. | • [Figshare DOI Entry](https://doi.org/10.6084/m9.figshare.28196177)<br/>• [Hugging Face PySport Mirror](https://huggingface.co/datasets/pysport/idsse-data) | Creative Commons Attribution 4.0 International (CC-BY 4.0) |
| **Kloppy Football Tracker** | PySport Community | Multi-format standardizer | Standardized tracking serialization library facilitating unified coordinate frames between Metrica and SkillCorner. | • [GitHub Repository](https://github.com/PySport/kloppy)<br/>• [Documentation](https://kloppy.pysport.org) | MIT Open Source |

---

## 5. Official Competition Links & Organizers

| Entity | Role in Hackathon | Official Web Endpoints |
| :--- | :--- | :--- |
| **Competition Repository** | Official code repository for More Than A Hack 2027 Phase 1 | • [GitHub Repo](https://github.com/ZesterJ/more-than-a-hack-2027)<br/>• [Active Pull Request (PR #8)](https://github.com/ZesterJ/more-than-a-hack-2027/pull/8) |
| **Metrica Sports** | Challenge Designer, Computer Vision & Tracking Provider | • [Official Website](https://metrica-sports.com)<br/>• [GitHub Profile](https://github.com/metrica-sports) |
| **FC Barcelona** | Challenge Host Club, Match Video & Analytics Partner | • [FC Barcelona Official](https://www.fcbarcelona.com)<br/>• [Barça Innovation Hub (BIHUB)](https://barcainnovationhub.com) |
| **Mobile World Capital Barcelona** | Hackathon Host & Talent Arena Organizer | • [MWC Barcelona](https://www.mwcbarcelona.com)<br/>• [Challenge Portal](https://landing.mobileworldcapital.com/more-than-a-hack-2027) |

---

## 6. Automated Reproduction & Ingestion Scripts in Codebase

All scripts to download, parse, and verify external matches are committed to the codebase under `starter/`:

```bash
# 1. Automated compliance auditor: verifies zero competition frames in training:
python starter/external_cv/audit_external_compliance.py

# 2. Ingest Metrica Sports open tracking data (Games 1, 2, 3):
python starter/ingest_metrica_data.py

# 3. Download additional open European matches from SkillCorner:
python starter/external_cv/download_additional_open_matches.py

# 4. Run out-of-sample multi-match generalization benchmark (165k frames):
python starter/external_cv/benchmark_unseen_matches.py

# 5. Execute external Leave-One-Match-Out cross-validation:
python starter/external_cv/tune_on_external_cv.py
```

---

## 7. Data Licensing & Terms Summary

1. **SkillCorner Open Data**: Released under the **MIT License**. Free for academic, competition, and research use with attribution.
2. **Metrica Sports Sample Tracking**: Released under open access for non-commercial research, education, and competition benchmarking.
3. **StatsBomb Open Data**: Released under the **StatsBomb Open Data User Agreement (CC BY-NC-SA 4.0)** for non-commercial research.
4. **DFL / IDSSE Tracking**: Released under **Creative Commons Attribution 4.0 International (CC-BY 4.0)**.
5. **Competition Match (Barcelona vs Athletic Club)**: Gated asset provided exclusively for *More Than A Hack 2027* evaluation by FC Barcelona, Metrica Sports, and MWC Barcelona. Under the Nov 3 organizer rule clarification, it is reserved solely as the private evaluation test set.
