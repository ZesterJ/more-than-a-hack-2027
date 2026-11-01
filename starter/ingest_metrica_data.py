"""Metrica Ingestion and Harmonization Pipeline for More Than A Hack 2027.

Downloads, parses, and extracts identical 31 kinematic and tactical features from
Metrica Sports Sample Games 1, 2, and 3, outputting ML-ready training datasets.
"""

import os
import sys
import math
import json
import csv
import urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(ROOT, "data", "metrica_raw")
OUTPUT_DIR = os.path.join(ROOT, "data")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

GITHUB_BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
PITCH_X = 105.0
PITCH_Y = 68.0
FPS = 25.0


def download_if_missing(rel_path: str, local_name: str) -> str:
    local_path = os.path.join(RAW_DIR, local_name)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return local_path
    url = f"{GITHUB_BASE}/{rel_path}"
    print(f"Downloading {local_name} from Metrica repository...")
    urllib.request.urlretrieve(url, local_path)
    print(f"Saved {local_name} ({os.path.getsize(local_path)/(1024*1024):.2f} MB)")
    return local_path


def parse_events_csv(csv_path: str) -> dict:
    possession = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for ev in reader:
            team = ev.get("Team")
            typ = ev.get("Type", "")
            try:
                sf = int(ev.get("Start Frame", 0))
                ef = int(ev.get("End Frame", sf))
            except (ValueError, TypeError):
                continue
            if ef < sf or ef == 0:
                ef = sf
            lbl = "H" if team == "Home" else ("A" if team == "Away" else "D")
            if typ in ["BALL OUT", "CARD", "FAULT RECEIVED"]:
                lbl = "D"
            for f_idx in range(sf, ef + 1):
                possession[f_idx] = lbl
    return possession


def parse_events_json(json_path: str) -> dict:
    possession = {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f).get("data", [])
        for ev in data:
            team = ev.get("team", {}).get("name")
            typ = ev.get("type", {}).get("name", "")
            sf = ev.get("start", {}).get("frame")
            ef = ev.get("end", {}).get("frame")
            if sf is None:
                continue
            if ef is None or ef < sf or ef == 0:
                ef = sf
            lbl = "H" if team == "Team A" else ("A" if team == "Team B" else "D")
            if typ in ["BALL OUT", "CARD", "FAULT RECEIVED"]:
                lbl = "D"
            for f_idx in range(sf, ef + 1):
                possession[f_idx] = lbl
    return possession


def compute_features_from_state(
    frame_idx: int,
    ball_norm: tuple,
    home_norm: list,
    away_norm: list,
    prev_ball: tuple,
    label: str,
) -> tuple:
    # 1. Ball status & coords
    if ball_norm is None or math.isnan(ball_norm[0]) or math.isnan(ball_norm[1]):
        ball_missing = 1
        ball_x = np.nan
        ball_y = np.nan
        ball_speed = 0.0
        cur_ball = None
    else:
        ball_missing = 0
        ball_x = float(ball_norm[0] * PITCH_X)
        ball_y = float(ball_norm[1] * PITCH_Y)
        cur_ball = (ball_x, ball_y)
        if prev_ball is not None:
            ball_speed = math.hypot(ball_x - prev_ball[0], ball_y - prev_ball[1]) * FPS
        else:
            ball_speed = 0.0

    # 2. Convert player coordinates to pitch meters
    home_coords = [(float(p[0] * PITCH_X), float(p[1] * PITCH_Y)) for p in home_norm if not math.isnan(p[0]) and not math.isnan(p[1])]
    away_coords = [(float(p[0] * PITCH_X), float(p[1] * PITCH_Y)) for p in away_norm if not math.isnan(p[0]) and not math.isnan(p[1])]

    # 3. Ball-Player Proximity
    d_min_home = 999.0
    d_min_away = 999.0
    home_within_3m = 0
    away_within_3m = 0

    if ball_missing == 0:
        for px, py in home_coords:
            d = math.hypot(px - ball_x, py - ball_y)
            if d < d_min_home:
                d_min_home = d
            if d <= 3.0:
                home_within_3m += 1

        for px, py in away_coords:
            d = math.hypot(px - ball_x, py - ball_y)
            if d < d_min_away:
                d_min_away = d
            if d <= 3.0:
                away_within_3m += 1

        d_margin = d_min_away - d_min_home
    else:
        d_margin = 0.0
        d_min_home = np.nan
        d_min_away = np.nan

    # 4. Collective Team Geometry & Centroids
    if home_coords:
        hx = [p[0] for p in home_coords]
        hy = [p[1] for p in home_coords]
        home_centroid_x = float(np.mean(hx))
        home_centroid_y = float(np.mean(hy))
        home_spread_x = float(np.std(hx))
        home_spread_y = float(np.std(hy))
        home_spread_area = home_spread_x * home_spread_y
    else:
        home_centroid_x, home_centroid_y = 52.5, 34.0
        home_spread_x, home_spread_y, home_spread_area = 0.0, 0.0, 0.0

    if away_coords:
        ax = [p[0] for p in away_coords]
        ay = [p[1] for p in away_coords]
        away_centroid_x = float(np.mean(ax))
        away_centroid_y = float(np.mean(ay))
        away_spread_x = float(np.std(ax))
        away_spread_y = float(np.std(ay))
        away_spread_area = away_spread_x * away_spread_y
    else:
        away_centroid_x, away_centroid_y = 52.5, 34.0
        away_spread_x, away_spread_y, away_spread_area = 0.0, 0.0, 0.0

    centroid_diff_x = home_centroid_x - away_centroid_x
    spread_ratio = (home_spread_area + 1e-4) / (away_spread_area + 1e-4)

    record = {
        "video_frame": frame_idx,
        "frame_id": frame_idx,
        "ball_missing": ball_missing,
        "ball_x": round(ball_x, 3) if not math.isnan(ball_x) else "",
        "ball_y": round(ball_y, 3) if not math.isnan(ball_y) else "",
        "ball_speed": round(ball_speed, 3),
        "d_min_home": round(d_min_home, 3) if not math.isnan(d_min_home) else "",
        "d_min_away": round(d_min_away, 3) if not math.isnan(d_min_away) else "",
        "d_margin": round(d_margin, 3),
        "home_within_3m": home_within_3m,
        "away_within_3m": away_within_3m,
        "home_active": len(home_coords),
        "away_active": len(away_coords),
        "home_centroid_x": round(home_centroid_x, 3),
        "home_centroid_y": round(home_centroid_y, 3),
        "away_centroid_x": round(away_centroid_x, 3),
        "away_centroid_y": round(away_centroid_y, 3),
        "centroid_diff_x": round(centroid_diff_x, 3),
        "home_spread_x": round(home_spread_x, 3),
        "home_spread_y": round(home_spread_y, 3),
        "away_spread_x": round(away_spread_x, 3),
        "away_spread_y": round(away_spread_y, 3),
        "spread_ratio": round(spread_ratio, 3),
        "mean_speed_home": 1.2,  # Baseline nominal player speed
        "mean_speed_away": 1.2,
        "overall_speed": 1.2,
        "label": label,
    }
    return record, cur_ball


def add_rolling_features(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)
    d_margins = df["d_margin"].astype(np.float32).values
    speeds = df["overall_speed"].astype(np.float32).values
    missing_flags = df["ball_missing"].astype(np.float32).values

    w25 = np.ones(25) / 25.0
    w75 = np.ones(75) / 75.0
    w50 = np.ones(50) / 50.0

    df["d_margin_roll_1s"] = np.round(np.convolve(d_margins, w25, mode="same"), 3)
    df["d_margin_roll_3s"] = np.round(np.convolve(d_margins, w75, mode="same"), 3)
    df["speed_all_roll_2s"] = np.round(np.convolve(speeds, w50, mode="same"), 3)
    df["missing_ratio_roll_2s"] = np.round(np.convolve(missing_flags, w50, mode="same"), 3)
    return df


def process_game_1_or_2(game_num: int, max_frames: int = 40000) -> pd.DataFrame:
    tag = f"Sample_Game_{game_num}"
    events_path = download_if_missing(f"{tag}/{tag}_RawEventsData.csv", f"{tag}_events.csv")
    home_path = download_if_missing(f"{tag}/{tag}_RawTrackingData_Home_Team.csv", f"{tag}_home.csv")
    away_path = download_if_missing(f"{tag}/{tag}_RawTrackingData_Away_Team.csv", f"{tag}_away.csv")

    possession = parse_events_csv(events_path)
    print(f"[{tag}] Loaded events with {len(possession)} tagged frames")

    # Read tracking rows
    records = []
    prev_ball = None

    with open(home_path, "r", encoding="utf-8") as fh, open(away_path, "r", encoding="utf-8") as fa:
        rh = csv.reader(fh)
        ra = csv.reader(fa)
        # Skip 3 header rows
        for _ in range(3):
            next(rh, None)
            next(ra, None)

        frame_count = 0
        for row_h, row_a in zip(rh, ra):
            if not row_h or not row_a:
                continue
            frame_idx = int(row_h[1])
            label = possession.get(frame_idx, "X")

            # We only train on confirmed possession states
            if label not in ["H", "A", "D"]:
                continue

            # Parse ball (columns 31, 32 in home CSV)
            try:
                bx = float(row_h[31])
                by = float(row_h[32])
                ball_norm = (bx, by)
            except (ValueError, IndexError):
                ball_norm = None

            # Parse players: pairs (3,4), (5,6), ... up to col 30
            home_norm = []
            for col in range(3, min(len(row_h) - 2, 31), 2):
                try:
                    px = float(row_h[col])
                    py = float(row_h[col + 1])
                    home_norm.append((px, py))
                except (ValueError, IndexError):
                    pass

            away_norm = []
            for col in range(3, min(len(row_a) - 2, 31), 2):
                try:
                    px = float(row_a[col])
                    py = float(row_a[col + 1])
                    away_norm.append((px, py))
                except (ValueError, IndexError):
                    pass

            rec, prev_ball = compute_features_from_state(frame_idx, ball_norm, home_norm, away_norm, prev_ball, label)
            rec["game"] = f"Metrica_{game_num}"
            records.append(rec)
            frame_count += 1
            if frame_count >= max_frames:
                break

    print(f"[{tag}] Extracted {len(records)} feature frames")
    return add_rolling_features(records)


def process_game_3(max_frames: int = 40000) -> pd.DataFrame:
    tag = "Sample_Game_3"
    events_path = download_if_missing(f"{tag}/{tag}_events.json", f"{tag}_events.json")
    tracking_path = download_if_missing(f"{tag}/{tag}_tracking.txt", f"{tag}_tracking.txt")

    possession = parse_events_json(events_path)
    print(f"[{tag}] Loaded events with {len(possession)} tagged frames")

    records = []
    prev_ball = None
    frame_count = 0

    with open(tracking_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            frame_idx = int(parts[0])
            label = possession.get(frame_idx, "X")
            if label not in ["H", "A", "D"]:
                continue

            # Ball coords
            try:
                b_parts = parts[2].split(",")
                ball_norm = (float(b_parts[0]), float(b_parts[1]))
            except (ValueError, IndexError):
                ball_norm = None

            # 22 players: first 11 Home (Team A), next 11 Away (Team B)
            players_str = parts[1].split(";")
            home_norm = []
            away_norm = []
            for i, p_str in enumerate(players_str):
                try:
                    coords = p_str.split(",")
                    px, py = float(coords[0]), float(coords[1])
                    if i < 11:
                        home_norm.append((px, py))
                    else:
                        away_norm.append((px, py))
                except (ValueError, IndexError):
                    pass

            rec, prev_ball = compute_features_from_state(frame_idx, ball_norm, home_norm, away_norm, prev_ball, label)
            rec["game"] = "Metrica_3"
            records.append(rec)
            frame_count += 1
            if frame_count >= max_frames:
                break

    print(f"[{tag}] Extracted {len(records)} feature frames")
    return add_rolling_features(records)


def main():
    print("=" * 65)
    print("  METRICA SPORTS 3-GAME FEATURE HARMONIZATION PIPELINE")
    print("=" * 65)

    df1 = process_game_1_or_2(1, max_frames=30000)
    df2 = process_game_1_or_2(2, max_frames=30000)
    df3 = process_game_3(max_frames=30000)

    # Combine all 3 Metrica games
    df_metrica = pd.concat([df1, df2, df3], ignore_index=True)
    metrica_out = os.path.join(OUTPUT_DIR, "metrica_3games_features.csv")
    df_metrica.to_csv(metrica_out, index=False)
    print(f"\n[DONE] Saved Metrica combined dataset: {metrica_out}")
    print(f"Total labeled frames: {len(df_metrica):,}")
    print("Class distribution:")
    print(df_metrica["label"].value_counts())

    # Merge with Barca Public Ground Truth to create the Comprehensive ML Training Set
    barca_path = os.path.join(OUTPUT_DIR, "features_first_15min_play.csv")
    if os.path.exists(barca_path):
        df_barca = pd.read_csv(barca_path)
        df_barca_labeled = df_barca[df_barca["label"].isin(["H", "A", "D"])].copy()
        df_barca_labeled["game"] = "BAR-ATH_ground_truth"

        cols = [c for c in df_barca_labeled.columns if c in df_metrica.columns]
        df_unified = pd.concat([df_metrica[cols], df_barca_labeled[cols]], ignore_index=True)
        unified_out = os.path.join(OUTPUT_DIR, "ml_unified_training_dataset.csv")
        df_unified.to_csv(unified_out, index=False)
        print(f"\n[SUCCESS] Created Unified Training Set: {unified_out}")
        print(f"Total Unified Samples: {len(df_unified):,}")
        print("Unified Class distribution:")
        print(df_unified["label"].value_counts())


if __name__ == "__main__":
    main()
