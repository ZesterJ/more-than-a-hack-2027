"""Extract collective spatial and temporal features for ball possession classification.

Reads raw Metrica ATD tracking data (and optional public ground truth) and outputs
a tabular feature dataset suitable for LightGBM, XGBoost, or PyTorch models.

Usage:
    python starter/extract_features.py --window 15min    # 15 minutes of match play from kickoff (22,500 frames)
    python starter/extract_features.py --window public   # First 5 minutes with public ground truth (7,500 frames)
    python starter/extract_features.py --window full     # Full match: 199,319 frames
    python starter/extract_features.py --start 11185 --end 33684
"""

import argparse
import csv
import math
import os
import sys
import time
import numpy as np

# Project path resolution
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import games

# Constants
PITCH_LENGTH, PITCH_WIDTH = games.GAMES["BAR-ATH"]["pitch"]  # 105.3, 68.0
KICKOFF_FRAME = 11185        # First half kickoff
FPS = 25
HOME_SLOTS = set(range(0, 24))   # Track 1 to 24 (Barcelona)
AWAY_SLOTS = set(range(24, 46))  # Track 25 to 46 (Athletic)
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING_CHAR = "N"  # First char of "NaN"


def load_ground_truth(gt_path):
    """Load ground truth labels mapping frame -> label."""
    gt_map = {}
    if os.path.exists(gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                gt_map[int(row[0])] = row[1]
    return gt_map


def extract_features(raw_path, gt_map, start_frame=0, end_frame=199318, output_path=None):
    """Stream raw tracking file and compute collective geometric & temporal features."""
    t0 = time.time()
    n_frames = end_frame - start_frame + 1
    duration_min = n_frames / (FPS * 60.0)
    print(f"Extracting features for {n_frames:,} frames ({duration_min:.1f} minutes of video, frames {start_frame}..{end_frame})...")

    records = []
    
    with open(raw_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(FIELD_SEP)
            if len(parts) != 3:
                continue

            frame_id = int(parts[0])
            video_frame = frame_id - 1

            if video_frame < start_frame:
                continue
            if video_frame > end_frame:
                break

            # 1. Parse Ball Coordinates
            ball_part = parts[2]
            if not ball_part or ball_part[0] == MISSING_CHAR or ball_part.startswith("-1"):
                ball_missing = 1
                ball_x = np.nan
                ball_y = np.nan
                ball_speed = 0.0
            else:
                ball_missing = 0
                b_tokens = ball_part.split(",")
                ball_x = float(b_tokens[0]) * PITCH_LENGTH
                ball_y = float(b_tokens[1]) * PITCH_WIDTH
                ball_speed = float(b_tokens[2]) if len(b_tokens) > 2 and b_tokens[2][0] != MISSING_CHAR else 0.0

            # 2. Parse Active Players
            home_coords = []
            home_speeds = []
            away_coords = []
            away_speeds = []

            for slot_idx, group in enumerate(parts[1].split(GROUP_SEP)):
                if not group or group[0] == MISSING_CHAR or group.startswith("-1"):
                    continue
                tokens = group.split(",")
                px = float(tokens[0]) * PITCH_LENGTH
                py = float(tokens[1]) * PITCH_WIDTH
                ps = float(tokens[2]) if len(tokens) > 2 and tokens[2][0] != MISSING_CHAR else 0.0

                if slot_idx in HOME_SLOTS:
                    home_coords.append((px, py))
                    home_speeds.append(ps)
                elif slot_idx in AWAY_SLOTS:
                    away_coords.append((px, py))
                    away_speeds.append(ps)

            # 3. Ball-Player Proximity & Local Pressure
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

                d_margin = d_min_away - d_min_home  # Positive -> Home closer
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

            # 5. Speeds & Dead Ball / Stoppage Intensity
            mean_speed_home = float(np.mean(home_speeds)) if home_speeds else 0.0
            mean_speed_away = float(np.mean(away_speeds)) if away_speeds else 0.0
            all_speeds = home_speeds + away_speeds
            overall_speed = float(np.mean(all_speeds)) if all_speeds else 0.0

            # Ground truth label
            label = gt_map.get(video_frame, "X")

            records.append({
                "video_frame": video_frame,
                "frame_id": frame_id,
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
                "mean_speed_home": round(mean_speed_home, 3),
                "mean_speed_away": round(mean_speed_away, 3),
                "overall_speed": round(overall_speed, 3),
                "label": label
            })

    # 6. Compute Temporal Rolling Window Features
    print("Computing temporal rolling window features...")
    d_margins = np.array([r["d_margin"] for r in records], dtype=np.float32)
    speeds = np.array([r["overall_speed"] for r in records], dtype=np.float32)
    missing_flags = np.array([r["ball_missing"] for r in records], dtype=np.float32)

    # 1-second (25 frames) and 3-second (75 frames) moving averages
    w25 = np.ones(25) / 25.0
    w75 = np.ones(75) / 75.0
    w50 = np.ones(50) / 50.0

    d_margin_smooth25 = np.convolve(d_margins, w25, mode="same")
    d_margin_smooth75 = np.convolve(d_margins, w75, mode="same")
    overall_speed_smooth50 = np.convolve(speeds, w50, mode="same")
    missing_ratio_smooth50 = np.convolve(missing_flags, w50, mode="same")

    for i, r in enumerate(records):
        r["d_margin_roll_1s"] = round(float(d_margin_smooth25[i]), 3)
        r["d_margin_roll_3s"] = round(float(d_margin_smooth75[i]), 3)
        r["speed_all_roll_2s"] = round(float(overall_speed_smooth50[i]), 3)
        r["missing_ratio_roll_2s"] = round(float(missing_ratio_smooth50[i]), 3)

    # 7. Write to CSV
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fieldnames = list(records[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"Features saved to: {output_path} ({len(records):,} rows, {len(fieldnames)} columns)")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.2f} seconds ({len(records)/max(elapsed, 0.001):.0f} frames/sec).")
    return records


def main():
    parser = argparse.ArgumentParser(description="Extract ML features from tracking data.")
    parser.add_argument("--window", choices=["15min", "public", "full"], default="15min",
                        help="Window choice: '15min' (first 15 min of play, 22,500 frames), 'public' (5 min, 7500 frames), or 'full'.")
    parser.add_argument("--start", type=int, default=None, help="Custom start frame index.")
    parser.add_argument("--end", type=int, default=None, help="Custom end frame index.")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path.")
    args = parser.parse_args()

    paths = games.asset_paths(source="tactical")
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt_map = load_ground_truth(gt_path)

    if args.start is not None and args.end is not None:
        start_frame, end_frame = args.start, args.end
        default_out = os.path.join(ROOT, "data", f"features_{start_frame}_{end_frame}.csv")
    elif args.window == "15min":
        # 15 minutes of active match play from kickoff (15 * 60 * 25 = 22,500 frames)
        start_frame = KICKOFF_FRAME
        end_frame = KICKOFF_FRAME + (15 * 60 * FPS) - 1
        default_out = os.path.join(ROOT, "data", "features_first_15min_play.csv")
    elif args.window == "public":
        start_frame, end_frame = 11185, 18684
        default_out = os.path.join(ROOT, "data", "features_public_window.csv")
    else:
        start_frame, end_frame = 0, 199318
        default_out = os.path.join(ROOT, "data", "features_full_match.csv")

    output_path = args.output or default_out
    extract_features(paths["atd_raw"], gt_map, start_frame, end_frame, output_path)


if __name__ == "__main__":
    main()
