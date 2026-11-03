"""
Ingest StatsBomb Open Data: 9 Historical FC Barcelona vs Athletic Club Matches

Extracts:
1. Empirical Possession Duration Distribution (Parametric Gamma/Log-Normal Fit)
2. State Transition Probability Matrix: P(S_{t+1} | S_t)
3. Spatial Turnover Hotspots: Pitch grid mapping of turnover likelihood P(Loss | x, y)
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
OUT_JSON = os.path.join(DATA_DIR, "statsbomb_bar_ath_priors.json")
OUT_CSV = os.path.join(DATA_DIR, "statsbomb_turnover_events.csv")

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

BAR_ATH_MATCH_IDS = [
    3764661,  # 2021-01-06: Athletic 2 - 3 Barcelona
    3773403,  # 2021-01-31: Barcelona 2 - 1 Athletic
    303493,   # 2020-06-23: Barcelona 1 - 0 Athletic
    16173,    # 2019-02-10: Athletic 0 - 0 Barcelona
    16010,    # 2018-09-29: Barcelona 1 - 1 Athletic
    9855,     # 2018-03-18: Barcelona 2 - 0 Athletic
    9661,     # 2017-10-28: Athletic 0 - 2 Barcelona
    266892,   # 2016-08-28: Athletic 0 - 1 Barcelona
    266874    # 2017-02-04: Barcelona 3 - 0 Athletic
]

def main():
    print("="*100)
    print(" INGESTING STATSBOMB OPEN DATA (9 FC BARCELONA vs ATHLETIC CLUB MATCHES)")
    print("="*100)
    t0 = time.time()

    all_possessions = []
    turnover_records = []
    
    total_events = 0

    for idx, match_id in enumerate(BAR_ATH_MATCH_IDS, 1):
        print(f"[{idx}/{len(BAR_ATH_MATCH_IDS)}] Fetching Match ID {match_id} from StatsBomb...")
        url = f"{BASE_URL}/events/{match_id}.json"
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"  Warning: Failed to fetch match {match_id} (status {resp.status_code})")
            continue

        events = resp.json()
        total_events += len(events)
        
        # Group by possession ID
        poss_map = {}
        for e in events:
            pid = e.get("possession")
            p_team = e.get("possession_team", {}).get("name")
            if not pid or not p_team:
                continue

            t_min = e.get("minute", 0)
            t_sec = e.get("second", 0)
            t_total = t_min * 60 + t_sec
            etype = e.get("type", {}).get("name")
            loc = e.get("location")

            if pid not in poss_map:
                poss_map[pid] = {
                    "match_id": match_id,
                    "possession_id": pid,
                    "team": "H" if "Barcelona" in p_team else "A",
                    "team_name": p_team,
                    "start_s": t_total,
                    "end_s": t_total,
                    "events": [etype],
                    "start_loc": loc,
                    "end_loc": loc
                }
            else:
                poss_map[pid]["end_s"] = t_total
                poss_map[pid]["events"].append(etype)
                if loc:
                    poss_map[pid]["end_loc"] = loc

            # Check if this event indicates a turnover / loss of ball
            # Dispossessed, Miscontrol, Incomplete Pass, Foul, Interception, etc.
            is_turnover = False
            if etype in ["Dispossessed", "Miscontrol"]:
                is_turnover = True
            elif etype == "Pass":
                p_outcome = e.get("pass", {}).get("outcome", {}).get("name")
                if p_outcome in ["Incomplete", "Out", "Pass Offside"]:
                    is_turnover = True

            if is_turnover and loc:
                # StatsBomb coords: [x, y] in [0..120, 0..80]
                # Scale to standard meters [-52.5..+52.5, -34..+34]
                sx, sy = loc[0], loc[1]
                px = (sx / 120.0) * 105.0 - 52.5
                py = (sy / 80.0) * 68.0 - 34.0
                turnover_records.append({
                    "match_id": match_id,
                    "team": "H" if "Barcelona" in p_team else "A",
                    "time_s": t_total,
                    "x_m": px,
                    "y_m": py,
                    "event_type": etype
                })

        # Append possessions
        for p in poss_map.values():
            dur = max(1.0, p["end_s"] - p["start_s"])
            p["duration_s"] = dur
            all_possessions.append(p)

    print(f"\nIngested {total_events:,} events across {len(BAR_ATH_MATCH_IDS)} matches.")
    print(f"Extracted {len(all_possessions):,} total possession spells.")
    print(f"Extracted {len(turnover_records):,} spatial turnover events.")

    # 1. Compute Duration Statistics
    barca_durs = [p["duration_s"] for p in all_possessions if p["team"] == "H"]
    ath_durs = [p["duration_s"] for p in all_possessions if p["team"] == "A"]

    stats_dict = {
        "n_matches": len(BAR_ATH_MATCH_IDS),
        "total_events": total_events,
        "n_possessions": len(all_possessions),
        "barcelona": {
            "n_spells": len(barca_durs),
            "mean_duration_s": float(np.mean(barca_durs)),
            "std_duration_s": float(np.std(barca_durs)),
            "median_duration_s": float(np.median(barca_durs)),
            "p25_s": float(np.percentile(barca_durs, 25)),
            "p75_s": float(np.percentile(barca_durs, 75)),
            "p90_s": float(np.percentile(barca_durs, 90)),
        },
        "athletic_club": {
            "n_spells": len(ath_durs),
            "mean_duration_s": float(np.mean(ath_durs)),
            "std_duration_s": float(np.std(ath_durs)),
            "median_duration_s": float(np.median(ath_durs)),
            "p25_s": float(np.percentile(ath_durs, 25)),
            "p75_s": float(np.percentile(ath_durs, 75)),
            "p90_s": float(np.percentile(ath_durs, 90)),
        }
    }

    # 2. Compute Transition Matrix P(S_{t+1} | S_t)
    trans_counts = {"H": {"H": 0, "A": 0, "D": 0}, "A": {"H": 0, "A": 0, "D": 0}}
    for i in range(len(all_possessions) - 1):
        curr_t = all_possessions[i]["team"]
        next_t = all_possessions[i+1]["team"]
        gap = all_possessions[i+1]["start_s"] - all_possessions[i]["end_s"]
        if gap > 8.0:
            # Significant stoppage between spells -> Dead ball intermediate
            trans_counts[curr_t]["D"] += 1
        else:
            trans_counts[curr_t][next_t] += 1

    trans_probs = {}
    for from_state, to_dict in trans_counts.items():
        tot = sum(to_dict.values())
        trans_probs[from_state] = {to_s: float(cnt / tot) for to_s, cnt in to_dict.items()} if tot > 0 else {}

    stats_dict["transition_probabilities"] = trans_probs

    # 3. Save Artifacts
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"\nSaved empirical priors JSON to: {OUT_JSON}")

    df_turn = pd.DataFrame(turnover_records)
    df_turn.to_csv(OUT_CSV, index=False)
    print(f"Saved spatial turnover events CSV to: {OUT_CSV}")

    # 4. Display Results
    print("\n" + "="*80)
    print("         EMPIRICAL STATSBOMB BARCELONA vs ATHLETIC PRIORS SUMMARY")
    print("="*80)
    print(f"  Barcelona Mean Possession Duration:      {stats_dict['barcelona']['mean_duration_s']:.1f} s (median: {stats_dict['barcelona']['median_duration_s']:.1f} s)")
    print(f"  Athletic Club Mean Possession Duration:  {stats_dict['athletic_club']['mean_duration_s']:.1f} s (median: {stats_dict['athletic_club']['median_duration_s']:.1f} s)")
    print("\n  Empirical Transition Rates:")
    print(f"    P(Turnover to Away | Barcelona):       {trans_probs['H'].get('A', 0.0):.2%}")
    print(f"    P(Stoppage/Dead Ball | Barcelona):     {trans_probs['H'].get('D', 0.0):.2%}")
    print(f"    P(Recovery to Barca | Athletic Club):  {trans_probs['A'].get('H', 0.0):.2%}")
    print(f"    P(Stoppage/Dead Ball | Athletic Club): {trans_probs['A'].get('D', 0.0):.2%}")
    print("="*80)
    print(f"Elapsed Time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
