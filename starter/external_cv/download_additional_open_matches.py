"""
Download Additional Open European Matches (SkillCorner Open Data)
Strict Compliance with Organizer Rule Clarification (Nov 3, 2026)

MATCHES AVAILABLE (MIT Open Source License):
- Match 2417: FC Bayern Munchen vs Borussia Dortmund (2019-11-09)
- Match 3749: Borussia Dortmund vs FC Bayern Munchen (2020-05-26)
- Match 2269: Paris Saint-Germain vs Olympique de Marseille (2019-10-27)

Usage:
    python starter/external_cv/download_additional_open_matches.py
"""

import os
import time
import json
import urllib.request

COMMIT = '1ed50b775f93075c2a6d00d0d795f13b7b89709e'
BASE_URL = f'https://raw.githubusercontent.com/SkillCorner/opendata/{COMMIT}/data/matches'

MATCHES = [
    (2417, "Bayern_vs_Dortmund"),
    (3749, "Dortmund_vs_Bayern"),
    (2269, "PSG_vs_Marseille")
]

def download_matches():
    os.makedirs('data/skillcorner', exist_ok=True)
    print("="*90)
    print(" DOWNLOADING ADDITIONAL ELITE EUROPEAN TRACKING MATCHES (MIT LICENSE)")
    print("="*90)
    
    for mid, name in MATCHES:
        meta_url = f'{BASE_URL}/{mid}/match_data.json'
        meta_path = f'data/skillcorner/{mid}_match_data.json'
        struct_url = f'{BASE_URL}/{mid}/structured_data.json'
        struct_path = f'data/skillcorner/{mid}_structured_data.json'
        
        print(f"\n--- Checking Match {mid} ({name}) ---")
        if not os.path.exists(meta_path):
            req = urllib.request.Request(meta_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                with open(meta_path, 'wb') as f:
                    f.write(r.read())
            print(f"  Downloaded metadata: {meta_path}")
        else:
            print(f"  Metadata already exists: {meta_path}")
            
        if os.path.exists(struct_path) and os.path.getsize(struct_path) > 10_000_000:
            print(f"  Structured tracking already downloaded ({os.path.getsize(struct_path)/(1024*1024):.1f} MB).")
            continue
            
        print(f"  Downloading structured tracking data ({struct_url})...")
        t0 = time.time()
        req = urllib.request.Request(struct_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as r:
            with open(struct_path, 'wb') as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        print(f"  Downloaded {os.path.getsize(struct_path)/(1024*1024):.1f} MB in {time.time()-t0:.1f}s.")

if __name__ == '__main__':
    download_matches()
