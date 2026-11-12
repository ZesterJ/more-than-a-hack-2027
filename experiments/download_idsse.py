"""Download the open DFL / IDSSE Bundesliga matches used as external training data.

    python -m experiments.download_idsse                 # all 7 matches (about 2.6 GB)
    python -m experiments.download_idsse J03WMX J03WN1   # only these matches

Source: Bassek et al. (2025), "An integrated dataset of spatiotemporal and event data in elite
soccer", Scientific Data, https://doi.org/10.6084/m9.figshare.28196177 -- CC BY 4.0, released with
DFL authorisation. Files go to data/external/idsse/ (gitignored). Re-running skips files that are
already complete.
"""

import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import DATA_DIR  # noqa: E402

API = "https://api.figshare.com/v2/articles/28196177"
DEST = os.path.join(DATA_DIR, "external", "idsse")
KINDS = ("matchinformation", "positions_raw_observed")  # events are not needed here


def main(wanted):
    os.makedirs(DEST, exist_ok=True)
    with urllib.request.urlopen(API, timeout=60) as response:
        files = json.load(response)["files"]
    todo = [f for f in files if any(k in f["name"] for k in KINDS)
            and (not wanted or any(w in f["name"] for w in wanted))]
    if not todo:
        raise SystemExit("no files matched {}".format(wanted))
    for f in todo:
        path = os.path.join(DEST, f["name"])
        if os.path.isfile(path) and os.path.getsize(path) == f["size"]:
            print("have  ", f["name"])
            continue
        print("fetch ", f["name"], "({:.0f} MB)".format(f["size"] / 1e6), flush=True)
        partial = path + ".part"
        with urllib.request.urlopen(f["download_url"], timeout=60) as src, open(partial, "wb") as out:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        if os.path.getsize(partial) != f["size"]:
            raise SystemExit("{}: incomplete download".format(f["name"]))
        os.replace(partial, path)
    print("done ->", DEST)


if __name__ == "__main__":
    main(sys.argv[1:])
