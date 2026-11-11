"""Full-match submission from a model trained and tuned on external data only.

    python starter/external.py --fetch-idsse            # once, 2.6 GB into data/external/idsse/ (CC-BY 4.0)
    python starter/generate_full_submission.py --model lightgbm --out ext_lightgbm.csv

Competition rules: training uses only external data and every tuned parameter is tuned on external data. This
script therefore never reads the public ground truth. The recipe:

  1. features: 30 collective / kinematic features (features.py) + 20 multi-scale / dwell / shape ones
     (external_pipeline.advanced_block; +0.022 leave-one-match-out, better on 7/7 matches), pitch oriented so the home team attacks +x in
     every half of every match;
  2. model: trained on the seven IDSSE matches (provider labels), every 6th frame;
  3. decoder: smoothing / duration filter / contact filter chosen by leave-one-match-out on those same matches;
  4. each half of this match decoded separately; frames outside the two halves are X.

The run is deterministic, so the same inputs regenerate the same file.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import external as ext  # noqa: E402
import external_pipeline as ep  # noqa: E402
from baseline import write_submission  # noqa: E402
from benchmark import _full_proba  # noqa: E402
from features import load_tracking, team_of_each_track  # noqa: E402
from games import GAME, GAMES, asset_paths  # noqa: E402
from tactical import decode_pipeline  # noqa: E402


def predict_and_write(model, params, out_name, advanced=True, boundary=True, motion=True, temporal=True):
    """Decode this match's two halves with an already-trained `model` and write the submission CSV."""
    n_frames = GAMES[GAME]["n_video_frames"]
    offsets, halves = GAMES[GAME]["offsets"], GAMES[GAME]["halves"]
    spans = [(halves[h][0] - offsets[h], halves[h][1] - offsets[h] + 1) for h in (1, 2)]
    trk = load_tracking(n_frames)
    is_home = team_of_each_track(asset_paths()["atd_meta"]) == 0
    trk = ext.orient_home_attacks_right(trk, is_home, spans)
    X, aux = ep.featureset(trk, is_home, advanced, boundary, motion, temporal)

    codes = np.full(n_frames, 3, dtype=np.int8)
    for a, b in spans:
        codes[a:b] = decode_pipeline(_full_proba(model, X[a:b]), {k: v[a:b] for k, v in aux.items()}, **params)
    path = os.path.join(ROOT, "evaluation", "results", out_name)
    write_submission(path, np.array(list("HADX"))[codes])
    print("wrote", path)
    return path


def main(out_name, model_name, advanced=True, control_s=3.0, boundary=True, motion=True, augment=True, domain=True, temporal=True):
    sets = ep.load_idsse_sets(ep.ball_unseen_pattern(), advanced=advanced, control_s=control_s or None,
                              boundary=boundary, motion=motion, augment=augment, temporal=temporal)
    weights = None
    if domain:  # weight the external frames to resemble this match's tracking (inputs only, no labels)
        weights = ep.domain_weights(sets, ep.target_features(advanced, boundary, motion, temporal))
    posts = ep.lomo_posteriors(model_name, sets, weights)
    params, _ = ep.tune_decoder(posts, sets)
    print("decoder (leave-one-match-out on IDSSE): {}".format(params))
    return predict_and_write(ep.fit(model_name, sets, weights=weights), params, out_name, advanced, boundary, motion, temporal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="ext_lightgbm.csv")
    parser.add_argument("--model", choices=sorted(ep.MODELS), default="lightgbm")
    parser.add_argument("--base-features", action="store_true", help="30 features instead of 50")
    parser.add_argument("--control-s", type=float, default=3.0,
                        help="relabel IDSSE to controlled possession (runs shorter than this join the previous team); 0 = raw")
    parser.add_argument("--no-boundary", action="store_true", help="drop the 16 boundary features")
    parser.add_argument("--no-temporal", action="store_true", help="drop the 31 temporal-context features")
    parser.add_argument("--no-motion", action="store_true", help="drop the 12 motion features")
    parser.add_argument("--no-domain", action="store_true", help="no input-only domain weighting of the external frames")
    parser.add_argument("--no-augment", action="store_true", help="train without mirrored copies of the matches")
    args = parser.parse_args()
    main(args.out, args.model, advanced=not args.base_features, control_s=args.control_s,
         boundary=not args.no_boundary, motion=not args.no_motion, augment=not args.no_augment,
         domain=not args.no_domain, temporal=not args.no_temporal)
