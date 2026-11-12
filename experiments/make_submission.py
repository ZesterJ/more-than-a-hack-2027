"""Train on the external matches and write the full-match submission. Nothing is scored.

    python -m experiments.make_submission my_team.csv
    python -m experiments.make_submission my_team.csv extras=True label_min_dur=75 bags=3

This script never reads the ground-truth possession file and prints nothing derived from the
provided match data except the validator's one-line verdict. The model is trained only on the
external Bundesliga matches (see download_idsse.py); the Barcelona tracking is only the input it is
applied to. Run evaluation/run_evaluation.py yourself, once, after the configuration is frozen.

The configuration is chosen with experiments/lomo.py, which scores held-out EXTERNAL matches only.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from experiments import external, harness  # noqa: E402
from experiments.features import load_features  # noqa: E402
from experiments.methods import ExternalGBDT  # noqa: E402
from experiments.run import parse_params  # noqa: E402

# The frozen configuration. Change it only on the evidence of experiments/lomo.py.
# Chosen on held-out EXTERNAL matches (leave-one-match-out, spellF1 about 0.69-0.70 across decoder settings):
#   extras=True      touch/line features helped there
#   dynamics=True    ball-independent team-movement features (centroids, front lines, width, speed over the
#                    previous/next 2 s and 6 s): +0.07 spellF1 and +0.07 turnover detection, and they hold up when
#                    the ball is lost around turnovers (+0.10). The deepest-player series are excluded: on their
#                    own they separate Barcelona from every external match (AUC 0.96 vs 0.60), a tracking artefact.
#   train_stress     train on standard gaps plus copies with the ball also lost around possession changes
#   bags=3           an average of 3 models, +0.01 spellF1
#   25 / 40          decoder plateau (team switch cost 20-40 all within 0.012); 219 predicted vs 221 true possessions
#   label_min_dur=75 a 3 s minimum possession, from the challenge text ("keeps the ball for a few seconds")
DEFAULTS = dict(copies=2, drop=["ball_speed"], extras=True, dynamics=True, train_stress=["none", "both"],
                bags=3, label_min_dur=75, decode="viterbi", switch_team=25, switch_dead=40)


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    path, params = argv[0], dict(DEFAULTS, **parse_params(argv[1:]))
    matches = external.match_ids()
    if not matches:
        raise SystemExit("no external matches in data/external/idsse -- run: python -m experiments.download_idsse")
    params.setdefault("ext", matches)
    model = ExternalGBDT(**params)
    model.fit(load_features(), None, [])          # external-only: labels and train_idx are never read
    codes = harness.apply_known_structure(model.predict(load_features()))
    harness.write_submission(codes, path)         # raises if the official validator rejects it
    print("trained on {} external matches; wrote and validated {}".format(len(params["ext"]), path))


if __name__ == "__main__":
    main(sys.argv[1:])
