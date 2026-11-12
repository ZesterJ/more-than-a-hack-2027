"""Every feature is either PHYSICAL or TEMPORAL, never both.

physical  describes one moment: distances and counts around the ball, where the teams stand, team
          width, speeds, the ball's position relative to the lines. A new SIGNAL is physical.
temporal  describes how things change or how long they have lasted: rolling averages, forward and
          backward changes, time since or until the last touch, how stale the last seen ball is,
          anything carried forward from a past frame. A new WINDOW or AGGREGATION over an existing
          signal is temporal.

The rule that keeps two people from overlapping: temporal features are built from physical ones, never
the other way round. If you need a new signal, add it as a physical feature; if you want a new window
on a signal that exists, add it as a temporal feature.

    python -m experiments.feature_groups          # show the split and fail on anything unclassified
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

WINDOW_SUFFIX = re.compile(r"_(roll|fwd|back)\d+$")
TEMPORAL_PREFIXES = ("dyn_", "long_")
# history-dependent single columns: they look at frames other than the current one
TEMPORAL_NAMES = {"ball_gap", "ball_next_gap", "last_contact", "next_contact",
                  "since_contact", "until_contact"}


def group_of(name):
    """'physical' or 'temporal'."""
    if (WINDOW_SUFFIX.search(name) or name.startswith(TEMPORAL_PREFIXES)
            or name in TEMPORAL_NAMES or name.endswith("_ff")):
        return "temporal"
    return "physical"


def split(columns):
    out = {"physical": [], "temporal": []}
    for c in columns:
        out[group_of(c)].append(c)
    return out


def main():
    from experiments.features import load_features
    columns = list(load_features().columns)
    parts = split(columns)
    assert sorted(parts["physical"] + parts["temporal"]) == sorted(columns)
    assert not set(parts["physical"]) & set(parts["temporal"]), "a feature is in both groups"
    for name in ("physical", "temporal"):
        print("{} ({}):".format(name, len(parts[name])))
        print("  " + ", ".join(parts[name]))
        print()


if __name__ == "__main__":
    main()
