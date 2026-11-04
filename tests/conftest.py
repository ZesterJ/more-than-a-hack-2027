"""Make starter/ and evaluation/ importable for every test, and flag the data-dependent ones."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("starter", "evaluation", ""):
    sys.path.insert(0, os.path.join(ROOT, sub) if sub else ROOT)

import pytest  # noqa: E402


def have_atd():
    from games import asset_paths
    return os.path.isfile(asset_paths()["atd_raw"])


needs_atd = pytest.mark.skipif(not have_atd(), reason="match data not in data/ (see DATA.md)")
