"""Rebuilds starter/04_external_benchmark.ipynb (cells only, no outputs).

    python tests/notebook_builders/build_04_external_benchmark.py
    jupyter nbconvert --to notebook --execute starter/04_external_benchmark.ipynb --output 04_external_benchmark.ipynb
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook(); cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# 🧪 External-data benchmark (competition-compliant)

**Rules this notebook follows** (organisers' clarification): if you train, use only external data; every tuned parameter
is tuned on external data, never on the ground-truth possession data or the provided video. So:

* the **public ground truth is not read anywhere in this notebook**;
* models train on the seven **IDSSE** matches (Sportec/DFL, CC-BY 4.0; the provider's own `BallStatus` / `BallPossession`
  labels) and are validated **leave-one-match-out**: every match is predicted by a model that never saw it;
* decoder settings (smoothing width, minimum possession length, contact filter) are chosen on those external predictions;
  `tuned_on_others` below scores each match with settings picked from the *other* six, an honest transfer estimate;
* the provided match contributes only its tracking: as model input, and to copy where its ball goes unseen (so IDSSE's
  always-present ball has the same gaps).

Metrica's sample games are not used: their README states no licence (confirm with the organisers before adding them).
Caveat that matters: IDSSE's possession definition is more fragmented than the competition's, so absolute scores here are
not predictions of the leaderboard. Use them to compare model families and decoder rules.

Data: `python starter/external.py --fetch-idsse` (2.6 GB, git-ignored). Cite the IDSSE dataset (Bassek et al., figshare
doi 10.6084/m9.figshare.28196177).
""")
code("""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.abspath("..") if os.path.basename(os.getcwd()) == "starter" else os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "starter")); sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm
import external as ext
import external_pipeline as ep
pd.set_option("display.width", 200)
if not all(os.path.isfile(p) for m in ext.IDSSE_MATCHES for p in ext.idsse_files(m).values()):
    ext.fetch_idsse()
""")
md("## 1. Load the seven matches")
code("""
sets = ep.load_idsse_sets(ep.ball_unseen_pattern(), advanced=True, control_s=3, boundary=True,
                         motion=True, temporal=True, augment=True)  # 109 features, controlled-possession labels, mirrored copies for training
def stats(m, d):
    y = d["y"]; sp = pm.possession_spells(y, pm.valid_runs(y)); L = np.array([(b - a) / 25 for a, b, _ in sp])
    return dict(match=m, frames=len(y), H=(y == 0).mean(), A=(y == 1).mean(), D=(y == 2).mean(),
                per_5min=len(sp) / (len(y) / 25 / 300), median_s=np.median(L), short_lt_4s=(L < 4).mean())
pd.DataFrame([stats(m, d) for m, d in sets.items()]).round(3)
""")
md("## 2. Leave-one-match-out posteriors for each model family")
code("""
POSTS, t = {}, time.time()
for name in ep.MODELS:
    POSTS[name] = ep.lomo_posteriors(name, sets); print(name, round(time.time() - t), "s")
""")
md("""
## 3. Tune the decoder on external predictions; compare models
Per held-out match: `plain_w25` = smoothing only; `tuned_on_others` = decoder settings picked on the other six matches'
predictions. Means and standard deviations are over the seven matches (a match is the unit of evidence here: roughly
300-500 provider-labelled possessions each).
""")
code("""
TUNED, rows, details = {}, [], {}
for name in ep.MODELS:
    joint, table = ep.tune_decoder(POSTS[name], sets); TUNED[name] = joint; details[name] = pd.DataFrame(table)
    d = details[name]
    rows.append(dict(model=name, plain_w25=d.plain_w25.mean(), tuned_on_others=d.tuned_on_others.mean(),
                     sd_over_matches=d.tuned_on_others.std(), joint_decoder=str({k: v for k, v in joint.items() if k != "inflight_thr"})))
board = pd.DataFrame(rows).sort_values("tuned_on_others", ascending=False).reset_index(drop=True)
board.round(3)
""")
code("""
best = board.model.iloc[0]
print("selected on external data only:", best)
details[best].round(3)
""")
md("""
## 3b. Domain weighting (inputs only)
This match's tracking differs from the Bundesliga matches (the ball is unseen far more often, different tempo). A small
classifier learns to tell external frames from this match's frames **using the tracking features alone, no labels**, and
the external frames are weighted by how much they resemble this match. Weights are cross-fitted by match. Applies to the
LightGBM family; kept because it shrinks the input shift (7 strongly shifted features become 1) and costs at most 0.01 of
leave-one-match-out score (with the temporal features it is now 0.004 lower, so its benefit is mainly the smaller shift).
""")
code("""
Xt = ep.target_features()
W = ep.domain_weights(sets, Xt)
Xs = np.vstack([d["X"][::6] for d in sets.values()]); ws = np.concatenate([W[m][::6] for m in sets])
smd = lambda mu: np.abs(Xt.mean(0) - mu) / (Xs.std(0) + 1e-9)
before, after = smd(Xs.mean(0)), smd((Xs * ws[:, None]).sum(0) / ws.sum())
print("features shifted by > 0.3 sd:  before %d  after %d   (effective sample size %.0f%%)" %
      ((before > .3).sum(), (after > .3).sum(), 100 * ws.sum() ** 2 / (ws ** 2).sum() / len(ws)))
POSTS_W = ep.lomo_posteriors("lightgbm", sets, W)
joint_w, table_w = ep.tune_decoder(POSTS_W, sets)
pd.DataFrame([dict(model="lightgbm", weighting="none", tuned_on_others=details["lightgbm"].tuned_on_others.mean()),
              dict(model="lightgbm", weighting="domain", tuned_on_others=pd.DataFrame(table_w).tuned_on_others.mean())]).round(3)
""")
md("""
## 4. Export the full-match submission
Train the selected family on all seven matches, apply the externally-tuned decoder to this match's two halves, write the
CSV, and validate its format. Then run the official scorer **once** from a terminal, to report the number
(`python evaluation/run_evaluation.py`); nothing above this line depends on it, and it must not be used to pick between
models, weights or settings.
""")
code("""
import subprocess
from generate_full_submission import predict_and_write
use_w = best == "lightgbm"      # domain weighting applies to the LightGBM family
params = joint_w if use_w else TUNED[best]
path = predict_and_write(ep.fit(best, sets, weights=W if use_w else None), params, "ext_%s.csv" % best)
print(subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), path],
                     capture_output=True, text=True).stdout[:900])
""")
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "starter", "04_external_benchmark.ipynb"))
