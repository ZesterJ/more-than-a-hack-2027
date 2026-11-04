"""Decoders, models and scoring for the external-data benchmark (04_external_benchmark.ipynb).

Scoring uses evaluation/possession_metrics.py -- the same code that produces the board. Competition rules:
the public ground truth is never used to train, fit or tune anything; it is only ever an evaluation target.
`score` exists for that final, one-shot report, and for scoring held-out *external* matches.

Posteriors are P[:, 0:3] = (H, A, D), the same order as the scorer's codes.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import LABEL_START, LABEL_STOP, FPS, pm  # noqa: E402

H, A, D, X = 0, 1, 2, 3


# -- helpers -----------------------------------------------------------------------------------

def _full_proba(model, X):
    """predict_proba as an (n, 3) array even if a training set never saw one of the classes."""
    out = np.zeros((len(X), 3))
    out[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(X)
    return out


# -- decoders: posteriors -> labels ------------------------------------------------------------

def smooth_proba(P, width):
    """Centred rolling mean of each class posterior over `width` frames, then renormalised."""
    if width <= 1:
        return P
    half = width // 2
    csum = np.vstack([np.zeros((1, P.shape[1])), np.cumsum(P, axis=0)])
    idx = np.arange(len(P))
    lo, hi = np.maximum(0, idx - half), np.minimum(len(P), idx + half + 1)
    out = (csum[hi] - csum[lo]) / (hi - lo)[:, None]
    return out / out.sum(axis=1, keepdims=True)


def viterbi(P, p_stay=0.999, direct_switch=0.05, prior_power=0.0, prior=None):
    """Most likely H/A/D path given per-frame posteriors as emissions.

    The transition matrix encodes the football: you stay in a state ~1000 frames (40 s) on average,
    and a direct H<->A hand-over is `direct_switch` times as likely as going through a stoppage.
    `prior_power` > 0 divides emissions by prior**power (turns posteriors into scaled likelihoods).
    """
    leave = 1.0 - p_stay
    T = np.array([[p_stay, leave * direct_switch / (1 + direct_switch), leave / (1 + direct_switch)],
                  [leave * direct_switch / (1 + direct_switch), p_stay, leave / (1 + direct_switch)],
                  [leave / 2, leave / 2, p_stay]])
    logT = np.log(T)
    E = np.log(np.clip(P, 1e-6, 1.0))
    if prior_power and prior is not None:
        E = E - prior_power * np.log(np.clip(prior, 1e-6, 1.0))
    n = len(P)
    score = np.zeros((n, 3))
    back = np.zeros((n, 3), dtype=np.int8)
    score[0] = E[0]
    for t in range(1, n):
        cand = score[t - 1][:, None] + logT
        back[t] = cand.argmax(axis=0)
        score[t] = cand.max(axis=0) + E[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = score[-1].argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def min_duration(codes, min_frames):
    """Absorb every run shorter than `min_frames` into its longer neighbour, shortest first."""
    codes = np.asarray(codes).copy()
    while True:
        change = np.flatnonzero(np.diff(codes)) + 1
        starts = np.concatenate(([0], change))
        stops = np.concatenate((change, [len(codes)]))
        lengths = stops - starts
        if len(lengths) < 2 or lengths.min() >= min_frames:
            return codes
        i = int(lengths.argmin())
        left = lengths[i - 1] if i > 0 else -1
        right = lengths[i + 1] if i + 1 < len(lengths) else -1
        codes[starts[i]:stops[i]] = codes[starts[i - 1]] if left >= right else codes[starts[i + 1]]


# -- scoring -----------------------------------------------------------------------------------

def score(pred_window, gt_codes, start=LABEL_START, bootstrap_n=0):
    """Score predictions for the labelled window with the official evaluator.

    `pred_window` holds codes 0..3 for video frames [start, start + len). Everything else is X,
    which the scorer never grades -- exactly how a real submission is read against the public GT.
    """
    full = np.full(len(gt_codes), X, dtype=np.int8)
    full[start:start + len(pred_window)] = pred_window
    return pm.evaluate(gt_codes, full, bootstrap_n=bootstrap_n)


def row(name, result):
    """One benchmark-table row from pm.evaluate output, in the PDF's column order."""
    pc = result["per_class"]
    return {
        "model": name,
        "spellF1": result["spell_f1_mean"],
        "lo": result["spell_bootstrap_low"], "hi": result["spell_bootstrap_high"],
        "macroF1": result["macro_f1"], "acc": result["accuracy"],
        "F1_H": pc["H"]["f1"], "F1_A": pc["A"]["f1"], "F1_D": pc["D"]["f1"],
        "spells": result["n_pred_spells"], "mean_dur_s": result["mean_spell_duration_s"],
    }


# -- hierarchical model -------------------------------------------------------------------------

class Hierarchical:
    """Dead-ball gate + live router: P(D) from a gate trained on all frames, P(H | live) from a router
    trained on live frames only. The product keeps the three posteriors summing to one."""

    classes_ = np.arange(3)

    def __init__(self, make_gate, make_router):
        self.make_gate, self.make_router = make_gate, make_router

    def fit(self, X, y):
        self.gate = self.make_gate().fit(X, (y == D).astype(int))
        live = y != D
        self.router = self.make_router().fit(X[live], y[live])
        return self

    def predict_proba(self, X):
        p_dead = self.gate.predict_proba(X)[:, 1]
        r = np.zeros((len(X), 2))
        r[:, np.asarray(self.router.classes_, dtype=int)] = self.router.predict_proba(X)
        return np.column_stack([(1 - p_dead) * r[:, 0], (1 - p_dead) * r[:, 1], p_dead])


# -- optional: sequence models (torch) ---------------------------------------------------------

def torch_sequence_model(arch, n_features, context=35, hidden=32, epochs=15, crop=150, seed=0):
    """Factory for per-frame sequence classifiers. Needs torch; import is deferred so the rest of
    this module works without it.

    arch: 'conv1d'     stacked dilated Conv1d, no recurrence
          'bigru'      2-layer bidirectional GRU
          'tcn_bigru'  dilated residual TCN (dilations 1, 2, 4) feeding a BiGRU
    `context` is the conv kernel span in frames for the conv parts.
    """
    import torch
    from torch import nn

    class TCNBlock(nn.Module):
        def __init__(self, ch, dilation):
            super().__init__()
            self.conv = nn.Conv1d(ch, ch, 5, padding=2 * dilation, dilation=dilation)
            self.norm = nn.BatchNorm1d(ch)

        def forward(self, x):
            return x + torch.relu(self.norm(self.conv(x)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch = arch
            self.inp = nn.Conv1d(n_features, hidden, 1)
            if arch in ("conv1d", "tcn_bigru"):
                self.tcn = nn.Sequential(*[TCNBlock(hidden, d) for d in (1, 2, 4)])
            if arch in ("bigru", "tcn_bigru"):
                self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                                  bidirectional=True, dropout=0.1)
                out = 2 * hidden
            else:
                out = hidden
            self.head = nn.Linear(out, 3)

        def forward(self, x):  # x: (batch, time, features)
            h = self.inp(x.transpose(1, 2))
            if self.arch in ("conv1d", "tcn_bigru"):
                h = self.tcn(h)
            h = h.transpose(1, 2)
            if self.arch in ("bigru", "tcn_bigru"):
                h, _ = self.gru(h)
            return self.head(h)

    class Model:
        classes_ = np.arange(3)

        def fit(self, Xtr, ytr):
            """Train on Xtr/ytr (the CV training rows, concatenated; `self.breaks` marks purge gaps).

            Optional multi-source training: set `self.extra` to a list of contiguous (X, y) segments (e.g. the
            external matches, full frame rate) and `self.bar_share` to the fraction of crops drawn from the
            target-match rows; `self.n_steps` fixes the number of optimiser steps.
            """
            torch.manual_seed(seed)
            rng = np.random.RandomState(seed)
            extra = getattr(self, "extra", None) or []
            sample = np.vstack([Xtr] + [e[0][::10] for e in extra])
            self.mu, self.sd = sample.mean(0), sample.std(0) + 1e-6
            Z = ((Xtr - self.mu) / self.sd).astype(np.float32)
            breaks = np.asarray(getattr(self, "breaks", []), dtype=int)
            edges = np.concatenate(([0], breaks, [len(Z)]))
            near = [(Z, ytr, a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a > crop]
            far = [(((e[0] - self.mu) / self.sd).astype(np.float32), e[1], 0, len(e[1])) for e in extra if len(e[1]) > crop]
            share = getattr(self, "bar_share", 1.0 if not far else 0.3)
            far_p = np.array([b - a for _, _, a, b in far], dtype=float)
            far_p = far_p / far_p.sum() if len(far) else far_p
            labels = np.concatenate([ytr] + [e[1][::10] for e in extra])
            self.net = Net()
            opt = torch.optim.AdamW(self.net.parameters(), lr=3e-3, weight_decay=1e-3)
            weights = torch.tensor(len(labels) / (3 * np.maximum(np.bincount(labels, minlength=3), 1)),
                                   dtype=torch.float32) ** 0.5
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            n_steps = getattr(self, "n_steps", None) or epochs * max(1, len(Z) // crop)
            self.net.train()
            for _ in range(n_steps):
                xs, ys = [], []
                for _ in range(16):
                    if far and (not near or rng.rand() >= share):
                        zz, yy, a, b = far[rng.choice(len(far), p=far_p)]
                    else:
                        zz, yy, a, b = near[rng.randint(len(near))]
                    st = rng.randint(a, b - crop + 1)
                    xs.append(zz[st:st + crop]); ys.append(yy[st:st + crop])
                xb, yb = torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys), dtype=torch.long)
                opt.zero_grad()
                loss = loss_fn(self.net(xb).reshape(-1, 3), yb.reshape(-1))
                loss.backward(); opt.step()
            return self

        def predict_proba(self, Xte):
            Z = torch.tensor(((Xte - self.mu) / self.sd).astype(np.float32))[None]
            self.net.eval()
            with torch.no_grad():
                return torch.softmax(self.net(Z)[0], dim=-1).numpy()

    return Model
