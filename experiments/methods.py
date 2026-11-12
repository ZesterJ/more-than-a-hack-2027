"""Reference methods and the registry. Copy one to start your own.

A method is any object with:

    fit(feats, labels, train_idx)   learn from the labelled frames in train_idx (may do nothing)
    predict(feats) -> codes         one code (0 H, 1 A, 2 D, 3 X) for EVERY frame of the match

`feats` is the table from experiments/features.py (one row per video frame). `labels` is the
public ground truth as codes over the whole match (X outside the labelled window); only read it at
train_idx. To add a method, write a function that returns an object like the ones below and add it
to REGISTRY.
"""

import os

import numpy as np
import pandas as pd

from experiments.features import DYNAMICS_EXCLUDE, DYNAMICS_PREFIX, EXPERIMENTAL, LONG_CONTEXT_PREFIX
from experiments.postprocess import H, A, D, vote_smooth, smooth_probs, min_duration_filter


class NearestPlayer:
    """Whoever is nearest the ball has it, then a majority vote. No learning: the repo baseline."""

    def __init__(self, window=50):
        self.window = window

    def fit(self, feats, labels, train_idx):
        pass

    def predict(self, feats):
        d_home, d_away = feats["d_home"].to_numpy(), feats["d_away"].to_numpy()
        seen = ~(np.isnan(d_home) | np.isnan(d_away))
        raw = np.full(len(feats), -1, dtype=np.int8)  # -1: no ball, so nobody votes
        raw[seen] = np.where(d_home[seen] <= d_away[seen], H, A)
        return vote_smooth(raw, self.window)


class LogisticSmooth:
    """Multinomial logistic regression on the collective features, smoothed over time.

    Missing values are filled with the training median and a 'was missing' flag is kept for the
    columns that are often NaN. Probabilities are averaged over +/- window frames before the
    argmax, then runs shorter than min_dur frames are merged away.
    """

    def __init__(self, window=50, min_dur=0, C=1.0):
        self.window, self.min_dur, self.C = window, min_dur, C

    def _matrix(self, feats):
        x = feats[self.columns_].to_numpy(dtype=float)
        flags = np.isnan(x[:, self.flag_cols_]).astype(float)
        x = np.where(np.isnan(x), self.median_, x)
        return (x - self.mean_) / self.std_, flags

    def fit(self, feats, labels, train_idx):
        from sklearn.linear_model import LogisticRegression  # optional dependency
        self.columns_ = list(feats.columns)
        x = feats.iloc[train_idx].to_numpy(dtype=float)
        self.median_ = np.nan_to_num(np.nanmedian(x, axis=0), nan=0.0)
        self.flag_cols_ = [i for i in range(x.shape[1]) if np.isnan(x[:, i]).mean() > 0.01]
        filled = np.where(np.isnan(x), self.median_, x)
        self.mean_, self.std_ = filled.mean(axis=0), filled.std(axis=0) + 1e-6
        z, flags = self._matrix(feats.iloc[train_idx])
        self.model_ = LogisticRegression(C=self.C, max_iter=500)
        self.model_.fit(np.hstack([z, flags]), labels[train_idx])

    def predict(self, feats):
        z, flags = self._matrix(feats)
        proba = self.model_.predict_proba(np.hstack([z, flags]))
        full = np.zeros((len(feats), 3))
        full[:, self.model_.classes_] = proba  # classes_ are the codes seen in training
        codes = smooth_probs(full, self.window)
        return min_duration_filter(codes, self.min_dur)


_EXT_CACHE = {}   # (match, seed, label_min_dur) -> (features, labels), reused across folds and runs
_MODEL_CACHE = {}  # fitted models that do not depend on the fold, keyed by their parameters


def external_match(mid, seed, templates=None, stress="none"):
    """(features, raw labels, half) for one external match with Barcelona-style gaps, cached on disk.

    The features do not depend on any label convention, so one cache serves every label setting.
    """
    import pickle
    from experiments import external as ext
    from experiments.features import CACHE_VERSION
    key = (mid, seed, stress)
    if key in _EXT_CACHE:
        return _EXT_CACHE[key]
    suffix = "" if stress == "none" else "_" + stress
    path = os.path.join(ext.EXT_DIR, "featcache_v{}_{}_{}{}.pkl".format(CACHE_VERSION, mid, seed, suffix))
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            _EXT_CACHE[key] = pickle.load(handle)
    else:
        templates = templates or _EXT_CACHE.setdefault("templates", ext.barca_missingness())
        _EXT_CACHE[key] = ext.match_table(mid, templates, seed=seed, stress=stress)
        with open(path, "wb") as handle:
            pickle.dump(_EXT_CACHE[key], handle)
    return _EXT_CACHE[key]


def align_labels(labels, half, label_min_dur):
    """Drop possession blips shorter than label_min_dur frames, per half so no run crosses the join."""
    if not label_min_dur:
        return labels
    out = labels.copy()
    for h in (1, 2):
        sel = half == h
        out[sel] = min_duration_filter(labels[sel], label_min_dur)
    return out


def _external_tables(matches, copies, label_min_dur, step, seed_offset=0, stress="none"):
    """Training rows from the external matches, each replayed with Barcelona-style gaps."""
    xs, ys = [], []
    for mid in matches:
        for seed in range(seed_offset, seed_offset + copies):
            feats, raw, half = external_match(mid, seed, stress=stress)
            labels = align_labels(raw, half, label_min_dur)
            keep = np.flatnonzero(labels != 3)[::step]
            xs.append(feats.iloc[keep])
            ys.append(labels[keep])
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def _training_tables(matches, copies, label_min_dur, step, stresses, seed_offset=0):
    parts = [_external_tables(matches, copies, label_min_dur, step, seed_offset, stress=s) for s in stresses]
    return pd.concat([p[0] for p in parts], ignore_index=True), np.concatenate([p[1] for p in parts])


class ProbModel:
    """Frame classifier returning P(H), P(A), P(D) as an (n, 3) array.

    two_stage=False: one multiclass gradient-boosted model.
    two_stage=True : P(alive) from a live-vs-dead model, and P(home | alive) from a model trained on
                     live frames only, so stoppages cannot blur the team decision.
                     P(H) = P(alive) P(home | alive), P(A) = P(alive) (1 - P(home | alive)), P(D) = 1 - P(alive).
    """

    def __init__(self, two_stage=False, max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                 min_samples_leaf=100):
        self.two_stage = two_stage
        self.hgb = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                        min_samples_leaf=min_samples_leaf, l2_regularization=1.0, random_state=0)

    def fit(self, x, y, sample_weight=None):
        from sklearn.ensemble import HistGradientBoostingClassifier as HGB
        y = np.asarray(y)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        if not self.two_stage:
            self.model_ = HGB(**self.hgb).fit(x, y, sample_weight=w)
            return self
        self.alive_ = HGB(**self.hgb).fit(x, (y != 2).astype(int), sample_weight=w)
        live = y != 2
        self.team_ = HGB(**self.hgb).fit(x[live], (y[live] == 0).astype(int), sample_weight=w[live])
        return self

    def predict_proba(self, x):
        if not self.two_stage:
            out = np.zeros((len(x), 3))
            out[:, self.model_.classes_] = self.model_.predict_proba(x)
            return out
        alive = self.alive_.predict_proba(x)[:, 1]
        home = self.team_.predict_proba(x)[:, 1]
        return np.column_stack([alive * home, alive * (1 - home), 1 - alive])


class BaggedProbModel:
    """Average of several ProbModels, each trained on a different set of gap-corrupted copies."""

    def __init__(self, models):
        self.models = models

    def predict_proba(self, x):
        return np.mean([m.predict_proba(x) for m in self.models], axis=0)


def reset_caches():
    """Forget fitted models (not the external feature tables, which never depend on Barcelona
    labels). The harness calls this between its two passes so the leak check is meaningful."""
    _MODEL_CACHE.clear()


class ExternalGBDT:
    """Gradient boosting trained on external matches (plus, optionally, Barcelona frames), decoded
    into possessions.

    ext            external match ids to train on ([] trains on Barcelona frames only)
    copies         corrupted replays of each external match (different gap patterns)
    label_min_dur  drop external possession blips shorter than this many frames before training
    barca_weight   weight of the Barcelona training frames (0 = never trained on Barcelona)
    drop           feature-name prefixes to leave out (e.g. ['ball_speed'], see the note in __init__)
    long_context   also use the 6 s / 10 s window features (long_*)
    dynamics       also use the ball-independent team-movement features (dyn_*, back lines excluded)
    train_stress   gap/shift variants of the external matches to train on, e.g. ('none', 'both'); see external.match_table
    extras         also use the contact / line features (off: they hurt on Barcelona, see FINDINGS.md)
    bags           train this many models on different gap-corrupted copies and average them
    two_stage      live-vs-dead model, then a home-vs-away model on live frames (see ProbModel)
    d_bias         added to the log-probability of dead ball (corrects the dead-ball prior, which is
                   much higher in the Bundesliga matches than in the Barcelona truth)
    decode         'vote' (average probabilities over +/- window frames) or 'viterbi'
    window         half-width for 'vote'
    switch_team / switch_dead / temperature   for 'viterbi', see postprocess.viterbi
    """

    def __init__(self, ext=(), copies=2, label_min_dur=0, barca_weight=0.0, drop=(), extras=False, long_context=False, dynamics=False, train_stress=("none",), two_stage=False, bags=1, d_bias=0.0,
                 decode="vote", window=50, switch_team=40.0, switch_dead=40.0, switch_restart=None, temperature=1.0,
                 step=5, max_iter=150, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100):
        self.__dict__.update(locals())
        del self.__dict__["self"]
        self.ext = list(ext)
        self.drop = (list(drop) + ([] if extras else list(EXPERIMENTAL))
                     + ([] if long_context else [LONG_CONTEXT_PREFIX])
                     + (list(DYNAMICS_EXCLUDE) if dynamics else [DYNAMICS_PREFIX]))
        self.train_stress = list(train_stress)

    def _fit_params(self):
        keys = ("ext", "copies", "label_min_dur", "barca_weight", "drop", "train_stress", "two_stage", "bags", "step", "max_iter",
                "learning_rate", "max_leaf_nodes", "min_samples_leaf")
        return tuple((k, tuple(getattr(self, k)) if isinstance(getattr(self, k), list) else getattr(self, k))
                     for k in keys)

    def fit(self, feats, labels, train_idx):
        self.columns_ = [c for c in feats.columns if not any(c.startswith(p) for p in self.drop)]
        key = self._fit_params()
        fold_free = bool(self.ext) and self.barca_weight == 0   # never sees Barcelona frames
        if fold_free and key in _MODEL_CACHE:                   # so it is the same model in every fold
            self.model_ = _MODEL_CACHE[key]
            return
        if self.bags > 1:
            if not self.ext or self.barca_weight:
                raise ValueError("bags > 1 is only supported for external-only training")
            members = []
            for b in range(self.bags):
                xb, yb = _training_tables(self.ext, self.copies, self.label_min_dur, self.step, self.train_stress, b * self.copies)
                members.append(ProbModel(self.two_stage, self.max_iter, self.learning_rate, self.max_leaf_nodes,
                                         self.min_samples_leaf).fit(xb[self.columns_], yb))
            self.model_ = BaggedProbModel(members)
            _MODEL_CACHE[key] = self.model_
            return
        parts_x, parts_y, parts_w = [], [], []
        if self.ext:
            x, y = _training_tables(self.ext, self.copies, self.label_min_dur, self.step, self.train_stress)
            parts_x.append(x), parts_y.append(y), parts_w.append(np.ones(len(y)))
        if self.barca_weight > 0 or not self.ext:
            idx = np.asarray(train_idx)[::max(1, self.step // 5)]
            parts_x.append(feats.iloc[idx]), parts_y.append(labels[idx])
            parts_w.append(np.full(len(idx), self.barca_weight if self.ext else 1.0))
        x, y, w = pd.concat(parts_x, ignore_index=True), np.concatenate(parts_y), np.concatenate(parts_w)
        model = ProbModel(self.two_stage, self.max_iter, self.learning_rate, self.max_leaf_nodes,
                          self.min_samples_leaf)
        model.fit(x[self.columns_], y, sample_weight=w)
        self.model_ = model
        if fold_free:
            _MODEL_CACHE[key] = model

    def predict(self, feats):
        from experiments.postprocess import viterbi
        from games import GAME, half_video_blocks
        proba = self.model_.predict_proba(feats[self.columns_])
        proba[:, 2] *= np.exp(self.d_bias)
        proba = np.clip(proba / proba.sum(axis=1, keepdims=True), 1e-6, 1.0)
        out = np.full(len(feats), 3, dtype=np.int8)
        for _, first, last in half_video_blocks(GAME):       # decode each half on its own
            block = slice(first, last + 1)
            if self.decode == "viterbi":
                out[block] = viterbi(np.log(proba[block]), self.switch_team, self.switch_dead, self.temperature,
                                    self.switch_restart)
            else:
                out[block] = smooth_probs(proba[block], self.window)
        return out


REGISTRY = {
    "nearest": NearestPlayer,
    "logistic": LogisticSmooth,
    "gbdt": ExternalGBDT,
}
