"""Reference methods and the registry. Copy one to start your own.

A method is any object with:

    fit(feats, labels, train_idx)   learn from the labelled frames in train_idx (may do nothing)
    predict(feats) -> codes         one code (0 H, 1 A, 2 D, 3 X) for EVERY frame of the match

`feats` is the table from experiments/features.py (one row per video frame). `labels` is the
public ground truth as codes over the whole match (X outside the labelled window); only read it at
train_idx. To add a method, write a function that returns an object like the ones below and add it
to REGISTRY.
"""

import numpy as np

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


REGISTRY = {
    "nearest": NearestPlayer,
    "logistic": LogisticSmooth,
}
