"""Train-once C2ST with exact fixed-class-count holdout calibration.

A separately named variant: the historical Binomial implementation is unchanged.
Only training labels may influence fitting/tuning. The hypergeometric test is
conditional on held-out predictions and fixed class counts under exchangeability.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.stats import hypergeom
from sklearn.model_selection import train_test_split
from .c2st import C2STResult, _make_default_estimator


def prediction_hypergeometric_pvalue(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.ndim != 1 or truth.shape != prediction.shape or not len(truth):
        raise ValueError('Nonempty equal-sized label and prediction vectors required')
    if not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError('Binary labels/predictions required')
    n, n1, k = len(truth), int(truth.sum()), int(prediction.sum())
    overlap = int(np.sum((truth == 1) & (prediction == 1)))
    return float(hypergeom.sf(overlap - 1, n, k, n1))


def c2st_holdout_hypergeometric_test(X, Y, *, estimator_factory=None,
                                    split_prop=.7, alpha=.05, seed=2027):
    started = time.perf_counter()
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError('Compatible feature matrices required')
    z = np.vstack([X, Y])
    labels = np.concatenate([np.ones(len(X), dtype=int), np.zeros(len(Y), dtype=int)])
    training, testing = train_test_split(np.arange(len(z)), train_size=split_prop,
                                         stratify=labels, random_state=seed)
    if estimator_factory is None:
        rng = np.random.default_rng(seed)
        estimator_factory = lambda: _make_default_estimator(int(rng.integers(1, 2**31-1)))
    classifier = estimator_factory()
    classifier.fit(z[training], labels[training])
    prediction = classifier.predict(z[testing])
    p = prediction_hypergeometric_pvalue(labels[testing], prediction)
    accuracy = float(np.mean(labels[testing] == prediction))
    return C2STResult(stat_obs=accuracy, p_perm=p, p_mid=p, reject=p <= alpha,
                      alpha=alpha, metric='acc', cv_mode='holdout_hypergeometric',
                      folds=1, perms_used=0, train_seconds=time.perf_counter()-started,
                      note='Train once; exact hypergeometric holdout confusion-table tail.')


@dataclass
class C2STHypergeometric:
    split_prop: float = .7
    alpha: float = .05
    seed: int = 2027
    estimator_factory: object = None

    def run(self, X, Y):
        return c2st_holdout_hypergeometric_test(X, Y, split_prop=self.split_prop,
               alpha=self.alpha, seed=self.seed, estimator_factory=self.estimator_factory)
