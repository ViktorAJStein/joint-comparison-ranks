"""Fixed-bandwidth random-Fourier-feature (biased) MMD, full-budget plus-one.

Approximates k(x,y)=exp(-||x-y||^2/(2*bandwidth^2)), not exact MMD.
Features are label-independent and frozen for the full statistic calibration.
No tuning, stopping, antithetic pairing, or universal finite-feature consistency.
"""
from dataclasses import dataclass
import time
import numpy as np


def rff_features(z, frequencies, phases):
    return np.sqrt(2/len(phases))*np.cos(np.asarray(z)@frequencies.T+phases)


def feature_mmd_squared(fx, fy):
    # A common translation leaves the mean discrepancy unchanged and avoids
    # cancellation for identical features with unequal group sizes.
    anchor = fx[0]
    delta = (fx-anchor).mean(axis=0)-(fy-anchor).mean(axis=0)
    return float(delta@delta)


@dataclass
class RFFMMDResult:
    stat_obs: float
    p_perm: float
    reject: bool
    perms_used: int
    total_seconds: float
    features: int
    bandwidth: float


class RFFMMD:
    def __init__(self, features=256, bandwidth=1., B=100, alpha=.05, seed=2035):
        if not isinstance(features, int) or features < 1 or not isinstance(B, int) or B < 1:
            raise ValueError('Positive integer features/B required')
        if not np.isfinite(bandwidth) or bandwidth <= 0 or not 0 < alpha < 1:
            raise ValueError('Positive finite bandwidth and alpha in (0,1) required')
        self.features, self.bandwidth, self.B, self.alpha, self.seed = features, bandwidth, B, alpha, seed

    def run(self, X, Y):
        start = time.perf_counter()
        x, y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not len(x) or not len(y):
            raise ValueError('Nonempty compatible data matrices required')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Finite input required')
        feature_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0x524646]))
        frequencies = feature_rng.normal(size=(self.features, x.shape[1]))/self.bandwidth
        phases = feature_rng.uniform(0, 2*np.pi, self.features)
        fx, fy = rff_features(x, frequencies, phases), rff_features(y, frequencies, phases)
        observed = feature_mmd_squared(fx, fy)
        pooled = np.vstack([fx, fy]); n = len(x)
        permutation_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0x5045524D]))
        ge = 0
        for _ in range(self.B):
            order = permutation_rng.permutation(len(pooled))
            score = feature_mmd_squared(pooled[order[:n]], pooled[order[n:]])
            ge += score >= observed
        p = (1+ge)/(self.B+1)
        return RFFMMDResult(observed, p, p <= self.alpha, self.B, time.perf_counter()-start,
                            self.features, self.bandwidth)
