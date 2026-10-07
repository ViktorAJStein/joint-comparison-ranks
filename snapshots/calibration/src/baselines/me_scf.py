# src/baselines/me_scf.py
# -------------------------------------------------------------------
# Lightweight ME/SCF-style finite-feature two-sample baselines.
#
# This module implements a practical finite-feature kernel mean embedding
# test in the spirit of the Mean Embedding (ME) test of Chwialkowski et al.
# (2015): choose a finite set of test locations, compare empirical Gaussian
# kernel mean features at those locations, and calibrate the covariance-
# normalized quadratic statistic by label permutations.
#
# The implementation is intentionally conservative for rebuttal benchmarking:
# locations are selected at random from the pooled sample rather than optimized.
# -------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import math
import time

import numpy as np

Array = np.ndarray

__all__ = [
    "MESCFResult",
    "me_permutation_test",
    "scf_permutation_test",
    "MEFiniteFeatureTest",
    "SCFFiniteFeatureTest",
]


def _ndtri(p: float) -> float:
    if p <= 0.0:
        return -1e300
    if p >= 1.0:
        return 1e300
    a = [-39.6968302866538, 220.946098424521, -275.928510446969,
         138.357751867269, -30.6647980661472, 2.50662827745924]
    b = [-54.476098798225, 161.585836858041, -155.698979886,
         66.8013118877197, -13.2806815528857]
    c = [-0.007784894002430293, -0.322396458041136, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.93816398269878]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _wilson_ci(k: int, n: int, delta: float) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    z = _ndtri(1 - delta / 2.0)
    phat = k / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = z / denom * math.sqrt(max(0.0, phat*(1-phat)/n + z*z/(4*n*n)))
    return max(0.0, center - half), min(1.0, center + half)


def _pairwise_sq_dists(X: Array, Y: Array) -> Array:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0)


def _median_bandwidth(Z: Array, subsample: int = 1000, seed: int = 0) -> float:
    n = Z.shape[0]
    m = min(n, subsample)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    D2 = _pairwise_sq_dists(Z[idx], Z[idx])
    tri = D2[np.triu_indices(m, k=1)]
    med = float(np.sqrt(np.median(tri))) if tri.size else 1.0
    return med if med > 0 else 1.0


def _choose_locations(Z: Array, J: int, rng: np.random.Generator) -> Array:
    n = Z.shape[0]
    replace = J > n
    idx = rng.choice(n, size=J, replace=replace)
    return np.asarray(Z[idx], dtype=float).copy()


def _rbf_features(Z: Array, locations: Array, sigma: float) -> Array:
    D2 = _pairwise_sq_dists(Z, locations)
    return np.exp(-D2 / (2.0 * sigma * sigma))


def _choose_frequencies(d: int, J: int, sigma: float, rng: np.random.Generator) -> Array:
    """Random Fourier frequencies for a Gaussian kernel spectral measure.

    For k(x,y)=exp(-||x-y||^2/(2 sigma^2)), the spectral law is
    N(0, sigma^{-2} I_d) under the exp(i w^T x) convention.
    """
    return rng.normal(loc=0.0, scale=1.0 / max(float(sigma), 1e-12), size=(int(J), int(d)))


def _scf_features(Z: Array, frequencies: Array) -> Array:
    """Real-valued smoothed-characteristic-function features.

    We compare empirical characteristic features at fixed random frequencies by
    concatenating cos(w_j^T z) and sin(w_j^T z), yielding 2J real features.
    """
    phase = np.asarray(Z, dtype=float) @ np.asarray(frequencies, dtype=float).T
    return np.concatenate([np.cos(phase), np.sin(phase)], axis=1)


def _stat_from_features(F: Array, labels: Array, n_x: int, ridge: float) -> float:
    FX = F[labels == 1]
    FY = F[labels == 0]
    if FX.shape[0] == 0 or FY.shape[0] == 0:
        return 0.0
    delta = FX.mean(axis=0) - FY.mean(axis=0)
    pooled = np.vstack([FX - FX.mean(axis=0), FY - FY.mean(axis=0)])
    if pooled.shape[0] <= 1:
        cov = np.eye(F.shape[1])
    else:
        cov = np.cov(pooled, rowvar=False, bias=False)
        cov = np.atleast_2d(cov)
    cov = cov + float(ridge) * np.eye(cov.shape[0])
    scale = FX.shape[0] * FY.shape[0] / max(1, FX.shape[0] + FY.shape[0])
    try:
        val = scale * float(delta @ np.linalg.solve(cov, delta))
    except np.linalg.LinAlgError:
        val = scale * float(delta @ np.linalg.pinv(cov) @ delta)
    return val


@dataclass
class MESCFResult:
    stat_obs: float
    p_perm: float
    reject: bool
    alpha: float
    perms_used: int
    train_seconds: float
    J: int
    sigma: float
    note: str


def me_permutation_test(
    X: Array,
    Y: Array,
    *,
    J: int = 10,
    sigma: Optional[float] = None,
    bandwidth_scale: float = 1.0,
    B: int = 1000,
    alpha: float = 0.05,
    ridge: float = 1e-6,
    early_stop: Literal["none", "wilson"] = "wilson",
    delta_ci: float = 1e-2,
    min_b_check: int = 50,
    seed: int = 2031,
) -> MESCFResult:
    t0 = time.perf_counter()
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n_x, n_y = X.shape[0], Y.shape[0]
    Z = np.vstack([X, Y])
    labels = np.concatenate([np.ones(n_x, dtype=int), np.zeros(n_y, dtype=int)])
    rng = np.random.default_rng(seed)

    if sigma is None:
        sigma = _median_bandwidth(Z, seed=seed) * float(bandwidth_scale)
    sigma = float(max(sigma, 1e-12))
    locations = _choose_locations(Z, int(J), rng)
    F = _rbf_features(Z, locations, sigma)

    stat_obs = _stat_from_features(F, labels, n_x, ridge)
    ge = 0
    perms_used = 0
    for b in range(1, int(B) + 1):
        yb = np.random.default_rng(seed + b).permutation(labels)
        stat_b = _stat_from_features(F, yb, n_x, ridge)
        ge += int(stat_b >= stat_obs)
        perms_used = b
        if early_stop != "none" and b >= min_b_check:
            lo, hi = _wilson_ci(ge, b, delta_ci)
            if hi <= alpha or lo > alpha:
                break

    p_perm = (1 + ge) / (perms_used + 1)
    reject = p_perm <= alpha
    return MESCFResult(
        stat_obs=float(stat_obs),
        p_perm=float(p_perm),
        reject=bool(reject),
        alpha=float(alpha),
        perms_used=int(perms_used),
        train_seconds=float(time.perf_counter() - t0),
        J=int(J),
        sigma=float(sigma),
        note="Random-location ME-style finite-feature RBF test; permutation calibrated.",
    )


def scf_permutation_test(
    X: Array,
    Y: Array,
    *,
    J: int = 10,
    sigma: Optional[float] = None,
    bandwidth_scale: float = 1.0,
    B: int = 1000,
    alpha: float = 0.05,
    ridge: float = 1e-6,
    early_stop: Literal["none", "wilson"] = "wilson",
    delta_ci: float = 1e-2,
    min_b_check: int = 50,
    seed: int = 2033,
) -> MESCFResult:
    """Random-frequency SCF-style finite-feature two-sample test.

    This is a lightweight, non-optimized SCF comparator in the spirit of
    Chwialkowski et al. (2015): frequencies are sampled once from the Gaussian
    kernel spectral distribution, features are empirical characteristic
    function evaluations (cos/sin), and calibration is by label permutations.
    """
    t0 = time.perf_counter()
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n_x, n_y = X.shape[0], Y.shape[0]
    Z = np.vstack([X, Y])
    labels = np.concatenate([np.ones(n_x, dtype=int), np.zeros(n_y, dtype=int)])
    rng = np.random.default_rng(seed)

    if sigma is None:
        sigma = _median_bandwidth(Z, seed=seed) * float(bandwidth_scale)
    sigma = float(max(sigma, 1e-12))
    frequencies = _choose_frequencies(Z.shape[1], int(J), sigma, rng)
    F = _scf_features(Z, frequencies)

    stat_obs = _stat_from_features(F, labels, n_x, ridge)
    ge = 0
    perms_used = 0
    for b in range(1, int(B) + 1):
        yb = np.random.default_rng(seed + b).permutation(labels)
        stat_b = _stat_from_features(F, yb, n_x, ridge)
        ge += int(stat_b >= stat_obs)
        perms_used = b
        if early_stop != "none" and b >= min_b_check:
            lo, hi = _wilson_ci(ge, b, delta_ci)
            if hi <= alpha or lo > alpha:
                break

    p_perm = (1 + ge) / (perms_used + 1)
    reject = p_perm <= alpha
    return MESCFResult(
        stat_obs=float(stat_obs),
        p_perm=float(p_perm),
        reject=bool(reject),
        alpha=float(alpha),
        perms_used=int(perms_used),
        train_seconds=float(time.perf_counter() - t0),
        J=int(J),
        sigma=float(sigma),
        note="Random-frequency SCF-style finite-feature test; permutation calibrated.",
    )


class MEFiniteFeatureTest:
    def __init__(
        self,
        *,
        J: int = 10,
        sigma: Optional[float] = None,
        bandwidth_scale: float = 1.0,
        B: int = 1000,
        alpha: float = 0.05,
        ridge: float = 1e-6,
        early_stop: Literal["none", "wilson"] = "wilson",
        delta_ci: float = 1e-2,
        min_b_check: int = 50,
        seed: int = 2031,
    ):
        self.J = J
        self.sigma = sigma
        self.bandwidth_scale = bandwidth_scale
        self.B = B
        self.alpha = alpha
        self.ridge = ridge
        self.early_stop = early_stop
        self.delta_ci = delta_ci
        self.min_b_check = min_b_check
        self.seed = seed

    def run(self, X: Array, Y: Array) -> MESCFResult:
        return me_permutation_test(
            X, Y,
            J=self.J,
            sigma=self.sigma,
            bandwidth_scale=self.bandwidth_scale,
            B=self.B,
            alpha=self.alpha,
            ridge=self.ridge,
            early_stop=self.early_stop,
            delta_ci=self.delta_ci,
            min_b_check=self.min_b_check,
            seed=self.seed,
        )


class SCFFiniteFeatureTest:
    def __init__(
        self,
        *,
        J: int = 10,
        sigma: Optional[float] = None,
        bandwidth_scale: float = 1.0,
        B: int = 1000,
        alpha: float = 0.05,
        ridge: float = 1e-6,
        early_stop: Literal["none", "wilson"] = "wilson",
        delta_ci: float = 1e-2,
        min_b_check: int = 50,
        seed: int = 2033,
    ):
        self.J = J
        self.sigma = sigma
        self.bandwidth_scale = bandwidth_scale
        self.B = B
        self.alpha = alpha
        self.ridge = ridge
        self.early_stop = early_stop
        self.delta_ci = delta_ci
        self.min_b_check = min_b_check
        self.seed = seed

    def run(self, X: Array, Y: Array) -> MESCFResult:
        return scf_permutation_test(
            X, Y,
            J=self.J,
            sigma=self.sigma,
            bandwidth_scale=self.bandwidth_scale,
            B=self.B,
            alpha=self.alpha,
            ridge=self.ridge,
            early_stop=self.early_stop,
            delta_ci=self.delta_ci,
            min_b_check=self.min_b_check,
            seed=self.seed,
        )
