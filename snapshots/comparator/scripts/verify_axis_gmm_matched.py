#!/usr/bin/env python3
"""Sanity checks for the axis matched-moment GMM benchmark."""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from ranktwosample.power_datasets import sample_axis_gmm_matched_pair


def check_case(d: int, s: int, delta: float, n: int = 200_000, seed: int = 123) -> None:
    rng = np.random.default_rng(seed)
    X, Y = sample_axis_gmm_matched_pair(d, n, n, delta=delta, sparse_dims=s, rng=rng)
    mean_err = float(np.max(np.abs(X.mean(axis=0))))
    var_err = float(np.max(np.abs(X.var(axis=0) - 1.0)))
    cov_err = float(np.max(np.abs(np.cov(X, rowvar=False) - np.eye(d))))
    active_ks = [float(ks_2samp(X[:, j], Y[:, j]).statistic) for j in range(min(s, 4))]
    inactive_ks = float(ks_2samp(X[:, s], Y[:, s]).statistic) if s < d else float("nan")
    print(
        f"d={d:2d} s={s:2d} delta={delta:.1f} "
        f"max|mean|={mean_err:.4g} max|var-1|={var_err:.4g} "
        f"max|cov-I|={cov_err:.4g} active_KS={active_ks} inactive_KS={inactive_ks:.4g}"
    )


def main() -> None:
    for d, s, delta in [(20, 4, 1.8), (20, 2, 1.8), (50, 4, 1.8)]:
        check_case(d, s, delta)


if __name__ == "__main__":
    main()
