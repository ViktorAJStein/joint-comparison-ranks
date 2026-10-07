from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from ranktwosample.power_datasets import generate_pair, sample_axis_gmm_matched_pair


def test_generators_are_deterministic_for_fixed_seed():
    args = ("gmm-matched", 8, 100, 120)
    kwargs = {"params": {"delta": 0.8, "sparse_dims": 2}, "rng": 91}
    x1, y1 = generate_pair(*args, **kwargs)
    x2, y2 = generate_pair(*args, **kwargs)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)


def test_diffuse_matched_gmm_matches_first_two_moments():
    x, _ = generate_pair(
        "gmm-matched",
        8,
        30_000,
        30_000,
        params={"delta": 0.8, "sparse_dims": 2},
        rng=17,
    )
    assert np.max(np.abs(x.mean(axis=0))) < 0.025
    assert np.max(np.abs(np.cov(x, rowvar=False) - np.eye(8))) < 0.04


def test_axis_gmm_matches_moments_but_changes_active_marginal():
    x, y = sample_axis_gmm_matched_pair(
        8, 30_000, 30_000, delta=1.8, sparse_dims=4, rng=23
    )
    assert np.max(np.abs(x.mean(axis=0))) < 0.025
    assert np.max(np.abs(np.cov(x, rowvar=False) - np.eye(8))) < 0.04
    assert ks_2samp(x[:, 0], y[:, 0]).statistic > 0.06
    assert ks_2samp(x[:, 6], y[:, 6]).statistic < 0.025


def test_corrected_gumbel_generator_preserves_marginals():
    x, y = generate_pair(
        "dep-gumbel",
        4,
        30_000,
        30_000,
        params={"rho": 0.5, "nu": 6, "equicorr": True},
        rng=29,
    )
    marginal_ks = [ks_2samp(x[:, j], y[:, j]).statistic for j in range(4)]
    assert max(marginal_ks) < 0.02
    assert np.max(np.abs(x.mean(axis=0) - y.mean(axis=0))) < 0.01
