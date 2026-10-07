from __future__ import annotations

import numpy as np

from ranktwosample.rank_two_sample import stat_from_counts


def _manual_statistics(hx: np.ndarray, hy: np.ndarray, alpha0: float):
    n = int(hx.sum())
    ky = int(hy.sum())
    qhat = (hy + alpha0) / (ky + alpha0 * hx.size)
    expected = n * qhat
    pearson = np.sum((hx - expected) ** 2 / expected)
    positive = hx > 0
    gtest = 2.0 * np.sum(hx[positive] * np.log(hx[positive] / expected[positive]))
    return float(pearson), float(gtest)


def test_legacy_and_alias_statistics_match_definitions():
    hx = np.array([8, 17, 4, 21], dtype=float)
    hy = np.array([13, 11, 16, 10], dtype=float)
    alpha0 = 0.1
    expected_pearson, expected_gtest = _manual_statistics(hx, hy, alpha0)
    kwargs = dict(N_local=50, Ky_local=50, alpha0=alpha0)

    pearson = stat_from_counts(hx, hy, kind="chi2", **kwargs)
    gtest = stat_from_counts(hx, hy, kind="gtest", **kwargs)

    assert np.isclose(pearson, expected_pearson)
    assert np.isclose(gtest, expected_gtest)
    assert stat_from_counts(hx, hy, kind="pearson", **kwargs) == pearson
    assert stat_from_counts(hx, hy, kind="xlogx", **kwargs) == gtest
    assert np.isclose(stat_from_counts(hx, hy, kind="cr:1", **kwargs), pearson)
    assert np.isclose(stat_from_counts(hx, hy, kind="cr:0", **kwargs), gtest)


def test_statistics_are_finite_with_empty_observed_cells():
    hx = np.array([0, 5, 15, 0], dtype=float)
    hy = np.array([2, 8, 5, 5], dtype=float)
    kwargs = dict(N_local=20, Ky_local=20, alpha0=0.1)
    assert np.isfinite(stat_from_counts(hx, hy, kind="chi2", **kwargs))
    assert np.isfinite(stat_from_counts(hx, hy, kind="gtest", **kwargs))
