from __future__ import annotations

import numpy as np

from baselines.c2st import c2st_holdout_binomial_test
from baselines.me_scf import me_permutation_test, scf_permutation_test
from baselines.rank_chi2_subspaces import RankChi2SubspacesTest


def test_new_baselines_smoke():
    rng = np.random.default_rng(31)
    x = rng.normal(0.3, 1.0, size=(80, 4))
    y = rng.normal(0.0, 1.0, size=(80, 4))

    c2st = c2st_holdout_binomial_test(x, y, seed=7)
    me = me_permutation_test(x, y, J=4, B=20, early_stop="none", seed=11)
    scf = scf_permutation_test(x, y, J=4, B=20, early_stop="none", seed=13)

    for result in (c2st, me, scf):
        assert 0.0 <= result.p_perm <= 1.0
        assert np.isfinite(result.stat_obs)


def test_rank_subspace_is_reproducible_for_fixed_seed():
    rng = np.random.default_rng(37)
    x = rng.normal(size=(60, 5))
    y = rng.normal(size=(60, 5))
    test = RankChi2SubspacesTest(
        K_ref=4,
        L=4,
        k_dim=2,
        B=20,
        min_b_check=10,
        stat="chi2",
        tie="right",
        gpu="off",
    )
    first = test.run(x, y, seed=41)
    second = test.run(x, y, seed=41)
    assert first.stat_obs == second.stat_obs
    assert first.p_perm == second.p_perm
    assert first.p_mid == second.p_mid
