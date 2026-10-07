from itertools import combinations
import numpy as np
from baselines.rff_mmd import RFFMMD, rff_features, feature_mmd_squared


def test_feature_mmd_equals_explicit_biased_kernel_mmd():
    rng = np.random.default_rng(271)
    fx, fy = rng.normal(size=(7, 4)), rng.normal(size=(11, 4))
    expected = np.mean(fx@fx.T)+np.mean(fy@fy.T)-2*np.mean(fx@fy.T)
    assert abs(feature_mmd_squared(fx, fy)-expected) < 1e-12


def test_fixed_feature_statistic_exact_allocation_superuniformity():
    z = np.array([[-1., 0.], [0., 1.], [.2, .5], [1., 2.], [2., -1.]])
    f = rff_features(z, np.array([[.4, .1], [-1., .2]]), np.array([.2, .8]))
    scores = []
    for target in combinations(range(5), 2):
        reference = [j for j in range(5) if j not in target]
        scores.append(feature_mmd_squared(f[list(target)], f[reference]))
    scores = np.array(scores)
    p = np.mean(scores[None, :] >= scores[:, None], axis=1)
    for alpha in (.05, .1, .25, .5, 1.):
        assert np.mean(p <= alpha) <= alpha+1e-12


def test_rff_full_budget_repeatability_imbalance_and_constant_cloud():
    rng = np.random.default_rng(72)
    x, y = rng.normal(size=(7, 2)), rng.normal(size=(11, 2))
    test = RFFMMD(features=8, B=19, seed=37)
    a, b = test.run(x, y), test.run(x, y)
    assert (a.stat_obs, a.p_perm, a.reject) == (b.stat_obs, b.p_perm, b.reject)
    assert a.perms_used == 19
    constant = test.run(np.zeros((7, 2)), np.zeros((11, 2)))
    assert constant.stat_obs == 0 and constant.p_perm == 1 and not constant.reject


def test_gaussian_spectral_identity_by_quadrature():
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    # Integrating the independent uniform phase first gives cos(omega*(x-y)).
    for distance in (0., .2, 1., 2.):
        for bandwidth in (.5, 1., 2.):
            value = weights@np.cos(np.sqrt(2)*nodes*distance/bandwidth)/np.sqrt(np.pi)
            assert abs(value-np.exp(-distance**2/(2*bandwidth**2))) < 1e-12
