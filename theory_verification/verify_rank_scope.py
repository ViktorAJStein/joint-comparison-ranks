#!/usr/bin/env python3
"""Falsification/identity checks for the purple theory revision.

Run in the companion uv environment:
  uv run --project joint-comparison-ranks python \
    overleaf_submission/theory_verification/verify_rank_scope.py
Checks do not establish power of the empirical implementation.
"""
from itertools import combinations, product
from math import comb, pi, sqrt
from pathlib import Path
import argparse
import sys

import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.integrate import quad
from scipy.special import ndtr


def ranks(query, batch):
    return (batch[None, :, :] <= query[:, None, :]).sum(axis=1)


def hist(query, batch, k):
    r = ranks(query, batch)
    index = r @ ((k+1)**np.arange(query.shape[1]-1, -1, -1))
    return np.bincount(index, minlength=(k+1)**query.shape[1]).astype(float)


def pearson(p, q):
    p, q = np.asarray(p), np.asarray(q)
    if np.any((q == 0) & (p > 0)):
        return np.inf
    keep = q > 0
    return float(np.sum((p[keep]-q[keep])**2/q[keep]))


def score(x, y, batch=None):
    if batch is None:
        batch = y[:2]
    k = len(batch)
    hx, hy = hist(x, batch, k), hist(y, batch, k)
    q = (hy+0.1)/(len(y)+0.1*len(hy))
    return len(x)*pearson(hx/len(x), q)


def pca(z):
    centered = z-z.mean(0)
    c = centered.T@centered/(len(z)-1)
    eigenvalues, v = np.linalg.eigh(c)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, v = eigenvalues[order], v[:, order]
    return centered@v, eigenvalues, v


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])


def check_fixed_k():
    max_error = 0.0
    for k in range(1, 9):
        polynomial = Legendre.basis(k+1)
        density = lambda u: 1+0.5*polynomial(2*u-1)
        probabilities = []
        for j in range(k+1):
            p, err = quad(lambda u: comb(k,j)*u**j*(1-u)**(k-j)*density(u), 0, 1, epsabs=1e-12)
            probabilities.append(p)
            max_error = max(max_error, abs(p-1/(k+1)))
        assert np.max(np.abs(np.array(probabilities)-1/(k+1))) < 1e-11
        assert pearson(np.array(probabilities), np.full(k+1, 1/(k+1))) < 1e-20
        assert abs(density(1)-1.5) < 1e-12
        # A dependence-only copula perturbation with unchanged uniform marginals.
        moments = np.array([quad(lambda u: comb(k,j)*u**j*(1-u)**(k-j)*polynomial(2*u-1),0,1,epsabs=1e-12)[0] for j in range(k+1)])
        copula_rank_law = np.full((k+1,k+1),1/(k+1)**2)+0.5*np.outer(moments,moments)
        assert np.max(np.abs(copula_rank_law-1/(k+1)**2)) < 1e-11
    print(f"fixed_K_counterexamples=8_marginal_and8_copula max_cell_error={max_error:.3g}")


def check_symmetries(rng):
    max_cell_error = 0.0
    for _ in range(100):
        batch = rng.uniform(-2, 2, (4, 3))
        query = rng.uniform(-2, 2, (9, 3))
        r = ranks(query, batch)
        assert np.array_equal(r, ranks(np.exp(query), np.exp(batch)))
        assert np.array_equal(4-r, ranks(-query, -batch))
        assert np.array_equal(r, ranks(query, batch[rng.permutation(4)]))
        perm = rng.permutation(3)
        assert np.array_equal(r[:, perm], ranks(query[:, perm], batch[:, perm]))
        p, q = rng.dirichlet(np.ones(10), size=2)
        cells = rng.permutation(10)
        max_cell_error = max(max_cell_error, abs(pearson(p,q)-pearson(p[cells],q[cells])))
    assert max_cell_error < 1e-10
    # The exact reversal correction with ties/self-comparisons.
    batch = np.array([[0.], [0.], [1.]])
    query = batch.copy()
    tied = (batch[None,:,:] == query[:,None,:]).sum(1)
    r, reverse = ranks(query,batch), ranks(-query,-batch)
    assert np.array_equal(reverse, 3-r+tied)
    assert not np.array_equal(reverse, 3-r)
    mid = r-0.5*tied
    assert np.array_equal(reverse-0.5*tied, 3-mid)
    # Non-strict maps and fixed additive jitter are not exact symmetries.
    assert not np.array_equal(ranks(np.array([[0.1]]),np.array([[0.9]])),
                              ranks(np.floor([[0.1]]),np.floor([[0.9]])))
    x, b = np.array([[0.1]]), np.array([[0.]])
    ex, eb = np.array([[-0.2]]), np.array([[0.]])
    assert not np.array_equal(ranks(x+ex,b+eb), ranks(10*x+ex,10*b+eb))
    print(f"randomized_symmetry_cases=100 max_cell_relabel_error={max_cell_error:.3g}; tie/jitter counterexamples verified")


def check_rotations():
    q = rotation(pi/3)
    assert np.array_equal(ranks(np.array([[1.,0.]]), np.array([[0.,1.]])), [[1,0]])
    assert np.array_equal(ranks(np.array([[1.,0.]])@q, np.array([[0.,1.]])@q), [[1,1]])
    reference_cov = np.eye(2)
    target_cov = np.diag([4.,1.])
    divergences = []
    for angle in (0.,pi/4):
        q = rotation(angle)
        difference_cov = q.T@target_cov@q+reference_cov
        rho = difference_cov[0,1]/sqrt(difference_cov[0,0]*difference_cov[1,1])
        density = lambda z: np.exp(-z*z/2)/sqrt(2*pi)
        quadrant, _ = quad(lambda z: density(z)*ndtr(rho*z/sqrt(1-rho*rho)),0,np.inf,epsabs=1e-12)
        p = np.array([quadrant,0.5-quadrant,0.5-quadrant,quadrant])
        divergences.append(pearson(p, np.full(4,0.25)))
    assert divergences[0] < 1e-20 and divergences[1] > 0.01
    print(f"population_rotation_counterexample: D1_original={divergences[0]:.3g} D1_rotated={divergences[1]:.6f}")
    # Distinguish batch-averaged rank laws from fixed-cutoff partition laws.
    density=lambda z: np.exp(-z*z/2)/sqrt(2*pi)
    averaged,_=quad(lambda z: density(z)*ndtr(2*z),-np.inf,np.inf,epsabs=1e-12)
    assert abs(averaged-0.5)<1e-12
    q=np.array([ndtr(1.),1-ndtr(1.)])
    p=np.array([ndtr(0.5),1-ndtr(0.5)])
    assert pearson(p,q)>0
    print(f'fresh_vs_shared: averaged_K1_signal=0; fixed_cutoff_signal={pearson(p,q):.6f} (conditional population law, not implementation power)')


def check_dependence_discrepancies():
    # Gaussian copula rank signal by product Gauss-Hermite quadrature.
    from numpy.polynomial.hermite import hermgauss
    nodes, weights = hermgauss(80)
    z, weights = nodes*np.sqrt(2), weights/np.sqrt(pi)
    for rho, rounded_signal in ((0.2,0.0092),(0.4,0.0371)):
        u = ndtr(z[:,None]+np.zeros((80,80)))
        v = ndtr(rho*z[:,None]+np.sqrt(1-rho*rho)*z[None,:])
        basis = lambda u: ((1-u)**2,2*u*(1-u),u*u)
        probabilities = np.array([[np.sum(weights[:,None]*weights[None,:]*a*b) for b in basis(v)] for a in basis(u)])
        signal = 9*np.sum((probabilities-1/9)**2)
        assert abs(signal-rounded_signal) < 0.00005
        # Gaussian MMD on Gaussian laws with identical standard-normal marginals.
        p, q = np.array([[1.,rho],[rho,1.]]), np.eye(2)
        for sigma in (0.5,1.,2.):
            term = lambda c: 1/np.sqrt(np.linalg.det(np.eye(2)+c/(sigma*sigma)))
            mmd2 = term(2*p)+term(2*q)-2*term(p+q)
            assert mmd2 > 0
        print(f'copula_rho={rho}: K2_rank_signal={signal:.9f}; Gaussian MMD positive at 3 bandwidths')


def check_pca(rng):
    max_error = 0.0
    for _ in range(30):
        z = rng.normal(size=(80,4))*np.array([3.,2.,1.,0.5])
        scores, lam, v = pca(z)
        q, _ = np.linalg.qr(rng.normal(size=(4,4)))
        other, lam2, v2 = pca(z@q)
        sign = np.sign(np.sum(v2*(q.T@v),axis=0))
        error = np.max(np.abs(other*sign-scores))
        max_error = max(max_error,error)
        assert np.allclose(lam,lam2,rtol=1e-12,atol=1e-12)
        assert error < 1e-10
        # Independent query and reference rows avoid self-comparisons for reversal.
        assert np.array_equal(ranks(other[:20]*sign,other[20:24]*sign),ranks(scores[:20],scores[20:24]))
        white = scores/np.sqrt(lam)
        assert np.array_equal(ranks(scores[:20],scores[20:24]),ranks(white[:20],white[20:24]))
        assert score(scores[:40],scores[40:]) == score(white[:40],white[40:])
    # A spherical covariance does not determine axes: C=cI for this cloud.
    z = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    c = z.T@z/3
    q = rotation(pi/6)
    assert np.allclose(q.T@c@q,c)
    # The equally legitimate identity basis before/after rotation leaves rotated
    # scores that are not related by any signed coordinate permutation.
    transformed = z@q
    signed_permutations = [z[:,p]*signs for p in ((0,1),(1,0)) for signs in product((-1,1),repeat=2)]
    assert all(not np.allclose(transformed,a) for a in signed_permutations)
    # Verify the explicit eigengap bound over perturbations.
    v = np.array([1.,0.,0.])
    gap = 0.4
    c = np.eye(3)+gap*np.outer(v,v)
    for _ in range(20):
        e = rng.normal(size=(3,3)); e=(e+e.T)/2
        e *= 0.05/np.linalg.norm(e,2)
        _, eig = np.linalg.eigh(c+e)
        top = eig[:,-1]
        sine = np.linalg.norm(top-np.dot(v,top)*v)
        assert sine <= np.linalg.norm(e,2)/(gap-np.linalg.norm(e,2))+1e-12
    print(f"PCA_simple_spectrum_cases=30 max_sign_aligned_error={max_error:.3g}; whitening ranks identical; spherical-axis failure and 20 eigengap checks pass")


def check_permutations(rng):
    allocations = list(combinations(range(6),3))
    z = rng.normal(size=(6,2))
    scores, eigenvalues, _ = pca(z)
    transforms = [z, z-z.mean(0), scores, scores/np.sqrt(eigenvalues),z[:,:1]]
    scores_by_map = []
    for transformed in transforms:
        ts = []
        for subset in allocations:
            keep = np.zeros(6,dtype=bool);keep[list(subset)]=True
            ts.append(score(transformed[keep],transformed[~keep]))
        scores_by_map.append(np.array(ts))
    scores_by_map.append((scores_by_map[0]+scores_by_map[2])/2)
    scores_by_map.append(np.maximum(scores_by_map[0],scores_by_map[2]))
    for ts in scores_by_map:
        p = np.array([np.mean(ts >= t) for t in ts])
        for alpha in (0.05,0.1,0.2,0.3,0.5,0.9):
            assert np.mean(p <= alpha) <= alpha+1e-12
    # Exhaustively integrate the plus-one Monte Carlo rank for B=3 and tied scores.
    all_scores = np.array([0.,0.,1.,3.])
    pvalues = []
    for indices in product(range(4),repeat=4):
        sampled = all_scores[list(indices)]
        pvalues.append((1+np.sum(sampled[1:]>=sampled[0]))/4)
    pvalues = np.array(pvalues)
    for alpha in (0.1,0.25,0.5,0.75,0.9):
        assert np.mean(pvalues <= alpha) <= alpha+1e-12
    print("exact_allocation_checks=7_statistics_x20_allocations; plus_one_MC=256_cases; superuniformity verified")


def check_subspace_interaction():
    max_error=0.0
    epsilon=0.5
    for k in (1,2,4):
        h=np.array([quad(lambda u: comb(k,j)*u**j*(1-u)**(k-j)*(2*u-1),0,1,epsabs=1e-12)[0] for j in range(k+1)])
        formula=(2*np.arange(k+1)-k)/((k+1)*(k+2))
        assert np.max(np.abs(h-formula)) < 1e-12
        for r in (2,3,4):
            grid=np.array(list(product(range(k+1),repeat=r)))
            q=np.full(len(grid),1/(k+1)**r)
            p=q+epsilon*np.prod(h[grid],axis=1)
            assert np.all(p>0) and abs(p.sum()-1)<1e-12
            computed=pearson(p,q)
            expected=epsilon**2*(k/(3*(k+2)))**r
            max_error=max(max_error,abs(computed-expected))
            assert abs(computed-expected) < 1e-12
    # Missing an active factor integrates the interaction perturbation to zero.
    assert abs(quad(lambda u: 2*u-1,0,1)[0]) < 1e-12
    active={0,1,2};d=6
    for s in (2,3,4):
        candidates=list(combinations(range(d),s))
        count=sum(active.issubset(j) for j in map(set,candidates))
        expected=comb(d-len(active),s-len(active)) if s>=len(active) else 0
        assert count==expected
        for a in (1,2):
            invisible=sum(all(not active.issubset(set(j)) for j in chosen) for chosen in combinations(candidates,a))
            assert invisible==comb(len(candidates)-count,a)
    print(f'subspace_interaction=9_rank_laws max_divergence_error={max_error:.3g}; zero missing-coordinate signal and combinatorial coverage verified')


def check_group_average(rng):
    x, y = rng.normal(size=(7,2)), rng.normal(size=(8,2))
    batch = rng.normal(size=(3,2))
    group = [rotation(j*pi/2) for j in range(4)]
    avg = lambda r: np.mean([score(x@r@q,y@r@q,batch@r@q) for q in group])
    a, b = avg(np.eye(2)), avg(group[1])
    assert abs(a-b)<1e-10
    print(f"finite_closed_group_average_error={abs(a-b):.3g} (algebraic sanity check; not a Haar-integration or power claim)")


def check_zero_cells():
    assert np.isinf(pearson(np.array([0.5,0.5]),np.array([1.,0.])))
    assert pearson(np.array([1.,0.]),np.array([1.,0.])) == 0
    counts = np.array([3.,0.]); gamma=0.1
    q = (counts+gamma)/(counts.sum()+gamma*len(counts))
    assert np.isfinite(pearson(np.array([0.5,0.5]),q))
    print("zero_cells: empty/structural/target-only/smoothed cases pass")


def check_companion_code(code_root):
    sys.path.insert(0,str(code_root/'src'))
    from baselines.rank_chi2_subspaces import RankChi2SubspacesTest
    from scripts.rotation_bench import pooled_pca_transform
    rng = np.random.default_rng(3810)
    x,y = rng.normal(size=(35,3)), rng.normal(size=(35,3))
    px,py=pooled_pca_transform(x,y,whiten=False)
    wx,wy=pooled_pca_transform(x,y,whiten=True)
    test=RankChi2SubspacesTest(K_ref=4,B=19,L=3,k_dim=1,tie='right',stat='gtest',decision='pvalue',stop_ci='none')
    a=test.run(px,py,seed=7151)
    b=test.run(wx,wy,seed=7151)
    assert a.stat_obs==b.stat_obs and a.p_perm==b.p_perm
    print(f"companion_seed_coupled_whitening: identical T={a.stat_obs:.6f} and p={a.p_perm:.6f} (identity check only)")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--code-root',type=Path,default=Path(__file__).resolve().parents[2]/'joint-comparison-ranks')
    args=parser.parse_args()
    rng=np.random.default_rng(20261001)
    check_fixed_k();check_symmetries(rng);check_rotations();check_dependence_discrepancies();check_pca(rng)
    check_permutations(rng);check_subspace_interaction();check_group_average(rng);check_zero_cells()
    if args.code_root.is_dir():
        check_companion_code(args.code_root)
    else:
        print('companion code unavailable: implementation parity not checked')
    print('outcome=PASS (proof sanity checks/counterexamples, not empirical power guarantees)')


if __name__=='__main__':
    main()
