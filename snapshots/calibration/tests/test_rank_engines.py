from dataclasses import replace
import numpy as np
import pytest
from ranktwosample.rank_two_sample import RankGTest, RankGTestConfig, precompute_orders, ranks_prefixsum, ranks_searchsorted
from ranktwosample.rank_two_sample_subspaces import RankGTestSubspaces, SubspaceOpts


def signature(out):
    return out.reject,out.p_perm,out.p_mid,out.T_obs,out.perms_used


@pytest.mark.parametrize('tie',['right','left','jitter'])
def test_prefix_equals_direct_comparisons_with_self_ties(tie):
    z=np.array([[0.,1.],[0.,0.],[1.,0.],[1.,1.],[-1.,1.]])
    ids=np.arange(len(z)); refs=np.array([0,1,3]); mask=np.zeros(len(z));mask[refs]=1
    jitter=np.random.default_rng(37).normal(scale=.1,size=(2,len(z))) if tie=='jitter' else None
    side='right' if tie=='jitter' else tie
    work=z if jitter is None else z+jitter.T
    orders=precompute_orders(work,tie=side)
    pick='last_right' if side=='right' else 'first_left'
    direct=ranks_searchsorted(ids,refs,[z[:,0],z[:,1]],side=side,jitter=jitter)
    fast=ranks_prefixsum(ids,mask,orders,side_pick=pick)
    reference=np.sum((work[refs][None,:,:] <= work[:,None,:]) if side=='right' else (work[refs][None,:,:] < work[:,None,:]),axis=1)
    assert np.array_equal(direct,reference)
    assert np.array_equal(fast,reference)


@pytest.mark.parametrize('tie',['right','left','jitter'])
@pytest.mark.parametrize('subspaces',[False,True])
def test_coupled_engines_and_disabled_stopping(tie,subspaces):
    rng=np.random.default_rng(173)
    x=np.round(rng.normal(size=(12,3)))
    y=np.round(rng.normal(size=(13,3)))
    cfg=RankGTestConfig(K_ref=4,B=31,tie=tie,decision='pvalue',stop_ci='none',min_b_check=1,antithetic=False,jitter_scale=.01)
    outputs=[]
    for engine in ('searchsorted','prefixsum','auto'):
        c=replace(cfg,engine=engine)
        test=RankGTestSubspaces(c,SubspaceOpts(L=3,k_dim=1)) if subspaces else RankGTest(c)
        a=test.test(x,y,seed=721)
        b=test.test(x,y,seed=721)
        assert signature(a)==signature(b)
        assert a.perms_used==cfg.B
        outputs.append(signature(a))
    assert outputs[0]==outputs[1]==outputs[2]


@pytest.mark.parametrize('mode',['shared_pool','fresh_iid'])
@pytest.mark.parametrize('subspaces',[False,True])
def test_randomized_tie_decision_uses_seed_and_full_budget(mode,subspaces):
    cfg=RankGTestConfig(K_ref=2,B=19,reference_mode=mode,tie='right',decision='randomized',stop_ci='none',min_b_check=1,alpha=.25,antithetic=False)
    test=RankGTestSubspaces(cfg,SubspaceOpts(L=1,k_dim=1)) if subspaces else RankGTest(cfg)
    x=y=np.zeros((5,1))
    for seed in range(5):
        assert signature(test.test(x,y,seed=seed))==signature(test.test(x,y,seed=seed))
        assert test.test(x,y,seed=seed).perms_used==cfg.B


def test_pca_selection_terminates_above_unique_capacity_and_for_wide_data():
    rng=np.random.default_rng(8)
    x,y=rng.normal(size=(3,7)),rng.normal(size=(3,7))
    test=RankGTestSubspaces(RankGTestConfig(K_ref=2,B=3,stop_ci='none'),SubspaceOpts(L=64,k_dim=1,pca=True))
    dims=test._sample_dims_list(x,y,rng)
    assert 1<=len(dims)<=7 and len(set(dims))==len(dims)
    assert test.test(x,y,seed=81).perms_used==3


def test_wilson_can_stop_inside_a_single_large_chunk():
    cfg=RankGTestConfig(K_ref=4,B=100,chunk=512,min_b_check=50,tie='right',decision='pvalue',stop_ci='wilson',antithetic=False)
    x=y=np.zeros((20,2))
    test=RankGTestSubspaces(cfg,SubspaceOpts(L=2,k_dim=1))
    stopped=test.test(x,y,seed=381)
    full=RankGTestSubspaces(replace(cfg,stop_ci='none'),SubspaceOpts(L=2,k_dim=1)).test(x,y,seed=381)
    assert stopped.perms_used==50 and full.perms_used==100


def test_unknown_stopping_rule_is_not_silently_hoeffding():
    with pytest.raises(ValueError):
        RankGTest(RankGTestConfig(stop_ci='typo'))
