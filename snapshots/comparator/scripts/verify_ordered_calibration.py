"""Exhaustively audit the actual ordered-role permutation loop (CPU only).

An enumeration RNG and a temporary statistic recorder expose all 720 candidate
scores. They are verification instrumentation, not benchmark configurations.
"""
from itertools import permutations
from pathlib import Path
import sys
import numpy as np
from scipy.stats import binom
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ranktwosample.rank_two_sample import RankGTestConfig
import ranktwosample.rank_two_sample_subspaces as module
from ranktwosample.rank_two_sample_subspaces import RankGTestSubspaces, SubspaceOpts


def main():
    z=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.],[2.,0.],[0.,2.]])
    maximum=0.
    original_stat=module.stat_from_counts
    recorded=[]
    def recorder(*args,**kwargs):
        value=original_stat(*args,**kwargs)
        recorded.append(value)
        return value
    try:
        module.stat_from_counts=recorder
        for tie in ('left','right','jitter'):
            for engine in ('searchsorted','prefixsum'):
                cfg=RankGTestConfig(K_ref=2,engine=engine,tie=tie,B=1,stop_ci='none',antithetic=False,decision='pvalue')
                test=RankGTestSubspaces(cfg,SubspaceOpts(dims_list=[(0,),(1,)],L=2,k_dim=1))
                observed,env=test._compute_T_obs_sum_shared_pool(z[:3],z[3:],np.random.default_rng(781))
                k,d,ss,orders,pick,jitter,side,cols,views,n,m,positions=env
                work=z if jitter is None else z+jitter.T
                def direct(ix,iy):
                    result=0.
                    for dims,pos in zip(views,positions):
                        refs=iy[pos]
                        def histogram(ids):
                            values=work[refs][:,dims]
                            queries=work[ids][:,dims]
                            compare=values[None,:,:] <= queries[:,None,:] if side=='right' else values[None,:,:] < queries[:,None,:]
                            return np.bincount(compare.sum(axis=1).ravel(),minlength=k+1)
                        hx,hy=histogram(ix),histogram(iy)
                        expected=n*(hy+cfg.alpha0)/(m+cfg.alpha0*(k+1))
                        positive=hx>0
                        result+=2*np.sum(hx[positive]*np.log(hx[positive]/expected[positive]))
                    return result
                assert abs(observed-direct(np.arange(3),np.arange(3,6)))<1e-12
                class ReplayRng:
                    def __init__(self,p): self.p=p; self.view=0
                    def normal(self,**kwargs): return jitter.copy()
                    def permutation(self,length):
                        if length==len(z): return self.p.copy()
                        assert length==m
                        selected=positions[self.view]; self.view+=1
                        return np.concatenate([selected,np.setdiff1d(np.arange(m),selected)])
                scores=[]
                for p in permutations(range(6)):
                    p=np.array(p); iy,ix=p[:m],p[m:]
                    test._runner._rng=lambda seed=None: ReplayRng(p)
                    recorded.clear()
                    out=test.test(z[:3],z[3:])
                    assert len(recorded)==2*len(views) and out.perms_used==1
                    candidate=sum(recorded[-len(views):])
                    truth=direct(ix,iy)
                    maximum=max(maximum,abs(candidate-truth))
                    assert abs(candidate-truth)<1e-12
                    assert out.p_perm==(1+int(candidate>=out.T_obs))/2
                    scores.append(candidate)
                scores=np.array(scores)
                exact=np.mean(scores[None,:]>=scores[:,None],axis=1)
                for alpha in (.01,.05,.1,.25,.5,1.):
                    assert np.mean(exact<=alpha)<=alpha+1e-12
    finally:
        module.stat_from_counts=original_stat
    b=100;alpha=.05
    continuous_plusone=np.floor(alpha*(b+1))/(b+1)
    continuous_midp=(np.floor(alpha*b)+1)/(b+1)
    duplicate_antithetic=(np.floor((alpha*(b+1)-1)/2)+1)/(b/2+1)
    discrete_midp=.08*binom.cdf(int(2*b*alpha),b,.08)
    assert continuous_midp>alpha and duplicate_antithetic>alpha and discrete_midp>alpha
    print({'ordered_cases':6*720,'max_map_error':maximum,
           'independent_plusone_continuous_size':continuous_plusone,
           'midp_continuous_size':continuous_midp,
           'duplicate_antithetic_size':duplicate_antithetic,
           'midp_two_point_size':discrete_midp})

if __name__=='__main__':main()
