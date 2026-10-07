"""Finite-population identities behind the scoped shared-pool proof.

These are arithmetic checks, not Monte Carlo rank size/power experiments.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.stats import hypergeom, norm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ranktwosample.rank_two_sample import RankGTest, RankGTestConfig


def main():
    k,gamma,m=4,.1,5
    worst=0.;error=0.;cases=0
    for reference_size in (32,128,512,2048):
        for ratio in (.25,1.,4.):
            target_size=int(reference_size*ratio);v=target_size+reference_size-k
            c=target_size/(reference_size+gamma*m);a=(reference_size-k)/v
            rlo,rhi=.25,4.
            a0=1/(2*(1+rhi));c0=rlo/(1+gamma*m/(2*k))
            A=rhi*(2*k+gamma*(m+1));C=2+rhi+rhi*(k+gamma)
            b=(1-np.log(2))*a0/2
            bound=max(rhi*(k+gamma),2*((1+rhi)**2/4+A*A)/(c0*a0)+C*C*4*np.exp(b-2)/(c0*gamma*b*b))
            for u in np.unique(np.r_[np.arange(min(v,20)+1),np.linspace(0,v,60,dtype=int)]):
                y=np.arange(max(0,u-target_size),min(u,reference_size-k)+1)
                probability=hypergeom.pmf(y,v,u,reference_size-k)
                for selected in range(k+1):
                    e=u-(1+c)*y-c*(selected+gamma)
                    expected_error=c*(u*(k+gamma*m)/v-(selected+gamma))
                    error=max(error,abs(np.sum(e*probability)-expected_error))
                    cell=np.sum(probability*e*e/(c*(y+selected+gamma)))
                    assert cell<=bound
                    worst=max(worst,cell);cases+=1
                # Direct mgf bound for sampling without replacement.
                assert np.sum(probability*2.**(-y))<=(1-a/2)**u+1e-12
    assert error<1e-8
    # Shared cutoff Gaussian scale signal despite fresh k=1 averaged blindness.
    cutoffs=np.random.default_rng(715).normal(size=20000)
    difference=norm.cdf(cutoffs/1.5)-norm.cdf(cutoffs)
    assert np.all(difference!=0)
    # Conditional signal at fixed selected views; no claim from tiny probabilities.
    print({'hypergeometric_cell_cases':cases,'maximum_identity_error':error,
           'maximum_checked_cell_expectation':worst,'gaussian_cutoff_witnesses':len(cutoffs)})
    # For the PROTECTED B=100/alpha=.05/min_b=50 configuration, early
    # pvalue-Wilson stopping cannot change the fixed-budget binary decision:
    # no early-reject bound can fire; every early nonreject has ge>4.
    test=RankGTest(RankGTestConfig())
    for budget in range(50,101):
        for ge in range(budget+1):
            lo,hi=test._wilson_ci(ge,budget,.01)
            assert hi>.05
            if lo>.05: assert ge>4
    print('PASS: scoped Wilson binary-decision identity; NOT a general stopped-p-value guarantee')

if __name__=='__main__':main()
