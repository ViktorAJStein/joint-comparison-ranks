from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from scripts import run_power_bench as common
from scripts import rotation_bench as rotation


def test_common_rank_wrapper_forwards_seed(monkeypatch):
    received=[]
    class Dummy:
        def __init__(self, **kw): pass
        def run(self,x,y,seed=None):
            received.append(seed)
            return SimpleNamespace(reject=False,stat_obs=0.,p_perm=1.)
    monkeypatch.setattr(common,'RankChi2SubspacesTest',Dummy)
    common.run_rank_subspaces(np.zeros((5,2)),np.zeros((6,2)),.05,19,{'seed':381})
    assert received==[381]


def test_one_rep_preserves_data_and_derives_rank_stream():
    spec=common.BenchSpec('locshift',2,5,6,{})
    observed=[]
    def capture(x,y,alpha,b,cfg):
        observed.append((x.copy(),y.copy(),cfg['seed']))
        return False,0.,1.,0.,'OK'
    a=common.one_rep(601,spec,.05,19,{}, {'RankChi2_Subspaces':capture},experiment='calibration')
    b=common.one_rep(601,spec,.05,19,{}, {'RankChi2_Subspaces':capture},experiment='calibration')
    rng=np.random.default_rng(601)
    x,y=common.generate_null_pair(spec,rng)
    assert np.array_equal(observed[0][0],x) and np.array_equal(observed[0][1],y)
    assert observed[0][2]==observed[1][2] and a==b
    common.one_rep(601,spec,.05,19,{'RankChi2_Subspaces':{'seed':777}}, {'RankChi2_Subspaces':capture},experiment='calibration')
    assert observed[-1][2]==777


def test_rotation_rank_wrapper_forwards_seed(monkeypatch):
    received=[]
    class Dummy:
        def __init__(self,**kw): pass
        def run(self,x,y,seed=None):
            received.append(seed)
            return SimpleNamespace(reject=False,stat_obs=0.)
    monkeypatch.setattr(rotation,'RankChi2SubspacesTest',Dummy)
    args=SimpleNamespace(K_ref=4,min_b_check=50,rank_stat='gtest',L_subspaces=64,rank_subspace_dim=1)
    rotation.run_method('RankChi2_Subspaces',np.zeros((5,2)),np.zeros((6,2)),.05,19,912,args)
    assert received==[912]
