"""No performance assertions: pairing, information boundaries and archive guards."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
import scipy
import sklearn
from threadpoolctl import threadpool_info
from scripts import fair_comparator_bench as bench

spec=importlib.util.spec_from_file_location('fair_analysis',bench.ROOT/'scripts/analyze_fair_comparators.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def tiny():
    p=json.loads(bench.PROTOCOL.read_text())
    p.update(training_per_class=6,evaluation_per_class=10,evaluation_permutations=19,
             training_permutations_per_candidate=3,dimensions=[3])
    p['rank'].update(K_ref=2,L=3);p['scf']['J']=2;p['rff']['features']=4
    return p


def test_tiny_all_methods_actual_budgets_and_no_mutation():
    p=tiny();rows=[];case=p['cases'][0]
    bench.run_block(p,case,3,100000,rows.append,'test')
    assert len(rows)==16 and all(r['status']=='OK' for r in rows)
    for orient in ('raw','rotated'):
        group=[r for r in rows if r['orientation']==orient]
        assert len({r['evaluation_hash'] for r in group})==len({r['training_hash'] for r in group})==1
        for row in group:
            assert row['perms_used']==(0 if row['method'].startswith('C2ST') else 19)
            assert row['reject']==(row['pvalue']<=.05)
            assert len(row['candidate_trace'])==(5 if row['method'].endswith('TrainGrid') else 0)


def test_raw_rotated_training_and_test_pairing():
    p=tiny();a,q=bench.make_data(p,p['cases'][4],3,100000)
    assert np.allclose(q.T@q,np.eye(3),atol=1e-14)
    assert np.allclose(a[2]@q@q.T,a[2],atol=1e-14)
    b,_=bench.make_data(p,p['cases'][4],3,100000)
    assert bench.arrays_hash(*a)==bench.arrays_hash(*b)
    assert not np.shares_memory(a[0],a[2])


def test_seeds_disjoint_across_dimension_case_rep_and_pilot():
    p=tiny();case=p['cases'][0]
    seeds={bench.seed_for(p,case,d,r,1) for d in (20,50) for r in (0,1,100000)}
    assert len(seeds)==6
    assert bench.seed_for(p,p['cases'][1],20,0,1) not in seeds


def test_common_preprocessing_never_uses_evaluation_data():
    p=tiny();a,_=bench.make_data(p,p['cases'][0],3,100000)
    first,h1,c1=bench.prepare(a)
    changed=(*a[:2],a[2]+1000,a[3]-1000)
    second,h2,c2=bench.prepare(changed)
    assert h1==h2 and c1==c2
    assert bench.arrays_hash(*first[:2])==bench.arrays_hash(*second[:2])
    assert not first[0].flags.writeable


@pytest.mark.parametrize('kind',['SCF','RFF'])
def test_train_only_scale_selection_and_first_tie(kind):
    p=tiny();a=np.zeros((6,3));trace=[]
    choice=bench.train_scale(kind,a,a,1.,123,p,trace)
    assert choice==.25 and len(trace)==5 and all(t['score']==0 for t in trace)
    assert all(t['perms_used']==3 for t in trace)


@pytest.mark.parametrize('kind',['SCF','RFF'])
def test_grid_score_matches_formula_and_is_reproducible(kind):
    p=tiny();rng=np.random.default_rng(4);a=rng.normal(size=(6,3));b=rng.normal(size=(6,3));t=[];u=[]
    c=bench.train_scale(kind,a,b,2.,17,p,t);d=bench.train_scale(kind,a,b,2.,17,p,u)
    assert c==d and t==u
    assert c==t[np.argmax([v['score'] for v in t])]['scale']
    for v in t: assert v['score']==(v['stat_obs']-v['permutation_mean'])/max(v['permutation_sd'],1e-12)


@pytest.mark.parametrize('kind',['SCF','MMD'])
def test_grid_and_fixed_scale_share_the_actual_evaluation_statistic(kind):
    p=tiny();a,_=bench.make_data(p,p['cases'][3],3,100000);arrays,scale,_=bench.prepare(a)
    prefix='SCF_GaussianWindow' if kind=='SCF' else 'MMD_RFF'
    tuned={'candidate_trace':[]};fixed={'candidate_trace':[]}
    bench.evaluate(prefix+'_TrainGrid',arrays,scale,17,p,tuned)
    bench.evaluate(prefix+'_Median',arrays,tuned['selected_scale'],17,p,fixed)
    assert tuned['stat_obs']==fixed['stat_obs'] and tuned['pvalue']==fixed['pvalue']
    assert tuned['perms_used']==fixed['perms_used']==19


def test_external_c2st_fit_uses_common_training_only(monkeypatch):
    p=tiny();arrays,_=bench.make_data(p,p['cases'][0],3,100000);arrays,scale,_=bench.prepare(arrays)
    calls=[]
    class Classifier:
        def fit(self,z,labels):
            calls.append(('fit',z.copy(),labels.copy()));return self
        def predict(self,z):
            calls.append(('predict',z.copy()));return np.zeros(len(z),int)
    monkeypatch.setattr(bench,'make_pipeline',lambda *args:Classifier())
    row={'candidate_trace':[]}
    bench.evaluate('C2ST_RBF_Hypergeometric',arrays,scale,7,p,row)
    assert np.array_equal(calls[0][1],np.vstack(arrays[:2]))
    assert np.array_equal(calls[1][1],np.vstack(arrays[2:]))
    assert row['perms_used']==0 and row['pvalue']==1


def test_decision_cli_refuses_missing_budget_authorization(tmp_path):
    command=[sys.executable,'-m','scripts.fair_comparator_bench','--stage','decision','--cohort','tu-cpu','--out',str(tmp_path/'forbidden')]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode!=0 and 'approved inclusive CPU budget' in result.stderr
    assert not (tmp_path/'forbidden').exists()


def test_no_survivor_only_rates_or_paired_differences():
    rows=[{'status':'OK','reject':True}]
    assert analysis.rate_summary(rows,2)['rate'] is None
    assert analysis.rate_summary([{'status':'ERROR','reject':None}],1)['rate'] is None
    assert analysis.paired_summary({0:rows[0]}, {},1,24)['difference'] is None


def test_exact_paired_counts_and_holm_reserved_family_slots():
    a={i:{'status':'OK','reject':i<3} for i in range(10)}
    b={i:{'status':'OK','reject':i in (0,3)} for i in range(10)}
    result=analysis.paired_summary(a,b,10,24)
    assert result['rank_only']==2 and result['comparator_only']==1 and result['difference']==.1
    assert result['lo']<=.1<=result['hi']
    assert np.allclose(analysis.holm([.001,.03,.2],24),[.024,.69,1.])


def test_power_significance_withheld_when_matching_size_is_unresolved():
    p=json.loads(bench.PROTOCOL.read_text());p.update(null_repetitions=4,power_repetitions=12)
    rows=[]
    for cohort,dims in p['cohort_dimensions'].items():
        for d in dims:
            for case in p['cases']:
                for orient in case['orientations']:
                    for method in p['methods']:
                        for r in range(4 if case['null'] else 12):
                            rows.append(dict(cohort=cohort,d=d,case=case['id'],orientation=orient,method=method,
                                             rep=r,status='OK',reject=(not case['null'] and method=='Rank_Pairs_Conservative')))
    rates,contrasts=analysis.decision_analysis(p,rows)
    assert len(contrasts)==24
    assert all(c['holm_pvalue']<.05 for c in contrasts)
    assert all(not c['calibration_eligible'] and not c['controlled_size_superiority'] for c in contrasts)
    # Empty full-data groups stay present, with NA, not a smaller evaluation.
    rates,contrasts=analysis.decision_analysis(p,[])
    assert len(rates)==128 and all(r['rate'] is None for r in rates)


@pytest.fixture(scope='module')
def archive(tmp_path_factory):
    path=tmp_path_factory.mktemp('fair-archive');p=json.loads(bench.PROTOCOL.read_text());rows=[]
    # One actual target-shape block, explicitly incomplete overall and NOT evidence.
    elapsed=bench.run_block(p,p['cases'][2],20,p['pilot_rep_start'],rows.append,'test-local')
    for r in rows:r['protocol_hash']=bench.digest(p)
    assert all(r['status']=='OK' for r in rows)
    (path/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    meta=dict(protocol=p,protocol_hash=bench.digest(p),stage='preflight',cohort='test-local',
              arguments=dict(pilot_reps=1,rep_start=0,rep_stop=None),source_dirty=True,
              source_commit=subprocess.check_output(['git','-C',str(bench.ROOT),'rev-parse','HEAD'],text=True).strip(),
              source_hashes=bench.source_snapshot(path),cpu_model=bench.cpu_model(),python=sys.version,
              numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,threadpools=threadpool_info(),
              written_calls=len(rows),failed_calls=0,status='incomplete_or_failed',
              blocks=[dict(case='null_ar1',d=20,rep=p['pilot_rep_start'],seconds=elapsed)],
              raw_sha256=hashlib.sha256((path/'raw.jsonl').read_bytes()).hexdigest(),
              source_archive_sha256=hashlib.sha256((path/'source.tar.gz').read_bytes()).hexdigest())
    meta['expected_calls']=len(analysis.expected_keys(meta));(path/'metadata.json').write_text(json.dumps(meta))
    return path


def test_incomplete_archive_audited_without_claiming_completion(archive):
    meta,rows=analysis.audit(archive)
    assert len(rows)==8 and meta['expected_calls']==128
    assert (archive/'input_replay_verification.json').is_file()


@pytest.mark.parametrize('mutation',['input','seed','decision','budget','scale','confusion','duplicate','protocol','source'])
def test_deliberate_archive_corruption_is_rejected(archive,tmp_path,mutation):
    path=tmp_path/'copy';shutil.copytree(archive,path)
    meta=json.loads((path/'metadata.json').read_text());rows=[json.loads(s) for s in (path/'raw.jsonl').read_text().splitlines()]
    if mutation=='input': rows[0]['evaluation_hash']='0'*64
    elif mutation=='seed': rows[0]['method_seed']+=1
    elif mutation=='decision': rows[0]['reject']=not rows[0]['reject']
    elif mutation=='budget': rows[0]['perms_used']+=1
    elif mutation=='scale': next(r for r in rows if r['method'].endswith('TrainGrid'))['selected_scale']*=3
    elif mutation=='confusion': next(r for r in rows if r['method'].startswith('C2ST'))['c2st_confusion']['correct_ones']+=1
    elif mutation=='duplicate': rows.append(rows[0]);meta['written_calls']+=1
    elif mutation=='protocol':meta['protocol']['alpha']=.1
    elif mutation=='source':meta['source_hashes'][next(iter(meta['source_hashes']))]='0'*64
    (path/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    meta['raw_sha256']=hashlib.sha256((path/'raw.jsonl').read_bytes()).hexdigest()
    (path/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(AssertionError):analysis.audit(path)
