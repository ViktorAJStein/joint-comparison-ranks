"""New paired generator benchmark; frozen numerical source remains unchanged.

Batching evaluates precisely the same rank tables and generator equations as
RankGTestSubspaces, with identical RNG order. No new statistical test is proposed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings
import numpy as np
from scipy import __version__ as scipy_version
from sklearn import __version__ as sklearn_version
from threadpoolctl import threadpool_info
from scripts.fair_comparator_bench import make_data, prepare, arrays_hash, seed_for, cpu_model, digest
from ranktwosample.rank_two_sample import RankGTestConfig
from ranktwosample.rank_two_sample_subspaces import RankGTestSubspaces, SubspaceOpts

HERE=Path(__file__).resolve().parent
KINDS=('chi2','gtest','freeman_tukey','cr:2')


def batch_values(hx,hy,n,m,gamma,kind):
    """Allocation-by-cell arrays; identical arithmetic to frozen scalar helper."""
    hx=hx.astype(np.float64);hy=hy.astype(np.float64)
    # Match qhat followed by N multiplication, NOT N*(HY+gamma)/denominator.
    expected=float(n)*((hy+float(gamma))/(float(m)+float(gamma)*hx.shape[1]))
    if kind=='chi2':return ((hx-expected)**2/expected).sum(axis=1)
    if kind=='gtest':
        ratio=np.where(hx>0,hx/expected,1.)
        return 2*np.where(hx>0,hx*np.log(ratio),0.).sum(axis=1)
    lam=-.5 if kind=='freeman_tukey' else 2.
    if kind not in ('freeman_tukey','cr:2'):raise ValueError(kind)
    ratio=hx/expected
    f=(ratio**(lam+1)-ratio-lam*(ratio-1))/(lam*(lam+1))
    return 2*(expected*f).sum(axis=1)


def joint_test(x,y,seed,p,kinds=KINDS):
    """Shared allocation tables, independent fixed-B plus-one p-value for each f."""
    cfg=RankGTestConfig(alpha=p['alpha'],B=p['evaluation_permutations'],
                       **{k:v for k,v in p['rank'].items() if k!='L'})
    test=RankGTestSubspaces(cfg,SubspaceOpts(L=p['rank']['L'],k_dim=2,dedup=True,pca=False))
    rng=test._runner._rng(seed)
    began=time.perf_counter()
    oracle_observed,env=test._compute_T_obs_sum_shared_pool(x,y,rng)
    init_seconds=time.perf_counter()-began
    kr,d,_,_,_,jitter,side,cols,views,n,m,refpositions=env
    if side!='right' or jitter is None or cfg.gpu!='off':raise RuntimeError('Unqualified rank mode')
    alloc=p['evaluation_permutations']+1;total=n+m
    permutations=np.asarray([rng.permutation(total) for _ in range(alloc-1)])
    iy=np.vstack([np.arange(n,total),permutations[:,:m]])
    ix=np.vstack([np.arange(n),permutations[:,m:]])
    z=np.vstack([x,y])+jitter.T
    scores={kind:np.zeros(alloc) for kind in kinds}
    construction_seconds=init_seconds
    generator_seconds={kind:0. for kind in kinds}
    table_hash=hashlib.sha256()
    offsets=(np.arange(alloc)*(kr+1)**2)[:,None]
    for view,rp in zip(views,refpositions):
        before=time.perf_counter()
        refs=iy[:,rp]
        a,b=view
        # Right-continuous comparisons, including tied values and reference self counts.
        ra=(z[None,:,a,None]>=z[refs,a][:,None,:]).sum(axis=2)
        rb=(z[None,:,b,None]>=z[refs,b][:,None,:]).sum(axis=2)
        cells=ra*(kr+1)+rb;ncells=(kr+1)**2
        hx=np.bincount((offsets+np.take_along_axis(cells,ix,axis=1)).ravel(),minlength=alloc*ncells).reshape(alloc,ncells)
        hy=np.bincount((offsets+np.take_along_axis(cells,iy,axis=1)).ravel(),minlength=alloc*ncells).reshape(alloc,ncells)
        if not (np.all(hx.sum(axis=1)==n) and np.all(hy.sum(axis=1)==m)):raise RuntimeError('Histogram mass')
        table_hash.update(hx.astype('<i4').tobytes());table_hash.update(hy.astype('<i4').tobytes())
        construction_seconds+=time.perf_counter()-before
        for kind in kinds:
            before=time.perf_counter();scores[kind]+=batch_values(hx,hy,n,m,cfg.alpha0,kind)
            generator_seconds[kind]+=time.perf_counter()-before
    # Initialization has a frozen scalar G observation, serving as a bound replay check.
    if 'gtest' in scores and scores['gtest'][0]!=oracle_observed:
        raise RuntimeError(f'G observation not bitwise identical: {scores["gtest"][0]} != {oracle_observed}')
    results={}
    for kind,values in scores.items():
        if not np.isfinite(values).all():raise FloatingPointError('Nonfinite statistic')
        pv=float((1+np.sum(values[1:]>=values[0]))/alloc)
        results[kind]={'stat_obs':float(values[0]),'pvalue':pv,'reject':pv<=p['alpha'],
                       'perms_used':alloc-1,'permutation_statistics':values[1:].tolist(),
                       'generator_seconds':generator_seconds[kind]}
    common={'effective_L':len(views),'views_hash':digest(views),
            'reference_positions_hash':digest([r.tolist() for r in refpositions]),
            'permutations_hash':arrays_hash(permutations),'jitter_hash':arrays_hash(jitter),
            'rank_tables_hash':table_hash.hexdigest(),'construction_seconds':construction_seconds,
            'joint_seconds':time.perf_counter()-began}
    return results,common


def standalone_timings(x,y,seed,p,rep):
    order=np.random.default_rng(np.random.SeedSequence([seed,0x54494D45])).permutation(len(KINDS))
    result={}
    for index in order:
        kind=KINDS[index];start=time.perf_counter()
        scores,common=joint_test(x,y,seed,p,kinds=(kind,))
        result[kind]={'standalone_seconds':time.perf_counter()-start,'timing_order_position':len(result),
                      'stat_obs':scores[kind]['stat_obs'],'pvalue':scores[kind]['pvalue'],
                      'rank_tables_hash':common['rank_tables_hash']}
    return result


def source_receipt(base,out):
    paths=sorted((base/'src').rglob('*.py'))+[base/'pyproject.toml',base/'uv.lock']
    paths+=sorted(HERE.glob('*.py'))+[HERE/'protocol.json']
    result={str(q.relative_to(base)) if q.is_relative_to(base) else 'overlay/'+q.name:
            hashlib.sha256(q.read_bytes()).hexdigest() for q in paths}
    (out/'source_hashes.json').write_text(json.dumps(result,indent=2))
    return result


def run(p,args):
    base=Path(os.environ['GENERATOR_BASE_SOURCE']).resolve()
    commit=subprocess.check_output(['git','-C',str(base),'rev-parse','HEAD'],text=True).strip()
    if commit!=p['base_source_commit']:raise RuntimeError('Frozen base mismatch')
    if subprocess.check_output(['git','-C',str(base),'status','--porcelain'],text=True).strip():raise RuntimeError('Dirty numerical base')
    if any(t['num_threads']!=1 for t in threadpool_info()):raise RuntimeError('Multiple numerical threads')
    if args.stage=='decision' and args.authorization is None:raise PermissionError('Decision-scale cap not yet admitted')
    if args.stage=='decision':
        approval=json.loads(args.authorization.read_text())
        if not approval.get('approved_by_user') or approval.get('protocol_hash')!=digest(p):raise PermissionError('Unapproved protocol/budget')
        if approval.get('maximum_total_cpu_core_hours',0)>8 or approval.get('maximum_total_cpu_core_hours',0)<=0:raise PermissionError('CPU cap')
        if approval.get('reserved_cpu_core_hours',float('inf'))>approval['maximum_total_cpu_core_hours']:raise PermissionError('Overcommitted cap')
        if approval.get('source_commit')!=commit or approval.get('cpu_model')!=cpu_model():raise PermissionError('Source/CPU admission')
        for name,expected in approval.get('numeric_overlay_hashes',{}).items():
            if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Numeric overlay changed')
        if set(approval.get('numeric_overlay_hashes',{}))!={'runner.py','protocol.json'}:raise PermissionError('Unbound numeric overlay')
    if args.pilot_reps<1 or args.rep_start<0 or (args.stage=='decision' and args.rep_stop<=args.rep_start):raise ValueError('Empty/invalid replication range')
    if args.dimensions and not set(args.dimensions)<=set(p['dimensions']):raise ValueError('Unprotected dimension')
    args.out.mkdir(parents=True,exist_ok=False)
    receipt=source_receipt(base,args.out)
    meta={'protocol':p,'protocol_hash':digest(p),'stage':args.stage,'source_commit':commit,'source_hashes':receipt,
          'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
          'cpu_model':cpu_model(),'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,
          'scipy':scipy_version,'sklearn':sklearn_version,'node':platform.node(),'slurm_job':os.environ.get('SLURM_JOB_ID'),
          'task':os.environ.get('SLURM_ARRAY_TASK_ID'),'status':'running','groups_completed':0,'groups_failed':0,
          'start_unix':time.time(),'blocks':[]}
    if args.stage=='preflight':ranges=range(p['pilot_rep_start'],p['pilot_rep_start']+args.pilot_reps)
    else:ranges=range(args.rep_start,args.rep_stop)
    tasks=[(case,d,r) for d in args.dimensions or p['dimensions'] for case in p['cases'] for r in ranges
           if args.stage=='preflight' or r<(p['null_repetitions'] if case['null'] else p['power_repetitions'])]
    meta['expected_groups']=sum(len(c['orientations']) for c,d,r in tasks)
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2))
    began=time.perf_counter()
    with (args.out/'raw.jsonl').open('x') as stream:
        try:
            for case,d,rep in tasks:
                block_start=time.perf_counter();raw,rotation=make_data(p,case,d,rep)
                for orientation in case['orientations']:
                    row={'case':case['id'],'null':case['null'],'matched_null':case.get('matched_null'),
                         'd':d,'rep':rep,'orientation':orientation,'data_seed':seed_for(p,case,d,rep,1),
                         'rotation_seed':seed_for(p,case,d,rep,2),'method_seed':seed_for(p,case,d,rep,10),
                         'base_data_hash':arrays_hash(*raw),'protocol_hash':digest(p),'status':'OK'}
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter('always')
                        try:
                            oriented=raw if orientation=='raw' else tuple(a@rotation for a in raw)
                            arrays,_,_=prepare(oriented);x,y=arrays[2:]
                            row['evaluation_hash']=arrays_hash(x,y)
                            result,common=joint_test(x,y,row['method_seed'],p)
                            if args.stage=='preflight' or rep<p['standalone_timing_repetitions']:
                                timings=standalone_timings(x,y,row['method_seed'],p,rep)
                                for kind,value in timings.items():
                                    if value['stat_obs']!=result[kind]['stat_obs'] or value['pvalue']!=result[kind]['pvalue'] or value['rank_tables_hash']!=common['rank_tables_hash']:
                                        raise RuntimeError('Standalone/shared mismatch')
                                    result[kind].update(value)
                            row.update(common=common,results=result)
                            meta['groups_completed']+=1
                        except Exception as e:
                            row.update(status=f'ERROR:{type(e).__name__}:{e}',results=None)
                            meta['groups_failed']+=1
                        row['warnings']=[str(w.message) for w in captured]
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                meta['blocks'].append({'case':case['id'],'d':d,'rep':rep,'seconds':time.perf_counter()-block_start})
        finally:
            meta.update(status='complete' if meta['groups_completed']==meta['expected_groups'] and not meta['groups_failed'] else 'incomplete_or_failed',
                        elapsed_seconds=time.perf_counter()-began,raw_sha256=hashlib.sha256((args.out/'raw.jsonl').read_bytes()).hexdigest())
            (args.out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:meta[k] for k in ('status','groups_completed','groups_failed','elapsed_seconds')}))
    if meta['status']!='complete':raise SystemExit(1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--stage',choices=('preflight','decision'),default='preflight')
    ap.add_argument('--pilot-reps',type=int,default=2);ap.add_argument('--rep-start',type=int,default=0)
    ap.add_argument('--rep-stop',type=int,default=0);ap.add_argument('--dimensions',type=int,nargs='+')
    ap.add_argument('--authorization',type=Path)
    args=ap.parse_args();run(json.loads((HERE/'protocol.json').read_text()),args)
if __name__=='__main__':main()
