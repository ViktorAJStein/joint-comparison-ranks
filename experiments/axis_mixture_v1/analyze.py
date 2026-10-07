"""Fail-closed paired analysis; preflight outputs are cost-only, never power evidence."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom, binomtest

if not __debug__:
    raise RuntimeError('Analysis must run without Python optimization')


def holm(values):
    adjusted=[1.]*len(values);running=0.
    for rank,i in enumerate(sorted(range(len(values)),key=lambda k:values[k])):
        running=max(running,min(1.,(len(values)-rank)*values[i]))
        adjusted[i]=running
    return adjusted


def read_archives(paths, wanted_stage):
    rows=[];metas=[];keys=set();signature={}
    for path in paths:
        meta=json.loads((path/'metadata.json').read_text());p=meta['protocol']
        assert meta['stage']==wanted_stage and meta['status']=='complete' and meta['failed_calls']==0
        raw=(path/'raw.jsonl').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==meta['raw_sha256']
        assert hashlib.sha256((path/'source.tar.gz').read_bytes()).hexdigest()==meta['source_archive_sha256']
        receipt=json.loads((path/'input_replay_verification.json').read_text())
        assert receipt['status']=='pass' and receipt['calls']==meta['written_calls']
        assert receipt['metadata_sha256']==hashlib.sha256((path/'metadata.json').read_bytes()).hexdigest()
        assert receipt['raw_sha256']==meta['raw_sha256'] and receipt['source_archive_sha256']==meta['source_archive_sha256']
        # Receipt bytes are preserved; worker replay is not fabricated on another CPU.
        if metas:
            assert meta['protocol_hash']==metas[0]['protocol_hash'] and meta['source_hashes']==metas[0]['source_hashes']
        for d in (meta['arguments']['dimensions'] or p['dimensions']):
            numerical=(meta['cpu_model'],meta['python'],meta['numpy'],meta['scipy'],meta['sklearn'])
            assert d not in signature or signature[d]==numerical, 'Numerical/hardware cohort mismatch'
            signature[d]=numerical
        part=[json.loads(line) for line in raw.splitlines()]
        assert len(part)==meta['written_calls']==meta['expected_calls']
        for row in part:
            key=(row['case'],row['d'],row['rep'],row['method'])
            assert key not in keys and row['status']=='OK' and not row['warnings']
            assert row['protocol_hash']==meta['protocol_hash']
            assert row['reject']==(row['pvalue']<=p['alpha'])
            keys.add(key)
        rows.extend(part);metas.append(meta)
    return metas,rows


def analyze(paths,stage):
    metas,rows=read_archives(paths,stage);p=metas[0]['protocol']
    byblock=defaultdict(list)
    for meta in metas:
        for b in meta['blocks']:byblock[(b['d'],b['case'])].append(b['seconds'])
    if stage=='preflight':
        details=[]
        for (d,case),seconds in sorted(byblock.items()):
            n=p['null_repetitions'] if case=='null_gaussian' else p['power_repetitions']
            details.append({'d':d,'case':case,'observed_blocks':len(seconds),
                            'mean_seconds':float(np.mean(seconds)),'max_seconds':max(seconds),
                            'mean_forecast_cpu_hours':n*float(np.mean(seconds))/3600,
                            'two_max_planning_cpu_hours':2*n*max(seconds)/3600})
        return {'stage':'preflight','scope':'feasibility/cost only; no power conclusion',
                'calls':len(rows),'details':details,
                'mean_forecast_cpu_hours':sum(r['mean_forecast_cpu_hours'] for r in details),
                'two_max_planning_cpu_hours':sum(r['two_max_planning_cpu_hours'] for r in details),
                'cpu_models':sorted(set(m['cpu_model'] for m in metas))}
    # Full protected rows, no survivor frequencies or smaller replacement strata.
    wanted={(c['id'],d,rep,method) for d in p['dimensions'] for c in p['cases']
            for rep in range(p['null_repetitions'] if c['null'] else p['power_repetitions'])
            for method in p['methods']}
    assert {(r['case'],r['d'],r['rep'],r['method']) for r in rows}==wanted
    groups=defaultdict(dict)
    for r in rows:groups[r['case'],r['d'],r['method']][r['rep']]=r
    summary=[];screens={}
    for d in p['dimensions']:
        for case in p['cases']:
            n=p['null_repetitions'] if case['null'] else p['power_repetitions']
            for method in p['methods']:
                group=groups[case['id'],d,method]
                assert set(group)==set(range(n))
                k=sum(v['reject'] for v in group.values())
                lo=0. if k==0 else float(beta.ppf(.025,k,n-k+1))
                hi=1. if k==n else float(beta.ppf(.975,k+1,n-k))
                row={'case':case['id'],'d':d,'method':method,'n':n,'rejections':k,'rate':k/n,
                     'pointwise_exact_95_interval':[lo,hi],
                     'mean_total_seconds':float(np.mean([v['total_seconds'] for v in group.values()]))}
                if case['null']:
                    adjusted=min(1.,p['analysis']['null_family_size']*binomtest(k,n,p['alpha'],alternative='greater').pvalue)
                    upper=1. if k==n else float(beta.ppf(1-.05/p['analysis']['null_family_size'],k+1,n-k))
                    status='overreject' if adjusted<=.05 else 'inconclusive' if upper>p['alpha']+p['analysis']['null_upper_margin'] else 'resolved'
                    row.update(adjusted_overrejection_pvalue=adjusted,simultaneous_upper=upper,size_status=status)
                    screens[d,method]=status
                summary.append(row)
    contrasts=[]
    for d in p['dimensions']:
        for case in p['cases'][1:]:
            primary=p['analysis']['primary_rank'];a=groups[case['id'],d,primary];n=len(a)
            for method in p['analysis']['primary_comparators']:
                b=groups[case['id'],d,method]
                plus=sum(a[i]['reject'] and not b[i]['reject'] for i in range(n))
                minus=sum(b[i]['reject'] and not a[i]['reject'] for i in range(n))
                discord=plus+minus;delta=(plus-minus)/n
                pv=1. if discord==0 else min(1.,2*float(binom.cdf(min(plus,minus),discord,.5)))
                se=math.sqrt(max(0.,discord/n-delta**2)/(n-1))
                contrasts.append({'case':case['id'],'d':d,'rank':primary,'comparator':method,
                                  'a_only':plus,'b_only':minus,'difference':delta,'exact_pvalue':pv,
                                  'paired_pointwise_normal_95_interval':[max(-1.,delta-1.96*se),min(1.,delta+1.96*se)],
                                  'matching_nulls_resolved':screens[d,primary]==screens[d,method]=='resolved'})
    assert len(contrasts)==p['analysis']['primary_power_family_size']==45
    for row,pv in zip(contrasts,holm([r['exact_pvalue'] for r in contrasts])):
        row.update(holm_pvalue=pv,qualified_rank_gain=row['difference']>0 and pv<=.05 and row['matching_nulls_resolved'])
    return {'stage':'decision','calls':len(rows),'summary':summary,'paired_contrasts':contrasts,
            'platforms_preserved':sorted(set(m['platform'] for m in metas))}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archives',nargs='+',type=Path,required=True)
    parser.add_argument('--stage',choices=('preflight','decision'),required=True)
    parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    a.out.write_text(json.dumps(analyze(a.archives,a.stage),indent=2))
