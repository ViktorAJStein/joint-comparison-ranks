"""Strict archive audit, pricing-only pilot output, and protected paired analysis."""
import argparse
from collections import defaultdict
from functools import lru_cache
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import numpy as np
from scipy.stats import beta, binomtest, hypergeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from scripts.fair_comparator_bench import PROTOCOL, arrays_hash, cpu_model, digest, make_data, prepare, seed_for


def expected_keys(meta):
    p, a = meta['protocol'], meta['arguments']
    dims = p['dimensions'] if meta['stage']=='preflight' else p['cohort_dimensions'][meta['cohort']]
    keys = set()
    for d in dims:
        for case in p['cases']:
            n = p['null_repetitions'] if case['null'] else p['power_repetitions']
            reps = range(p['pilot_rep_start'], p['pilot_rep_start']+a['pilot_reps']) if meta['stage']=='preflight' else range(a['rep_start'], min(n,a['rep_stop']) if a['rep_stop'] is not None else n)
            keys.update((case['id'],d,r,o,m) for r in reps for o in case['orientations'] for m in p['methods'])
    return keys


def audit(path):
    meta = json.loads((path/'metadata.json').read_text())
    p = meta['protocol']
    assert p == json.loads(PROTOCOL.read_text()), 'Protocol drift'
    assert meta['protocol_hash'] == digest(p), 'Protocol hash'
    assert hashlib.sha256((path/'raw.jsonl').read_bytes()).hexdigest()==meta['raw_sha256'], 'Raw checksum'
    assert hashlib.sha256((path/'source.tar.gz').read_bytes()).hexdigest()==meta['source_archive_sha256'], 'Source archive checksum'
    with tarfile.open(path/'source.tar.gz') as archive:
        assert len(archive.getmembers()) == len(meta['source_hashes']), 'Source file set'
        for member in archive:
            assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
            body = archive.extractfile(member).read()
            assert hashlib.sha256(body).hexdigest()==meta['source_hashes'][member.name], 'Archived source hash'
            if not meta['source_dirty']:
                frozen = subprocess.check_output(['git','-C',str(ROOT),'show',meta['source_commit']+':'+member.name])
                assert frozen==body, 'Frozen commit differs'
    if meta['stage']=='decision': assert not meta['source_dirty'], 'Dirty decision source'
    assert all(v['num_threads']==1 for v in meta['threadpools']), 'Thread cohort'
    fingerprint = {'cpu_model':cpu_model(), 'python':sys.version, 'numpy':np.__version__}
    import scipy, sklearn
    fingerprint.update(scipy=scipy.__version__, sklearn=sklearn.__version__)
    same_environment = all(meta[k] == value for k,value in fingerprint.items())
    binding = {'metadata_hash':digest(meta), 'raw_sha256':meta['raw_sha256'],
               'source_archive_sha256':meta['source_archive_sha256'], 'protocol_hash':meta['protocol_hash']}
    receipt = path/'input_replay_verification.json'
    if not same_environment:
        assert receipt.is_file(), 'Need original-cohort input replay receipt; do not assume cross-hardware bitwise equality'
        proof = json.loads(receipt.read_text())
        assert proof['binding']==binding and proof['status']=='pass', 'Stale input replay receipt'
        assert proof['environment']=={k:meta[k] for k in fingerprint}, 'Receipt environment'
    @lru_cache(maxsize=4)
    def replay(case_id,d,rep,orientation):
        case = next(c for c in p['cases'] if c['id']==case_id)
        raw,q = make_data(p,case,d,rep)
        oriented = raw if orientation=='raw' else tuple(a@q for a in raw)
        prepared,scale,center = prepare(oriented)
        return {'base_data_hash':arrays_hash(*raw), 'oriented_data_hash':arrays_hash(*oriented),
                'training_hash':arrays_hash(*prepared[:2]), 'evaluation_hash':arrays_hash(*prepared[2:]),
                'median_scale':scale, 'center_hash':center}
    rows = [json.loads(line) for line in (path/'raw.jsonl').read_text().splitlines()]
    wanted = expected_keys(meta)
    assert meta['expected_calls']==len(wanted) and meta['written_calls']==len(rows)
    seen=set(); failed=0
    for row in rows:
        key = (row['case'],row['d'],row['rep'],row['orientation'],row['method'])
        assert key in wanted and key not in seen, 'Duplicate/undeclared call'
        seen.add(key)
        assert row['cohort']==meta['cohort'] and row['protocol_hash']==meta['protocol_hash']
        case = next(c for c in p['cases'] if c['id']==row['case'])
        assert row['null']==case['null']
        assert row['data_seed']==seed_for(p,case,row['d'],row['rep'],1)
        assert row['rotation_seed']==seed_for(p,case,row['d'],row['rep'],2)
        domain=10 if row['method'].startswith('Rank') else 11 if row['method'].startswith('SCF') else 12 if row['method'].startswith('MMD') else 13
        assert row['method_seed']==seed_for(p,case,row['d'],row['rep'],domain), 'Method seed'
        if same_environment:
            hashes = replay(row['case'],row['d'],row['rep'],row['orientation'])
            assert all(row[k]==value for k,value in hashes.items()), 'Input replay hash'
        scale = row['median_scale']
        assert np.isfinite([row['call_seconds'],row['total_seconds'],row['common_prep_seconds']]).all()
        assert row['call_seconds']>=0 and row['total_seconds']>=row['call_seconds']
        if row['status']!='OK':
            failed+=1; assert row['reject'] is None; continue
        assert np.isfinite([row['stat_obs'],row['pvalue']]).all() and 0<=row['pvalue']<=1
        assert row['reject']==(row['pvalue']<=p['alpha']), 'Decision identity'
        assert row['perms_used']==(0 if row['method'].startswith('C2ST') else p['evaluation_permutations']), 'Budget'
        if row['method'].startswith('C2ST'):
            c=row['c2st_confusion'];n=c['n'];n1=c['true_ones'];k=c['predicted_ones'];u=c['correct_ones']
            assert n==2*p['evaluation_per_class'] and n1==p['evaluation_per_class']
            assert row['pvalue']==float(hypergeom.sf(u-1,n,k,n1)), 'C2ST tail'
            assert row['stat_obs']==(n-n1-k+2*u)/n, 'C2ST accuracy'
        else:
            assert row['c2st_confusion'] is None
            scaled=row['pvalue']*(p['evaluation_permutations']+1)
            assert scaled>=1-1e-12 and abs(scaled-round(scaled))<1e-10, 'Plus-one grid'
        if row['method'].startswith('Rank'):
            expected=p['rank']['L'] if 'Pairs' in row['method'] else min(p['rank']['L'],row['d'])
            assert row['effective_L']==expected
            assert len(row['views_hash'])==len(row['reference_positions_hash'])==64
        trace=row['candidate_trace']
        if row['method'].endswith('TrainGrid'):
            assert len(trace)==len(p['scale_multipliers'])
            for t,mult in zip(trace,p['scale_multipliers']):
                assert t['multiplier']==mult and t['scale']==scale*mult
                assert t['perms_used']==p['training_permutations_per_candidate']
                assert np.isclose(t['score'],(t['stat_obs']-t['permutation_mean'])/max(t['permutation_sd'],1e-12), rtol=1e-12,atol=1e-12)
            assert row['selected_scale']==trace[int(np.argmax([t['score'] for t in trace]))]['scale'], 'Training selection'
        else:
            assert not trace
            assert row['selected_scale']==(scale if row['method'].startswith(('SCF','MMD')) else None)
    assert failed==meta['failed_calls']
    if meta['status']=='complete': assert seen==wanted and not failed, 'Incomplete complete archive'
    # Each complete representation block has the same training/evaluation hashes.
    groups=defaultdict(list)
    for row in rows: groups[(row['case'],row['d'],row['rep'],row['orientation'])].append(row)
    for group in groups.values():
        assert len({(r['training_hash'],r['evaluation_hash']) for r in group})==1, 'Unpaired input'
        assert len({r['order_position'] for r in group})==len(group), 'Duplicate scheduling position'
    if same_environment:
        receipt.write_text(json.dumps({'status':'pass', 'binding':binding,
                                      'environment':fingerprint, 'verified_calls':len(rows)},indent=2))
    return meta,rows


def cp(k,n,tail_alpha=.05):
    return (0. if k==0 else float(beta.ppf(tail_alpha/2,k,n-k+1)),
            1. if k==n else float(beta.ppf(1-tail_alpha/2,k+1,n-k)))


def holm(values, family):
    order=np.argsort(values); adjusted=[1.]*len(values); maximum=0.
    for rank,i in enumerate(order):
        maximum=max(maximum,min(1.,(family-rank)*values[i]));adjusted[int(i)]=maximum
    return adjusted


def rate_summary(rows, expected):
    if len(rows)!=expected or any(r['status']!='OK' for r in rows):
        return {'complete':False,'rate':None,'lo':None,'hi':None}
    k=sum(r['reject'] for r in rows);lo,hi=cp(k,expected)
    result={'complete':True,'rejections':k,'rate':k/expected,'lo':lo,'hi':hi}
    if all('total_seconds' in r for r in rows):
        for field in ('common_prep_seconds','training_seconds','evaluation_seconds','total_seconds'):
            result['mean_'+field]=float(np.mean([r[field] for r in rows]))
        result['median_total_seconds']=float(np.median([r['total_seconds'] for r in rows]))
        result['p95_total_seconds']=float(np.quantile([r['total_seconds'] for r in rows],.95))
    return result


def paired_summary(a,b,expected,family):
    if set(a)!=set(b) or len(a)!=expected or any(r['status']!='OK' for r in [*a.values(),*b.values()]):
        return {'complete':False,'difference':None,'pvalue':None,'lo':None,'hi':None}
    win=sum(a[i]['reject'] and not b[i]['reject'] for i in a)
    loss=sum(b[i]['reject'] and not a[i]['reject'] for i in a)
    wl,wh=cp(win,expected,.05/(2*family));ll,lh=cp(loss,expected,.05/(2*family))
    return {'complete':True,'rank_only':win,'comparator_only':loss,'difference':(win-loss)/expected,
            'lo':wl-lh,'hi':wh-ll,'pvalue':float(binomtest(win,win+loss,.5).pvalue) if win+loss else 1.}


def decision_analysis(p, rows):
    groups=defaultdict(list)
    for row in rows: groups[(row['cohort'],row['case'],row['d'],row['orientation'],row['method'])].append(row)
    summaries=[];lookup={};contrasts=[]
    for cohort,dims in p['cohort_dimensions'].items():
        for d in dims:
            for case in p['cases']:
                for orient in case['orientations']:
                    for method in p['methods']:
                        key=(cohort,case['id'],d,orient,method)
                        expected=p['null_repetitions'] if case['null'] else p['power_repetitions']
                        result=rate_summary(groups[key],expected)
                        item=dict(cohort=cohort,case=case['id'],d=d,orientation=orient,method=method,**result)
                        if case['null'] and result['complete']:
                            k=result['rejections'];family=p['analysis']['null_family_size']
                            item['adjusted_overrejection_p']=min(1.,family*float(binomtest(k,expected,p['alpha'],alternative='greater').pvalue))
                            item['simultaneous_upper']=1. if k==expected else float(beta.ppf(1-.05/family,k+1,expected-k))
                            item['calibration_screen']=item['adjusted_overrejection_p']>=.05 and item['simultaneous_upper']<=p['alpha']+p['analysis']['null_upper_margin']
                        else: item['calibration_screen']=None
                        summaries.append(item);lookup[key]=item
                    if not case['null']:
                        primary=p['analysis']['primary_rank']
                        a={r['rep']:r for r in groups[(cohort,case['id'],d,orient,primary)]}
                        for method in p['analysis']['primary_comparators']:
                            b={r['rep']:r for r in groups[(cohort,case['id'],d,orient,method)]}
                            result=paired_summary(a,b,p['power_repetitions'],p['analysis']['primary_power_family_size'])
                            nk=(cohort,case['matched_null'],d,orient)
                            eligible=lookup[(*nk,primary)]['calibration_screen'] is True and lookup[(*nk,method)]['calibration_screen'] is True
                            contrasts.append(dict(cohort=cohort,case=case['id'],d=d,orientation=orient,comparator=method,
                                                  calibration_eligible=eligible,**result))
    # Keep every declared family slot, including incomplete or ineligible groups.
    adj=holm([c['pvalue'] if c['complete'] else 1. for c in contrasts],p['analysis']['primary_power_family_size'])
    for c,v in zip(contrasts,adj):
        c['holm_pvalue']=v if c['complete'] else None
        c['controlled_size_superiority']=bool(c['complete'] and c['calibration_eligible'] and v<.05 and c['lo']>0)
    return summaries,contrasts


def pricing(p, archives):
    # These are LOCAL COHORT forecasts, not measured TU/LRZ prices or approvals.
    by_group=defaultdict(list)
    for meta,rows in archives:
        for b in meta['blocks']: by_group[(meta['cohort'],meta['cpu_model'],b['d'],b['case'])].append(b['seconds'])
    details=[]
    for (cohort,cpu,d,case),seconds in sorted(by_group.items()):
        spec=next(c for c in p['cases'] if c['id']==case)
        reps=p['null_repetitions'] if spec['null'] else p['power_repetitions']
        mean=float(np.mean(seconds));maximum=max(seconds)
        details.append(dict(cohort=cohort,cpu=cpu,d=d,case=case,pilot_blocks=len(seconds),mean_block_seconds=mean,
                            max_block_seconds=maximum,full_repetitions=reps,mean_forecast_core_hours=mean*reps/3600,
                            twice_max_forecast_core_hours=2*maximum*reps/3600))
    return details


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archives',nargs='+',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    archives=[audit(path) for path in args.archives];p=archives[0][0]['protocol']
    assert len({m['stage'] for m,_ in archives})==1, 'Do not mix pilots and evidence'
    assert len({m['source_commit'] for m,_ in archives})==1, 'Source cohort mismatch'
    seen=set();allrows=[];cohorts={}
    for meta,rows in archives:
        signature=tuple(meta[k] for k in ('cpu_model','python','numpy','scipy','sklearn','platform'))
        assert meta['cohort'] not in cohorts or cohorts[meta['cohort']]==signature, 'Hardware/software cohort mismatch'
        cohorts[meta['cohort']]=signature
        for row in rows:
            key=(row['case'],row['d'],row['rep'],row['orientation'],row['method'])
            assert key not in seen, 'Duplicate shard or independent-evidence duplication'
            seen.add(key);allrows.append(row)
    if archives[0][0]['stage']=='preflight':
        assert all(m['status']=='complete' for m,_ in archives), 'Incomplete pilots cannot price a complete campaign'
        output={'stage':'preflight','performance_evidence':False,'verified_calls':len(allrows),
                'pricing':pricing(p,archives),'hardware':cohorts}
        details=output['pricing']
        with (args.out/'pricing.csv').open('x',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    else:
        rates,contrasts=decision_analysis(p,allrows)
        output={'stage':'decision','rates':rates,'paired_contrasts':contrasts,'hardware':cohorts,
                'incomplete_groups':sum(not r['complete'] for r in rates)}
    (args.out/'analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps({'stage':output['stage'],'verified_calls':len(allrows),'performance_evidence':output['stage']=='decision'}))


if __name__=='__main__':main()
