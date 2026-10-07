"""Fail-closed raw-tail audit and predeclared inference for dependence v1."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import platform
import sys
import tarfile
import numpy as np
from scipy.stats import beta, binomtest
from runner import digest, make_data, arrays_hash, seed_for, strata, cpu_model

HERE = Path(__file__).resolve().parent


def interval(k, n, level=.95):
    a = (1-level)/2
    return [float(beta.ppf(a,k,n-k+1)) if k else 0.,
            float(beta.ppf(1-a,k+1,n-k)) if k<n else 1.]


def holm(values, family):
    ordered = sorted(values.items(), key=lambda v:v[1])
    if len(ordered)>family: raise ValueError('Expanded hypothesis family')
    adjusted = {}; running = 0.
    for i,(key,value) in enumerate(ordered):
        running = max(running,min(1.,(family-i)*value));adjusted[key]=running
    return adjusted


def audit(path, p, replay=False):
    meta = json.loads((path/'metadata.json').read_text())
    rawpath = path/'raw.jsonl'
    if meta['protocol_hash'] != digest(p) or meta['protocol'] != p:
        raise RuntimeError('Protocol drift')
    if meta['source_commit'] != p['base_source_commit'] or meta['status'] != 'complete' or meta['failed_blocks']:
        raise RuntimeError('Incomplete or wrong source')
    if hashlib.sha256(rawpath.read_bytes()).hexdigest() != meta['raw_sha256']:
        raise RuntimeError('Raw checksum drift')
    if meta['completed_blocks'] != meta['expected_blocks']:
        raise RuntimeError('Wrong metadata denominator')
    with tarfile.open(path/'source.tar.gz', 'r:gz') as archive:
        members=archive.getmembers()
        if len(members)!=len({m.name for m in members}) or set(m.name for m in members)!=set(meta['source_hashes']):
            raise RuntimeError('Source archive membership mismatch')
        for member in members:
            stream=archive.extractfile(member)
            if stream is None or hashlib.sha256(stream.read()).hexdigest()!=meta['source_hashes'][member.name]:
                raise RuntimeError('Source archive content mismatch')
    # Current numerical overlay must equal the archived one; never regenerate outputs.
    for name in ('runner.py','protocol.json'):
        if meta['source_hashes'].get('overlay/'+name) != hashlib.sha256((HERE/name).read_bytes()).hexdigest():
            raise RuntimeError('Numerical overlay drift')
    selected = [s for s in strata(p) if (meta['arguments']['study']=='both' or s['study']==meta['arguments']['study'])
                and (meta['arguments']['stratum'] is None or s['id'] in meta['arguments']['stratum'])]
    args = meta['arguments']
    reps = range(p['pilot_rep_start'],p['pilot_rep_start']+args['pilot_reps']) if meta['stage']=='preflight' else range(args['rep_start'],args['rep_stop'])
    expected = {(s['id'],r) for s in selected for r in reps if meta['stage']=='preflight'
                or r<p['null_repetitions' if s['null'] else 'power_repetitions']}
    result=[];seen=set();replayed=set()
    with rawpath.open() as stream:
        for line in stream:
            row=json.loads(line);s=row['stratum'];rep=row['rep'];key=(s['id'],rep)
            if key in seen or key not in expected or s!=strata(p)[s['id']]:
                raise RuntimeError('Duplicate/unexpected row')
            seen.add(key)
            if row['status']!='OK' or row['warnings'] or row['protocol_hash']!=digest(p):
                raise RuntimeError('Failed/warning/raw protocol mismatch')
            if row['data_seed']!=seed_for(p,s,rep,1) or row['method_seed']!=seed_for(p,s,rep,10):
                raise RuntimeError('Wrong seed')
            methods=p[s['study']]['methods']
            if set(row['results'])!=set(methods): raise RuntimeError('Missing method')
            for name,value in row['results'].items():
                scores=value['permutation_statistics']
                if len(scores)!=p['B'] or value['perms_used']!=p['B'] or not np.isfinite([value['stat_obs'],*scores]).all():
                    raise RuntimeError('Budget/nonfinite scores')
                tail=sum(score>=value['stat_obs'] for score in scores)
                pv=(1+tail)/(p['B']+1)
                if tail!=value['tail_count'] or pv!=value['pvalue'] or value['reject']!=(pv<=p['alpha']):
                    raise RuntimeError('Raw tail/decision mismatch')
                del value['permutation_statistics']
            if replay and (meta['stage']=='preflight' or s['id'] not in replayed):
                x,y=make_data(p,s,rep)
                if arrays_hash(x,y)!=row['data_hash']: raise RuntimeError('Input replay mismatch')
                replayed.add(s['id'])
            result.append(row)
    if seen!=expected: raise RuntimeError('Incomplete archive coverage')
    if replay:
        receipt=dict(status='pass',original_worker_input_replay=True,platform=platform.platform(),
                     cpu_model=cpu_model(),python=sys.version,raw_sha256=meta['raw_sha256'],
                     replayed_blocks=len(result) if meta['stage']=='preflight' else len(replayed))
        if receipt['platform']!=meta['platform'] or receipt['cpu_model']!=meta['cpu_model'] or receipt['python']!=meta['python']:
            raise RuntimeError('Replay not on original worker cohort')
        (path/'input_replay_verification.json').write_text(json.dumps(receipt,indent=2))
    return meta,result


def analyze(paths,p):
    rows=[];seen=set();cohorts={};baseline_sources=None
    for path in paths:
        meta,batch=audit(path,p)
        if meta['stage']!='decision': raise RuntimeError('Pilot cannot become evidence')
        receipt=json.loads((path/'input_replay_verification.json').read_text())
        if receipt['raw_sha256']!=meta['raw_sha256'] or not receipt.get('original_worker_input_replay'):
            raise RuntimeError('Missing original-worker replay receipt')
        if baseline_sources is None: baseline_sources=meta['source_hashes']
        if meta['source_hashes']!=baseline_sources: raise RuntimeError('Different source snapshots')
        for row in batch:
            key=(row['stratum']['id'],row['rep'])
            if key in seen: raise RuntimeError('Overlapping shards')
            seen.add(key)
            study=row['stratum']['study'];cohort=(meta['cpu_model'],meta['python'],meta['numpy'],meta['scipy'])
            if study in cohorts and cohorts[study]!=cohort: raise RuntimeError('Within-study cohort mismatch')
            cohorts[study]=cohort
            row['platform']=meta['platform'];row['cpu_model']=meta['cpu_model'];rows.append(row)
    expected={(s['id'],r) for s in strata(p) for r in range(p['null_repetitions'] if s['null'] else p['power_repetitions'])}
    if seen!=expected: raise RuntimeError('Incomplete protected campaign; no survivor rates')
    groups=collections.defaultdict(list)
    for row in rows: groups[row['stratum']['id']].append(row)
    rates=[];nulls={};raw_over={}
    for s in strata(p):
        group=sorted(groups[s['id']],key=lambda r:r['rep']);n=len(group)
        for method in p[s['study']]['methods']:
            k=sum(r['results'][method]['reject'] for r in group)
            record=dict(stratum=s,method=method,rejections=k,repetitions=n,rate=k/n,
                        interval95=interval(k,n),mean_call_seconds=float(np.mean([r['results'][method]['call_seconds'] for r in group])),
                        timing_scope='shared-field diagnostic only' if s['study']=='gaussian' else 'measured method components plus charged common allocation cost; not independent standalone latency')
            rates.append(record)
            if s['null']:
                key=f"{s['id']}:{method}"
                upper=float(beta.ppf(1-p['analysis']['null_family_alpha']/p['analysis']['null_family_size'],k+1,n-k)) if k<n else 1.
                record.update(simultaneous_null_upper=upper,over_rejection_p=float(binomtest(k,n,p['alpha'],alternative='greater').pvalue))
                nulls[key]=record;raw_over[key]=record['over_rejection_p']
    if len(nulls)!=p['analysis']['null_family_size']: raise RuntimeError('Wrong null family')
    for key,record in nulls.items():
        record['bonferroni_over_rejection_p']=min(1.,raw_over[key]*p['analysis']['null_family_size'])
        record['null_resolved']=record['simultaneous_null_upper']<=p['alpha']+p['analysis']['null_upper_margin'] and record['bonferroni_over_rejection_p']>=p['analysis']['null_family_alpha']
    contrasts=[];raw_ps={}
    for s in strata(p):
        if s['null']:continue
        group=sorted(groups[s['id']],key=lambda r:r['rep']);a=p[s['study']]['methods'][0]
        for b in p[s['study']]['methods'][1:]:
            win=sum(r['results'][a]['reject'] and not r['results'][b]['reject'] for r in group)
            lose=sum(r['results'][b]['reject'] and not r['results'][a]['reject'] for r in group)
            key=f"{s['id']}:{a}:{b}";pv=float(binomtest(win,win+lose,.5).pvalue) if win+lose else 1.
            matching=[t for t in strata(p) if t['null'] and t['study']==s['study'] and t['d']==s['d'] and t['n']==s['n']]
            eligible=all(nulls[f"{t['id']}:{method}"]['null_resolved'] for t in matching for method in (a,b))
            delta=(win-lose)/len(group)
            se=float(np.sqrt(max(0.,((win+lose)/len(group)-delta**2)/len(group))))
            contrasts.append(dict(id=key,stratum=s,method_a=a,method_b=b,a_only=win,b_only=lose,
                repetitions=len(group),difference=delta,paired_se=se,pointwise_normal_interval95=[delta-1.96*se,delta+1.96*se],
                pvalue=pv,matching_null_ids=[t['id'] for t in matching],calibration_eligible=eligible))
            raw_ps[key]=pv
    if len(contrasts)!=p['analysis']['primary_power_family_size']:raise RuntimeError('Wrong power family')
    adj=holm(raw_ps,p['analysis']['primary_power_family_size'])
    for record in contrasts:
        record['holm_pvalue']=adj[record['id']]
        record['qualified_difference']=record['calibration_eligible'] and record['holm_pvalue']<=p['analysis']['power_family_alpha']
    secondary=[]
    for (sid,platform_name),group in itertools_group(rows).items():
        s=group[0]['stratum']
        for method in p[s['study']]['methods']:
            secondary.append(dict(stratum=s,platform=platform_name,cpu_model=group[0]['cpu_model'],
                method=method,repetitions=len(group),rate=sum(v['results'][method]['reject'] for v in group)/len(group),
                mean_call_seconds=float(np.mean([v['results'][method]['call_seconds'] for v in group])),
                scope='secondary platform-stratified descriptive; not a replacement primary cohort'))
    return dict(protocol=p,protocol_hash=digest(p),status='complete',blocks=len(rows),
                method_results=sum(len(r['results']) for r in rows),cohorts=cohorts,rates=rates,contrasts=contrasts,platform_stratified_rates=secondary,
                null_resolved=sum(v['null_resolved'] for v in nulls.values()),null_screens=len(nulls),
                qualified_contrasts=sum(v['qualified_difference'] for v in contrasts))


def itertools_group(rows):
    groups=collections.defaultdict(list)
    for row in rows:groups[row['stratum']['id'],row['platform']].append(row)
    return groups


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('paths',type=Path,nargs='+');ap.add_argument('--out',type=Path)
    ap.add_argument('--audit-only',action='store_true');ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    p=json.loads((HERE/'protocol.json').read_text())
    if args.audit_only:
        for path in args.paths:
            meta,rows=audit(path,p,args.replay)
            print(json.dumps(dict(status='pass',path=str(path),blocks=len(rows),stage=meta['stage'])))
    else:
        result=analyze(args.paths,p)
        if args.out is None:raise ValueError('--out required')
        args.out.mkdir(parents=True,exist_ok=False)
        (args.out/'analysis.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:result[k] for k in ('status','blocks','method_results','null_resolved','null_screens','qualified_contrasts')}))
