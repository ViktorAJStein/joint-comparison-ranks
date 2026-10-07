"""Protected complete-stratum null screens / paired generator power analysis."""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta,binom,norm
from audit_archive import audit
from runner import KINDS
from scripts.fair_comparator_bench import digest
HERE=Path(__file__).resolve().parent

def interval(k,n):
    return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k))]

def holm(values,family):
    ordered=sorted((v,key) for key,v in values.items() if v is not None)
    result={key:None for key in values};running=0.
    for rank,(v,key) in enumerate(ordered):
        running=max(running,min(1.,v*(family-rank)))
        result[key]=running
    return result

def analyze(paths):
    p=json.loads((HERE/'protocol.json').read_text());ph=digest(p)
    rows={};metadata=[];runtime_cohorts=set()
    for path in paths:
        path=Path(path);meta=json.loads((path/'metadata.json').read_text())
        if meta['stage']!='decision' or meta['protocol_hash']!=ph:raise RuntimeError('Only exact decision protocol archives')
        if meta['source_commit']!=p['base_source_commit']:raise RuntimeError('Source commit')
        receipt=json.loads((path/'input_replay_verification.json').read_text())
        if not receipt['original_worker_input_replay'] or receipt['status']!='pass':raise RuntimeError('Missing replay')
        if receipt['metadata_sha256']!=hashlib.sha256((path/'metadata.json').read_bytes()).hexdigest():raise RuntimeError('Stale replay')
        if receipt['raw_sha256']!=meta['raw_sha256']:raise RuntimeError('Unbound raw replay')
        audit(path,replay=False)
        sig=tuple(meta[k] for k in ('cpu_model','python','numpy','scipy','sklearn'))
        runtime_cohorts.add(sig)
        if len(runtime_cohorts)>1:raise RuntimeError('CPU/Python/library cohort mismatch')
        if meta['cpu_model']!=p['execution']['cpu_model']:raise RuntimeError('Wrong CPU')
        if not any(k in meta['platform'] for k in ('6.12.110-1.el8.x86_64','6.12.111-1.el8.x86_64')):raise RuntimeError('Unqualified platform')
        metadata.append({'path':str(path),'platform':meta['platform'],'node':meta['node'],'status':meta['status']})
        for line in (path/'raw.jsonl').open():
            row=json.loads(line);key=(row['case'],row['d'],row['orientation'],row['rep'])
            if key in rows:raise RuntimeError('Duplicate independent evidence')
            row['platform']=meta['platform'];rows[key]=row
    summary=[];null={};strata={};pairs=[];kernel=[]
    for case in p['cases']:
        n=p['null_repetitions'] if case['null'] else p['power_repetitions']
        for d in p['dimensions']:
            for orientation in case['orientations']:
                rs=[rows.get((case['id'],d,orientation,r)) for r in range(n)]
                complete=all(r is not None and r['status']=='OK' and not r['warnings'] for r in rs)
                strata[case['id'],d,orientation]=(rs,complete)
                for kind in KINDS:
                    row={'case':case['id'],'d':d,'orientation':orientation,'generator':kind,'null':case['null'],
                         'expected_blocks':n,'complete':complete,'rejections':None,'rate':None,'ci95':None,
                         'null_adjusted_pvalue':None,'simultaneous_null_upper':None,'size_status':None,
                         'standalone_timing_count':0,'standalone_mean_seconds':None,'standalone_median_seconds':None}
                    if complete:
                        k=sum(r['results'][kind]['reject'] for r in rs)
                        row.update(rejections=k,rate=k/n,ci95=interval(k,n))
                        ts=[r['results'][kind]['standalone_seconds'] for r in rs if 'standalone_seconds' in r['results'][kind]]
                        if len(ts)!=p['standalone_timing_repetitions']:raise RuntimeError('Incomplete fixed timing sample')
                        row.update(standalone_timing_count=len(ts),standalone_mean_seconds=float(np.mean(ts)),standalone_median_seconds=float(np.median(ts)))
                        if case['null']:
                            adj=min(1.,float(binom.sf(k-1,n,p['alpha']))*p['analysis']['null_family_size'])
                            upper=1. if k==n else float(beta.ppf(1-p['analysis']['null_family_alpha']/p['analysis']['null_family_size'],k+1,n-k))
                            status='overreject' if adj<=p['analysis']['null_family_alpha'] else 'inconclusive' if upper>p['alpha']+p['analysis']['null_upper_margin'] else 'resolved'
                            row.update(null_adjusted_pvalue=adj,simultaneous_null_upper=upper,size_status=status)
                    summary.append(row)
                    if case['null']:null[case['id'],d,orientation,kind]=row['size_status']
                # Secondary kernel strata do not replace primary full-group denominators.
                if complete:
                    for platform_name in sorted({r['platform'] for r in rs}):
                        kr=[r for r in rs if r['platform']==platform_name]
                        for kind in KINDS:
                            ts=[r['results'][kind]['standalone_seconds'] for r in kr if 'standalone_seconds' in r['results'][kind]]
                            kernel.append({'case':case['id'],'d':d,'orientation':orientation,'platform':platform_name,
                                           'generator':kind,'blocks':len(kr),'rate':sum(r['results'][kind]['reject'] for r in kr)/len(kr),
                                           'timing_blocks':len(ts),'standalone_mean_seconds':float(np.mean(ts)) if ts else None,
                                           'scope':'secondary stratification, not substitute primary evidence'})
    for case in (c for c in p['cases'] if not c['null']):
        for d in p['dimensions']:
            for orientation in case['orientations']:
                rs,complete=strata[case['id'],d,orientation];n=p['power_repetitions']
                for a,b in itertools.combinations(KINDS,2):
                    calibrated=null.get((case['matched_null'],d,orientation,a))=='resolved' and null.get((case['matched_null'],d,orientation,b))=='resolved'
                    row={'case':case['id'],'d':d,'orientation':orientation,'a':a,'b':b,'complete':complete,
                         'matching_nulls_resolved':calibrated,'a_only':None,'b_only':None,'delta_a_minus_b':None,
                         'pointwise_normal_ci95':None,'exact_mcnemar_pvalue':None,'holm_pvalue':None,
                         'supported_calibrated_difference':False}
                    if complete:
                        av=np.array([r['results'][a]['reject'] for r in rs]);bv=np.array([r['results'][b]['reject'] for r in rs])
                        x=int(np.sum(av&~bv));y=int(np.sum(bv&~av));discord=x+y
                        pv=1. if discord==0 else min(1.,2*float(binom.sf(max(x,y)-1,discord,.5)))
                        delta=(x-y)/n;se=np.sqrt((discord-(x-y)**2/n)/(n*(n-1)))
                        ci=[max(-1.,delta-norm.ppf(.975)*se),min(1.,delta+norm.ppf(.975)*se)]
                        row.update(a_only=x,b_only=y,delta_a_minus_b=delta,pointwise_normal_ci95=ci,exact_mcnemar_pvalue=pv)
                    pairs.append(row)
    if len(pairs)!=48 or len([s for s in summary if s['null']])!=32:raise RuntimeError('Protected families changed')
    adjusted=holm({i:r['exact_mcnemar_pvalue'] for i,r in enumerate(pairs)},p['analysis']['primary_power_family_size'])
    for i,row in enumerate(pairs):
        row['holm_pvalue']=adjusted[i]
        row['supported_calibrated_difference']=row['matching_nulls_resolved'] and adjusted[i] is not None and adjusted[i]<=p['analysis']['power_family_alpha']
    return {'protocol_hash':ph,'summary':summary,'paired_contrasts':pairs,'kernel_strata':kernel,
            'archives':metadata,'complete_primary':all(r['complete'] for r in summary),'row_groups':len(rows),
            'no_adaptive_generator_selection_claim':True}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archives',nargs='+',required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();result=analyze(a.archives);a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'analysis.json').write_text(json.dumps(result,indent=2))
    for key in ('summary','paired_contrasts','kernel_strata'):
        with (a.out/(key+'.csv')).open('w') as f:
            wr=csv.DictWriter(f,fieldnames=list(result[key][0]));wr.writeheader();wr.writerows(result[key])
    print(json.dumps({'groups':result['row_groups'],'complete':result['complete_primary'],'null_screens':32,'paired_contrasts':48}))
