"""Original-worker replay and full archived statistic-tail audit."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scripts.fair_comparator_bench import make_data,prepare,arrays_hash,cpu_model,digest,seed_for
from runner import KINDS

def audit(out,replay=True):
    out=Path(out);meta=json.loads((out/'metadata.json').read_text());p=meta['protocol']
    raw=out/'raw.jsonl'
    if hashlib.sha256(raw.read_bytes()).hexdigest()!=meta['raw_sha256']:raise RuntimeError('Raw hash')
    if meta['protocol_hash']!=digest(p):raise RuntimeError('Protocol hash')
    arguments=meta['arguments']
    reps=range(p['pilot_rep_start'],p['pilot_rep_start']+arguments['pilot_reps']) if meta['stage']=='preflight' else range(arguments['rep_start'],arguments['rep_stop'])
    dimensions=arguments['dimensions'] or p['dimensions']
    planned={(c['id'],d,r,o) for d in dimensions for c in p['cases'] for r in reps
             if meta['stage']=='preflight' or r<(p['null_repetitions'] if c['null'] else p['power_repetitions']) for o in c['orientations']}
    if meta['expected_groups']!=len(planned):raise RuntimeError('Expected group enumeration')
    seen=set();groups=0;failed=0;cache={}
    for line in raw.open():
        row=json.loads(line);key=(row['case'],row['d'],row['rep'],row['orientation'])
        if key in seen:raise RuntimeError('Duplicate group')
        if key not in planned:raise RuntimeError('Unplanned group')
        seen.add(key);groups+=1
        case=next(c for c in p['cases'] if c['id']==row['case'])
        if row['null']!=case['null'] or row['matched_null']!=case.get('matched_null'):raise RuntimeError('Null role')
        for field,domain in (('data_seed',1),('rotation_seed',2),('method_seed',10)):
            if row[field]!=seed_for(p,case,row['d'],row['rep'],domain):raise RuntimeError('Wrong seed')
        if row['protocol_hash']!=meta['protocol_hash']:raise RuntimeError('Row protocol')
        if row['status']!='OK':failed+=1;continue
        if row['warnings']:raise RuntimeError('Warnings not qualified')
        if set(row['results'])!=set(KINDS):raise RuntimeError('Partial generator group')
        if row['common']['effective_L']!=64:raise RuntimeError('Wrong view budget')
        for kind,result in row['results'].items():
            values=np.array(result['permutation_statistics']);obs=result['stat_obs']
            if len(values)!=p['evaluation_permutations'] or result['perms_used']!=len(values):raise RuntimeError('Wrong B')
            if not np.isfinite(values).all() or not np.isfinite(obs):raise RuntimeError('Nonfinite')
            pv=float((1+np.sum(values>=obs))/(len(values)+1))
            if pv!=result['pvalue'] or result['reject']!=(pv<=p['alpha']):raise RuntimeError('Tail/decision mismatch')
        if replay:
            if cpu_model()!=meta['cpu_model']:raise RuntimeError('Replay must run on original CPU cohort')
            block=key[:3]
            if block not in cache:
                case=next(c for c in p['cases'] if c['id']==row['case'])
                cache.clear();cache[block]=make_data(p,case,row['d'],row['rep'])
            arrays,q=cache[block]
            if arrays_hash(*arrays)!=row['base_data_hash']:raise RuntimeError('Base input replay')
            oriented=arrays if row['orientation']=='raw' else tuple(a@q for a in arrays)
            prepared,_,_=prepare(oriented)
            if arrays_hash(*prepared[2:])!=row['evaluation_hash']:raise RuntimeError('Evaluation replay')
    if groups!=meta['groups_completed']+meta['groups_failed'] or failed!=meta['groups_failed']:raise RuntimeError('Counts')
    if meta['status']=='complete' and (failed or seen!=planned):raise RuntimeError('False completion')
    receipt={'status':'pass','raw_sha256':meta['raw_sha256'],'metadata_sha256':hashlib.sha256((out/'metadata.json').read_bytes()).hexdigest(),
             'groups_audited':groups,'failed_groups':failed,'original_worker_input_replay':replay,'cpu_model':meta['cpu_model'],
             'protocol_hash':meta['protocol_hash']}
    if replay:(out/'input_replay_verification.json').write_text(json.dumps(receipt,indent=2))
    return receipt
if __name__=='__main__':print(json.dumps(audit(Path(sys.argv[1]))))
