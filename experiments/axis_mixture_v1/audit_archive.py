"""Original-worker archive/input replay; review summaries are never power evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
if not __debug__:
    raise RuntimeError('Audits must run without Python optimization')
from baselines.conditional_c2st import prediction_hypergeometric_pvalue

spec=importlib.util.spec_from_file_location('axis',Path(__file__).with_name('runner.py'))
axis=importlib.util.module_from_spec(spec);spec.loader.exec_module(axis)


def audit(out):
    meta=json.loads((out/'metadata.json').read_text());p=meta['protocol']
    assert meta['protocol_hash']==axis.digest(p)
    assert meta['source_hashes']==axis.source_hashes(), 'Source changed since producing rows'
    raw=(out/'raw.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==meta['raw_sha256']
    assert hashlib.sha256((out/'source.tar.gz').read_bytes()).hexdigest()==meta['source_archive_sha256']
    rows=[json.loads(line) for line in raw.splitlines()]
    args=meta['arguments']
    dimensions=args['dimensions'] or p['dimensions']
    reps=range(p['pilot_rep_start'],p['pilot_rep_start']+args['pilot_reps']) if meta['stage']=='preflight' else range(args['rep_start'],args['rep_stop'])
    wanted={(case['id'],d,rep,method) for d in dimensions for case in p['cases'] for rep in reps
            if meta['stage']=='preflight' or rep<(p['null_repetitions'] if case['null'] else p['power_repetitions'])
            for method in p['methods']}
    keys=[(r['case'],r['d'],r['rep'],r['method']) for r in rows]
    assert len(keys)==len(set(keys))==len(wanted)==meta['expected_calls']==meta['written_calls']
    assert set(keys)==wanted and not meta['failed_calls'] and meta['status']=='complete'
    replay={}
    for row in rows:
        assert row['status']=='OK' and not row['warnings']
        case=next(c for c in p['cases'] if c['id']==row['case'])
        assert row['null']==case['null'] and row['protocol_hash']==meta['protocol_hash']
        d,rep=row['d'],row['rep'];key=(case['id'],d,rep)
        if key not in replay:
            raw_data,_=axis.make_data(p,case,d,rep);arrays,scale,center=axis.prepare(raw_data)
            replay[key]=(axis.arrays_hash(*raw_data),axis.arrays_hash(*arrays[:2]),axis.arrays_hash(*arrays[2:]),scale,center)
        assert (row['base_data_hash'],row['training_hash'],row['evaluation_hash'],row['median_scale'],row['center_hash'])==replay[key]
        assert row['data_seed']==axis.seed_for(p,case,d,rep,1)
        name=row['method']
        domain=10 if name.startswith('Rank') else 11 if name.startswith('SCF_Gaussian') else 12 if name.startswith('MMD') else 13 if name.startswith('C2ST') else 14 if name.startswith('ME_') else 15
        assert row['method_seed']==axis.seed_for(p,case,d,rep,domain)
        assert row['reject']==(row['pvalue']<=p['alpha']) and 0<=row['pvalue']<=1
        assert np.isfinite([row['stat_obs'],row['pvalue'],row['total_seconds']]).all()
        assert row['perms_used']==(0 if name.startswith('C2ST') else p['evaluation_permutations'])
        if name.startswith('Rank'):
            values=row['permutation_statistics']
            assert len(values)==p['evaluation_permutations']
            assert row['pvalue']==(1+sum(v>=row['stat_obs'] for v in values))/(len(values)+1)
        if name.startswith('C2ST'):
            c=row['c2st_confusion'];n=c['n'];n1=c['true_ones'];k=c['predicted_ones'];u=c['correct_ones']
            # Construct one equivalent confusion table independently for exact-tail replay.
            true=np.r_[np.ones(n1,int),np.zeros(n-n1,int)]
            predicted=np.r_[np.ones(u,int),np.zeros(n1-u,int),np.ones(k-u,int),np.zeros(n-n1-k+u,int)]
            assert prediction_hypergeometric_pvalue(true,predicted)==row['pvalue']
        if name.endswith('TrainGrid'):
            assert len(row['candidate_trace'])==len(p['scale_multipliers'])
            assert all(v['perms_used']==p['training_permutations_per_candidate'] for v in row['candidate_trace'])
    return {'status':'pass','calls':len(rows),'complete_blocks':len(replay),
            'input_replay':True,'source_and_raw_hashes':True,'stage':meta['stage'],
            'metadata_sha256':hashlib.sha256((out/'metadata.json').read_bytes()).hexdigest(),
            'raw_sha256':meta['raw_sha256'],'source_archive_sha256':meta['source_archive_sha256']}


if __name__=='__main__':
    out=Path(sys.argv[1]);result=audit(out)
    (out/'input_replay_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
