"""Portable execution of frozen paper functions, separate from private admissions."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
# Set before importing any numerical library; enforce after import too.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
BASE = ROOT / 'snapshots/comparator'
os.environ['AXIS_BASE_SOURCE'] = str(BASE)
os.environ['AXIS_BATCH_RUNNER'] = str(ROOT/'experiments/generator_comparison_v1/runner.py')
sys.path.insert(0, str(BASE/'src'))
import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_info
from scripts import fair_comparator_bench as common


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def setup(study):
    if study == 'comparator':
        return common, json.loads((BASE/'experiments/fair_comparators_v1.json').read_text())
    directory = ROOT/'experiments'/f'{study if study != "generator" else "generator_comparison"}_v1'
    if study == 'axis': directory = ROOT/'experiments/axis_mixture_v1'
    if study == 'dependence': directory = ROOT/'experiments/dependence_power_v1'
    return load(f'paper_{study}', directory/'runner.py'), json.loads((directory/'protocol.json').read_text())


def tasks(study, mod, p, mode, start, stop):
    if study == 'dependence':
        for s in mod.strata(p):
            count = p['null_repetitions' if s['null'] else 'power_repetitions']
            reps = [p['pilot_rep_start']] if mode == 'smoke' else range(start, min(count, stop or count))
            for rep in reps: yield s, s['d'], rep
    else:
        for d in p['dimensions']:
            for case in p['cases']:
                count = p['null_repetitions' if case['null'] else 'power_repetitions']
                reps = [p['pilot_rep_start']] if mode == 'smoke' else range(start, min(count, stop or count))
                for rep in reps: yield case, d, rep


def run_generator(mod, p, case, d, rep, emit, smoke):
    raw, rotation = common.make_data(p, case, d, rep)
    for orientation in case['orientations']:
        arrays, _, _ = common.prepare(raw if orientation == 'raw' else tuple(a@rotation for a in raw))
        x, y = arrays[2:]
        seed = common.seed_for(p, case, d, rep, 10)
        results, fields = mod.joint_test(x, y, seed, p)
        if smoke or rep < p['standalone_timing_repetitions']:
            timings = mod.standalone_timings(x, y, seed, p, rep)
            for kind, value in timings.items():
                assert all(value[k] == results[kind][k] for k in ('stat_obs', 'pvalue'))
                assert value['rank_tables_hash'] == fields['rank_tables_hash']
                results[kind].update(value)
        emit(dict(case=case['id'], null=case['null'], matched_null=case.get('matched_null'),
                  d=d, rep=rep, orientation=orientation,
                  data_seed=common.seed_for(p,case,d,rep,1),
                  rotation_seed=common.seed_for(p,case,d,rep,2), method_seed=seed,
                  base_data_hash=common.arrays_hash(*raw), evaluation_hash=common.arrays_hash(x,y),
                  protocol_hash=common.digest(p), status='OK', warnings=[], common=fields, results=results))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study', choices=('calibration','comparator','generator','axis','dependence'), required=True)
    ap.add_argument('--mode', choices=('smoke','full'), default='smoke')
    ap.add_argument('--confirm-full', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rep-start', type=int, default=0)
    ap.add_argument('--rep-stop', type=int)
    ap.add_argument('--wall-seconds', type=int, default=600)
    args = ap.parse_args()
    if args.mode == 'full' and not args.confirm_full: ap.error('Full study needs --confirm-full')
    if args.rep_start < 0 or args.rep_stop is not None and args.rep_stop <= args.rep_start: ap.error('Invalid range')
    if args.wall_seconds <= 0: ap.error('Positive wall cap required')
    if args.out.exists(): ap.error('Output exists; never overwrite evidence')
    if args.study == 'calibration':
        if args.rep_start or args.rep_stop is not None: ap.error('Calibration supports complete range or smoke only')
        command = [sys.executable, str(ROOT/'snapshots/calibration/scripts/run_calibration_audit.py'),
                   '--out',str(args.out),'--reps', '8000' if args.mode=='full' else '1',
                   '--power-reps','500' if args.mode=='full' else '1','--seed','6060',
                   '--workers','4' if args.mode=='full' else '1','--cohort','portable-rerun']
        # Own process group so a timeout terminates child workers as well.
        proc = subprocess.Popen(command, env=os.environ, start_new_session=True)
        try: code = proc.wait(timeout=args.wall_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            raise SystemExit('Calibration wall cap reached; incomplete output retained')
        raise SystemExit(code)
    if any(v['num_threads'] != 1 for v in threadpool_info()): raise RuntimeError('Multiple numerical threads')
    mod, p = setup(args.study)
    todo = list(tasks(args.study, mod, p, args.mode, args.rep_start, args.rep_stop))
    if not todo: raise ValueError('Empty shard')
    args.out.mkdir(parents=True)
    def timeout(signum, frame): raise TimeoutError('Public rerun wall cap reached')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(args.wall_seconds)
    meta = dict(study=args.study, mode=args.mode, status='running', protocol=p,
                protocol_hash=common.digest(p), cpu_model=common.cpu_model(), platform=platform.platform(),
                python=sys.version, numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                threadpools=threadpool_info(), expected_blocks=len(todo), completed_blocks=0, written_rows=0,
                provenance_scope='portable rerun; not original author worker admission',
                source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in ROOT.rglob('*.py') if not any(v in f.parts for v in ('.venv','runs','.git'))})
    began=time.perf_counter()
    try:
        with (args.out/'raw.jsonl').open('x') as stream:
            def emit(row):
                if row['status'] != 'OK' or row.get('warnings'): raise RuntimeError(f'Failed/warning row: {row}')
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush(); meta['written_rows']+=1
            for case,d,rep in todo:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    if args.study=='comparator': mod.run_block(p,case,d,rep,emit,'portable-rerun')
                    elif args.study=='axis': mod.run_block(p,case,d,rep,emit)
                    elif args.study=='generator': run_generator(mod,p,case,d,rep,emit,args.mode=='smoke')
                    else:
                        before=time.perf_counter();x,y=mod.make_data(p,case,rep)
                        row=dict(stratum=case,rep=rep,data_seed=mod.seed_for(p,case,rep,1),
                                 method_seed=mod.seed_for(p,case,rep,10),status='OK',warnings=[],
                                 protocol_hash=common.digest(p), data_hash=common.arrays_hash(x,y),
                                 data_seconds=time.perf_counter()-before)
                        row['results'],row['common']=mod.evaluate(p,case,x,y,row['method_seed'])
                        assert common.arrays_hash(x,y)==row['data_hash']
                        row['block_seconds']=time.perf_counter()-before;emit(row)
                    if caught: raise RuntimeError('Numerical warnings: '+str([str(w.message) for w in caught]))
                meta['completed_blocks']+=1
        meta['status']='complete'
    finally:
        signal.alarm(0)
        if meta['status']!='complete': meta['status']='incomplete_or_failed'
        meta.update(elapsed_seconds=time.perf_counter()-began,
                    raw_sha256=hashlib.sha256((args.out/'raw.jsonl').read_bytes()).hexdigest())
        (args.out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({k:meta[k] for k in ('study','mode','status','completed_blocks','written_rows')}))


if __name__=='__main__': main()
