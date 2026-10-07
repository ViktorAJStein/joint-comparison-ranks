"""Separate prospective axis-mixture comparison; frozen older studies stay untouched."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
import warnings

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_info
from baselines.me_scf import me_permutation_test, scf_permutation_test
from scripts.fair_comparator_bench import (arrays_hash, cpu_model, digest, evaluate,
                                          make_data, prepare, seed_for)
from scripts.run_power_bench import time_limit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = Path(os.environ.get('AXIS_BASE_SOURCE', ROOT / 'joint-comparison-ranks')).resolve()
BATCH_PATH = Path(os.environ.get('AXIS_BATCH_RUNNER', HERE.parent / 'generator_comparison_v1/runner.py')).resolve()
spec = importlib.util.spec_from_file_location('qualified_generator_batch', BATCH_PATH)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def sources():
    files = sorted((BASE / 'src').rglob('*.py')) + [BASE / 'pyproject.toml', BASE / 'uv.lock']
    files += sorted(HERE.glob('*.py')) + [HERE / 'protocol.json', BATCH_PATH]
    return {str(f.relative_to(BASE)) if f.is_relative_to(BASE) else
            ('batch/runner.py' if f == BATCH_PATH else 'axis/' + f.name): f for f in files}


def source_hashes():
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources().items()}


def admit(p, args):
    commit = subprocess.check_output(['git', '-C', str(BASE), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != p['base_source_commit']:
        raise RuntimeError('Frozen base changed')
    if subprocess.check_output(['git', '-C', str(BASE), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Frozen base dirty')
    if any(info['num_threads'] != 1 for info in threadpool_info()):
        raise RuntimeError('Only one numerical thread allowed')
    if args.dimensions and not set(args.dimensions) <= set(p['dimensions']):
        raise ValueError('Unprotected dimension')
    if not 1 <= args.pilot_reps <= 2:
        raise ValueError('Preflight is at most two diagnostic repetitions')
    if args.stage == 'decision':
        if not args.authorization:
            raise PermissionError('Main campaign not budget-authorized')
        approval = json.loads(args.authorization.read_text())
        required = (approval.get('approved_by_user') is True,
                    approval.get('protocol_hash') == digest(p),
                    approval.get('source_hashes') == source_hashes(),
                    approval.get('source_commit') == commit,
                    approval.get('cohort') == args.cohort,
                    approval.get('cpu_model') == cpu_model(),
                    approval.get('kernel_releases') and platform.release() in approval['kernel_releases'],
                    0 < approval.get('maximum_total_cpu_core_hours', 0) < float('inf'),
                    0 < approval.get('reserved_cpu_core_hours', 0) <=
                    approval.get('maximum_total_cpu_core_hours', 0),
                    args.rep_start >= 0 and args.rep_stop > args.rep_start,
                    tuple(args.dimensions or p['dimensions']) == tuple(p['cohort_dimensions'].get(args.cohort, [])))
        if not all(required):
            raise PermissionError('Failed protocol/source/cohort/budget admission')
        actual = {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                  'sklearn': sklearn.__version__}
        if approval.get('software') != actual:
            raise PermissionError('Unqualified numerical environment')
    return commit


def call_method(name, arrays, scale, seed, p, row):
    x, y = arrays[2:]
    if name == 'Rank_Pairs_Conservative':
        started = time.perf_counter()
        values, common = batch.joint_test(x, y, seed, p, kinds=('gtest',))
        value = values['gtest']
        row.update(stat_obs=value['stat_obs'], pvalue=value['pvalue'], reject=value['reject'],
                   perms_used=value['perms_used'], permutation_statistics=value['permutation_statistics'],
                   rank_common=common, training_seconds=0.0,
                   evaluation_seconds=time.perf_counter()-started)
    elif name in ('ME_Archived_FixedBudget', 'SCF_Unsmooth_Archived_FixedBudget'):
        started = time.perf_counter()
        func = me_permutation_test if name.startswith('ME_') else scf_permutation_test
        value = func(x, y, **p['legacy_features'], B=p['evaluation_permutations'],
                     alpha=p['alpha'], early_stop='none', seed=seed)
        row.update(stat_obs=value.stat_obs, pvalue=value.p_perm, reject=value.reject,
                   perms_used=value.perms_used, selected_scale=value.sigma,
                   training_seconds=0.0, evaluation_seconds=time.perf_counter()-started)
    else:
        evaluate(name, arrays, scale, seed, p, row)
    if row['perms_used'] != (0 if name.startswith('C2ST') else p['evaluation_permutations']):
        raise RuntimeError('Incomplete permutation budget')
    if not np.isfinite([row['stat_obs'], row['pvalue']]).all():
        raise FloatingPointError('Nonfinite score/p-value')
    if not 0 <= row['pvalue'] <= 1 or row['reject'] != (row['pvalue'] <= p['alpha']):
        raise RuntimeError('Invalid decision')


def run_block(p, case, d, rep, emit):
    before = time.perf_counter()
    raw, _ = make_data(p, case, d, rep)
    arrays, scale, center_hash = prepare(raw)
    prep_seconds = time.perf_counter() - before
    common = {'case': case['id'], 'null': case['null'], 'd': d, 'rep': rep,
              'orientation': 'raw', 'data_seed': seed_for(p, case, d, rep, 1),
              'base_data_hash': arrays_hash(*raw), 'evaluation_hash': arrays_hash(*arrays[2:]),
              'training_hash': arrays_hash(*arrays[:2]), 'center_hash': center_hash,
              'median_scale': scale, 'common_prep_seconds': prep_seconds,
              'protocol_hash': digest(p)}
    s = case['params']['sparse_dims']
    a = min(case['params']['delta'], .98*np.sqrt(s))
    common.update(sparse_dims=s, requested_delta=case['params']['delta'],
                  effective_delta=float(a), active_variance=float(1-a*a/s))
    order = np.random.default_rng(seed_for(p, case, d, rep, 3)).permutation(len(p['methods']))
    for position, index in enumerate(order):
        name = p['methods'][index]
        # Domains stay distinct; no collisions between bridge and modern features.
        domain = 10 if name.startswith('Rank') else 11 if name.startswith('SCF_Gaussian') else 12 if name.startswith('MMD') else 13 if name.startswith('C2ST') else 14 if name.startswith('ME_') else 15
        row = dict(common, method=name, method_seed=seed_for(p, case, d, rep, domain),
                   order_position=position, status='OK', candidate_trace=[],
                   selected_scale=None, training_seconds=None, evaluation_seconds=None,
                   stat_obs=None, pvalue=None, reject=None, perms_used=None)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                with time_limit(p['method_timeout_seconds']):
                    call_method(name, arrays, scale, row['method_seed'], p, row)
                if arrays_hash(*arrays[2:]) != common['evaluation_hash'] or arrays_hash(*arrays[:2]) != common['training_hash']:
                    raise RuntimeError('Input mutation')
            except Exception as error:
                row.update(status=f'ERROR:{type(error).__name__}:{error}', reject=None)
            row['warnings'] = [str(v.message) for v in caught]
        row.update(call_seconds=time.perf_counter()-start)
        row['total_seconds'] = prep_seconds + row['call_seconds']
        emit(row)
    return time.perf_counter()-before


def run(p, args):
    commit = admit(p, args)  # Refuse unapproved main runs before creating an output.
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = source_hashes()
    with tarfile.open(args.out / 'source.tar.gz', 'w:gz') as archive:
        for name, path in sources().items():
            archive.add(path, arcname=name)
    dimensions = args.dimensions or p['dimensions']
    reps = range(p['pilot_rep_start'], p['pilot_rep_start']+args.pilot_reps) if args.stage == 'preflight' else range(args.rep_start, args.rep_stop)
    tasks = [(case, d, rep) for d in dimensions for case in p['cases'] for rep in reps
             if args.stage == 'preflight' or rep < (p['null_repetitions'] if case['null'] else p['power_repetitions'])]
    meta = {'protocol': p, 'protocol_hash': digest(p), 'source_commit': commit,
            'source_hashes': receipt, 'stage': args.stage, 'cohort': args.cohort,
            'cpu_model': cpu_model(), 'platform': platform.platform(), 'kernel_release': platform.release(),
            'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
            'sklearn': sklearn.__version__, 'threadpools': threadpool_info(),
            'slurm_job': os.environ.get('SLURM_JOB_ID'), 'status': 'running',
            'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'expected_calls': len(tasks)*len(p['methods']), 'written_calls': 0, 'failed_calls': 0,
            'blocks': []}
    began = time.perf_counter()
    with (args.out / 'raw.jsonl').open('x') as stream:
        def emit(row):
            meta['written_calls'] += 1
            meta['failed_calls'] += row['status'] != 'OK'
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        try:
            for case, d, rep in tasks:
                seconds = run_block(p, case, d, rep, emit)
                meta['blocks'].append({'case': case['id'], 'd': d, 'rep': rep, 'seconds': seconds})
        finally:
            meta.update(elapsed_seconds=time.perf_counter()-began,
                        status='complete' if meta['written_calls'] == meta['expected_calls'] and not meta['failed_calls'] else 'incomplete_or_failed',
                        raw_sha256=hashlib.sha256((args.out/'raw.jsonl').read_bytes()).hexdigest(),
                        source_archive_sha256=hashlib.sha256((args.out/'source.tar.gz').read_bytes()).hexdigest())
            (args.out/'metadata.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ('status', 'written_calls', 'failed_calls', 'elapsed_seconds')}))
    if meta['status'] != 'complete':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('preflight', 'decision'), default='preflight')
    parser.add_argument('--pilot-reps', type=int, default=2)
    parser.add_argument('--rep-start', type=int, default=0)
    parser.add_argument('--rep-stop', type=int, default=0)
    parser.add_argument('--dimensions', type=int, nargs='+')
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--authorization', type=Path)
    args = parser.parse_args()
    run(json.loads((HERE/'protocol.json').read_text()), args)


if __name__ == '__main__':
    main()
