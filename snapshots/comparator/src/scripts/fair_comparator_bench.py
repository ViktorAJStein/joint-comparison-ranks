"""Prospective paired benchmark; no changes to historical runners/defaults.

Every method uses the same independent training/evaluation arrays. This is a
finite, training-grid comparison, not author-optimized SCF or exact-kernel MMD.
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
import tarfile
import time
import warnings
import numpy as np
import scipy
import sklearn
from scipy.spatial.distance import pdist
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_info
from baselines.analytic_scf import GaussianWindowSCF, gaussian_scf_features, paired_feature_statistic
from baselines.conditional_c2st import prediction_hypergeometric_pvalue
from baselines.rff_mmd import RFFMMD, feature_mmd_squared, rff_features
from ranktwosample.power_datasets import generate_pair
from ranktwosample.rank_two_sample import RankGTestConfig
from ranktwosample.rank_two_sample_subspaces import RankGTestSubspaces, SubspaceOpts
from scripts.run_power_bench import time_limit

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'experiments/fair_comparators_v1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def arrays_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a, dtype='<f8').tobytes())
    return h.hexdigest()


def seed_for(p, case, d, rep, domain):
    case_index = [c['id'] for c in p['cases']].index(case['id'])
    return int(np.random.SeedSequence([p['base_seed'], case_index, d, rep, domain]).generate_state(1)[0])


def cpu_model():
    if sys.platform == 'darwin':
        return subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    try:
        info = json.loads(subprocess.check_output(['lscpu', '-J'], text=True))
        return next(v['data'] for v in info['lscpu'] if v['field'] == 'Model name:')
    except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
        return platform.processor() or platform.machine()


def make_data(p, case, d, rep):
    seed = seed_for(p, case, d, rep, 1)
    rng = np.random.default_rng(seed)
    total = p['training_per_class'] + p['evaluation_per_class']
    if case['null']:
        _, pooled = generate_pair(case['family'], d, 0, 2*total, params=case['params'], rng=rng)
        x, y = pooled[:total], pooled[total:]
    else:
        x, y = generate_pair(case['family'], d, total, total, params=case['params'], rng=rng)
    if case['rounded']:
        x, y = np.round(x), np.round(y)
    n = p['training_per_class']
    arrays = (x[:n].copy(), y[:n].copy(), x[n:].copy(), y[n:].copy())
    q, r = np.linalg.qr(np.random.default_rng(seed_for(p, case, d, rep, 2)).normal(size=(d,d)))
    q *= np.where(np.diag(r) >= 0, 1., -1.)
    return arrays, q


def prepare(arrays):
    tx, ty, x, y = arrays
    center = np.vstack([tx, ty]).mean(axis=0)
    distances = pdist(np.vstack([tx, ty]))
    positive = distances[distances > 0]
    scale = float(np.median(positive)) if len(positive) else 1.
    prepared = tuple(np.ascontiguousarray(a-center) for a in arrays)
    for a in prepared: a.flags.writeable = False
    return prepared, scale, arrays_hash(center)


class AuditedRank(RankGTestSubspaces):
    def _compute_T_obs_sum_shared_pool(self, x, y, rng):
        result, env = super()._compute_T_obs_sum_shared_pool(x, y, rng)
        self.effective_L = len(env[8])
        self.views_hash = digest(env[8])
        self.reference_positions_hash = digest([v.tolist() for v in env[-1]])
        return result, env


def train_scale(kind, tx, ty, scale, seed, p, trace):
    """Training arrays ONLY. Candidate traces are appended before any test call."""
    if kind == 'SCF':
        frequencies = np.random.default_rng(seed).normal(size=(p['scf']['J'], tx.shape[1]))
        feature = lambda a, c: gaussian_scf_features(a, frequencies, c)
        statistic = lambda a, b: paired_feature_statistic(a, b, p['scf']['ridge'])
    else:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x524646]))
        frequencies = rng.normal(size=(p['rff']['features'], tx.shape[1]))
        phases = rng.uniform(0, 2*np.pi, p['rff']['features'])
        feature = lambda a, c: rff_features(a, frequencies/c, phases)
        statistic = feature_mmd_squared
    for multiplier in p['scale_multipliers']:
        candidate = scale*multiplier
        a, b = feature(tx, candidate), feature(ty, candidate)
        obs = statistic(a, b)
        pooled = np.vstack([a, b]); n = len(a)
        # Same training orders for every candidate; distinct from test shuffles.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x54524149]))
        scores = []
        for _ in range(p['training_permutations_per_candidate']):
            order = rng.permutation(len(pooled))
            scores.append(statistic(pooled[order[:n]], pooled[order[n:]]))
        mean, sd = float(np.mean(scores)), float(np.std(scores, ddof=1))
        z = (obs-mean)/max(sd, 1e-12)
        if not np.isfinite([obs, mean, sd, z]).all(): raise FloatingPointError('Nonfinite training score')
        trace.append({'scale': candidate, 'multiplier': multiplier, 'stat_obs': obs,
                      'permutation_mean': mean, 'permutation_sd': sd, 'score': z,
                      'perms_used': len(scores)})
    return trace[int(np.argmax([v['score'] for v in trace]))]['scale']


def evaluate(name, arrays, scale, seed, p, row):
    tx, ty, x, y = arrays
    started = time.perf_counter()
    training_start = started
    if name.endswith('TrainGrid'):
        scale = train_scale('SCF' if name.startswith('SCF') else 'RFF', tx, ty, scale, seed, p, row['candidate_trace'])
    elif name.startswith('C2ST'):
        if 'Logistic' in name:
            classifier = LogisticRegression(C=p['classifier']['C'], max_iter=p['classifier']['logistic_max_iter'],
                                            class_weight=p['classifier']['class_weight'], random_state=seed)
        else:
            classifier = SVC(C=p['classifier']['C'], gamma=p['classifier']['rbf_gamma'],
                             class_weight=p['classifier']['class_weight'])
        fitted = make_pipeline(StandardScaler(), classifier)
        fitted.fit(np.vstack([tx, ty]), np.r_[np.ones(len(tx), int), np.zeros(len(ty), int)])
    row['training_seconds'] = time.perf_counter()-training_start
    row['selected_scale'] = scale if name.startswith(('SCF', 'MMD')) else None
    evaluation_start = time.perf_counter()
    if name.startswith('Rank'):
        cfg = RankGTestConfig(alpha=p['alpha'], B=p['evaluation_permutations'],
                             **{k:v for k,v in p['rank'].items() if k != 'L'})
        opts = SubspaceOpts(L=p['rank']['L'], k_dim=1 if 'Singleton' in name else 2, dedup=True, pca=False)
        # L is a subspace option, not a core configuration field.
        test = AuditedRank(cfg, opts)
        result = test.test(x, y, seed=seed)
        stat, pv, used = result.T_obs, result.p_perm, result.perms_used
        row.update(effective_L=test.effective_L, views_hash=test.views_hash,
                   reference_positions_hash=test.reference_positions_hash)
    elif name.startswith('SCF'):
        result = GaussianWindowSCF(gamma=scale, seed=seed, B=p['evaluation_permutations'],
                                   alpha=p['alpha'], calibration='permutation', **p['scf']).run(x, y)
        stat, pv, used = result.stat_obs, result.p_perm, result.perms_used
    elif name.startswith('MMD'):
        result = RFFMMD(bandwidth=scale, seed=seed, B=p['evaluation_permutations'], alpha=p['alpha'], **p['rff']).run(x, y)
        stat, pv, used = result.stat_obs, result.p_perm, result.perms_used
    elif name.startswith('C2ST'):
        truth = np.r_[np.ones(len(x), int), np.zeros(len(y), int)]
        prediction = fitted.predict(np.vstack([x, y]))
        stat = float(np.mean(truth == prediction))
        pv = prediction_hypergeometric_pvalue(truth, prediction); used = 0
        row['c2st_confusion'] = {'n':len(truth), 'true_ones':int(truth.sum()),
                                 'predicted_ones':int(prediction.sum()),
                                 'correct_ones':int(np.sum((truth==1)&(prediction==1)))}
    else:
        raise ValueError(name)
    row['evaluation_seconds'] = time.perf_counter()-evaluation_start
    if not np.isfinite([stat, pv]).all() or not 0 <= pv <= 1: raise FloatingPointError('Invalid result')
    expected = 0 if name.startswith('C2ST') else p['evaluation_permutations']
    if used != expected: raise RuntimeError('Permutation-budget mismatch')
    row.update(stat_obs=float(stat), pvalue=float(pv), reject=bool(pv <= p['alpha']), perms_used=int(used))


def run_block(p, case, d, rep, emit, cohort):
    started = time.perf_counter()
    raw, q = make_data(p, case, d, rep)
    for orientation_index, orientation in enumerate(case['orientations']):
        before = time.perf_counter()
        oriented = raw if orientation == 'raw' else tuple(a @ q for a in raw)
        prepared, scale, center_hash = prepare(oriented)
        prep_seconds = time.perf_counter()-before
        common = {'case': case['id'], 'null': case['null'], 'd': d, 'rep': rep,
                  'orientation': orientation, 'cohort': cohort, 'data_seed': seed_for(p, case, d, rep, 1),
                  'rotation_seed': seed_for(p, case, d, rep, 2), 'base_data_hash': arrays_hash(*raw),
                  'oriented_data_hash': arrays_hash(*oriented), 'training_hash': arrays_hash(*prepared[:2]),
                  'evaluation_hash': arrays_hash(*prepared[2:]), 'center_hash': center_hash,
                  'median_scale': scale, 'common_prep_seconds': prep_seconds}
        order = np.random.default_rng(seed_for(p, case, d, rep, 3+orientation_index)).permutation(len(p['methods']))
        for position, index in enumerate(order):
            name = p['methods'][index]
            # Median/grid feature controls share seeds; rank controls share seed too.
            domain = 10 if name.startswith('Rank') else 11 if name.startswith('SCF') else 12 if name.startswith('MMD') else 13
            seed = seed_for(p, case, d, rep, domain)
            row = {**common, 'method': name, 'method_seed': seed, 'order_position': position,
                   'status': 'OK', 'warnings': [], 'candidate_trace': [], 'selected_scale': None,
                   'training_seconds': None, 'evaluation_seconds': None, 'stat_obs': None,
                   'pvalue': None, 'reject': None, 'perms_used': None, 'effective_L': None,
                   'views_hash': None, 'reference_positions_hash': None, 'c2st_confusion': None}
            call_start = time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                warnings.simplefilter('error', ConvergenceWarning)
                try:
                    with time_limit(p['method_timeout_seconds']): evaluate(name, prepared, scale, seed, p, row)
                    if arrays_hash(*prepared[:2]) != common['training_hash'] or arrays_hash(*prepared[2:]) != common['evaluation_hash']:
                        raise RuntimeError('Input mutation')
                except Exception as error:
                    row.update(status=f'ERROR:{type(error).__name__}:{error}', reject=None)
                row['warnings'] = [str(w.message) for w in captured]
            row['call_seconds'] = time.perf_counter()-call_start
            row['total_seconds'] = prep_seconds + row['call_seconds']
            emit(row)
    return time.perf_counter()-started


def source_snapshot(out):
    paths = [ROOT/'pyproject.toml', ROOT/'uv.lock']
    for directory in ('src', 'scripts', 'tests', 'experiments'):
        paths += sorted(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.md', '.slurm'))
    files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with tarfile.open(out/'source.tar.gz', 'w:gz') as archive:
        for path in paths: archive.add(path, arcname=str(path.relative_to(ROOT)))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stage', choices=['preflight', 'decision'], default='preflight')
    ap.add_argument('--pilot-reps', type=int, default=2)
    ap.add_argument('--cohort', required=True)
    ap.add_argument('--authorization', type=Path)
    ap.add_argument('--rep-start', type=int, default=0)
    ap.add_argument('--rep-stop', type=int)
    args = ap.parse_args()
    p = json.loads(PROTOCOL.read_text())
    if args.pilot_reps < 1 or args.rep_start < 0: raise ValueError('Positive pilot count/nonnegative ranges required')
    if args.stage == 'decision':
        if args.authorization is None: raise PermissionError('Decision scale needs separately approved inclusive CPU budget.')
        approval = json.loads(args.authorization.read_text())
        commit = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip()
        assert approval['protocol_hash'] == digest(p) and approval['source_commit'] == commit
        assert approval['maximum_total_cpu_core_hours'] > 0 and approval['approved_by_user'] is True
        assert approval['cpu_models'][args.cohort] == cpu_model()
        dims = p['cohort_dimensions'][args.cohort]
        if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'], text=True).strip():
            raise RuntimeError('Uncommitted source is forbidden at decision scale')
    else:
        dims = p['dimensions']
    args.out.mkdir(parents=True, exist_ok=False)
    pools = threadpool_info()
    if any(v['num_threads'] != 1 for v in pools): raise RuntimeError('Use one BLAS/OpenMP thread')
    meta = {'protocol': p, 'protocol_hash': digest(p), 'stage': args.stage, 'cohort': args.cohort,
            'arguments': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'source_commit': subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip(),
            'source_dirty': bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'], text=True).strip()),
            'source_hashes': source_snapshot(args.out), 'cpu_model': cpu_model(), 'node': platform.node(),
            'platform': platform.platform(), 'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
            'sklearn': sklearn.__version__, 'threadpools': pools, 'slurm_job': os.environ.get('SLURM_JOB_ID'),
            'start_unix': time.time(), 'status': 'running', 'expected_calls': 0, 'written_calls': 0, 'failed_calls': 0,
            'blocks': []}
    tasks = []
    for d in dims:
        for case in p['cases']:
            count = p['null_repetitions'] if case['null'] else p['power_repetitions']
            if args.stage == 'preflight': reps = range(p['pilot_rep_start'], p['pilot_rep_start']+args.pilot_reps)
            else:
                stop = count if args.rep_stop is None else min(count, args.rep_stop)
                reps = range(args.rep_start, stop)
            tasks += [(case, d, r) for r in reps]
    if not tasks: raise ValueError('Empty shard')
    meta['expected_calls'] = sum(len(c['orientations'])*len(p['methods']) for c,_,_ in tasks)
    (args.out/'metadata.json').write_text(json.dumps(meta, indent=2))
    began = time.perf_counter()
    with (args.out/'raw.jsonl').open('x') as file:
        def emit(row):
            row['protocol_hash'] = meta['protocol_hash']
            file.write(json.dumps(row, allow_nan=False)+'\n'); file.flush()
            meta['written_calls'] += 1
            meta['failed_calls'] += row['status'] != 'OK'
        try:
            for case,d,rep in tasks:
                elapsed = run_block(p, case, d, rep, emit, args.cohort)
                meta['blocks'].append({'case':case['id'],'d':d,'rep':rep,'seconds':elapsed})
        finally:
            meta.update(status='complete' if meta['written_calls']==meta['expected_calls'] and not meta['failed_calls'] else 'incomplete_or_failed',
                        elapsed_seconds=time.perf_counter()-began,
                        raw_sha256=hashlib.sha256((args.out/'raw.jsonl').read_bytes()).hexdigest(),
                        source_archive_sha256=hashlib.sha256((args.out/'source.tar.gz').read_bytes()).hexdigest())
            (args.out/'metadata.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({k:meta[k] for k in ('status','expected_calls','written_calls','failed_calls','elapsed_seconds')}))
    if meta['status'] != 'complete': raise SystemExit(1)


if __name__ == '__main__': main()
