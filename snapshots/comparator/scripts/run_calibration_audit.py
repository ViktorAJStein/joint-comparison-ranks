"""Seeded, CPU-only calibration blocks; historical files are never overwritten."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ranktwosample.rank_two_sample import RankGTestConfig
from ranktwosample.rank_two_sample_subspaces import RankGTestSubspaces, SubspaceOpts
from scripts.rotation_bench import generate_null, generate_sparse_scale, random_orthogonal, pooled_pca_transform

CASES = {'gaussian_balanced': (0, 500, 500), 'rounded_balanced': (1, 500, 500),
         'gaussian_200_800': (2, 200, 800), 'gaussian_800_200': (3, 800, 200)}
BASE = RankGTestConfig(alpha=.05, K_ref=4, B=100, alpha0=.1, engine='auto',
                      tie='jitter', jitter_scale=1e-12, decision='pvalue',
                      stop_ci='none', min_b_check=50, chunk=512, antithetic=False,
                      stat='gtest', gpu='off', reference_mode='shared_pool')
VARIANTS = {'conservative': BASE, 'midp_only': replace(BASE, decision='midp'),
            'wilson_only': replace(BASE, stop_ci='wilson'),
            'antithetic_only': replace(BASE, antithetic=True),
            'legacy_combined': replace(BASE, decision='midp', stop_ci='wilson', antithetic=True)}
SUBS = SubspaceOpts(L=64, k_dim=1, dedup=True, pca=False)


def data_hash(x, y):
    h = hashlib.sha256()
    for a in (x, y):
        h.update(str(a.shape).encode()); h.update(np.ascontiguousarray(a, dtype='<f8').tobytes())
    return h.hexdigest()


class AuditedTest(RankGTestSubspaces):
    def _compute_T_obs_sum_shared_pool(self, x, y, rng):
        result, env = super()._compute_T_obs_sum_shared_pool(x, y, rng)
        self.effective_L = len(env[8])
        self.views_hash = hashlib.sha256(json.dumps(env[8]).encode()).hexdigest()
        self.reference_positions_hash = hashlib.sha256(json.dumps(
            [p.tolist() for p in env[-1]]).encode()).hexdigest()
        return result, env


def evaluate(x, y, cfg, seed):
    test = AuditedTest(cfg, SUBS)
    out = test.test(x, y, seed=seed)
    assert np.isfinite(out.T_obs) and 0 <= out.p_perm <= 1
    if cfg.stop_ci == 'none':
        assert out.perms_used == cfg.B
    return {'reject': int(out.reject), 'stat_obs': out.T_obs, 'p_perm': out.p_perm,
            'p_mid': out.p_mid, 'perms_used': out.perms_used, 'elapsed_ms': out.elapsed_ms,
            'effective_L': test.effective_L, 'views_hash': test.views_hash,
            'reference_positions_hash': test.reference_positions_hash,
            'config_hash': hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()}


def block(task):
    case, rep, base_seed, power = task
    case_id, n, m = CASES[case]
    key = (rep,) if case_id == 0 else (case_id, rep)
    seed = int(np.random.SeedSequence(base_seed, spawn_key=key).generate_state(1)[0])
    method_seed = seed + 17
    rng = np.random.default_rng(seed)
    if power:
        x, y = generate_sparse_scale(20, n, m, 1.5, rng)
    else:
        x, y = generate_null(20, n, m, rng)
    if case == 'rounded_balanced':
        x, y = np.round(x), np.round(y)
    rows = []
    common = {'case': case, 'rep': rep, 'alternative': int(power), 'N': n, 'M': m,
              'd': 20, 'data_seed': seed, 'method_seed': method_seed, 'requested_L': 64,
              'data_hash': data_hash(x, y)}

    def add(method, xx, yy, cfg, orientation='original'):
        out = evaluate(xx, yy, cfg, method_seed)
        rows.append({**common, 'method': method, 'orientation': orientation,
                     'score_input_hash': data_hash(xx, yy), 'reuse_certificate': '', **out})
        return out

    if not power:
        add('conservative', x, y, BASE)
        for name, cfg in VARIANTS.items():
            if name == 'conservative': continue
            # Stopping checks also occur INSIDE permutation loops. Every
            # variant must execute its own actual loop; no score reuse.
            add(name, x, y, cfg)
    else:
        add('conservative', x, y, BASE)

    if case == 'gaussian_balanced':
        q = random_orthogonal(20, rng)
        for orientation, xx, yy in [('original', x, y), ('rotated', x @ q, y @ q)]:
            if orientation == 'rotated': add('conservative', xx, yy, BASE, orientation)
            px, py = pooled_pca_transform(xx, yy, canonical_order=True)
            wx, wy = pooled_pca_transform(xx, yy, whiten=True, canonical_order=True)
            add('canonical_pca', px, py, BASE, orientation)
            add('canonical_whiten', wx, wy, BASE, orientation)
            # Check, but never force, the whitening identity in the actual
            # fixed-scale post-transform jitter implementation.
            a, b = rows[-2:]
            equal = all(a[k] == b[k] for k in ('stat_obs', 'p_perm', 'p_mid', 'reject', 'perms_used'))
            a['whitening_identity'] = b['whitening_identity'] = int(equal)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reps', type=int, default=4)
    parser.add_argument('--power-reps', type=int, default=2)
    parser.add_argument('--rep-start', type=int, default=0)
    parser.add_argument('--seed', type=int, default=6060)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--cohort', required=True)
    args = parser.parse_args()
    assert args.reps > 0 and args.workers > 0 and args.power_reps >= 0
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / 'raw.csv'
    if output.exists(): raise FileExistsError(output)
    files = [ROOT / 'uv.lock', ROOT / 'pyproject.toml', Path(__file__)] + sorted((ROOT / 'src').rglob('*.py'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    metadata = {'arguments': {**vars(args), 'out': str(args.out)}, 'python': sys.version,
                'numpy': np.__version__, 'scipy': scipy.__version__, 'platform': platform.platform(),
                'node': platform.node(), 'cpu': platform.processor(), 'slurm_job': os.environ.get('SLURM_JOB_ID'),
                'configurations': {k: asdict(v) for k, v in VARIANTS.items()}, 'subspaces': asdict(SUBS),
                'source_hashes': hashes, 'start_unix': time.time(), 'status': 'running'}
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    tasks = [(case, r, args.seed, False) for case in CASES for r in range(args.rep_start, args.rep_start + args.reps)]
    tasks += [('gaussian_balanced', r, args.seed, True) for r in range(args.rep_start, args.rep_start + args.power_reps)]
    fields = list(block(tasks[0])[0]) + ['whitening_identity', 'cohort']
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, output.open('x', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i, rows in enumerate(pool.map(block, tasks, chunksize=8)):
            for row in rows: writer.writerow({**row, 'cohort': args.cohort})
            file.flush()
            if (i+1) % 100 == 0: print(f'{i+1}/{len(tasks)} blocks; elapsed={time.perf_counter()-started:.1f}s', flush=True)
    metadata.update(status='complete', elapsed_seconds=time.perf_counter()-started,
                    complete_blocks=len(tasks), raw_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: metadata[k] for k in ('status', 'elapsed_seconds', 'complete_blocks', 'raw_sha256')}))

if __name__ == '__main__': main()
