"""Prospective dependence rerun; imports but never modifies frozen source."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tarfile
import time
import warnings

import numpy as np
import scipy
from threadpoolctl import threadpool_info
from ranktwosample.power_datasets import (gaussian_copula_u, sample_clayton_u,
    sample_gumbel_u, kendall_tau_of_gaussian, clayton_theta_from_tau,
    gumbel_theta_from_tau, safe_beta_ppf)
from ranktwosample.rank_two_sample import RankGTestConfig, _rank_f_statistic_from_expected
from ranktwosample.rank_two_sample_subspaces import RankGTestSubspaces, SubspaceOpts
from baselines.me_scf import (_median_bandwidth, _choose_locations, _rbf_features,
    _choose_frequencies, _scf_features, _stat_from_features)
from baselines.kernel_mmd import _median_bandwidths, _gram_rbf
from scripts.fair_comparator_bench import arrays_hash, digest, cpu_model

HERE = Path(__file__).resolve().parent


def strata(p):
    result = []
    for d in p['copula']['dimensions']:
        for n in p['copula']['sample_sizes']:
            for law in p['copula']['generators']:
                result.append(dict(study='copula', d=d, n=n, null=True, law=law, rho=.6))
            for law in ('clayton', 'gumbel'):
                result.append(dict(study='copula', d=d, n=n, null=False, law=law, rho=.6))
    for n in p['gaussian']['sample_sizes']:
        for rho in (0., *p['gaussian']['rho_alternatives']):
            result.append(dict(study='gaussian', d=2, n=n, null=rho == 0., law='gaussian', rho=rho))
    for index, item in enumerate(result):
        item['id'] = index
    return result


def seed_for(p, s, rep, domain):
    return int(np.random.SeedSequence([p['base_seed'], s['id'], rep, domain]).generate_state(1)[0])


def draw(law, n, d, rho, rng, beta=False):
    tau = kendall_tau_of_gaussian(rho)
    if law == 'gaussian':
        u = gaussian_copula_u(n, d, rho, rng)
    elif law == 'clayton':
        u = sample_clayton_u(clayton_theta_from_tau(tau), n, d, rng)
    elif law == 'gumbel':
        u = sample_gumbel_u(gumbel_theta_from_tau(tau), n, d, rng)
    else:
        raise ValueError(law)
    if not np.isfinite(u).all() or np.any((u <= 0) | (u >= 1)):
        raise FloatingPointError('Non-interior copula draw')
    return safe_beta_ppf(u) if beta else u


def make_data(p, s, rep):
    rng = np.random.default_rng(seed_for(p, s, rep, 1))
    beta = s['study'] == 'copula'
    x = draw(s['law'], s['n'], s['d'], s['rho'], rng, beta)
    reference_law = s['law'] if s['null'] else 'gaussian'
    reference_rho = s['rho'] if s['null'] else p[s['study']]['rho_reference']
    y = draw(reference_law, s['n'], s['d'], reference_rho, rng, beta)
    return x, y


def mmd_scores(gram, allocations):
    """Unbiased Gram statistic; allocations have target ones, reference zeros."""
    a = np.asarray(allocations, dtype=float)
    n = a.sum(axis=1)
    m = gram.shape[0] - n
    if np.any(n < 2) or np.any(m < 2):
        raise ValueError('At least two observations per role')
    ag = a @ gram
    xx = (ag * a).sum(axis=1)
    row = ag.sum(axis=1)
    diagx = a @ np.diag(gram)
    diagy = np.trace(gram) - diagx
    yy = gram.sum() - 2 * row + xx
    return (xx-diagx)/(n*(n-1)) + (yy-diagy)/(m*(m-1)) - 2*(row-xx)/(n*m)


def feature_scores(features, allocations, ridge):
    return np.asarray([_stat_from_features(features, a, int(a.sum()), ridge) for a in allocations])


def allocation_fields(n, m, permutations):
    iy = np.vstack([np.arange(n, n+m), permutations[:, :m]])
    ix = np.vstack([np.arange(n), permutations[:, m:]])
    labels = np.zeros((len(ix), n+m), dtype=np.int8)
    np.put_along_axis(labels, ix, 1, axis=1)
    return ix, iy, labels


def rank_table_scores(z, ix, iy, views, reference_positions, k, gamma, kind):
    scores = np.zeros(len(ix))
    table_hash = hashlib.sha256()
    for view, rp in zip(views, reference_positions):
        refs = iy[:, rp]
        cells = np.zeros((len(ix), len(z)), dtype=np.int64)
        for col in view:
            ranks = (z[None, :, col, None] >= z[refs, col][:, None, :]).sum(axis=2)
            cells = cells*(k+1) + ranks
        count = (k+1)**len(view)
        offset = (np.arange(len(ix))*count)[:, None]
        hx = np.bincount((offset+np.take_along_axis(cells, ix, axis=1)).ravel(),
                         minlength=len(ix)*count).reshape(len(ix), count)
        hy = np.bincount((offset+np.take_along_axis(cells, iy, axis=1)).ravel(),
                         minlength=len(ix)*count).reshape(len(ix), count)
        n, m = ix.shape[1], iy.shape[1]
        if not (np.all(hx.sum(axis=1) == n) and np.all(hy.sum(axis=1) == m)):
            raise RuntimeError('Histogram mass mismatch')
        expected = n*((hy+gamma)/(m+gamma*count))
        scores += np.asarray([_rank_f_statistic_from_expected(np, h, e, kind=kind)
                              for h, e in zip(hx, expected)])
        table_hash.update(hx.astype('<i4').tobytes())
        table_hash.update(hy.astype('<i4').tobytes())
    return scores, table_hash.hexdigest()


def calibrate(scores, p):
    if len(scores) != p['B']+1 or not np.isfinite(scores).all():
        raise FloatingPointError('Bad score vector')
    tail = int(np.sum(scores[1:] >= scores[0]))
    pv = (1+tail)/(p['B']+1)
    return dict(stat_obs=float(scores[0]), permutation_statistics=scores[1:].tolist(),
                tail_count=tail, pvalue=pv, reject=pv <= p['alpha'], perms_used=p['B'])


def evaluate(p, s, x, y, seed):
    settings = p[s['study']]
    kind = 'gtest' if s['study'] == 'copula' else 'chi2'
    cfg = RankGTestConfig(alpha=p['alpha'], B=p['B'], K_ref=settings['K'],
                         stat=kind, **p['rank'])
    rank = RankGTestSubspaces(cfg, SubspaceOpts(L=settings['L'], k_dim=2, dedup=True, pca=False))
    rng = rank._runner._rng(seed)
    before = time.perf_counter()
    observed, env = rank._compute_T_obs_sum_shared_pool(x, y, rng)
    k, _, _, _, _, jitter, side, _, views, n, m, rp = env
    if jitter is None or side != 'right':
        raise RuntimeError('Unqualified rank mode')
    allocations_before = time.perf_counter()
    permutations = np.asarray([rng.permutation(n+m) for _ in range(p['B'])])
    ix, iy, labels = allocation_fields(n, m, permutations)
    allocation_seconds = time.perf_counter()-allocations_before
    z = np.vstack([x, y])
    zrank = z + jitter.T
    scores, tables_hash = rank_table_scores(zrank, ix, iy, views, rp, k, cfg.alpha0, kind)
    if scores[0] != observed:
        raise RuntimeError('Frozen rank observation mismatch')
    name = settings['methods'][0]
    results = {name: {**calibrate(scores, p), 'call_seconds': time.perf_counter()-before}}
    common = dict(effective_L=len(views), views=views,
                  reference_positions=[a.tolist() for a in rp], rank_tables_hash=tables_hash,
                  jitter_hash=arrays_hash(jitter), permutations_hash=arrays_hash(permutations),
                  allocation_seconds=allocation_seconds)
    if s['study'] == 'gaussian':
        before = time.perf_counter()
        scores, th = rank_table_scores(zrank, ix, iy, [(0,)], [rp[0]], k, cfg.alpha0, kind)
        results[settings['methods'][1]] = {**calibrate(scores, p),
            'call_seconds': time.perf_counter()-before, 'rank_tables_hash': th,
            'shared_field_timing': True}
        # Marginal timing shares construction; not a standalone runtime comparison.
        return results, common
    f = p['features']
    # Distinct feature draws, same ordered allocation space and same data.
    for name in ('ME', 'Fourier_unsmoothed', 'MMD_exact_median_grid'):
        before = time.perf_counter()
        if name == 'MMD_exact_median_grid':
            bandwidths = _median_bandwidths(z, subsample=p['mmd']['median_subsample'],
                num_scales=p['mmd']['num_scales'], base=p['mmd']['base'])
            gram = _gram_rbf(z, z, bandwidths)
            scores = mmd_scores(gram, labels)
            extras = {'bandwidths': bandwidths.tolist()}
        else:
            domain = 11 if name == 'ME' else 12
            fs = int(np.random.SeedSequence([seed, domain]).generate_state(1)[0])
            sigma = _median_bandwidth(z, subsample=f['median_subsample'], seed=fs)*f['bandwidth_scale']
            frng = np.random.default_rng(fs)
            field = (_choose_locations(z, f['J'], frng) if name == 'ME'
                     else _choose_frequencies(z.shape[1], f['J'], sigma, frng))
            features = (_rbf_features(z, field, sigma) if name == 'ME' else _scf_features(z, field))
            scores = feature_scores(features, labels, f['ridge'])
            extras = dict(sigma=sigma, feature_field_hash=arrays_hash(field), feature_seed=fs)
        results[name] = {**calibrate(scores, p),
                         'call_seconds': time.perf_counter()-before+allocation_seconds,
                         'shared_allocation_seconds_charged': allocation_seconds, **extras}
    return results, common


def receipt(base, out, p):
    paths = sorted((base/'src').rglob('*.py')) + [base/'pyproject.toml', base/'uv.lock']
    paths += sorted(HERE.glob('*.py')) + [HERE/'protocol.json', HERE/'protocol.md']
    paths += [HERE.parents[1]/'scripts/verify_dependence_power.py']
    records = {}
    with tarfile.open(out/'source.tar.gz', 'w:gz') as archive:
        for path in paths:
            name = str(path.relative_to(base)) if path.is_relative_to(base) else 'overlay/'+path.name
            records[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=name)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--study', choices=('copula', 'gaussian', 'both'), default='both')
    ap.add_argument('--stage', choices=('preflight', 'decision'), default='preflight')
    ap.add_argument('--pilot-reps', type=int, default=1)
    ap.add_argument('--rep-start', type=int, default=0)
    ap.add_argument('--rep-stop', type=int, default=0)
    ap.add_argument('--stratum', type=int, nargs='+')
    ap.add_argument('--wall-seconds', type=int, default=180)
    ap.add_argument('--authorization', type=Path)
    args = ap.parse_args()
    p = json.loads((HERE/'protocol.json').read_text())
    base = Path(os.environ['DEPENDENCE_BASE_SOURCE']).resolve()
    commit = subprocess.check_output(['git', '-C', str(base), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != p['base_source_commit'] or subprocess.check_output(
            ['git', '-C', str(base), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Frozen base mismatch or dirty source')
    if any(t['num_threads'] != 1 for t in threadpool_info()):
        raise RuntimeError('Multiple numerical threads')
    if args.wall_seconds < 1 or args.pilot_reps < 1:
        raise ValueError('Positive caps required')
    if args.stage == 'decision':
        if args.authorization is None:
            raise PermissionError('No priced decision-scale authorization')
        approval = json.loads(args.authorization.read_text())
        if not approval.get('approved_by_user') or approval.get('protocol_hash') != digest(p):
            raise PermissionError('Missing actual user approval/protocol binding')
        if approval.get('reserved_cpu_core_hours', float('inf')) > approval.get('maximum_total_cpu_core_hours', 0):
            raise PermissionError('Overcommitted campaign')
        if args.rep_stop <= args.rep_start or args.rep_start < 0:
            raise ValueError('Invalid decision range')
        expected = {q.name: hashlib.sha256(q.read_bytes()).hexdigest() for q in (HERE/'runner.py', HERE/'protocol.json')}
        if approval.get('numeric_overlay_hashes') != expected:
            raise PermissionError('Unbound numerical overlay')
        qualified = approval.get('qualified_cohorts', {}).get(args.study)
        actual = [cpu_model(), sys.version, np.__version__, scipy.__version__]
        if qualified != actual:
            raise PermissionError('Unqualified CPU/Python/library cohort')
        if approval.get('maximum_total_cpu_core_hours', 0) > p['execution']['maximum_total_cpu_core_hours']:
            raise PermissionError('Protocol CPU cap exceeded')
    selected = [s for s in strata(p) if (args.study == 'both' or s['study'] == args.study)
                and (args.stratum is None or s['id'] in args.stratum)]
    if not selected or args.stratum is not None and set(args.stratum) != {s['id'] for s in selected}:
        raise ValueError('Empty/invalid strata')
    ranges = range(p['pilot_rep_start'], p['pilot_rep_start']+args.pilot_reps) if args.stage == 'preflight' else range(args.rep_start, args.rep_stop)
    tasks = [(s, r) for s in selected for r in ranges if args.stage == 'preflight'
             or r < p['null_repetitions' if s['null'] else 'power_repetitions']]
    if not tasks:
        raise ValueError('Empty task set')
    args.out.mkdir(parents=True, exist_ok=False)
    hashes = receipt(base, args.out, p)
    meta = dict(protocol=p, protocol_hash=digest(p), source_commit=commit, source_hashes=hashes,
                stage=args.stage, cpu_model=cpu_model(), platform=platform.platform(), python=sys.version,
                numpy=np.__version__, scipy=scipy.__version__, threadpools=threadpool_info(),
                arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                slurm_job=os.environ.get('SLURM_JOB_ID'), status='running', start_unix=time.time(),
                expected_blocks=len(tasks), completed_blocks=0, failed_blocks=0, blocks=[])
    def stop(signum, frame):
        raise TimeoutError('Hard experiment wall cap reached')
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(args.wall_seconds)
    began = time.perf_counter()
    (args.out/'metadata.json').write_text(json.dumps(meta, indent=2))
    try:
        with (args.out/'raw.jsonl').open('x') as stream:
            for s, rep in tasks:
                before = time.perf_counter()
                row = dict(stratum=s, rep=rep, data_seed=seed_for(p,s,rep,1),
                           method_seed=seed_for(p,s,rep,10), status='OK', protocol_hash=digest(p))
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    try:
                        x,y = make_data(p,s,rep)
                        row['data_seconds'] = time.perf_counter()-before
                        row['data_hash'] = arrays_hash(x,y)
                        row['results'], row['common'] = evaluate(p,s,x,y,row['method_seed'])
                        if arrays_hash(x,y) != row['data_hash']:
                            raise RuntimeError('Input mutation')
                    except Exception as error:
                        row.update(status=f'ERROR:{type(error).__name__}:{error}', results=None)
                    row['warnings'] = [str(w.message) for w in captured]
                row['block_seconds'] = time.perf_counter()-before
                good = row['status'] == 'OK' and not row['warnings']
                meta['completed_blocks' if good else 'failed_blocks'] += 1
                meta['blocks'].append(dict(stratum=s['id'], rep=rep, seconds=row['block_seconds']))
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                if not good:
                    raise RuntimeError('Failed/warning block; no automatic retry')
        meta['status'] = 'complete'
    finally:
        signal.alarm(0)
        meta.update(elapsed_seconds=time.perf_counter()-began, end_unix=time.time())
        if meta['status'] != 'complete':
            meta['status'] = 'incomplete_or_failed'
        raw = args.out/'raw.jsonl'
        meta['raw_sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest() if raw.exists() else None
        (args.out/'metadata.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({k:meta[k] for k in ('status','completed_blocks','failed_blocks','elapsed_seconds')}))


if __name__ == '__main__':
    main()
