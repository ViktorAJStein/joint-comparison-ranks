#!/usr/bin/env python3
"""Rotation sensitivity benchmark for rebuttal experiments.

Compares original sparse-coordinate alternatives with the same testing problem
after applying a common random orthogonal rotation to both samples.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from baselines.c2st import C2STBinomial
from baselines.hotelling_t2 import HotellingT2
from baselines.kernel_mmd import KernelMMDTest
from baselines.me_scf import MEFiniteFeatureTest
from baselines.rank_chi2_subspaces import RankChi2SubspacesTest


def se_binom(p: float, n: int) -> float:
    return float(math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1)))


def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs[None, :]


def pooled_pca_transform(X: np.ndarray, Y: np.ndarray, *, whiten: bool = False, eps: float = 1e-12, canonical_order: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Apply pooled PCA, optionally with explicitly row-equivariant fitting.

    canonical_order=True lexicographically sorts the unlabeled fitting cloud
    before centering/SVD; even repeated eigenspaces then receive the same
    numerical basis under row permutations on a fixed software/hardware cohort.
    The default retains historical fitting order. Calibration ALSO requires an
    appropriate identical allocation map and conservative permutation route.
    """
    Z = np.vstack([np.asarray(X, dtype=float), np.asarray(Y, dtype=float)])
    fit = Z[np.lexsort(Z.T[::-1])] if canonical_order else Z
    mean = fit.mean(axis=0, keepdims=True)
    Zc = Z - mean
    _, singular_values, Vt = np.linalg.svd(fit - mean, full_matrices=False)
    scores = Zc @ Vt.T
    if whiten:
        scale = np.sqrt(max(Z.shape[0] - 1, 1)) / np.maximum(singular_values, eps)
        scores = scores * scale[None, :]
    n = X.shape[0]
    return scores[:n], scores[n:]


def generate_sparse_shift(d: int, n: int, m: int, delta: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    Y = rng.normal(size=(m, d))
    X = rng.normal(size=(n, d))
    X[:, 0] += float(delta)
    return X, Y


def generate_sparse_scale(d: int, n: int, m: int, sigma: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    Y = rng.normal(size=(m, d))
    X = rng.normal(size=(n, d))
    X[:, 0] *= float(sigma)
    return X, Y


def generate_null(d: int, n: int, m: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    return rng.normal(size=(n, d)), rng.normal(size=(m, d))


def run_method(name: str, X: np.ndarray, Y: np.ndarray, alpha: float, B: int, seed: int, args) -> Tuple[bool, float, float, str]:
    t0 = time.perf_counter()
    try:
        if name == "C2ST_Binomial":
            out = C2STBinomial(alpha=alpha, split_prop=args.c2st_split_prop, seed=seed).run(X, Y)
            return bool(out.reject), float(out.stat_obs), 1e3 * (time.perf_counter() - t0), "OK"
        if name == "ME_SCF":
            out = MEFiniteFeatureTest(J=args.me_J, B=B, alpha=alpha, seed=seed, min_b_check=args.min_b_check).run(X, Y)
            return bool(out.reject), float(out.stat_obs), 1e3 * (time.perf_counter() - t0), "OK"
        if name in ("RankChi2_Subspaces", "RankChi2_Subspaces_PooledPCA", "RankChi2_Subspaces_PooledPCAWhiten"):
            X_rank, Y_rank = X, Y
            if name == "RankChi2_Subspaces_PooledPCA":
                X_rank, Y_rank = pooled_pca_transform(X, Y, whiten=False)
            elif name == "RankChi2_Subspaces_PooledPCAWhiten":
                X_rank, Y_rank = pooled_pca_transform(X, Y, whiten=True)
            test = RankChi2SubspacesTest(
                K_ref=args.K_ref,
                alpha=alpha,
                B=B,
                engine="auto",
                tie="jitter",
                decision="midp",
                stop_ci="wilson",
                min_b_check=args.min_b_check,
                stat=args.rank_stat,
                gpu="off",
                L=args.L_subspaces,
                k_dim=args.rank_subspace_dim,
                dedup=True,
                pca=False,
            )
            out = test.run(X_rank, Y_rank, seed=seed)
            return bool(out.reject), float(getattr(out, "stat_obs", np.nan)), 1e3 * (time.perf_counter() - t0), "OK"
        if name == "MMD":
            test = KernelMMDTest(B=B, alpha=alpha, decision="pvalue", early_stop="wilson", min_b_check=args.min_b_check)
            out = test.run(X, Y)
            return bool(out.reject), float(getattr(out, "stat_obs", np.nan)), 1e3 * (time.perf_counter() - t0), "OK"
        if name == "HotellingT2":
            test = HotellingT2(B=B, alpha=alpha, decision="pvalue", early_stop="wilson", min_b_check=args.min_b_check, shrinkage="ridge", ridge_lambda=1e-6, seed=seed)
            out = test.run(X, Y)
            return bool(out.reject), float(getattr(out, "stat_obs", np.nan)), 1e3 * (time.perf_counter() - t0), "OK"
        return False, float("nan"), 1e3 * (time.perf_counter() - t0), "SKIP"
    except Exception as e:
        return False, float("nan"), 1e3 * (time.perf_counter() - t0), f"ERROR:{type(e).__name__}:{e}"


@dataclass(frozen=True)
class Job:
    rep: int
    seed: int
    experiment: str
    orientation: str


def one_rep(job: Job, args) -> Dict[str, Dict]:
    rng = np.random.default_rng(job.seed)
    if job.experiment == "calibration":
        X, Y = generate_null(args.d, args.N, args.M, rng)
    else:
        if args.scenario == "sparse_shift":
            X, Y = generate_sparse_shift(args.d, args.N, args.M, args.delta, rng)
        elif args.scenario == "sparse_scale":
            X, Y = generate_sparse_scale(args.d, args.N, args.M, args.delta, rng)
        else:
            raise ValueError(f"Unknown scenario {args.scenario}")
    if job.orientation == "rotated":
        Q = random_orthogonal(args.d, rng)
        X = X @ Q
        Y = Y @ Q
    out: Dict[str, Dict] = {}
    for method in args.methods:
        reject, stat, ms, status = run_method(method, X, Y, args.alpha, args.B, job.seed + 17, args)
        out[method] = {"reject": reject, "stat": stat, "ms": ms, "status": status}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["power", "calibration"], default="power")
    ap.add_argument("--d", type=int, default=20)
    ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--M", type=int, default=300)
    ap.add_argument("--scenario", choices=["sparse_shift", "sparse_scale"], default="sparse_shift")
    ap.add_argument("--delta", type=float, default=0.6, help="shift size for sparse_shift or scale factor for sparse_scale")
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--B", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=6060)
    ap.add_argument("--out_csv", type=str, default="rotation_results.csv")
    ap.add_argument("--methods", nargs="+", default=["C2ST_Binomial", "ME_SCF", "RankChi2_Subspaces", "RankChi2_Subspaces_PooledPCA", "MMD", "HotellingT2"])
    ap.add_argument("--K_ref", type=int, default=4)
    ap.add_argument("--L_subspaces", type=int, default=64)
    ap.add_argument("--rank_subspace_dim", type=int, default=2)
    ap.add_argument("--rank_stat", type=str, default="gtest")
    ap.add_argument("--me_J", type=int, default=10)
    ap.add_argument("--c2st_split_prop", type=float, default=0.7)
    ap.add_argument("--min_b_check", type=int, default=50)
    args = ap.parse_args()

    ss = np.random.SeedSequence(args.seed)
    seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(args.R)]
    rows = []
    for orientation in ["original", "rotated"]:
        jobs = [Job(rep=i, seed=seeds[i], experiment=args.experiment, orientation=orientation) for i in range(args.R)]
        counts = {m: 0 for m in args.methods}
        completed = {m: 0 for m in args.methods}
        errors = {m: 0 for m in args.methods}
        times = {m: [] for m in args.methods}
        print(f"\n=== {args.experiment} orientation={orientation} d={args.d} N={args.N} delta={args.delta} ===")
        if args.jobs == 1:
            iterator = ((j, one_rep(j, args)) for j in jobs)
        else:
            ex = ProcessPoolExecutor(max_workers=args.jobs)
            futs = {ex.submit(one_rep, j, args): j for j in jobs}
            iterator = ((futs[f], f.result()) for f in as_completed(futs))
        done = 0
        for _job, out in iterator:
            done += 1
            for m in args.methods:
                info = out[m]
                times[m].append(info["ms"])
                if info["status"] == "OK":
                    completed[m] += 1
                    counts[m] += int(info["reject"])
                else:
                    errors[m] += 1
            if done % max(1, args.R // 10) == 0:
                print(f"  progress {done}/{args.R} ...", end="\r")
        if args.jobs != 1:
            ex.shutdown()
        for m in args.methods:
            c = completed[m]
            rate = counts[m] / c if c else float("nan")
            se = se_binom(rate, c) if c else float("nan")
            avg_ms = float(np.mean(times[m])) if times[m] else float("nan")
            print(f"  {m:18s}: rate={rate:.3f} SE={se:.3f} avg_ms={avg_ms:.1f} completed={c}/{args.R} errors={errors[m]}")
            rows.append({
                "experiment": args.experiment,
                "orientation": orientation,
                "method": m,
                "d": args.d,
                "N": args.N,
                "M": args.M,
                "scenario": args.scenario,
                "delta": args.delta,
                "alpha": args.alpha,
                "B": args.B,
                "R": args.R,
                "R_completed": c,
                "reject_rate": f"{rate:.6f}",
                "se": f"{se:.6f}",
                "avg_ms": f"{avg_ms:.2f}",
                "errors": errors[m],
                "K_ref": args.K_ref if m.startswith("RankChi2_Subspaces") else "",
                "L_subspaces": args.L_subspaces if m.startswith("RankChi2_Subspaces") else "",
                "rank_subspace_dim": args.rank_subspace_dim if m.startswith("RankChi2_Subspaces") else "",
            })
    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Done] wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
