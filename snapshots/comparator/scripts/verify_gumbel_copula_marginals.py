#!/usr/bin/env python3
"""Diagnostic for dep-gumbel benchmark marginal and moment artifacts."""
from __future__ import annotations

import argparse
import numpy as np
from scipy.stats import ks_2samp

from ranktwosample.power_datasets import generate_pair


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--N", type=int, default=200000)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    X, Y = generate_pair(
        "dep-gumbel",
        args.d,
        args.N,
        args.N,
        params={"rho": args.rho, "nu": 6, "equicorr": True},
        rng=args.seed,
    )
    mean_diff = X.mean(axis=0) - Y.mean(axis=0)
    var_diff = X.var(axis=0) - Y.var(axis=0)
    cov_diff = np.cov(X, rowvar=False) - np.cov(Y, rowvar=False)
    ks = [ks_2samp(X[:, j], Y[:, j]).statistic for j in range(args.d)]
    print(f"d={args.d} N={args.N} rho={args.rho}")
    print("mean_X", np.array2string(X.mean(axis=0), precision=5))
    print("mean_Y", np.array2string(Y.mean(axis=0), precision=5))
    print(f"max_abs_mean_diff={np.max(np.abs(mean_diff)):.6g}")
    print(f"max_abs_var_diff={np.max(np.abs(var_diff)):.6g}")
    print(f"max_abs_cov_diff={np.max(np.abs(cov_diff)):.6g}")
    print(f"max_marginal_KS={max(ks):.6g}")
    print("cov_diag_X", np.array2string(np.diag(np.cov(X, rowvar=False)), precision=5))
    print("cov_diag_Y", np.array2string(np.diag(np.cov(Y, rowvar=False)), precision=5))
    if args.d >= 2:
        print(f"corr01_X={np.corrcoef(X[:,0], X[:,1])[0,1]:.6g}")
        print(f"corr01_Y={np.corrcoef(Y[:,0], Y[:,1])[0,1]:.6g}")


if __name__ == "__main__":
    main()
