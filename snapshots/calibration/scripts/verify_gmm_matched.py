#!/usr/bin/env python3
"""Verify moment matching for the high-dimensional matched GMM benchmark."""
from __future__ import annotations

import argparse
import numpy as np

from ranktwosample.power_datasets import generate_pair


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=20)
    ap.add_argument("--N", type=int, default=200000)
    ap.add_argument("--delta", type=float, default=0.8)
    ap.add_argument("--sparse_dims", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    X, Y = generate_pair(
        "gmm-matched",
        args.d,
        args.N,
        args.N,
        params={"delta": args.delta, "sparse_dims": args.sparse_dims},
        rng=args.seed,
    )
    mean_x = X.mean(axis=0)
    mean_y = Y.mean(axis=0)
    cov_x = np.cov(X, rowvar=False)
    cov_y = np.cov(Y, rowvar=False)
    ident = np.eye(args.d)
    print(f"d={args.d} N={args.N} delta={args.delta} sparse_dims={args.sparse_dims}")
    print(f"max_abs_mean_X={np.max(np.abs(mean_x)):.4g}")
    print(f"max_abs_mean_Y={np.max(np.abs(mean_y)):.4g}")
    print(f"fro_cov_X_minus_I={np.linalg.norm(cov_x-ident, ord='fro'):.4g}")
    print(f"fro_cov_Y_minus_I={np.linalg.norm(cov_y-ident, ord='fro'):.4g}")
    print(f"max_abs_cov_X_minus_I={np.max(np.abs(cov_x-ident)):.4g}")
    print(f"max_abs_cov_Y_minus_I={np.max(np.abs(cov_y-ident)):.4g}")


if __name__ == "__main__":
    main()
