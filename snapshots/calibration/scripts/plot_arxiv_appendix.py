#!/usr/bin/env python3
"""Regenerate the three post-review appendix figures from archived CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

METHODS = ["ME", "SCF", "RankChi2_Subspaces", "MMD"]
LABELS = {
    "ME": "ME",
    "SCF": "SCF",
    "RankChi2_Subspaces": "Rank-sub",
    "MMD": "MMD",
}
COLORS = {
    "ME": "#1f77b4",
    "SCF": "#ff7f0e",
    "RankChi2_Subspaces": "#17becf",
    "MMD": "#d62728",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, output: Path) -> None:
    for extension in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{extension}"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_common(data: Path, output: Path) -> None:
    rows = read_rows(data / "scf_common_d4_power_curves_r30.csv")
    families = ["mixture", "dep-clayton", "dep-gumbel"]
    titles = ["Mixture", "Clayton", "Gumbel"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.8), sharex="col")
    for column, (family, title) in enumerate(zip(families, titles)):
        for method in METHODS:
            values = sorted(
                (int(row["N"]), float(row["power"]), float(row["avg_ms"]))
                for row in rows
                if row["family"] == family and row["method"] == method
            )
            if values:
                sample_sizes, power, runtime = zip(*values)
                axes[0, column].plot(
                    sample_sizes, power, marker="o", label=LABELS[method],
                    color=COLORS[method], linewidth=2,
                )
                axes[1, column].plot(
                    sample_sizes, runtime, marker="o", label=LABELS[method],
                    color=COLORS[method], linewidth=2,
                )
        axes[0, column].set_title(title)
        axes[0, column].set_ylim(-0.03, 1.05)
        axes[0, column].grid(alpha=0.25)
        axes[1, column].set_yscale("log")
        axes[1, column].grid(alpha=0.25, which="both")
        axes[1, column].set_xlabel("$N=M$")
    axes[0, 0].set_ylabel("Power")
    axes[1, 0].set_ylabel("Runtime (ms, log)")
    axes[0, 2].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, output / "appendix_me_scf_common_power_runtime")


def plot_axis(data: Path, output: Path) -> None:
    rows = read_rows(data / "scf_axis_ablation_d_s_delta_r20.csv")
    rows += read_rows(data / "scf_axis_ablation_mmd_d_s_delta_r20.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for axis, active in zip(axes, (2, 4, 8)):
        for method in METHODS:
            values = sorted(
                (float(row["delta_param"]), float(row["power"]))
                for row in rows
                if int(row["d"]) == 20
                and int(row["sparse_dims"]) == active
                and row["method"] == method
            )
            if values:
                delta, power = zip(*values)
                axis.plot(
                    delta, power, marker="o", label=LABELS[method],
                    color=COLORS[method], linewidth=2,
                )
        axis.set_title(f"active $s={active}$")
        axis.set_xlabel("$\\delta$")
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("Power")
    axes[2].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, output / "appendix_axis_gmm_ablation")


def plot_rotation(data: Path, output: Path) -> None:
    rows = read_rows(data / "rotation_sparse_scale_d20_N500_sigma15_with_pooled_pca_power.csv")
    selected = ["RankChi2_Subspaces", "RankChi2_Subspaces_PooledPCA", "MMD"]
    labels = {
        "RankChi2_Subspaces": "Rank-sub",
        "RankChi2_Subspaces_PooledPCA": "Rank-sub + PCA",
        "MMD": "MMD",
    }
    values: defaultdict[str, dict[str, float]] = defaultdict(dict)
    errors: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["method"] in selected:
            values[row["method"]][row["orientation"]] = float(row["reject_rate"])
            errors[row["method"]][row["orientation"]] = float(row["se"])

    fig, axis = plt.subplots(figsize=(6, 3.4))
    positions = list(range(len(selected)))
    width = 0.35
    original = [values[m]["original"] for m in selected]
    rotated = [values[m]["rotated"] for m in selected]
    original_error = [errors[m]["original"] for m in selected]
    rotated_error = [errors[m]["rotated"] for m in selected]
    axis.bar(
        [x - width / 2 for x in positions], original, width,
        yerr=original_error, label="Original", color="#1f77b4", capsize=3,
    )
    axis.bar(
        [x + width / 2 for x in positions], rotated, width,
        yerr=rotated_error, label="Rotated", color="#ff7f0e", capsize=3,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels([labels[m] for m in selected])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Power")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    save(fig, output / "appendix_rotation_pooled_pca")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("results/data"))
    parser.add_argument("--output", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_common(args.data, args.output)
    plot_axis(args.data, args.output)
    plot_rotation(args.data, args.output)


if __name__ == "__main__":
    main()
