#!/usr/bin/env python3
"""Plot the fully audited d=50 study; d=20 pooling remains blocked, not omitted as a result."""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["Rank_Singleton_Conservative", "Rank_Pairs_Conservative", "SCF_GaussianWindow_Median",
           "SCF_GaussianWindow_TrainGrid", "MMD_RFF_Median", "MMD_RFF_TrainGrid",
           "C2ST_Logistic_Hypergeometric", "C2ST_RBF_Hypergeometric"]
LABELS = ["Rank singleton", "Rank pairs", "SCF median", "SCF train grid", "RFF-MMD median",
          "RFF-MMD train grid", "Logistic C2ST", "RBF C2ST"]
CASES = [("alt_locshift", "raw", "Location shift"), ("alt_sparse_scale", "raw", "Sparse scale: raw"),
         ("alt_sparse_scale", "rotated", "Sparse scale: rotated"),
         ("alt_gmm_matched", "raw", "Matched-moment mixture")]
COLORS = ["#7ba6c4", "#135e93", "#80bd9e", "#218b58", "#e5bf7b", "#ce8721", "#bc9ad0", "#8246a1"]


def save(fig, stem):
    fig.savefig(ROOT / "reproduced-figures" / (stem + ".pdf"), bbox_inches="tight")
    fig.savefig(ROOT / "reproduced-figures" / (stem + ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "plot-inputs/comparator/d50.json")
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    (ROOT / "reproduced-figures").mkdir(exist_ok=True)
    rates = {(r["case"], r["orientation"], r["method"]): r for r in data["rates"] if r["cohort"] == "lrz-cpu"}
    protocol = json.loads((ROOT / "snapshots/comparator/experiments/fair_comparators_v1.json").read_text())
    specs = {c["id"]: c for c in protocol["cases"]}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, (case, orient, title) in zip(axes.ravel(), CASES):
        for i, method in enumerate(METHODS):
            r = rates[(case, orient, method)]
            assert r["complete"]
            calibrated = rates[(specs[case]["matched_null"], orient, method)]["calibration_screen"]
            bar = ax.barh(i, 100 * r["rate"], color=COLORS[i], edgecolor="black", linewidth=.4,
                          xerr=np.array([[100 * (r["rate"] - r["lo"])], [100 * (r["hi"] - r["rate"])]]),
                          capsize=2)
            if not calibrated:
                bar[0].set_hatch("///")
            ax.text(100 * r["hi"] + 1.0, i, f'{100*r["rate"]:.1f}' + ("*" if not calibrated else ""), va="center", fontsize=9)
        ax.set_yticks(range(8), LABELS)
        ax.invert_yaxis()
        ax.axvline(5, color="black", linestyle=":", linewidth=.8)
        ax.set_title(title)
        ax.set_xlim(0, 75)
        ax.set_xlabel("Rejection frequency (%)")
        ax.grid(axis="x", alpha=.18)
    fig.suptitle("Complete d=50 / LRZ study: 500 alternative and 4000 matched-null repetitions", fontsize=13)
    fig.text(.5, .015, "Bars: all 8 methods. Error bars: pointwise 95% exact intervals. * / hatching: failed predeclared matched-null screen.\n"
             "64-screen / 24-primary families unchanged; d=20 aggregation pending. These are G-test rank variants, not a generator ablation.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95))
    save(fig, "comparator_d50_power")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, (case, orient, title) in zip(axes.ravel(), CASES):
        values = [rates[(case, orient, method)]["mean_total_seconds"] for method in METHODS]
        ax.barh(range(8), values, color=COLORS)
        ax.set_yticks(range(8), LABELS)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(.005, 5)
        ax.set_title(title)
        ax.set_xlabel("Mean recorded total method time (seconds, log scale)")
        ax.grid(axis="x", alpha=.18)
    fig.suptitle("d=50 / LRZ Intel Xeon Gold 6148: one allocated CPU, one numerical thread", fontsize=13)
    fig.text(.5, .015, "Includes shared preprocessing, training/selection and evaluation/calibration. Excludes archive/output overhead.\n"
             "Matched 200 training + 500 evaluation observations/class; these implementations only, not universal complexity or cross-hardware scaling.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    save(fig, "comparator_d50_runtime")

    contrasts = [c for c in data["paired_contrasts"] if c["cohort"] == "lrz-cpu"]
    fig, ax = plt.subplots(figsize=(11, 7))
    labels = []
    for i, c in enumerate(contrasts):
        eligible = c["calibration_eligible"]
        color = "#218b58" if c["controlled_size_superiority"] else "#c44236" if eligible and c["holm_pvalue"] < .05 and c["hi"] < 0 else "#666666"
        ax.errorbar(100*c["difference"], i, xerr=np.array([[100*(c["difference"]-c["lo"])], [100*(c["hi"]-c["difference"])]]),
                    fmt="o" if eligible else "x", color=color, capsize=3)
        name = dict(zip(METHODS, LABELS))[c["comparator"]]
        title = next(t for case, orient, t in CASES if (case, orient) == (c["case"], c["orientation"]))
        labels.append(title + " vs " + name + (" *" if not eligible else ""))
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_xlabel("Rank-pairs power minus comparator power (percentage points)")
    ax.set_title("All 12 d=50 primary paired contrasts; original 24-slot family retained")
    ax.grid(axis="x", alpha=.2)
    fig.text(.5, .015, "Intervals: frozen simultaneous paired bounds. Green: certified rank gain; red: eligible contrast favors comparator.\n"
             "* / cross: calibration-screen-ineligible. d=20 analysis slots unresolved, not removed from the Holm family.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, 1))
    save(fig, "comparator_d50_paired")


if __name__ == "__main__":
    main()
