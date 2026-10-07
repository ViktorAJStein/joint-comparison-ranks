#!/usr/bin/env python3
"""Plot all observed d20 counts without claiming the environment guard is cleared."""
import json
from plot_comparator_main_results import ROOT, METHODS, LABELS, COLORS, plt


def main():
    source = ROOT / "plot-inputs/comparator/d20.json"
    data = json.loads(source.read_text())
    assert data["audited_calls"] == 144000 and not data["merger_guard_waived"]
    rates = {(r["case"],r["orientation"],r["method"]):r for r in data["rates"]}
    panels = {
        "power": [("alt_locshift","raw","Location shift"), ("alt_sparse_scale","raw","Sparse scale: raw"),
                  ("alt_sparse_scale","rotated","Sparse scale: rotated"), ("alt_gmm_matched","raw","Matched-moment mixture")],
        "null": [("null_gaussian","raw","Gaussian null: raw"), ("null_gaussian","rotated","Gaussian null: rotated"),
                 ("null_rounded","raw","Rounded null"), ("null_ar1","raw","AR1 null")]
    }
    for kind, cases in panels.items():
        fig, axes = plt.subplots(2,2,figsize=(13,9),sharex=True)
        for ax,(case,orient,title) in zip(axes.ravel(),cases):
            values = [100*rates[(case,orient,m)]["rate"] for m in METHODS]
            ax.barh(range(8),values,color=COLORS)
            ax.set_yticks(range(8),LABELS);ax.invert_yaxis()
            ax.tick_params(axis="y",labelleft=True)
            for i,value in enumerate(values):ax.text(value+.12 if kind=="null" else value+.7,i,f'{value:.3f}' if kind=="null" else f'{value:.1f}',va="center",fontsize=9)
            ax.axvline(5,color="black",linestyle=":",linewidth=.8)
            ax.set_xlim(0,8 if kind=="null" else 100)
            ax.set_title(title);ax.set_xlabel("Observed rejection frequency (%)");ax.grid(axis="x",alpha=.18)
        repetitions = 4000 if kind=="null" else 500
        fig.suptitle(f"d=20 / TU: complete descriptive counts, {repetitions} repetitions per cell",fontsize=13)
        fig.text(.5,.015,"All 40 archives included, no omitted seeds. Environment-cohort guard unresolved; no calibration-cleared superiority claim.\n"
                 "Original metadata and frozen merger preserved. Dotted line: nominal 5%; bars are observed frequencies, not confidence bounds.",ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.065,1,.95))
        for extension in ("pdf","png"):
            fig.savefig(ROOT / f"images/comparator_d20_descriptive_{kind}.{extension}",bbox_inches="tight",dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
