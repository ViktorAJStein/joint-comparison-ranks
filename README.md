# Two-Sample Testing with Joint Comparison Ranks

Research code and reproducibility material for **Two-Sample Testing with Joint Comparison Ranks**.

Repository: [ViktorAJStein/joint-comparison-ranks](https://github.com/ViktorAJStein/joint-comparison-ranks). The `reproducibility-20261007` snapshot covers the five verified synthetic studies mapped in `REPRODUCING.md`. Final submitted-paper binding, citation metadata and the arXiv identifier remain pending; this snapshot is not an `arxiv-v1` release.

## Reproduce the paper

There are two distinct workflows:

1. **Reconstruct the recorded results:** use the supplied numerical tables, complete per-repetition release assets, and plotting scripts. This preserves the paper's seeds, outcomes and recorded hardware-specific timings without rerunning ~61 CPU-core-hours of experiments.
2. **Rerun the experiments:** generate the same seeded datasets and execute the same numerical routines, using the immutable protocols. Floating-point libraries/CPU architecture can affect bitwise results; measured timings cannot be reproduced exactly on different hardware. Reruns are labelled separately and never overwrite paper evidence.

The five retained studies are calibration/PCA, matched modern comparators, generator ablation, nine coordinate-aligned mixtures, and dependence power. No external dataset or GPU is needed. CIFAR and superseded low-repetition experiments are **not part of this manuscript snapshot**. Historical entry points in frozen source are not paper reproduction commands.

### Install

Install [uv](https://docs.astral.sh/uv/), then from the repository root:

```bash
git clone https://github.com/ViktorAJStein/joint-comparison-ranks.git
cd joint-comparison-ranks
uv sync --locked --extra dev --extra plots --python 3.12
uv run --locked python scripts/check_manifest.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run --locked pytest snapshots/comparator/tests
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH=snapshots/calibration/src uv run --locked pytest snapshots/calibration/tests
```

`pyproject.toml` and `uv.lock` are copied from the frozen comparator source; do not regenerate the lock. Paper workers used Python 3.12.14, NumPy 2.5.2, SciPy 1.18.1 and scikit-learn 1.9.0. `results/*/environment.json` records actual per-study hardware/software; the local calibration cohort differs from cluster cohorts. The portable setup is tested separately, not claimed to recreate an OS/BLAS image.

### Reconstruct archived evidence

Download the five complete raw-data assets (~576 MB) from the [matching snapshot release](https://github.com/ViktorAJStein/joint-comparison-ranks/releases/tag/reproducibility-20261007). Asset filenames and SHA256 hashes are in `release-assets.json`.

```bash
mkdir -p downloads
for study in calibration comparator generator axis dependence; do
  curl --fail --location --retry 2 \
    --output "downloads/${study}-raw.tar.gz" \
    "https://github.com/ViktorAJStein/joint-comparison-ranks/releases/download/reproducibility-20261007/${study}-raw.tar.gz" || exit 1
done
uv run --locked python scripts/verify_raw.py --assets downloads
uv run --locked python scripts/plot_paper.py --out reproduced-figures
```

The raw verifier checks asset checksums, full expected keys/denominators, seed domains, independent permutation tails where archived, every published rejection count, and paired discordance counts. It refuses missing/duplicate/failed records. It does not infer validity from a favorable subset. The plotting command saves PDF+PNG from all archived empirical plot inputs and two analytic illustrations. A third analytic Gaussian-copula illustration is supplied as its original rendered PDF; its original plotting source is unavailable. Exact published TeX tables and CSVs are in `results/`; numbered figure/table references may change during final editing, so `REPRODUCING.md` maps by caption/topic and source fragment.

### Rerun the protected experiments

See [REPRODUCING.md](REPRODUCING.md) for every full command, compute/storage requirements and seed scheme. Start with a smoke test:

```bash
uv run --locked python scripts/run_paper.py --study comparator --mode smoke --out runs/comparator-smoke
```

Smoke runs retain full sample sizes/permutation budgets but use one diagnostic repetition; they are not paper results. Full runs require explicit `--confirm-full` and `--wall-seconds` and write new raw files. The public driver calls frozen numerical functions directly. It does **not** claim to inherit the authors' private cluster authorizations or to reproduce scheduler/job metadata. Original research admission/analysis code is retained separately for provenance, not as a requirement for readers to have author accounts or private approval JSON.

## Repository layout

- `snapshots/comparator/`: frozen base `8f2de62`, numerical source, tests, original analyzers and lock.
- `snapshots/calibration/`: distinct frozen calibration source `bd81b8a`; do not silently substitute newer source.
- `experiments/*_v1/`: exact overlay source and protocols for generator/axis/dependence studies.
- `scripts/`: portable rerun driver, raw verifier, all-paper plotting, manifest verification.
- `results/`: current paper numerical exports, TeX tables, plot inputs and hardware records.
- `paper-map/`: empirical source fragments used to bind the mapping; not a replacement for the final paper.
- `theory_verification/`: analytic illustrations and sanity checks (not empirical power evidence).
- `release-assets.json`: complete raw-asset filenames, checksums and sizes; assets are not committed to git.
- `UPLOAD_MANIFEST.tsv`, `manifest.sha256`: exact public file allowlist and integrity hashes.

## Interpretation and limitations

Preserve all negatives and inconclusive screens. The comparator d=20 pooled-inference guard remains unresolved; those rows are descriptive and the original 64-null/24-contrast families stay fixed. Generator and copula size screens include unresolved cells. Axis-mixture gains are coordinate-aligned/fixed-budget, not universal dominance; rotation, hard-mixture and latency limitations remain. Calibration-switch ablations include invalid/over-rejecting configurations intentionally, not recommended defaults. No empirical-reference consistency proof is asserted.

## Attribution and license

Inherited source retains its original MIT notice in `LICENSE` and `snapshots/*/LICENSE`. It descends from `josemanuel22/ranktwosample`, source commit `42bbe058fb8b25188b30b61f52b61765506cbb0c`; the separate ICML repository is not modified by this release. See `THIRD_PARTY.md` for dependencies. The inherited MIT notice is preserved. Licensing/ownership confirmation for newly added overlays, scripts and data remains a maintainer checklist item; publication does not invent that confirmation or automatically apply the inherited notice to every new file. Do not redistribute third-party papers, images, datasets or private research logs.

Citation metadata will be added from the final author-approved paper and verified arXiv record, not guessed. Please cite that paper once available.
