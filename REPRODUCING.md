# Experiment-to-artifact map and exact rerun commands

Run commands at repository root after `uv sync --locked --extra dev --extra plots --python 3.12`. One BLAS/OpenMP thread per process is enforced by the public driver. CPU-only, float64. Output directories must not exist. The driver uses unchanged scientific source/protocols; new metadata explicitly identifies portable reruns.

| Paper topic / source fragment | Source snapshot and protocol | Full calls/results | Released result directory |
|---|---|---:|---|
| Calibration switches and coupled raw/rotated/PCA/whitened representations; `empirical_calibration_audit.tex` | calibration snapshot; `scripts/run_calibration_audit.py` BASE/VARIANTS/CASES | 203000 | `results/calibration` |
| Matched eight-method comparison; `empirical_comparator_main.tex`, `empirical_comparator_comparison.tex` | comparator snapshot; `experiments/fair_comparators_v1.json` | 288000 | `results/comparator` |
| Pearson/G/Freeman–Tukey/CR lambda=2; `empirical_generator_comparison.tex` | comparator + generator overlay/protocol | 144000 (36000 groups) | `results/generator` |
| All nine axis mixtures; `empirical_axis_main.tex`, `empirical_axis_mixture.tex` | comparator + axis overlay/protocol + generator batched routine | 99000 | `results/axis` |
| Gaussian-copula curves and corrected Clayton/Gumbel comparison; `empirical_dependence_power.tex` | comparator + dependence overlay/protocol | 144000 (46000 blocks) | `results/dependence` |

## Full commands (not executed during release preparation)

```bash
uv run --locked python scripts/run_paper.py --study calibration --mode full --confirm-full --wall-seconds 21600 --out runs/calibration
uv run --locked python scripts/run_paper.py --study comparator --mode full --confirm-full --wall-seconds 259200 --out runs/comparator
uv run --locked python scripts/run_paper.py --study generator --mode full --confirm-full --wall-seconds 43200 --out runs/generator
uv run --locked python scripts/run_paper.py --study axis --mode full --confirm-full --wall-seconds 86400 --out runs/axis
uv run --locked python scripts/run_paper.py --study dependence --mode full --confirm-full --wall-seconds 86400 --out runs/dependence
```

These are conservative wall stops, **not measured runtime promises or permission to use a cluster**. Full costs on original cohorts were approximately 5.46, 34.07, 3.60, 11.94 (axis includes pilots/failures/repair) and 5.65 CPU-core-hours (dependence includes pilots), respectively; no GPUs. Python/CPU changes may be slower. Provide several GB free disk for raw outputs. For diagnostic-only runs replace `--mode full --confirm-full` with `--mode smoke`; full dimensions/cases/B stay unchanged, only repetition coverage is diagnostic. Never use smoke output for conclusions.

To distribute independent work, `--rep-start A --rep-stop B` defines a half-open repetition range, capped by each case's protected count; for calibration only the full range or smoke is currently supported. Use disjoint ranges covering 0:4000 for all null cases and 0:500 for alternatives, not selected seeds. Keep all methods of each block on the same hardware/software cohort. Preserve each output and metadata; no automatic retries. Portable output schema retains original scientific row fields, but private admission metadata is not fabricated. To compare a new complete run with paper counts, pass its directories to `scripts/verify_raw.py --rerun-dir ...`; the checker reports discrepancies rather than altering reference data.

## Protected settings and randomness

- **Calibration:** seed6060; R8000 per four null cases, R500 sparse-scale power; d20, N/M500/500,200/800,800/200; K4, requested singleton L64 (effective20), B100, alpha.05, jitter1e-12, gamma.1. `SeedSequence(6060,spawn_key=(rep,))` for balanced data and `(case_id,rep)` otherwise; method seed=data seed+17. Same field for PCA/whitening; five calibration-switch variants all execute actual loops. Original source—not current recommended defaults—defines mid-p/Wilson/antithetic arms.
- **Comparators:** d20/50; train200+evaluate500 per class; K4/L64 pairs and singleton; B199; train-only scale grid .25,.5,1,2,4 with B49/candidate; R4000/500; seed20261006. `seed_for` binds case index,d,rep,domain. C2ST conditional hypergeometric allocation, Gaussian-window SCF, biased RFF-MMD; archived unsmoothed Fourier is not that SCF. Preserve 64-screen and24-primary families even when d20 is descriptive.
- **Generator:** exact `experiments/generator_comparison_v1/protocol.json`; d20/50,N=M500,K4,L64 pairs,B199,gamma.1, R4000/500. All four generators share data/views/reference draws/jitter/permutations. 32 fixed standalone timing reps; no lambda=2/3 arm. Preserve32 size screens/48 contrasts.
- **Axis:** exact `experiments/axis_mixture_v1/protocol.json`; d10/20/50,N=M1000,K4,L128 pairs,B199,R4000/500. All3 sparse supports and separation cap retained; six matched methods,18 null screens/45 contrasts. No changed mixture separation or feature budget.
- **Dependence:** exact `experiments/dependence_power_v1/protocol.json`; B100,R4000 null/R500 alternative. Copula d4/20,N=M1000; Gaussian d2,K2,N=M250/500/1000/2000,rho.2/.4 versus independent reference; fixed-coordinate0 is a negative control, not all-coordinate marginal aggregation. MMD uses the original five-bandwidth median grid. Preserve32 size screens/20 contrasts.

## Exact evidence reconstruction versus fresh computation

`results/` contains published counts/uncertainties/paired inference, all exported tables, timing distributions and plot inputs. `verify_raw.py` independently reconstructs all count and paired-discordance inputs from complete release raw records; interval/multiplicity calculations remain auditable in the frozen `analyze*.py` files and published exports. It is not a replacement proof of calibration or the original worker input-replay audit. `plot_paper.py` regenerates the publication plots from those bound inputs; PDF byte hashes may differ due to backend/version/timestamp. For exact originally rendered graphics use the supplied `figures/` copies.

Release raw files retain scientific numeric rows unchanged, except calibration's redundant cohort label is replaced by `paper-local`. Private job/node/path/authorization metadata is omitted. `environment.json` retains hardware/software/protocol/source records. Original worker raw hashes are recorded in asset provenance; calibration's public raw hash differs solely because of the disclosed cohort-label change. Internal author logs/approval receipts are not needed for scientific reproduction.

**Before publication:** compare this map against the final submitted PDF/source. This preparation binds the empirical-refresh source snapshot `8bb5c6c`, not an unseen final submission. Bind final title/authors/arXiv ID/figure numbers and release URL after verification. No paper source is compiled here.
