# Preparation verification (October 7, 2026)

This is a local preparation record, not an upload or submission-readiness certificate.

- Locked clean `uv sync --locked --extra dev --extra plots --python 3.12`: passed on macOS/ARM64, Python3.12.13. This is a portable test environment, not the original Linux worker cohort.
- Source regression suites: **79 comparator tests + 29 calibration tests passed**, with one BLAS/OpenMP thread. The initial unthreaded invocation failed an expected thread-cohort guard; the documented command and CI environment were corrected, not the scientific test.
- Frozen source/configuration parity: **98 files byte-identical** to the two recorded base sources and three scientific overlays.
- Five full raw release assets verified: **203000 calibration rows, 288000 comparator rows, 36000 generator groups, 99000 axis calls, 46000 dependence blocks**; **878000 method results** in total. All **359 published rate groups** and **125 published paired contrasts** match. Seeds, actual fixed permutation budgets/tails where archived, pointwise exact intervals, null simultaneous bounds, paired discordances/exact p-values and original Holm family sizes checked. d20 comparator remains descriptive; no statistical guard was relaxed.
- Seven intentionally bad raw fixtures (missing/duplicate coverage, corrupted data/method seed, decision, budget, nonfinite score): rejected.
- Full-shape smoke execution of every public driver: passed; calibration5 blocks, comparator12 blocks/128 rows, generator12 blocks/16 groups, axis12 blocks/72 calls, dependence22 blocks/64 method results. Diagnostic repetition only, not fresh paper power estimates.
- Plotting: **18 empirical PDF+PNG pairs**, including comparator diagnostics, and **two analytic PDF+PNG pairs** regenerated. The **16 original manuscript figure PDFs** are included. The Gaussian-copula analytic PDF is a supplied reference asset, with original plotting source unavailable; no source-regeneration claim for it.
- Public payload SHA256/allowlist checks and automated private-path/token scan pass. Original scientific source and author manuscript checkouts are not changed. Manual security/licensing/policy review remains required.

Engineering errors found and fixed during preparation: root analyzer/protocol dependencies missing from the first staged layout; streaming tar text-wrapper incompatibility; missing calibration plot-audit input; duplicate destinations in the first upload TSV. Earlier diagnostic staging/logs are retained privately, not part of the public payload.

No full fresh campaign, GPU/cluster allocation, LaTeX compilation, GitHub remote creation, push, paper submission or arXiv upload was performed. Final paper binding, owner, new-file licensing and release URL are still human decisions. See `PUBLISHING.md`.
