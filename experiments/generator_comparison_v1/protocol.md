# Prospective generator comparison v1 (2026-10-07)

User explicitly requests the proposed generator comparison on TU Berlin now.
This is separate from fair-comparators-v1, which remains numerically frozen.
The new operational ceiling is **8 inclusive CPU-core-hours**, voluntarily
imposed under the direct run request/standing research scope authorization;
not a claim that the user explicitly supplied the number8, not an extension
of the earlier80-hour or8-hour studies. Zero GPUs, no automatic retries,
no cap extension. Reserve .25h for preparation INCLUDING the300s one-core
qualification, and .5h for central auditing/analysis. Admit all main child
wall caps beforehand only if their sum plus these reserves is <=8h. If pricing
fails, stop for a larger budget rather than shrink protected primary cohorts.
Explicit TU-only user instruction supersedes distributing this study across
both clusters; both availability checks retained. Original other-cluster jobs
and unrelated SympFormer jobs remain untouched.

## Mathematical method (before decision outcomes)

Same fresh base datasets and training/evaluation construction as the earlier
comparison, but separately seeded base_seed202610071. All generators use the
same200 training +500 evaluation observations per class; training labels are
unused and the common training-only centering is retained. K_ref4,64 unique
pair views, gamma.1 on the reference only, fixed B199 independent uniform
permutations, >= ties and plus-one p-values, alpha.05. One variable: f.
No tuning of f on the evaluation outcomes.

Generators: Pearson f(t)=(t-1)^2,c_f1; centered KL f(t)=t log t-t+1,c_f2;
Freeman–Tukey f(t)=2(sqrt(t)-1)^2,c_f2; Cressie–Read lambda2,
f(t)=(t^3-3t+2)/6,c_f2. All have finite f(0); Freeman–Tukey also has finite
recession slope2. All match the existing scalar engine's actual normalizations.
Histogram q_i=(HY_i+.1)/(M+.1*m), p_i=HX_i/N, m25 for each pair;
T_f=sum_views c_f N D_f(p||q).

For each representation, one label-independent jitter field, coordinate-view
list and reference-position list are generated in exactly the frozen engine's
RNG order. The observed allocation has targets firstN and references lastM;
permutations put references in the firstM permuted slots, exactly as the
existing engine. Batched comparisons count indicators z_j>=ref_j (right ties),
linearize each pair as5*r1+r2 and use allocation-offset bincount to construct
all observed/permuted histograms. Vectorized row arithmetic is identical to
the original scalar generator helper; the original G observation is checked
bitwise. This is implementation batching, NOT a new statistical method or a
novelty claim requiring a different testing reference.

Standalone timing samples run the ENTIRE new test with one generator at a time
on the first32 decision blocks per stratum in randomized generator order.
Shared multi-generator calculation cost is separately recorded; do NOT present
its four shared timings as measured standalone test latency. The standalone
routine includes the common frozen-G observation/check overhead for every
method and uses the same batched engine, source and precision for all four.

## Decision cohort and hypotheses

Dimensions20 and50, six base cases with Gaussian raw/rotated, rounded and AR1
null representations plus AR1 mean shift, raw/rotated sparse scale and
matched-moment GMM alternatives; exact JSON definitions unchanged. R4000
per null representation, R500 per alternative representation. 27000 base
blocks,36000 representation groups,144000 generator results. Pilot rep IDs
100000/100001 are excluded from decision evidence. Timing subset32 per stratum
is fixed before outcomes, not selected on power or runtime.

Hypothesis: different generator growth may yield different finite-sample
sensitivity to concentrated versus distributed histogram discrepancies; no
universal winner expected or promised. The common local-quadratic oracle
limit bounds expectations for large gains near equality. Confirmation means
nonzero paired differences supported after Holm and resolved matching null
screens; no supported differences is negative/inconclusive evidence for
adaptive gains, not an implementation failure. Similar performance can
illustrate interchangeability but does not prove practical improvement.

32 predeclared Bonferroni null screens (2d x4null representations x4generators):
no adjusted over-rejection versus alpha.05 AND simultaneous one-sided upper
size bound<=.065 required to resolve a method's matched null. 48 primary
paired two-sided exact McNemar contrasts (2d x4alternative representations
x6pairs), Holm family .05, including all predeclared comparisons when missing.
Pointwise exact binomial intervals for frequencies; paired differences and
standard errors reported with clearly labeled pointwise normal intervals.
Incomplete/failed/warning groups do not produce survivor-only rates; retain
NA strata and failures, never outcome-driven retries/reseeding/cohort changes.

All four methods stay within each paired block on one CPU, same EPYC7302
software/precision cohort. Full Linux kernels and Python build/packages are
retained; prospective admissible TU kernel list is in JSON. Kernel-stratified
secondary effects/timings are reported, not smaller substitute primary groups.
This policy DOES NOT relax the earlier campaign's frozen exact-platform guard.

## Runtime-budget winners

Winners can be legitimate descriptive summaries of a predeclared eligible
set under a fixed measured runtime budget, with uncertainty, calibrated size,
full tables and explicit selection rule. They are not direct causal evidence
about f if N/K/views/model budgets also change. This study holds those fixed;
report all four generator power/runtime points and their descriptive Pareto
frontier, rather than treating a noisy empirical maximum as proven superiority.
Any adaptive f-selection method needs separate training/selection calibration.

## Tiered checks

Tier1: `scripts/verify_generator_comparison.py`840 exact checks pass, including
Gaussian/rounded/constant/unequal-size/large-value ties, four frozen scalar
engines, all observed scores, decisions and permutation budgets, standalone/
shared equality and vectorized scalar-row arithmetic.
Tier2: local full shapes32 groups/128 results,0 errors;24.19s INCLUDING four
standalone timing passes per group. This is not cluster pricing or power evidence.
Next: one CPU/4GiB/TU smpshort EPYC7302/5-minute qualification; main admission
only after actual assigned-worker qualification and cap accounting.
