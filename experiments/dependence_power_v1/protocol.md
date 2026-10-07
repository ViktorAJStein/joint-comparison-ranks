# Dependence evidence v1 — prospective protocol, 2026-10-07

## Scope and authorization

User requests rerunning the exploratory Clayton/Gumbel comparisons with usable
calibration and extending the Gaussian-copula illustration to empirical power.
This is a NEW study; historical results and numerical sources remain unchanged.
Both clusters were checked at startup and respond. Complete method blocks stay
on one hardware/software cohort; studies may be assigned to different clusters.
No old 80h/8h authorization is reused. The initial voluntarily imposed8h ceiling
rejected the12.6h inclusive full child-cap reservation. The user subsequently
explicitly approves the proposed NEW **14 CPU-core-hour total cap, zero GPUs**,
including prior pilots/preparation and analysis (reply “Yes” on2026-10-07).
Retain .25h preparation/qualification and .5h analysis reserves and the original
2xmaximum measured-block/child-cap pricing, leaving1.4h unallocated buffer.
Only this operational ceiling changes; all statistical settings/seeds/cohorts
and numerical code remain identical to the qualified version. Bind the old
pilot protocol/source and new resource-only version explicitly; do not rewrite
pilot metadata, shrink repetitions/methods/sample sizes or reuse old budgets.
No automatic retries or further extension.

## Questions, bounds and decision criteria (before outcomes)

1. Can the corrected rank-pair G statistic detect the original copula alternatives
   at controlled size, and how does it compare with the original ME, unsmoothed
   Fourier comparator and exact median-grid MMD? No universal winner assumed.
2. At d=2,K=2, does the empirical-reference Pearson rank test exhibit power against
   Gaussian-copula rho=.2/.4 with unchanged uniform marginals? A fixed coordinate-0
   rank test is the negative control, NOT a multiple-coordinate marginal aggregator.
   This is empirical evidence, NOT a proof of empirical-reference consistency or
   a verification of the oracle threshold-crossing approximation.

Maximum power is 1; at R=500 the maximum pointwise binomial SE is .02236.
Fixed-K identification and subspace blindness limitations remain. Confirmation
requires resolved matched-null size screens and multiplicity-adjusted contrasts;
otherwise report failure/inconclusive evidence, not a winner.

## Protected design

Original-comparison study: d in {4,20}, N=M=1000, K=4,L=64 requested unique pair
views, reference pseudocount .1, J=10, covariance ridge1e-6. Reference Gaussian
copula equicorrelation rho=.6, targets Clayton/Gumbel matched in Kendall tau
2*arcsin(.6)/pi, all with Beta(2,5) marginals. Nulls use both samples from each
of the three laws: Gaussian, Clayton, Gumbel. This preserves the earlier d4,N1000
slice and adds d20; it is not a resurrection of the disabled pre-fix panels.
ME locations and Fourier frequencies are pooled-label-independent, unoptimized,
median scale multiplier1. The Fourier comparator is explicitly unsmoothed, not
published analytic SCF. Source inspection found that the original runner's nominal “median MMD”
actually defaults to five equally weighted Gaussian bandwidths median*2^{j},
j=-2,-1,0,1,2. Preserve that exact fixed multiscale kernel, with no tuning,
and explicitly call it exact median-grid MMD (not single-bandwidth/RFF MMD). Label-independent choices remain fixed per test.

Gaussian illustration study: d=2,K=2,N=M in {250,500,1000,2000}; uniform independent
reference, Gaussian copula target rho in {.2,.4} with uniform marginals. One null
rho=0 per N. Full-joint Pearson has one pair view. Fixed coordinate-0 Pearson has
one singleton view. Both use the same data/jitter/reference-position/permutation
fields, so the only construction change is using joint versus marginal ranks.
Do not replace the shared-pool reference procedure with a fresh-batch oracle.

ALL tests use alpha=.05, fixed B=100 independent uniform permutations, conservative
>= ties and plus-one (1+tail)/(B+1), no mid-p/antithetic/stopping. R=4000 per null
stratum, R=500 per power stratum. Separate disjoint pilot IDs >=100000.
Predeclare source, seeds, protocol and software before execution. Archive every
observed/permuted score, tail count, decision, data/input hashes and full source
receipt; fail closed for warnings, incomplete indices or duplicate coverage.
No survivor-only rates. Do not pool ARM and cluster timings or mix old/new backends.

46,000 complete blocks: original 26,000 (104,000 method results), Gaussian20,000
(40,000 method results). Exact primary family: 12 copula contrasts (2d*2alt*3 baselines, each G versus
one baseline), and 8 Gaussian joint-vs-coordinate contrasts; total20, Holm
two-sided exact McNemar. Clayton-versus-Gumbel comparisons are NOT primary tests. Null family:
2d*3laws*4methods=24 plus4N*2methods=8, total32. Resolve with simultaneous exact
one-sided upper bound<=.065 AND no Bonferroni-significant over-rejection at .05.
Copula primary contrasts require all three corresponding distributional null
screens for both methods; Gaussian contrasts require matching independent null.
Report all frequencies with exact pointwise95% binomial intervals, paired
contrast discordances, adjusted p-values, failures and kernel/hardware-stratified
secondary effects. Timing includes all test construction/calibration; data
sampling is separately timed and counted in resource pricing.

## Derivation before implementation

For an allocation a in {0,1}^{n+m} with n ones, pooled Gram G, diagonal g,
let v=Ga, s=a^TGa, r=a^TG1, q=1^TG1. Then off-diagonal within sums are
s-a^Tg and q-2r+s-(1-a)^Tg; cross sum is r-s. The unbiased MMD is these
within sums divided by n(n-1),m(m-1), minus2(r-s)/(nm). Batched matrix
multiplication changes implementation, not estimator, kernel or calibration.
Compare against direct Gram slicing on random, tied, unequal-size and degenerate
inputs and verify permutation decisions, including exact ties.

Finite-feature quadratic score is nm/(n+m)*delta^T(S+ridge I)^-1 delta,
S=[sum_X(phi-mean_X)(phi-mean_X)^T+sum_Y(...)]/(n+m-1). Recompute S for EVERY
allocation. Rank scores use exactly the frozen positive-reference-pseudocount
histogram equations; shared role positions and jitter remain fixed across
allocations. Right-continuous self-comparisons are retained.

Prior work/statistics are already documented and source-audited in the project;
this is an evidence rerun and implementation batching, not a new method/novelty
claim. No new citations or proof scope are introduced.

## Tiered execution

Tier1: scorer/sampler/engine equivalence and negative archive/admission tests.
Tier2: all full shapes on disjoint pilot IDs, local CPU pricing (NOT a cluster
price or power result). Then cluster qualifications only with bounded authorization.
Tier3: full fixed cohorts only after source/environment/budget admission. Plot
power versus N for Gaussian and all copula methods/null sizes, PDF+PNG with source
CSV. Incorporation occurs only after raw/source/statistics checks; no manuscript
compilation, readiness claim or changes to empirical-reference proofs.
