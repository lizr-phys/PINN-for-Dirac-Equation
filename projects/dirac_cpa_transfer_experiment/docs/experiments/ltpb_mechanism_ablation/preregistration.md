# LTPB-CPA Mechanism Ablation Preregistration

This file records the fixed run set and decision logic used before the formal
metrics were available. It is an experiment control record, not a claim of
statistical preregistration with an external registry.

## Fixed problem and run set

- Free one-dimensional Dirac equation on `x in [0, 2*pi]`, `t in [0, 5]`.
- `m=5`, `k=2`, Case M with `eta=1`.
- Models: `ltpb_cpa`, `branch_complex`, `component_polar`,
  `rotated_ltpb_15`, and `rotated_ltpb_30`.
- Seeds: `0, 1, 2`; exactly 15 formal runs.
- Every model uses one four-output, four-hidden-layer, width-64 `tanh` MLP,
  float64/complex128 arithmetic, 1024 freshly resampled points per step,
  Adam at `1e-3`, and 5000 steps.
- The only optimized objective is the strong-form Dirac PDE residual.
- Model initialization and collocation random-number generators are separate;
  matched seeds share the same trainable initialization and collocation stream.
- No failed or unfavorable completed run is selectively repeated, and no
  hyperparameter is changed after inspecting formal results.

## Fixed core outcomes

The three core errors are `spinor_rel_l2`, `current_rel_l2`, and
`relative_phase_rmse_rad`. Means use the three seeds and spreads use population
SD (`ddof=0`). No inferential significance test is claimed for `n=3`.

Strong mechanism support requires all of the following:

1. For every core error, the LTPB mean is no greater than 0.2 times both the
   Branch-Complex and Component-Polar means.
2. At least two core errors show a clear rotation trend: Rotated-15/LTPB is at
   least 1.25, Rotated-30/Rotated-15 is at least 1.25, and at least two of three
   paired seeds increase at each rotation step.
3. For at least two core errors, Rotated-30/LTPB is at least 3.

For descriptive attribution only, Branch-Complex or Component-Polar is called
close to LTPB when all three error ratios to LTPB are at most 2 and its absolute
mean-fidelity gap is at most 0.01. This descriptive rule does not replace the
strong-support decision.

## Interpretation boundaries

- Branch-Complex close to LTPB points toward the energy basis as the dominant
  prior under this budget; it does not prove phase separation irrelevant.
- Component-Polar close to LTPB suggests a generic componentwise polar form may
  suffice for this problem. Here the component basis is only about 10.9007
  degrees from the energy basis, so that observation cannot be generalized.
- Failure of rotation to degrade the core errors prevents the claim that
  energy-channel alignment is essential.
- All five representations can express the analytic single-mode solution.
  Error differences diagnose carrier preconditioning and optimization burden,
  not mathematical non-representability.
- Conclusions are restricted to the free, periodic, single-momentum Case M
  problem with a known eigenspinor basis; they do not establish a general Dirac
  solver or novelty priority.

The executable thresholds are defined once in
`src/ltpb_mechanism_decision.py`; the figure evidence contract is recorded in
`LTPB_MECHANISM_ABLATION_FIGURE_CONTRACT.md`.
