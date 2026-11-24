# LTPB-CPA Mechanism Ablation Figure Contract

## Core conclusion

At matched capacity, initialization, collocation sequence, and PDE-only budget, determine whether LTPB-CPA's Case M accuracy specifically requires one independently learned phase per energy-aligned channel, rather than merely an eigenspinor basis or a generic two-channel polar representation.

## Figure archetype and export contract

- Archetype: quantitative grid with a dominant core-error comparison and subordinate mechanism/time-series panels.
- Backend: Python only (`matplotlib`).
- Final width: 183 mm; white background; restrained, model-consistent palette.
- Exports: 600 dpi PNG plus editable-text PDF and SVG for every figure group.
- Statistics: three independent seeds; center is the arithmetic mean; spread is population SD (`ddof=0`); all seed points remain visible; no inferential test is claimed for `n=3`.
- Source data: every quantitative figure has a CSV generated from the immutable per-run artifacts.

## Pre-result decision rules

The three core errors are full-grid spinor relative L2, current relative L2, and spinor-component relative-phase RMSE.

1. LTPB dominance passes only if each LTPB mean is at most `0.2` times the corresponding Branch-Complex mean and at most `0.2` times the corresponding Component-Polar mean.
2. A core error has a clear rotation trend only if its mean at 15 degrees is at least `1.25` times the LTPB mean, its mean at 30 degrees is at least `1.25` times its 15-degree mean, and at least two of the three paired seeds increase at each angle step. At least two of the three errors must pass this two-step rule.
3. The 30-degree degradation rule passes only if at least two core-error means are at least `3` times the LTPB means.
4. Strong mechanism support requires all three rules above. These thresholds are not changed after training.
5. For descriptive attribution only, Branch-Complex or Component-Polar is called close to LTPB only if all three core-error ratios are at most `2` and its mean-fidelity difference from LTPB is at most `0.01`.

## Panel map and evidence hierarchy

1. `01_core_error_comparison` — hero evidence: log-scale core errors for all five models, showing every seed plus mean and population SD.
2. `02_per_seed_summary` — robustness: per-seed normalized error matrix and model-level mean/SD summary without hiding failed seeds.
3. `03_current_x0` — physical validation: predicted and exact Case M probability-current trajectories at `x=0`.
4. `04_relative_phase_x0` — phase validation: circularly aligned and unwrapped spinor-component relative-phase trajectories at `x=0`.
5. `05_rotation_degradation` — mechanism test: core-error ratios at 0, 15, and 30 degrees with the `3x` reference.
6. `06_amplitude_phase_complexity` — representation burden: internal-channel amplitude drift, phase-velocity fluctuation, and amplitude-weighted phase curvature, including the exact representation's required complexity.
7. `07_loss_histories` — optimization context only: PDE training trajectories; never used as the success criterion.

The hero evidence is the joint three-metric comparison and rotation response. Time curves validate that any scalar improvement represents restored oscillatory physics. Complexity panels distinguish intrinsic basis-induced modulation from optimization failure.

## Metric definitions for representation complexity

Internal complex channel coefficients are extracted after removing `exp(ikx)`. Their phases are unwrapped along time independently at each spatial point. Quantities involving phase use `|c_j|^2` weights so phase near a coefficient zero contributes negligibly.

- Amplitude drift: RMS change of `|c_j|` from its hard-initial value, normalized by the corresponding initial-amplitude RMS.
- Phase-slope fluctuation: amplitude-weighted RMS deviation of instantaneous phase velocity from that channel's amplitude-weighted mean velocity.
- Weighted phase curvature: amplitude-weighted RMS of the second temporal derivative of the unwrapped phase.

Predicted and exact-representation complexity are both retained; a rotated or component basis may require genuine amplitude/phase modulation even for the analytic solution.

## Reviewer risks and controls

- All five networks have four real outputs and identical trainable parameter counts.
- Model initialization and collocation generators are independent; same seed gives identical MLP parameters and the same collocation sequence across models.
- No forward pass, loss, or optimizer uses the exact energy or analytic carrier; exact quantities appear only in tests and post-training evaluation.
- Component and rotated bases have nonzero hard-initial coefficients; rotated coefficients are computed by the exact orthogonal transform.
- Training is not selectively repeated and hyperparameters are not changed after observing results.
- Conclusions remain limited to the free, single-momentum, periodic Case M problem.
