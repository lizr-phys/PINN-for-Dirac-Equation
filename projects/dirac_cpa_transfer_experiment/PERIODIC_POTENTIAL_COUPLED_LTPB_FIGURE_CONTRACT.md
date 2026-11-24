# Periodic-potential coupled-LTPB figure evidence contract

Status: **frozen before formal learned-model results**.  The contract applies to
`periodic_potential_coupled_ltpb` only.  All plots are generated with Python
from saved CSV/NPZ evidence; every figure is exported as PNG, editable SVG and
PDF, and every PDF is rendered for visual QA.  Population standard deviations
use `ddof=0`, matching the preregistered five-seed population summary.

## Claim and evidence chain

The only primary claim under test is whether the existing shared-spectral LTPB
transfers from uncoupled free Fourier channels to dynamically coupled momentum
channels.  Evidence is ordered as follows:

1. establish a nontrivial, converged reference problem without consulting a
   learned model;
2. show all five paired seeds for the preregistered primary errors;
3. inspect the fixed seed-0 dynamics rather than selecting a visually favorable
   run;
4. expose branch activation, phase-validity masks and coherences so that a
   failure can be distinguished from a shared optimization failure;
5. report time, memory, parameter count and loss alongside accuracy.

No panel may introduce a metric, threshold, seed-selection rule or physical
parameter that is absent from the preregistration and saved source data.

## Figure inventory

| ID | File stem | Archetype | Required evidence | Intended conclusion |
|---|---|---|---|---|
| F1 | `calibration_modal_transfer` | small multiples and threshold table | all candidate potentials at T=5 and, only if invoked, T=8; sideband transfer, boundary occupation, J=4/J=6 diagnostics | the selected physics is measurable and its reference limitations are explicit |
| F2 | `all_seed_primary_metrics` | log-scale grouped dot plot with mean and population SD | all five seeds for spinor, current, modal-population error and `1-fidelity`; all three learned models | absolute transfer and paired comparative evidence are visible without aggregation hiding seeds |
| F3 | `modal_occupation_heatmaps_seed0` | aligned heatmaps | seed-0 reference, LTPB prediction and absolute error for every ladder mode and dense test time | whether coupled momentum transfer is reproduced |
| F4 | `center_sideband_trajectories_seed0` | time-series small multiples | reference and seed-0 predictions at k0 and k0 plus/minus q | accuracy on the directly generated sidebands |
| F5 | `energy_branch_occupations_seed0` | energy-branch heatmaps | reference and seed-0 LTPB P(j,+), P(j,-), plus the saved validity mask | where free-basis branch phases are physically identifiable |
| F6 | `density_current_spacetime_seed0` | two-row reference/prediction/error heatmap matrix | seed-0 physical density and current fields | whether observable physical-space dynamics, especially current, transfers |
| F7 | `active_channel_diagnostics_seed0` | masked amplitude/coherence time series | reference and prediction on active channels only, validity masks, coherence magnitude/phase diagnostics and t=0 sideband tangent evidence | whether failure is associated with activation from zero amplitude rather than unoccupied-phase scoring |
| F8 | `training_efficiency_comparison` | loss curves plus resource dot plots | every seed's loss history, training time, peak RSS increment and actual parameter count | optimization/resource context for interpretation |

## Visual and statistical rules

- Model colors remain fixed across figures; reference is neutral dark gray.
- Error axes that span orders of magnitude use a logarithmic scale with positive
  clipping only for display; the unclipped source values remain in CSV.
- Means never replace seed points.  Error bars are mean plus/minus population SD.
- All heatmaps share documented color limits within a comparison row.  Error
  heatmaps use nonnegative sequential scales; signed current uses a diverging
  scale centered on zero.
- Phase/coherence marks are omitted wherever the saved reference occupation mask
  is false (`P(j,branch) < 1e-6`).  Missing phase is not encoded as zero.
- All spatiotemporal and trajectory panels use seed 0.  No best-seed panel is
  permitted.
- Source data live under
  `results/figures/periodic_potential_coupled_ltpb/source_data/` with one or
  more descriptively named CSV files per figure.

## QA acceptance

Each PNG and rendered PDF must have nonzero dimensions, legible labels, no
cropping, no empty axes, no accidental transparency and the expected panel
count.  SVG and PDF text must remain editable.  `FIGURE_QA.md` records the
rendered-page inspection and the machine checks; a failed figure blocks the
final artifact audit.
