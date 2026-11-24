# Learned two-phase pilot figure contract

Core conclusion: Under the matched 5,000-step PDE-only budget, test whether
LTPB-CPA learns the two opposite temporal phase slopes and jointly restores
Case M spinor, probability current, component relative phase, and fidelity
without degrading Case P.

Figure archetype: Quantitative grid with one primary metric comparison and
time-resolved mechanism/validation panels.

Target/output: Reproducibility report; Python/matplotlib only; white
background; editable SVG and PDF plus 600 dpi PNG.

Final size: Double-column, 183 mm wide; 5--7 pt text and 8 pt bold panel labels.

Panel map:

- `01_main_metric_comparison`: Case P/M seed distributions and mean plus
  population SD for spinor, density, current, infidelity, and component
  relative-phase error; SP-CPA and LTPB use three seeds, oracle is marked n=1.
- `02_learned_branch_phases`: spatial-mean `S_+` and `S_-` against the exact
  evaluation curves, with full-grid OLS slopes.  Case P `S_-` is explicitly
  non-identifiable and is not success evidence.
- `03_case_M_current_x0`: the mixed-branch `2E` probability-current
  oscillation, with exact reference, learned-model mean plus population SD,
  and the one-seed oracle.
- `04_case_M_relative_phase_x0`: the spinor-component phase
  `arg(conj(psi_1) psi_2)`, aligned through circular residuals before
  unwrapping.
- `05_amplitude_and_occupations`: real-amplitude drift and positive/negative
  energy branch occupations, separating phase learning from amplitude
  compensation.
- `06_loss_histories`: PDE-only optimization histories, shown as a diagnostic
  rather than an accuracy criterion.

Evidence hierarchy:

- Hero evidence: Case M spinor/current/relative-phase/fidelity comparison.
- Mechanism evidence: learned `dS_+/dt`, `dS_-/dt` and phase trajectories.
- Validation evidence: Case P compatibility, current/phase time traces, and
  amplitude/occupation stability.
- Control: the exact-carrier Two-Branch oracle is shown only as a diagnostic
  upper bound and always labelled n=1.

Statistics: LTPB-CPA and SP-CPA use independent seeds `{0,1,2}`.  Lines and
error bars show the arithmetic mean and population standard deviation
(`ddof=0`); no hypothesis test is claimed.  The oracle has one pre-registered
seed and no uncertainty bar.

Source data: the dedicated pilot comparison CSV, each run's `metrics.json`,
`time_series.csv`, and `loss_history.csv`; plotting exports seven clean source
CSVs beside the figures.

Image integrity: no selective clipping or smoothing.  Strictly positive
display floors are used only on logarithmic axes and never alter saved source
metrics.  Wrapped phase is aligned by a documented circular-residual procedure
before unwrapping.

Reviewer risks:

- LTPB uses the known eigenspinor basis `u_+`, `u_-`; it does not discover the
  branches without prior physics.
- Its forward pass does not use `E`, but exact `E` is used after training for
  evaluation and fitted-slope targets.
- The signed real-amplitude representation has a discrete pi phase/sign
  equivalence; raw and circular phase diagnostics must be read together.
- Case P contains no negative branch, so `S_-` is not identifiable.
- One free, single-k plane wave cannot establish performance for wave packets,
  external potentials, open boundaries, or higher dimensions.
