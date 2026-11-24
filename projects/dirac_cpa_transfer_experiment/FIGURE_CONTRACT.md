# Figure contract

Core conclusion: Under an identical PINN budget, determine whether the optimization advantage of a common single phase degrades for a positive/negative-energy mixture, and whether conservation losses can look successful while current and relative phase remain wrong.

Figure archetype: Quantitative grid with time-series and error-map validation panels.

Target/output: Reproducibility report; Python/matplotlib only; white background; editable SVG as the primary format, with PDF and 300 dpi PNG companions.

Final size: Double-column figures, approximately 183 mm wide; 6--8 pt text at final size.

Panel map:

- Main comparison: Case P and Case M mean +/- population standard deviation for spinor, density, current, and global phase-insensitive fidelity.
- Case M current: exact and predicted current at x=0, exposing the 2E oscillation.
- Case M relative phase: exact and predicted angle(conj(psi1) psi2) at x=0, with undefined points masked.
- Case M error maps: absolute errors in rho, j, psi1, and psi2 on the same dense x-t grid.
- Conservation diagnostics: time-resolved continuity RMS, norm drift, and absolute energy drift.
- Optimization diagnostics: total/PDE/structure loss histories and, when available, PDE-versus-structure gradient cosine.
- Sensitivity: fixed-seed Case M results for lambda_struct in {0.1, 1, 10}; lambda=1 reuses the identically configured main run.

Evidence hierarchy:

- Hero evidence: Case M spinor/current/fidelity error versus conservation errors, backed by current and relative-phase traces.
- Validation evidence: Case P comparison and the Two-Branch diagnostic oracle.
- Controls/robustness: three seeds for the three principal learned models, one predeclared Two-Branch seed, and the lambda sensitivity analysis.

Statistics:

- Independent initialization/collocation seeds are {0,1,2} for Vanilla, SP-CPA, and SP-CPA-Conservative.
- Two-Branch uses seed 0, as predeclared by the requested minimum diagnostic budget.
- Error bars are population standard deviations across available seeds; n is printed or documented for every group.
- No hypothesis test or multiple-comparison correction is used; raw per-seed values are retained.

Source data: results/summary.csv, results/summary_aggregated.csv, per-run metrics.json, loss_history.csv, time_series.csv, and dense_grid.npz.

Image integrity: Error maps use one global normalization per metric across methods; no selective clipping or local contrast adjustment. Log scales use strictly positive floors only for display and never modify source metrics.

Reviewer risks:

- A complex spinor amplitude makes SP-CPA representationally complete, so observed failure can establish optimization/preconditioning degradation but not mathematical non-representability.
- Two-Branch contains the exact +/-E carriers and is therefore an oracle/diagnostic baseline, not a proposed fair deployable method.
- Conservation laws are analytically blind to an arbitrary relative phase in this spatially homogeneous mixture; PDE loss remains the only term that fixes its time evolution.
- One-seed lambda sensitivity and Two-Branch results are diagnostic, not population-level uncertainty estimates.
