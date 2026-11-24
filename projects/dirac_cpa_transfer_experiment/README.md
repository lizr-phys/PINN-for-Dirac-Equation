# CPA-to-Dirac transfer experiment

This directory is a self-contained, reproducible numerical experiment for the 1+1 dimensional free Dirac equation

\[
i\partial_t\psi=(-i\sigma_x\partial_x+m\sigma_z)\psi,
\qquad x\in[-\pi,\pi],\quad t\in[0,5],
\]

with `m=5`, `k=2`, `float64`, and periodic features `[sin(x), cos(x), t/T]`. It compares:

1. `vanilla`: a direct two-component complex-spinor PINN;
2. `sp_cpa`: one shared phase plus an arbitrary complex spinor amplitude;
3. `sp_cpa_conservative`: `sp_cpa` plus local continuity, global norm, and global energy losses;
4. `two_branch`: a diagnostic oracle with the exact positive/negative-energy carriers hard-coded.

The code never imports from or writes to `../dirac_pinn_simple`.

## Interpretation boundary

Because the SP-CPA amplitude is an unrestricted complex two-spinor, `A exp(iS)` can still represent any spinor. The experiment therefore tests a finite-budget optimization/preconditioning claim: a single phase cannot make both `exp(-iEt)` and `exp(+iEt)` slow at once. It does **not** test or claim mathematical non-representability.

For the mixed case, an arbitrary additional phase on one energy branch leaves density, continuity, norm, and energy unchanged while altering current and spinor relative phase. `python -m src.validate_analytic` saves a direct numerical demonstration in `results/raw/analytic_validation.json`.

## Environment

The recorded experiment used:

- Python 3.13.7;
- PyTorch 2.12.0 CPU build;
- NumPy 2.3.5, SciPy 1.16.3, pandas 2.3.3, Matplotlib 3.10.8;
- AMD Ryzen 7 4700U, with three independent two-thread training processes used only to reduce wall-clock time.

CUDA is selected automatically when available. On the recorded host, CUDA was unavailable. Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Fixed configuration

- network: four hidden `tanh` layers, width 64, shared trunk;
- initialization: Xavier-normal weights and zero biases for every linear layer;
- parameters: 12,996 for Vanilla/Two-Branch, 13,061 for SP-CPA variants;
- optimizer: Adam, learning rate `1e-3`, 5,000 steps;
- collocation: 1,024 newly sampled points per step;
- structure quadrature: four random time slices and 64 endpoint-free uniform x points per slice;
- main seeds: `{0,1,2}` for Vanilla, SP-CPA, and SP-CPA-Conservative;
- Two-Branch: seed `0` only, as a predeclared oracle/diagnostic minimum;
- sensitivity: Case M, seed `0`, `lambda_struct` in `{0.1,1,10}`; the identical main `lambda=1` run is reused;
- test grid: 128 endpoint-free spatial points by 201 time points;
- no L-BFGS stage was run because it was optional and the CPU-only multi-seed Adam experiment already required several hours.

Independent random generators make the collocation sequence identical across models for a given seed. Structure-time samples use a separate generator so adding a structure loss does not change later collocation points.

## Loss definitions

All models use the same mean squared complex Dirac residual

\[
R_D=i\psi_t+i\sigma_x\psi_x-m\sigma_z\psi,
\qquad L_{PDE}=\operatorname{mean}|R_D|^2.
\]

For the conservative model,

\[
L_{cont}=\operatorname{mean}\left[\frac{\rho_t+j_x}{E\rho_0}\right]^2,
\quad
L_N=\operatorname{mean}[N(t)-1]^2,
\quad
L_E=\operatorname{mean}\left[\frac{E_D(t)-E_D(0)}{E}\right]^2,
\]

where `rho0=1/(2*pi)`. Thus `L_struct=L_cont+L_N+L_E` and `L=L_PDE+lambda_struct L_struct`. The spectral energy `E=sqrt(29)` is used as the energy scale even when the mixed-state total energy is zero.

## Run sequence

From this directory:

```powershell
python -m pytest -q
python -m src.validate_analytic
python -m src.run_experiments --mode smoke
python -m src.run_experiments --mode main
python -m src.run_experiments --mode sensitivity
python -m src.run_experiments --mode aggregate
python -m src.plot_results
```

Completed runs are skipped unless `--force` is supplied. Filters such as `--only-case M`, `--only-model sp_cpa`, and `--only-seed 1` allow exact run-level recovery. The recorded parallel execution used these equivalent commands:

```powershell
python -m src.run_experiments --mode main --only-case P --no-aggregate --cpu-threads 2
python -m src.run_experiments --mode main --only-case M --no-aggregate --cpu-threads 2
python -m src.run_experiments --mode sensitivity --no-aggregate --cpu-threads 2
python -m src.run_experiments --mode aggregate
```

## Outputs

- `results/raw/analytic_validation.json`: analytic/autodiff validation and deliberate wrong-phase counterexample;
- `results/raw/smoke/`: four 200-step smoke runs;
- `results/raw/main/`: per-seed configs, checkpoints, histories, metrics, dense grids, and time series;
- `results/raw/sensitivity/`: fixed-seed lambda runs;
- `results/summary.csv`: every formal run, including the reused sensitivity `lambda=1` row;
- `results/summary_aggregated.csv`: Case/model means and population standard deviations;
- `results/figures/`: SVG/PDF/PNG versions of all required plots plus a manifest;
- `EXPERIMENT_REPORT.md`: equations, complete per-seed results, aggregate results, conclusions, limitations, and next steps.

The figure logic, seed definitions, error-bar convention, and reviewer risks were fixed before plotting in `FIGURE_CONTRACT.md`.

## Learned two-phase pilot extension

The independent `ltpb_cpa` extension replaces the oracle's hard-coded
`exp(-iEt)` / `exp(+iEt)` carriers with two MLP-predicted real phases while
retaining the eigenspinor basis computed from the known free Dirac symbol.  It
uses a four-output 4 x 64 network (12,996 parameters) and the same PDE-only
training budget, random sources, and dense test grid as the original runs.

```powershell
python -m pytest -q
python -m src.run_learned_two_phase_pilot --mode smoke
python -m src.run_learned_two_phase_pilot --mode full --no-aggregate --cpu-threads 2
python -m src.run_learned_two_phase_pilot --mode aggregate
python -m src.plot_learned_two_phase_pilot --check-only
python -m src.plot_learned_two_phase_pilot
python -m src.build_learned_two_phase_pilot_report
python -m src.audit_learned_two_phase_pilot
```

Formal per-run artifacts are isolated under
`results/raw/learned_two_phase_pilot/`; figures are isolated under
`results/figures/learned_two_phase_pilot/`.  Dedicated run, aggregate, and
comparison CSVs use the `learned_two_phase_pilot` prefix, so the original
`results/summary*.csv`, `results/raw/main/`, `results/raw/sensitivity/`, and
`EXPERIMENT_REPORT.md` remain the original experiment's outputs.

## LTPB mechanism-attribution ablation

The independent mechanism ablation retrains Case M on the explicitly requested
periodic interval `[0, 2*pi]` and compares five matched four-output networks:
LTPB-CPA, Branch-Complex, Component-Polar, and Rotated-LTPB at 15 and 30
degrees.  Every formal run is PDE-only; exact energies and analytic carriers
are used only after training for evaluation.  Model-initialization and
collocation RNGs are separate, so the five models share both the trainable
initial state and collocation stream within each seed.

```powershell
python -m pytest -q
python -m src.run_ltpb_mechanism_ablation --mode smoke --smoke-steps 20 --cpu-threads 2
python -m src.run_ltpb_mechanism_ablation --mode full --no-aggregate --cpu-threads 2
python -m src.run_ltpb_mechanism_ablation --mode aggregate
python -m src.plot_ltpb_mechanism_ablation --check-only
python -m src.plot_ltpb_mechanism_ablation
python -m src.build_ltpb_mechanism_ablation_report
python -m src.audit_ltpb_mechanism_ablation
```

Formal raw artifacts are isolated under
`results/raw/ltpb_mechanism_ablation/`; smoke artifacts use the separate
`results/raw/ltpb_mechanism_ablation_smoke/` root, and all seven figure groups
plus source CSVs are written to
`results/figures/ltpb_mechanism_ablation/`.  The fixed run set and decision
thresholds are recorded in `LTPB_MECHANISM_ABLATION_PREREGISTRATION.md`, while
`LTPB_MECHANISM_ABLATION_FIGURE_CONTRACT.md` fixes the evidence hierarchy.
`results/ltpb_mechanism_ablation_protected_baseline.json` freezes per-file
SHA-256 values for all prior raw results, figures, summaries, audits, and the
two earlier reports; the final audit rejects any protected change.
