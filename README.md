# PINN for Dirac Equation

Research code and experiments on physics-informed neural networks (PINNs) for the one-dimensional Dirac equation. The main question is whether learning separate phases for the positive- and negative-energy branches improves time evolution. The studies start with free plane waves and Gaussian wave packets, then examine mode coupling in a periodic potential and test changes to the learned carrier and probability normalization.

## Start here

- [Minimal free Dirac example](projects/dirac_pinn_simple/README.md) compares a PINN prediction with an analytical plane-wave solution.
- [Main time-evolution study](projects/dirac_cpa_transfer_experiment/README.md) contains the model implementations, experiment scripts, and evaluation code.
- [Research assessment through R13](docs/reviews/r1_r13_assessment_2026_10_11.md) reviews the evidence, failure cases, and limitations. Most detailed reports are in Chinese.

## Time-evolution experiments

| Round | Research question | Report |
| --- | --- | --- |
| R1–R6 | How do shared and branch-specific phases behave on free states, wave packets, and the first coupled cases? | [Combined report](projects/dirac_cpa_transfer_experiment/docs/comprehensive/report.md) |
| R7 | What changes when phase constraints on initially empty modes are relaxed? | [Phase-release experiments](projects/dirac_cpa_transfer_experiment/R7_PHASE_RELEASE_EXPERIMENT_REPORT.md) |
| R8–R9 | Can a learned free carrier and a complex envelope recover periodic-potential dynamics? | [Coupling recovery](projects/dirac_cpa_transfer_experiment/R8_COUPLING_RECOVERY_REPORT.md) · [Gaussian wave packet](projects/dirac_cpa_transfer_experiment/R9_WAVEPACKET_PERIODIC_POTENTIAL_REPORT.md) |
| R10–R11 | How do training budget, random seed, and carrier training strategy affect results? | [Training budget](projects/dirac_cpa_transfer_experiment/R10_WAVEPACKET_BUDGET_DIAGNOSIS_REPORT.md) · [Replication and ablation](projects/dirac_cpa_transfer_experiment/R11_INDEPENDENT_REPLICATION_ABLATION_REPORT.md) |
| R12 | How robust are the methods across changes in time horizon, mass, and potential? | [Robustness](projects/dirac_cpa_transfer_experiment/R12_ROBUSTNESS_REPORT.md) |
| R13 | Can a learned static frequency and a norm-preserving envelope improve the selected difficult cases? | [Recovery experiments](projects/dirac_cpa_transfer_experiment/R13_LONG_TIME_RECOVERY_REPORT.md) |
| R14 | What do independent checks and comparisons with numerical baselines establish? | [Progress and scope](projects/dirac_cpa_transfer_experiment/studies/r14_prewriting_closure_2026_10_11/progress.md) · [Report at pause](projects/dirac_cpa_transfer_experiment/studies/r14_prewriting_closure_2026_10_11/analysis_outputs/paused_work_report_20261013.md) |

## Companion projects

| Project | Focus |
| --- | --- |
| [Dirac and Klein–Gordon bound states](projects/bound_state_dirac_kg/README.md) | Stationary energies and wavefunctions in a Gaussian potential |
| [Dirac and Klein–Gordon scattering](projects/scattering_dirac_kg/README.md) | Reflection, transmission, and phase shifts in one-dimensional scattering |
| [Quantum harmonic oscillator](projects/pinn_harmonic_oscillator/README.md) | Multiple eigenstates using normalization, orthogonality, and parity constraints |

Project READMEs describe dependencies and how to run the examples. This repository includes source code, experiment protocols, reports, final figures, and selected result tables. Full training archives and model checkpoints are stored separately. See [literature notes and references](literature/README.md) for the related work.
