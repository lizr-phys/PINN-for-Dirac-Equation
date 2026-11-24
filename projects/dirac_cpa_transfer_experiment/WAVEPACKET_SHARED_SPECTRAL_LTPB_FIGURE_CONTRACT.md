# Wavepacket Shared-Spectral-LTPB Figure Contract

Backend: Python/matplotlib only. Archetype: quantitative evidence grid with one primary comparison as the hero figure.

## Core conclusion

The figure set must show, without relying on training loss alone, whether one shared `(k,t)` network recovers the occupied training spectrum, interpolates its learned phase slopes to held-out midpoint wavenumbers, and reconstructs the split free-Dirac wavepacket in density, current, center, and width.

## Evidence chain and planned figures

| Figure | Unique evidentiary role | Planned content |
|---|---|---|
| `primary_error_comparison` | Main learned-model ranking | Five-seed means and population SD for spinor, density, current, infidelity on log scale |
| `per_seed_results` | Robustness and paired advantage | Spinor/current/fidelity for every seed and method |
| `wavepacket_density_spacetime` | Physical reconstruction | Exact, representative Shared-Spectral-LTPB, and absolute density error over `(x,t)` |
| `wavepacket_current_spacetime` | Coherent transport | Exact, representative prediction, and current error on a shared diverging scale |
| `branch_center_width` | Wavepacket splitting | Exact/predicted positive and negative branch centers and widths with seed variability |
| `phase_slope_train_midpoint` | Continuous spectral interpolation | Learned vs exact slopes and relative errors at 33 training k and 32 unseen midpoints |
| `spectral_coefficient_errors` | Fourier and occupation fidelity | Error versus k and time, weighted occupation summaries |
| `training_curves` | Optimization evidence | PDE loss histories for all models and seeds |
| `accuracy_time_memory` | Cost-aware comparison | Error versus wall time and peak RSS increment, with parameter counts |

## Export and integrity contract

- Every quantitative panel has a clean source CSV generated from formal raw artifacts.
- Every figure is exported from Python as PNG (600 dpi), SVG with editable text, and PDF with TrueType text.
- White background; restrained method-consistent palette; no rainbow colormap; exact reference is neutral black/gray and Shared-Spectral-LTPB is the signal color.
- Error bars are five-seed population SD (`ddof=0`); all individual seeds remain visible where space permits.
- Heatmaps use identical extents and comparable color limits for exact/predicted panels; error panels are separately labeled and never share a misleading range.
- Visual QA checks raster dimensions, clipped labels, legends, panel labels, colorbars, PDF rendering, selectable PDF text, editable SVG text, and consistency with source data.
- No smoothing, selective run omission, local image adjustment, or manual data editing is permitted.

## Review risks to inspect explicitly

1. The held-out midpoint plot must label those points as interpolation probes, not periodic Fourier modes.
2. A low total-field error must not conceal a failed midpoint phase test or incorrect branch width.
3. Oracle curves must be visually marked as reference and excluded from learned ranking.
4. The log-scale primary plot must not hide zero/near-zero values or reverse the fidelity direction; plot `1-fidelity` there and report fidelity itself in text/tables.
5. Training time and memory measurements must use the same process-level definition for every model.
