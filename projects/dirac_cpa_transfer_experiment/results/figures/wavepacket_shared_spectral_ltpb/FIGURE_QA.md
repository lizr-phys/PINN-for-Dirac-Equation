# Wavepacket Shared-Spectral-LTPB Figure QA

Overall status: **PASS**

All figures were generated exclusively with Python/matplotlib. Source-data rows, schemas and SHA-256 hashes were checked; PNG dimensions were inspected; SVG editable text was verified; PDFs were rendered with Poppler and checked for extractable text with pdftotext or pypdf.

| Figure | Automated | Source rows | PDF preview | Problems |
|---|---|---:|---|---|
| `primary_error_comparison` | PASS | 100 | `E:/PINN for Quantum Physics/dirac_cpa_transfer_experiment/tmp/pdfs/wavepacket_shared_spectral_ltpb/primary_error_comparison.png` | None |
| `per_seed_results` | PASS | 75 | `E:/PINN for Quantum Physics/dirac_cpa_transfer_experiment/tmp/pdfs/wavepacket_shared_spectral_ltpb/per_seed_results.png` | None |
| `wavepacket_density_spacetime` | PASS | 51456 | `E:/PINN for Quantum Physics/dirac_cpa_transfer_experiment/tmp/pdfs/wavepacket_shared_spectral_ltpb/wavepacket_density_spacetime.png` | None |
| `wavepacket_current_spacetime` | PASS | 51456 | `E:/PINN for Quantum Physics/dirac_cpa_transfer_experiment/tmp/pdfs/wavepacket_shared_spectral_ltpb/wavepacket_current_spacetime.png` | None |
| `branch_center_width` | PASS | 2010 | `E:/PINN for Quantum Physics/dirac_cpa_transfer_experiment/tmp/pdfs/wavepacket_shared_spectral_ltpb/branch_center_width.png` | None |
| `phase_slope_train_midpoint` | PASS | 650 | `E:/PINN for Quantum Physics/dirac_cpa_transfer_experiment/tmp/pdfs/wavepacket_shared_spectral_ltpb/phase_slope_train_midpoint.png` | None |
| `spectral_coefficient_errors` | PASS | 66330 | `E:/PINN for Quantum Physics/dirac_cpa_transfer_experiment/tmp/pdfs/wavepacket_shared_spectral_ltpb/spectral_coefficient_errors.png` | None |
| `training_curves` | PASS | 12525 | `E:/PINN for Quantum Physics/dirac_cpa_transfer_experiment/tmp/pdfs/wavepacket_shared_spectral_ltpb/training_curves.png` | None |
| `accuracy_time_memory` | PASS | 25 | `E:/PINN for Quantum Physics/dirac_cpa_transfer_experiment/tmp/pdfs/wavepacket_shared_spectral_ltpb/accuracy_time_memory.png` | None |

## Manual visual review

PASS: every PNG and Poppler-rendered PDF preview was visually inspected at final delivery. No clipped labels, overlapping legends, broken colorbars, misleading shared scales, unreadable seed marks, or panel hierarchy defects were found.

The midpoint panels are explicitly labelled as interpolation probes rather than periodic Fourier modes. Heatmap error panels use separately labelled scales, and `1-fidelity` is used only in the log-error comparison while fidelity itself remains reported in tables.

## Post-freeze layout-only correction

Formal visual QA found that the original `per_seed_results` axes were compressed by the interaction between an axes-level external legend and the tight save boundary. The unchanged 75-row source CSV was redrawn with fixed margins and a figure-level legend. Original PNG/SVG/PDF files and the original figure manifest are retained in `pre_layout_correction/`; before/after SHA-256 values and the unchanged numerical scope are recorded in `POSTFREEZE_FIGURE_CORRECTION.json`. No run, checkpoint, metric, aggregate, decision threshold, source-data row, or frozen formal-code file changed.
