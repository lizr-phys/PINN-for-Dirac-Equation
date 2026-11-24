# Learned Two-Phase Pilot Figure QA

Date: 2026-08-31

All six formal figure groups were regenerated from the validated formal-run artifacts and inspected at the exported raster resolution.

- `01_main_metric_comparison`: panel labels, log axes, method labels, seed points, and uncertainty crossbars are readable. Non-positive additive lower whiskers are explicitly omitted on the log axis rather than drawn to an arbitrary floor.
- `02_learned_branch_phases`: all four titles and annotations are visible; Case P's unidentifiable negative branch is marked; the fitted and exact Case M slopes are distinguishable in the source data and visually coincident at plotting scale.
- `03_case_M_current_x0`: exact, SP-CPA, LTPB-CPA, oracle, and uncertainty bands are visible and identified.
- `04_case_M_relative_phase_x0`: the unwrapped relative-phase convention and circular-residual alignment note are visible.
- `05_amplitude_and_occupations`: sparse circle/square markers distinguish coincident positive/negative occupation curves without shifting the data.
- `06_loss_histories`: log-scale trajectories are legible; mean-minus-SD bands are shown only where positive.

Each group has PNG (600 dpi), PDF, and SVG exports. All PDFs are single-page files with extractable text, and every SVG contains editable `<text>` elements. The seven source-data CSV files and `figure_manifest.json` record the evidence, statistical summaries, and plotting conventions.
