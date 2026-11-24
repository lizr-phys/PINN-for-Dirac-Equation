# Figure QA

- Backend: Python/matplotlib exclusively for plotting, preview, export, and QA.
- Export: every figure has editable-text SVG (`svg.fonttype=none`), PDF (`pdf.fonttype=42`), and 300 dpi PNG.
- Source traceability: `figure_manifest.json` maps each figure to its source CSV/NPZ.
- Statistics: main comparison shows individual seeds plus mean and population SD; learned models use n=3, Two-Branch oracle uses n=1 and is labeled accordingly.
- Shared scales: each error-map row uses a single global logarithmic normalization across all four methods; no per-model rescaling or selective clipping is used.
- Log display floors: strictly positive floors are used only to display exact/near-zero values on log axes; source values remain unchanged.
- Visual inspection: all eight PNG previews were inspected at high resolution. Panel labels, legends, tick labels, color bars, and math text are readable with no material overlap or clipping.
- Color/accessibility: method colors are consistent across figures; markers and direct x labels provide non-color identification; no rainbow color map is used.
- Interpretation risk: the current and phase traces show seed 0 as explicitly stated; aggregate uncertainty is carried by the separate main comparison figure and raw per-seed tables.
- Integrity: no smoothing, local contrast edits, point removal, or post-hoc axis truncation was applied. Loss spikes and non-monotonic lambda behavior are retained.
