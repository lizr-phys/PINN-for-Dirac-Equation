# R9 FIGURE_QA

Status: PASS

Python/matplotlib exports; native PNG and independent Poppler-rendered PDF previews inspected.
All five seeds shown in statistical figures; seed300 fixed for dynamics.
Population SD, ddof=0. Shared heatmap scales, no smoothing or selective omission.

- reference_calibration: layout bounding boxes PASS; editable SVG PASS; PDF text PASS; source CSV complete.
- all_seed_primary_metrics: layout bounding boxes PASS; editable SVG PASS; PDF text PASS; source CSV complete.
- paired_comparisons: layout bounding boxes PASS; editable SVG PASS; PDF text PASS; source CSV complete.
- dynamics_seed300: layout bounding boxes PASS; editable SVG PASS; PDF text PASS; source CSV complete.
- modal_branch_seed300: layout bounding boxes PASS; editable SVG PASS; PDF text PASS; source CSV complete.
- branch_center_width: layout bounding boxes PASS; editable SVG PASS; PDF text PASS; source CSV complete.
- diagnostics_all_seeds: layout bounding boxes PASS; editable SVG PASS; PDF text PASS; source CSV complete.
- carrier_defects_all_seeds: layout bounding boxes PASS; editable SVG PASS; PDF text PASS; source CSV complete.
- training_costs: layout bounding boxes PASS; editable SVG PASS; PDF text PASS; source CSV complete.

Visual inspection: All nine native PNGs and independent Poppler PDF contact sheets inspected: readable labels, no clipping or overlap, all five seeds retained, shared heatmap scales, seed300 fixed. The initial automated failures were nonrendered out-of-view tick artists; the corrected checker excludes only those artists. Initial exports and QA are preserved in initial_export_snapshot.
