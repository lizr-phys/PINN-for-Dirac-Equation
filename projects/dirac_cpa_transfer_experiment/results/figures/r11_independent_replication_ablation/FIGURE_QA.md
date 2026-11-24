# R11 FIGURE_QA

Status: PASS

Python/matplotlib; SVG editable text; PDF TrueType/selectable text; PNG 300dpi.
All fresh paired seeds400–404; dynamics seed400; mean ± population SD(ddof=0).
E* uses analytic prior and is diagnostic only. All fixed checkpoints retained.
No smoothing, prediction renormalization, data modification or seed omission.

- primary_metrics_final: bounding boxes=True, SVG text=True, PDF text=True; source CSV retained.
- primary_metrics_coupled_20000: bounding boxes=True, SVG text=True, PDF text=True; source CSV retained.
- paired_ablation_differences: bounding boxes=True, SVG text=True, PDF text=True; source CSV retained.
- error_vs_total_cost: bounding boxes=True, SVG text=True, PDF text=True; source CSV retained.
- branch_trajectories_all_seeds: bounding boxes=True, SVG text=True, PDF text=True; source CSV retained.
- norm_balance_all_seeds: bounding boxes=True, SVG text=True, PDF text=True; source CSV retained.
- complex_carrier_residual_diagnostics: bounding boxes=True, SVG text=True, PDF text=True; source CSV retained.
- modal_branch_occupations_all_seeds: bounding boxes=True, SVG text=True, PDF text=True; source CSV retained.
- dynamics_seed400: bounding boxes=True, SVG text=True, PDF text=True; source CSV retained.

Visual inspection: 逐张查看全部9份原生PNG及9份Poppler独立PDF预览。文字、坐标、阈值、均值与population SD、种子400–404、参考及失败轨迹完整可见；E为解析诊断，动力学固定seed400并使用统一色范围。成本图和范数图的图例位置已修正，初版完整保存于qa_history；10份source CSV逐文件SHA一致，其余7份PNG逐字节未改变。未发现文字裁切、曲线被图例遮挡或PDF字体异常。
