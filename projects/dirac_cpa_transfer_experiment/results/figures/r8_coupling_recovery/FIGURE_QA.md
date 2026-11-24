# R8 FIGURE QA

统计图展示全部seed；中心为均值，离散度为population SD(ddof=0)，不是SEM。正式动力学示例固定seed100。
文件、源数据与导出哈希检查不替代目视检查；最终PNG和PDF渲染记录见visual_review.md及qa/pdf_render_manifest.json。

| Figure | Integrity and actual visual review | PNG dimensions | Editable SVG texts |
| --- | --- | --- | ---: |
| 01_exploration | PASS | (3274, 1984) | 68 |
| 02_validation | PASS | (3274, 1984) | 56 |
| 03_modal_dynamics | PASS | (3184, 2315) | 195 |
| 04_branch_coherence | PASS | (3364, 2580) | 151 |
| 05_density_current | PASS | (3181, 3335) | 217 |
| 06_carrier_gradient | PASS | (3184, 1980) | 76 |
| 07_residual_bound | PASS | (3334, 1080) | 52 |
| 08_learning_cost | PASS | (3364, 2040) | 91 |
| 09_conditional_comparison | PASS | (3274, 1144) | 35 |
| 10_extensions | PASS | (3244, 2014) | 29 |
| 11_free_packet_regression | PASS | (3184, 1980) | 67 |
| 12_window_conditioning | PASS | (3244, 2044) | 57 |
