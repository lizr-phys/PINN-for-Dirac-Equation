# Periodic-potential coupled-LTPB figure QA

Status: **PASS**

全部图由 Python/matplotlib 生成；PNG/SVG/PDF、source CSV 哈希、可编辑文字、五 seed 完整性及相位 mask 已检查。PDF 全部经 Poppler 渲染。

| Figure | Automated | PDF preview |
|---|---|---|
| calibration_modal_transfer | PASS | [preview](pdf_previews/calibration_modal_transfer.png) |
| all_seed_primary_metrics | PASS | [preview](pdf_previews/all_seed_primary_metrics.png) |
| modal_occupation_heatmaps_seed0 | PASS | [preview](pdf_previews/modal_occupation_heatmaps_seed0.png) |
| center_sideband_trajectories_seed0 | PASS | [preview](pdf_previews/center_sideband_trajectories_seed0.png) |
| energy_branch_occupations_seed0 | PASS | [preview](pdf_previews/energy_branch_occupations_seed0.png) |
| density_current_spacetime_seed0 | PASS | [preview](pdf_previews/density_current_spacetime_seed0.png) |
| active_channel_diagnostics_seed0 | PASS | [preview](pdf_previews/active_channel_diagnostics_seed0.png) |
| training_efficiency_comparison | PASS | [preview](pdf_previews/training_efficiency_comparison.png) |

视觉检查：PASS。逐幅检查 PNG 与 PDF 渲染图：文字、图例、色条和面板无遮挡或裁切；参考/预测共用色限。统计图均包含五个种子；精度–时间图中接近的点可能重合，逐 seed 数值均保存在 source CSV。

统计点为五个配对随机 seeds；中心为均值，误差棒为 population SD（ddof=0）。没有显著性检验或 p 值。动力学图固定 seed 0。

F2 的范数漂移面板使用 symlog（线性阈值 10⁻³），完整显示均值 ± SD 跨零的统计区间；负端不表示观测到负漂移。其余五个面板使用普通对数轴。

热图数据层以 400 dpi 嵌入 SVG/PDF，避免 PDF 查看器产生多边形接缝；文字、坐标轴和曲线保持可编辑矢量。未平滑、裁剪或修改原始场值，全部热图值均提供 source CSV。

F7 只在保存的有效 mask 内展示相对相位。无效相位保持缺失，不作为零误差。
