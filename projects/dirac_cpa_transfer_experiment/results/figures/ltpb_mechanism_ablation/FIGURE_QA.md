# LTPB mechanism-ablation figure QA

检查日期：2026-08-31  
检查对象：七组 600 dpi PNG，以及同组 PDF/SVG 导出合同。

## 逐图目视检查

1. `01_core_error_comparison`：三项核心误差均采用对数轴；全部 15 个 seed 点和 mean ± population SD 可辨；LTPB 与其余模型的数量级差异没有被线性轴压缩。
2. `02_per_seed_summary`：同 seed 配对比值热图和 mean ± population SD 数值表均完整；色条、行列标签和 `1-F` 定义可读。
3. `03_current_x0`：解析 current、三个 seed、模型均值和 population SD 均存在；LTPB 的持续振荡与其余模型的快速衰减清晰可辨。
4. `04_relative_phase_x0`：解析相对相位与按圆周差对齐后的预测曲线均存在；图例明确 n=3 且不作推断性检验；跨 seed 的整数 `2π` 显示平移不改变源数据中的圆周误差。
5. `05_rotation_degradation`：预注册的 0°/15°/30° 路径、三个 seed、均值、3×LTPB 阈值均可辨；Component-Polar 以精确有效角 `-10.9007°` 的特例显示，未混入旋转趋势判据。
6. `06_amplitude_phase_complexity`：预测 seed、mean ± population SD 和同一基底中的解析表示负担使用不同标记；三个复杂度量及归一化单位清楚。
7. `07_loss_histories`：五个模型的三个 seed、均值和 population SD 均可辨；对数轴上当 `mean-SD <= 0` 时不绘制该阴影下臂，避免把非正对称区间压到绘图下界形成误导性竖带；所有原始 seed loss 仍保留。

## 结论

未发现裁切、重叠遮挡、不可读标签、颜色语义冲突或与源数据相矛盾的视觉编码。所有图均使用白底；PNG 用于目视与栅格交付，PDF/SVG 用于矢量交付，SVG 文本保持可编辑。最终机器审计另行检查像素尺寸、600 dpi 元数据、PDF 可提取文本、SVG `<text>` 以及七份源 CSV 的严格重算。
