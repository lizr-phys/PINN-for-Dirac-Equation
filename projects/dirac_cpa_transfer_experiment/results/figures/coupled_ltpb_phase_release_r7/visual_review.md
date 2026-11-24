# R7 图表目视审查

日期：2026-10-01。最终结果：PASS（研究报告阅读尺度，不宣称已符合特定期刊最终印刷规范）。

方法：以 Python 调用 Poppler 将最终 PDF 逐图渲染为 140 dpi PNG，实际打开并检查全部 11 张图。渲染命令、PDF/PNG 哈希见 PDF 渲染记录（本地档案），预览保存在 `qa/pdf_renders/`。另外检查了最初三张 PNG 导出。文件完整性与目视检查分开记录；目视通过不表示模型科学成功。

| 图 | 目视结果 | 重点检查 |
| --- | --- | --- |
| all_seed_primary_metrics | PASS | 六配置五种子全部显示，普通圆点/因果三角；均值±population SD，绝对门槛清楚；未隐藏失败种子 |
| checkpoint_accuracy | PASS | 1000/2500/5000 固定检查点及每种子轨迹；虚线因果；不挑最佳检查点 |
| initial_tangent_and_short_dynamics | PASS | 全部三个短时种子；早期时间 symlog 含零和小正时间；参考与实/虚系数清楚，不在零占据处画普通相位误差 |
| main_initial_tangent | PASS | 全部五种子；零敏感度用 symlog 保留；切向敏感度、导数误差和最小奇异值不混同；最终梯度批次注明 |
| modal_occupation_heatmaps | PASS | seed0；离散模态 nearest 插值；参考/预测及绝对误差各有色标，共用 0–1 范围；超范围值明确标出实际最大值，不靠颜色饱和掩盖发散 |
| sideband_trajectories | PASS | 固定 seed0，中心及两个一阶侧带，保留因果 Complex 的发散；大值压缩小轨迹是共同坐标的真实结果，精确数值在源表 |
| coherence_current | PASS | 全种子相干/覆盖率；单位置流明确标注 x=0 only；覆盖率不冒充准确率；空间误差由独立全场图补充 |
| density_current_fields | PASS | seed0 修正版普通/因果；参考与预测共用色标，误差用单独标明色标；密度均匀预测与真实空间调制、流的空间相位差可辨 |
| residual_weights | PASS | 八段权重和未加权残差；黑色 raw 误差与模型色积分界分开；展示被压低权重的后段发散，不仅画小加权目标 |
| accuracy_cost | PASS | 全部种子的 wall time/RSS 与精度；所有训练轨迹及均值保留；来源表含参数量、实际采样数；不把相同步数当相同成本 |
| factor_interactions | PASS | 五个配对原始值、零线、均值±population SD；不标显著性，不将 SD 写成置信区间 |

修复的展示问题：长标题与面板字母曾重叠，面板字母移到左侧；灰色积分界曾与半透明黑误差线近似同色，改为不透明黑误差线；超 1 的值最初两位小数导致 `1.00 >1` 不清楚，改为五位有效数字。修复后重新渲染并检查了对应最终 PDF。仅改布局/线条/标注，没有改原始指标、训练权重、预算或阈值。

限制：动态实例固定 seed0，不代表最优或均值轨迹；统计比较只有五个配对种子，不是总体显著性证明。T=8 未触发训练，没有该 learned 结果图。最终 PNG/SVG/PDF 和 CSV 逐项哈希见 figure_manifest.json；本审查不能替代对原始科学数据的独立数值审计。
