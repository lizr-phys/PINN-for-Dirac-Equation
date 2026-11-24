# R8 最终图表实际目视记录

核验日期：2026-10-02。逐幅打开最终300dpi PNG及Poppler 140dpi PDF渲染图，并核对源表与图例。此记录不是自动文件存在检查的替代品；导出/源表哈希由figure_manifest.json记录，实际PDF及渲染图哈希由qa/pdf_render_manifest.json记录。

| Figure | Actual PNG and PDF review | 核验重点 |
| --- | --- | --- |
| 01_exploration | PASS | 六配置各3个配对种子全部显示；均值±population SD，log误差棒未截断，绝对阈值与探索标识清楚。 |
| 02_validation | PASS | 四配置各5个新种子全部显示；含seed100的范数超标点，没有选取最佳种子。 |
| 03_modal_dynamics | PASS | 固定seed100；参考、A、C占据和绝对误差共用对应色标；中心衰减与第一侧带轨迹可辨。最终PDF没有伪网格接缝。 |
| 04_branch_coherence | PASS | 实/虚部及正/负支用线型区分；有效mask对应源表，原始空通道不虚构相位曲线；图例未裁切。 |
| 05_density_current | PASS | 固定seed100；密度和概率流分别保持公共范围；预测与误差均清楚，PDF无伪网格接缝。 |
| 06_carrier_gradient | PASS | 精确±E仅作评价的图例已明确；五seed载波误差点和全部梯度/分模态残差保留；轴刻度可读。 |
| 07_residual_bound | PASS | 全五seed raw场误差及未加权残差数值积分；参考线没有被称为连续认证；图例和坐标完整。 |
| 08_learning_cost | PASS | 五seed损失、时间、内存、参数和残差点数；训练步刻度间距已修正；核竞争黑边标识明确，不作公平加速结论。 |
| 09_conditional_comparison | PASS | 五个配对载波移除消融差值全部显示；零线保留，负差方向清楚；未伪造未触发救援数据。 |
| 10_extensions | PASS | 四范围算例各3新seed、两模型、旋量/概率流均显示；图只表示两项误差，V0=.4范数失败由报告完整五项判据说明。 |
| 11_free_packet_regression | PASS | 三新seed均展示；A与参考重合程度高，低范数Complex的错误几何也保留，不把其中心当成正常传播。 |
| 12_window_conditioning | PASS | 仅参考活跃、预测非零交接；对数刻度不伪造下限，局部输出坐标诊断与物理相位误差明确区分。 |

最终排版修复仅包括：统计轴根据误差棒范围选择log/symlog；相位斜率图补充“±E仅评价”图例；成本图减少重叠刻度；热图的数据层在SVG/PDF中以300dpi栅格嵌入，轴、文字和曲线保持可编辑矢量。热图完整数值仍在source CSV，可由绘图代码重新生成。没有图像修饰、数据平滑、色标择优或移除失败种子。

固定动力学种子100恰为A主验证中范数未联合通过的种子；保留这个预定示例，不以通过种子替换。
