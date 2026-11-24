# R7 图表证据契约

后端固定 Python/matplotlib，三种表示固定灰/蓝/橙系列，ordinary/causal 用标记或线型区分。结论由结果决定，契约只规定需呈现的证据，不预设成功。

| 图标识 | 要回答的问题与面板证据 | 数据/种子 |
| --- | --- | --- |
| all_seed_primary_metrics | 六配置是否达到绝对标准；spinor、current、fidelity、占据及 norm 全种子与 mean±population SD、门槛 | 主30run，n=5，不隐藏失败 |
| checkpoint_accuracy | 1000/2500/5000 的固定检查点如何演化，不挑最优 | 每个 seed 和配置 |
| initial_tangent_and_short_dynamics | 能表示虚方向是否转化为正确早期侧带；初始导数、参数敏感度、实虚系数/占据 | 短9run全部统计，轨迹 seed0，保留很小正时间 |
| modal_occupation_heatmaps | 中心外模态是否真正生成；reference/prediction/abs error | 主六配置 seed0，共用色标 |
| sideband_trajectories | 中心和 ±1 模态轨迹与 norm 是否正确 | 六配置 seed0，不选最佳种子 |
| coherence_current | 有效相干量、覆盖率与代表位置流是否恢复 | 全种子相干统计与 seed0 流，mask见原数据 |
| residual_weights | 小加权 loss 是否掩盖后段误差；训练八段 loss/weight、dense分段未加权残差、积分误差估计 | 全种子训练源表，seed0分段轨迹 |
| accuracy_cost | 普通/因果精度、wall time、RSS、参数量和实际残差评价量 | 主30run全部种子 |
| factor_interactions | 相位修正和因果工具的配对差与交互是否可区分 | 五个配对 seed，mean±population SD，无显著性夸大 |
| long_time_validation | 条件触发时 T8 的绝对精度和侧带是否通过 | 若启动，全15run和seed0轨迹；未启动明确说明 |

图型采用定量网格、热图与轨迹组合，面板分别承担独立证据任务。统计图显示全部种子，动态单run固定seed0。数字以 CSV 为准，不用训练 loss 单独定成功。

PNG/PDF/SVG 三格式，SVG 文本可编辑、PDF TrueType；白背景、克制配色、统一标签与 SI/自然单位约定，字号适合报告阅读。各图保存 source CSV 和 manifest，记录源路径/摘要、样本量和误差条含义。检查文件齐全、矢量标签、绘图区/图例裁切、色标与对照范围，目视 QA 记录在 FIGURE_QA.md。禁用隐藏 seed、错用标准误或事后换代表 seed。
