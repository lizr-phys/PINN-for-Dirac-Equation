# R9 图表契约（正式训练前冻结）

Python/matplotlib；量化网格与动力学图。目标是展示绝对成功、与容量匹配控制的实际配对差、动力学与诊断是否一致，不能预定有利结论。

1. reference_calibration：全部已执行预定截断对、两端占据、自由/外势差。参考曲线不涉及learned选择。
2. all_seed_primary_metrics：5项绝对指标，每模型5个种子均显示，mean±population SD；门槛线与逐seed联合标记。
3. paired_comparisons：A/B、A/C、C/B实际旋量/流配对差、比值，不添加优势门槛。
4. dynamics_seed300：完整参考与A/B/C的密度、概率流时空图；固定300，不更换代表seed。误差用统一跨模型色标。
5. modal_branch_seed300：总模态与正负支占据图，显示初始支持[0,32]边界；不把中心外都标为新侧带。
6. branch_center_width：参考和全部种子中心/宽度轨迹，seed300为较粗线，其余保留；低范数控制仍显示并说明诊断限制。
7. diagnostics_all_seeds：相干、复谱、未加权残差积分、泄漏与载波缺陷，所有种子均显示。
8. training_costs：全种子载波/耦合时间、评价时间、RSS、残差工作量；相同步数不解释成相同成本。

白底；A、B、C在各图颜色一致。字体可编辑、面板字母、轴单位清楚；整图宽约180–230mm，字体≥7pt（报告阅读图可9pt）。输出PNG 300dpi、SVG editable text、PDF TrueType；每幅全部数据用source CSV溯源，不删除不利种子。误差棒population SD(ddof=0)、n=5，非SEM/CI。导出前检查文字包围盒和边界，导出后视觉核验PNG及SVG/PDF存在完整，记录FIGURE_QA；不以视觉修正变更科学数据。
