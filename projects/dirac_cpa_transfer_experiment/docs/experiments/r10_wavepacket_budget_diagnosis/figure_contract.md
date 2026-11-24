# R10图表冻结口径

全部统计以五个配对祖先300–304为单位，显示所有点；均值±population SD（ddof=0），不用SEM/CI。阶段为R9 A总20000步与R10 A总26000步，不作为独立两组。原始数据不平滑、不裁掉失败种子。

预定图：

1. paired_primary_metrics：五项固定判据、每seed前后配对连线，统计同一尺度。
2. norm_balance_all_seeds：全部五seed两阶段范数漂移曲线及残差投影积分闭合误差；0.02线保留。
3. residual_decomposition：载波、包络及势的全局范数投影；原始机器精度分量与有限网格积分差均保存source。
4. dynamics_seed300：参考、R9 A、R10 A的密度/流与旋量误差，固定seed300，不换最好seed。
5. branch_trajectories：±支中心和宽度相对同一耦合参考，全五seed轨迹误差。
6. budget_and_residuals：继承/新增/累计耗时与实际残差工作量，全部seed，范数与残差变化。

每图PNG300dpi、可编辑SVG、矢量PDF、长表source CSV；共享颜色、变量单位、明确phase和总步数。检查标签边界、非空PDF文字和栅格，使用Poppler预览逐图人工可视核验，保留FIGURE_QA。任何导出修订只涉及排版，不能改变科学数据或评价规则。
