# R11 图表契约

后端沿用已有 Python/matplotlib 科学工作流。白背景，方法色在所有图一致；PNG 300dpi及可编辑SVG、TrueType PDF，每图source CSV、统计定义与FIGURE_QA。全seed400–404显示；动力学固定400。n=5，均值与population SD(ddof=0)，不冒充SEM、置信区间或总体显著性。

1. quantitative grid：五项阈值/全seed，主终点与同耦合20000步分开，A独立复现由绝对标准决定，E明确诊断。
2. quantitative grid：A/B、A/C、A/D、E/A和C/B全配对旋量/概率流差，解释预训练、冻结、容量混合因素；实际比值另表。
3. quantitative curves：全部115个预定误差—累计残差工作量/优化耗时点；自由预训练包括在A成本中，评价和落盘独立分列。不同工作量不能混同相同步数。
4. dynamics plate：seed400参考与五组密度、概率流以及绝对误差；固定色范围，失败保持可见。
5. quantitative curves：全部seed分支中心、宽度和新增支持占据；低预测范数注明失败诊断。
6. quantitative grid：复系数、相干、载波单位缺陷、残差积分及截断外泄漏，E无载波缺陷仅浮点误差，D无载波诊断明确不适用。
7. quantitative curves：全部seed范数轨迹及残差投影数值闭合，说明范数不能唯一诊断相位或载波精度。

风险检查：不把E算学习成功；不单独归因A/C差异为冻结；不按最好seed或检查点挑图；不将中心外已有初始高斯谱称新侧带；mask只按参考；数值积分不称连续认证；成本标清实际共享硬件环境。检查文字/曲线是否裁切、图例遮挡、PDF文本可选、source CSV可追溯。
