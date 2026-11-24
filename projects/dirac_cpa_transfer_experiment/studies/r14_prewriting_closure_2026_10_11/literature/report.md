# 原始文献查新与贡献边界（2026-10-11）

本次已重新检索 arXiv 和 CrossRef 原始API，并核对作者/出版商页面。最近更新窗口为2026-09-30至检索日，同时保留不限定年份的关键先例搜索。查询、来源响应、检索时间及SHA均保存于 [retrieval_log.json](retrieval_log.json) 和 sources；候选逐条筛选见 [screening_all_candidates.csv](screening_all_candidates.csv)。没有学术专用MCP或机构授权Scopus/WoS；宽查询有大量无关命中，未发现完整重复不构成优先权证明。

最直接的先例是 [Lorin–Yang Dirac PINN](https://arxiv.org/abs/2204.02959) 与 [SINN](https://arxiv.org/html/2408.16414v2)。前者已经处理时变Dirac旋量、硬初值及范数问题；后者直接使用(k,t)输入和谱系数输出。已下载并核查方法全文与图。故一般Dirac PINN、共享谱网络和谱乘法替代空间微分，均不能列为本项目独创。

[CPA-PINNs](https://www.sciencedirect.com/science/article/pii/S0377042726002505)、[WKB-PINNs](https://www.sciencedirect.com/science/article/pii/S0167278926002903)及[半经典非线性WKB复包络](https://www.sciencedirect.com/science/article/abs/pii/S1007570426010075)是相位—包络与守恒设计的近邻。核对出版商可见方法/摘要及CrossRef记录；全文直连403，细节比较范围明确受限。CrossRef的DOI创建日不是首发日，2026-11卷期也不是尚未公开的证据；仍未取得这些文章的可核实Available online日期。不能依据摘要排除所有潜在重合。

解析自由载波乘复包络，与已知自由算符的相互作用绘景变换有直接数学关系。此路线在成熟积分方法中早已使用，例如 [Hult RK4IP](https://opg.optica.org/jlt/abstract.cfm?uri=JLT-25-12-3770)（已核对作者全文）。本项目学习近似载波时保留其自由缺陷，但这不会使相互作用绘景本身成为新概念。

量子归一的结构先例已通过[Pu等原始全文](https://arxiv.org/abs/2303.03934)及[正式出版全文](https://hepnp.ihep.ac.cn/fileZGWLC/journal/article/zgwlc/2023/5/PDF/CPC-2023-0002.pdf)核查。其单调CDF生成非负实基态波函数，但式4以CDF两端软损失控制归一；有限区间总积分等于端点差，不能将它标为任意权重下精确单位范数的硬球面层。这是全文检查后的具体区分，仍不能据此主张一般结构化归一首创。[KKT-hPINN](https://arxiv.org/abs/2402.07251)则通过投影实施线性等式。本项目全复谱径向投影是二次范数结构；机器精度总概率不能代替正确动力学。

振荡基线来自 [SIREN](https://arxiv.org/abs/2006.09661) 和 [Fourier features](https://arxiv.org/abs/2006.10739)，本轮为谱PINN适配并按固定小规模开发规则选择，未声称复现原论文全部任务。成熟谱算法依据 [Dirac时间分裂Fourier伪谱](https://arxiv.org/abs/2112.03616)，本轮测试TSFP2。未覆盖所有因果训练、时间分解、复权重网络或神经算子，结论应限定于实际比较对象。

新增[Khairullin–Rykovanov原始预印本](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5901547)及[正式论文元数据](https://doi.org/10.1016/j.cnsns.2026.109926)显示，波幅比值、透明边界硬结构及相位积分恢复已有振荡PINN相邻路线。当前仅核到作者摘要，未取得方法全文，也未复现其性能；不能把其报道的速度比当本轮基准。它进一步限制“通过结构分离快相位”这一宽泛原创主张。对象是Helmholtz空间散射，不能未经推导等同Dirac时变能支。

本次补查保留最初147条原始候选元数据，另加1条定向原始记录，派生筛选表共148条，先例表13行、BibTeX15条。完整获取8篇独立原文、9个PDF版本（Pu等预印本与正式版分别保存），两页正式方法实际渲染查看。补查响应、获取记录与SHA在public_*20261011及norm_and_helmholtz_fulltext_web_20261011.json等文件中；更新前的派生表和报告保留于revisions/before_public_fulltext_2026_10_11。三篇2026 CPA/WKB全文访问限制仍未消除。

详细的重合部件、核对深度和可区分内容见 [prior_art_map.csv](prior_art_map.csv)，可导入文献库的原始元数据见 [references.bib](references.bib)。本项目较有区分度的候选贡献是：在已知Dirac多能支中用相干量检验相位对齐；定位具体空通道硬门控的初始切向缺陷并展示局部修复不足；分离载波退化与范数结构的作用；诚实呈现有限条件的成功与失败边界。最终有效性、学习载波价值和投稿定位必须等待R14完整结果与独立审计，不能先写成普适求解器或效率突破。
