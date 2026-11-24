本节在全部预定训练结束后撰写，检索截止日期为 **2026-10-02**。结论是：R8有可复核的耦合恢复和失效机制证据，但不能把“相位—复包络、自由载波预条件、预训练或时间窗”单独宣称为新发明，更不能据本次有限检索宣称首创。

### 已有方法与重合边界

| 已核实的原始来源 | 与本课题重合的部分 | 本次能够确认的差别与限制 |
| --- | --- | --- |
| Lorin、Yang，2022，[Time-dependent Dirac Equation with Physics-Informed Neural Networks](https://arxiv.org/abs/2204.02959)，期刊DOI 10.1016/j.cpc.2022.108474 | 用PINN求含时间的Dirac方程，自动微分构造物理残差 | “Dirac-PINN”本身不是新方法；该记录不建立R8的载波预训练与动态侧带恢复证据 |
| Balac等，2016，[Interaction Picture method](https://www.numdam.org/item/M2AN_2016__50_4_945_0/)，DOI 10.1051/m2an/2015060 | 分离已知线性演化以减轻剩余动力学的数值困难 | 相互作用绘景及相应预条件思想是已有工具。R8用自由PDE学习近似载波，并保留其缺陷项，不等同于写入精确自由传播算子 |
| Liu等，2024，[Preconditioning for Physics-Informed Neural Networks](https://arxiv.org/abs/2402.00531) | 用条件数解释、缓解PINN训练困难 | PINN预条件与条件性分析已有先例；R8的局部输出坐标条件数不是该文的完整网络条件数或其定理 |
| Wang等，[Respecting causality](https://arxiv.org/abs/2203.07404)，2022预印本、2024期刊 DOI 10.1016/j.cma.2024.116813 | 尊重时间因果结构以改善PINN训练 | R8的固定十窗顺序训练不是该文因果加权损失；不能以“按时间训练”作独立首创点 |
| Penwarden等，2023，[Causal sweeping strategies and temporal decompositions](https://arxiv.org/abs/2302.14227)，DOI 10.1016/j.jcp.2023.112464 | 时间推进、分域、上一窗预测作为下一窗初态 | C、D采用这些常见训练思想。本轮C未成功；不把时间窗包装成R8成功的必要机制 |
| Kumar等，2026，[Conservative phase-amplitude PINNs](https://www.sciencedirect.com/science/article/pii/S0377042726002505)，DOI 10.1016/j.cam.2026.117607 | 相位—振幅表示与守恒约束，正是早期课题的出发点 | 本课题已将其作为来源；不能把CPA表示重新命名后宣称原创 |
| [A deep learning WKB-PINNs framework](https://www.sciencedirect.com/science/article/pii/S1007570426010075)，DOI 10.1016/j.cnsns.2026.110653；[WKB-PINNs for highly oscillatory Schrödinger-type equations](https://www.sciencedirect.com/science/article/pii/S0167278926002903)，DOI 10.1016/j.physd.2026.135394 | 学习相位并分离快速振荡与包络；后者的可检索摘要明确含复振幅 | 与“学习相位＋复包络”存在实质方法邻近性，不能称该组合本身全新。可读摘要未给出本轮Dirac耦合、零初始侧带、独立种子验证的同一证据链；受访问限制，未做这两篇全文代码的逐项等价审查 |
| Misumi，2026-07-16，[Lattice fermion formulation via PINNs](https://journals.aps.org/prd/abstract/10.1103/756m-stz5) | Dirac算符相关的物理约束学习 | 问题是格点费米子算符/Ginsparg–Wilson关系，不是本轮的含时侧带动力学 |
| Giotis等，[A unified Python framework for solving Schrödinger and Dirac equations in exotic lepton–antilepton atoms](https://www.sciencedirect.com/science/article/abs/pii/S0010465526003176)，DOI 10.1016/j.cpc.2026.110335 | Dirac相关PINN、结构与优化消融 | 已检索摘要/结论讨论径向束缚态ODE及能量估计，不是本轮的周期耦合含时问题；不借此否定其成果 |

上述2026年CPA及两篇WKB论文的Crossref记录未提供精确published-online日期。CPA卷期为2026年10月，两篇WKB卷期为2026年11月；WKB DOI记录created日期分别为8月4日、9月2日，**created不等于首次公开日期**。这些论文现在可被索引检索，应纳入相关工作，但不能仅以未来卷期排除，也不能把DOI创建时间当成“三个月内发表”或优先权证据。

### 本课题可主张什么

可主张的是一个**限定物理问题上的诊断与恢复证据链**：自由问题的相位载波优势，在动量耦合及零初始占据下遭遇失效；相位释放解决局部梯度缺陷而不恢复完整动力学；由非零单位自由初态进行PDE载波预训练、冻结但保留时间导数，再用复包络学习完整耦合方程，在预注册预算和五个独立新种子中恢复真实侧带。载波移除消融支持“实际使用载波预条件”在该预算中的贡献。

这不是“原纯实振幅LTPB已经被证明普遍正确”，也不是对每个部件独立贡献的完整证明。尚缺少相同总容量的端到端双网络控制、精确载波只作诊断时的进一步定量分解、更一般动量耦合与外势下的跨算例验证，以及与成熟谱积分器的统一误差—总成本比较。R8也没有证明新训练轨迹优于所有已发表方法。

### 检索范围与可复核性

采用学术检索技能的多源工作流：当前会话没有学术MCP连接，改用技能自带OpenAlex检索脚本发现条目，再到arXiv、原始期刊页面/数学文献库和Crossref元数据核实；以技能自带转换器导出10条去重文献的BibTeX。PubMed、Crossref、arXiv公开端点预检均可达，但本课题不依赖PubMed的医学主题覆盖。

历史方法检索不限年份；近期补充检索以2026年为召回范围，并关注2026-07-02至2026-10-02的相关条目。**年份过滤、搜索引擎相对日期和DOI创建日期均不能替代精确发表日期**。已查看的有限条目中，没有核实到与上面整条证据链完全相同的工作；这不是无重复工作的证明。未覆盖Scopus、Web of Science、CNKI的授权全文，也未完成所有术语、语言与未索引预印本的穷尽检索。

过程与访问限制见[检索记录](literature_search_record.json)；可导入文献管理器的[去重BibTeX](references/references.bib)。这些文件只供研究定位，不进入训练配置或检查点。
