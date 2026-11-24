# 核实的原始文献

核查日期：2026-10-04。学术检索 MCP 在本会话未挂载；使用出版社/作者预印本页面和 Crossref 原始 DOI API 交叉核查。完整元数据与访问状态保存在 `results/literature`。这些文献提供物理/方法背景，不作为本项目原创性或算法优越性的证据。

1. V. M. Villalba, W. Greiner. *Transmission resonances and supercritical states in a one-dimensional cusp potential*. Physical Review A **67**, 052707 (2003). [DOI:10.1103/PhysRevA.67.052707](https://doi.org/10.1103/PhysRevA.67.052707)；[出版社记录](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.67.052707)。研究一维 Dirac 散射与束缚态，势为 cusp，不能把其解析结论直接搬到本高斯势。该文使用“transmission resonance”的语境与本轮没有求解的复能极点要区分。

2. E. J. Aquino Curi, L. B. Castro, A. S. de Castro. *Proper treatment of scalar and vector exponential potentials in the Klein-Gordon equation: Scattering and bound states*. European Physical Journal Plus **134**, 248 (2019). [DOI:10.1140/epjp/i2019-12571-8](https://doi.org/10.1140/epjp/i2019-12571-8)；[作者原文](https://arxiv.org/html/1902.02872)，第 2 节式 (1)–(2) 区分 S、V 及电荷流。该文流除以质量，本文采用整体相差固定因子的 ρ=2(E−V)|φ|²、j=2 Im(φ*φ')，通量比不受此因子影响。其势形不是本高斯势。

3. M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics **378**, 686–707 (2019). [DOI:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)；[出版社原始记录](https://www.sciencedirect.com/science/article/pii/S0021999118307125)。本轮常规 MLP 的自动微分方程残差框架具有已有方法基础。

4. V. Dwivedi, B. Srinivasan. *Physics Informed Extreme Learning Machine (PIELM)—A rapid method for the numerical solution of partial differential equations*. Neurocomputing **391**, 96–118 (2020). [DOI:10.1016/j.neucom.2019.12.099](https://doi.org/10.1016/j.neucom.2019.12.099)；[作者预印本](https://arxiv.org/abs/1907.03507)，[出版社记录](https://www.sciencedirect.com/science/article/pii/S0925231219318144)。随机隐藏层与物理损失输出层最小二乘已有研究，不把本轮此结构称为原创 PINN。Crossref 标题分隔符存在字符编码损坏，展示标题据作者预印本规范化，原 JSON 保持原样。

5. S. Dong, Z. Li. *Local extreme learning machines and domain decomposition for solving linear and nonlinear partial differential equations*. Computer Methods in Applied Mechanics and Engineering **387**, 114129 (2021). [DOI:10.1016/j.cma.2021.114129](https://doi.org/10.1016/j.cma.2021.114129)；[出版社记录](https://www.sciencedirect.com/science/article/pii/S0045782521004606)。说明固定随机隐藏层、线性/非线性最小二乘的相关方法谱系；本项目没有复现该文域分解算法，不能直接沿用其性能结论。

6. S. Wang, X. Yu, P. Perdikaris. *When and why PINNs fail to train: A neural tangent kernel perspective*. Journal of Computational Physics **449**, 110768 (2022；2021 在线发表). [DOI:10.1016/j.jcp.2021.110768](https://doi.org/10.1016/j.jcp.2021.110768)；[作者原文](https://arxiv.org/abs/2007.14527)。多损失项训练速率差异提供失败机制背景；本轮未计算 NTK，不能声称已证实该论文的机制就是当前全部失败的因果解释。

7. M. Tancik et al. *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*. NeurIPS 2020. [会议原文](https://papers.neurips.cc/paper_files/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html)；[预印本 DOI:10.48550/arXiv.2006.10739](https://doi.org/10.48550/arXiv.2006.10739)。用于已知振荡特征的背景；本轮载波使用物理 k(E)，不是逐字复现该文随机 Fourier 特征方案。
