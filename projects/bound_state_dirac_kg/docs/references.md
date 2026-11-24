# 核实后的来源

核实日期：2026-09-26。以下来源提供模型和方法依据，不是本高斯势阱计算结果的外部验证。

1. Hideo Hasegawa, *Bound states of the one-dimensional Dirac equation for scalar and vector double square-well potentials*, Physica E **59**, 192–201 (2014).
   DOI：[10.1016/j.physe.2014.01.011](https://doi.org/10.1016/j.physe.2014.01.011)。
   已核实 [arXiv 元数据](https://arxiv.org/abs/1309.0308)与[原文 II 节](https://arxiv.org/html/1309.0308v4)。
   用于核对一维 Dirac 的矢量/标量耦合区别与束缚态匹配方法；该文势形不是本项目的高斯势。

2. Elvis J. Aquino Curi, Luis B. Castro, Antonio S. de Castro, *Proper treatment of scalar and vector exponential potentials in the Klein-Gordon equation: Scattering and bound states*, European Physical Journal Plus **134**, 248 (2019).
   DOI：[10.1140/epjp/i2019-12571-8](https://doi.org/10.1140/epjp/i2019-12571-8)。
   已核实 [arXiv 期刊元数据](https://arxiv.org/abs/1902.02872)及[原文第 2 节、式 (1)–(2)](https://arxiv.org/pdf/1902.02872)。
   用于区分 KG 两类耦合、束缚态条件及电荷解释。该文电荷密度为 (E−V)|φ|²/m；
   本报告不除以 2m，采用 2(E−V)|φ|²，因子约定在 THEORY.md 完整展开，两者相差固定正因子 2m。

3. Françoise Tisseur and Karl Meerbergen, *The Quadratic Eigenvalue Problem*, SIAM Review **43**, 235–286 (2001).
   DOI：[10.1137/S0036144500381988](https://doi.org/10.1137/S0036144500381988)。
   核实[出版社记录](https://epubs.siam.org/doi/abs/10.1137/S0036144500381988)及[作者机构原文](https://eprints.maths.manchester.ac.uk/466/1/38198.pdf)。
   用于二次矩阵铅笔与伴随线性化背景；本代码的 KG 铅笔符号另由回归测试核验。

4. C. W. J. Beenakker, A. Donis Vela, G. Lemut, M. J. Pacholski, and J. Tworzydło, *Tangent fermions: Dirac or Majorana fermions on a lattice without fermion doubling*, Annalen der Physik **535**, 2300081 (2023).
   DOI：[10.1002/andp.202300081](https://doi.org/10.1002/andp.202300081)。
   核实[作者预印本及期刊信息](https://arxiv.org/abs/2302.12793)。
   用于差分色散和倍增背景；本项目采用非局域 Fourier 导数，并未实现该文的 tangent 方法。

5. Maziar Raissi, Paris Perdikaris, George E. Karniadakis, *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*, Journal of Computational Physics **378**, 686–707 (2019).
   DOI：[10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)。
   核实[出版社索引](https://www.sciencedirect.com/science/article/pii/S0021999118307125)、[Crossref 记录](https://crossmark.crossref.org/dialog-content?doi=10.1016%2Fj.jcp.2018.10.045&domain=pdf)和[早期方法预印本](https://arxiv.org/abs/1711.10561)。
   用于 PINN 自动微分残差框架背景，不作为本任务具有原创性的证据。
