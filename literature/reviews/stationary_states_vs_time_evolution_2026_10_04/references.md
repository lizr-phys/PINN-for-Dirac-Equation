# 核实的代表文献

核查日：2026-10-04。34 条期刊/会议 DOI 由 Crossref 核实；另 6 条为已核实的 arXiv 版本，其中一条明确为硕士论文。arXiv 版本不等于已完成同行评议；没有核实期刊记录不等于确认没有发表。

| ID | 年份/状态 | 原始论文与出处 | 相关性 |
|---|---|---|---|
| B01 | 2022 | [Physics-Informed Neural Networks for Quantum Eigenvalue Problems](https://doi.org/10.1109/IJCNN55064.2022.9891944)；2022 International Joint Conference on Neural Networks (IJCNN) | 方程、归一化、正交约束的量子本征问题先例。 |
| B02 | 2025 | [A deep neural network approach to solve the Dirac equation](https://doi.org/10.1140/epja/s10050-025-01630-5)；The European Physical Journal A | 逆 Hamiltonian 避免 Dirac 变分塌缩，并计算低激发态。 |
| B03 | 2026 卷期；2025-12 在线 | [Solving nuclear Dirac Woods–Saxon potential with a physics-informed neural network](https://doi.org/10.1088/1572-9494/ae1a5c)；Communications in Theoretical Physics | 径向 Dirac Woods–Saxon 的残差 PINN，须区别标量/矢量势。 |
| B04 | 2024 | [Accurate computation of quantum excited states with neural networks](https://doi.org/10.1126/science.adn0137)；Science | NES-VMC 求多电子激发态、跃迁观测；不是低维残差 PINN。 |
| B05 | 2017 | [Solving the quantum many-body problem with artificial neural networks](https://doi.org/10.1126/science.aag2302)；Science | 神经量子态处理多体波函数的基础工作。 |
| B06 | 2020 | [Ab initio solution of the many-electron Schrödinger equation with deep neural networks](https://doi.org/10.1103/PhysRevResearch.2.033429)；Physical Review Research | FermiNet 与变分 Monte Carlo，针对多电子反对称波函数。 |
| B07 | 2020 | [Deep-neural-network solution of the electronic Schrödinger equation](https://doi.org/10.1038/s41557-020-0544-y)；Nature Chemistry | PauliNet 将物理先验与神经多电子波函数结合。 |
| B08 | 2026；预印本版本 | [Excited Pfaffians: Generalized Neural Wave Functions Across Structure and State](https://arxiv.org/abs/2603.14515)；arXiv | 2026 多状态神经波函数与重要性采样预印本。 |
| S01 | 2025 | [Application of physics-informed neural networks to multidimensional quantum tunneling](https://doi.org/10.1088/1674-1137/add9fc)；Chinese Physics C | 多通道近势垒隧穿；复 SIREN，输入 r 和 E，对照 FEM。 |
| S02 | 2026-09 期刊；2026-05 预印本 | [Stable minimum principles for scattering states](https://doi.org/10.1088/1751-8121/aea52e)；Journal of Physics A: Mathematical and Theoretical | 稳定散射最小原理与振幅误差界；非相对论定理不能直接移植。 |
| S03 | 2026；预印本版本 | [Light nuclear scattering from neural quantum states](https://arxiv.org/abs/2605.27807)；arXiv | 神经量子态求弹性/非弹性中子–氘核散射，无需含时推进。 |
| S04 | 2003 | [Transmission resonances and supercritical states in a one-dimensional cusp potential](https://doi.org/10.1103/PhysRevA.67.052707)；Physical Review A | cusp 势 Dirac 的透射、束缚与超临界；势形不同于当前高斯势。 |
| S05 | 2019 | [Proper treatment of scalar and vector exponential potentials in the Klein-Gordon equation: Scattering and bound states](https://doi.org/10.1140/epjp/i2019-12571-8)；The European Physical Journal Plus | KG 标量/矢量耦合、流约定、束缚/散射区别。 |
| R01 | 1968 | [On the use of resonant states in eigenfunction expansions of scattering and reaction amplitudes](https://doi.org/10.1016/0375-9474(68)90593-9)；Nuclear Physics A | Berggren 共振态与连续谱展开的经典基础。 |
| R02 | 1985 | [Interfering resonances and bound states in the continuum](https://doi.org/10.1103/PhysRevA.32.3231)；Physical Review A | 干涉共振和 BIC 经典先例，不能把全透射当 BIC。 |
| R03 | 2026 | [Single-particle resonant states in relativistic hartree-fock theory: A Green's function approach](https://doi.org/10.1088/1674-1137/ae6a85)；Chinese Physics C | 2026 相对论 Hartree–Fock 与 Green 函数统一束缚态和共振。 |
| A01 | 1929 | [Über die Quantenmechanik der Elektronen in Kristallgittern](https://doi.org/10.1007/BF01339455)；Zeitschrift f�r Physik | 空间周期 Bloch 谱，与时间周期 Floquet 问题区别。 |
| A02 | 2025；硕士论文 | [Physics-Informed Neural Solvers for Periodic Quantum Eigenproblems](https://arxiv.org/abs/2512.21349)；arXiv; Master's thesis | 二维 Bloch 量子本征问题；明确为硕士论文。 |
| T01 | 2022 | [Time-dependent Dirac Equation with Physics-Informed Neural Networks: Computation and Properties](https://doi.org/10.1016/j.cpc.2022.108474)；Computer Physics Communications | PINN 求含时 Dirac 的已有直接先例。 |
| T02 | 2023 | [Scattering with neural operators](https://doi.org/10.1103/PhysRevD.108.L101701)；Physical Review D | 标题为 scattering，但实际学习波包含时传播的神经算子。 |
| T03 | 2024 | [Physics-Informed Neural Networks for Quantum Control](https://doi.org/10.1103/PhysRevLett.132.010801)；Physical Review Letters | 开放量子系统控制，属于动力学与控制而非静态束缚谱。 |
| M01 | 2020 | [Physics Informed Extreme Learning Machine (PIELM)–A rapid method for the numerical solution of partial differential equations](https://doi.org/10.1016/j.neucom.2019.12.099)；Neurocomputing | 随机隐藏层与物理残差最小二乘，当前 PIELM 的方法谱系。 |
| M02 | 2022 | [When and why PINNs fail to train: A neural tangent kernel perspective](https://doi.org/10.1016/j.jcp.2021.110768)；Journal of Computational Physics | PINN 多损失训练困难的机制研究；不自动解释本项目失败。 |
| M03 | 2026 | [Spectral informed neural networks](https://doi.org/10.1016/j.cam.2025.117178)；Journal of Computational and Applied Mathematics | 谱坐标 PINN，空间谱与时间输入的已有方法。 |
| P01 | 2026；预印本版本 | [Physics-Informed Classical and Quantum Neural Networks for One-Dimensional Schrodinger Eigenvalue Problems](https://arxiv.org/abs/2609.22189)；arXiv | 2026 一维谱 benchmark 的经典/量子神经网络预印本。 |
| R04 | 2026 | [Existence of Friedrich–Wintgen Bound States in the Continuum: System of Schrödinger Equations](https://doi.org/10.1137/25M1733926)；SIAM Journal on Applied Mathematics | 2026 给三耦合 Schrödinger 方程 BIC 存在性严格论证。 |
| X01 | 2009 | [Quantum interference and Klein tunnelling in graphene heterojunctions](https://doi.org/10.1038/nphys1198)；Nature Physics | 石墨烯输运干涉实验，散射相位与电导相关。 |
| X02 | 2013 | [Observing Atomic Collapse Resonances in Artificial Nuclei on Graphene](https://doi.org/10.1126/science.1234320)；Science | STM 测石墨烯人工核的原子塌缩共振。 |
| X03 | 2010 | [Quantum simulation of the Dirac equation](https://doi.org/10.1038/nature08688)；Nature | 离子自由 Dirac 模拟，直接关联含时主线。 |
| X04 | 2011 | [Quantum Simulation of the Klein Paradox with Trapped Ions](https://doi.org/10.1103/PhysRevLett.106.060503)；Physical Review Letters | 离子散射波包重建，含时散射的实验实例。 |
| X05 | 2020 | [Direct observation of Klein tunneling in phononic crystals](https://doi.org/10.1126/science.abe2011)；Science | 声子晶体 Klein 隧穿，属于有效 Dirac 波的类比平台。 |
| X06 | 2026；预印本版本 | [Learning Boson Star Solution Families with Physics-Informed Neural Networks](https://arxiv.org/abs/2608.21845)；arXiv | 2026 Einstein–KG 定态解族代理，含监督与物理损失。 |
| M04 | 2021 | [Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators](https://doi.org/10.1038/s42256-021-00302-5)；Nature Machine Intelligence | 学习函数空间映射的神经算子，区别有限参数条件网络。 |
| S06 | 2026 | [Exterior complex scaling enables physics-informed neural networks for nuclear reactions](https://doi.org/10.1103/sjz4-pq6p)；Physical Review C | 2026 ECS 驱动方程 PINN 求核反应，阅读最新版弱源锚损失。 |
| S07 | 2026；预印本版本 | [Constructing Inverse Potentials from Scattering Phase Shifts using Physics-Informed Neural Networks: Application to Neutron-Alpha Scattering](https://arxiv.org/abs/2605.02264)；arXiv | 相移数据反演势，网络势＋可微 RK4 传统积分的混合路线。 |
| B09 | 2023 | [Physics-informed neural networks for discovering localised eigenstates in disordered media](https://doi.org/10.1016/j.jocs.2023.102136)；Journal of Computational Science | 无序局域谱 PINN，已于 2023 年期刊发表，对照 IGA。 |
| O01 | 2019 | [Variational Neural-Network Ansatz for Steady States in Open Quantum Systems](https://doi.org/10.1103/PhysRevLett.122.250503)；Physical Review Letters | Lindblad 密度矩阵稳态；不同于纯态定态能谱。 |
| S08 | 2026 | [Bidirectional neural networks for global nucleon-nucleus optical model calculations](https://doi.org/10.1103/qw54-df4l)；Physical Review C | 2026 由 Numerov 标签训练的全局核散射可微模拟器。 |
| S09 | 2026 | [Reduced basis emulator for elastic scattering in continuum-discretized coupled-channels calculations](https://doi.org/10.1103/n24x-d9gm)；Physical Review C | 2026 POD＋Galerkin 的非神经散射降阶模拟器。 |
| S10 | 2026 | [Direct boundary matching : A bound-state technique for nuclear scattering with Lagrange-Legendre functions](https://doi.org/10.1103/ddcx-cslb)；Physical Review C | 2026 直接边界匹配把内部 L2 基用于开放边界散射。 |

元数据原响应保存在 raw/，读取层级见 annotations.json。条目标题去掉出版脚注星号、HTML 格式和多余空格；Bloch 原始标题包含编码替代符，展示为规范德文。Du/Gao 作者中英混排名字按原始出版页面规范化；S07 的五位作者按原 PDF 首页分开，原 arXiv 合并字段仍保留。S06 期刊标题改为 nuclear reactions，与预印本 scattering 归为一篇，不重复统计。

年份优先采用期刊印刷年份，另保留线上、印刷、arXiv 日期。M03 当前 Crossref 卷期为 2026，不沿用旧综述的 2025 年简写；X02 的 arXiv 于 2015 年补存，原 Science 论文为 2013 年。P01 页面 DOI 标注注册待定，因此只提供已核实的 arXiv 链接，不把待注册字符串当成已生效 DOI。
