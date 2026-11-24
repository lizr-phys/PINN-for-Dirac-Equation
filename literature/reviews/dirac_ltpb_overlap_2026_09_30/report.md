# Dirac LTPB 近期文献重合度调研

检索日期：2026-09-30。主要窗口：2026-06-30—2026-09-30（三个日历月）。更早的关键先例单独列出，不与窗口内新公开工作混计。本记录是文献检索与方法比较，不改变任何正式实验配置、结果或预注册判据。

## 1. 结论先行

在本轮能检索并核对的公开来源中，**未发现完整重复本项目 LTPB 方法组合及其机制验证链的论文**：自由 Dirac 正负能支基底、各支实振幅与独立可学习相位、训练不输入精确时间频率、同预算机制消融、共享波数网络及耦合外势下零初始占据通道诊断。

这不是“已证明无人做过”，更不是“世界首次”。组成部件已有显著先例，其中共享谱网络的先例尤其直接：SINN 在 2024 年已经以 Fourier 波数和时间为输入、预测谱系数、用谱空间算子训练。因此，不能把一般的共享谱网络、振幅—相位 PINN、硬初值或 Dirac PINN 本身列为本项目独创。[SINN 原文](https://arxiv.org/html/2408.16414v2)、[时变 Dirac PINN](https://arxiv.org/abs/2204.02959)。

较有区分度的研究主线应收窄为：**能支—相位对齐对混合 Dirac 态载波优化和物理相干量的作用，以及该表示在动态产生空通道时的适用边界。**这是根据已有实验与下列文献比较作出的判断，仍需投稿前继续查新。

## 2. 本项目实际做了什么

比较对象以 [六轮完整实验报告](../../../projects/dirac_cpa_transfer_experiment/docs/comprehensive/report.md) 和各轮原始研究记录为依据。

| 轮次 | 对文献比较重要的方法或证据 | 不能扩大的结论 |
| --- | --- | --- |
| 公共 CPA 迁移 | 正能态表现好，混合正负能态相干量误差大；守恒约束不充分识别相对相位 | 不是证明公共相位在数学上不能表示混合态 |
| Learned Two-Phase | 已知自由能支基底，实有符号振幅、独立学习相位；仅 PDE 残差，无精确时间载波 | 基底含已知物理先验，不是完全无先验求解 |
| 机制消融 | Branch-Complex、Component-Polar、旋转基底，同预算、配对种子 | 不能把结果写成所有预注册强机制条件均通过 |
| 双波数 | 四个已知离散通道独立相位 | 不是未知谱发现 |
| 共享谱波包 | 一个共享网络接收归一化波数和时间；训练 33 模态，评价相邻中点 | 是给定 Fourier 表示下的自由波包和区间内插值验证 |
| 周期外势 | 完整耦合谱 Dirac 方程；中心有占据，侧带初始为空 | 两个共享谱模型均未达绝对标准，结论是 INCONCLUSIVE，非外势迁移成功 |

本项目关键参数化为 $a_s=a_{s,0}+\tau r_s$、$S_s=\tau q_s$、$c_s=a_se^{iS_s}$。已知 $u_\pm(k)$ 是表示基底；精确 $E(k)$ 不进入 learned 时间载波和训练目标，而仅用于参考或事后评价。相近论文只有某个关键词相同，不足以认定重复；反之，名称不同也不能排除实际方法重合。

## 3. 窗口内能确认公开日期的工作

“相近程度”是本次阅读方法段或摘要后的比较判断，不是论文作者的自我声明。预印本未经同行评议，不与已发表论文混淆。

| 工作与日期证据 | 交叉核对的内容 | 与本项目的关系及核对深度 |
| --- | --- | --- |
| Barman、Ray、Chatterjee，*Split Complex-Valued Physics-Informed Neural Networks for Forward and Inverse Nonlinear PDEs*；arXiv 首发 2026-07-16。[论文](https://arxiv.org/abs/2607.15087)、[方法全文](https://arxiv.org/html/2607.15087v1) | 复权重和复中间表示，实部、虚部分别激活；包括非线性 Schrödinger 等振荡 PDE | 复值/振荡 PINN 的重要相邻工作。实际架构不是能支独立极坐标相位。核对全文关键方法段；本项目实权重四输出 Complex 基线也不等于它的复权重网络 |
| Ibata 等，*NestyNet. IV. Laws Chosen by Nothing in Advance*；arXiv 首发 2026-08-21，修订 08-25。[论文](https://arxiv.org/abs/2608.21491)、[全文](https://arxiv.org/html/2608.21491v2) | 从数据和候选算子库识别微分规律；基准包含 1+1D Dirac | 应用对象有交集，但主要任务是方程发现，不是已知 Dirac PDE 下无解标签的能支相位训练。核对方法及 Dirac 基准段 |
| Tang 等，*Physics-informed neural network amplitude and phase modeling of steady-state thermal waves in layered structures*；J. Applied Physics，在线 2026-09-01。[出版页面](https://pubs.aip.org/aip/jap/article-abstract/140/9/095102/3403518/Physics-informed-neural-network-amplitude-and)、[DOI](https://doi.org/10.1063/5.0339545) | 以振幅、相位建模稳态热波 | 说明一般极坐标/振幅—相位 PINN 不是专属新想法。不是 Dirac 正负能载波学习。核对出版摘要和 CrossRef 日期，未取得全文 |
| Issa 等，*A Spectral–Physics-Informed Neural Network Hybrid with Energy-Constrained Training for the Allen–Cahn Equation: Forward and Inverse Problems*；提交 2026-09-23，公开 09-24。[预印本及全文](https://www.preprints.org/manuscript/202609.2113) | 时间输入输出整个谱系数向量，加能量约束；长时间场精度失效时约束仍可成立 | 与谱 PINN、结构指标不等于解准确的证据有部分重合。输出不是共享 $(k,t)$ 的固定通道网络；没有 Dirac 能支相位。核对全文关键段，未经同行评议 |
| Misumi，*Lattice fermion formulation via Physics-Informed Neural Networks: Ginsparg-Wilson relation and Overlap fermions*；Phys. Rev. D 在线 2026-07-16，但 arXiv 首发 05-07，修订 06-24。[期刊](https://journals.aps.org/prd/abstract/10.1103/756m-stz5)、[预印本](https://arxiv.org/abs/2605.06022) | PINN 构造满足 Ginsparg–Wilson 关系的格点 Dirac 算子 | 是最近发表的相关工作，不是最近首次公开，也不是旋量时间初值问题。核对原始记录和方法摘要 |
| Yoshimura、Sekizawa，*Neural-Network-Based Variational Method in Nuclear Density Functional Theory: Application to the Kohn--Sham method*；arXiv 首发 2026-09-01。[论文](https://arxiv.org/abs/2609.00836) | 复旋量轨道、能量变分和正交化，用于核密度泛函 | 与另一条定态/旋量网络课题相邻，不是本项目时变 Dirac 正负能支相位方法。核对原始摘要，未做完整算法复现 |

以上没有一篇经本次核对可认定为“本项目完整方法已被独立重复”。其中 SCV-PINN 是需要认真讨论的复网络基线；Allen–Cahn 工作提示“守恒或结构保证不充分”也不应被写成一般性首次发现。

## 4. 近期检出的重要候选：首发日期待核

下表不能直接当作“近三个月新论文数量”。CrossRef 的 `created` 是 DOI 元数据记录创建时间，**不是论文上线日期**；2026-11 的卷期日期同样不能代表首次公开时间。以下出版记录均已存在，但 CrossRef 未提供 `published-online`，部分出版商全文访问失败。

| 候选工作 | 已核对的日期与 DOI | 方法重合及结论边界 |
| --- | --- | --- |
| Kumar、Dwivedi、Rajeev，*A deep learning WKB-PINNs framework for forward and inverse problems of semiclassical nonlinear Schrödinger equations* | DOI 记录创建 2026-08-04；卷期 2026-11；[出版摘要](https://www.sciencedirect.com/science/article/pii/S1007570426010075)，[DOI](https://doi.org/10.1016/j.cnsns.2026.110653) | 实相位、复包络和 WKB/相位相关约束，应用于标量非线性 Schrödinger。与相位预条件思想接近；摘要未显示 Dirac 能支对齐验证。仅核对出版摘要及元数据 |
| Cen、Dwivedi、Kumar、Rajeev、Zeng，*WKB-PINNs: Physics-informed neural networks for highly oscillatory Schrödinger–type equations* | DOI 记录创建 2026-09-02；卷期 2026-11；[出版摘要](https://www.sciencedirect.com/science/article/pii/S0167278926002903)，[DOI](https://doi.org/10.1016/j.physd.2026.135394) | 高振荡标量方程的相位—振幅表示及 eikonal 辅助约束。是必须引用和区别的近邻，不能因非 Dirac 就忽略；不能仅凭摘要排除所有潜在重合 |
| Giotis、Gkrepis、Kosmas、Kosmas，*A unified Python framework for solving Schrödinger and Dirac equations in exotic lepton–antilepton atoms* | DOI 记录创建 2026-07-30；卷期 2026-11；[出版摘要](https://www.sciencedirect.com/science/article/pii/S0010465526003176)，[DOI](https://doi.org/10.1016/j.cpc.2026.110335) | Dirac/Schrödinger 定态能谱与原子问题。对工作区束缚态项目相关度较高，摘要方法不同于时变 LTPB；本项目早期开题文档已经列过它 |
| Lai 等，*A spectral PINN approach to unresolved-scale closure in multiscale dynamics* | DOI 记录创建 2026-06-19，早于窗口；卷期 2026-09；[出版摘要](https://www.sciencedirect.com/science/article/pii/S0960077926008209)，[DOI](https://doi.org/10.1016/j.chaos.2026.118679) | 已解析尺度数据驱动未解析谱闭合，应用于多尺度动力学；不是无解标签 Dirac 能支相位训练。九月卷期不能证明九月首次公开 |

两篇 WKB 工作 DOI 不同，保留为两篇，不把同一作者团队的不同论文误去重。也不把期刊卷期尚未来到等同于论文尚未公开。投稿前需补查这些候选的出版商“Available online”日期和全文。

## 5. 更早先例及其对创新表述的影响

| 必须承认的先例 | 与本项目重合的部件 | 可以区别的内容 |
| --- | --- | --- |
| Yu 等，*Spectral Informed Neural Network: An Efficient and Low-Memory PINN*，arXiv 2024-08-29，2025 年 JCAM 发表；[方法原文](https://arxiv.org/html/2408.16414v2)，[期刊 DOI](https://doi.org/10.1016/j.cam.2025.117178) | 共享网络以 Fourier 波数和时间为输入，输出谱系数；谱算子代替空间自动微分 | 本项目在自由 Dirac 能支基底中独立学习相位及其物理相干机制。共享谱形式和参数量不随模态数线性增大不能单独作为原创贡献 |
| Lorin、Yang，*Computation of the Time-Dependent Dirac Equation with Physics-Informed Neural Networks*，arXiv 2022-04-06；[原始记录](https://arxiv.org/abs/2204.02959)，[CPC DOI](https://doi.org/10.1016/j.cpc.2022.108474) | 用 PINN 求时变 Dirac，复旋量的四实输出，自动微分及硬初值构造 | 能支与载波相位对齐、同容量机制对照、混合态相干量审计，不是 Dirac PINN 首次应用 |
| Kumar、Dwivedi、Rajeev，*Structure-preserving conservative phase-amplitude PINNs for semiclassical Schrödinger-type equations*；DOI 记录已于 2026-03-18 创建，卷期 2026-10，首发日期另需核实；[出版页面](https://www.sciencedirect.com/science/article/pii/S0377042726002505)，[正确 DOI](https://doi.org/10.1016/j.cam.2026.117607) | 相位—振幅、连续性和全局守恒约束，是本项目 CPA 迁移的已有来源 | 本项目研究迁移到旋量多能支时的载波优化退化，不主张发明 CPA |

## 6. 建议如何表述本课题

### 可由目前证据支持的表述

本项目在已知自由 Dirac 能支基底下研究独立可学习相位，并通过混合态、基底/表示消融、多波数、共享谱波包和耦合外势逐步检验载波优化优势与适用边界。贡献强调可检验的物理机制与失败证据，而不是把几种已有技术拼在一起就宣称全新通用框架。

自由共享谱波包五种子 mean spinor error 为 0.041038、current error 为 0.048057、fidelity 为 0.998249，满足该轮预注册条件；这是给定有限 Fourier 表示及训练区间中点检验的证据。[本地波包报告](../../../projects/dirac_cpa_transfer_experiment/docs/experiments/wavepacket_shared_spectral_ltpb/report.md)。

弱周期外势下，LTPB mean spinor error 为 0.366703、current error 为 0.456121、fidelity 为 0.872527。虽然两项主误差对 Complex 为 5/5 配对胜出，二者均未过绝对标准，仍是 INCONCLUSIVE。不能把该轮写成一般外势有效。[本地外势报告](../../../projects/dirac_cpa_transfer_experiment/docs/experiments/periodic_potential_coupled_ltpb/report.md)。

外势轮另有具体表示诊断：空通道 $c(t)=\tau r(t)e^{i\tau q(t)}$ 的 $c'(0)=r(0)/T$ 为实，而所选实基底中的耦合 PDE 需要非零纯虚初始导数。这是当前光滑硬参数化的端点一阶切线限制，**不是所有极坐标表示不可能，也不足以单独解释整场训练失败**；Complex 同样失败。该数学诊断具有区别度，但本次检索不足以证明此前无人研究过一般零振幅相位奇异性。

### 不应采用的表述

- “首次用 PINN 求 Dirac 方程”“首次使用共享波数网络”“首次把振幅与相位分开”。
- “首次发现守恒量不能保证解正确”或“公共单相位数学上不能表示混合态”。
- “已解决连续未知谱发现、一般外势、通用 Dirac 求解器”。
- “外势实验已经证明失败完全来自零振幅相位退化”。
- “三个月没找到相同论文，因此具有确定的新颖性或必定可以发表”。

工作区定态 Dirac/KG、自由 Dirac 入门示例和谐振子项目应继续定位为基础验证/复现。近期 Giotis 等及更早 Dirac 神经网络论文已有同类大方向；当前检索没有逐一查遍每种高斯势参数，不作“具体 benchmark 完全重复”或“不重复”的保证。[本地束缚态报告](../../../projects/bound_state_dirac_kg/docs/report.md)。

## 7. 检索方法、可重复性与局限

采用 `nature-academic-search` 多源检索、原始出处交叉核验、日期辨析和 DOI/arXiv 去重工作流。实际可用来源为 arXiv API 与原文、CrossRef DOI 元数据、出版商页面、Preprints.org；OpenAlex 作补充发现。没有可用的学术专用连接器，未检索机构授权的 Web of Science、Scopus 或 CNKI。

关键词覆盖 Dirac PINN、positive/negative energy branches、phase-amplitude、two/multi-phase、polar、WKB-PINN、Fourier/spectral PINN、shared frequency network 与 coupled modes。公开网页检索叠加日期过滤，不只搜索 LTPB 这个自定名称。

arXiv API 的结构化查询示例如下，`submittedDate` 使用 UTC 日期范围；每次最多请求 100 条，此处返回均小于上限。

| 查询 | API 返回条数 | 解释 |
| --- | ---: | --- |
| `(all:Dirac AND (all:PINN OR all:"physics-informed" OR all:"neural network")) AND submittedDate:[202606300000 TO 202609302359]` | 4 | 包含 Dirac delta 等非目标命中，不能视为 4 篇同类方法 |
| `(all:Dirac AND (all:neural OR all:PINN OR all:"physics-informed")) AND submittedDate:[202606300000 TO 202609302359]` | 14 | 用宽查询降低漏检，含凝聚态/方程发现等邻近工作 |
| `((all:Dirac OR all:Schrodinger OR all:Schrödinger) AND (all:PINN OR all:"physics-informed") AND (all:phase OR all:amplitude OR all:spectral)) AND submittedDate:[202606300000 TO 202609302359]` | 2 | 其中有 SCV-PINN，另一条为量子态重构而非本项目方法 |
| `all:spinor AND (all:neural OR all:PINN OR all:"physics-informed") AND submittedDate:[202606300000 TO 202609302359]` | 2 | 纳入核密度泛函等旋量网络近邻 |
| `all:Dirac AND (all:PINN OR all:"physics-informed")`，不设首发日期，按更新日期检查 | 20 | 防止漏掉较早首发而近期修改的预印本；结果并非全部 Dirac 初值 PINN |

这些数量是不同查询的原始返回数，相互重叠，**不能相加为筛选论文总数**。方法判断只基于已核对的论文；未命中不等于不存在。可复查的查询、日期和来源级别保存在 [检索证据](search_evidence.json)。

具体盲区：若干 Elsevier 全文直连返回 403，只取得出版商索引摘要与 DOI 元数据；未发表工作、索引延迟、中文学位论文、会议现场材料可能漏检。CrossRef 某次限流后已重试，不能用失败响应得出不存在结论。此次不是专利新颖性法律意见，也没有复现他人代码。

## 8. 下一步优先级

1. 投稿前取得两篇 WKB 和 CPA 原文，把相位训练目标、是否硬编码频率、振幅类型和额外约束逐项比较；核准首发日期。
2. 论文相关工作必须补入 SINN 和 Lorin–Yang，避免共享谱与 Dirac PINN 的宽泛原创表述；复权重 SCV-PINN 可作为下一轮备选强基线，而非声称已有实权重 Complex 对照完全覆盖它。
3. 本轮只完成查新，不增加训练、不修改正式失败记录、不加入补救模型。后续若设计混合复数/极坐标表示，应另行预注册，分别验证初始切向可达性和共同优化问题。
