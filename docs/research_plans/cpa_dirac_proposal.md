# CPA-PINN 在含时 Dirac 方程中的直接迁移局限及旋量双相位 PINN

## 开题报告

**建议英文题目：** *Limits of Direct CPA-PINN Transfer to the Time-Dependent Dirac Equation and a Spinor-Aware Dual-Phase PINN*

**研究对象：** 一维线性含时 Dirac 方程

**方法暂称：** 旋量双相位 Dirac-PINN（SDP-Dirac-PINN）

**文献检索截止日期：** 2026 年 8 月 30 日

---

## 摘要

CPA-PINN 通过“复振幅+实相位”表示以及连续性、质量和能量约束，改善了半经典 Schrödinger 型方程中的强振荡与长时间演化问题 [1]。然而，Schrödinger 波函数是复标量，Dirac 波函数是复旋量；前者的概率流显式依赖公共相位梯度，后者的概率流则由旋量内部结构决定。因此，CPA-PINN 可以在表示层面改写为 Dirac 求解器，但其在 Schrödinger 方程中的“相位—概率流—守恒”机制未必能等价迁移。

本课题不预设“CPA-PINN 数学上不能求解 Dirac 方程”，而检验一个更严格的假设：在正负能支混合和长时间演化中，单一公共相位与总守恒约束无法充分识别旋量相对相位和概率流误差，直接 CPA-Dirac-PINN 因而可能失去原有优势。研究将依次完成理论分析、最小失效实验、旋量双相位方法构造、多层次 benchmark、消融、传统方法对照及稀疏数据反演。公开文献中已有含时 Dirac-PINN、Dirac 相位-振幅方法和 Dirac 流体表示，但截至检索日尚未发现与“CPA 直接迁移局限—混合能支反例—旋量双相位修正”完全重合的公开工作。该结论是有边界的检索结果，不构成绝对首创性证明。

**关键词：** PINN；Dirac 方程；CPA-PINN；相位-振幅表示；概率流；正负能支；长时间演化

---

## 1. 研究问题与范围

考虑一维线性含时 Dirac 方程

$$
i\hbar\partial_t\psi=
\left(-i\hbar c\sigma_x\partial_x+mc^2\sigma_z+V(x,t)I_2\right)\psi,
\qquad \psi\in\mathbb C^2,
$$

其中

$$
\sigma_x=\begin{pmatrix}0&1\\1&0\end{pmatrix},
\qquad
\sigma_z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}.
$$

概率密度与概率流为

$$
\rho=\psi^\dagger\psi,
\qquad
j=c\psi^\dagger\sigma_x\psi,
\qquad
\partial_t\rho+\partial_xj=0.
$$

这里的 $V I_2$ 表示静电型势能（时分量矢量势的有效形式），不是与 $\sigma_z$ 耦合的 Lorentz 标量势。首篇工作限定于单粒子、一维、自由或光滑亚临界势，不涉及粒子创生、超临界强场和量子场论。研究只围绕一个核心问题：

> CPA-PINN 中的单公共相位和总守恒约束，能否在 Dirac 正负能支混合时继续控制旋量相对相位与概率流；若不能，能否用旋量双相位和特征通道交换约束修正？

自由平面波、波包、势散射和数据反演仅是同一问题的递进验证，不分拆为多个独立创新方向。

---

## 2. 文献现状与创新边界

### 2.1 已有工作

| 方向 | 代表工作 | 已解决内容 | 本课题不能重复宣称的创新 |
|---|---|---|---|
| CPA-PINN | Kumar 等，2026 [1] | 二维半经典 Schrödinger/NLS；相位-振幅表示；连续性、质量和能量约束；含噪反演 | 不能将“相位-振幅+守恒损失”本身称为创新 |
| 含时 Dirac-PINN | Lorin 与 Yang，2022 [2] | 含时 Dirac 求解、$L^2$ 范数约束、长时间域分解、费米子倍增讨论 | 不能声称首次 Dirac-PINN、首次长时间或首次归一化约束 |
| 定态 Dirac 神经网络 | Wang 等，2025；Du 与 Zhang，2026；Giotis 等，2026 [3-5] | Coulomb/Woods-Saxon 束缚态、变分塌缩或径向方程 | 不能将“用神经网络求解 Dirac 方程”作为创新 |
| Dirac 相位-振幅 | Thylwe，2008 [6] | 径向定态 Dirac 散射的公共相位与复旋量振幅 | 不能声称首次将相位-振幅用于 Dirac 方程 |
| Dirac 极分解/流体表示 | Fabbri，2023 [7] | 旋量极变量和多种等价流体方程 | 不能声称首次引入 Dirac 内部相位 |
| 传统分支/多尺度算法 | Bao 等，2016、2017；Chai 等，2019 [8-10] | 正负能空间、非相对论极限快振荡和一致精度算法 | “能支分解”不是独立新概念 |

已有研究还表明，附加守恒损失可改善守恒量，却不保证场解误差同步降低 [11]。因此，本课题必须以独立的旋量、相位和概率流指标验证方法，不能只比较训练损失或总范数。

### 2.2 本次查重结论

检索使用 Crossref、arXiv、出版社页面及可获取全文，覆盖 `Dirac CPA-PINN`、`Dirac phase-amplitude PINN`、`Dirac probability current PINN`、`positive negative energy PINN`、`spinor hydrodynamics PINN` 和 `characteristic decomposition Dirac PINN` 等检索式。截至 2026-08-30：

1. **已经完成：** Dirac-PINN、全局范数约束、长时间域分解、Dirac 相位-振幅和流体表示。
2. **未检出高度相似工作：** 系统论证 CPA 公共相位在混合能支 Dirac 动力学中的守恒盲点，并将其与“旋量双相位 PINN+通道交换约束+专用 benchmark”组成完整方法。
3. **严格性边界：** “未检出”不等于“世界首次”。投稿前须复查 Scopus、Web of Science、Google Scholar、CNKI、最新预印本及 CPA-PINN 的后续引用。

---

## 3. 核心假设与理论依据

### 3.1 可证伪假设

> 对单一能支、相位结构简单的 Dirac 解，直接 CPA-Dirac-PINN 可能有效；对正负能支混合解，总范数、总能量和总连续性无法充分识别旋量相对相位误差，单公共相位因而可能失去 CPA 在标量 Schrödinger 问题中的优化优势。显式表示两个旋量通道的相位，并约束通道间概率交换，可能减少该误差。

该假设同时要求两个实验事实：直接 CPA 在受控混合能支条件下出现可重复退化；新方法在等参数量、等采样和等训练预算下改善独立指标。任一条不成立，均应修改假设，不得选择性报告结果。

### 3.2 Schrödinger 公共相位直接进入概率流

对半经典 Schrödinger 型方程

$$
i\varepsilon\partial_t v=
\left(-\frac{\varepsilon^2}{2}\Delta+V+\alpha|v|^2\right)v,
\qquad
v=Ae^{iS/\varepsilon},
$$

有

$$
\rho=|A|^2,
\qquad
\mathbf J=
\varepsilon\operatorname{Im}(\bar A\nabla A)+|A|^2\nabla S.
$$

因此，连续性或概率流损失可直接约束公共相位梯度。若

$$
R_S=i\varepsilon\partial_t v-H_Sv,
$$

则任意可微近似场均满足

$$
\partial_t\rho+\nabla\cdot\mathbf J
=\frac{2}{\varepsilon}\operatorname{Im}(\bar vR_S).
$$

连续性损失本质上是强残差的状态依赖投影；它可以改变优化梯度，但不是独立的方程信息。

### 3.3 Dirac 概率流消去公共相位

直接仿照 CPA 写为

$$
\psi=Ae^{iS/\hbar},
\qquad A\in\mathbb C^2,
$$

则

$$
\rho=A^\dagger A,
\qquad
j=cA^\dagger\sigma_xA.
$$

公共相位 $S$ 在 $\rho$ 和 $j$ 中都被消去。$S$ 仍进入 Dirac 强残差，故直接 CPA 并非没有表达能力；缺失的是 Schrödinger 情形中由概率流对 $\nabla S$ 产生的直接约束。同时，

$$
R_D=i\hbar\partial_t\psi-H_D\psi
$$

导出

$$
\partial_t\rho+\partial_xj
=\frac{2}{\hbar}\operatorname{Im}(\psi^\dagger R_D),
$$

所以总连续性仍是 Dirac 残差投影，不能单独补足旋量相对相位信息。此外，复旋量振幅存在规范冗余

$$
A'=Ae^{i\chi},
\qquad
S'=S-\hbar\chi,
$$

直接 CPA 实际输出 5 个实变量，但其中 1 个是冗余规范自由度。

### 3.4 概率流取决于旋量相对相位

令

$$
\psi_1=a_1e^{iS_1/\hbar},
\qquad
\psi_2=a_2e^{iS_2/\hbar},
$$

则

$$
\rho=a_1^2+a_2^2,
\qquad
j=2ca_1a_2\cos\left(\frac{S_2-S_1}{\hbar}\right).
$$

Dirac 概率流依赖旋量分量的相对相位，而不是整个旋量的公共相位。在“公共实相位+复旋量振幅”中，正负能干涉所需的快速相对相位只能隐含于复振幅，可能破坏“振幅慢变、相位快变”的预期分离。

### 3.5 混合能支的最小反例

对自由 Dirac 方程，

$$
H(k)=\hbar ck\sigma_x+mc^2\sigma_z,
\qquad
E_k=\sqrt{\hbar^2c^2k^2+m^2c^4},
$$

且 $H(k)u_\pm=\pm E_ku_\pm$。选取

$$
u_+=\begin{pmatrix}\cos(\theta/2)\\\sin(\theta/2)\end{pmatrix},
\qquad
u_-=\begin{pmatrix}-\sin(\theta/2)\\\cos(\theta/2)\end{pmatrix},
$$

$$
\sin\theta=\frac{\hbar ck}{E_k},
\qquad
\cos\theta=\frac{mc^2}{E_k},
$$

并在长度 $L$ 的周期区域上构造

$$
\psi_\eta(x,t)=
\frac{e^{ikx}}{\sqrt{L(1+\eta^2)}}
\left(u_+e^{-iE_kt/\hbar}+\eta u_-e^{iE_kt/\hbar}\right).
$$

可直接得到

$$
\rho_\eta=\frac1L,
\qquad
\langle H\rangle=E_k\frac{1-\eta^2}{1+\eta^2},
$$

$$
j_\eta(t)=\frac1L\left[
\frac{\hbar c^2k}{E_k}\frac{1-\eta^2}{1+\eta^2}
+\frac{2\eta mc^3}{E_k(1+\eta^2)}
\cos\left(\frac{2E_kt}{\hbar}\right)
\right].
$$

当 $\eta=1$ 时，密度、范数、能量期望和局部连续性均严格保持，概率流却以 $2E_k/\hbar$ 振荡。因而，错误的正负能支相对相位可与正确的总守恒诊断并存。这只证明守恒诊断存在盲点，不证明完整 Dirac 残差无法唯一确定解。本课题的实验问题是：该盲点是否会在有限采样和非凸优化中形成稳定退化。

### 3.6 开放边界的附加限制

有限区域 $[a,b]$ 满足

$$
\frac{d}{dt}\int_a^b\rho\,dx+j(b,t)-j(a,t)=0.
$$

散射中区域内范数可因出射通量而下降，故不能盲目加入固定总范数。Dirac 方程是一阶双曲系统，开放边界应约束入射特征分量，并以边界通量评价守恒。

---

## 4. 拟提出的旋量双相位方法

### 4.1 特征通道表示

定义

$$
U=\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix},
\qquad
U^\dagger\sigma_xU=\sigma_z,
\qquad
\phi=U^\dagger\psi=\begin{pmatrix}\phi_R\\\phi_L\end{pmatrix}.
$$

原方程等价于

$$
i\hbar(\partial_t+c\partial_x)\phi_R=mc^2\phi_L+V\phi_R,
$$

$$
i\hbar(\partial_t-c\partial_x)\phi_L=mc^2\phi_R+V\phi_L.
$$

$\phi_R$ 和 $\phi_L$ 沿 $\pm c$ 特征方向传播，质量项使两通道耦合。该变换是局部、精确且可逆的，不依赖 Fourier 空间的非局部能支投影。

### 4.2 双相位参数化

令

$$
\phi_R=a_Re^{iS_R/\hbar},
\qquad
\phi_L=a_Le^{iS_L/\hbar},
\qquad a_R,a_L\ge0.
$$

网络输出 $(r_R,r_L,S_R,S_L)$，并取 $a_s=\operatorname{softplus}(r_s)$。该表示具有 4 个实输出，显式给出 $S_L-S_R$，避免直接 CPA 的冗余公共规范变量。当 $a_s\approx0$ 时相位无定义，因此使用

$$
w_s=\frac{a_s^2}{a_s^2+\delta^2}
$$

对相位项加权，并将局部 Cartesian 复分量作为节点附近的回退表示。

### 4.3 通道概率交换

定义 $n_R=|\phi_R|^2$ 和 $n_L=|\phi_L|^2$，则

$$
(\partial_t+c\partial_x)n_R=Q,
\qquad
(\partial_t-c\partial_x)n_L=-Q,
$$

$$
Q=\frac{2mc^2}{\hbar}\operatorname{Im}(\phi_R^*\phi_L)
=\frac{2mc^2}{\hbar}a_Ra_L
\sin\left(\frac{S_L-S_R}{\hbar}\right),
$$

且

$$
\rho=n_R+n_L,
\qquad
j=c(n_R-n_L).
$$

$Q$ 显式依赖相对相位，可区分“总概率守恒但通道交换错误”的近似解。但该方程仍由 Dirac 残差导出，它的候选作用是改善表示和梯度条件，而非增加新物理定律。

### 4.4 损失函数与分支诊断

$$
\mathcal L=
\lambda_D\mathcal L_D+
\lambda_{\rm ic}\mathcal L_{\rm ic}+
\lambda_{\rm bc}\mathcal L_{\rm bc}+
\lambda_{\rm ch}\mathcal L_{\rm ch}+
\lambda_{\rm data}\mathcal L_{\rm data},
$$

其中 $\mathcal L_D$ 是两个复特征方程的强残差，

$$
\mathcal L_{\rm ch}=
\operatorname{MSE}[(\partial_t+c\partial_x)n_R-Q]
+\operatorname{MSE}[(\partial_t-c\partial_x)n_L+Q].
$$

总连续性、总范数和总能量原则上作为独立评价量；若加入训练，必须单独消融。总能量守恒仅适用于 $\partial_tV=0$ 的封闭系统。损失权重先比较固定权重与梯度范数平衡，不把通用权重算法另立为创新。

在自由或周期问题中，使用

$$
\Pi_\pm(k)=\frac12\left[
I_2\pm\frac{\hbar ck\sigma_x+mc^2\sigma_z}{E_k}
\right]
$$

计算分支占据 $N_\pm=\|\Pi_\pm\widehat\psi\|_2^2$。特征通道不等于正负能支，故 $\Pi_\pm$ 首先只作为独立诊断，避免用评价定义偏向新方法。

---

## 5. 研究目标与技术路线

### 5.1 研究目标

1. 给出公共相位消去、连续性残差投影和混合能支守恒盲点的可核验推导。
2. 用纯正能支与混合能支的受控实验，绘制直接 CPA 关于 $m,k,T,\eta$ 的适用边界。
3. 构造 SDP-Dirac-PINN，分离验证双相位表示和通道交换损失的作用。
4. 在解析解、长时间波包、光滑势散射和稀疏数据反演中形成完整证据链。

### 5.2 实施步骤

| 阶段 | 主要任务 | 判定标准 |
|---|---|---|
| 1. 基线复现 | 复现普通 Dirac-PINN；在一个 Schrödinger 例中复现 CPA 核心现象；实现直接 CPA-Dirac | 先排除方程、符号、自动微分和复数实现错误 |
| 2. 最小失效实验 | 用第 3.5 节解析解扫描 $\eta,m,k,T$ | 检验直接 CPA 是否“守恒指标小而电流/保真度误差大” |
| 3. 方法与消融 | 实现双相位、通道损失、节点掩膜和权重平衡 | 新方法必须改善独立测试指标，不只是自定义损失 |
| 4. 波包与长时间 | 对纯支/混合支高斯波包扫描 $k_0,\sigma,\eta,T$ | 验证宽频谱、相位累积和 Zitterbewegung 电流 |
| 5. 光滑势散射 | 加入开放特征边界与通量平衡 | 验证方法不依赖周期解析结构 |
| 6. 数据与反演 | 加入密度、电流或赝自旋稀疏含噪数据，反演 $m,V_0,\eta$ | 检验可辨识性和实验数据同化价值 |

若阶段 2 表明直接 CPA 未出现稳定退化，则否定原假设，将课题转为 CPA-Dirac 的适用边界与规范固定研究，不人为构造失败结论。

---

## 6. Benchmark 与对比设计

### 6.1 基准集

| 编号 | 问题 | 主要作用 |
|---|---|---|
| B0 | 半经典 Schrödinger 平面波 | 验证 CPA 复现正确，不作为 Dirac 结果 |
| B1 | 自由 Dirac 纯正能支平面波 | 给出直接 CPA 的正常对照 |
| B2 | 自由 Dirac 正负能支叠加 | 最小失效与守恒盲点实验 |
| B3 | 纯支/混合支高斯波包 | 长时间、宽频和电流振荡 |
| B4 | $V(x)=V_0e^{-x^2/(2w^2)}$ 光滑亚临界势 | 散射、开放边界与通量 |
| B5 | 稀疏含噪参数反演 | 数据融合、稳健性和可辨识性 |

高斯波包在 Fourier 空间取

$$
\widehat\psi(k,0)=g_{\sigma,k_0}(k)[u_+(k)+\eta u_-(k)].
$$

对 B4，波包反射率与透射率应由时间积分边界通量定义，而不是单个时刻的电流比值：

$$
R=\frac{\int_{t_0}^{t_1}[-j(a,t)]_+\,dt}{\int_{t_0}^{t_1}[j_{\rm in}(t)]_+\,dt},
\qquad
T=\frac{\int_{t_0}^{t_1}[j(b,t)]_+\,dt}{\int_{t_0}^{t_1}[j_{\rm in}(t)]_+\,dt}.
$$

### 6.2 PINN 基线

1. Cartesian 复分量 Vanilla Dirac-PINN。
2. Lorin-Yang 型 Dirac-PINN，含范数约束或时间域分解 [2]。
3. 单公共相位的直接 CPA-Dirac，仅含强残差。
4. 直接 CPA-Dirac，加入总连续性、范数和适用时的能量约束。
5. 时间分段或因果 PINN，排除改进仅来自时间分块。
6. 所提 SDP-Dirac-PINN。

### 6.3 传统数值参考

- 自由和周期问题：解析 Fourier 传播或 Fourier 伪谱方法。
- 光滑静势：时间分裂 Fourier 伪谱或指数波积分方法。
- 非相对论极限：一致精度多尺度时间积分方法 [9,10]。

参考解必须通过网格和时间步收敛检查，使其误差显著小于 PINN 误差。一维前向问题中，PINN 通常不会全面快于谱方法；应将传统方法作为可信真值，将 PINN 优势聚焦于稀疏数据同化和参数反演。

### 6.4 公平性要求

- 匹配可训练参数量、collocation 点、初边值点、优化器、迭代次数和数值精度。
- 各方法使用相同的调参预算与预先声明的搜索空间。
- 主要配置至少 5 个、建议 10 个随机种子；报告中位数、四分位区间、bootstrap 置信区间和失败率。
- 同时比较固定训练成本下的误差，以及达到指定误差所需成本。
- 结论必须基于独立测试网格，不使用训练损失作为主要精度证据。

---

## 7. 评价指标与消融实验

### 7.1 独立评价指标

旋量相对误差和去除最优整体相位后的误差为

$$
E_\psi(t)=
\frac{\|\widehat\psi-\psi_{\rm ref}\|_2}{\|\psi_{\rm ref}\|_2},
$$

$$
E_{\rm proj}(t)=
\min_{\gamma\in[0,2\pi)}
\frac{\|e^{i\gamma}\widehat\psi-\psi_{\rm ref}\|_2}
{\|\psi_{\rm ref}\|_2}.
$$

保真度为

$$
F(t)=
\frac{|\langle\psi_{\rm ref},\widehat\psi\rangle|^2}
{\|\psi_{\rm ref}\|_2^2\|\widehat\psi\|_2^2}.
$$

密度和概率流误差为

$$
E_\rho=
\frac{\|\widehat\rho-\rho_{\rm ref}\|_2}{\|\rho_{\rm ref}\|_2},
\qquad
E_j=
\frac{\|\widehat j-j_{\rm ref}\|_2}{\|j_{\rm ref}\|_2+\epsilon_j}.
$$

同时报告：

- 非零振幅区域的包裹相对相位误差；
- $N_+(t)$、$N_-(t)$ 及分支相对相位；
- Zitterbewegung/电流振荡的频率、振幅和相位偏差；
- 连续性误差、范数漂移、适用时的能量漂移和开放边界通量误差；
- 参数量、采样点、总训练时间、峰值显存和达到给定 $E_j$ 或 $1-F$ 的成本。

### 7.2 消融矩阵

1. 公共相位改为双相位，不加通道损失。
2. Cartesian 复输出+总连续性。
3. Cartesian 复输出+通道交换。
4. 双相位+总连续性。
5. 双相位+通道交换，即完整方法。
6. 完整方法去掉节点掩膜。
7. 固定损失权重与梯度平衡权重。

该矩阵用于判断改进来自变量表示、物理投影、权重重分配，还是单纯增加模型容量。必要时补充损失梯度夹角和条件数分析。

---

## 8. 真实数据约束与通用性

### 8.1 数据约束

数据损失可写为

$$
\mathcal L_{\rm data}=
w_\rho\|\widehat\rho-\rho^{\rm obs}\|_2^2+
w_j\|\widehat j-j^{\rm obs}\|_2^2+
w_s\|\widehat{\mathbf s}-\mathbf s^{\rm obs}\|_2^2,
$$

其中 $\mathbf s=(\psi^\dagger\sigma_x\psi,\psi^\dagger\sigma_y\psi,\psi^\dagger\sigma_z\psi)$ 为赝自旋观测量。反演参数可取有效质量 $m$、势强 $V_0$ 或初始混合系数 $\eta$。设置 $0\%$、$1\%$、$5\%$ 噪声，比较“仅密度”与“密度+概率流/赝自旋”的可辨识性。由传统求解器生成的合成数据只用于方法验证，不称为真实实验数据。

囚禁离子已用于量子模拟一维 Dirac 方程及 Zitterbewegung [12]，光子晶格和光量子元胞自动机也已观测相关振荡 [13,14]。方法的潜在用途是将稀疏位置分布、内态布居或光强数据与 Dirac 动力学联合，重建连续旋量、反演有效参数并诊断相位漂移。未获得公开真实数据前，这些只能表述为潜在应用。

### 8.2 方程推广

- **无质量 Dirac/Weyl 方程：** $m=0$ 时 $Q=0$，左右通道解耦，可用于检验算法的退化一致性。
- **光学耦合模方程：** 具有相似的一阶双通道结构，通道布居和相对相位可直接观测。
- **二维 Dirac 材料模型：** 可扩展至石墨烯或光子晶格，但需重新处理角向赝自旋和多方向特征。
- **Pauli、Bogoliubov-de Gennes 和多分量 Schrödinger 方程：** 同样含内部相对相位，但内积、谱对称性和守恒律不同，不能无修改地直接套用。

若目标是通用科学机器学方法论，至少需再选一类非 Dirac 多分量方程进行验证；若只完成一维 Dirac，则应定位为计算物理中的专用算法。

---

## 9. 预期创新、难点与成功标准

### 9.1 候选创新

1. **机制创新：** 从概率流和残差恒等式出发，区分 CPA 的“表示可迁移”与“结构保持机制不等价”。
2. **方法创新：** 以旋量特征通道的双相位表示显式暴露内部相对相位，并用质量耦合引起的通道交换关系调节优化。
3. **证据创新：** 建立混合能支最小反例和 failure map，并以旋量保真度、相对相位、分支占据和概率流而非单一守恒量评判 Dirac-PINN。

上述三项需形成“理论命题—反例实验—算法修正—多层次验证”闭环，不应拆成多个松散创新点。

### 9.2 主要风险

| 风险 | 应对 |
|---|---|
| 直接 CPA 未显著退化 | 否定负迁移假设，转向适用边界图，不选择性报告 |
| 双相位在节点奇异 | 振幅掩膜、Cartesian 回退、从无节点解析例递进 |
| 通道损失只是残差重加权 | 做表示/损失消融、梯度夹角和独立指标比较 |
| 新方法仅凭容量取胜 | 报告等参数量和等训练时间两组比较 |
| 一个解析例无法支撑论文 | 完成混合平面波、波包、光滑势和参数扫描 |
| 传统谱方法更快 | 不声称全面替代，聚焦 PINN 变体、数据同化和反演 |

### 9.3 达到投稿水平的最低闭环

面向 SCI 中科院 2–3 区水平，至少需要：理论命题及适用条件；直接 CPA 的纯支对照和混合支退化证据；稳定的新方法；三类以上主要 Dirac benchmark；四类以上 PINN 基线和可信传统方法；完整消融、参数扫描、多种子统计、失败案例和成本报告；可复现代码与原始指标。

若追求 CCF-B 级计算机会议，一维 Dirac 专用方法通常不足，还需将方法提炼为一阶多分量双曲系统的内部相位感知 PINN，完成跨方程验证、现代科学机器学基线和复杂度分析。分区与会议录用不能由课题设计保证，最终取决于效果量级、证据完整性和投稿时的分区口径。

---

## 10. 进度安排

| 时间 | 任务 | 停止/通过条件 |
|---|---|---|
| 第 1–2 个月 | 补充 Dirac 旋量与谱分支；复现 Dirac-PINN、CPA 和直接 CPA-Dirac | 自由平面波/波包与 Schrödinger 对照通过 |
| 第 3 个月 | 完成推导、B1–B2 和失效扫描 | 接受或否定负迁移假设 |
| 第 4–5 个月 | 实现双相位、通道损失和核心消融 | 必须改善独立旋量/电流指标，否则重设计 |
| 第 6–7 个月 | B3 长时间波包、多种子与 failure map | 完成机制层面证据 |
| 第 8 个月 | B4 光滑势、开放边界和传统方法对照 | 完成通量与散射验证 |
| 第 9 个月 | B5 含噪反演、成本与全部消融 | 形成独立统计证据 |
| 第 10 个月 | 复查文献、整理代码和论文 | 提供可复现配置、种子和原始表格 |

---

## 11. 预期结论

本课题不应以“CPA-PINN 完全无法推广到 Dirac 方程”立题。更准确的研究命题是：直接迁移在表示上可行，但其公共相位与总守恒机制在混合能支 Dirac 动力学中存在可检验的相对相位盲点。课题首先确认该理论差异是否会导致数值退化，再以旋量特征通道双相位和通道概率交换修正。只有当新方法在匹配预算下稳定改善旋量保真度、相对相位和概率流，并经受波包、光滑势、消融、传统方法与含噪数据反演验证时，才能形成足以投稿的算法贡献。

---

## 12. 参考文献

[1] S. Kumar, H. K. Dwivedi, and Rajeev, “Structure-preserving conservative phase-amplitude PINNs for semiclassical Schrödinger-type equations,” *Journal of Computational and Applied Mathematics*, 485, 117607, 2026. [https://doi.org/10.1016/j.cam.2026.117607](https://doi.org/10.1016/j.cam.2026.117607)

[2] E. Lorin and X. Yang, “Time-dependent Dirac Equation with Physics-Informed Neural Networks: Computation and Properties,” *Computer Physics Communications*, 280, 108474, 2022. [https://doi.org/10.1016/j.cpc.2022.108474](https://doi.org/10.1016/j.cpc.2022.108474)

[3] X.-K. Du and S.-Q. Zhang, “Solving nuclear Dirac Woods-Saxon potential with a physics-informed neural network,” *Communications in Theoretical Physics*, 78, 035303, 2026. [https://doi.org/10.1088/1572-9494/ae1a5c](https://doi.org/10.1088/1572-9494/ae1a5c)

[4] C. Wang, T. Naito, J. Li, and H. Liang, “A deep neural network approach to solve the Dirac equation,” *The European Physical Journal A*, 61, 162, 2025. [https://doi.org/10.1140/epja/s10050-025-01630-5](https://doi.org/10.1140/epja/s10050-025-01630-5)

[5] A. P. Giotis, A. Gkrepis, O. Kosmas, and T. S. Kosmas, “A unified Python framework for solving Schrödinger and Dirac equations in exotic lepton-antilepton atoms,” *Computer Physics Communications*, 328, 110335, 2026. [https://doi.org/10.1016/j.cpc.2026.110335](https://doi.org/10.1016/j.cpc.2026.110335)

[6] K.-E. Thylwe, “Amplitude-phase methods for analyzing the radial Dirac equation: calculation of scattering phase shifts,” *Physica Scripta*, 77, 065005, 2008. [https://doi.org/10.1088/0031-8949/77/06/065005](https://doi.org/10.1088/0031-8949/77/06/065005)

[7] L. Fabbri, “Dirac Hydrodynamics in 19 Forms,” *Symmetry*, 15, 1685, 2023. [https://doi.org/10.3390/sym15091685](https://doi.org/10.3390/sym15091685)

[8] L. Chai, E. Lorin, and X. Yang, “Frozen Gaussian Approximation for the Dirac Equation in Semiclassical Regime,” *SIAM Journal on Numerical Analysis*, 57(5), 2383-2412, 2019. [https://doi.org/10.1137/18M1222831](https://doi.org/10.1137/18M1222831)

[9] W. Bao, Y. Cai, X. Jia, and Q. Tang, “Numerical Methods and Comparison for the Dirac Equation in the Nonrelativistic Limit Regime,” *Journal of Scientific Computing*, 71, 1094-1134, 2017. [https://doi.org/10.1007/s10915-016-0333-3](https://doi.org/10.1007/s10915-016-0333-3)

[10] W. Bao, Y. Cai, X. Jia, and Q. Tang, “A Uniformly Accurate Multiscale Time Integrator Pseudospectral Method for the Dirac Equation in the Nonrelativistic Limit Regime,” *SIAM Journal on Numerical Analysis*, 54(3), 1785-1812, 2016. [https://doi.org/10.1137/15M1032375](https://doi.org/10.1137/15M1032375)

[11] C. Bajaj, L. McLennan, T. Andeen, and A. Roy, “Recipes for when physics fails: recovering robust learning of physics informed neural networks,” *Machine Learning: Science and Technology*, 4, 015013, 2023. [https://doi.org/10.1088/2632-2153/acb416](https://doi.org/10.1088/2632-2153/acb416)

[12] R. Gerritsma et al., “Quantum simulation of the Dirac equation,” *Nature*, 463, 68-71, 2010. [https://doi.org/10.1038/nature08688](https://doi.org/10.1038/nature08688)

[13] F. Dreisow et al., “Classical Simulation of Relativistic Zitterbewegung in Photonic Lattices,” *Physical Review Letters*, 105, 143902, 2010. [https://doi.org/10.1103/PhysRevLett.105.143902](https://doi.org/10.1103/PhysRevLett.105.143902)

[14] A. Suprano et al., “Photonic cellular automaton simulation of relativistic quantum fields: Observation of Zitterbewegung,” *Physical Review Research*, 6, 033136, 2024. [https://doi.org/10.1103/PhysRevResearch.6.033136](https://doi.org/10.1103/PhysRevResearch.6.033136)
