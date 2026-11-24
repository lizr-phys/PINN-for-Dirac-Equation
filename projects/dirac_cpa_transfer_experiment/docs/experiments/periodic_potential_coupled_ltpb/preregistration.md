# 弱周期外势耦合谱 LTPB 迁移实验预注册

实验标识：`periodic_potential_coupled_ltpb`。登记日期：2026-09-29。
本文件在任何神经网络正式训练之前建立。此前中断的实现阶段已做过参考候选的
初步计算；因此不声称参考校准结果对登记者完全盲。所有物理候选、门槛与后备规则
来自用户原始要求，下面的规则不会根据 learned-model 结果修改。

## 研究问题与禁止扩展

仅检验现有共享实振幅/相位表示在动量通道动态耦合时的迁移性。
不训练混合复数/极坐标模型，不添加振幅阈值正则化，不增加监督项或精确时间载波。
自由本征基底只是表示坐标；外势动力学必须由完整非对角耦合旋量残差约束。

## 参考校准与启动门槛

固定 $L=16\pi$、$m=5$、$k_0=2$、$q=1$，$k_j=k_0+jq$。
初态仅中心模态有 $(u_+(k_0)+u_-(k_0))/\sqrt2$，其余严格为零。
基底取 $u_+=(\cos(\alpha/2),\sin(\alpha/2))^T$、
$u_-=(-\sin(\alpha/2),\cos(\alpha/2))^T$，$\alpha=\operatorname{atan2}(k,m)$。

Hamiltonian 对角块为 $k_j\sigma_x+m\sigma_z$，邻接块为 $V_0 I_2/2$。
用 Hermitian 本征分解计算参考，独立矩阵指数交叉验证。
每个候选用含端点的 4001 个等距时间点校准。

依次考察 $T=5$ 的 $V_0\in\{0.1,0.2,0.5\}$；若均失败，再考察 $T=8$
的同一集合。选取首先有合格候选的时间区间中最小的 $V_0$。必须同时满足：

1. J=6 参考的最大总侧带占据在闭区间 [0.10, 0.60]；
2. J=6 最外两层总占据的时间最大值小于 $10^{-8}$（也报告各端点值）；
3. J=4 解零填充到 J=6 全状态空间后，与 J=6 复旋量完整轨迹的相对 L2
   小于 $10^{-6}$，即 $\|\iota\psi^{(4)}-\psi^{(6)}\|_{t,j,s}/\|\psi^{(6)}\|_{t,j,s}$。

同时报告最大逐时间、final-time、公共模态切片、单一全局相位对齐后的旋量差和
占据差；这些是诊断，不替代第 3 项。Hermiticity 误差不超过 $2\times10^{-15}$，
范数漂移和时间逆转相对误差不超过 $2\times10^{-13}$。
若选择成功，正式 learned 阶梯用 J=6；若候选全部失败，状态为
`CALIBRATION_BLOCKED`，禁止自动启动正式训练。不得把“占据收敛”改称“旋量收敛”。
额外 J=6/J=8 比较可作为用户决策所需的参考诊断，但不能自动替代用户指定的门槛。

## 模型与训练

三个模型及四实输出：

- `physical_vanilla`：$\psi_0(x)+\tau\,\Delta\psi(x,t)$，复旋量笛卡尔输出。
- `coupled_shared_spectral_complex`：$c_\pm=c_{\pm,0}+\tau(r_\pm+iq_\pm)$。
- `coupled_shared_spectral_ltpb`：$a_\pm=c_{\pm,0}+\tau r_\pm$，
  $S_\pm=\tau q_\pm$，$\widetilde\psi=\sum_\pm a_\pm u_\pm e^{iS_\pm}$。

$\tau=t/T$，$c_{\pm,0}=1/\sqrt2$ 仅在中心模态非零。
实振幅允许符号变化，无指数振幅参数化。所有模型严格满足初值。
4 层、每层 64 tanh；物理输入为全域周期 sin/cos 和 $t/T$，谱输入为
$(k-k_0)/(Jq)$ 和 $t/T$。物理模型 12996 参数，两共享谱模型各 12932 参数。
共享谱参数量与模态数无关，两个谱模型初始 MLP 权重逐元素相同。

固定 float64/complex128、CPU 单线程、Adam lr=0.001、5000 steps、seeds 0–4。
显式独立随机流：初始化 `500000+seed`，时间 `600000+seed`，位置 `700000+seed`。
每步重采样 32 个时间；谱模型与全部 13 模态作笛卡尔积（416 点），物理模型
与 32 个新位置作笛卡尔积（1024 点）。各模型同 seed 的逐步时间批完全一致。

唯一损失是 PDE 强残差。谱残差直接使用 $H(k)$ 和邻接旋量，
$\mathcal L=\operatorname{mean}_{t,s}\sum_j|R_j|^2/L$。
物理残差包括 $V_0\cos(qx)\psi$，损失是样本/分量均方。
依据 Parseval，两者在相同有限 Fourier 场的保留投影内对应；密集物理残差另计
外层预测模态向阶梯外生成的两项残差，避免隐去边界泄漏。
精确自由能量、矩阵谱、参考轨迹和参考占据不能进入模型、损失、优化器、配置或 checkpoint。

Smoke 仅用 20 steps、seed 0、缩小评价网格，检查尺寸、硬初值、梯度、有限值与
六件套落盘；不得用于选择超参数。无正式选择时 smoke 的候选 V0=0.2、T=5、J=6
只作工程测试，不产生正式成功判定。

## 评价、规范与相位 mask

统一密集网格 Nx=256（周期端点不重复）、Nt=201（含 t=0,T）。参考解仅在训练后调用。
主 spinor L2 对全模态/全时间计算，仅消去一个全局常数相位；物理模型额外报告
全物理场旋量误差和阶梯外占据，防止投影隐藏伪模态。密度和流使用完整物理预测。
保存 raw/aligned spinor、final-time、density、current、mean/min fidelity、最大范数
漂移、总模态占据误差、分支占据误差、中心与 ±1 侧带轨迹误差、PDE 残差、时间、
参数量与 peak RSS。

活跃条件仅由参考定义：$P_{j,\pm}(t)\ge10^{-6}$。通道相干量为
$C_{ab}=c_a c_b^*$，两通道均活跃才计入，排除 a=b。相干误差保留预测振幅塌缩的惩罚。
相位误差另要求预测占据至少 $10^{-12}$，同时报告覆盖率；保存 reference-active、
predicted-defined、pair-valid 三类 mask。未占据通道不报告普通相位误差。
初始导数和相位梯度是预定机制诊断，不改变训练。

## 自动判定

每个共享谱模型绝对成功需同时满足五 seed 均值：spinor ≤0.10、current ≤0.15、
fidelity ≥0.98、modal-population error ≤0.10、max-norm-drift ≤0.02，
且至少 4/5 seeds 各自同时满足这五个绝对条件。SD 使用 population SD（ddof=0）。

- `COMPARATIVE_ADVANTAGE`：LTPB 绝对成功，mean spinor/current 均低于同基底 Complex，
  且至少 4/5 配对 seeds 在两项同时获胜。
- `TRANSFER_SUCCESS`：LTPB 绝对成功，但未同时满足比较优势条件。
- `REPRESENTATION_FAILURE`：LTPB 未达绝对条件而同基底 Complex 达到条件。
- `INCONCLUSIVE`：两个共享谱模型均未达绝对条件。

报告全部实际误差比、逐 seed 差和其 population SD；不设置额外倍数阈值。
`CALIBRATION_BLOCKED` 属于训练前流程状态，不能冒充上述四种训练结果。

## 产物、冻结和保护

正式目录独立：`results/raw/periodic_potential_coupled_ltpb/`；图和日志使用同名独立目录。
各 run 保存 config、metrics、loss_history、time_series、dense_grid、model_state 六件套，
拒绝覆盖已完成 run；训练中断保留 staging 证据，禁止选择性重跑。
正式训练前保存本文件、图证据契约、正式代码/测试及传递依赖的 SHA-256。
目前已捕获 887 个既有研究产物的 SHA-256，完成时逐一复核。
图表契约见 [图表证据契约](figure_contract.md)。
统计图包含五个 seeds，动力学图固定 seed 0。PNG/SVG/PDF 与 source CSV 配套，PDF 渲染检查。

本实验不证明未知谱发现、一般外势或通用 Dirac 求解器能力。
零振幅下的相位不可辨识、固定相位门导致的初始导数限制、有限预算优化失败和
已知自由能支先验必须分别讨论；未经对照支持，不归因于某一种表示。
