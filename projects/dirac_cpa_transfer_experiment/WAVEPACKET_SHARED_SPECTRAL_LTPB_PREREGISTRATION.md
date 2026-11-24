# 自由 Dirac 连续高斯波包共享谱 LTPB 实验预注册

状态：**正式结果产生前冻结**  
登记日期：2026-09-01（Asia/Shanghai）

## 1. 唯一研究问题与边界

本实验只检验：在自由一维 Dirac 方程、已知 Fourier 表示和一个固定的有限周期谱截断下，单个共享网络能否在不接收精确色散关系的情况下，用波数和时间的连续函数统一表示正、负能支相位，并把在 33 个训练波数上学到的规律插值到相邻中点。

即使总判定通过，也只说明已知 Fourier 表示下自由高斯波包的有限谱结果；不说明网络发现了未知谱，不外推到连续无穷维谱、外势导致的动量耦合、散射、未知 Hamiltonian 或通用 Dirac 求解器。中点不是长度 `L` 上的整数周期 Fourier 模态，只作为参数化谱 ODE `H(k)` 的 held-out 插值诊断，不能被描述为额外的周期物理解。

## 2. 冻结的物理问题与谱规范

采用自然单位

\[
i\partial_t\psi=(-i\sigma_x\partial_x+m\sigma_z)\psi,
\quad x\in[-8\pi,8\pi),\quad t\in[0,5],\quad m=5,
\]

其中 `L=16π`。训练波数固定为 `k_n=n/8, n=0,...,32`，共 33 个；held-out 中点固定为 `k_{n+1/2}=(n+1/2)/8, n=0,...,31`，且绝不进入训练采样器或损失。

谱振幅为

\[
g_n=C\exp[-(k_n-2)^2/(4\times0.5^2)],\qquad \sum_n|g_n|^2=1.
\]

采用正交 Fourier 规范

\[
\psi(x,t)=\frac1{\sqrt L}\sum_n e^{ik_nx}\widetilde\psi_n(t),
\quad
\widetilde\psi_n=\sqrt L\,\langle e^{-ik_nx}\psi\rangle_x.
\]

所以分支初值系数为 `c_{n,+}(0)=c_{n,-}(0)=g_n/sqrt(2)`，每支总占据为 `1/2`，总范数为 1。`u_±(k)` 由已知矩阵 `H(k)=kσ_x+mσ_z` 的确定性连续实规范本征向量得到；learned model 只保存 `k,u_±,g`，不保存本征值。

精确能量、解析时间载波和目标斜率只能在 reference、post-training evaluator 与显式 oracle 中计算，不得进入 learned forward、loss、输入、optimizer、训练配置或 checkpoint。

## 3. 冻结的模型与参数预算

比较五个 learned models：

1. `physical_vanilla`：`psi0(x)+tau delta_psi(x,t)`；
2. `physical_sp_cpa`：`[psi0(x)+tau delta_A(x,t)] exp[i tau q(x,t)]`；
3. `physical_component_polar`：两个原始旋量分量分别采用硬初值实振幅和相位；
4. `shared_spectral_complex`：共享网络学习两个能支的普通复系数修正；
5. `shared_spectral_ltpb`：共享网络学习 `(r_+,r_-,q_+,q_-)`，并令 `a_±=g(k)/sqrt(2)+tau r_±`, `S_±=tau q_±`。

物理模型输入固定为与 `L` 周期匹配的 `[sin(2π(x-x_min)/L), cos(2π(x-x_min)/L), t/T]`；谱模型输入固定为 `[(k-2)/2,t/T]`。所有模型均为 4 个 hidden layers、每层 64、`tanh`。预期参数量为：Vanilla/ComponentPolar `12996`，SP-CPA `13061`，两个 Shared-Spectral 模型 `12932`，最大差约 `0.99%`。

两个 Shared-Spectral 模型必须使用完全相同的 MLP 结构、参数量、初始化权重和逐步时间样本。有效模态数改成 17、33、65 时只改变无参数的 `k,u,g` buffers，Shared-Spectral-LTPB 的可训练参数量必须保持不变；该测试不代表三个分辨率均训练成功。

Spectral oracle 直接使用精确 `E(k)`，不训练且不进入 learned 排名或判据。

## 4. 冻结的残差、采样与训练协议

物理模型直接最小化物理空间 Dirac 强残差。谱模型严格使用

\[
R_n=i\partial_t\widetilde\psi_n-[k_n\sigma_x+m\sigma_z]\widetilde\psi_n,
\]

不得将其对角化改写成含 `E(k)` 的标量支方程。由 Parseval 恒等式

\[
\int |R(x,t)|^2dx=\sum_n|R_n(t)|^2,
\]

谱训练 loss 固定为 `mean_t mean_component sum_n |R_n|^2 / L`，与物理空间 `mean_{x,t,component}|R|^2` 在离散 Fourier 子空间上数值等价。

- seeds：`0,1,2,3,4`，共 25 个 learned formal runs；
- `torch.float64/complex128`，CPU deterministic algorithms；
- Adam，学习率 `1e-3`，恰好 5000 steps；
- 每步先从 seed `300000+seed` 的独立 generator 抽取 32 个时间点；
- 谱模型将这 32 个时间与全部 33 个训练 `k` 作笛卡尔积，共 1056 点；
- 物理模型使用同一批 32 个时间，并从 seed `400000+seed` 抽取 32 个空间点作笛卡尔积，共 1024 点；
- 初始化使用与采样分离的本地 generator，seed `200000+seed`；
- loss 只有 PDE residual，无解析标签、能量、phase、norm、continuity 或监督辅助项；
- dense grid 固定为 endpoint-free `Nx=256` 与含 `0,T` 的 `Nt=201`；
- history 保存 step 1、每 10 step 和 step 5000；
- peak memory 使用训练进程的 RSS 定时采样，报告训练期间 peak RSS 与相对训练起点增量；
- smoke 仅检查形状、硬初值、梯度、NaN 和六件套落盘，使用隔离目录，不据此修改任何正式超参数或阈值；正式结果不得选择性重跑。

## 5. 冻结的评价定义

主指标为全时空、只消去一个全时空常数整体相位后的 spinor relative L2，density relative L2，current relative L2 与 mean fidelity。禁止逐时间、逐波数或逐分支规范对齐。

所有训练模态的物理系数均从预测总场作 Fourier 投影，再投影到 `u_±(k_n)`；即使模型内部提供 `S_±`，主相位指标也不直接使用内部相位。相位先 unwrap，再以预测系数模平方作时间拟合；每个通道若 `|c_hat| > 1e-6 |c_0|` 的时间比例低于 90%，则该通道不可辨识。33 个训练波数全部预先定义为有效占据模态。训练谱加权 phase-slope relative error 是按精确初始分支占据 `|g_n|^2/2` 对 66 个通道绝对相对斜率误差的加权平均；无效通道使该项失败。

held-out 指标直接把两个共享网络输入 32 个中点，使用同一训练谱归一化常数 `C` 构造中点高斯初值。相位仍由输出总谱旋量投影到 `u_±(k)` 后获得。记录 held-out spectral PDE residual、占据加权 phase-slope error、加权振幅漂移、相邻 `k` 误差跳变和斜率曲率；中点完全不进入训练。

分别用训练 Fourier 系数重建正、负能支场。对每支密度使用第一圆周矩定义中心并沿时间 unwrap；宽度使用圆周标准差 `L sqrt(-2 log |z|)/(2π)`。速度由 201 个中心点的全时段线性最小二乘斜率得到。记录每支中心轨迹 RMSE、速度及相对误差、宽度 trajectory relative L2、终态中心方向和宽化方向。若某支范数低于 `1e-8`，则中心与宽度判为不可辨识。

其他指标包括 Fourier 系数误差、模态占据误差、总范数最大漂移、dense physical-space PDE RMS、final PDE loss、参数量、训练时间和 RSS peak。均值与标准差跨五 seeds 计算，标准差固定为 population SD (`ddof=0`)。

## 6. 冻结的自动成功判据

learned baseline 固定为前四个非 oracle 模型；每个误差指标分别从其五 seed mean 取最佳值。总判定 `PASS` 当且仅当以下全部成立：

1. Shared-Spectral-LTPB mean spinor error `<=0.05`；
2. mean current error `<=0.10`；
3. mean fidelity `>=0.99`；
4. mean spinor 和 current 均不超过各自最佳 baseline mean 的 `0.1`；
5. 至少 `4/5` paired seeds 中，两项误差同时不超过同 seed 最佳 baseline 的 `0.1`；
6. 五 seed mean 的训练模态占据加权 phase-slope error 严格 `<0.01`，且各 seed 全部有效通道满足 90% 可辨识比例；
7. 五 seed mean 的 held-out 中点占据加权 phase-slope error 严格 `<0.02`，且各 seed 全部中点通道满足 90% 可辨识比例；
8. 五 seed mean 下，正、负支速度符号均正确且各自相对精确速度误差 `<0.10`，两支中心 trajectory RMSE 均 `<=0.25`，两支宽度 trajectory relative L2 均 `<0.10`，并且两支终态中心向相反方向分离、两支终态宽度相对初态的增减方向与参考一致。

边界严格按上述 `<`、`<=` 与 `>=`。中点项失败时，即使训练模态和总场通过，也只能报告“33 模态共享拟合成功”，不能报告连续谱插值成功。失败后不追加特定模型预算或只保留有利重跑；归因必须区分表示能力、优化、低占据可辨识性、共享网络预算和已知 Fourier/能支先验。

## 7. 冻结的产物与保护

每个 learned run 固定六件套：`config.json`、`metrics.json`、`loss_history.csv`、`time_series.csv`、`dense_grid.npz`、`model_state.pt`。正式目录为 `results/raw/wavepacket_shared_spectral_ltpb/learned/{model}/seed_{seed}/`；oracle 单独存放。另需逐 run、summary、comparisons CSV，decision/audit JSON，逐图 source CSV，PNG/SVG/PDF，`FIGURE_QA.md` 和最终报告中的全部 raw 直链。

密度、概率流等需要单个 run 的时空图固定使用 `Shared-Spectral-LTPB, seed=0`，不根据正式误差选择“最好”或“最典型”种子；统计图始终显示全部五个 seeds。

正式实验前的旧产物保护清单为 [results/wavepacket_shared_spectral_ltpb_protected_baseline.json](results/wavepacket_shared_spectral_ltpb_protected_baseline.json)。其记录 670 个文件，聚合 SHA-256 为 `5fa6382c7c0d88721ace5fe5067c6a1e60a33a9682933e0e8acb68a2a28164a5`。最终审计必须确认全部文件逐字节未变化。

预注册、图表证据契约、所有正式代码和测试的 SHA-256 将在 smoke 与正式前完整 pytest 通过后写入独立 companion JSON；冻结后不得修改这些文件。
