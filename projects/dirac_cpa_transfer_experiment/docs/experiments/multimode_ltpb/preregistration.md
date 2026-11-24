# Multi-LTPB 双波数实验预注册

状态：**正式结果封存前预注册**  
登记日期：2026-08-31（Asia/Shanghai）

## 1. 研究问题与适用边界

本实验检验：当初值明确给出两个离散整数波数 `k=2,5` 时，将已知空间 Fourier 模态及其 Dirac 正负能本征旋量作为基底、但让四个通道的实振幅和相位由 PDE 残差学习，是否能把单动量 LTPB 的载波预条件优势推广到四个不同时间载频。

本实验只检验“两个已知离散动量”的自由 Dirac 问题。无论结果如何，都不外推到未知或连续动量波包、一般外势、散射、非平稳 Hamiltonian 或通用 Dirac 求解器。

## 2. 冻结的物理问题

采用自然单位下一维自由 Dirac 方程

\[
i\partial_t\psi=(-i\sigma_x\partial_x+m\sigma_z)\psi,
\quad x\in[0,2\pi),\quad t\in[0,5],\quad m=5.
\]

固定通道顺序为 `(k2,+), (k2,-), (k5,+), (k5,-)`。已知 `k=(2,5)` 及由 `H_k=k sigma_x+m sigma_z` 计算的确定性实规范本征旋量可进入 learned model；精确能量、目标相位斜率和解析时间载波不得进入 learned forward、loss、optimizer、训练配置或 checkpoint metadata。

评价阶段参考能量为 `E_l=sqrt(k_l^2+m^2)`，目标斜率按固定顺序为 `(-sqrt(29), +sqrt(29), -sqrt(50), +sqrt(50))`。精确解为

\[
\psi=\frac{1}{2\sqrt{L}}\sum_{l=1}^2
\left[u_+(k_l)e^{ik_lx-iE_lt}+u_-(k_l)e^{ik_lx+iE_lt}\right],\quad L=2\pi.
\]

四个内部初始系数均为 `a0=1/(2 sqrt(L))`，归一化空间模态的物理投影占据均为 `1/4`。

## 3. 冻结的 learned models 与参数预算

比较 Vanilla、SP-CPA、ComponentPolar、MultiBranchComplex、Multi-LTPB。前三者为 `4×64 tanh`；两个八输出多通道模型为 `4×63 tanh`。输出和参数量预期为：Vanilla `4/12996`、SP-CPA `5/13061`、ComponentPolar `4/12996`、MultiBranchComplex `8/12860`、Multi-LTPB `8/12860`。最大参数量差约 1.56%。

所有模型以 `tau=t/T` 硬满足同一初值：

- Vanilla：`psi0 + tau delta_psi`；
- SP-CPA：`(psi0 + tau delta_A) exp(i tau q)`；
- ComponentPolar：`(|psi0_j|+tau r_j)(psi0_j/|psi0_j|)exp(i tau q_j)`；
- MultiBranchComplex：每通道 `a0+tau(delta_R+i delta_I)`；
- Multi-LTPB：每通道 `(a0+tau r)exp(i tau q)`。

另设一个直接写入四个精确时间载频的 Four-carrier oracle，只作机制上限诊断，不训练、不进入 learned 排名或成功判据。

## 4. 冻结的训练和随机流协议

- learned seeds：`0,1,2,3,4`，共 25 个正式 run；不根据结果选择性重跑；
- `torch.float64/complex128`，CPU；
- 输入 `[sin(x),cos(x),t/T]`；
- Adam，学习率 `1e-3`，恰好 5000 steps；
- 每步独立重采样 1024 个均匀时空配点；
- 损失严格为 Dirac 强形式 `L_PDE`，不加入解析标签、相位、守恒量或其他监督；
- 同 seed 的各模型使用同一初始化种子协议及相同配点 generator seed；模型初始化与配点各用独立 generator；
- 正式测试网格固定为 128 个不重复周期端点的 `x` 点和 201 个含 `t=0,5` 的时间点；
- loss history 保存 step 1 以及每 10 step，直至 step 5000；
- smoke 仅检查形状、梯度、NaN 和落盘，不改模型、步数、学习率、宽度或判据；smoke 结果与正式目录隔离。

## 5. 冻结的指标定义

主要指标为全时空 spinor relative L2、current relative L2 和 mean fidelity；辅助指标为 density relative L2、四个物理模态投影系数误差、占据误差、四相位斜率及相对误差、两个支间和两个波数间相对相位、dense PDE residual、范数漂移、训练时间、参数量和训练曲线。

模态系数从预测总场统一投影：

\[
\hat c_{l,s}(t)=\frac1L\int_0^L e^{-ik_lx}u_s(k_l)^\dagger\hat\psi(x,t)\,dx.
\]

它们而非 Multi-LTPB 的内部 `S(x,t)` 用于预注册相位判定；内部相位只作重参数化复杂度诊断。

为不计一个不可观测的整体常数相位，使用全时空所有点共同确定的**唯一常数**相位对齐；同一相位同时作用于总旋量和全部四个模态系数。禁止逐时间、逐空间或逐模态对齐，以免消去共同的错误时间频率。另保存 raw spinor error。Fidelity、密度、电流、占据和相对相位本身规范不变。

四条投影相位用幅度平方加权线性拟合。有效阈值固定为 `|c_hat| > 1e-6 a0`，有效比例低于 90% 时该 seed/通道的 slope 判为无效，成功判据中按不通过处理。通道平均 slope error 是五 seed 的绝对相对误差算术平均，不允许先平均有符号斜率抵消误差。

均值和标准差均跨五 seeds 计算，标准差使用 population SD (`ddof=0`)。

## 6. 冻结的自动成功判据

learned baseline 集合固定为 Vanilla、SP-CPA、ComponentPolar、MultiBranchComplex；每个主要误差分别从四者的五 seed 均值中取最佳者。总判定 PASS 当且仅当以下全部成立：

1. Multi-LTPB mean spinor error `<=0.05`；
2. Multi-LTPB mean current error `<=0.10`；
3. Multi-LTPB mean fidelity `>=0.99`；
4. Multi-LTPB mean spinor error 不超过最佳 baseline mean 的 `0.1`；
5. Multi-LTPB mean current error不超过最佳 baseline mean 的 `0.1`；
6. 至少 4/5 seeds 中，Multi-LTPB 的 spinor 和 current 同时不超过同 seed 最佳 baseline 的 `0.1`；
7. 四个可辨识投影相位斜率中，至少三个的五 seed 平均绝对相对误差严格 `<0.01`。

边界比较按上述 `<=`、`>=` 和严格 `<` 执行。若失败，不增加特定模型预算或只报告有利重跑；报告将区分表示能力、优化未收敛、模态投影不可辨识、参数预算和物理先验的作用。

## 7. 产物、审计与保护

正式 learned run 写入 `results/raw/multimode_ltpb/learned/{model}/seed_{seed}/`，每个 run 固定保存 `config.json`、`metrics.json`、`loss_history.csv`、`time_series.csv`、`dense_grid.npz`、`model_state.pt`。Oracle 单独存放。汇总、判定、图表 source data、三种矢量/位图格式、FIGURE_QA、报告、链接检查和独立结果审计均为必需产物。

旧结果保护清单已在正式实验前生成：`results/multimode_ltpb_protected_baseline.json`，记录 438 个旧文件，聚合 SHA-256 为 `037ffc9af31449f2a5c280f48c4105d9bef71fab1d07f58fbe7a9bae6ec3e9b3`。正式审计必须验证这些哈希未变化。

## 8. 预注册时读取证据的 SHA-256

| 文件 | SHA-256 |
|---|---|
| `RESEARCH_PROGRESS_SUMMARY.md` | `52f4aea9944ae8ec74257e60fe0e063a164bb003976b670218beb1367b9d0e61` |
| `LEARNED_TWO_PHASE_PILOT_REPORT.md` | `ffbd3adb4c50697623dea90e40f6559278af2ee5785430cc24009a6690ebb1a4` |
| `LTPB_MECHANISM_ABLATION_REPORT.md` | `86e72777b0adc6fcab918bf31c9321a73741999cdcef27333a97c90955ecfef6` |
| `src/physics.py` | `14469c6dabdec935136dd73fe3d053057834af337fddf27fbe0a122d0e058d1b` |
| `src/models.py` | `4d85a6035b5f07bdffbee9aef5ae3d5f497be716e756d55973febe6053b8797f` |
| `src/train.py` | `c72b3cc3129f0561c5da92d77e8539a8d25bb8adb2e8e3aab43376a7a0298312` |

预注册本身及正式实验代码的 SHA-256 将在 smoke 通过、正式 run 启动前写入独立的 `results/multimode_ltpb_preregistration_sha256.json`；独立 companion 避免 Markdown 自引用哈希问题。
