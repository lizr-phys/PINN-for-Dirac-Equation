# 基于结构保持 PINN 的一维含时 Dirac 方程求解研究方案

## 1. 课题名称

建议中文题目：

**基于特征分解与概率流守恒约束的结构保持 PINN 方法及其在一维含时 Dirac 方程中的应用**

建议英文题目：

**A Structure-Preserving Physics-Informed Neural Network with Characteristic Decomposition and Probability-Current Conservation for the One-Dimensional Time-Dependent Dirac Equation**

为了便于后续写论文和程序命名，可将方法暂称为：

**SPCC-Dirac-PINN**  
即 **Structure-Preserving Characteristic-Current Dirac PINN**。

## 2. 课题定位

本课题的核心不是简单地验证“PINN 能否求解 Dirac 方程”，因为 PINN 求解 Dirac 方程已经有相关工作。更合理的定位是：

> 针对普通 PINN 在一维含时 Dirac 方程长时间演化、高频波包传播、不连续势垒散射和 Klein tunneling 条件下容易出现的相位误差累积、概率流不守恒、界面匹配不稳定等问题，构造一种利用 Dirac 方程物理结构的改进 PINN 方法。

因此，本课题的创新重点放在算法上，而不是放在复杂物理背景上。物理模型保持相对简单，主要选择一维含时 Dirac 方程；算法层面则引入特征分解、概率流守恒、多区域界面约束和自适应采样。

## 3. 研究对象

研究一维含时 Dirac 方程：

```math
i\partial_t\psi(x,t)
=
\left[-i\alpha\partial_x+\beta m+V(x)\right]\psi(x,t),
\qquad
\psi=(\psi_1,\psi_2)^T .
```

其中：

```math
\alpha=\sigma_x,\qquad \beta=\sigma_z,
```

```math
\sigma_x=
\begin{pmatrix}
0 & 1\\
1 & 0
\end{pmatrix},
\qquad
\sigma_z=
\begin{pmatrix}
1 & 0\\
0 & -1
\end{pmatrix}.
```

波函数 \(\psi\) 是二分量复旋量。程序实现时可将其拆为四个实函数：

```math
\psi_1=u_1+iv_1,\qquad
\psi_2=u_2+iv_2.
```

神经网络输出为：

```math
\mathcal{N}_{\theta}(x,t)
=
\left(u_1,v_1,u_2,v_2\right).
```

## 4. 由简单到复杂的物理场景

### 4.1 自由 Dirac 方程

最简单情形为：

```math
V(x)=0.
```

此时方程为：

```math
i\partial_t\psi
=
\left(-i\sigma_x\partial_x+m\sigma_z\right)\psi .
```

该情形可以使用平面波解析解：

```math
\psi(x,t)=u(k)e^{i(kx-Et)},
```

其中色散关系为：

```math
E^2=k^2+m^2.
```

这一部分主要用于检查程序是否正确，包括矩阵符号、复旋量拆分、自动微分 residual、初始条件和边界条件。

### 4.2 自由高斯波包传播

在平面波之后，引入更接近真实传播问题的高斯波包：

```math
\psi(x,0)
=
A\exp\left[-\frac{(x-x_0)^2}{2\sigma^2}\right]
u(k_0)e^{ik_0x}.
```

其中 \(x_0\) 为初始中心位置，\(\sigma\) 为波包宽度，\(k_0\) 为中心动量。

这一问题开始出现相位传播、群速度、波包展宽和长期误差累积，是从简单例子过渡到论文 benchmark 的关键。

### 4.3 阶跃势散射

阶跃势可写为：

```math
V(x)=V_0H(x),
```

其中 \(H(x)\) 为 Heaviside 函数。

该问题的重点是考察不连续势场界面处的旋量匹配和概率流连续性。

### 4.4 方势垒散射

方势垒可写为：

```math
V(x)=
\begin{cases}
0, & x<0,\\
V_0, & 0\le x\le a,\\
0, & x>a.
\end{cases}
```

方势垒散射是本课题最重要的物理 benchmark。通过调节势垒高度 \(V_0\)、宽度 \(a\)、入射动量 \(k_0\) 和质量 \(m\)，可以研究反射率、透射率以及 Klein tunneling 区域中的物理量误差。

## 5. 普通 PINN 可能遇到的困难

### 5.1 长时间演化中的相位误差累积

Dirac 方程的自由解含有相位因子：

```math
e^{i(kx-Et)}.
```

如果神经网络学到的有效能量存在误差：

```math
\widehat{E}=E+\delta E,
```

则相位误差会随时间线性累积：

```math
\delta\phi(t)=\delta E\,t.
```

因此，短时间内看起来较小的误差，在长时间传播中可能导致波函数相位明显偏离。

### 5.2 高频波包中的 spectral bias

普通神经网络通常更容易拟合低频函数，而较难拟合高频振荡。对于：

```math
\psi(x,0)\sim e^{ik_0x},
```

当 \(k_0\) 较大时，波函数振荡更快，普通 PINN 可能出现训练困难、局部误差较大或传播速度不准。

### 5.3 不连续势场中的界面误差

对于阶跃势和方势垒，势场在界面处不连续。单个光滑神经网络直接跨越界面时，可能在界面附近产生明显误差。由于 Dirac 方程是一阶方程，界面条件与二阶 Schrodinger 方程不同，需要重点关注旋量连续性和概率流连续性。

### 5.4 概率流不守恒

Dirac 方程满足概率密度和概率流定义：

```math
\rho=\psi^\dagger\psi,
\qquad
j=\psi^\dagger\alpha\psi.
```

对应连续性方程：

```math
\partial_t\rho+\partial_xj=0.
```

普通 PINN 只最小化 PDE residual 时，不一定能严格保持该守恒律，特别是在长时间传播、边界附近和势垒界面附近。

### 5.5 散射物理量误差

在散射问题中，仅仅比较波函数 MSE 不够。更重要的是反射率和透射率：

```math
R=\frac{|j_{\mathrm{reflected}}|}{|j_{\mathrm{incident}}|},
\qquad
T=\frac{|j_{\mathrm{transmitted}}|}{|j_{\mathrm{incident}}|}.
```

物理上应满足：

```math
R+T=1.
```

如果 PINN 的概率流预测不准确，即使波函数误差看似不大，也可能得到错误的散射物理量。

## 6. 拟提出的算法框架

总损失函数可写为：

```math
\mathcal{L}
=
\lambda_{\mathrm{pde}}\mathcal{L}_{\mathrm{pde}}
+
\lambda_{\mathrm{ic}}\mathcal{L}_{\mathrm{ic}}
+
\lambda_{\mathrm{bc}}\mathcal{L}_{\mathrm{bc}}
+
\lambda_{\mathrm{cur}}\mathcal{L}_{\mathrm{cur}}
+
\lambda_{\mathrm{int}}\mathcal{L}_{\mathrm{int}}.
```

其中前 3 项是普通 PINN 的基本组成，后 2 项是本课题的主要结构保持约束。

### 6.1 Dirac 方程 residual

定义：

```math
\mathcal{R}_{\theta}
=
i\partial_t\widehat{\psi}
-
\left[-i\alpha\partial_x+\beta m+V(x)\right]\widehat{\psi}.
```

普通 PDE loss 为：

```math
\mathcal{L}_{\mathrm{pde}}
=
\left\|\mathcal{R}_{\theta}\right\|_2^2.
```

### 6.2 特征分解 residual

由于一维 Dirac 方程的主传播矩阵为 \(\alpha=\sigma_x\)，其本征值为：

```math
\lambda_+=1,\qquad \lambda_-=-1.
```

通过矩阵对角化，可以将旋量转换为左右传播特征分量。记：

```math
\chi=P^{-1}\psi,
\qquad
\chi=(\chi_+,\chi_-)^T.
```

其中 \(\chi_+\) 和 \(\chi_-\) 分别对应向右和向左传播的特征分量。这样可以将 residual 的构造从原始旋量空间转移到特征变量空间，使网络训练更符合 Dirac 方程的一阶双曲传播结构。

### 6.3 概率流守恒约束

加入概率流连续性损失：

```math
\mathcal{L}_{\mathrm{cur}}
=
\left\|
\partial_t\widehat{\rho}
+
\partial_x\widehat{j}
\right\|_2^2,
```

其中：

```math
\widehat{\rho}
=
\widehat{\psi}^{\dagger}\widehat{\psi},
\qquad
\widehat{j}
=
\widehat{\psi}^{\dagger}\alpha\widehat{\psi}.
```

该约束的目标不是替代 Dirac residual，而是从物理守恒律角度抑制长期误差累积。

### 6.4 多区域界面约束

对于阶跃势和方势垒，将空间区域拆分为多个子域。若 \(x=x_I\) 是势场界面，则加入：

```math
\widehat{\psi}_L(x_I,t)
=
\widehat{\psi}_R(x_I,t),
```

```math
\widehat{j}_L(x_I,t)
=
\widehat{j}_R(x_I,t).
```

对应界面 loss：

```math
\mathcal{L}_{\mathrm{int}}
=
\left\|
\widehat{\psi}_L-\widehat{\psi}_R
\right\|_2^2
+
\left\|
\widehat{j}_L-\widehat{j}_R
\right\|_2^2.
```

### 6.5 基于 current defect 的自适应采样

定义概率流缺陷：

```math
D_{\mathrm{cur}}(x,t)
=
\left|
\partial_t\widehat{\rho}(x,t)
+
\partial_x\widehat{j}(x,t)
\right|.
```

训练过程中在 \(D_{\mathrm{cur}}\) 较大的区域增加 collocation points，尤其关注：

- 势垒边界附近；
- 反射波和透射波干涉区；
- 长时间演化后误差积累明显的区域。

## 7. Benchmark 设计

为了达到 SCI 中科院 2-3 区的工作量，建议至少设计以下 benchmark。

| 编号 | Benchmark | 主要目的 | 难度 |
|---|---|---|---|
| B1 | 自由平面波 | 检查 Dirac residual 和复旋量表示是否正确 | 低 |
| B2 | 自由高斯波包短时间传播 | 验证基本含时传播能力 | 中 |
| B3 | 自由高斯波包长时间传播 | 暴露相位误差累积和概率漂移 | 中 |
| B4 | 高频窄波包传播 | 检验 spectral bias | 中高 |
| B5 | 阶跃势散射 | 检验不连续界面处理能力 | 中高 |
| B6 | 方势垒 Klein tunneling | 检验散射物理量和概率流守恒 | 高 |
| B7 | 双势垒散射，可选 | 检验多界面干涉和长期稳定性 | 高 |

实际推进时可以先完成 B1-B3，再进入 B5-B6。B7 可作为冲更高水平论文的加强实验。

## 8. 对比方法

### 8.1 传统数值方法

传统数值方法主要用于生成 reference solution，而不是必须被 PINN 全面超过。建议至少实现一种高精度参考方法：

- Fourier pseudospectral method；
- split-step Fourier method；
- finite difference method；
- norm-preserving finite difference；
- absorbing boundary 或 transparent boundary 方法。

对于最初阶段，建议优先实现 Fourier pseudospectral 或 split-step Fourier，因为它们适合自由传播和周期边界问题。

### 8.2 已有 PINN baseline

建议比较：

- vanilla PINN；
- Fourier feature PINN；
- SIREN-PINN；
- causal PINN；
- XPINN 或 domain-decomposition PINN。

如果只和 vanilla PINN 比，论文说服力不足。至少需要 3 个 baseline；如果目标是中科院 2 区，应尽量做到 4-5 个 baseline。

### 8.3 消融实验

为了证明算法每个模块的作用，需要构造消融实验：

- 去掉特征分解；
- 去掉概率流守恒 loss；
- 去掉界面约束；
- 去掉自适应采样；
- 只保留 vanilla PINN。

消融实验是达到投稿水准的关键。没有消融实验，审稿人很容易认为方法只是经验性地堆叠训练技巧。

## 9. 评价指标

### 9.1 波函数误差

```math
E_{\psi}(t)
=
\frac{
\|\widehat{\psi}(\cdot,t)-\psi_{\mathrm{ref}}(\cdot,t)\|_2
}{
\|\psi_{\mathrm{ref}}(\cdot,t)\|_2
}.
```

### 9.2 概率归一化误差

```math
E_N(t)
=
\left|
\int\widehat{\rho}(x,t)\,dx
-
\int\widehat{\rho}(x,0)\,dx
\right|.
```

### 9.3 连续性方程误差

```math
E_c(t)
=
\left\|
\partial_t\widehat{\rho}
+
\partial_x\widehat{j}
\right\|_2.
```

### 9.4 概率流误差

```math
E_j(t)
=
\frac{
\|\widehat{j}(\cdot,t)-j_{\mathrm{ref}}(\cdot,t)\|_2
}{
\|j_{\mathrm{ref}}(\cdot,t)\|_2
}.
```

### 9.5 散射物理量误差

```math
E_R=
|R_{\mathrm{PINN}}-R_{\mathrm{ref}}|,
\qquad
E_T=
|T_{\mathrm{PINN}}-T_{\mathrm{ref}}|,
```

```math
E_{RT}
=
|R_{\mathrm{PINN}}+T_{\mathrm{PINN}}-1|.
```

### 9.6 计算成本

还应记录：

- 训练时间；
- collocation points 数量；
- 网络参数量；
- 相同训练预算下的误差；
- 达到指定误差所需的训练轮数。

## 10. 具体研究路径

### 第一阶段：完成最小 Dirac-PINN

目标：求解自由 Dirac 方程平面波解析解。

需要完成：

- 复旋量拆分；
- Dirac residual；
- 初始条件；
- 周期边界条件；
- 与解析解对比；
- 输出 loss 曲线、波函数图像、概率密度和概率流。

该阶段已经可以作为代码基础，但还不是论文创新点。

### 第二阶段：自由高斯波包传播

目标：从平面波过渡到真正的波包传播。

需要完成：

- 构造高斯波包初值；
- 使用传统数值方法生成 reference solution；
- 比较 PINN 与 reference solution；
- 计算概率密度、概率流和波函数误差。

### 第三阶段：暴露普通 PINN 的失败条件

目标：系统说明普通 PINN 在哪些条件下表现不好。

重点测试：

- 增大最终时间 \(T\)；
- 增大中心动量 \(k_0\)；
- 减小波包宽度 \(\sigma\)；
- 改变质量 \(m\)；
- 增加 collocation 点但保持网络结构不变；
- 改变网络深度和宽度。

预期现象：

- 相位误差随时间累积；
- 波包中心位置偏移；
- 概率归一化发生漂移；
- 概率流误差增大；
- 高频情况下训练误差下降变慢。

### 第四阶段：加入概率流守恒 loss

目标：验证 current loss 是否能改善长期物理一致性。

需要比较：

- vanilla PINN；
- vanilla PINN + current loss。

重点看：

- \(E_N(t)\) 是否下降；
- \(E_c(t)\) 是否下降；
- 长时间 \(E_{\psi}(t)\) 是否改善；
- 是否引入新的训练不稳定。

### 第五阶段：加入特征分解 residual

目标：利用 Dirac 方程的一阶双曲传播结构改善训练。

需要比较：

- 原始旋量 residual；
- 特征变量 residual；
- 二者结合的 residual。

重点看：

- 高频波包误差；
- 长时间相位误差；
- 训练收敛速度；
- 对不同 \(k_0\) 的泛化能力。

### 第六阶段：加入阶跃势和方势垒

目标：进入散射问题。

需要完成：

- 阶跃势 benchmark；
- 方势垒 benchmark；
- 反射率 \(R\) 和透射率 \(T\) 计算；
- 界面处旋量连续性和概率流连续性检查。

### 第七阶段：多区域界面 PINN

目标：解决不连续势场下单网络界面拟合不稳定的问题。

需要比较：

- 单网络 PINN；
- XPINN/domain decomposition PINN；
- 加入 \(\psi\) 连续约束的多区域 PINN；
- 同时加入 \(\psi\) 连续和 \(j\) 连续约束的多区域 PINN。

### 第八阶段：自适应采样

目标：提高关键区域训练效率。

采样依据：

```math
D_{\mathrm{cur}}(x,t)
=
\left|
\partial_t\widehat{\rho}
+
\partial_x\widehat{j}
\right|.
```

比较：

- 均匀采样；
- residual-based adaptive sampling；
- current-defect adaptive sampling；
- 二者结合。

### 第九阶段：完整实验与论文写作

最终论文应包括：

- 方法推导；
- benchmark 设计；
- reference solver 说明；
- 多 baseline 对比；
- 消融实验；
- 参数扫描；
- 误差机理分析；
- 代码和数据可复现说明。

## 11. 预期创新点

本课题可以形成以下创新点：

1. 提出一种面向一维含时 Dirac 方程的结构保持 PINN 框架，将特征分解和概率流守恒显式嵌入训练过程。
2. 针对长时间演化中的相位误差和概率漂移问题，构造基于连续性方程的物理一致性约束。
3. 针对不连续势场中的界面误差，提出旋量连续与概率流连续相结合的多区域界面约束。
4. 针对势垒散射中的误差集中区域，设计基于 current defect 的自适应采样策略。
5. 建立一套面向 Dirac-PINN 的 benchmark，包括自由传播、高频波包、阶跃势、方势垒和 Klein tunneling。

## 12. 主要难点

主要难点包括：

- Dirac 方程复旋量 residual 的正确实现；
- 长时间训练时 loss 权重平衡；
- probability-current loss 与 PDE residual 的协调；
- 势垒界面处子网络的稳定匹配；
- reference solver 的可靠性；
- 多 baseline 和消融实验的工作量；
- 如何证明方法改进来自物理结构，而不是单纯调参。

## 13. 投稿目标与工作量判断

如果只完成自由平面波或短时间自由波包，该工作只能作为入门训练，不能支撑投稿。

如果完成自由波包、长时间传播、阶跃势、方势垒，并与 vanilla PINN 做系统比较，可能达到普通 SCI 或中科院 3 区边缘水平。

如果目标是 SCI 中科院 2-3 区，建议至少做到：

- 5 个以上 benchmark；
- 3-5 个 PINN baseline；
- 1-2 个传统数值 reference solver；
- 完整消融实验；
- 参数扫描；
- 误差随时间增长的系统分析；
- 概率流、反射率、透射率等物理指标；
- 较完整的代码和数据复现。

如果目标是 CCF-B，则需要进一步提升算法通用性。课题不能只写成 Dirac 方程应用，而应表述为：

> 面向一阶双曲型守恒 PDE 的结构保持 PINN 框架，Dirac 方程作为主要量子力学验证对象。

这一路线难度明显更高，建议作为后续拓展，而不是第一篇论文的直接目标。

## 14. 当前最合理的推进顺序

当前建议按以下路线推进：

1. 完成自由平面波 Dirac-PINN。
2. 扩展到自由高斯波包。
3. 建立 Fourier pseudospectral 或 split-step Fourier reference solver。
4. 做长时间传播和高频波包实验，系统暴露 vanilla PINN 的不足。
5. 加入 probability-current loss。
6. 加入特征分解 residual。
7. 进入阶跃势和方势垒散射。
8. 加入多区域界面约束。
9. 加入 current-defect adaptive sampling。
10. 完成 baseline 对比、消融实验和参数扫描。

## 15. 论文主线

论文可以围绕下面这条主线展开：

> 普通 PINN 通过最小化 Dirac 方程 residual 来逼近解，但这种形式残差并不必然保证长时间演化中的相位准确性、概率归一化和概率流守恒。针对这一问题，本文从一维 Dirac 方程的一阶双曲结构和概率流守恒律出发，提出一种结构保持 PINN 方法。数值实验表明，该方法在自由波包长时间传播、高频波包传播以及势垒散射问题中，相比 vanilla PINN 和若干已有 PINN 变体，能够更稳定地保持概率流守恒，并降低波函数误差和散射物理量误差。

## 16. 近期任务

短期内不建议直接进入复杂势垒。最合理的近期任务是：

- 在已有自由平面波程序基础上，加入高斯波包初值；
- 实现一个传统数值 reference solver；
- 比较 vanilla PINN 在不同 \(T\)、\(k_0\)、\(\sigma\) 下的误差；
- 画出 \(E_{\psi}(t)\)、\(E_N(t)\)、\(E_c(t)\) 随时间变化的曲线；
- 初步判断普通 PINN 最明显失败的参数区域。

这一阶段完成后，就可以明确后续算法创新应该优先解决“长时间误差”还是“高频误差”。
