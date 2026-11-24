# 论文前理论闭环核验

状态：冻结模型的推导核验材料，非论文正文或全实验验收。物理问题为周期一维Dirac初值演化；本轮有限模态、已知自由能支基底、固定实标量势与初态范围均保持原协议。

## 符号与单位

| 符号 | 定义与范围 | 自然单位下量纲 |
| --- | --- | --- |
| (L,k_n,n) | (L=16\pi, k_n=2\pi n/L=n/8)，冻结161模态 | (L, L^{-1},1) |
| (m,E_n,V,\Omega_{n,b}) | 质量、自由能量、标量势、学习频率；(E_n=\sqrt{k_n^2+m^2}) | (t^{-1})，取(\hbar=c=1) |
| (T,t,\tau) | 固定窗口、物理时间、(\tau=t/T) | (t,t,1) |
| (s_n,c_{0,n},r_n,\Gamma_n) | 物理旋量谱系数、初始能支系数、原始/归一复包络，各有两复分量 | 无量纲 |
| (U_n,P_n,\phi_{n,b}) | 正交自由本征基、单位模对角载波、实相位 | 无量纲 |
| (R_n) | (i\dot s_n-(Hs)_n)，包括需要注明的谱外部分 | (t^{-1}) |

波函数采用 (\psi(x,t)=L^{-1/2}\sum_n s_n(t)e^{ik_nx})，故(\psi)量纲为(L^{-1/2})，范数积分无量纲。本文量纲表只说明已冻结数值单位，未引入新物理参数。

## 推导链审查

**A1：旧相位门的限制是初始虚部切向。** [原LTPB源码](frozen/source/closure/vendor/periodic_potential_models.py)实际为
\[
c_{n,b}(t)=[c_{0,n,b}+\tau a_{n,b}(t)]e^{i\tau q_{n,b}(t)},\qquad a,q\in\mathbb R.
\]
在原始能支谱系数(c_0)为实、网络参数有限且网络平滑时，
\[
\dot c_{n,b}(0)=a_{n,b}(0)/T+i c_{0,n,b}q_{n,b}(0)/T.
\]
若该通道初始为空，虚部切向恒为零，且其对任何有限网络权重的灵敏度为零。振幅仍可在后续时间非零；不能将此写成“空通道永远为零”。

实的初始旋量谱系数(s(0))与实谱Hamiltonian要求初始导数为(-iHs(0))。这里“实”指谱系数，并不声称叠加(e^{ik_nx})后的空间波函数为实。例：将旧公式用于本轮原Gaussian谱初态，(V=0.2\cos x)连接(n\leftrightarrow n\pm8)；初始(s_{40}=s_{48}=0, s_{32}\ne0)，从而 (\dot s_{40}(0)=-0.1i s_{32}(0)\ne0)。转换到实正交能支基底后仍为非零虚向量，说明该相位门不能表示这项初始切向。此例是已冻结物理下的代数检验，不将旧R6的单模初态改称Gaussian试验。

相位释放使空通道成为(\tau a e^{iq})，其初始导数为(a(0)e^{iq(0)}/T)，在(a(0)\ne0)时允许不同复方向；见[释放源码](frozen/source/closure/vendor/r7_models.py)及[历史切向核验](frozen/source/closure/vendor/r7_theory.py)。(a(0)=0)仍有退化。局部切向可表示不是长时训练成功证明；时刻零的缺陷亦不能单独下界整个时间积分损失。R7后续失败必须保留。

此结论对应历史实振幅/相位门。R14的`original_pde`、`original_hard`亦已采用[同形B_complex复包络](frozen/source/closure/models.py)，名称中的original指旧的含时间载波；不能将它们当前的失败直接归因于A1的旧包络切向限制。

**A2：最终复包络允许复切向，硬范数只投影径向分量。** [最终源码](frozen/source/closure/models.py)采用 (r=c_0+\tau z_\theta, z_\theta\in\mathbb C^2)，对每个时刻的完整模态/能支集合计算
\[
N=(\sum_{n,b}|r_{n,b}|^2)^{1/2},\quad \Gamma=r/N,
\quad \dot\Gamma={\dot r\over N}-{r\,\operatorname{Re}(r^\dagger\dot r)\over N^3}.
\]
因此 (\sum|\Gamma|^2=1)，且 (2\operatorname{Re}(\Gamma^\dagger\dot\Gamma)=0)。初始空通道(c_{0,n,b}=0)的导数为复值(z_{n,b}(0)/T)，没有A1强制的纯实切向限制。这是参数化的代数许可，有限共享网络容量、优化及可达性仍需实验。

[投影与导数源码](frozen/source/closure/vendor/r13_models.py)对零/非有限(N)明确失败，不加clamp。推导在(N>0)、网络平滑时成立；不能声称零点也已定义。

**A2补充：单轨迹范数恒定不等于酉传播算子。** [本轮初态是固定buffer](frozen/source/closure/vendor/r9_models.py)，[实际前向](frozen/source/closure/models.py)为固定任务参数下的 \(c_0+\tau z_\theta(k,\tau)\) 归一化；模型没有实现以任意初态为输入的线性传播算子接口。即使形式地在同一网络/载波上替换初态，其映射也通常是非线性的：
\[
F_\tau(c_0)=C_\tau\frac{c_0+\tau z_\theta}{\|c_0+\tau z_\theta\|},\qquad C_\tau=UP_\tau.
\]
\(C_\tau\)在保留的谱子空间上保持内积，但前面的仿射偏移与状态依赖归一化不保证内积保持。代数例：取两个单位正交谱向量\(v,w\)，在\(\tau=1\)令固定输出\(z_\theta=v\)，则
\[
F_1(v)=C_1v,\quad F_1(w)=C_1(v+w)/\sqrt2,\quad
\langle F_1(v),F_1(w)\rangle=1/\sqrt2\ne\langle v,w\rangle=0.
\]
此例可用均匀正能支\(v\)、均匀负能支\(w\)及网络末层常数偏置表示，两个分母均非零。它是归一结构的代数反例，不是新增初态实验；没有替换任何现存模型buffer、增加训练或改变45项零样本方案。因此应写为“固定初态轨迹的精确总范数参数化”，不能仅凭范数结构推出学习了线性酉演化算子、两条轨迹内积守恒或未知初态的叠加推广。新初态重训、载波复用与原模型零样本预测仍按预定结果分别报告。

**A3：精确范数不等于正确演化。** 因(U_n^\dagger U_n=P_n^\dagger P_n=I)，(s_n=U_nP_n\Gamma_n)经Parseval保持单位总概率，任意实载波相位均如此。令同一自伴Hamiltonian的初态轨迹错误地以双倍时间传播，(\widetilde\psi(t)=e^{-2iHt}\psi_0)，初态及范数仍正确，但固定方程残差为(H\widetilde\psi)，一般不为零。故范数机器精度必须标为结构恒等式；旋量、概率流、占据、保真度及相干量是独立动力学证据。实际反例/实现一致性见[独立指标检查](engineering/independent_metric_tests.json)。

**A4：近似载波必须保留自由缺陷。** 对物理谱Hamiltonian，(U^\dagger H_0U=D=\operatorname{diag}(E_n,-E_n))，势项记为(W=U^\dagger VU)。定义 (P=e^{-it\Omega}) 后
\[
i\dot\Gamma=(D-\Omega)\Gamma+P^\dagger W P\Gamma.
\]
仅解析载波(\Omega=D)消去自由缺陷；这与相互作用绘景有关。一般 (P=e^{i\phi(t)}) 的自由项为(D+\dot\phi)。[完整残差源码](frozen/source/closure/models.py)直接计算(i\dot s-Hs)，保留该项；学习组不将近似频率冒充精确频率。

所有组都使用已知自由本征基；“无未来参考轨迹/无能量标签”不等于“无解析物理先验”。载波与复包络之间的相位重分配可保持同一物理(s)，故联合训练的潜相位不具唯一物理解读；不声称任意重分配都可由有限网络精确实现。

**A4补充：静态自由预训练等价于已知本征频率的回归。** [实际自由残差](frozen/source/closure/vendor/r13_residual.py)分别检验两条单位旋量(u_+p_+,u_-p_-)；已知(H_0u_\pm=\pm E_nu_\pm)、(\|u_\pm\|=1)且(p_b=e^{-it\Omega_b})，故
\[
R_{n,+}=(\Omega_{n,+}-E_n)u_{n,+}p_{n,+},\qquad
R_{n,-}=(\Omega_{n,-}+E_n)u_{n,-}p_{n,-}.
\]
[本轮free_loss](frozen/source/closure/models.py)对样本、两能支及两旋量分量取平方模均值。全模态的(N_tN_k)配对点由此给出
\[
\mathcal L_{\rm free}=\frac{1}{4N_tN_k}\sum_{j=1}^{N_t}\sum_{n=1}^{N_k}
\big[(\Omega_{n,+}-E_n)^2+(\Omega_{n,-}+E_n)^2\big]
=\frac{1}{4N_k}\sum_n\big[(\Omega_{n,+}-E_n)^2+(\Omega_{n,-}+E_n)^2\big].
\]
数学上静态损失不依赖采样时间，浮点实现另有舍入。本轮仍保留原32个时间样本、私有RNG、6000步及全部工作量计数；此推导不授权删去采样或改训练。一般实相位(p=e^{i\phi})的两支缺陷是(-\dot\phi_+-E_n)与(-\dot\phi_-+E_n)，平方模同样约束已知自由谱上的相位斜率。

因此“不在训练接口输入(E_n)数组”是源码事实，不意味着预训练未知自由物理：已知算符和已知自由本征基已经使其等价于频率回归。它仍与未来耦合参考轨迹隔离。应将此阶段写成已知自由动力学的参数化/近似，实际价值须由同结构解析载波的完整五配对精度与全费用比较判定。本段是冻结方程的代数核验，未新增训练、调参或数值实验。

**A5：谱内残差与完整物理残差必须分清。** [Hamiltonian及谱外源码](frozen/source/closure/vendor/r12_hamiltonian.py)对零延拓计算谱外(-V\psi)，双谐波重叠来源需相干相加。训练投影后的有限Hamiltonian仍可厄米并保持范数，不能据此证明谱外残差为零。参考的独立截断校准和模型谱外泄漏记录分别提供证据。

**A6：离散残差积分不是连续证书。** 在同一自伴Hamiltonian、足够正则轨迹下，变分常数公式给出
\[
\|\widetilde\psi(t)-\psi(t)\|\le\|\widetilde\psi(0)-\psi(0)\|+\int_0^t\|i\dot{\widetilde\psi}-H\widetilde\psi\|\,ds.
\]
此式依赖真实连续完整残差。本轮201个残差样本的独立JVP与4001/8001表的梯形求和不控制所有未采样时刻，也未给出严格积分余项包络；细/粗积分差仅作一致性诊断。参考截断误差另列，不能将离散积分标签改成连续误差上界。

## 量纲检查

通过：A1/A2导数均含(T^{-1})；A4的(D,\Omega,\dot\phi,W)均为频率量纲，自由预训练平方损失量纲为(T^{-2})；A6残差时间积分与状态误差同为无量纲。耦合空间平均损失为(\sum_n\|R_n\|^2/(2L))，自由单位载波RMS另有不同平均方式，二者数值不作跨任务直接比较。

## 假设与可验证性

必要假设为周期正交Fourier基、实势和自伴Hamiltonian、完整每时刻模态归一、单位模实相位载波、(N>0)及平滑网络。已知初值/自由基允许进入训练，未来解仅在独立评价中使用。

已执行的对应验证为[导数/参数梯度及精确恢复](engineering/tests_and_recovery.json)、[载波初态控制](analysis_outputs/carrier_initialization_audit.json)、[实际权重及随机流配对](analysis_outputs/pairing_audit.json)、[独立残差](analysis_outputs/residual_audit.json)和[参考校准](references/calibration.json)。各文件有范围与缺项，available级通过不表示全部新训练完成。

## 修改建议与证据边界

必须修正的表述：空通道问题应写为“具体相位门的初始虚部切向限制”；总概率与潜相位一致性不替代动力学；近似自由载波保留缺陷；离散残差不宣称连续证书。

需等待实验的结论：载波修复、范数结构的独立收益及交互；联合/无载波是否达到成功规则；学习载波相对解析载波的精度与总成本；六条件与零样本的适用边界。预训练与联合比较同时涉及辅助目标、初始化、冻结及活跃参数，不能唯一拆分每一贡献，也不预设预训练不可少。

载波修复对照亦是组合参数化的比较：由含时间相位变为自主频率，包含输入维度、输出单位/缩放及64个参数的变化。实际初始物理斜率匹配和共同包络权重控制已分别核验；不能将全部差异唯一归因于删去时间输入或单一缩放操作。
