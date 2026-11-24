# R10范数投影与解释边界

沿用R9全旋量残差R=iψ̇−Hψ。有限Fourier矩阵H为Hermitian，故

$$\dot N=2\operatorname{Re}\langle\psi,\dot\psi\rangle
=2\operatorname{Im}\langle\psi,R\rangle,\qquad N=\sum_n\|\psi_n\|^2.$$

这是复旋量PDE的恒等式；不能反向用范数正确证明旋量或相位正确。

## 1. 冻结单位模载波

定义每模态F_n=[u₊p₊,u₋p₋]，p_s=exp(iθ_s)，自由本征旋量构成正交基。F†F=I，与相位是否准确无关。令ψ_n=F_nd_n及G_n=iḞ_n−H_nF_n，H_n=k_nσ_x+5σ_z。

$$R_n=iF_n\dot d_n+G_nd_n-\frac{0.2}{2}(\psi_{n-8}+\psi_{n+8}).$$

由(F†F)̇=0可知F†Ḟ反Hermitian，iF†Ḟ及F†H_nF均Hermitian。因此

$$2\operatorname{Im}\sum_n\langle F_nd_n,G_nd_n\rangle=0.$$

这对任意可微的实载波相位成立，包括明显不准确的相位。单位载波缺陷的RMS可以很大而直接全局范数投影仍为零。

## 2. 外势耦合

截断内n±8成对耦合，实系数0.1，使势矩阵Hermitian。因此

$$2\operatorname{Im}\sum_n\langle\psi_n,0.1(\psi_{n-8}+\psi_{n+8})\rangle=0.$$

截断外预测为零，外残差等于负耦合源项；其与预测状态的全局内积也为零。保存其完整复符号仍是档案完整性的要求。

结合前述结果，

$$\dot N=2\operatorname{Im}\sum_n\langle\psi_n,iF_n\dot d_n\rangle
=2\operatorname{Re}\sum_n d_n^\dagger\dot d_n.$$

因此R9范数变化直接对应包络幅值随时间的变化。载波近似会改变包络要学习的耦合动力学和优化条件，不能由“直接投影为零”推断其对误差没有间接影响。

## 3. 可核验诊断

使用R9全部五个A固定终点档案重新计算，不改变R9任何文件。4001点完整残差投影积分对N(t)−N(0)的最大差为约0.6–1.9×10⁻⁹；载波缺陷直接投影约3×10⁻¹⁶、势投影约10⁻¹⁷。这支持恒等式和符号处理正确，不构成连续时间上界认证。保存4001/2001积分差，不以离散小差替代连续认证。

R10对同一固定祖先增加耦合预算，可以评估条件终点对预算的敏感性。它不控制优化轨迹上的所有变量，不是无预训练/准确预训练的单因素消融，也不能证明“所有范数失败均由14000步不足造成”。

所有参考Hamiltonian轨迹、精确E(k)及诊断分解只在独立后评价中出现；优化继续采用原物理H_n及完整耦合残差，不使用目标频率。
