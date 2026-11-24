# 简单 Dirac 方程 PINN 求解示例

这个目录是你后续 Dirac-PINN 研究的第一个最小实验。它不做势垒、不做波包、不做概率流改进，只做一个有解析解的自由 Dirac 方程平面波，用来确认 PINN 程序的基本结构是正确的。

## 1. 物理问题

采用自然单位：

```text
hbar = c = 1
```

求解一维含时 Dirac 方程：

```text
i psi_t = (-i alpha psi_x + beta m psi)
```

其中

```text
alpha = sigma_x, beta = sigma_z
```

也就是

```text
sigma_x = [[0, 1],
           [1, 0]]

sigma_z = [[1,  0],
           [0, -1]]
```

波函数是二分量复旋量：

```text
psi = [psi_1, psi_2]^T
```

程序中神经网络输出四个实数：

```text
[Re psi_1, Im psi_1, Re psi_2, Im psi_2]
```

## 2. 解析解

自由情形下取正能平面波：

```text
psi(x,t) = u exp(i(kx - Et))
```

色散关系为：

```text
E = sqrt(k^2 + m^2)
```

正能本征旋量取为：

```text
u = N [1, k/(E+m)]^T
```

其中 `N` 是归一化系数。默认参数为：

```text
m = 1
k = 1
E = sqrt(2)
```

计算区间取：

```text
x in [-pi, pi], t in [0, 1]
```

因为 `k=1`，平面波在 `[-pi, pi]` 上满足周期边界条件。

## 3. PINN loss

总损失为：

```text
L = lambda_pde L_pde + lambda_ic L_ic + lambda_bc L_bc
```

其中：

- `L_pde`：Dirac 方程 residual；
- `L_ic`：初始条件 `t=0` 上与解析解匹配；
- `L_bc`：周期边界条件 `psi(-pi,t)=psi(pi,t)`。

这个例子暂时不加入 probability-current loss。原因是第一步应先验证 vanilla PINN 的基础实现，后面再加入结构保持改进。

## 4. 运行

在项目根目录运行：

```powershell
python .\dirac_pinn_simple\main.py
```

如果想先快速测试：

```powershell
python .\dirac_pinn_simple\main.py --epochs 500 --collocation 1024
```

结果会保存到：

```text
dirac_pinn_simple/results
```

主要输出文件：

- `loss_curve.png`：训练损失曲线；
- `final_components.png`：最终时刻四个实/虚分量的预测与解析解对比；
- `density_current.png`：概率密度和概率流对比；
- `metrics.txt`：最终相对误差；
- `prediction_final_time.csv`：最终时刻的数值数据。

## 5. 下一步

这个例子跑通后，建议按下面顺序继续：

1. 自由高斯波包；
2. 长时间自由波包；
3. 阶跃势散射；
4. 方势垒 Klein tunneling；
5. 加入概率流守恒 loss；
6. 加入多区域界面约束；
7. 和 vanilla PINN、Fourier feature PINN、SIREN-PINN 对比。
