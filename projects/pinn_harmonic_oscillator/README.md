# 基于 PINN 的一维量子谐振子多个本征能级求解

这个项目用 PyTorch 实现一个适合初学者阅读的 PINN 程序，用于求解一维量子线性谐振子的多个定态本征能级。

程序最初只求基态。现在在原有结构上做了增量改进：保留原来的 `HarmonicOscillatorPINN`、`compute_losses`、`evaluate_wavefunction`、`save_wavefunction_plot`、`save_loss_plot` 和主训练结构，同时加入多个本征态训练所需的正交损失。

改进思路参考论文 *Physics-Informed Neural Networks for Quantum Eigenvalue Problems* 中的几个核心原则：

- 用神经网络表示未知本征函数；
- 用可训练参数表示未知本征值；
- 用参数化输出自动满足边界条件；
- 用 `norm-loss` 防止零解；
- 用 `ortho-loss` 让不同本征函数彼此正交；
- 对有对称性的体系，把偶/奇宇称嵌入网络结构中。

## 1. 物理问题

无量纲一维谐振子定态薛定谔方程为：

```text
[-1/2 d^2/dx^2 + 1/2 x^2] psi_n(x) = E_n psi_n(x)
```

求解区间取为：

```text
x in [-L, L],  L = 5
```

解析能级为：

```text
E_n = n + 1/2
```

因此前三个能级分别为：

```text
E0 = 0.5
E1 = 1.5
E2 = 2.5
```

解析本征函数为：

```text
psi_n(x) = 1 / (pi^(1/4) sqrt(2^n n!)) * H_n(x) * exp(-x^2/2)
```

其中 `H_n(x)` 是物理学家约定的 Hermite 多项式。

## 2. 程序如何表示波函数

神经网络输入位置坐标：

```text
x
```

输出波函数值：

```text
psi_theta(x)
```

普通网络输出记为：

```text
N_theta(x)
```

为了保留原程序结构，模型类仍然叫：

```python
HarmonicOscillatorPINN
```

新增参数：

```python
state_index
```

用来表示当前训练第几个本征态。

谐振子本征函数具有确定宇称：

```text
n = 0, 2, 4, ... 为偶函数
n = 1, 3, 5, ... 为奇函数
```

程序中：

```text
偶态: N_sym(x) = [N(x) + N(-x)] / 2
奇态: N_sym(x) = x * [N(x) + N(-x)] / 2
```

奇态使用 `x * 偶函数` 的形式，因为偶函数乘以 `x` 后一定是奇函数，而且第一激发态解析解本身就正比于 `x exp(-x^2/2)`。

最终波函数为：

```text
psi_theta(x) = (1 - x^2/L^2) * N_sym(x)
```

当 `x = ±L` 时，边界因子为 0，因此自动满足：

```text
psi(-L) = psi(L) = 0
```

## 3. 多本征态的核心：正交损失

只训练 PDE residual 时，网络可能反复找到同一个低能态。量子力学中，厄米算符不同本征值对应的本征函数彼此正交。

因此，训练第 `n` 个态时，程序会让它与之前已经训练好的态正交：

```text
integral psi_n(x) psi_m(x) dx = 0,  m < n
```

代码中对应的正交损失为：

```text
L_orth = sum_m [ integral psi_new(x) psi_old_m(x) dx ]^2
```

这就是程序可以依次求出多个能级的关键。

## 4. 损失函数

总损失为：

```text
L_total = L_pde
        + lambda_norm * L_norm
        + lambda_orth * L_orth
        + lambda_rayleigh * R[psi]
        + lambda_energy_match * L_energy_match
```

各项含义：

- `L_pde`：薛定谔方程 residual 的平方平均；
- `L_norm`：归一化损失，防止零解；
- `L_orth`：正交损失，防止重复找到同一个本征态；
- `R[psi]`：Rayleigh 能量项，帮助网络在当前约束下寻找最低可用能量态；
- `L_energy_match`：让可训练能量 `E` 与当前波函数的 Rayleigh 能量一致。

## 5. 数据点约束

现在程序加入了少量数据点约束，用来实现“物理方程 + 数据点”的混合训练。

数据损失定义为：

```text
L_data = mean((psi_theta(x_data) - psi_data)^2)
```

其中：

```text
x_data   是给定数据点的位置
psi_data 是这些位置上的已知波函数值
```

总损失中新增：

```text
lambda_data * L_data
```

这样训练时，网络不仅要满足薛定谔方程、归一化和正交性，还要在五组给定数据点上贴近已知数据。

程序提供两种数据模式：

```python
DATA_MODE = "random"
```

表示程序从解析解中随机抽取 5 个点作为测试数据，方便验证程序。

```python
DATA_MODE = "manual"
```

表示使用 `MANUAL_DATA_POINTS` 中手动输入的数据。每个数据点格式是：

```python
(x, psi_data)
```

例如：

```python
MANUAL_DATA_POINTS = {
    0: [
        (-3.0, 0.008344251073246043),
        (-1.5, 0.2442517225407128),
        (0.0, 0.7511255444649425),
        (1.5, 0.2442517225407128),
        (3.0, 0.008344251073246043),
    ],
}
```

如果导师给出实验数据或数值数据，只需要把这里的 `psi_data` 替换掉即可。

训练时使用的数据点会保存到：

```text
results/training_data_points.csv
```

这样可以明确看到模型到底用了哪五组数据点。

## 6. 能量参数

程序仍然使用可训练参数表示能量。

实际训练的是：

```text
E_raw
```

真正进入方程的是：

```text
E = softplus(E_raw)
```

这样可以保证：

```text
E > 0
```

## 7. 如何运行

安装依赖：

```bash
pip install -r requirements.txt
```

运行程序：

```bash
python main.py
```

默认会求前三个本征态：

```text
n = 0, 1, 2
```

可以在 `main.py` 中修改：

```python
NUM_STATES = 3
```

来控制求几个能级。

## 8. 输出结果

程序会输出每个态的预测能量，例如：

```text
n=0  predicted E ≈ 0.5
n=1  predicted E ≈ 1.5
n=2  predicted E ≈ 2.5
```

并保存：

```text
results/wavefunction_comparison.png
results/loss_curve.png
results/wavefunction_comparison_n0.png
results/wavefunction_comparison_n1.png
results/wavefunction_comparison_n2.png
results/loss_curve_n0.png
results/loss_curve_n1.png
results/loss_curve_n2.png
results/multi_wavefunction_comparison.png
results/energy_levels.png
results/predicted_energies.csv
results/predicted_wavefunctions.csv
results/training_data_points.csv
```

其中：

- `multi_wavefunction_comparison.png`：多个本征函数的 PINN 与解析解对比；
- `energy_levels.png`：预测能级与解析能级对比；
- `predicted_energies.csv`：预测能量表；
- `predicted_wavefunctions.csv`：每个 `x` 点上的预测波函数数值；
- `training_data_points.csv`：训练中实际使用的五组数据点。

## 9. 程序流程总结

整体流程如下：

```text
设置随机种子和设备
    ↓
生成区间内部配点
    ↓
训练 n=0
    ↓
保存 psi_0
    ↓
训练 n=1，并加入与 psi_0 的正交损失
    ↓
保存 psi_1
    ↓
训练 n=2，并加入与 psi_0、psi_1 的正交损失
    ↓
输出多个预测能量
    ↓
保存波函数图、能级图、loss 图和 CSV 数值表
```

## 10. 初学者可以如何继续修改

可以尝试：

- 修改 `NUM_STATES`，求更多能级；
- 修改 `NUM_EPOCHS`，增加训练轮数；
- 修改 `LAMBDA_ORTH`，观察正交损失权重对高激发态的影响；
- 修改 `DATA_MODE`，在随机测试数据和手动输入数据之间切换；
- 修改 `MANUAL_DATA_POINTS`，输入你自己的五组数据点；
- 修改 `LAMBDA_DATA`，观察数据点约束强弱对预测结果的影响；
- 修改 `L`，观察区间长度对边界近似的影响；
- 修改网络层数和宽度，观察表达能力对高激发态的影响。

如果求更高能级，建议适当增加训练轮数、网络宽度和采样点数量，因为高激发态有更多节点，更难拟合。

