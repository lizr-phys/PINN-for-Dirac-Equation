"""
用 PINN（Physics-Informed Neural Network，物理信息神经网络）求解
一维量子线性谐振子的多个定态本征能级。

本文件是在原来“只求基态”的程序结构上做增量改进：
1. 原来的类名、主要函数名和训练主线尽量保留，便于对照学习。
2. 新增 multiple eigenstates 的思想：依次训练 n=0,1,2,... 的本征态。
3. 根据论文 Physics-Informed Neural Networks for Quantum Eigenvalue Problems
   的核心思想，加入 norm-loss 防止零解，加入 ortho-loss 让不同本征函数正交。
4. 对谐振子利用偶/奇宇称：n=0 偶，n=1 奇，n=2 偶，依次交替。

物理方程（无量纲形式）：

    [-1/2 d^2/dx^2 + 1/2 x^2] psi_n(x) = E_n psi_n(x),  x in [-L, L]

解析结果：

    E_n = n + 1/2

其中 n=0 是基态，n=1 是第一激发态，n=2 是第二激发态。
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib

# 使用 Agg 后端可以在没有图形界面的环境中直接保存图片。
# 这对服务器、终端、CI 等环境更友好。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# 1. 基础设置
# ---------------------------------------------------------------------------


def set_random_seed(seed: int = 42) -> None:
    """固定随机种子，让每次运行的结果更容易复现。

    神经网络训练本身带有随机性，例如：
    - 网络参数一开始是随机初始化的；
    - 如果使用随机采样点，采样点也会有随机性。

    固定随机种子不能保证不同电脑、不同 PyTorch 版本结果完全一样，
    但通常可以让同一环境下的结果非常接近。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """选择训练设备：有 NVIDIA GPU 时用 CUDA，否则用 CPU。"""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 2. 物理参数和训练超参数
# ---------------------------------------------------------------------------


# 区间半宽度。谐振子低能级波函数在 |x| 很大时迅速趋近于 0，
# L=5 时前几个低能态在边界处已经很小，因此可以近似认为 psi(-L)=psi(L)=0。
L = 5.0

# 新增：要求解的本征态数量。
# NUM_STATES=3 表示依次求 n=0,1,2 三个能级。
# 如果想尝试更多能级，可以改成 4 或 5，但高激发态会更难训练。
NUM_STATES = 3

# 配点数量。PINN 会在这些 x 点上检查薛定谔方程是否成立。
# 点数越多，方程约束越细，但每一轮训练也越慢。
# 对这个一维入门问题，400 个内部点通常已经足够看到合理结果。
NUM_COLLOCATION_POINTS = 400

# 每一个本征态训练的轮数。
# 原程序只训练一个基态；现在会对 n=0,1,2 分别训练 NUM_EPOCHS 轮。
NUM_EPOCHS = 4000

# Adam 是深度学习中常用的一阶优化器。PINN 入门时先用 Adam 最直观。
LEARNING_RATE = 1.0e-3

# 归一化损失权重。
# 本征值问题中 psi=0 会让 PDE residual 变成 0，但这是没有物理意义的零解。
# 因此归一化约束非常重要，权重要明显大于 1。
LAMBDA_NORM = 50.0

# 新增：正交损失权重。
# 求多个本征态时，仅仅让 PDE residual 小是不够的，因为网络可能反复学到同一个低能态。
# ortho-loss 会惩罚当前波函数与之前已找到波函数的重叠，从而推动网络寻找新的本征态。
LAMBDA_ORTH = 120.0

# Rayleigh quotient 辅助项权重。
# 定态薛定谔方程的本征态不止一个，基态是能量最低的那个。
# 在多本征态训练中，Rayleigh 项会在“满足正交条件后”的可行空间里偏向最低能量态。
LAMBDA_RAYLEIGH = 0.05

# 让可训练能量 E 与由当前波函数计算出的 Rayleigh 能量保持一致。
# 这不是必须项，但能让 E 的学习更稳定、更容易接近物理能量。
LAMBDA_ENERGY_MATCH = 0.1

# 新增：数据点约束权重。
# 数据点约束的作用是让网络在少量已知位置上贴近给定波函数值。
# 它不是替代 PDE loss，而是给 PINN 增加“观测数据锚点”。
LAMBDA_DATA = 20.0

# 新增：是否启用数据点约束。
# 如果想恢复纯 PINN 训练，可以改成 False。
USE_DATA_LOSS = True

# 新增：数据点来源。
# "random"：程序从解析解随机抽取 5 个点作为测试数据；
# "manual"：使用下面 MANUAL_DATA_POINTS 中手动输入的 5 组数据点。
DATA_MODE = "random"

# 新增：每个本征态使用的数据点数量。
# 导师要求五组数据点，因此默认设置为 5。
NUM_DATA_POINTS = 5

# 新增：随机测试数据的种子。修改它可以得到另一组随机测试点。
DATA_RANDOM_SEED = 2026

# 新增：随机测试数据噪声强度。
# 0.0 表示数据点来自精确解析值；如果想模拟有噪声数据，可以改为 0.01 等小数。
DATA_NOISE_STD = 0.0

# 新增：手动输入的数据点。
# 当 DATA_MODE = "manual" 时，程序会使用这里的数据。
# 每个元组是 (x, psi_data)。这里给出前三个本征态各 5 个示例点，
# 你可以把这些数替换成导师给出的数据或自己测得的数据。
MANUAL_DATA_POINTS: dict[int, list[tuple[float, float]]] = {
    0: [
        (-3.0, 0.008344251073246043),
        (-1.5, 0.2442517225407128),
        (0.0, 0.7511255444649425),
        (1.5, 0.2442517225407128),
        (3.0, 0.008344251073246043),
    ],
    1: [
        (-3.0, -0.03540135430468704),
        (-1.5, -0.5181377696375343),
        (0.0, 0.0),
        (1.5, 0.5181377696375343),
        (3.0, 0.03540135430468704),
    ],
    2: [
        (-3.0, 0.1001584802209448),
        (-1.5, 0.6471753240913234),
        (0.0, -0.5311259660135984),
        (1.5, 0.6471753240913234),
        (3.0, 0.1001584802209448),
    ],
}

# 每隔多少轮打印一次训练信息。
PRINT_EVERY = 500


# ---------------------------------------------------------------------------
# 3. 神经网络定义
# ---------------------------------------------------------------------------


def inverse_softplus(value: float) -> float:
    """softplus 的反函数，用来设置一个直观的正能量初值。

    PyTorch 中 softplus(z)=log(1+exp(z))，它总是大于 0。
    我们训练 E_raw，再令 E=softplus(E_raw)，这样 E 不会变成负数。
    """

    return math.log(math.exp(value) - 1.0)


class HarmonicOscillatorPINN(nn.Module):
    """表示谐振子某一个本征态的 PINN 模型。

    这仍然保留原程序的类名 HarmonicOscillatorPINN，只是在原来的基态模型上
    增加了 state_index。state_index 决定当前模型要学习第几个本征态：

        state_index = 0 -> 基态，偶函数
        state_index = 1 -> 第一激发态，奇函数
        state_index = 2 -> 第二激发态，偶函数

    为什么要加入奇偶宇称？
    谐振子势能 V(x)=x^2/2 是偶函数，因此本征函数具有确定宇称。
    论文中也提到，把 even/odd symmetry 直接嵌入网络结构可以改善收敛。

    原程序中基态使用：

        psi_theta(x) = (1 - x^2/L^2) * N_even(x)

    现在扩展为：

        偶态：N_sym(x) = [N(x)+N(-x)]/2
        奇态：N_sym(x) = x * [N(x)+N(-x)]/2

    然后仍然乘以边界因子：

        psi_theta(x) = (1 - x^2/L^2) * N_sym(x)

    这样既保留原来的边界条件硬约束，又能求奇偶交替的多个本征函数。
    """

    def __init__(
        self,
        domain_half_width: float,
        hidden_dim: int = 32,
        hidden_layers: int = 3,
        initial_energy: float = 0.7,
        state_index: int = 0,
    ) -> None:
        super().__init__()
        self.L = domain_half_width
        self.state_index = state_index

        layers: list[nn.Module] = []
        input_dim = 1
        output_dim = 1

        # 第一层：把一维坐标 x 映射到 hidden_dim 维特征。
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # 中间隐藏层：tanh 是 PINN 中很常见的激活函数。
        # 原因是它光滑可导，而本问题需要对网络输出求二阶导数。
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # 最后一层：输出一个标量 N_theta(x)。
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

        self._initialize_weights()

        # E_raw 是真正交给优化器训练的参数。
        # energy() 方法会返回 softplus(E_raw)，保证预测能量 E > 0。
        raw_energy_value = inverse_softplus(initial_energy)
        self.energy_raw = nn.Parameter(torch.tensor([raw_energy_value], dtype=torch.float32))

    def _initialize_weights(self) -> None:
        """初始化网络权重。

        Xavier 初始化适合 tanh 这类激活函数，通常比完全默认初始化更稳定。

        注意这里没有把 bias 初始化为 0。
        因为我们后面会构造 [N(x)+N(-x)]/2 或 [N(x)-N(-x)]/2。
        如果 tanh 网络所有 bias 都是 0，某些对称组合在初始化时可能出现严重抵消，
        使 psi 接近零函数，不利于训练。
        """

        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.normal_(module.bias, mean=0.0, std=0.05)

    def energy(self) -> torch.Tensor:
        """返回正的可训练能量 E。"""

        return F.softplus(self.energy_raw)

    def raw_even_network(self, x: torch.Tensor) -> torch.Tensor:
        """构造具有正确宇称的网络输出 N_sym(x)。

        函数名保留为 raw_even_network，是为了尽量不改变原程序结构。
        但现在它会根据 self.state_index 自动选择偶函数或奇函数：

        - 偶态 n=0,2,4,... 使用 [N(x)+N(-x)]/2；
        - 奇态 n=1,3,5,... 使用 x*[N(x)+N(-x)]/2。

        这样扩展后，原来的基态代码逻辑仍然成立；当 state_index=0 时，
        它和原程序完全一样，仍然是偶函数网络输出。
        """

        n_x = self.network(x)
        n_minus_x = self.network(-x)

        even_part = 0.5 * (n_x + n_minus_x)
        if self.state_index % 2 == 0:
            return even_part

        # 对奇态，使用 x * even_part。偶函数乘以 x 后一定是奇函数。
        # 对谐振子第一激发态尤其自然，因为解析解正比于 x*exp(-x^2/2)。
        return x * even_part

    def psi(self, x: torch.Tensor) -> torch.Tensor:
        """计算满足边界条件的波函数 psi_theta(x)。"""

        boundary_factor = 1.0 - (x**2) / (self.L**2)
        return boundary_factor * self.raw_even_network(x)


# ---------------------------------------------------------------------------
# 4. 自动求导：计算一阶导数和二阶导数
# ---------------------------------------------------------------------------


def compute_derivatives(
    psi_value: torch.Tensor,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用 PyTorch autograd 计算 dpsi/dx 和 d2psi/dx2。"""

    dpsi_dx = torch.autograd.grad(
        outputs=psi_value,
        inputs=x,
        grad_outputs=torch.ones_like(psi_value),
        create_graph=True,
    )[0]

    d2psi_dx2 = torch.autograd.grad(
        outputs=dpsi_dx,
        inputs=x,
        grad_outputs=torch.ones_like(dpsi_dx),
        create_graph=True,
    )[0]

    return dpsi_dx, d2psi_dx2


# ---------------------------------------------------------------------------
# 5. 损失函数
# ---------------------------------------------------------------------------


def integral_mean(values: torch.Tensor, domain_half_width: float) -> torch.Tensor:
    """用均值近似积分 integral f(x) dx。

    训练点在 [-L,L] 上近似均匀分布，因此：

        integral_{-L}^{L} f(x) dx ≈ 2L * mean(f(x_i))

    原程序在归一化损失中已经使用了这个思想。
    现在多本征态需要正交损失，所以把这个积分近似单独写成小函数。
    """

    return 2.0 * domain_half_width * torch.mean(values)


def compute_losses(
    model: HarmonicOscillatorPINN,
    x_base: torch.Tensor,
    previous_psi_values: list[torch.Tensor] | None = None,
    data_points: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """计算 PINN 的各项损失。

    这个函数保留原程序 compute_losses 的名称和主要结构。
    新增参数 previous_psi_values，用来保存之前已经训练好的本征函数值。
    新增参数 data_points，用来传入五组数据点 (x_data, psi_data)。

    当训练 n=0 时，previous_psi_values 为空，程序退化为原来的基态训练。
    当训练 n=1 时，previous_psi_values 中有 psi_0，需要让 psi_1 与 psi_0 正交。
    当训练 n=2 时，需要让 psi_2 与 psi_0、psi_1 都正交。

    新增的 ortho-loss 来源于论文中的核心思想：
    不同本征函数应当两两正交，因此惩罚内积 overlap = integral psi_new psi_old dx。

    新增的 data-loss 表示：在少量已知数据点上，网络预测值也要接近给定值。
    这会把 PINN 从纯物理约束训练扩展成“物理方程 + 数据点”的混合约束训练。
    """

    if previous_psi_values is None:
        previous_psi_values = []

    # 每轮训练都重新创建带 requires_grad 的 x。
    # clone().detach() 可以避免上一轮训练留下的计算图被复用。
    x = x_base.clone().detach().requires_grad_(True)

    psi_value = model.psi(x)
    dpsi_dx, d2psi_dx2 = compute_derivatives(psi_value, x)
    energy = model.energy()

    # 薛定谔方程残差：
    # [-1/2 d^2/dx^2 + 1/2 x^2] psi - E psi
    residual = -0.5 * d2psi_dx2 + 0.5 * (x**2) * psi_value - energy * psi_value
    pde_loss = torch.mean(residual**2)

    # 归一化损失：避免 psi=0 的平凡解。
    psi_square_integral = integral_mean(psi_value**2, model.L)
    norm_loss = (psi_square_integral - 1.0) ** 2

    # Rayleigh quotient：当前波函数对应的能量期望。
    kinetic_density = 0.5 * dpsi_dx**2
    potential_density = 0.5 * (x**2) * psi_value**2
    energy_numerator = integral_mean(kinetic_density + potential_density, model.L)
    rayleigh_energy = energy_numerator / (psi_square_integral + 1.0e-8)

    energy_match_loss = (energy - rayleigh_energy) ** 2

    # 新增：正交损失。
    # 对每一个已经训练好的旧本征函数 psi_old，要求：
    #     integral psi_new(x) psi_old(x) dx = 0
    # 这里用 overlap^2 作为损失，因为正负重叠都应该被惩罚。
    orth_loss = torch.zeros((), device=x.device)
    for previous_psi in previous_psi_values:
        overlap = integral_mean(psi_value * previous_psi, model.L)
        orth_loss = orth_loss + overlap**2

    # 新增：数据点损失。
    # x_data 和 psi_data 是五组已知数据点。
    # 数据损失要求模型在这些点上的预测值接近给定数据：
    #     L_data = mean((psi_theta(x_data) - psi_data)^2)
    # 注意它只约束少数点；PDE loss 仍然负责整个区间上的物理一致性。
    data_loss = torch.zeros((), device=x.device)
    if USE_DATA_LOSS and data_points is not None:
        x_data, psi_data = data_points
        psi_data_pred = model.psi(x_data)
        data_loss = torch.mean((psi_data_pred - psi_data) ** 2)

    total_loss = (
        pde_loss
        + LAMBDA_NORM * norm_loss
        + LAMBDA_ORTH * orth_loss
        + LAMBDA_DATA * data_loss
        + LAMBDA_RAYLEIGH * rayleigh_energy
        + LAMBDA_ENERGY_MATCH * energy_match_loss
    )

    return {
        "total": total_loss,
        "pde": pde_loss,
        "norm": norm_loss,
        "orth": orth_loss,
        "data": data_loss,
        "rayleigh": rayleigh_energy,
        "energy_match": energy_match_loss,
        "energy": energy,
        "norm_integral": psi_square_integral,
    }


# ---------------------------------------------------------------------------
# 6. 解析解、绘图和评估
# ---------------------------------------------------------------------------


def exact_ground_state_wavefunction(x: np.ndarray) -> np.ndarray:
    """一维无量纲谐振子基态解析波函数。保留原函数，方便对照。"""

    return (math.pi ** (-0.25)) * np.exp(-0.5 * x**2)


def physicists_hermite(n: int, x: np.ndarray) -> np.ndarray:
    """计算物理学家约定的 Hermite 多项式 H_n(x)。

    多个谐振子本征函数的解析式为：

        psi_n(x) = 1/(pi^(1/4) sqrt(2^n n!)) H_n(x) exp(-x^2/2)

    为了不增加 scipy 依赖，这里用递推公式计算 H_n：

        H_0 = 1
        H_1 = 2x
        H_{n+1} = 2x H_n - 2n H_{n-1}
    """

    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x

    h_nm1 = np.ones_like(x)
    h_n = 2.0 * x
    for k in range(1, n):
        h_np1 = 2.0 * x * h_n - 2.0 * k * h_nm1
        h_nm1, h_n = h_n, h_np1
    return h_n


def exact_energy(state_index: int) -> float:
    """一维无量纲谐振子的解析能量 E_n = n + 1/2。"""

    return state_index + 0.5


def exact_wavefunction(state_index: int, x: np.ndarray) -> np.ndarray:
    """第 state_index 个谐振子本征函数的解析解。"""

    if state_index == 0:
        return exact_ground_state_wavefunction(x)

    coefficient = 1.0 / (
        math.pi**0.25 * math.sqrt((2.0**state_index) * math.factorial(state_index))
    )
    return coefficient * physicists_hermite(state_index, x) * np.exp(-0.5 * x**2)


def build_data_points(
    state_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """为当前本征态准备五组数据点。

    这个函数提供两种模式：

    1. DATA_MODE = "random"
       程序从解析解中随机抽取 NUM_DATA_POINTS 个 x 点，并计算对应 psi_n(x)。
       这适合测试“少量数据点 + 物理约束”的预测效果。

    2. DATA_MODE = "manual"
       程序读取 MANUAL_DATA_POINTS[state_index] 中手动输入的五组 (x, psi) 数据。
       以后如果导师给了数据，只需要替换 MANUAL_DATA_POINTS 里的数值。

    返回值中既有 PyTorch Tensor，也有 NumPy 数组：
    - Tensor 用于训练时计算 data loss；
    - NumPy 数组用于打印和保存数据点表格。
    """

    if not USE_DATA_LOSS:
        empty_x = np.empty((0,), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32)
        empty_tensor = torch.empty((0, 1), device=device)
        return empty_tensor, empty_tensor, empty_x, empty_y

    if DATA_MODE == "manual":
        if state_index not in MANUAL_DATA_POINTS:
            raise ValueError(f"MANUAL_DATA_POINTS does not contain state n={state_index}")
        raw_points = MANUAL_DATA_POINTS[state_index]
        if len(raw_points) != NUM_DATA_POINTS:
            raise ValueError(
                f"state n={state_index} needs {NUM_DATA_POINTS} manual data points, "
                f"but got {len(raw_points)}"
            )
        x_np = np.array([point[0] for point in raw_points], dtype=np.float32)
        psi_np = np.array([point[1] for point in raw_points], dtype=np.float32)
    elif DATA_MODE == "random":
        # 每个 state 使用不同但可复现的随机数流。
        rng = np.random.default_rng(DATA_RANDOM_SEED + state_index)

        # 避免随机点太靠近边界，因为边界附近波函数极小，对训练帮助有限。
        x_np = np.sort(rng.uniform(-0.85 * L, 0.85 * L, size=NUM_DATA_POINTS)).astype(np.float32)
        psi_np = exact_wavefunction(state_index, x_np).astype(np.float32)

        if DATA_NOISE_STD > 0.0:
            psi_np = psi_np + rng.normal(0.0, DATA_NOISE_STD, size=psi_np.shape).astype(np.float32)
    else:
        raise ValueError("DATA_MODE must be 'random' or 'manual'")

    x_tensor = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float32, device=device)
    psi_tensor = torch.tensor(psi_np.reshape(-1, 1), dtype=torch.float32, device=device)
    return x_tensor, psi_tensor, x_np, psi_np


def save_data_points_table(
    all_data_points: list[tuple[int, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """保存训练中使用的五组数据点，方便检查和复现实验。"""

    rows: list[list[float]] = []
    for state_index, x_np, psi_np in all_data_points:
        for x_value, psi_value in zip(x_np, psi_np):
            rows.append([float(state_index), float(x_value), float(psi_value)])

    table = np.array(rows, dtype=float)
    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header="state_index,x_data,psi_data",
        comments="",
    )

def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """梯形积分，兼容不同 NumPy 版本。"""

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


@torch.no_grad()
def evaluate_wavefunction(
    model: HarmonicOscillatorPINN,
    device: torch.device,
    num_points: int = 1200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在密集网格上计算 PINN 波函数和对应解析波函数，方便画图比较。

    原程序默认比较基态。现在根据 model.state_index 自动选择解析的第 n 态。
    """

    x_np = np.linspace(-model.L, model.L, num_points, dtype=np.float32)
    x_tensor = torch.tensor(x_np.reshape(-1, 1), device=device)

    psi_pred = model.psi(x_tensor).cpu().numpy().reshape(-1)
    psi_exact = exact_wavefunction(model.state_index, x_np)

    # 训练中的归一化已经使 integral psi^2 dx 接近 1。
    # 这里再用更密集网格做一次轻微归一化，只是为了画图更公平。
    pred_norm = math.sqrt(max(trapezoid_integral(psi_pred**2, x_np), 1.0e-12))
    psi_pred = psi_pred / pred_norm

    # 本征函数的整体符号没有物理意义：psi 和 -psi 对应同一个概率密度。
    overlap = trapezoid_integral(psi_pred * psi_exact, x_np)
    if overlap < 0:
        psi_pred = -psi_pred

    return x_np, psi_pred, psi_exact


@torch.no_grad()
def normalized_training_values(
    model: HarmonicOscillatorPINN,
    x_base: torch.Tensor,
) -> torch.Tensor:
    """保存当前训练好的波函数值，供后续本征态的正交损失使用。

    注意这里返回 detach 后的 Tensor。
    这表示旧本征态已经固定，不会在训练新本征态时继续被更新。
    """

    psi_value = model.psi(x_base)
    norm = torch.sqrt(torch.clamp(integral_mean(psi_value**2, model.L), min=1.0e-12))
    return (psi_value / norm).detach()


def save_wavefunction_plot(
    x: np.ndarray,
    psi_pred: np.ndarray,
    psi_exact: np.ndarray,
    output_path: Path,
    state_index: int = 0,
) -> None:
    """保存 PINN 波函数与解析波函数对比图。"""

    plt.figure(figsize=(8, 5))
    plt.plot(x, psi_exact, "k--", linewidth=2.0, label=f"Analytical state n={state_index}")
    plt.plot(x, psi_pred, "r", linewidth=2.0, label=f"PINN prediction n={state_index}")
    plt.xlabel("x")
    plt.ylabel(r"$\psi_n(x)$")
    plt.title(f"1D Quantum Harmonic Oscillator Eigenstate n={state_index}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_loss_plot(history: dict[str, list[float]], output_path: Path) -> None:
    """保存 loss 曲线。

    原函数名保留。现在除了 total/PDE/norm，也会画新增的 orth loss。
    """

    epochs = np.arange(1, len(history["total"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.semilogy(epochs, history["total"], label="total loss")
    plt.semilogy(epochs, history["pde"], label="PDE loss")
    plt.semilogy(epochs, history["norm"], label="normalization loss")
    if "orth" in history:
        plt.semilogy(epochs, np.array(history["orth"]) + 1.0e-16, label="orthogonality loss")
    if "data" in history:
        plt.semilogy(epochs, np.array(history["data"]) + 1.0e-16, label="data loss")
    plt.xlabel("epoch")
    plt.ylabel("loss (log scale)")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_multi_wavefunction_plot(
    evaluations: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """把多个本征函数画到一张多子图中。"""

    num_states = len(evaluations)
    fig, axes = plt.subplots(num_states, 1, figsize=(8, 3.0 * num_states), sharex=True)
    if num_states == 1:
        axes = [axes]

    for state_index, (axis, (x, psi_pred, psi_exact)) in enumerate(zip(axes, evaluations)):
        axis.plot(x, psi_exact, "k--", linewidth=2.0, label=f"Exact n={state_index}")
        axis.plot(x, psi_pred, "r", linewidth=1.8, label=f"PINN n={state_index}")
        axis.set_ylabel(r"$\psi_n(x)$")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("x")
    fig.suptitle("PINN Eigenfunctions vs Analytical Eigenfunctions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_energy_plot(results: list[dict[str, object]], output_path: Path) -> None:
    """保存预测能级与解析能级对比图。"""

    state_indices = np.array([int(result["state_index"]) for result in results])
    predicted = np.array([float(result["predicted_energy"]) for result in results])
    exact = np.array([float(result["exact_energy"]) for result in results])

    plt.figure(figsize=(7, 4.5))
    plt.plot(state_indices, exact, "ko--", linewidth=2.0, label="Exact energies")
    plt.plot(state_indices, predicted, "ro", markersize=7, label="PINN predicted energies")
    plt.xlabel("state index n")
    plt.ylabel("energy E")
    plt.title("Harmonic Oscillator Energy Levels")
    plt.xticks(state_indices)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_energy_table(results: list[dict[str, object]], output_path: Path) -> None:
    """保存预测能量到 CSV 表格。"""

    table = np.array(
        [
            [
                int(result["state_index"]),
                float(result["predicted_energy"]),
                float(result["exact_energy"]),
                float(result["absolute_error"]),
            ]
            for result in results
        ],
        dtype=float,
    )
    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header="state_index,predicted_energy,exact_energy,absolute_error",
        comments="",
    )


def save_wavefunction_table(
    evaluations: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """保存多个预测波函数的数值表。

    神经网络预测出的函数通常不是一个简洁解析公式。
    因此输出它在密集 x 网格上的函数值，是查看预测函数的实用方法。
    """

    x = evaluations[0][0]
    columns = [x]
    header_parts = ["x"]

    for state_index, (_, psi_pred, psi_exact) in enumerate(evaluations):
        columns.extend([psi_pred, psi_exact, np.abs(psi_pred - psi_exact)])
        header_parts.extend(
            [
                f"psi_pinn_n{state_index}",
                f"psi_exact_n{state_index}",
                f"abs_error_n{state_index}",
            ]
        )

    table = np.column_stack(columns)
    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header=",".join(header_parts),
        comments="",
    )


# ---------------------------------------------------------------------------
# 7. 训练主程序
# ---------------------------------------------------------------------------


def main() -> None:
    set_random_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 配点使用区间内部点，不包含 x=-L 和 x=L。
    # 边界点已经由输出变换 psi=(1-x^2/L^2)N(x) 自动满足 psi=0。
    x_collocation = torch.linspace(
        -L,
        L,
        NUM_COLLOCATION_POINTS + 2,
        device=device,
    )[1:-1].reshape(-1, 1)

    results: list[dict[str, object]] = []
    previous_psi_values: list[torch.Tensor] = []
    all_data_points: list[tuple[int, np.ndarray, np.ndarray]] = []

    # 多本征态训练的核心：按 n=0,1,2,... 顺序训练。
    # 每训练好一个态，就把它的波函数值保存下来，作为后续态的正交约束。
    for state_index in range(NUM_STATES):
        print()
        print(f"Training eigenstate n={state_index}")

        x_data, psi_data, x_data_np, psi_data_np = build_data_points(state_index, device)
        data_points = (x_data, psi_data) if USE_DATA_LOSS else None
        all_data_points.append((state_index, x_data_np, psi_data_np))

        if USE_DATA_LOSS:
            print("Data points used for data loss:")
            for x_value, psi_value in zip(x_data_np, psi_data_np):
                print(f"  x = {x_value: .6f}, psi_data = {psi_value: .8f}")

        # 谐振子的解析能量大约为 n+1/2。
        # 这里用 0.7+n 只是为了给可训练 E 一个合理初值，不是固定答案。
        model = HarmonicOscillatorPINN(
            domain_half_width=L,
            initial_energy=0.7 + state_index,
            state_index=state_index,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        history: dict[str, list[float]] = {
            "total": [],
            "pde": [],
            "norm": [],
            "orth": [],
            "data": [],
            "rayleigh": [],
            "energy": [],
        }

        for epoch in range(1, NUM_EPOCHS + 1):
            optimizer.zero_grad(set_to_none=True)

            losses = compute_losses(model, x_collocation, previous_psi_values, data_points)
            losses["total"].backward()

            # 二阶导数会让 PINN 的梯度偶尔比较大。
            # 梯度裁剪可以减少训练早期的剧烈震荡。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

            # 记录每一轮的标量值，后面用于画 loss 曲线。
            history["total"].append(float(losses["total"].detach().cpu()))
            history["pde"].append(float(losses["pde"].detach().cpu()))
            history["norm"].append(float(losses["norm"].detach().cpu()))
            history["orth"].append(float(losses["orth"].detach().cpu()))
            history["data"].append(float(losses["data"].detach().cpu()))
            history["rayleigh"].append(float(losses["rayleigh"].detach().cpu()))
            history["energy"].append(float(losses["energy"].detach().cpu()))

            if epoch == 1 or epoch % PRINT_EVERY == 0:
                print(
                    f"State {state_index} | "
                    f"Epoch {epoch:5d} | "
                    f"total loss = {history['total'][-1]:.6e} | "
                    f"PDE loss = {history['pde'][-1]:.6e} | "
                    f"norm loss = {history['norm'][-1]:.6e} | "
                    f"orth loss = {history['orth'][-1]:.6e} | "
                    f"data loss = {history['data'][-1]:.6e} | "
                    f"Rayleigh E = {history['rayleigh'][-1]:.6f} | "
                    f"predicted E = {history['energy'][-1]:.6f}"
                )

        predicted_energy = float(model.energy().detach().cpu())
        e_exact = exact_energy(state_index)
        absolute_error = abs(predicted_energy - e_exact)

        print()
        print(f"Predicted energy for state n={state_index}: E = {predicted_energy:.8f}")
        print(f"Exact energy for state n={state_index}: E = {e_exact:.8f}")
        print(f"Absolute error = {absolute_error:.8e}")

        results.append(
            {
                "state_index": state_index,
                "model": model,
                "history": history,
                "predicted_energy": predicted_energy,
                "exact_energy": e_exact,
                "absolute_error": absolute_error,
            }
        )

        # 保存当前态的归一化波函数值，供后续态计算 ortho-loss。
        previous_psi_values.append(normalized_training_values(model, x_collocation))

    print()
    print("Predicted energy levels")
    print("state | predicted E | exact E | absolute error")
    for result in results:
        print(
            f"n={int(result['state_index']):<2d} | "
            f"{float(result['predicted_energy']): .8f} | "
            f"{float(result['exact_energy']): .8f} | "
            f"{float(result['absolute_error']):.8e}"
        )

    evaluations: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for result in results:
        model = result["model"]
        assert isinstance(model, HarmonicOscillatorPINN)
        x_np, psi_pred, psi_exact = evaluate_wavefunction(model, device)
        evaluations.append((x_np, psi_pred, psi_exact))

        state_index = int(result["state_index"])
        save_wavefunction_plot(
            x_np,
            psi_pred,
            psi_exact,
            results_dir / f"wavefunction_comparison_n{state_index}.png",
            state_index=state_index,
        )
        save_loss_plot(
            result["history"],
            results_dir / f"loss_curve_n{state_index}.png",
        )

    # 为了兼容原程序输出文件名，仍然把 n=0 的图另存为原来的文件名。
    save_wavefunction_plot(
        evaluations[0][0],
        evaluations[0][1],
        evaluations[0][2],
        results_dir / "wavefunction_comparison.png",
        state_index=0,
    )
    save_loss_plot(results[0]["history"], results_dir / "loss_curve.png")

    multi_wavefunction_path = results_dir / "multi_wavefunction_comparison.png"
    energy_plot_path = results_dir / "energy_levels.png"
    energy_table_path = results_dir / "predicted_energies.csv"
    wavefunction_table_path = results_dir / "predicted_wavefunctions.csv"
    data_points_table_path = results_dir / "training_data_points.csv"

    save_multi_wavefunction_plot(evaluations, multi_wavefunction_path)
    save_energy_plot(results, energy_plot_path)
    save_energy_table(results, energy_table_path)
    save_wavefunction_table(evaluations, wavefunction_table_path)
    save_data_points_table(all_data_points, data_points_table_path)

    print()
    print(f"Saved ground-state wavefunction comparison to: {results_dir / 'wavefunction_comparison.png'}")
    print(f"Saved ground-state loss curve to: {results_dir / 'loss_curve.png'}")
    print(f"Saved multi-state wavefunction comparison to: {multi_wavefunction_path}")
    print(f"Saved energy level plot to: {energy_plot_path}")
    print(f"Saved predicted energies to: {energy_table_path}")
    print(f"Saved predicted wavefunctions to: {wavefunction_table_path}")
    print(f"Saved training data points to: {data_points_table_path}")


if __name__ == "__main__":
    main()






