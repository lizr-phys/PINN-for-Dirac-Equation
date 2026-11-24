"""
Minimal PINN example for the 1+1 dimensional free time-dependent Dirac equation.

The equation is written in natural units hbar = c = 1:

    i psi_t = (-i alpha psi_x + beta m psi),
    alpha = sigma_x, beta = sigma_z.

The network predicts the real and imaginary parts of a two-component spinor:

    psi = (psi_1, psi_2)^T,
    output = [Re psi_1, Im psi_1, Re psi_2, Im psi_2].

This first example uses a positive-energy plane wave as the exact solution. It
is intentionally simple: the purpose is to verify the Dirac residual, complex
spinor representation, initial condition, periodic boundary condition, and
basic error metrics before moving to wave packets and scattering.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    x_min: float = -math.pi
    x_max: float = math.pi
    t_min: float = 0.0
    t_max: float = 1.0
    mass: float = 1.0
    wave_number: float = 1.0
    hidden_dim: int = 64
    hidden_layers: int = 4
    num_collocation: int = 2048
    num_initial: int = 512
    num_boundary: int = 512
    epochs: int = 3000
    learning_rate: float = 1.0e-3
    lambda_pde: float = 1.0
    lambda_ic: float = 20.0
    lambda_bc: float = 5.0
    seed: int = 42
    log_every: int = 200

    @property
    def energy(self) -> float:
        return math.sqrt(self.wave_number**2 + self.mass**2)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiracPINN(nn.Module):
    def __init__(self, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 2
        for _ in range(hidden_layers):
            linear = nn.Linear(in_dim, hidden_dim)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.Tanh()])
            in_dim = hidden_dim
        final = nn.Linear(hidden_dim, 4)
        nn.init.xavier_normal_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1))


def exact_solution(x: torch.Tensor, t: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Positive-energy plane wave solution.

    For H(k) = k sigma_x + m sigma_z, the positive-energy eigenspinor can be
    written as u = N [1, k / (E + m)]^T.
    """

    k = cfg.wave_number
    energy = cfg.energy
    ratio = k / (energy + cfg.mass)
    norm = 1.0 / math.sqrt(1.0 + ratio**2)
    theta = k * x - energy * t

    psi1_re = norm * torch.cos(theta)
    psi1_im = norm * torch.sin(theta)
    psi2_re = norm * ratio * torch.cos(theta)
    psi2_im = norm * ratio * torch.sin(theta)
    return torch.cat([psi1_re, psi1_im, psi2_re, psi2_im], dim=1)


def sample_uniform(
    n: int,
    lower: float,
    upper: float,
    device: torch.device,
    requires_grad: bool = False,
) -> torch.Tensor:
    values = lower + (upper - lower) * torch.rand(n, 1, device=device)
    values.requires_grad_(requires_grad)
    return values


def derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def dirac_residual(
    model: DiracPINN,
    x: torch.Tensor,
    t: torch.Tensor,
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = model(x, t)
    a1, b1, a2, b2 = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]

    a1_t, b1_t, a2_t, b2_t = (
        derivative(a1, t),
        derivative(b1, t),
        derivative(a2, t),
        derivative(b2, t),
    )
    a1_x, b1_x, a2_x, b2_x = (
        derivative(a1, x),
        derivative(b1, x),
        derivative(a2, x),
        derivative(b2, x),
    )

    m = cfg.mass
    real_1 = -b1_t - b2_x - m * a1
    real_2 = -b2_t - b1_x + m * a2
    imag_1 = a1_t + a2_x - m * b1
    imag_2 = a2_t + a1_x + m * b2
    return real_1, imag_1, real_2, imag_2


def compute_losses(
    model: DiracPINN,
    cfg: Config,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    x_f = sample_uniform(cfg.num_collocation, cfg.x_min, cfg.x_max, device, True)
    t_f = sample_uniform(cfg.num_collocation, cfg.t_min, cfg.t_max, device, True)
    residuals = dirac_residual(model, x_f, t_f, cfg)
    pde_loss = sum(torch.mean(r**2) for r in residuals)

    x_ic = sample_uniform(cfg.num_initial, cfg.x_min, cfg.x_max, device)
    t_ic = torch.zeros_like(x_ic)
    pred_ic = model(x_ic, t_ic)
    target_ic = exact_solution(x_ic, t_ic, cfg)
    ic_loss = F.mse_loss(pred_ic, target_ic)

    t_bc = sample_uniform(cfg.num_boundary, cfg.t_min, cfg.t_max, device)
    x_left = torch.full_like(t_bc, cfg.x_min)
    x_right = torch.full_like(t_bc, cfg.x_max)
    bc_loss = F.mse_loss(model(x_left, t_bc), model(x_right, t_bc))

    total = (
        cfg.lambda_pde * pde_loss
        + cfg.lambda_ic * ic_loss
        + cfg.lambda_bc * bc_loss
    )
    return {
        "total": total,
        "pde": pde_loss,
        "ic": ic_loss,
        "bc": bc_loss,
    }


def density_and_current(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a1, b1, a2, b2 = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
    rho = a1**2 + b1**2 + a2**2 + b2**2
    current = 2.0 * (a1 * a2 + b1 * b2)
    return rho, current


def evaluate(
    model: DiracPINN,
    cfg: Config,
    device: torch.device,
    results_dir: Path,
) -> dict[str, float]:
    x = torch.linspace(cfg.x_min, cfg.x_max, 600, device=device).view(-1, 1)
    t = torch.full_like(x, cfg.t_max)
    with torch.no_grad():
        pred = model(x, t)
        target = exact_solution(x, t, cfg)
        rho_pred, j_pred = density_and_current(pred)
        rho_target, j_target = density_and_current(target)

    rel_l2 = torch.linalg.norm(pred - target) / torch.linalg.norm(target)
    rho_l2 = torch.linalg.norm(rho_pred - rho_target) / torch.linalg.norm(rho_target)
    j_l2 = torch.linalg.norm(j_pred - j_target) / torch.linalg.norm(j_target)

    x_np = x.detach().cpu().numpy().reshape(-1)
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    rho_pred_np = rho_pred.detach().cpu().numpy().reshape(-1)
    rho_target_np = rho_target.detach().cpu().numpy().reshape(-1)
    j_pred_np = j_pred.detach().cpu().numpy().reshape(-1)
    j_target_np = j_target.detach().cpu().numpy().reshape(-1)

    np.savetxt(
        results_dir / "prediction_final_time.csv",
        np.column_stack(
            [
                x_np,
                pred_np,
                target_np,
                rho_pred_np,
                rho_target_np,
                j_pred_np,
                j_target_np,
            ]
        ),
        delimiter=",",
        header=(
            "x,pred_psi1_re,pred_psi1_im,pred_psi2_re,pred_psi2_im,"
            "exact_psi1_re,exact_psi1_im,exact_psi2_re,exact_psi2_im,"
            "pred_rho,exact_rho,pred_current,exact_current"
        ),
        comments="",
    )

    plt.figure(figsize=(10, 6))
    labels = [
        "Re psi1",
        "Im psi1",
        "Re psi2",
        "Im psi2",
    ]
    for i, label in enumerate(labels):
        plt.subplot(2, 2, i + 1)
        plt.plot(x_np, target_np[:, i], "k-", linewidth=2, label="exact")
        plt.plot(x_np, pred_np[:, i], "r--", linewidth=1.5, label="PINN")
        plt.title(label)
        plt.xlabel("x")
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "final_components.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x_np, rho_target_np, "k-", linewidth=2, label="exact")
    plt.plot(x_np, rho_pred_np, "r--", linewidth=1.5, label="PINN")
    plt.title("Density rho")
    plt.xlabel("x")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_np, j_target_np, "k-", linewidth=2, label="exact")
    plt.plot(x_np, j_pred_np, "r--", linewidth=1.5, label="PINN")
    plt.title("Probability current j")
    plt.xlabel("x")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "density_current.png", dpi=200)
    plt.close()

    metrics = {
        "relative_spinor_l2_at_final_time": float(rel_l2.detach().cpu()),
        "relative_density_l2_at_final_time": float(rho_l2.detach().cpu()),
        "relative_current_l2_at_final_time": float(j_l2.detach().cpu()),
        "mass": cfg.mass,
        "wave_number": cfg.wave_number,
        "energy": cfg.energy,
        "t_final": cfg.t_max,
    }
    with (results_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    return metrics


def save_loss_plot(history: list[dict[str, float]], results_dir: Path) -> None:
    epochs = np.arange(1, len(history) + 1)
    plt.figure(figsize=(8, 5))
    for key in ["total", "pde", "ic", "bc"]:
        values = np.array([row[key] for row in history])
        plt.semilogy(epochs, values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "loss_curve.png", dpi=200)
    plt.close()

    np.savetxt(
        results_dir / "loss_history.csv",
        np.array([[row["total"], row["pde"], row["ic"], row["bc"]] for row in history]),
        delimiter=",",
        header="total,pde,ic,bc",
        comments="",
    )


def train(cfg: Config, results_dir: Path) -> dict[str, float]:
    set_random_seed(cfg.seed)
    device = get_device()
    results_dir.mkdir(parents=True, exist_ok=True)

    model = DiracPINN(cfg.hidden_dim, cfg.hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    history: list[dict[str, float]] = []

    print(f"Device: {device}")
    print(
        "Equation: i psi_t = -i sigma_x psi_x + m sigma_z psi, "
        f"m={cfg.mass}, k={cfg.wave_number}, E={cfg.energy:.6f}"
    )

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(model, cfg, device)
        losses["total"].backward()
        optimizer.step()

        row = {key: float(value.detach().cpu()) for key, value in losses.items()}
        history.append(row)

        if epoch == 1 or epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            print(
                f"epoch {epoch:5d} | total {row['total']:.3e} | "
                f"pde {row['pde']:.3e} | ic {row['ic']:.3e} | bc {row['bc']:.3e}"
            )

    save_loss_plot(history, results_dir)
    return evaluate(model, cfg, device, results_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a minimal PINN for the 1+1D free Dirac equation."
    )
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--collocation", type=int, default=Config.num_collocation)
    parser.add_argument("--initial", type=int, default=Config.num_initial)
    parser.add_argument("--boundary", type=int, default=Config.num_boundary)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--hidden-layers", type=int, default=Config.hidden_layers)
    parser.add_argument("--lr", type=float, default=Config.learning_rate)
    parser.add_argument("--mass", type=float, default=Config.mass)
    parser.add_argument("--wave-number", type=float, default=Config.wave_number)
    parser.add_argument("--t-max", type=float, default=Config.t_max)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        mass=args.mass,
        wave_number=args.wave_number,
        t_max=args.t_max,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        num_collocation=args.collocation,
        num_initial=args.initial,
        num_boundary=args.boundary,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
    )
    metrics = train(cfg, args.results_dir)
    print("Final metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"Saved results to: {args.results_dir}")


if __name__ == "__main__":
    main()
