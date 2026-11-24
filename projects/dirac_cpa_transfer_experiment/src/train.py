from __future__ import annotations

import json
import hashlib
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, RunSpec, write_json
from .evaluate import evaluate_model
from .losses import LossComputer, gradient_cosine
from .models import TwoChannelDiracModel, build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment_snapshot(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
        ),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "default_dtype": str(torch.get_default_dtype()),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def trainable_state_sha256(model: torch.nn.Module) -> str:
    """Hash trainable tensors only, excluding fixed physical-basis buffers."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def train_one_run(
    cfg: ExperimentConfig,
    spec: RunSpec,
    run_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        required_artifacts = (
            "config.json",
            "metrics.json",
            "loss_history.csv",
            "time_series.csv",
            "dense_grid.npz",
            "model_state.pt",
        )
        missing = [name for name in required_artifacts if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(
                f"Refusing to skip incomplete run {spec.run_name}; missing={missing}. "
                "Inspect the partial directory and rerun explicitly with --force."
            )
        with metrics_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        print(f"SKIP completed {spec.run_name}", flush=True)
        return cached

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    set_seed(spec.seed)
    device = select_device()
    model = build_model(spec.model, cfg, spec.eta).to(device=device, dtype=torch.float64)
    parameter_count = count_parameters(model)
    conservative = spec.model == "sp_cpa_conservative"
    collocation_generator = torch.Generator(device=device.type)
    collocation_generator.manual_seed(100_000 + spec.seed)
    structure_generator = torch.Generator(device=device.type)
    structure_generator.manual_seed(200_000 + spec.seed)
    loss_computer = LossComputer(
        cfg,
        model,
        conservative=conservative,
        lambda_struct=spec.lambda_struct,
        device=device,
        collocation_generator=collocation_generator,
        structure_generator=structure_generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    snapshot = {
        "experiment_config": cfg.as_dict(),
        "run_spec": spec.as_dict(),
        "environment": environment_snapshot(device),
        "initialization": "Xavier normal for every linear weight; zero biases",
        "model_initialization_seed": spec.seed,
        "trainable_initial_state_sha256": trainable_state_sha256(model),
        "sampling": {
            "collocation_generator_seed": 100_000 + spec.seed,
            "structure_generator_seed": 200_000 + spec.seed,
            "collocation_resampled_each_step": True,
            "periodic_spatial_quadrature_endpoint": False,
        },
    }
    eig = model.eigensystem
    snapshot["eigensystem"] = {
        "energy": eig.energy,
        "u_plus": eig.u_plus.detach().cpu().tolist(),
        "u_minus": eig.u_minus.detach().cpu().tolist(),
        "u_plus_sigma_x_u_minus": eig.cross_current,
    }
    if isinstance(model, TwoChannelDiracModel):
        if model.model_name in {"ltpb_cpa", "branch_complex"}:
            rotation_degrees: float | None = 0.0
        else:
            rotation_degrees = getattr(model, "angle_degrees", None)
        snapshot["two_channel_parameterization"] = {
            "basis_matrix_columns": model.channel_basis_matrix.detach().cpu().tolist(),
            "initial_coefficients": model.initial_channel_coefficients.detach().cpu().tolist(),
            "rotation_degrees": rotation_degrees,
            "energy_basis_aligned": model.model_name
            in {"ltpb_cpa", "branch_complex"},
            "uses_exact_energy_or_time_carrier_in_forward": False,
        }
    write_json(run_dir / "config.json", snapshot)

    print(
        f"START {spec.run_name} | device={device} | params={parameter_count} | "
        f"steps={spec.steps}",
        flush=True,
    )
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    final_loss_values: dict[str, float] = {}
    for step in range(1, spec.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = loss_computer()
        cosine = float("nan")
        if conservative and (
            step == 1
            or step % cfg.gradient_diagnostic_every == 0
            or step == spec.steps
        ):
            cosine = gradient_cosine(losses["pde"], losses["structure"], model)
        losses["total"].backward()
        optimizer.step()
        final_loss_values = {
            key: float(value.detach().cpu()) for key, value in losses.items()
        }
        if step == 1 or step % cfg.history_every == 0 or step == spec.steps:
            history.append(
                {
                    "step": step,
                    **final_loss_values,
                    "gradient_cosine": cosine,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        if step == 1 or step % cfg.log_every == 0 or step == spec.steps:
            print(
                f"{spec.run_name} step {step:5d}/{spec.steps} "
                f"total={final_loss_values['total']:.4e} "
                f"pde={final_loss_values['pde']:.4e} "
                f"struct={final_loss_values['structure']:.4e}",
                flush=True,
            )
        if not np.isfinite(final_loss_values["total"]):
            raise FloatingPointError(
                f"Non-finite loss in {spec.run_name} at step {step}: {final_loss_values}"
            )

    training_seconds = time.perf_counter() - started
    pd.DataFrame(history).to_csv(run_dir / "loss_history.csv", index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "experiment_config": cfg.as_dict(),
            "run_spec": spec.as_dict(),
        },
        run_dir / "model_state.pt",
    )
    metrics = evaluate_model(
        model,
        cfg,
        spec,
        run_dir,
        parameter_count=parameter_count,
        training_seconds=training_seconds,
        final_losses=final_loss_values,
    )
    print(
        f"DONE {spec.run_name} | spinor={metrics['spinor_rel_l2']:.4e} | "
        f"current={metrics['current_rel_l2']:.4e} | "
        f"fidelity={metrics['fidelity_mean']:.6f} | {training_seconds:.1f}s",
        flush=True,
    )
    return metrics
