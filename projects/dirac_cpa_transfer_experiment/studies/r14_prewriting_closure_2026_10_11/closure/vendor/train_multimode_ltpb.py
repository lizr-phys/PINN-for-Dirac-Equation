from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from .multimode_config import (
    MULTIMODE_EXPERIMENT,
    MULTIMODE_MODELS,
    MULTIMODE_SEEDS,
    MultimodeConfig,
    MultimodeRunSpec,
)
from .multimode_models import (
    build_multimode_model,
    count_trainable_parameters,
    trainable_state_sha256,
)
from .physics import dirac_residual, spinor_derivatives


INITIALIZATION_SEED_OFFSET = 200_000
COLLOCATION_SEED_OFFSET = 300_000
RUN_ARTIFACTS = (
    "config.json",
    "metrics.json",
    "loss_history.csv",
    "time_series.csv",
    "dense_grid.npz",
    "model_state.pt",
)


def paired_initialization_seed(seed: int) -> int:
    """Return the model-independent initialization stream for a paired seed."""

    return INITIALIZATION_SEED_OFFSET + int(seed)


def paired_collocation_seed(seed: int) -> int:
    """Return the model-independent collocation stream for a paired seed."""

    return COLLOCATION_SEED_OFFSET + int(seed)


def collocation_seed(seed: int) -> int:
    """Backward-compatible public name used by stream-pairing tests."""

    return paired_collocation_seed(seed)


def sample_collocation(
    cfg: MultimodeConfig,
    generator: torch.Generator,
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw one fresh batch from an explicitly supplied random stream."""

    target = torch.device(device)
    if target.type != "cpu":
        raise ValueError("The preregistered paired sampler uses the CPU stream")
    x = cfg.x_min + cfg.length * torch.rand(
        (cfg.collocation_points, 1),
        generator=generator,
        device=target,
        dtype=torch.float64,
    )
    t = cfg.t_max * torch.rand(
        (cfg.collocation_points, 1),
        generator=generator,
        device=target,
        dtype=torch.float64,
    )
    return x.requires_grad_(True), t.requires_grad_(True)


class PairedCollocationSampler:
    """Fresh uniform points from a stream shared by models with the same seed."""

    def __init__(
        self,
        cfg: MultimodeConfig,
        seed: int,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.seed = paired_collocation_seed(seed)
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError("The preregistered paired sampler uses the CPU stream")
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.batch_index = 0

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        x, t = sample_collocation(
            self.cfg, self.generator, device=self.device
        )
        self.batch_index += 1
        return x.requires_grad_(True), t.requires_grad_(True)


def multimode_pde_loss(
    model: torch.nn.Module,
    cfg: MultimodeConfig,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Mean-squared strong Dirac residual; this is the sole training term."""

    psi = model(x, t)
    if psi.shape != (x.shape[0], 2) or psi.dtype != torch.complex128:
        raise TypeError(
            "Multimode model must return complex128 with shape (batch, 2); "
            f"received shape={tuple(psi.shape)}, dtype={psi.dtype}"
        )
    psi_x, psi_t = spinor_derivatives(psi, x, t, create_graph=True)
    residual = dirac_residual(psi, psi_x, psi_t, mass=cfg.mass)
    return torch.mean(torch.abs(residual) ** 2)


def _set_runtime(seed: int, cpu_threads: int) -> torch.device:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this only before the first parallel operation.
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return torch.device("cpu")


def _environment_snapshot(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": str(device),
        "default_dtype": str(torch.get_default_dtype()),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _validate_formal_contract(
    cfg: MultimodeConfig, spec: MultimodeRunSpec
) -> None:
    if spec.experiment != MULTIMODE_EXPERIMENT:
        return
    expected: dict[str, Any] = {
        "x_min": 0.0,
        "x_max": 2.0 * math.pi,
        "t_max": 5.0,
        "mass": 5.0,
        "wave_numbers": (2.0, 5.0),
        "hidden_layers": 4,
        "baseline_width": 64,
        "multichannel_width": 63,
        "collocation_points": 1024,
        "adam_steps": 5000,
        "learning_rate": 1.0e-3,
        "test_x_points": 128,
        "test_t_points": 201,
        "history_every": 10,
        "dtype": "float64",
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(cfg, key)}
        for key, value in expected.items()
        if getattr(cfg, key) != value
    }
    if spec.model not in MULTIMODE_MODELS:
        mismatches["model"] = {
            "expected": list(MULTIMODE_MODELS),
            "actual": spec.model,
        }
    if spec.seed not in MULTIMODE_SEEDS:
        mismatches["seed"] = {
            "expected": list(MULTIMODE_SEEDS),
            "actual": spec.seed,
        }
    if spec.steps != cfg.adam_steps:
        mismatches["steps"] = {
            "expected": cfg.adam_steps,
            "actual": spec.steps,
        }
    if mismatches:
        raise ValueError(
            "Formal multimode run violates the preregistered training contract: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _classify_existing_run(run_dir: Path) -> str:
    if not run_dir.exists():
        return "absent"
    if not run_dir.is_dir():
        return "invalid"
    names = {path.name for path in run_dir.iterdir()}
    required = set(RUN_ARTIFACTS)
    return "complete" if names == required else "invalid"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            _jsonable(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def _dense_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate_in_memory(
    model: torch.nn.Module,
    cfg: MultimodeConfig,
    spec: MultimodeRunSpec,
    *,
    parameter_count: int,
    training_seconds: float,
    final_pde_loss: float,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    # Delayed import keeps the trainer usable while the independent evaluator is
    # tested or replaced, and prevents reference-only code from entering loss setup.
    from .multimode_evaluate import compute_multimode_evaluation

    metrics, time_series, dense = compute_multimode_evaluation(
        model,
        cfg,
        model_name=spec.model,
        seed=spec.seed,
        parameter_count=parameter_count,
        training_seconds=training_seconds,
        final_pde_loss=final_pde_loss,
    )
    if not isinstance(time_series, pd.DataFrame):
        raise TypeError("Multimode evaluator must return time_series as DataFrame")
    if not isinstance(dense, dict):
        raise TypeError("Multimode evaluator must return dense arrays as a dict")
    return dict(metrics), time_series, dense


def _config_snapshot(
    cfg: MultimodeConfig,
    spec: MultimodeRunSpec,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "experiment_config": cfg.as_dict(),
        "run_spec": spec.as_dict(),
        "environment": _environment_snapshot(device),
        "architecture": {
            "input_features": ["sin(x)", "cos(x)", "t/T"],
            "hidden_layers": cfg.hidden_layers,
            "hidden_width": cfg.width_for_model(spec.model),
            "output_dim": int(getattr(model, "output_dim")),
            "activation": "tanh",
            "hard_initial_condition": True,
            "known_symbol_basis_from_initial_data": spec.model
            in {"multi_branch_complex", "multi_ltpb"},
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": cfg.learning_rate,
            "steps": spec.steps,
        },
        "training_objective": {
            "terms": ["strong_dirac_pde_residual"],
            "supervised_labels": False,
            "auxiliary_constraints": False,
        },
        "random_streams": {
            "model_initialization_seed": paired_initialization_seed(spec.seed),
            "collocation_seed": paired_collocation_seed(spec.seed),
            "streams_are_separate": True,
            "paired_across_models_for_same_seed": True,
            "collocation_resampled_each_step": True,
        },
        "trainable_initial_state_sha256": trainable_state_sha256(model),
        "parameter_count": count_trainable_parameters(model),
        "artifact_contract": list(RUN_ARTIFACTS),
    }


def _checkpoint_payload(
    cfg: MultimodeConfig,
    spec: MultimodeRunSpec,
    model: torch.nn.Module,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": spec.experiment,
        "model": spec.model,
        "seed": spec.seed,
        "steps": spec.steps,
        "experiment_config": cfg.as_dict(),
        "architecture": {
            "hidden_layers": cfg.hidden_layers,
            "hidden_width": cfg.width_for_model(spec.model),
            "output_dim": int(getattr(model, "output_dim")),
        },
        "random_streams": {
            "model_initialization_seed": paired_initialization_seed(spec.seed),
            "collocation_seed": paired_collocation_seed(spec.seed),
        },
        "state_dict": model.state_dict(),
    }


def train_multimode_run(
    cfg: MultimodeConfig,
    spec: MultimodeRunSpec,
    run_dir: Path,
) -> dict[str, Any]:
    """Train and atomically commit one learned-model run.

    A complete existing run is read without modification. Any other existing
    target is rejected, so neither formal nor smoke output can be overwritten
    accidentally.
    """

    run_dir = Path(run_dir)
    _validate_formal_contract(cfg, spec)
    state = _classify_existing_run(run_dir)
    if state == "complete":
        with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        print(f"SKIP completed {spec.run_name}", flush=True)
        return cached
    if state != "absent":
        names = sorted(path.name for path in run_dir.iterdir()) if run_dir.is_dir() else []
        raise RuntimeError(
            f"Refusing to overwrite incomplete or invalid run target {run_dir}; "
            f"found={names}, required={list(RUN_ARTIFACTS)}"
        )

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = run_dir.parent / f".{run_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    committed = False
    try:
        device = _set_runtime(spec.seed, cfg.cpu_threads)
        model = build_multimode_model(
            spec.model,
            cfg,
            initialization_seed=paired_initialization_seed(spec.seed),
        ).to(device=device, dtype=torch.float64)
        parameter_count = count_trainable_parameters(model)
        _write_json(staging / "config.json", _config_snapshot(cfg, spec, model, device))

        sampler = PairedCollocationSampler(cfg, spec.seed, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        history: list[dict[str, float | int]] = []
        final_pde_loss = math.nan
        started = time.perf_counter()
        print(
            f"START {spec.run_name} | device={device} | params={parameter_count} | "
            f"steps={spec.steps} | collocation_seed={sampler.seed}",
            flush=True,
        )
        for step in range(1, spec.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            x, t = sampler.sample()
            loss = multimode_pde_loss(model, cfg, x, t)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite PDE loss in {spec.run_name} at step {step}: {loss}"
                )
            loss.backward()
            optimizer.step()
            final_pde_loss = float(loss.detach().cpu())
            elapsed = time.perf_counter() - started
            if step == 1 or step % cfg.history_every == 0 or step == spec.steps:
                history.append(
                    {
                        "step": step,
                        "total": final_pde_loss,
                        "pde": final_pde_loss,
                        "elapsed_seconds": elapsed,
                        "collocation_batch_index": sampler.batch_index,
                    }
                )
            if step == 1 or step % cfg.log_every == 0 or step == spec.steps:
                print(
                    f"{spec.run_name} step {step:5d}/{spec.steps} "
                    f"pde={final_pde_loss:.4e}",
                    flush=True,
                )

        training_seconds = time.perf_counter() - started
        history_frame = pd.DataFrame(history)
        expected_history_steps = sorted(
            {1, spec.steps}
            | set(range(cfg.history_every, spec.steps + 1, cfg.history_every))
        )
        if history_frame["step"].tolist() != expected_history_steps:
            raise RuntimeError("Loss-history cadence violated its artifact contract")
        history_frame.to_csv(staging / "loss_history.csv", index=False)

        torch.save(_checkpoint_payload(cfg, spec, model), staging / "model_state.pt")
        model.eval()
        metrics, time_series, dense = _evaluate_in_memory(
            model,
            cfg,
            spec,
            parameter_count=parameter_count,
            training_seconds=training_seconds,
            final_pde_loss=final_pde_loss,
        )
        metrics.update(
            {
                "experiment": spec.experiment,
                "model": spec.model,
                "seed": spec.seed,
                "steps": spec.steps,
                "parameter_count": parameter_count,
                "training_seconds": training_seconds,
                "final_pde_loss": final_pde_loss,
                "loss_terms_used": ["pde"],
            }
        )
        _write_json(staging / "metrics.json", metrics)
        time_series.to_csv(staging / "time_series.csv", index=False)
        np.savez_compressed(
            staging / "dense_grid.npz",
            **{str(key): _dense_numpy(value) for key, value in dense.items()},
        )

        actual = {path.name for path in staging.iterdir()}
        if actual != set(RUN_ARTIFACTS):
            raise RuntimeError(
                f"Staged run does not contain exactly six artifacts: {sorted(actual)}"
            )
        for artifact in RUN_ARTIFACTS:
            path = staging / artifact
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty staged artifact: {path}")
            _artifact_sha256(path)
        staging.rename(run_dir)
        committed = True
        print(
            f"DONE {spec.run_name} | pde={final_pde_loss:.4e} | "
            f"training={training_seconds:.1f}s",
            flush=True,
        )
        return metrics
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


# Deliberately explicit alias for runners and tests that use the older naming.
train_one_multimode_run = train_multimode_run
