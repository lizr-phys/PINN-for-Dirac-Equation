from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from .physics import complex_derivative, dirac_residual, spinor_derivatives
from .wavepacket_config import (
    WAVEPACKET_EXPERIMENT,
    WAVEPACKET_MODELS,
    WAVEPACKET_SEEDS,
    WavepacketConfig,
    WavepacketRunSpec,
    paired_collocation_seed,
    paired_initialization_seed,
)
from .wavepacket_physics import (
    spectral_dirac_residual,
    training_wave_numbers,
)


PHYSICAL_X_SEED_OFFSET = 400_000
SPECTRAL_MODELS = frozenset(
    {"shared_spectral_complex", "shared_spectral_ltpb"}
)
PHYSICAL_MODELS = frozenset(set(WAVEPACKET_MODELS).difference(SPECTRAL_MODELS))
RUN_ARTIFACTS = (
    "config.json",
    "metrics.json",
    "loss_history.csv",
    "time_series.csv",
    "dense_grid.npz",
    "model_state.pt",
)


def paired_time_seed(seed: int) -> int:
    """Model-independent time stream used by every learned model."""

    return paired_collocation_seed(seed)


def paired_physical_x_seed(seed: int) -> int:
    """Independent physical-position stream used by physical baselines."""

    return PHYSICAL_X_SEED_OFFSET + int(seed)


def sample_time_points(
    cfg: WavepacketConfig,
    generator: torch.Generator,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    target = torch.device(device)
    if target.type != "cpu":
        raise ValueError("The preregistered paired sampler uses CPU generators")
    return cfg.t_max * torch.rand(
        (cfg.spectral_time_points, 1),
        generator=generator,
        device=target,
        dtype=torch.float64,
    )


def sample_physical_positions(
    cfg: WavepacketConfig,
    generator: torch.Generator,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    target = torch.device(device)
    if target.type != "cpu":
        raise ValueError("The preregistered paired sampler uses CPU generators")
    points_per_axis = math.isqrt(cfg.physical_collocation_points)
    if points_per_axis**2 != cfg.physical_collocation_points:
        raise ValueError(
            "physical_collocation_points must be a square Cartesian batch"
        )
    return cfg.x_min + cfg.length * torch.rand(
        (points_per_axis, 1),
        generator=generator,
        device=target,
        dtype=torch.float64,
    )


class PairedTimeSampler:
    """Fresh 32-time batches paired across all models for one seed."""

    def __init__(
        self,
        cfg: WavepacketConfig,
        seed: int,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.seed = paired_time_seed(seed)
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError("The preregistered paired sampler uses CPU generators")
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.batch_index = 0

    def sample(self) -> torch.Tensor:
        values = sample_time_points(
            self.cfg, self.generator, device=self.device
        )
        self.batch_index += 1
        return values


class PhysicalCollocationSampler:
    """A fresh 32-time by 32-position Cartesian grid each Adam step."""

    def __init__(
        self,
        cfg: WavepacketConfig,
        seed: int,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.time_sampler = PairedTimeSampler(cfg, seed, device=device)
        self.x_seed = paired_physical_x_seed(seed)
        self.x_generator = torch.Generator(device="cpu").manual_seed(self.x_seed)

    @property
    def batch_index(self) -> int:
        return self.time_sampler.batch_index

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        times = self.time_sampler.sample()
        positions = sample_physical_positions(
            self.cfg, self.x_generator, device=self.device
        )
        n_t = times.shape[0]
        n_x = positions.shape[0]
        x_flat = positions.repeat(n_t, 1).detach().clone().requires_grad_(True)
        t_flat = (
            times.repeat_interleave(n_x, dim=0)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        if x_flat.shape[0] != self.cfg.physical_collocation_points:
            raise RuntimeError("Physical Cartesian batch size violates its contract")
        return x_flat, t_flat


class SpectralCollocationSampler:
    """All 33 training wave numbers crossed with one fresh 32-time batch."""

    def __init__(
        self,
        cfg: WavepacketConfig,
        seed: int,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.time_sampler = PairedTimeSampler(cfg, seed, device=device)
        self.wave_numbers = training_wave_numbers(
            cfg, device=self.device, dtype=torch.float64
        ).reshape(-1, 1)

    @property
    def batch_index(self) -> int:
        return self.time_sampler.batch_index

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        times = self.time_sampler.sample()
        n_t = times.shape[0]
        n_k = self.wave_numbers.shape[0]
        k_flat = self.wave_numbers.repeat(n_t, 1)
        t_flat = (
            times.repeat_interleave(n_k, dim=0)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        if k_flat.shape[0] != self.cfg.spectral_collocation_points:
            raise RuntimeError("Spectral Cartesian batch size violates its contract")
        return k_flat, t_flat


def physical_pde_loss(
    model: torch.nn.Module,
    cfg: WavepacketConfig,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Physical-space strong Dirac residual, with no auxiliary terms."""

    psi = model(x, t)
    if psi.shape != (x.shape[0], 2) or psi.dtype != torch.complex128:
        raise TypeError(
            "Physical wavepacket model must return complex128 (batch,2); "
            f"received shape={tuple(psi.shape)}, dtype={psi.dtype}"
        )
    psi_x, psi_t = spinor_derivatives(psi, x, t, create_graph=True)
    residual = dirac_residual(psi, psi_x, psi_t, mass=cfg.mass)
    return torch.mean(torch.abs(residual) ** 2)


def spectral_pde_loss(
    model: torch.nn.Module,
    cfg: WavepacketConfig,
    wave_numbers: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Parseval-scaled spectral Dirac strong residual.

    This evaluates the original matrix symbol directly.  The exact dispersion
    and a diagonal energy equation are deliberately absent.  With residual
    shape ``(n_t,n_k,2)``, the registered loss is
    ``mean_t mean_component sum_k |R|^2 / L``.
    """

    psi = model(wave_numbers, t)
    if psi.shape != (wave_numbers.shape[0], 2) or psi.dtype != torch.complex128:
        raise TypeError(
            "Shared spectral model must return complex128 (batch,2); "
            f"received shape={tuple(psi.shape)}, dtype={psi.dtype}"
        )
    psi_t = torch.stack(
        [
            complex_derivative(psi[:, component], t, create_graph=True)
            for component in range(2)
        ],
        dim=1,
    )
    residual = spectral_dirac_residual(
        psi, psi_t, wave_numbers.reshape(-1), cfg.mass
    )
    expected = (cfg.spectral_time_points, cfg.mode_count, 2)
    if residual.numel() != math.prod(expected):
        raise RuntimeError(
            f"Spectral residual cannot be reshaped to {expected}: {residual.shape}"
        )
    power = torch.abs(residual.reshape(expected)) ** 2
    return torch.sum(power, dim=1).mean() / cfg.length


def wavepacket_pde_loss(
    model_name: str,
    model: torch.nn.Module,
    cfg: WavepacketConfig,
    first: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    if model_name in SPECTRAL_MODELS:
        return spectral_pde_loss(model, cfg, first, t)
    if model_name in PHYSICAL_MODELS:
        return physical_pde_loss(model, cfg, first, t)
    raise ValueError(f"Unknown wavepacket model: {model_name}")


def _windows_rss_bytes() -> tuple[int, int]:
    """Current and OS peak working set on Windows, or ``(0,0)`` elsewhere."""

    if os.name != "nt":
        return 0, 0

    size_t = ctypes.c_size_t

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", size_t),
            ("WorkingSetSize", size_t),
            ("QuotaPeakPagedPoolUsage", size_t),
            ("QuotaPagedPoolUsage", size_t),
            ("QuotaPeakNonPagedPoolUsage", size_t),
            ("QuotaNonPagedPoolUsage", size_t),
            ("PagefileUsage", size_t),
            ("PeakPagefileUsage", size_t),
            ("PrivateUsage", size_t),
        ]

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return 0, 0
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def _portable_rss_bytes() -> tuple[int, int, str]:
    current, peak = _windows_rss_bytes()
    if current > 0:
        return current, peak, "Windows GetProcessMemoryInfo + polling thread"

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        current = int(page_size * resident_pages)
    except (AttributeError, FileNotFoundError, IndexError, OSError, ValueError):
        current = 0
    try:
        import resource

        raw_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak = raw_peak if sys.platform == "darwin" else raw_peak * 1024
    except (ImportError, ValueError):
        peak = current
    return current, max(current, peak), "procfs/resource + polling thread"


class ProcessRSSMonitor:
    """Poll native process RSS so PyTorch tensor allocations are included."""

    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.interval_seconds = interval_seconds
        self.initial_bytes = 0
        self.current_bytes = 0
        self.sampled_peak_bytes = 0
        self.os_peak_bytes = 0
        self.method = "uninitialized"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _sample(self) -> None:
        current, os_peak, method = _portable_rss_bytes()
        with self._lock:
            self.current_bytes = current
            self.sampled_peak_bytes = max(self.sampled_peak_bytes, current)
            self.os_peak_bytes = max(self.os_peak_bytes, os_peak)
            self.method = method

    def _poll(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("RSS monitor is already started")
        self._sample()
        self.initial_bytes = self.current_bytes
        self._thread = threading.Thread(
            target=self._poll, name="wavepacket-rss-monitor", daemon=True
        )
        self._thread.start()

    def snapshot(self) -> dict[str, int | str]:
        self._sample()
        with self._lock:
            return {
                "rss_current_bytes": self.current_bytes,
                "rss_sampled_peak_bytes": self.sampled_peak_bytes,
                "rss_os_peak_bytes": self.os_peak_bytes,
                "rss_measurement_method": self.method,
            }

    def stop(self) -> dict[str, int | str]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(1.0, 10.0 * self.interval_seconds))
            self._thread = None
        return self.snapshot()


def _set_runtime(seed: int, cpu_threads: int) -> torch.device:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
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
    cfg: WavepacketConfig, spec: WavepacketRunSpec
) -> None:
    if spec.experiment != WAVEPACKET_EXPERIMENT:
        return
    expected: dict[str, Any] = {
        "x_min": -8.0 * math.pi,
        "x_max": 8.0 * math.pi,
        "t_max": 5.0,
        "mass": 5.0,
        "n_min": 0,
        "n_max": 32,
        "k0": 2.0,
        "sigma_k": 0.5,
        "hidden_layers": 4,
        "hidden_width": 64,
        "physical_collocation_points": 1024,
        "spectral_time_points": 32,
        "adam_steps": 5000,
        "learning_rate": 1.0e-3,
        "test_x_points": 256,
        "test_t_points": 201,
        "history_every": 10,
        "dtype": "float64",
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(cfg, key)}
        for key, value in expected.items()
        if getattr(cfg, key) != value
    }
    if cfg.mode_count != 33 or cfg.spectral_collocation_points != 1056:
        mismatches["spectral_cartesian_batch"] = {
            "expected": {"modes": 33, "points": 1056},
            "actual": {
                "modes": cfg.mode_count,
                "points": cfg.spectral_collocation_points,
            },
        }
    if spec.model not in WAVEPACKET_MODELS:
        mismatches["model"] = {
            "expected": list(WAVEPACKET_MODELS),
            "actual": spec.model,
        }
    if spec.seed not in WAVEPACKET_SEEDS:
        mismatches["seed"] = {
            "expected": list(WAVEPACKET_SEEDS),
            "actual": spec.seed,
        }
    if spec.steps != cfg.adam_steps:
        mismatches["steps"] = {
            "expected": cfg.adam_steps,
            "actual": spec.steps,
        }
    if mismatches:
        raise ValueError(
            "Formal wavepacket run violates its preregistered contract: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _classify_existing_run(run_dir: Path) -> str:
    if not run_dir.exists():
        return "absent"
    if not run_dir.is_dir():
        return "invalid"
    names = {path.name for path in run_dir.iterdir()}
    return "complete" if names == set(RUN_ARTIFACTS) else "invalid"


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
            allow_nan=False,
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


def _load_model_helpers() -> tuple[Any, Any, Any]:
    from .wavepacket_models import (
        build_wavepacket_model,
        count_trainable_parameters,
        trainable_state_sha256,
    )

    return (
        build_wavepacket_model,
        count_trainable_parameters,
        trainable_state_sha256,
    )


def _evaluate_in_memory(
    model: torch.nn.Module,
    cfg: WavepacketConfig,
    spec: WavepacketRunSpec,
    *,
    parameter_count: int,
    training_seconds: float,
    final_pde_loss: float,
    peak_rss_mb: float,
    peak_rss_increment_mb: float,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    # Reference-only quantities first enter through this delayed, post-training
    # evaluator import; they are absent from model construction and the loss.
    from .wavepacket_evaluate import compute_wavepacket_evaluation

    metrics, time_series, dense = compute_wavepacket_evaluation(
        model,
        cfg,
        model_name=spec.model,
        seed=spec.seed,
        parameter_count=parameter_count,
        training_seconds=training_seconds,
        final_pde_loss=final_pde_loss,
        peak_rss_mb=peak_rss_mb,
        peak_rss_increment_mb=peak_rss_increment_mb,
    )
    if not isinstance(time_series, pd.DataFrame):
        raise TypeError("Wavepacket evaluator must return time_series as DataFrame")
    if not isinstance(dense, dict):
        raise TypeError("Wavepacket evaluator must return dense arrays as a dict")
    return dict(metrics), time_series, dense


def _architecture_snapshot(model: torch.nn.Module, model_name: str) -> dict[str, Any]:
    spectral = model_name in SPECTRAL_MODELS
    return {
        "domain": "spectral" if spectral else "physical",
        "input_features": ["normalized_k", "t/T"]
        if spectral
        else ["sin(2*pi*(x-x_min)/L)", "cos(2*pi*(x-x_min)/L)", "t/T"],
        "input_dim": int(getattr(model, "input_dim", 2 if spectral else 3)),
        "hidden_layers": int(getattr(model, "hidden_layers", 4)),
        "hidden_width": int(getattr(model, "width", 64)),
        "output_dim": int(getattr(model, "output_dim")),
        "activation": "tanh",
        "hard_initial_condition": True,
        "known_symbol_basis_from_initial_data": spectral,
    }


def _config_snapshot(
    cfg: WavepacketConfig,
    spec: WavepacketRunSpec,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    _, count_parameters, initial_hash = _load_model_helpers()
    spectral = spec.model in SPECTRAL_MODELS
    return {
        "experiment_config": cfg.as_dict(),
        "run_spec": spec.as_dict(),
        "environment": _environment_snapshot(device),
        "architecture": _architecture_snapshot(model, spec.model),
        "optimizer": {
            "name": "Adam",
            "learning_rate": cfg.learning_rate,
            "steps": spec.steps,
        },
        "training_objective": {
            "terms": ["strong_dirac_pde_residual"],
            "supervised_labels": False,
            "auxiliary_constraints": False,
            "spectral_matrix_symbol_used_directly": spectral,
            "spectral_parseval_scale": "mean_t mean_component sum_k |R|^2 / L"
            if spectral
            else None,
        },
        "collocation": {
            "time_points_per_step": cfg.spectral_time_points,
            "physical_position_points_per_step": 32 if not spectral else 0,
            "training_wave_number_count": cfg.mode_count if spectral else 0,
            "total_points_per_step": cfg.spectral_collocation_points
            if spectral
            else cfg.physical_collocation_points,
            "cartesian_product": True,
        },
        "random_streams": {
            "model_initialization_seed": paired_initialization_seed(spec.seed),
            "shared_time_seed": paired_time_seed(spec.seed),
            "physical_x_seed": paired_physical_x_seed(spec.seed)
            if not spectral
            else None,
            "streams_are_separate": True,
            "paired_across_models_for_same_seed": True,
            "resampled_each_step": True,
        },
        "memory_measurement": {
            "metric": "native process resident set size",
            "sampling_interval_seconds": 0.02,
            "includes_native_tensor_allocations": True,
        },
        "trainable_initial_state_sha256": initial_hash(model),
        "parameter_count": count_parameters(model),
        "artifact_contract": list(RUN_ARTIFACTS),
    }


def _checkpoint_payload(
    cfg: WavepacketConfig,
    spec: WavepacketRunSpec,
    model: torch.nn.Module,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": spec.experiment,
        "model": spec.model,
        "seed": spec.seed,
        "steps": spec.steps,
        "experiment_config": cfg.as_dict(),
        "architecture": _architecture_snapshot(model, spec.model),
        "random_streams": {
            "model_initialization_seed": paired_initialization_seed(spec.seed),
            "shared_time_seed": paired_time_seed(spec.seed),
            "physical_x_seed": paired_physical_x_seed(spec.seed)
            if spec.model in PHYSICAL_MODELS
            else None,
        },
        "state_dict": model.state_dict(),
    }


def train_wavepacket_run(
    cfg: WavepacketConfig,
    spec: WavepacketRunSpec,
    run_dir: Path,
) -> dict[str, Any]:
    """Train one learned model and atomically commit its six artifacts."""

    run_dir = Path(run_dir)
    _validate_formal_contract(cfg, spec)
    state = _classify_existing_run(run_dir)
    if state == "complete":
        cached = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        print(f"SKIP completed {spec.run_name}", flush=True)
        return cached
    if state != "absent":
        names = (
            sorted(path.name for path in run_dir.iterdir())
            if run_dir.is_dir()
            else []
        )
        raise RuntimeError(
            f"Refusing to overwrite incomplete or invalid target {run_dir}; "
            f"found={names}, required={list(RUN_ARTIFACTS)}"
        )

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = run_dir.parent / f".{run_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    committed = False
    monitor: ProcessRSSMonitor | None = None
    try:
        device = _set_runtime(spec.seed, cfg.cpu_threads)
        build_model, count_parameters, _ = _load_model_helpers()
        model = build_model(
            spec.model,
            cfg,
            initialization_seed=paired_initialization_seed(spec.seed),
        ).to(device=device)
        if any(parameter.dtype != torch.float64 for parameter in model.parameters()):
            raise TypeError("All learned wavepacket parameters must be float64")
        parameter_count = count_parameters(model)
        _write_json(
            staging / "config.json", _config_snapshot(cfg, spec, model, device)
        )

        sampler: PhysicalCollocationSampler | SpectralCollocationSampler
        if spec.model in SPECTRAL_MODELS:
            sampler = SpectralCollocationSampler(cfg, spec.seed, device=device)
        else:
            sampler = PhysicalCollocationSampler(cfg, spec.seed, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        history: list[dict[str, float | int]] = []
        final_pde_loss = math.nan
        monitor = ProcessRSSMonitor()
        monitor.start()
        started = time.perf_counter()
        print(
            f"START {spec.run_name} | device={device} | params={parameter_count} | "
            f"steps={spec.steps} | time_seed={paired_time_seed(spec.seed)}",
            flush=True,
        )
        for step in range(1, spec.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            first, t = sampler.sample()
            loss = wavepacket_pde_loss(spec.model, model, cfg, first, t)
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
        training_memory = monitor.snapshot()
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
            peak_rss_mb=float(training_memory["rss_sampled_peak_bytes"]) / 2**20,
            peak_rss_increment_mb=max(
                0.0,
                (
                    float(training_memory["rss_sampled_peak_bytes"])
                    - float(monitor.initial_bytes)
                )
                / 2**20,
            ),
        )
        initial_rss_bytes = monitor.initial_bytes
        run_memory = monitor.stop()
        monitor = None
        metrics.update(
            {
                "experiment": spec.experiment,
                "model": spec.model,
                "seed": spec.seed,
                "steps": spec.steps,
                "parameter_count": parameter_count,
                "training_seconds": training_seconds,
                "final_pde_loss": final_pde_loss,
                "final_loss_pde": final_pde_loss,
                "loss_terms_used": ["pde"],
                "rss_initial_bytes": int(initial_rss_bytes),
                "rss_current_after_training_bytes": int(
                    training_memory["rss_current_bytes"]
                ),
                "rss_peak_training_bytes": int(
                    training_memory["rss_sampled_peak_bytes"]
                ),
                "rss_current_after_evaluation_bytes": int(
                    run_memory["rss_current_bytes"]
                ),
                "peak_rss_bytes": int(run_memory["rss_sampled_peak_bytes"]),
                "os_peak_rss_bytes": int(run_memory["rss_os_peak_bytes"]),
                "rss_measurement_method": str(
                    run_memory["rss_measurement_method"]
                ),
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
                f"Staged run must contain exactly six artifacts: {sorted(actual)}"
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
            f"training={training_seconds:.1f}s | "
            f"peak_rss={int(metrics['peak_rss_bytes']) / 2**20:.1f} MiB",
            flush=True,
        )
        return metrics
    finally:
        if monitor is not None:
            monitor.stop()
        if not committed and staging.exists():
            shutil.rmtree(staging)


train_one_wavepacket_run = train_wavepacket_run


__all__ = [
    "RUN_ARTIFACTS",
    "PHYSICAL_MODELS",
    "SPECTRAL_MODELS",
    "PairedTimeSampler",
    "PhysicalCollocationSampler",
    "SpectralCollocationSampler",
    "ProcessRSSMonitor",
    "paired_initialization_seed",
    "paired_time_seed",
    "paired_physical_x_seed",
    "physical_pde_loss",
    "spectral_pde_loss",
    "wavepacket_pde_loss",
    "train_wavepacket_run",
    "train_one_wavepacket_run",
]
