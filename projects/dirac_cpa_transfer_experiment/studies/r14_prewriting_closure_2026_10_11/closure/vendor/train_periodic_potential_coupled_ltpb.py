from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .physics import complex_derivative, dirac_residual, spinor_derivatives
from .periodic_potential_config import (
    PERIODIC_POTENTIAL_EXPERIMENT, PERIODIC_POTENTIAL_MODELS,
    PERIODIC_POTENTIAL_SEEDS, PeriodicPotentialConfig,
    paired_initialization_seed, paired_time_collocation_seed, paired_space_collocation_seed,
)
from .periodic_potential_models import (
    build_periodic_potential_model, count_trainable_parameters, trainable_state_sha256,
)
from .periodic_potential_physics import coupled_spectral_residual
from .train_wavepacket_shared_spectral_ltpb import ProcessRSSMonitor, _set_runtime, _environment_snapshot

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw" / PERIODIC_POTENTIAL_EXPERIMENT
RUN_ARTIFACTS = ("config.json", "metrics.json", "loss_history.csv", "time_series.csv", "dense_grid.npz", "model_state.pt")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def load_formal_config() -> PeriodicPotentialConfig:
    selection = RAW / "formal_selection.json"
    if not selection.is_file():
        raise RuntimeError("CALIBRATION_BLOCKED: no approved, passing formal physics selection exists")
    data = json.loads(selection.read_text(encoding="utf-8"))
    if data.get("status") != "SELECTED" or not data.get("all_selection_gates_passed"):
        raise RuntimeError("CALIBRATION_BLOCKED: the formal selection did not pass its registered gates")
    return replace(PeriodicPotentialConfig(), **data["selected"])


class PairedSampler:
    def __init__(self, cfg: PeriodicPotentialConfig, seed: int, *, physical: bool):
        self.cfg, self.physical = cfg, physical
        self.times_rng = torch.Generator().manual_seed(paired_time_collocation_seed(seed))
        self.space_rng = torch.Generator().manual_seed(paired_space_collocation_seed(seed))
        self.time_digest = hashlib.sha256()
        self.batch_index = 0

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        times = cfg.t_max * torch.rand((cfg.spectral_time_points, 1), generator=self.times_rng, dtype=torch.float64)
        self.time_digest.update(times.numpy().tobytes())
        if self.physical:
            n_x = cfg.physical_collocation_points // cfg.spectral_time_points
            if n_x * cfg.spectral_time_points != cfg.physical_collocation_points:
                raise ValueError("Physical samples must be an integer Cartesian product")
            coordinates = cfg.x_min + cfg.length * torch.rand((n_x, 1), generator=self.space_rng, dtype=torch.float64)
        else:
            coordinates = torch.tensor(cfg.wave_numbers, dtype=torch.float64).reshape(-1, 1)
        first = coordinates.repeat(len(times), 1).detach().requires_grad_(self.physical)
        time_flat = times.repeat_interleave(len(coordinates), dim=0).detach().requires_grad_(True)
        self.batch_index += 1
        return first, time_flat


def physical_pde_loss(model: torch.nn.Module, cfg: PeriodicPotentialConfig,
                      x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    psi = model(x, t)
    psi_x, psi_t = spinor_derivatives(psi, x, t, create_graph=True)
    residual = dirac_residual(psi, psi_x, psi_t, mass=cfg.mass)
    residual = residual - cfg.potential_strength * torch.cos(cfg.coupling_wave_number*x) * psi
    return torch.mean(torch.abs(residual)**2)


def spectral_pde_loss(model: torch.nn.Module, cfg: PeriodicPotentialConfig,
                      k: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    psi = model(k, t)
    psi_t = torch.stack([complex_derivative(psi[:, c], t, create_graph=True) for c in range(2)], dim=1)
    n_t = len(t) // cfg.mode_count
    if n_t * cfg.mode_count != len(t):
        raise ValueError("Residual requires full momentum ladder at every time")
    expected_k = torch.tensor(cfg.wave_numbers, dtype=torch.float64, device=k.device)
    if not torch.equal(k.reshape(n_t, cfg.mode_count), expected_k.expand(n_t, -1)):
        raise ValueError("Momentum ladder order is inconsistent")
    times = t.reshape(n_t, cfg.mode_count)
    if not torch.equal(times.detach(), times[:, :1].detach().expand_as(times)):
        raise ValueError("Coupled neighbors must share the same time")
    residual = coupled_spectral_residual(
        psi.reshape(n_t, cfg.mode_count, 2), psi_t.reshape(n_t, cfg.mode_count, 2),
        expected_k, cfg.mass, cfg.potential_strength,
    )
    return torch.sum(torch.abs(residual)**2, dim=1).mean()/cfg.length


def pde_loss(model_name: str, model: torch.nn.Module, cfg: PeriodicPotentialConfig,
             first: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if model_name == "physical_vanilla":
        return physical_pde_loss(model, cfg, first, t)
    if model_name in PERIODIC_POTENTIAL_MODELS:
        return spectral_pde_loss(model, cfg, first, t)
    raise ValueError(model_name)


def train_run(cfg: PeriodicPotentialConfig, model_name: str, seed: int,
              run_dir: Path, *, formal: bool) -> dict:
    if model_name not in PERIODIC_POTENTIAL_MODELS or seed not in PERIODIC_POTENTIAL_SEEDS:
        raise ValueError("Model or seed outside registered inventory")
    freeze_hash = None
    if formal:
        from .freeze_periodic_potential_coupled_ltpb import verify_freeze
        freeze_hash = verify_freeze()
        if cfg != load_formal_config():
            raise ValueError("Formal configuration differs from the frozen selection")
    if run_dir.exists():
        if {p.name for p in run_dir.iterdir()} == set(RUN_ARTIFACTS):
            cached = json.loads((run_dir/"metrics.json").read_text(encoding="utf-8"))
            if cached["steps"] == cfg.adam_steps and cached["model"] == model_name and cached["seed"] == seed:
                print(f"SKIP complete {model_name} seed={seed}", flush=True)
                return cached
        raise FileExistsError(f"Refusing to overwrite existing run {run_dir}")
    existing_staging = list(run_dir.parent.glob(f".{run_dir.name}.staging-*"))
    if existing_staging:
        raise RuntimeError(f"Prior interrupted attempt preserved; cannot selectively restart: {existing_staging}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = run_dir.parent / f".{run_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    device = _set_runtime(seed, cfg.cpu_threads)
    model = build_periodic_potential_model(model_name, cfg, paired_initialization_seed(seed))
    params = count_trainable_parameters(model)
    initial_hash = trainable_state_sha256(model)
    streams = {"initialization_seed": paired_initialization_seed(seed),
               "time_seed": paired_time_collocation_seed(seed),
               "space_seed": paired_space_collocation_seed(seed) if model_name == "physical_vanilla" else None}
    config = {"experiment": PERIODIC_POTENTIAL_EXPERIMENT, "model": model_name,
              "seed": seed, "formal": formal, "configuration": cfg.as_dict(),
              "initial_trainable_sha256": initial_hash, "random_streams": streams,
              "parameter_count": params, "loss_terms": ["pde"],
              "freeze_manifest_sha256": freeze_hash, "environment": _environment_snapshot(device)}
    write_json(staging/"config.json", config)
    sampler = PairedSampler(cfg, seed, physical=model_name == "physical_vanilla")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    monitor = ProcessRSSMonitor()
    history = []
    monitor.start()
    started = time.perf_counter()
    print(f"START {model_name} seed={seed} steps={cfg.adam_steps} params={params}", flush=True)
    try:
        for step in range(1, cfg.adam_steps+1):
            optimizer.zero_grad(set_to_none=True)
            first, t = sampler.sample()
            loss = pde_loss(model_name, model, cfg, first, t)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"Nonfinite or missing parameter gradient at step {step}")
            if step == 1 or step % cfg.history_every == 0 or step == cfg.adam_steps:
                gradient = model.mlp.net[-1].weight.grad
                history.append({"step": step, "pde": float(loss.detach()), "total": float(loss.detach()),
                                "elapsed_seconds": time.perf_counter()-started,
                                "output_first_two_gradient_norm": float(torch.linalg.vector_norm(gradient[:2])),
                                "output_last_two_gradient_norm": float(torch.linalg.vector_norm(gradient[2:])),
                                "collocation_batch_index": sampler.batch_index})
            optimizer.step()
            if step == 1 or step % cfg.log_every == 0 or step == cfg.adam_steps:
                print(f"{model_name} seed={seed} step={step}/{cfg.adam_steps} loss={float(loss.detach()):.6g}", flush=True)
        training_seconds = time.perf_counter()-started
        memory = monitor.snapshot()
        checkpoint = {"model": model_name, "seed": seed, "steps": cfg.adam_steps,
                      "configuration": cfg.as_dict(), "state_dict": model.state_dict(),
                      "random_streams": streams, "time_stream_sha256": sampler.time_digest.hexdigest(),
                      "initial_trainable_sha256": initial_hash}
        torch.save(checkpoint, staging/"model_state.pt")
        pd.DataFrame(history).to_csv(staging/"loss_history.csv", index=False)
        # Reference code is called only here, after optimization and checkpoint serialization.
        from .periodic_potential_evaluate import compute_periodic_potential_evaluation
        metrics, series, dense = compute_periodic_potential_evaluation(
            model, cfg, model_name=model_name, seed=seed, parameter_count=params,
            training_seconds=training_seconds, final_pde_loss=float(loss.detach()),
            peak_rss_mb=float(memory["rss_sampled_peak_bytes"])/2**20,
            peak_rss_increment_mb=max(0.0, float(memory["rss_sampled_peak_bytes"])-monitor.initial_bytes)/2**20,
        )
        metrics.update({"formal": formal, "time_stream_sha256": sampler.time_digest.hexdigest(),
                        "initial_trainable_sha256": initial_hash, "loss_terms_used": ["pde"],
                        "rss_measurement_method": memory["rss_measurement_method"]})
        write_json(staging/"metrics.json", metrics)
        series.to_csv(staging/"time_series.csv", index=False)
        np.savez_compressed(staging/"dense_grid.npz", **dense)
        if {p.name for p in staging.iterdir()} != set(RUN_ARTIFACTS):
            raise AssertionError("Run artifact inventory is incomplete")
        staging.rename(run_dir)
        print(f"DONE {model_name} seed={seed} training_seconds={training_seconds:.1f}", flush=True)
        return metrics
    except Exception as exc:
        # Keep all partial evidence; never silently rerun a failed formal attempt.
        write_json(staging/"failure.json", {"error_type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        monitor.stop()
