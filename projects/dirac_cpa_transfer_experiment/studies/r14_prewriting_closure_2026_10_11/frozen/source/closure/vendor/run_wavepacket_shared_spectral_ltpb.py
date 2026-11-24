from __future__ import annotations

import argparse
import json
import shutil
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .wavepacket_config import (
    WAVEPACKET_EXPERIMENT,
    WAVEPACKET_MODELS,
    WAVEPACKET_SEEDS,
    WavepacketConfig,
    WavepacketRunSpec,
)
from .wavepacket_physics import (
    exact_spectral_state,
    reference_energies,
    training_wave_numbers,
)
from .train_wavepacket_shared_spectral_ltpb import train_wavepacket_run


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = (
    ROOT / "results" / "raw" / "wavepacket_shared_spectral_ltpb"
)
SMOKE_ROOT = (
    ROOT / "results" / "raw" / "wavepacket_shared_spectral_ltpb_smoke"
)


class SpectralOracle(torch.nn.Module):
    """Exact-dispersion upper bound, deliberately isolated from learned models."""

    model_name = "spectral_oracle"
    output_dim = 0

    def __init__(self, cfg: WavepacketConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("device_anchor", torch.zeros((), dtype=torch.float64))

    def forward(
        self, wave_numbers: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return exact_spectral_state(wave_numbers, t, self.cfg)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_oracle(cfg: WavepacketConfig, oracle_dir: Path) -> None:
    """Write an exact spectral reference with no optimizer or learned ranking role."""

    required = {"config.json", "metrics.json", "time_series.csv", "dense_grid.npz"}
    if oracle_dir.exists():
        names = (
            {path.name for path in oracle_dir.iterdir()}
            if oracle_dir.is_dir()
            else set()
        )
        if names == required:
            print(f"SKIP completed oracle {oracle_dir}", flush=True)
            return
        raise RuntimeError(
            f"Refusing to overwrite incomplete oracle directory {oracle_dir}: "
            f"{sorted(names)}"
        )
    oracle_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = oracle_dir.parent / f".{oracle_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    committed = False
    try:
        torch.set_default_dtype(torch.float64)
        wave_numbers = training_wave_numbers(cfg)
        rates = reference_energies(wave_numbers, cfg.mass)
        config = {
            "experiment": WAVEPACKET_EXPERIMENT,
            "model": "spectral_oracle",
            "role": "exact_spectral_upper_bound_excluded_from_learned_ranking",
            "uses_exact_dispersion": True,
            "wave_numbers": list(cfg.wave_numbers),
            "reference_energies": rates,
            "test_grid": {
                "x_points": cfg.test_x_points,
                "t_points": cfg.test_t_points,
            },
        }
        _write_json(staging / "config.json", config)

        from .wavepacket_evaluate import compute_wavepacket_evaluation

        oracle = SpectralOracle(cfg)
        metrics, time_series, dense = compute_wavepacket_evaluation(
            oracle,
            cfg,
            model_name="spectral_oracle",
            seed=-1,
            parameter_count=0,
            training_seconds=0.0,
            final_pde_loss=0.0,
            peak_rss_mb=0.0,
            peak_rss_increment_mb=0.0,
        )
        metrics.update(
            {
                "experiment": WAVEPACKET_EXPERIMENT,
                "model": "spectral_oracle",
                "seed": -1,
                "excluded_from_learned_ranking": True,
                "uses_exact_dispersion": True,
            }
        )
        _write_json(staging / "metrics.json", dict(metrics))
        time_series.to_csv(staging / "time_series.csv", index=False)
        np.savez_compressed(staging / "dense_grid.npz", **dense)
        names = {path.name for path in staging.iterdir()}
        if names != required:
            raise RuntimeError(f"Oracle artifact contract mismatch: {sorted(names)}")
        for path in staging.iterdir():
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty oracle artifact: {path}")
        staging.rename(oracle_dir)
        committed = True
        print(f"DONE spectral oracle -> {oracle_dir}", flush=True)
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


def run(
    mode: str,
    *,
    only_models: tuple[str, ...] | None = None,
    only_seeds: tuple[int, ...] | None = None,
    smoke_steps: int = 20,
    cpu_threads: int = 1,
    oracle_only: bool = False,
    no_oracle: bool = False,
    no_postprocess: bool = False,
) -> None:
    models = only_models or WAVEPACKET_MODELS
    seeds = only_seeds or WAVEPACKET_SEEDS
    unknown = sorted(set(models).difference(WAVEPACKET_MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    if any(seed not in WAVEPACKET_SEEDS for seed in seeds):
        raise ValueError(f"Seed choices must be drawn from {WAVEPACKET_SEEDS}")

    if mode == "smoke":
        cfg = replace(
            WavepacketConfig(),
            adam_steps=smoke_steps,
            test_x_points=32,
            test_t_points=21,
            history_every=max(1, min(5, smoke_steps)),
            log_every=max(1, smoke_steps),
            cpu_threads=cpu_threads,
        )
        learned_root = SMOKE_ROOT / f"steps_{smoke_steps}" / "learned"
        experiment = "wavepacket_shared_spectral_ltpb_smoke"
    elif mode == "full":
        cfg = replace(WavepacketConfig(), cpu_threads=cpu_threads)
        learned_root = FORMAL_ROOT / "learned"
        experiment = WAVEPACKET_EXPERIMENT
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if oracle_only:
        if mode != "full":
            raise ValueError("--oracle-only is available only in full mode")
        write_oracle(cfg, FORMAL_ROOT / "oracle")
        return

    for seed in seeds:
        for model in models:
            spec = WavepacketRunSpec(
                model=model,
                seed=seed,
                steps=cfg.adam_steps,
                experiment=experiment,
            )
            train_wavepacket_run(
                cfg, spec, learned_root / model / f"seed_{seed}"
            )

    if mode == "full" and not no_oracle:
        write_oracle(cfg, FORMAL_ROOT / "oracle")

    if mode == "full" and not no_postprocess:
        expected = [
            learned_root / model / f"seed_{seed}" / "metrics.json"
            for model in WAVEPACKET_MODELS
            for seed in WAVEPACKET_SEEDS
        ]
        if all(path.is_file() for path in expected):
            from .aggregate_wavepacket_shared_spectral_ltpb import (
                write_wavepacket_outputs,
            )
            from .plot_wavepacket_shared_spectral_ltpb import generate_all

            write_wavepacket_outputs()
            generate_all()
        else:
            print(
                "Postprocessing deferred: the 25-run formal inventory is incomplete.",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered shared-spectral wavepacket experiment"
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--only-model", "--model", action="append", choices=WAVEPACKET_MODELS
    )
    parser.add_argument("--only-seed", "--seed", action="append", type=int)
    parser.add_argument("--smoke-steps", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--no-postprocess", action="store_true")
    args = parser.parse_args()
    run(
        args.mode,
        only_models=tuple(args.only_model) if args.only_model else None,
        only_seeds=tuple(args.only_seed) if args.only_seed else None,
        smoke_steps=args.smoke_steps,
        cpu_threads=args.cpu_threads,
        oracle_only=args.oracle_only,
        no_oracle=args.no_oracle,
        no_postprocess=args.no_postprocess,
    )


if __name__ == "__main__":
    main()


__all__ = ["FORMAL_ROOT", "SMOKE_ROOT", "SpectralOracle", "write_oracle", "run"]
