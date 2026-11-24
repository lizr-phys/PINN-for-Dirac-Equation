from __future__ import annotations

import argparse
import json
import math
import shutil
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .multimode_config import (
    MULTIMODE_EXPERIMENT,
    MULTIMODE_MODELS,
    MULTIMODE_SEEDS,
    MultimodeConfig,
    MultimodeRunSpec,
)
from .multimode_models import FourCarrierOracle
from .multimode_physics import reference_energies
from .train_multimode_ltpb import train_multimode_run


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "results" / "raw" / "multimode_ltpb"
SMOKE_ROOT = ROOT / "results" / "raw" / "multimode_ltpb_smoke"


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_oracle(cfg: MultimodeConfig, oracle_dir: Path) -> None:
    """Write a deterministic exact-carrier diagnostic, isolated from ranking."""

    required = {"config.json", "metrics.json", "time_series.csv", "dense_grid.npz"}
    if oracle_dir.exists():
        names = {path.name for path in oracle_dir.iterdir()} if oracle_dir.is_dir() else set()
        if names == required:
            print(f"SKIP completed oracle {oracle_dir}", flush=True)
            return
        raise RuntimeError(f"Refusing to overwrite incomplete oracle directory {oracle_dir}: {sorted(names)}")
    oracle_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = oracle_dir.parent / f".{oracle_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    committed = False
    try:
        # Reference energies are intentionally created only inside this explicit
        # oracle/evaluation path, never inside a learned model or optimizer.
        energies = reference_energies(cfg.mass, cfg.wave_numbers)
        config = {
            "experiment": MULTIMODE_EXPERIMENT,
            "model": "oracle",
            "role": "exact_four_carrier_diagnostic_excluded_from_learned_ranking",
            "uses_exact_time_carriers": True,
            "wave_numbers": list(cfg.wave_numbers),
            "reference_energies": energies.tolist(),
            "test_grid": {"x_points": cfg.test_x_points, "t_points": cfg.test_t_points},
        }
        (staging / "config.json").write_text(
            json.dumps(config, indent=2, default=_json_default) + "\n", encoding="utf-8"
        )
        from .multimode_evaluate import compute_multimode_evaluation

        oracle = FourCarrierOracle(cfg).to(dtype=torch.float64)
        metrics, time_series, dense = compute_multimode_evaluation(
            oracle,
            cfg,
            model_name="oracle",
            seed=-1,
            parameter_count=0,
            training_seconds=0.0,
            final_pde_loss=0.0,
        )
        metrics.update(
            {
                "experiment": MULTIMODE_EXPERIMENT,
                "model": "oracle",
                "seed": -1,
                "excluded_from_learned_ranking": True,
                "uses_exact_time_carriers": True,
            }
        )
        (staging / "metrics.json").write_text(
            json.dumps(metrics, indent=2, default=_json_default, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        time_series.to_csv(staging / "time_series.csv", index=False)
        np.savez_compressed(staging / "dense_grid.npz", **dense)
        names = {path.name for path in staging.iterdir()}
        if names != required:
            raise RuntimeError(f"Oracle artifact contract mismatch: {sorted(names)}")
        staging.rename(oracle_dir)
        committed = True
        print(f"DONE exact-carrier oracle -> {oracle_dir}", flush=True)
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
    no_oracle: bool = False,
    no_postprocess: bool = False,
) -> None:
    models = only_models or MULTIMODE_MODELS
    seeds = only_seeds or MULTIMODE_SEEDS
    unknown = sorted(set(models).difference(MULTIMODE_MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    if any(seed not in MULTIMODE_SEEDS for seed in seeds):
        raise ValueError(f"Formal seed choices must be drawn from {MULTIMODE_SEEDS}")
    if mode == "smoke":
        cfg = replace(
            MultimodeConfig(),
            collocation_points=64,
            adam_steps=smoke_steps,
            test_x_points=16,
            test_t_points=11,
            history_every=max(1, min(5, smoke_steps)),
            log_every=max(1, smoke_steps),
            cpu_threads=cpu_threads,
        )
        root = SMOKE_ROOT / f"steps_{smoke_steps}"
        experiment = "multimode_ltpb_smoke"
    elif mode == "full":
        cfg = replace(MultimodeConfig(), cpu_threads=cpu_threads)
        root = FORMAL_ROOT / "learned"
        experiment = MULTIMODE_EXPERIMENT
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for seed in seeds:
        for model in models:
            spec = MultimodeRunSpec(
                model=model,
                seed=seed,
                steps=cfg.adam_steps,
                experiment=experiment,
            )
            train_multimode_run(cfg, spec, root / model / f"seed_{seed}")

    if mode == "full" and not no_oracle:
        write_oracle(cfg, FORMAL_ROOT / "oracle")

    if mode == "full" and not no_postprocess:
        expected = [
            root / model / f"seed_{seed}" / "metrics.json"
            for model in MULTIMODE_MODELS
            for seed in MULTIMODE_SEEDS
        ]
        if all(path.is_file() for path in expected):
            from .aggregate_multimode_ltpb import write_outputs
            from .plot_multimode_ltpb import generate_all

            write_outputs()
            generate_all()
        else:
            print("Postprocessing deferred: the 25-run formal inventory is incomplete.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered Multi-LTPB experiment")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--only-model", action="append", choices=MULTIMODE_MODELS)
    parser.add_argument("--only-seed", action="append", type=int)
    parser.add_argument("--smoke-steps", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--no-postprocess", action="store_true")
    args = parser.parse_args()
    run(
        args.mode,
        only_models=tuple(args.only_model) if args.only_model else None,
        only_seeds=tuple(args.only_seed) if args.only_seed else None,
        smoke_steps=args.smoke_steps,
        cpu_threads=args.cpu_threads,
        no_oracle=args.no_oracle,
        no_postprocess=args.no_postprocess,
    )


if __name__ == "__main__":
    main()
