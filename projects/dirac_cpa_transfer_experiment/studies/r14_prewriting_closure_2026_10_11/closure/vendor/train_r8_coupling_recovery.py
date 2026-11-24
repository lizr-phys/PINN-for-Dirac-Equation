from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .r8_config import (allocation, EXPERIMENT, config_for, RAW, CANDIDATES, CONTROLS,
                        EXPLORATION_SEEDS, VALIDATION_SEEDS, EXTENSION_SEEDS)
from .r8_models import R8Model, parameter_counts
from .r8_sampling import TimeSampler
from .r8_loss import training_loss, free_carrier_residual, ResidualBalance
from .freeze_r7_phase_release import write_json
from .train_r7_phase_release import atomic_torch_save, event
from .periodic_potential_models import trainable_state_sha256
from .train_wavepacket_shared_spectral_ltpb import ProcessRSSMonitor, _set_runtime, _environment_snapshot

ARTIFACTS = ("config.json", "metrics.json", "loss_history.csv", "time_series.csv", "dense_grid.npz", "model_state.pt")


def train_run(cfg, name, seed, run_dir, *, stage, balanced=False, stop_after=None):
    run_dir = Path(run_dir)
    freeze_hash = None
    if stage != "smoke":
        from .freeze_r8_coupling_recovery import verify_freeze
        freeze_hash = verify_freeze()
        if cfg != config_for(stage, cfg.case, cfg.ladder_half_width):
            raise ValueError("Run configuration differs from frozen stage")
        if stage == "exploration":
            allowed, seeds = CANDIDATES+CONTROLS, EXPLORATION_SEEDS
        elif stage == "validation":
            selection = json.loads((RAW/"exploration_selection.json").read_text(encoding="utf-8"))
            allowed, seeds = tuple(selection["selected_candidates"])+CONTROLS, VALIDATION_SEEDS
        elif stage in ("ablation", "extension", "rescue"):
            plan = json.loads((RAW/"conditional_execution_plan.json").read_text(encoding="utf-8"))
            allowed = plan["ablations"] if stage == "ablation" else (plan["extension_models"] if stage == "extension" else [plan["rescue_model"]])
            seeds = VALIDATION_SEEDS if stage == "ablation" else (EXTENSION_SEEDS if stage == "extension" else EXPLORATION_SEEDS)
            if stage == "extension" and cfg.case not in plan["extension_cases"]:
                raise ValueError("Unregistered extension case")
        else:
            raise ValueError("Unregistered training stage")
        if name not in allowed or seed not in seeds or balanced != (stage == "rescue"):
            raise ValueError("Run is outside frozen execution branch")
    if (run_dir/"metrics.json").exists():
        if not all((run_dir/f).is_file() for f in ARTIFACTS):
            raise RuntimeError("Completed run has incomplete evidence")
        r = json.loads((run_dir/"metrics.json").read_text(encoding="utf-8"))
        if r["steps"] != cfg.adam_steps or r["model"] != name or r["balanced"] != balanced:
            raise RuntimeError("Cached completed run differs from requested protocol")
        print(f"SKIP {stage}/{name}/seed{seed}", flush=True)
        return r
    run_dir.mkdir(parents=True, exist_ok=True)
    device = _set_runtime(seed, 1)
    model = R8Model(cfg, name, seed)
    initial_hash = trainable_state_sha256(model.fields[0])
    carrier_initial_hash = None if model.carrier is None else trainable_state_sha256(model.carrier)
    pre, n_windows, per_window = allocation(cfg, name)
    config = dict(experiment=EXPERIMENT, model=name, seed=seed, stage=stage, balanced=balanced,
        configuration=cfg.as_dict(), total_budget=cfg.adam_steps, carrier_steps=pre,
        coupled_steps=cfg.adam_steps-pre, window_count=n_windows, coupled_steps_per_window=per_window,
        initial_envelope_sha256=initial_hash, carrier_initial_sha256=carrier_initial_hash,
        random_streams=dict(envelope_init=800000+seed, carrier_init=900000+seed,
                            free_times=910000+seed, coupled_times=920000+seed,
                            window_init_increment=10000),
        loss_terms=["full_coupled_pde"], freeze_manifest_sha256=freeze_hash,
        environment=_environment_snapshot(device), cpu_affinity_mask=os.environ.get("R7_CPU_AFFINITY_MASK"))
    if (run_dir/"config.json").exists():
        previous = json.loads((run_dir/"config.json").read_text(encoding="utf-8"))
        for key in ("model", "seed", "stage", "balanced", "configuration", "freeze_manifest_sha256"):
            if previous[key] != json.loads(json.dumps(config))[key]:
                raise RuntimeError(f"Cannot resume a changed run: {key}")
    else:
        write_json(run_dir/"config.json", config)
    free_sampler, coupled_sampler = TimeSampler(cfg, seed, carrier=True), TimeSampler(cfg, seed)
    phase = "carrier" if pre else "coupled"
    optimizer = torch.optim.Adam(model.carrier.parameters() if pre else model.active_field.parameters(), lr=cfg.learning_rate)
    balance = ResidualBalance(cfg.mode_count) if balanced else None
    seconds, pre_seconds, history, start, peak, initial_rss = 0., 0., [], 1, 0, 0
    recovery = run_dir/"recovery_state.pt"
    if recovery.exists():
        saved = torch.load(recovery, weights_only=False)
        for _ in range(saved["window_count"]-1):
            model.add_window()
        model.load_state_dict(saved["state_dict"])
        phase = saved["phase"]
        if phase == "coupled" and model.carrier is not None:
            model.carrier.freeze()
        optimizer = torch.optim.Adam(model.carrier.parameters() if phase == "carrier" else model.active_field.parameters(), lr=cfg.learning_rate)
        optimizer.load_state_dict(saved["optimizer"])
        free_sampler.load_state_dict(saved["free_sampler"])
        coupled_sampler.load_state_dict(saved["coupled_sampler"])
        if balance is not None:
            balance.ema = saved["balance_ema"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        start, seconds, pre_seconds, history = saved["step"]+1, saved["seconds"], saved["pre_seconds"], saved["history"]
        peak, initial_rss = saved["peak"], saved["initial_rss"]
        event(run_dir, "exact_resume", step=saved["step"])
    else:
        event(run_dir, "start", budget=cfg.adam_steps)
    monitor = ProcessRSSMonitor()
    monitor.start()
    initial_rss = initial_rss or monitor.initial_bytes
    session_started = time.perf_counter()

    def save(step):
        nonlocal peak
        peak = max(peak, int(monitor.snapshot()["rss_sampled_peak_bytes"]))
        atomic_torch_save(recovery, dict(step=step, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
            window_count=len(model.fields), phase=phase, configuration=cfg.as_dict(), model=name, seed=seed,
            free_sampler=free_sampler.state_dict(), coupled_sampler=coupled_sampler.state_dict(),
            balance_ema=None if balance is None else balance.ema, seconds=seconds, pre_seconds=pre_seconds,
            history=history, peak=peak, initial_rss=initial_rss, python_rng=random.getstate(),
            numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state()))
        pd.DataFrame(history).to_csv(run_dir/"loss_history.csv", index=False)

    try:
        for step in range(start, cfg.adam_steps+1):
            tick = time.perf_counter()
            if step > pre and phase == "carrier":
                model.carrier.freeze()
                phase = "coupled"
                optimizer = torch.optim.Adam(model.active_field.parameters(), lr=cfg.learning_rate)
                atomic_torch_save(run_dir/"pretrained_carrier_state.pt", dict(state_dict=model.carrier.state_dict(),
                    configuration=cfg.as_dict(), seed=seed, steps=pre, training_seconds=pre_seconds,
                    initial_sha256=carrier_initial_hash, normalized_time_stream_sha256=free_sampler.digest(True)))
                event(run_dir, "carrier_frozen_time_derivative_retained", step=pre)
            optimizer.zero_grad(set_to_none=True)
            if step <= pre:
                k, t = free_sampler.sample()
                residual = free_carrier_residual(model.carrier, cfg, k, t)
                objective = residual.abs().square().mean()
                power = residual.detach().abs().square().reshape(32, cfg.mode_count, 2, 2).mean((0, 2, 3))
                weights = torch.ones(cfg.mode_count, dtype=torch.float64)
                active_parameters = tuple(model.carrier.parameters())
                head = model.carrier.mlp.net[-1]
                window = -1
            else:
                coupled_index = step-pre-1
                window = coupled_index//per_window
                while len(model.fields) < window+1:
                    model.add_window()
                    optimizer = torch.optim.Adam(model.active_field.parameters(), lr=cfg.learning_rate)
                    event(run_dir, "hard_predicted_complex_handoff", window=window+1, step=step)
                field = model.active_field
                k, t = coupled_sampler.sample(field.start, field.duration)
                # Evaluate the active field directly; completed windows are not retrained.
                objective, power, weights = training_loss(lambda k, t: field(k, t, model.carrier), cfg, k, t, balance)
                active_parameters, head = tuple(field.parameters()), field.mlp.net[-1]
            if not torch.isfinite(objective):
                raise FloatingPointError(f"Nonfinite objective at {step}")
            objective.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in active_parameters):
                raise FloatingPointError(f"Nonfinite/missing gradient at {step}")
            grads = head.weight.grad
            gradient_r = float(torch.linalg.vector_norm(grads[:2]))
            gradient_b = float(torch.linalg.vector_norm(grads[2:])) if grads.shape[0] == 4 else 0.
            optimizer.step()
            elapsed = time.perf_counter()-tick
            seconds += elapsed
            if step <= pre:
                pre_seconds += elapsed
            if step == 1 or step % cfg.history_every == 0 or step in (pre, cfg.adam_steps) or (step-pre-1) % per_window == 0:
                row = dict(step=step, phase=phase, window=window+1, objective=float(objective.detach()),
                    unweighted_pde=float(power.detach().sum()) if step>pre else float(power.detach().mean()),
                    elapsed_training_seconds=seconds, output_r_gradient_norm=gradient_r,
                    output_b_gradient_norm=gradient_b)
                for j in range(cfg.mode_count):
                    row[f"mode_{j-cfg.ladder_half_width:+d}_unweighted_power"] = float(power[j].detach())
                    row[f"mode_{j-cfg.ladder_half_width:+d}_weight"] = float(weights[j])
                history.append(row)
            if step == 1 or step % cfg.log_every == 0 or step == cfg.adam_steps:
                print(f"{stage}/{name} seed={seed} step={step}/{cfg.adam_steps} phase={phase} window={window+1} loss={float(objective.detach()):.6g}", flush=True)
            if step % cfg.recovery_every == 0 or step in (pre, cfg.adam_steps) or step == stop_after:
                save(step)
            if step == stop_after:
                event(run_dir, "controlled_test_interruption", step=step)
                return dict(interrupted_at=step)
        if model.carrier is not None:
            model.carrier.freeze()
        save(cfg.adam_steps)
        snapshot = monitor.snapshot()
        peak = max(peak, int(snapshot["rss_sampled_peak_bytes"]))
        monitor.stop()
        atomic_torch_save(run_dir/"model_state.pt", dict(state_dict=model.state_dict(), configuration=cfg.as_dict(),
            model=name, seed=seed, window_count=len(model.fields), steps=cfg.adam_steps,
            carrier_steps=pre, frozen_carrier=True, final_checkpoint_only=True,
            initial_envelope_sha256=initial_hash, carrier_initial_sha256=carrier_initial_hash,
            coupled_time_stream_sha256=coupled_sampler.digest(), free_time_stream_sha256=free_sampler.digest(),
            normalized_coupled_time_stream_sha256=coupled_sampler.digest(True)))
        from .r8_evaluate import evaluate_run
        last_loss = history[-1]["unweighted_pde"]
        eval_start = time.perf_counter()
        metrics, series, dense, tables = evaluate_run(model, cfg, seed=seed, seconds=seconds,
            final_loss=last_loss, rss=peak/2**20, rss_increment=max(peak-initial_rss, 0)/2**20, full=stage != "smoke")
        metrics.update(stage=stage, balanced=balanced, carrier_steps=pre, coupled_steps=cfg.adam_steps-pre,
            carrier_training_seconds=pre_seconds, coupled_training_seconds=seconds-pre_seconds,
            initial_envelope_sha256=initial_hash, carrier_initial_sha256=carrier_initial_hash,
            coupled_time_stream_sha256=coupled_sampler.digest(), free_time_stream_sha256=free_sampler.digest(),
            normalized_coupled_time_stream_sha256=coupled_sampler.digest(True),
            normalized_free_time_stream_sha256=free_sampler.digest(True),
            residual_training_batches=cfg.adam_steps,
            residual_training_spinor_points=(2*pre+cfg.adam_steps-pre)*cfg.spectral_collocation_points,
            residual_training_complex_components=2*(2*pre+cfg.adam_steps-pre)*cfg.spectral_collocation_points,
            fixed_final_step=cfg.adam_steps, evaluation_seconds=time.perf_counter()-eval_start,
            evaluation_coupled_spinor_points=(cfg.test_t_points+(cfg.residual_time_points if stage != "smoke" else 0)+1)*cfg.mode_count,
            run_elapsed_seconds_this_session=time.perf_counter()-session_started,
            rss_measurement_method=snapshot["rss_measurement_method"])
        series.to_csv(run_dir/"time_series.csv", index=False)
        np.savez_compressed(run_dir/"dense_grid.npz", **dense)
        for file, table in tables.items():
            table.to_csv(run_dir/file, index=False)
        write_json(run_dir/"metrics.json", metrics)
        event(run_dir, "complete", budget=cfg.adam_steps, time_sha256=coupled_sampler.digest())
        print(f"DONE {stage}/{name} seed={seed} seconds={seconds:.1f}", flush=True)
        return metrics
    except Exception as error:
        event(run_dir, "failure_preserved", error_type=type(error).__name__, message=str(error))
        raise
    finally:
        monitor.stop()
