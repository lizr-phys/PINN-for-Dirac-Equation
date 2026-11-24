from __future__ import annotations
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .r9_config import EXPERIMENT, ARTIFACTS, MODELS, SEEDS
from .r9_models import R9Model, parameter_counts, tensor_digest
from .r9_loss import training_loss, free_carrier_residual
from .r8_sampling import TimeSampler
from .freeze_r7_phase_release import write_json
from .train_r7_phase_release import atomic_torch_save, event
from .train_wavepacket_shared_spectral_ltpb import ProcessRSSMonitor, _set_runtime, _environment_snapshot


def train_run(cfg, name, seed, run_dir, *, smoke=False, stop_after=None):
    run_dir = Path(run_dir)
    freeze_hash = None
    if not smoke:
        from .freeze_r9_wavepacket import verify_freeze, selected_config
        freeze_hash = verify_freeze()
        if cfg != selected_config() or name not in MODELS or seed not in SEEDS:
            raise ValueError("Run differs from frozen R9 protocol")
    if (run_dir/"metrics.json").exists():
        if not all((run_dir/p).is_file() for p in ARTIFACTS+("recovery_state.pt",)):
            raise RuntimeError("Incomplete cached run")
        saved = json.loads((run_dir/"metrics.json").read_text(encoding="utf-8"))
        if saved["steps"] != cfg.adam_steps or saved["model"] != name or saved["seed"] != seed:
            raise RuntimeError("Different cached run")
        return saved
    run_dir.mkdir(parents=True, exist_ok=True)
    device = _set_runtime(seed, 1)
    model = R9Model(cfg, name, seed)
    network_hashes = [tensor_digest(n.parameters()) for n in model.networks]
    carrier_hash = None if model.carrier is None else tensor_digest(model.carrier.parameters())
    pre = cfg.carrier_steps if model.carrier is not None else 0
    configuration = dict(experiment=EXPERIMENT, model=name, seed=seed, smoke=smoke, configuration=cfg.as_dict(),
        total_budget=cfg.adam_steps, carrier_steps=pre, coupled_steps=cfg.adam_steps-pre,
        initial_complex_network_sha256=network_hashes, initial_carrier_sha256=carrier_hash,
        random_streams=dict(complex_network_1_init=800000+seed, complex_network_2_init=830000+seed,
                            carrier_init=900000+seed, free_time=910000+seed, coupled_time=920000+seed),
        loss_terms=["original_full_coupled_spinor_pde"], freeze_manifest_sha256=freeze_hash,
        cpu_affinity_mask=os.environ.get("R7_CPU_AFFINITY_MASK"), environment=_environment_snapshot(device))
    if (run_dir/"config.json").exists():
        old = json.loads((run_dir/"config.json").read_text(encoding="utf-8"))
        for key in ("model", "seed", "smoke", "configuration", "freeze_manifest_sha256", "cpu_affinity_mask"):
            if old[key] != json.loads(json.dumps(configuration))[key]:
                raise RuntimeError(f"Changed recovery configuration: {key}")
    else:
        write_json(run_dir/"config.json", configuration)
    free_sampler, coupled_sampler = TimeSampler(cfg, seed, carrier=True), TimeSampler(cfg, seed)
    phase = "carrier" if pre else "coupled"
    parameters = model.carrier.parameters() if pre else model.networks.parameters()
    optimizer = torch.optim.Adam(parameters, lr=cfg.learning_rate)
    seconds, pre_seconds, history, start, peak = 0., 0., [], 1, 0
    recovery = run_dir/"recovery_state.pt"
    if recovery.exists():
        saved = torch.load(recovery, weights_only=False)
        model.load_state_dict(saved["state_dict"])
        phase = saved["phase"]
        if phase == "coupled" and model.carrier is not None:
            model.carrier.freeze()
        optimizer = torch.optim.Adam(model.carrier.parameters() if phase == "carrier" else model.networks.parameters(),
                                     lr=cfg.learning_rate)
        optimizer.load_state_dict(saved["optimizer"])
        free_sampler.load_state_dict(saved["free_sampler"])
        coupled_sampler.load_state_dict(saved["coupled_sampler"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        start, seconds, pre_seconds, history, peak = (saved[k] for k in ("step", "seconds", "pre_seconds", "history", "peak"))
        start += 1
        event(run_dir, "exact_resume", step=start-1)
    else:
        event(run_dir, "start", budget=cfg.adam_steps)
    monitor = ProcessRSSMonitor()
    monitor.start()
    session_start = time.perf_counter()

    def save(step):
        nonlocal peak
        peak = max(peak, int(monitor.snapshot()["rss_sampled_peak_bytes"]))
        atomic_torch_save(recovery, dict(step=step, phase=phase, state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(), configuration=cfg.as_dict(), model=name, seed=seed,
            free_sampler=free_sampler.state_dict(), coupled_sampler=coupled_sampler.state_dict(),
            seconds=seconds, pre_seconds=pre_seconds, history=history, peak=peak,
            python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state()))
        pd.DataFrame(history).to_csv(run_dir/"loss_history.csv", index=False)

    try:
        for step in range(start, cfg.adam_steps+1):
            tick = time.perf_counter()
            if step > pre and phase == "carrier":
                model.carrier.freeze()
                phase = "coupled"
                optimizer = torch.optim.Adam(model.networks.parameters(), lr=cfg.learning_rate)
                atomic_torch_save(run_dir/"pretrained_carrier_state.pt", dict(state_dict=model.carrier.state_dict(),
                    configuration=cfg.as_dict(), steps=pre, seed=seed, training_seconds=pre_seconds,
                    time_stream_sha256=free_sampler.digest()))
                event(run_dir, "carrier_frozen_time_graph_retained", step=pre)
            optimizer.zero_grad(set_to_none=True)
            if step <= pre:
                k, t = free_sampler.sample()
                residual = free_carrier_residual(model.carrier, cfg, k, t)
                loss = residual.abs().square().mean()
                power = residual.detach().abs().square().reshape(32, cfg.mode_count, 2, 2).mean((0, 2, 3))
                current_parameters = tuple(model.carrier.parameters())
            else:
                k, t = coupled_sampler.sample()
                loss, power = training_loss(model, cfg, k, t)
                current_parameters = tuple(model.networks.parameters())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in current_parameters):
                raise FloatingPointError(f"Missing/nonfinite gradient at step {step}")
            gradient = math_norm(current_parameters)
            optimizer.step()
            elapsed = time.perf_counter()-tick
            seconds += elapsed
            if step <= pre:
                pre_seconds += elapsed
            if step == 1 or step % cfg.history_every == 0 or step in (pre, pre+1, cfg.adam_steps):
                row = dict(step=step, phase=phase, objective=float(loss.detach()),
                           training_seconds=seconds, gradient_norm=gradient)
                row.update({f"n_{n}_unweighted_component_mean_power": float(v) for n, v in zip(cfg.indices, power)})
                history.append(row)
            if step == 1 or step % cfg.log_every == 0 or step == cfg.adam_steps:
                print(f"{name} seed={seed} step={step}/{cfg.adam_steps} phase={phase} loss={float(loss.detach()):.7g} seconds={seconds:.1f}", flush=True)
            if step % cfg.recovery_every == 0 or step in (pre, cfg.adam_steps) or step == stop_after:
                save(step)
            if step == stop_after:
                event(run_dir, "controlled_smoke_interruption", step=step)
                return dict(interrupted_at=step)
        if model.carrier is not None:
            model.carrier.freeze()
        save(cfg.adam_steps)
        training_peak = max(peak, int(monitor.snapshot()["rss_sampled_peak_bytes"]))
        checkpoint = dict(state_dict=model.state_dict(), configuration=cfg.as_dict(), model=name, seed=seed,
            steps=cfg.adam_steps, carrier_steps=pre, final_checkpoint_only=True,
            initial_complex_network_sha256=network_hashes, initial_carrier_sha256=carrier_hash,
            coupled_time_stream_sha256=coupled_sampler.digest(), free_time_stream_sha256=free_sampler.digest())
        atomic_torch_save(run_dir/"model_state.pt", checkpoint)
        event(run_dir, "fixed_final_checkpoint_saved", step=cfg.adam_steps)
        evaluation_start = time.perf_counter()
        if smoke:
            from .r9_evaluate import spectral_grid
            times = np.linspace(0, cfg.t_max, 3)
            states = spectral_grid(model, cfg, times)
            metrics = dict(initial_condition_max_abs_error=float(abs(states[0]-model(
                torch.tensor(cfg.wave_numbers), torch.zeros(cfg.mode_count)).detach().numpy()).max()),
                all_finite=bool(np.isfinite(states).all()), engineering_only=True, accuracy_not_used=True)
            series, dense, tables = pd.DataFrame(dict(t=times)), dict(t=times, spectral_prediction=states), {}
        else:
            from .r9_evaluate import evaluate
            metrics, series, dense, tables = evaluate(model, cfg)
        evaluation_seconds = time.perf_counter()-evaluation_start
        peak = max(training_peak, int(monitor.snapshot()["rss_sampled_peak_bytes"]))
        metrics.update(experiment=EXPERIMENT, model=name, seed=seed, smoke=smoke, steps=cfg.adam_steps,
            fixed_final_step=cfg.adam_steps, carrier_steps=pre, coupled_steps=cfg.adam_steps-pre,
            training_seconds=seconds, carrier_training_seconds=pre_seconds, coupled_training_seconds=seconds-pre_seconds,
            evaluation_seconds=evaluation_seconds, training_peak_rss_mib=training_peak/2**20, peak_rss_mib=peak/2**20,
            rss_measurement_method=monitor.snapshot()["rss_measurement_method"],
            run_elapsed_seconds_this_session=time.perf_counter()-session_start,
            training_residual_batches=cfg.adam_steps,
            training_carrier_unit_spinor_residual_points=2*pre*cfg.spectral_collocation_points,
            training_coupled_spinor_residual_points=(cfg.adam_steps-pre)*cfg.spectral_collocation_points,
            training_complex_scalar_residual_evaluations=2*(cfg.adam_steps+pre)*cfg.spectral_collocation_points,
            initial_complex_network_sha256=network_hashes, initial_carrier_sha256=carrier_hash,
            coupled_time_stream_sha256=coupled_sampler.digest(), free_time_stream_sha256=free_sampler.digest(),
            cpu_affinity_mask=configuration["cpu_affinity_mask"], **parameter_counts(model))
        series.to_csv(run_dir/"time_series.csv", index=False)
        np.savez_compressed(run_dir/"dense_grid.npz", **dense)
        for filename, table in tables.items():
            table.to_csv(run_dir/filename, index=False)
        write_json(run_dir/"metrics.json", metrics)
        event(run_dir, "complete", budget=cfg.adam_steps)
        print(f"DONE {name} seed={seed} training_seconds={seconds:.1f} evaluation_seconds={evaluation_seconds:.1f}", flush=True)
        return metrics
    except Exception as error:
        event(run_dir, "failure_preserved", error_type=type(error).__name__, message=str(error))
        raise
    finally:
        monitor.stop()


def math_norm(parameters):
    return float(torch.sqrt(sum(p.grad.square().sum() for p in parameters)))
