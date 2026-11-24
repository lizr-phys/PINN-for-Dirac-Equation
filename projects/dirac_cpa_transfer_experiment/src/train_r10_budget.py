from __future__ import annotations
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .r10_config import EXPERIMENT, MODEL, ADDITIONAL_STEPS, BASE_GLOBAL_STEP, ARTIFACTS, source_directory
from .r10_state import restore_source
from .r9_models import tensor_digest, parameter_counts
from .r9_loss import training_loss
from .r9_runtime import actual_affinity_mask
from .freeze_r7_phase_release import write_json, sha256
from .train_r7_phase_release import atomic_torch_save, event
from .train_wavepacket_shared_spectral_ltpb import ProcessRSSMonitor, _set_runtime, _environment_snapshot


def train_continuation(seed, run_dir, *, additional_steps=ADDITIONAL_STEPS, smoke=False, stop_after=None):
    run_dir = Path(run_dir)
    frozen = None
    if not smoke:
        from .freeze_r10_budget import verify_freeze
        frozen = verify_freeze()
        if additional_steps != ADDITIONAL_STEPS:
            raise ValueError("Formal continuation budget differs from preregistration")
    if (run_dir/"metrics.json").exists():
        saved = json.loads((run_dir/"metrics.json").read_text(encoding="utf-8"))
        if saved["additional_optimizer_steps"] != additional_steps or saved["seed"] != seed:
            raise RuntimeError("Changed completed continuation")
        if not all((run_dir/name).is_file() for name in ARTIFACTS+("recovery_state.pt",)):
            raise RuntimeError("Incomplete completed continuation")
        return saved
    run_dir.mkdir(parents=True, exist_ok=True)
    device = _set_runtime(seed, 1)
    cfg, model, optimizer, free, coupled, ancestry = restore_source(seed)
    original = json.loads((source_directory(seed)/"metrics.json").read_text(encoding="utf-8"))
    config = dict(experiment=EXPERIMENT, model=MODEL, seed=seed, smoke=smoke, source_configuration=cfg.as_dict(),
        inherited_global_step=BASE_GLOBAL_STEP, additional_steps=additional_steps,
        fixed_final_global_step=BASE_GLOBAL_STEP+additional_steps, new_pretraining_steps=0,
        final_checkpoint_only=True, loss_terms=["original_full_coupled_spinor_pde"],
        optimizer_reset=False, sampler_reset=False, fresh_independent_seed=False,
        cpu_affinity_mask=actual_affinity_mask(), environment=_environment_snapshot(device),
        frozen_protocol_sha256=frozen, ancestry=ancestry)
    config_path = run_dir/"config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("seed", "smoke", "additional_steps", "fixed_final_global_step", "ancestry", "frozen_protocol_sha256"):
            if old[key] != json.loads(json.dumps(config))[key]:
                raise RuntimeError("Continuation recovery configuration changed: "+key)
    else:
        write_json(config_path, config)
    recovery_path = run_dir/"recovery_state.pt"
    start, seconds, history, peak = 1, 0., [], 0
    if recovery_path.exists():
        state = torch.load(recovery_path, weights_only=False, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        model.carrier.freeze()
        optimizer.load_state_dict(state["optimizer"])
        free.load_state_dict(state["free_sampler"])
        coupled.load_state_dict(state["coupled_sampler"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        start, seconds, history, peak = state["additional_step"]+1, state["seconds"], state["history"], state["peak"]
        event(run_dir, "exact_resume", additional_step=start-1)
    else:
        event(run_dir, "conditional_continuation_start", source_global_step=BASE_GLOBAL_STEP,
              additional_budget=additional_steps, all_five_sources_included=True)
    monitor = ProcessRSSMonitor()
    monitor.start()
    session_start = time.perf_counter()
    parameters = tuple(model.networks.parameters())

    def save(step):
        nonlocal peak
        peak = max(peak, int(monitor.snapshot()["rss_sampled_peak_bytes"]))
        atomic_torch_save(recovery_path, dict(additional_step=step, global_step=BASE_GLOBAL_STEP+step,
            state_dict=model.state_dict(), optimizer=optimizer.state_dict(), phase="coupled",
            free_sampler=free.state_dict(), coupled_sampler=coupled.state_dict(),
            seconds=seconds, history=history, peak=peak, ancestry=ancestry,
            python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state()))
        pd.DataFrame(history).to_csv(run_dir/"loss_history.csv", index=False)

    try:
        for step in range(start, additional_steps+1):
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            k, t = coupled.sample()
            loss, power = training_loss(model, cfg, k, t)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite continuation loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError("Missing/nonfinite continuation gradient")
            gradient = float(torch.sqrt(sum(p.grad.square().sum() for p in parameters)))
            optimizer.step()
            seconds += time.perf_counter()-tick
            if step == 1 or step % 25 == 0 or step == additional_steps:
                row = dict(additional_step=step, global_step=BASE_GLOBAL_STEP+step, objective=float(loss.detach()),
                           additional_training_seconds=seconds, gradient_norm=gradient)
                row.update({f"n_{n}_unweighted_component_mean_power": float(p) for n, p in zip(cfg.indices, power)})
                history.append(row)
            if step == 1 or step % 500 == 0 or step == additional_steps:
                print(f"R10 seed={seed} additional_step={step}/{additional_steps} global_step={BASE_GLOBAL_STEP+step} "
                      f"loss={float(loss.detach()):.7g} seconds={seconds:.1f}", flush=True)
            if step % 500 == 0 or step == additional_steps or step == stop_after:
                save(step)
            if step == stop_after:
                event(run_dir, "controlled_engineering_interruption", additional_step=step)
                return dict(interrupted_at=step)
        save(additional_steps)
        training_peak = max(peak, int(monitor.snapshot()["rss_sampled_peak_bytes"]))
        if tensor_digest(model.carrier.state_dict().values()) != ancestry["frozen_carrier_tensor_sha256"]:
            raise RuntimeError("Frozen carrier changed during continuation")
        checkpoint = dict(state_dict=model.state_dict(), model=MODEL, seed=seed,
            physical_configuration=cfg.as_dict(), steps=BASE_GLOBAL_STEP+additional_steps,
            additional_steps=additional_steps, inherited_steps=BASE_GLOBAL_STEP, carrier_steps=6000,
            final_checkpoint_only=True, ancestry=ancestry,
            free_time_stream_sha256=free.digest(), coupled_time_stream_sha256=coupled.digest())
        atomic_torch_save(run_dir/"model_state.pt", checkpoint)
        event(run_dir, "fixed_final_checkpoint_saved", global_step=BASE_GLOBAL_STEP+additional_steps)
        tick = time.perf_counter()
        if smoke:
            from .r9_evaluate import spectral_grid
            values = spectral_grid(model, cfg, [0., 1., 5.])
            metrics = dict(engineering_only=True, accuracy_not_used=True, all_finite=bool(np.isfinite(values).all()))
            series, dense, tables = pd.DataFrame(dict(t=[0., 1., 5.])), dict(spectral_prediction=values), {}
        else:
            from .r10_evaluate import evaluate_continuation
            metrics, series, dense, tables = evaluate_continuation(model, cfg)
        evaluation_seconds = time.perf_counter()-tick
        peak = max(training_peak, int(monitor.snapshot()["rss_sampled_peak_bytes"]))
        metrics.update(experiment=EXPERIMENT, model=MODEL, seed=seed, smoke=smoke,
            steps=BASE_GLOBAL_STEP+additional_steps, fixed_final_step=BASE_GLOBAL_STEP+additional_steps,
            additional_optimizer_steps=additional_steps, inherited_optimizer_steps=BASE_GLOBAL_STEP,
            carrier_steps=6000, coupled_steps=14000+additional_steps, new_carrier_steps=0,
            new_coupled_steps=additional_steps, additional_training_seconds=seconds, new_carrier_training_seconds=0.,
            cumulative_training_seconds=original["training_seconds"]+seconds,
            inherited_training_seconds=original["training_seconds"],
            inherited_carrier_training_seconds=original["carrier_training_seconds"],
            inherited_coupled_training_seconds=original["coupled_training_seconds"],
            evaluation_seconds=evaluation_seconds,
            training_peak_rss_mib=training_peak/2**20, peak_rss_mib=peak/2**20,
            rss_measurement_method=monitor.snapshot()["rss_measurement_method"],
            run_elapsed_seconds_this_session=time.perf_counter()-session_start,
            additional_coupled_spinor_residual_points=additional_steps*cfg.spectral_collocation_points,
            additional_complex_scalar_residual_evaluations=2*additional_steps*cfg.spectral_collocation_points,
            cumulative_training_spinor_residual_points=(26000+6000)*cfg.spectral_collocation_points if not smoke else None,
            coupled_time_stream_sha256=coupled.digest(), free_time_stream_sha256=free.digest(),
            cpu_affinity_mask=actual_affinity_mask(), ancestry=ancestry, **parameter_counts(model))
        series.to_csv(run_dir/"time_series.csv", index=False)
        np.savez_compressed(run_dir/"dense_grid.npz", **dense)
        for filename, table in tables.items():
            table.to_csv(run_dir/filename, index=False)
        write_json(run_dir/"metrics.json", metrics)
        event(run_dir, "complete", additional_budget=additional_steps, global_step=BASE_GLOBAL_STEP+additional_steps)
        print(f"DONE R10 seed={seed} additional_training_seconds={seconds:.1f} evaluation_seconds={evaluation_seconds:.1f}", flush=True)
        return metrics
    except Exception as error:
        event(run_dir, "failure_preserved", error_type=type(error).__name__, message=str(error))
        raise
    finally:
        monitor.stop()
