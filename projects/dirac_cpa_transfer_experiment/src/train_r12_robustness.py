from __future__ import annotations
import json
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .r12_config import EXPERIMENT, GROUPS, SEEDS, ARTIFACTS, CURVE_STEPS, configuration, budget
from .r12_models import R12Model, hashes, parameter_counts, coupled_parameters
from .r12_residual import training_loss, free_carrier_residual
from .r8_sampling import TimeSampler
from .freeze_r7_phase_release import write_json
from .train_r7_phase_release import atomic_torch_save, event
from .train_wavepacket_shared_spectral_ltpb import ProcessRSSMonitor, _set_runtime, _environment_snapshot
from .train_r9_wavepacket import math_norm

def train_run(case, group, seed, directory, *, smoke=False, stop_after=None):
    cfg, directory = configuration(case), Path(directory)
    pre, coupled_budget = (2 if group == GROUPS[0] else 0, 2) if smoke else budget(group)
    total = pre+coupled_budget
    freeze_hash = None
    if not smoke:
        from .freeze_r12_robustness import verify_freeze
        freeze_hash = verify_freeze()
        if group not in GROUPS or seed not in SEEDS:
            raise ValueError("Run outside frozen R12 protocol")
    if (directory/"metrics.json").exists():
        result = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
        if result["steps"] != total or result["seed"] != seed or result["model"] != group:
            raise RuntimeError("Different cached run")
        if not all((directory/p).is_file() for p in ARTIFACTS+("recovery_state.pt",)):
            raise RuntimeError("Incomplete cached run")
        return result
    directory.mkdir(parents=True, exist_ok=True)
    device = _set_runtime(seed, 1)
    model = R12Model(cfg, group, seed)
    initial = hashes(model)
    config = dict(experiment=EXPERIMENT, case=case, model=group, seed=seed, smoke=smoke,
        configuration=cfg.as_dict(), carrier_steps=pre, coupled_steps=coupled_budget,
        total_budget=total, fresh_initialization=True, initial_hashes=initial,
        analytic_prior=(group == GROUPS[-1]), freeze_manifest_sha256=freeze_hash,
        cpu_affinity_mask=os.environ.get("R7_CPU_AFFINITY_MASK"), environment=_environment_snapshot(device),
        random_streams=dict(carrier_init=900000+seed, envelope_init=800000+seed,
            second_complex_init=830000+seed, free_time=910000+seed, coupled_time=920000+seed),
        loss_terms=["unweighted_complete_spinor_PDE"],
        snapshot_policy="fixed_coupled_steps; no selection")
    if (directory/"config.json").exists():
        old = json.loads((directory/"config.json").read_text(encoding="utf-8"))
        for key in ("model", "seed", "smoke", "configuration", "freeze_manifest_sha256", "cpu_affinity_mask"):
            if old[key] != json.loads(json.dumps(config))[key]:
                raise RuntimeError("Changed recovery configuration: "+key)
    else:
        write_json(directory/"config.json", config)
        atomic_torch_save(directory/"initial_state.pt", dict(state_dict=model.state_dict(), model=group, seed=seed))
    free, coupled = TimeSampler(cfg, seed, carrier=True), TimeSampler(cfg, seed)
    phase = "carrier" if pre else "coupled"
    optimizer = torch.optim.Adam(model.carrier.parameters() if pre else coupled_parameters(model),
                                 lr=cfg.learning_rate)
    start, seconds, pre_seconds, evaluation_seconds, io_seconds, peak = 1, 0., 0., 0., 0., 0
    history, curves, completed = [], [], []
    recovery = directory/"recovery_state.pt"
    if recovery.exists():
        saved = torch.load(recovery, weights_only=False)
        model.load_state_dict(saved["state_dict"])
        phase = saved["phase"]
        if phase == "coupled" and group == GROUPS[0]:
            model.carrier.freeze()
        optimizer = torch.optim.Adam(model.carrier.parameters() if phase == "carrier" else coupled_parameters(model),
                                     lr=cfg.learning_rate)
        optimizer.load_state_dict(saved["optimizer"])
        free.load_state_dict(saved["free_sampler"]); coupled.load_state_dict(saved["coupled_sampler"])
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        start = saved["step"]+1
        seconds, pre_seconds, evaluation_seconds, io_seconds, peak = (
            saved[k] for k in ("seconds", "pre_seconds", "evaluation_seconds", "io_seconds", "peak"))
        history, curves, completed = saved["history"], saved["curves"], saved["completed_snapshots"]
        event(directory, "exact_resume", step=start-1)
    else:
        event(directory, "start", budget=total, group=group, seed=seed)
    monitor = ProcessRSSMonitor(); monitor.start()
    session_start = time.perf_counter()

    def save(step):
        nonlocal peak, io_seconds
        tick = time.perf_counter()
        peak = max(peak, int(monitor.snapshot()["rss_sampled_peak_bytes"]))
        atomic_torch_save(recovery, dict(step=step, phase=phase, state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(), model=group, seed=seed, configuration=cfg.as_dict(),
            free_sampler=free.state_dict(), coupled_sampler=coupled.state_dict(),
            seconds=seconds, pre_seconds=pre_seconds, evaluation_seconds=evaluation_seconds,
            io_seconds=io_seconds, history=history, curves=curves, completed_snapshots=completed, peak=peak,
            python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state()))
        pd.DataFrame(history).to_csv(directory/"loss_history.csv", index=False)
        pd.DataFrame(curves).to_csv(directory/"cost_curve.csv", index=False)
        io_seconds += time.perf_counter()-tick

    def snapshot(step):
        nonlocal evaluation_seconds, io_seconds, peak
        coupled_step = step-pre
        target = directory/"snapshots"/f"coupled_{coupled_step:05d}"
        target.mkdir(parents=True, exist_ok=True)
        full = coupled_step in (20000, coupled_budget) and not smoke
        tick = time.perf_counter()
        write_json(target/"config.json", config | dict(snapshot_coupled_step=coupled_step, snapshot_total_step=step))
        atomic_torch_save(target/"model_state.pt", dict(state_dict=model.state_dict(), configuration=cfg.as_dict(),
            model=group, seed=seed, steps=step, carrier_steps=pre, coupled_steps=coupled_step,
            fixed_endpoint=True, analytic_prior=group == GROUPS[-1], initial_hashes=initial,
            coupled_time_stream_sha256=coupled.digest(), free_time_stream_sha256=free.digest(),
            coupled_normalized_time_stream_sha256=coupled.digest(normalized=True)))
        event(directory, "fixed_checkpoint_saved_before_evaluation", step=step, coupled_step=coupled_step, full=full)
        pd.DataFrame(history).to_csv(target/"loss_history.csv", index=False)
        if full or smoke:
            atomic_torch_save(target/"recovery_state.pt", torch.load(recovery, weights_only=False))
        io_seconds += time.perf_counter()-tick
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
                  free.generator.get_state().clone(), coupled.generator.get_state().clone())
        tick = time.perf_counter()
        if smoke:
            from .r9_evaluate import spectral_grid
            from .r9_models import compose
            times = np.linspace(0, cfg.t_max, 3)
            states = spectral_grid(model, cfg, times)
            metrics = dict(all_finite=bool(np.isfinite(states).all()), engineering_only=True,
                accuracy_not_used=True, initial_condition_max_abs_error=float(abs(states[0]-compose(
                    model.initial_coeff, torch.tensor(cfg.wave_numbers), cfg.mass).detach().numpy()).max()),
                evaluation_coupled_spinor_residual_points=0, evaluation_carrier_unit_spinor_residual_points=0)
            series, dense, tables = pd.DataFrame(dict(t=times)), dict(t=times, spectral_prediction=states), {}
        else:
            from .r12_evaluate import evaluate
            metrics, series, dense, tables = evaluate(model, cfg, full=full)
        elapsed = time.perf_counter()-tick
        evaluation_seconds += elapsed
        if (random.getstate() != before[0] or not np.array_equal(np.random.get_state()[1], before[1][1])
            or not torch.equal(torch.get_rng_state(), before[2])
            or not torch.equal(free.generator.get_state(), before[3])
            or not torch.equal(coupled.generator.get_state(), before[4])):
            raise RuntimeError("Independent evaluation changed a training random stream")
        peak = max(peak, int(monitor.snapshot()["rss_sampled_peak_bytes"]))
        metrics.update(experiment=EXPERIMENT, case=case, model=group, seed=seed, smoke=smoke,
            steps=step, fixed_endpoint=True, carrier_steps=pre, coupled_steps=coupled_step,
            training_seconds=seconds, carrier_training_seconds=pre_seconds,
            coupled_training_seconds=seconds-pre_seconds, evaluation_seconds=elapsed,
            cumulative_evaluation_seconds=evaluation_seconds, io_seconds=io_seconds,
            training_residual_batches=step,
            training_carrier_unit_spinor_residual_points=2*pre*cfg.spectral_collocation_points,
            training_coupled_spinor_residual_points=coupled_step*cfg.spectral_collocation_points,
            training_spinor_residual_points=(2*pre+coupled_step)*cfg.spectral_collocation_points,
            training_complex_scalar_residual_evaluations=2*(2*pre+coupled_step)*cfg.spectral_collocation_points,
            peak_rss_mib=peak/2**20,
            rss_measurement_method=monitor.snapshot()["rss_measurement_method"],
            run_elapsed_seconds_this_session=time.perf_counter()-session_start,
            initial_hashes=initial, current_hashes=hashes(model),
            coupled_time_stream_sha256=coupled.digest(), free_time_stream_sha256=free.digest(),
            coupled_normalized_time_stream_sha256=coupled.digest(normalized=True),
            cpu_affinity_mask=config["cpu_affinity_mask"], analytic_prior=group == GROUPS[-1],
            evaluation_does_not_change_training_rng=True, **parameter_counts(model))
        tick = time.perf_counter()
        series.to_csv(target/"time_series.csv", index=False)
        np.savez_compressed(target/"dense_grid.npz", **dense)
        for name, table in tables.items():
            table.to_csv(target/name, index=False)
        write_json(target/"metrics.json", metrics)
        io_seconds += time.perf_counter()-tick
        curves.append({k:v for k, v in metrics.items() if isinstance(v, (int, float, str, bool)) or v is None})
        completed.append(coupled_step)
        save(step)
        if coupled_step == coupled_budget:
            for name in ARTIFACTS:
                shutil.copyfile(target/name, directory/name)
            for name in tables:
                shutil.copyfile(target/name, directory/name)
            event(directory, "complete", budget=total, coupled_steps=coupled_step)
        print(f"CHECKPOINT {group} seed={seed} coupled={coupled_step} total={step} eval_seconds={elapsed:.1f}", flush=True)
        return metrics

    try:
        previous = start-1
        if previous > pre and previous-pre in (CURVE_STEPS if not smoke else (coupled_budget,)):
            if previous-pre not in completed:
                snapshot(previous)
        for step in range(start, total+1):
            if step > pre and phase == "carrier":
                transition = time.perf_counter()
                model.carrier.freeze(); phase = "coupled"
                optimizer = torch.optim.Adam(coupled_parameters(model), lr=cfg.learning_rate)
                atomic_torch_save(directory/"pretrained_carrier_state.pt", dict(state_dict=model.carrier.state_dict(),
                    steps=pre, seed=seed, training_seconds=pre_seconds, free_time_stream_sha256=free.digest()))
                event(directory, "carrier_frozen_time_graph_retained", step=pre)
                io_seconds += time.perf_counter()-transition
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if step <= pre:
                k, t = free.sample()
                residual = free_carrier_residual(model.carrier, cfg, k, t)
                loss = residual.abs().square().mean()
                power = residual.detach().abs().square().reshape(32, cfg.mode_count, 2, 2).mean((0, 2, 3))
                parameters = tuple(model.carrier.parameters())
            else:
                k, t = coupled.sample()
                loss, power = training_loss(model, cfg, k, t)
                parameters = coupled_parameters(model)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite objective")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError("Missing/nonfinite active parameter gradient")
            grad = math_norm(parameters)
            optimizer.step()
            elapsed = time.perf_counter()-tick
            seconds += elapsed
            if step <= pre:
                pre_seconds += elapsed
            if step == 1 or step % cfg.history_every == 0 or step in (pre, pre+1, total):
                history.append(dict(step=step, coupled_step=max(0, step-pre), phase=phase, objective=float(loss.detach()),
                    training_seconds=seconds, gradient_norm=grad,
                    **{f"n_{n}_unweighted_component_mean_power":float(v) for n,v in zip(cfg.indices, power)}))
            if step == 1 or step % cfg.log_every == 0 or step == total:
                print(f"{group} seed={seed} step={step}/{total} coupled={max(0,step-pre)} phase={phase} loss={float(loss.detach()):.7g} seconds={seconds:.1f}", flush=True)
            if step % cfg.recovery_every == 0 or step in (pre, total) or step == stop_after:
                save(step)
            if step > pre and (step-pre in CURVE_STEPS or step == total):
                save(step); snapshot(step)
            if step == stop_after:
                event(directory, "controlled_smoke_interruption", step=step)
                return dict(interrupted_at=step)
        write_json(directory/"execution_cost.json", dict(training_seconds=seconds,
            carrier_training_seconds=pre_seconds, coupled_training_seconds=seconds-pre_seconds,
            evaluation_seconds=evaluation_seconds, checkpoint_and_artifact_io_seconds=io_seconds,
            elapsed_seconds_this_session=time.perf_counter()-session_start, peak_rss_mib=peak/2**20,
            persistent_seed_worker=True, includes_resident_allocator_memory=True,
            pretraining_active_parameters=12802 if pre else 0,
            pretraining_idle_parameters=12932 if pre else 0, **parameter_counts(model)))
        return json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
    except Exception as error:
        event(directory, "failure_preserved", error_type=type(error).__name__, message=str(error))
        raise
    finally:
        monitor.stop()
