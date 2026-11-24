from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .freeze_r7_phase_release import write_json
from .periodic_potential_config import paired_initialization_seed, paired_time_collocation_seed
from .periodic_potential_models import count_trainable_parameters, trainable_state_sha256
from .r7_config import EXPERIMENT, MODELS, TRAINING_MODES, stage_config
from .r7_models import build_model
from .r7_sampling import StratifiedTimeSampler
from .r7_loss import training_loss, epsilon_for_step
from .train_wavepacket_shared_spectral_ltpb import ProcessRSSMonitor, _set_runtime, _environment_snapshot

ARTIFACTS = ("config.json", "metrics.json", "loss_history.csv", "time_series.csv", "dense_grid.npz", "model_state.pt")


def atomic_torch_save(path, payload):
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def event(run_dir, event_name, **kwargs):
    record = dict(utc=datetime.now(timezone.utc).isoformat(), event=event_name, **kwargs)
    with (run_dir/"execution_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False)+"\n")


def recovery_payload(model, optimizer, sampler, cfg, *, model_name, mode, seed, step,
                     history, training_seconds, peak_bytes, initial_bytes):
    return dict(model=model_name, training_mode=mode, seed=seed, step=step,
                configuration=cfg.as_dict(), state_dict=model.state_dict(),
                optimizer_state=optimizer.state_dict(), sampler_state=sampler.state_dict(),
                python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state(),
                torch_rng_state=torch.get_rng_state(), history=history,
                training_seconds=training_seconds, training_peak_rss_bytes=peak_bytes,
                training_initial_rss_bytes=initial_bytes,
                schedule_step=step, next_epsilon=epsilon_for_step(cfg, min(step+1, cfg.adam_steps)))


def restore_recovery(payload, model, optimizer, sampler):
    model.load_state_dict(payload["state_dict"])
    optimizer.load_state_dict(payload["optimizer_state"])
    sampler.load_state_dict(payload["sampler_state"])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])


def train_run(cfg, model_name, training_mode, seed, run_dir, *, stage, formal=True, stop_after=None):
    if model_name not in MODELS or training_mode not in TRAINING_MODES:
        raise ValueError("Unregistered representation or training mode")
    if seed not in range(5) or (stage == "short" and (seed > 2 or training_mode != "ordinary")):
        raise ValueError("Seed or short-time mode is outside protocol")
    frozen_hash = None
    if formal:
        from .freeze_r7_phase_release import verify_freeze
        frozen_hash = verify_freeze()
        expected = stage_config(stage, long_half_width=cfg.ladder_half_width if stage == "long" else None)
        if cfg != expected:
            raise ValueError("Formal config differs from preregistered stage")
    run_dir = Path(run_dir)
    if (run_dir/"metrics.json").exists():
        if not all((run_dir/f).is_file() for f in ARTIFACTS):
            raise ValueError("Completed run has missing evidence")
        cached = json.loads((run_dir/"metrics.json").read_text(encoding="utf-8"))
        if cached["steps"] != cfg.adam_steps or cached["model"] != model_name or cached["training_mode"] != training_mode:
            raise ValueError("Existing completed run is inconsistent")
        print(f"SKIP {stage}/{model_name}/{training_mode}/seed{seed}", flush=True)
        return cached
    run_dir.mkdir(parents=True, exist_ok=True)
    device = _set_runtime(seed, cfg.cpu_threads)
    model = build_model(model_name, cfg, paired_initialization_seed(seed))
    initial_hash = trainable_state_sha256(model)
    if count_trainable_parameters(model) != 12932:
        raise AssertionError("Matched parameter budget broken")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    sampler = StratifiedTimeSampler(cfg, seed)
    config = dict(experiment=EXPERIMENT, stage=stage, model=model_name, training_mode=training_mode,
                  seed=seed, formal=formal, configuration=cfg.as_dict(), parameter_count=12932,
                  initial_trainable_sha256=initial_hash, loss_terms=["pde"],
                  random_streams=dict(initialization_seed=paired_initialization_seed(seed),
                                      time_seed=paired_time_collocation_seed(seed)),
                  freeze_manifest_sha256=frozen_hash, environment=_environment_snapshot(device),
                  cpu_affinity_mask=os.environ.get("R7_CPU_AFFINITY_MASK"),
                  execution_policy="three distinct physical cores; one thread/worker; seed-rotated core assignment")
    previous = run_dir/"config.json"
    if previous.exists():
        saved = json.loads(previous.read_text(encoding="utf-8"))
        for key in ("configuration", "model", "training_mode", "seed", "initial_trainable_sha256", "freeze_manifest_sha256"):
            if saved[key] != json.loads(json.dumps(config))[key]:
                raise ValueError(f"Cannot resume changed config: {key}")
    else:
        write_json(previous, config)
    history, seconds, peak_bytes, initial_bytes, start_step = [], 0.0, 0, 0, 1
    recovery = run_dir/"recovery_state.pt"
    if recovery.exists():
        payload = torch.load(recovery, weights_only=False)
        restore_recovery(payload, model, optimizer, sampler)
        history, seconds = payload["history"], payload["training_seconds"]
        peak_bytes, initial_bytes = payload["training_peak_rss_bytes"], payload["training_initial_rss_bytes"]
        start_step = payload["step"]+1
        event(run_dir, "resume_exact_saved_state", saved_step=payload["step"])
    else:
        event(run_dir, "start", steps=cfg.adam_steps)
    print(f"START {stage}/{model_name}/{training_mode} seed={seed} step={start_step} steps={cfg.adam_steps}", flush=True)
    checkpoints = sorted(set(cfg.checkpoint_steps).intersection(range(1, cfg.adam_steps+1)))
    monitor = ProcessRSSMonitor()
    monitor.start()
    initial_bytes = initial_bytes or monitor.initial_bytes
    run_started = time.perf_counter()
    last_unweighted, last_objective = None, None

    def save_recovery(step):
        memory = monitor.snapshot()
        nonlocal peak_bytes
        peak_bytes = max(peak_bytes, int(memory["rss_sampled_peak_bytes"]))
        payload = recovery_payload(model, optimizer, sampler, cfg, model_name=model_name,
                                   mode=training_mode, seed=seed, step=step, history=history,
                                   training_seconds=seconds, peak_bytes=peak_bytes, initial_bytes=initial_bytes)
        atomic_torch_save(recovery, payload)
        pd.DataFrame(history).to_csv(run_dir/"loss_history.csv", index=False)
        return payload

    def evaluate_checkpoint(step, final):
        nonlocal monitor, peak_bytes
        memory = monitor.snapshot()
        peak_bytes = max(peak_bytes, int(memory["rss_sampled_peak_bytes"]))
        monitor.stop()
        destination = run_dir if final else run_dir/"checkpoints"/f"step_{step}"
        destination.mkdir(parents=True, exist_ok=True)
        from .r7_evaluate import evaluate_run
        evaluation_started = time.perf_counter()
        metrics, series, dense, extra = evaluate_run(
            model, cfg, model_name, seed, steps=step, training_seconds=seconds,
            final_loss=last_unweighted, peak_rss_mb=peak_bytes/2**20,
            peak_increment_mb=max(0, peak_bytes-initial_bytes)/2**20, full_diagnostics=final)
        metrics.update(training_mode=training_mode, stage=stage, formal=formal,
                       final_weighted_objective=last_objective,
                       time_stream_sha256=sampler.digest(), initial_trainable_sha256=initial_hash,
                       residual_training_points=step*cfg.spectral_collocation_points,
                       residual_training_batches=step, checkpoint_fixed_step=step,
                       evaluation_seconds=time.perf_counter()-evaluation_started,
                       rss_measurement_method=memory["rss_measurement_method"])
        write_json(destination/"metrics.json", metrics)
        series.to_csv(destination/"time_series.csv", index=False)
        np.savez_compressed(destination/"dense_grid.npz", **dense)
        for filename, table in extra.items():
            table.to_csv(destination/filename, index=False)
        model.train()
        monitor = ProcessRSSMonitor()
        monitor.start()
        return metrics

    try:
        # If a saved fixed checkpoint lacks evaluation, evaluate those exact weights first.
        if start_step > 1 and start_step-1 in checkpoints and start_step-1 < cfg.adam_steps:
            last_unweighted, last_objective = history[-1]["unweighted_pde"], history[-1]["objective"]
            if not (run_dir/"checkpoints"/f"step_{start_step-1}"/"metrics.json").exists():
                evaluate_checkpoint(start_step-1, False)
        for step in range(start_step, cfg.adam_steps+1):
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            k, t = sampler.sample()
            objective, segments, weights = training_loss(model, cfg, k, t, training_mode, step)
            if not torch.isfinite(objective):
                raise FloatingPointError(f"Nonfinite objective at {step}")
            objective.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"Nonfinite or missing gradient at {step}")
            last_unweighted = float(segments.detach().mean())
            last_objective = float(objective.detach())
            output_grad = model.mlp.net[-1].weight.grad
            gradients = (float(torch.linalg.vector_norm(output_grad[:2])),
                         float(torch.linalg.vector_norm(output_grad[2:])))
            optimizer.step()
            seconds += time.perf_counter()-tick
            if step == 1 or step % cfg.history_every == 0 or step in checkpoints or (step-1) % cfg.epsilon_steps == 0:
                row = dict(step=step, objective=last_objective, unweighted_pde=last_unweighted,
                           pde=last_unweighted, elapsed_training_seconds=seconds,
                           epsilon=epsilon_for_step(cfg, step) if training_mode == "causal" else 0,
                           output_r_gradient_norm=gradients[0], output_b_gradient_norm=gradients[1],
                           residual_training_points=step*cfg.spectral_collocation_points,
                           collocation_batch_index=sampler.batch_index)
                for i in range(cfg.time_bins):
                    row[f"segment_{i+1}_unweighted_pde"] = float(segments[i].detach())
                    row[f"segment_{i+1}_weight"] = float(weights[i].detach())
                history.append(row)
            if step % cfg.log_every == 0 or step == 1 or step == cfg.adam_steps:
                print(f"{stage}/{model_name}/{training_mode} s={seed} step={step}/{cfg.adam_steps} weighted={last_objective:.6g} raw={last_unweighted:.6g}", flush=True)
            if step % cfg.recovery_every == 0 or step in checkpoints or step == stop_after:
                payload = save_recovery(step)
                if step in checkpoints:
                    ck_dir = run_dir/"checkpoints"/f"step_{step}"
                    ck_dir.mkdir(parents=True, exist_ok=True)
                    if not (ck_dir/"training_state.pt").exists():
                        atomic_torch_save(ck_dir/"training_state.pt", payload)
                        atomic_torch_save(ck_dir/"model_state.pt", dict(
                            model=model_name, training_mode=training_mode, seed=seed, steps=step,
                            configuration=cfg.as_dict(), state_dict=model.state_dict(),
                            initial_trainable_sha256=initial_hash, time_stream_sha256=sampler.digest()))
                    if step < cfg.adam_steps:
                        evaluate_checkpoint(step, False)
                if stop_after == step:
                    event(run_dir, "test_controlled_interruption", saved_step=step)
                    return {"interrupted_at": step}
        if last_unweighted is None:
            last_unweighted, last_objective = history[-1]["unweighted_pde"], history[-1]["objective"]
        atomic_torch_save(run_dir/"model_state.pt", dict(
            model=model_name, training_mode=training_mode, seed=seed, steps=cfg.adam_steps,
            configuration=cfg.as_dict(), state_dict=model.state_dict(),
            initial_trainable_sha256=initial_hash, time_stream_sha256=sampler.digest()))
        save_recovery(cfg.adam_steps)
        pd.DataFrame(history).to_csv(run_dir/"loss_history.csv", index=False)
        metrics = evaluate_checkpoint(cfg.adam_steps, True)
        metrics["run_elapsed_seconds_this_session"] = time.perf_counter()-run_started
        metrics["interruption_and_resume_events"] = sum(1 for _ in (run_dir/"execution_events.jsonl").open(encoding="utf-8"))
        write_json(run_dir/"metrics.json", metrics)
        event(run_dir, "complete", steps=cfg.adam_steps, time_digest=sampler.digest())
        print(f"DONE {stage}/{model_name}/{training_mode} seed={seed} train_seconds={seconds:.1f}", flush=True)
        return metrics
    except Exception as exc:
        event(run_dir, "interrupted_error", error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        monitor.stop()
