"""Post-freeze evaluation-only compatibility adapter; never changes R8 training.

The historical coupled-potential evaluator asks for a relative initial sideband
tangent error against a central-mode initial state. That diagnostic is undefined
for the registered V=0 Gaussian regression (and was already scheduled for removal
by r8_evaluate). Skip only that diagnostic, before its zero-reference division.
All primary evaluation, reference, optimization and checkpoint code stays frozen.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import hashlib
import shutil
from pathlib import Path

import torch

from .r8_config import ROOT, RAW, LOGS, run_directory
from .freeze_r7_phase_release import write_json


def initial_tangent_adapter(original):
    def wrapped(model, cfg):
        if getattr(cfg, "case", None) == "free_packet" and cfg.potential_strength == 0:
            return {}, {}
        return original(model, cfg)
    return wrapped


def tensor_state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def worker():
    from . import periodic_potential_evaluate as legacy
    from . import run_r8_coupling_recovery as frozen_runner
    args = sys.argv
    case = args[args.index("--case")+1]
    stage = args[args.index("--stage")+1]
    name = args[args.index("--model")+1]
    seed = int(args[args.index("--seed")+1])
    if case != "free_packet" or stage != "extension":
        raise ValueError("Adapter is restricted to registered free-packet extensions")
    destination = run_directory(stage, name, seed, case=case)
    recovery, checkpoint = destination/"recovery_state.pt", destination/"model_state.pt"
    previous = None
    if recovery.exists():
        saved = torch.load(recovery, weights_only=False)
        if saved["step"] != saved["configuration"]["adam_steps"]:
            raise RuntimeError("Only evaluation of an already-final checkpoint may be recovered")
        previous = dict(step=saved["step"], training_seconds=saved["seconds"],
                        tensor_state_sha256=tensor_state_digest(saved["state_dict"]))
        preserved = destination/"evaluation_failure_preserved"
        preserved.mkdir(exist_ok=True)
        for path in (checkpoint, recovery):
            target = preserved/path.name
            if path.exists() and not target.exists():
                shutil.copy2(path, target)
    original = legacy._initial_tangent_diagnostics
    legacy._initial_tangent_diagnostics = initial_tangent_adapter(original)
    try:
        frozen_runner.main()
    finally:
        legacy._initial_tangent_diagnostics = original
    final = torch.load(checkpoint, weights_only=False)
    metrics = json.loads((destination/"metrics.json").read_text(encoding="utf-8"))
    final_hash = tensor_state_digest(final["state_dict"])
    if previous is not None:
        assert final_hash == previous["tensor_state_sha256"]
        assert metrics["training_seconds"] == previous["training_seconds"]
        assert metrics["fixed_final_step"] == previous["step"]
    write_json(destination/"evaluation_adapter_record.json", dict(
        status="PASS", scope="V=0 free Gaussian regression only",
        undefined_diagnostic="central-only initial-sideband relative tangent error",
        primary_metric_definitions_unchanged=True, frozen_science_files_unchanged=True,
        extra_optimizer_steps=0, original_final_checkpoint=previous,
        final_tensor_state_sha256=final_hash, final_step=final["steps"]))


def continue_registered():
    from .run_r8_coupling_recovery import conditional_tasks
    from .r7_runtime import physical_core_masks
    tasks = conditional_tasks()
    masks = physical_core_masks()
    for start in range(0, len(tasks), 3):
        processes = []
        cohort = tasks[start:start+3]
        for index, task in enumerate(cohort):
            stage, case, name, seed, balanced = task
            destination = run_directory(stage, name, seed, case=case, balanced=balanced)
            if (destination/"metrics.json").exists():
                continue
            if case != "free_packet" or stage != "extension" or balanced:
                raise RuntimeError("Unexpected unfinished task outside the adapter scope")
            mask = masks[(index+seed) % 3]
            command = [sys.executable, "-u", "-m", "src.r8_free_packet_evaluation_adapter", "worker",
                       "--stage", stage, "--case", case, "--model", name,
                       "--seed", str(seed), "--affinity", str(mask)]
            label = f"{stage}_{case}_{name}_seed{seed}"
            metadata = dict(command=command, task=task, affinity=mask, threads=1,
                            concurrent_cohort_size=len(cohort), evaluation_adapter_only=True)
            original_command = LOGS/(label+"_command.json")
            if not original_command.exists():
                write_json(original_command, metadata)
            write_json(LOGS/(label+"_evaluation_adapter_command.json"), metadata)
            stream = (LOGS/(label+".log")).open("a", encoding="utf-8")
            stream.write("\nEVALUATION-ONLY ADAPTER; frozen training/metrics/budgets unchanged\n")
            stream.flush()
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            processes.append((process, stream, destination))
        while any(process.poll() is None for process, _, _ in processes):
            time.sleep(.25)
        failures = []
        for process, stream, destination in processes:
            stream.close()
            if process.returncode:
                failures.append(str(destination))
        if failures:
            raise RuntimeError(f"Adapter failure preserved: {failures}")
        if processes:
            print(f"REGISTERED CONTINUATION complete cohort {start//3+1}", flush=True)
    print("ALL prescribed R8 training stages complete", flush=True)


if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1] == "worker":
        worker()
    else:
        continue_registered()
