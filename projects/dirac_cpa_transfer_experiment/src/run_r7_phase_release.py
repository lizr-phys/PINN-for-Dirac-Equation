from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .freeze_r7_phase_release import ROOT, RAW, write_json, capture_protected_history, freeze
from .r7_config import MODELS, TRAINING_MODES, stage_config
from .r7_reference import validate_references
from .train_r7_phase_release import train_run


def directory(stage, model, mode, seed):
    return RAW/stage/model/mode/f"seed_{seed}"


def prepare():
    history = capture_protected_history()
    reference = validate_references(ROOT, RAW)
    print(json.dumps(dict(protected_files=len(history["files"]), main_reference=reference["main"]["pass"],
                          long_reference_J=reference["selected_long_half_width"])), flush=True)
    return reference


def run_stage(stage, *, reference=None):
    reference = reference or json.loads((RAW/"reference_validation.json").read_text(encoding="utf-8"))
    if stage == "smoke":
        cfg, seeds, modes = stage_config(stage), (0,), TRAINING_MODES
    elif stage == "short":
        cfg, seeds, modes = stage_config(stage), (0, 1, 2), ("ordinary",)
    elif stage == "main":
        cfg, seeds, modes = stage_config(stage), range(5), TRAINING_MODES
    elif stage == "long":
        decision = json.loads((RAW/"decision.json").read_text(encoding="utf-8"))
        mode = decision["long_training_mode_selected"]
        if mode is None or not reference["long_reference_pass"]:
            result = dict(status="NOT_TRIGGERED" if mode is None else "REFERENCE_BLOCKED",
                          reason="No main release configuration passed absolute conditions" if mode is None else "Long reference truncation gates failed")
            write_json(RAW/"long_validation_status.json", result)
            print(json.dumps(result), flush=True)
            return []
        cfg = stage_config(stage, long_half_width=reference["selected_long_half_width"])
        seeds, modes = range(5), (mode,)
    else:
        raise ValueError(stage)
    records = []
    # Fixed three-worker cohorts, distinct physical cores, one thread per worker.
    # Rotate core assignment by seed; complete a cohort before starting the next.
    from .r7_runtime import physical_core_masks
    masks = physical_core_masks()
    if len(masks) < 3:
        raise RuntimeError("Three distinct physical cores are required by frozen execution protocol")
    log_root = ROOT/"results/logs"/"coupled_ltpb_phase_release_r7"
    log_root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for mode in modes:
            processes = []
            for model_index, model in enumerate(MODELS):
                mask = masks[(model_index+seed) % 3]
                command = [sys.executable, "-u", "-m", "src.run_r7_phase_release", "worker",
                           "--stage", stage, "--model", model, "--mode", mode,
                           "--seed", str(seed), "--affinity", str(mask)]
                label = f"{stage}_{model}_{mode}_seed_{seed}"
                environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                stream = (log_root/(label+".log")).open("a", encoding="utf-8")
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                           stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                processes.append((process, stream, model))
                write_json(log_root/(label+"_command.json"), dict(command=command, affinity_mask=mask,
                           threads=1, concurrent_cohort_size=3, stage=stage, seed=seed, mode=mode))
            while any(process.poll() is None for process, _, _ in processes):
                time.sleep(0.25)
            failures = []
            for process, stream, model in processes:
                stream.close()
                if process.returncode:
                    failures.append(dict(model=model, exit_code=process.returncode))
                else:
                    records.append(json.loads((directory(stage, model, mode, seed)/"metrics.json").read_text(encoding="utf-8")))
            if failures:
                raise RuntimeError(f"Cohort failed; preserved evidence; resume only exact states: {failures}")
            print(f"COHORT DONE {stage}/{mode}/seed{seed}: {len(records)} runs", flush=True)
    if stage == "smoke":
        write_json(RAW/"smoke_validation.json", dict(status="PASS", engineering_only=True,
                   completed_runs=len(records), all_initial_errors_below_1e_minus_12=all(r["initial_condition_max_abs_error"] < 1e-12 for r in records),
                   finite_metrics_and_artifacts=True))
    if stage == "long":
        write_json(RAW/"long_validation_status.json", dict(status="TRAINED", completed_runs=len(records),
                   training_mode=modes[0], half_width=cfg.ladder_half_width, time_horizon=8.0))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "smoke", "freeze", "short", "main", "long", "all", "worker"))
    parser.add_argument("--stage", choices=("smoke", "short", "main", "long"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--mode", choices=TRAINING_MODES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--affinity", type=int)
    args = parser.parse_args()
    if args.action == "worker":
        from .r7_runtime import set_cpu_affinity
        set_cpu_affinity(args.affinity)
        reference = json.loads((RAW/"reference_validation.json").read_text(encoding="utf-8"))
        cfg = stage_config(args.stage, long_half_width=reference["selected_long_half_width"] if args.stage == "long" else None)
        train_run(cfg, args.model, args.mode, args.seed, directory(args.stage, args.model, args.mode, args.seed),
                  stage=args.stage, formal=args.stage != "smoke")
    elif args.action == "prepare":
        prepare()
    elif args.action == "freeze":
        print(freeze())
    elif args.action == "all":
        reference = prepare()
        run_stage("smoke", reference=reference)
        freeze()
        run_stage("short", reference=reference)
        run_stage("main", reference=reference)
        from .aggregate_r7_phase_release import aggregate
        aggregate()
        run_stage("long", reference=reference)
        aggregate()
    else:
        run_stage(args.action)


if __name__ == "__main__":
    main()
