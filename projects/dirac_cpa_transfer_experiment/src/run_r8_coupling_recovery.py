from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from .r8_config import (ROOT, RAW, LOGS, CANDIDATES, CONTROLS, EXPLORATION_SEEDS,
    VALIDATION_SEEDS, EXTENSION_SEEDS, config_for, run_directory)
from .freeze_r7_phase_release import write_json
from .freeze_r8_coupling_recovery import capture_history, freeze


def run_tasks(tasks):
    from .r7_runtime import physical_core_masks
    masks = physical_core_masks()
    if len(masks) < 3:
        raise RuntimeError("Three distinct physical cores required")
    LOGS.mkdir(parents=True, exist_ok=True)
    records = []
    for start in range(0, len(tasks), 3):
        processes = []
        for index, task in enumerate(tasks[start:start+3]):
            stage, case, model, seed, balanced = task
            destination = run_directory(stage, model, seed, case=case, balanced=balanced)
            mask = masks[(index+seed) % 3]
            command = [sys.executable, "-u", "-m", "src.run_r8_coupling_recovery", "worker",
                       "--stage", stage, "--case", case, "--model", model,
                       "--seed", str(seed), "--affinity", str(mask)]
            if balanced:
                command.append("--balanced")
            label = f"{stage}_{case}_{model}_seed{seed}"+("_balanced" if balanced else "")
            stream = (LOGS/(label+".log")).open("a", encoding="utf-8")
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            processes.append((proc, stream, destination))
            write_json(LOGS/(label+"_command.json"), dict(command=command, task=task, affinity=mask,
                threads=1, concurrent_cohort_size=len(tasks[start:start+3])))
        while any(proc.poll() is None for proc, _, _ in processes):
            time.sleep(.25)
        failures = []
        for proc, stream, destination in processes:
            stream.close()
            if proc.returncode:
                failures.append(dict(run=str(destination), exit_code=proc.returncode))
            else:
                records.append(json.loads((destination/"metrics.json").read_text(encoding="utf-8")))
        if failures:
            raise RuntimeError(f"Preserved failed cohort; exact recovery only: {failures}")
        print(f"COHORT complete {min(start+3, len(tasks))}/{len(tasks)}", flush=True)
    return records


def tasks_for(stage):
    if stage == "smoke":
        seeds, names = (0,), CANDIDATES+CONTROLS
    elif stage == "exploration":
        seeds, names = EXPLORATION_SEEDS, CANDIDATES+CONTROLS
    elif stage == "validation":
        selection = json.loads((RAW/"exploration_selection.json").read_text(encoding="utf-8"))
        seeds, names = VALIDATION_SEEDS, tuple(selection["selected_candidates"])+CONTROLS
    else:
        raise ValueError(stage)
    return [(stage, "main", name, seed, False) for seed in seeds for name in names]


def conditional_tasks():
    from .r8_decision import conditional_plan
    plan = conditional_plan()
    tasks = [("ablation", "main", name, seed, False) for seed in VALIDATION_SEEDS for name in plan["ablations"]]
    tasks += [("extension", case, name, seed, False) for case in plan["extension_cases"]
              for seed in EXTENSION_SEEDS for name in plan["extension_models"]]
    if plan["rescue_model"]:
        tasks += [("rescue", "main", plan["rescue_model"], seed, True) for seed in EXPLORATION_SEEDS]
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "smoke", "freeze", "exploration", "select", "validation", "conditional", "all", "worker"))
    parser.add_argument("--stage")
    parser.add_argument("--case", default="main")
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--affinity", type=int)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()
    if args.action == "worker":
        from .r7_runtime import set_cpu_affinity
        from .train_r8_coupling_recovery import train_run
        set_cpu_affinity(args.affinity)
        reference = json.loads((RAW/"reference_validation.json").read_text(encoding="utf-8"))
        j = reference["cases"][args.case]["selected_half_width"]
        cfg = config_for(args.stage, args.case, j)
        train_run(cfg, args.model, args.seed,
                  run_directory(args.stage, args.model, args.seed, case=args.case, balanced=args.balanced),
                  stage=args.stage, balanced=args.balanced)
    elif args.action == "prepare":
        from .r8_reference import prepare_references
        print(f"Protected historical files: {len(capture_history()['files'])}", flush=True)
        print(json.dumps(prepare_references(), ensure_ascii=False), flush=True)
    elif args.action == "freeze":
        print(freeze(), flush=True)
    elif args.action == "select":
        from .r8_decision import select_candidates
        print(json.dumps(select_candidates(), ensure_ascii=False), flush=True)
    elif args.action == "conditional":
        run_tasks(conditional_tasks())
    elif args.action == "all":
        # Preparation, smoke, unit tests and first freeze must already exist.
        freeze()
        run_tasks(tasks_for("exploration"))
        from .r8_decision import select_candidates, validation_decision
        print(json.dumps(select_candidates(), ensure_ascii=False), flush=True)
        run_tasks(tasks_for("validation"))
        print(json.dumps(validation_decision(), ensure_ascii=False), flush=True)
        run_tasks(conditional_tasks())
        print("ALL prescribed training stages completed; postprocessing pending", flush=True)
    else:
        records = run_tasks(tasks_for(args.action))
        if args.action == "smoke":
            write_json(RAW/"smoke_validation.json", dict(status="PASS", engineering_only=True,
                runs=len(records), budgets=[r["steps"] for r in records],
                max_initial_error=max(r["initial_condition_max_abs_error"] for r in records),
                no_nan=True, no_hyperparameter_selection=True))


if __name__ == "__main__":
    main()
