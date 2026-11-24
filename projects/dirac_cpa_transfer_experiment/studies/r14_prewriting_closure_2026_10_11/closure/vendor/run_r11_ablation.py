from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from .r11_config import ROOT, RAW, LOGS, GROUPS, SEEDS, run_directory, protocol
from .freeze_r7_phase_release import write_json

def launch():
    from .freeze_r11_ablation import verify_freeze
    from .r7_runtime import physical_core_masks, set_cpu_affinity
    from .run_r9_wavepacket import validate_masks
    frozen = verify_freeze()
    available = physical_core_masks()
    masks = available[:5]
    if len(masks) != 5 or len(available) < 6:
        raise RuntimeError("Five worker cores plus a separate control core required")
    validate_masks(masks+[available[-1]])
    set_cpu_affinity(available[-1])
    LOGS.mkdir(parents=True, exist_ok=True)
    plan = dict(status="RUNNING", masks=masks, control_mask=available[-1],
        all_slots_pairwise_disjoint=True, frozen_protocol_sha256=frozen,
        threads_per_worker=1, schedule="fixed five-seed Latin rotation of group order", slots=[])
    processes = []
    for slot, (seed, mask) in enumerate(zip(SEEDS, masks)):
        order = list(GROUPS[slot:]+GROUPS[:slot])
        log = LOGS/f"slot_{slot}_seed{seed}.log"
        stream = log.open("a", encoding="utf-8")
        command = [sys.executable, "-X", "utf8", "-u", "-m", "src.run_r11_ablation",
                   "worker", "--seed", str(seed), "--affinity", str(mask), "--slot", str(slot)]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        plan["slots"].append(dict(slot=slot, seed=seed, mask=mask, group_order=order, pid=child.pid,
            started_utc=datetime.now(timezone.utc).isoformat(), log=log.relative_to(ROOT).as_posix()))
        processes.append((child, stream, seed))
    write_json(RAW/"runtime_plan.json", plan)
    while any(p.poll() is None for p,_,_ in processes):
        time.sleep(.25)
    failures = []
    for p, stream, seed in processes:
        stream.close()
        if p.returncode:
            failures.append(dict(seed=seed, exit_code=p.returncode))
    plan.update(status="FAIL" if failures else "COMPLETE", failures=failures,
                ended_utc=datetime.now(timezone.utc).isoformat())
    write_json(RAW/"runtime_plan.json", plan)
    if failures:
        raise RuntimeError("Failed runs preserved: "+str(failures))
    print("All 25 independent R11 runs complete", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "smoke", "freeze", "formal", "worker"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--affinity", type=int)
    parser.add_argument("--slot", type=int)
    args = parser.parse_args()
    if args.action == "worker":
        from .r7_runtime import set_cpu_affinity
        from .r9_runtime import actual_affinity_mask
        set_cpu_affinity(args.affinity)
        os.environ["R7_CPU_AFFINITY_MASK"] = str(args.affinity)
        if actual_affinity_mask() != args.affinity:
            raise RuntimeError("Actual CPU allocation differs from disjoint plan")
        from .train_r11_ablation import train_run
        order = GROUPS[args.slot:]+GROUPS[:args.slot]
        failures = []
        for group in order:
            try:
                train_run(group, args.seed, run_directory(group, args.seed))
            except Exception as error:
                failures.append(dict(group=group, error_type=type(error).__name__, message=str(error)))
                print("FAILED PRESERVED "+group+": "+str(error), flush=True)
        if failures:
            write_json(RAW/f"worker_seed{args.seed}_failures.json", failures)
            raise RuntimeError("Failed runs retained; other assigned groups still attempted")
    elif args.action == "prepare":
        from .r7_runtime import set_cpu_affinity
        set_cpu_affinity(128)
        from .r11_reference import prepare
        from .freeze_r11_ablation import preserve_sources
        write_json(RAW/"protocol.json", protocol())
        print(prepare(), flush=True)
        print("Prior files protected: "+str(len(preserve_sources()["files"])), flush=True)
    elif args.action == "smoke":
        from .r7_runtime import set_cpu_affinity
        set_cpu_affinity(128)
        from .train_r11_ablation import train_run
        records = []
        for group in GROUPS:
            result = train_run(group, 400, run_directory(group, 400, smoke=True), smoke=True)
            records.append(dict(group=group, all_finite=result["all_finite"],
                hard_initial_error=result["initial_condition_max_abs_error"],
                all_artifacts=all((run_directory(group,400,smoke=True)/p).is_file()
                    for p in ("config.json","metrics.json","loss_history.csv","time_series.csv",
                              "dense_grid.npz","model_state.pt","recovery_state.pt"))))
        write_json(RAW/"smoke_validation.json", dict(status="PASS" if all(r["all_finite"] and
            r["all_artifacts"] and r["hard_initial_error"] < 1e-14 for r in records) else "FAIL",
            records=records, accuracy_not_used=True, dimensions_initial_gradients_nan_disk_only=True))
    elif args.action == "freeze":
        from .freeze_r11_ablation import freeze
        print(freeze(), flush=True)
    else:
        launch()

if __name__ == "__main__":
    main()
