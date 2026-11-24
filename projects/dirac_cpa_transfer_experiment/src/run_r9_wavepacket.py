from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time
from .r9_config import ROOT, RAW, LOGS, MODELS, SEEDS, run_directory
from .freeze_r7_phase_release import write_json


def validate_masks(masks):
    if not masks or any(m <= 0 or m & (m-1) for m in masks):
        raise RuntimeError("Each CPU slot needs one physical-core representative")
    if any(a & b for i, a in enumerate(masks) for b in masks[i+1:]):
        raise RuntimeError("CPU core allocations overlap")
    return True


def launch():
    from .freeze_r9_wavepacket import verify_freeze
    from .r7_runtime import physical_core_masks
    manifest = verify_freeze()
    masks = physical_core_masks()[:5]
    if len(masks) != 5:
        raise RuntimeError("Frozen R9 runtime requires five distinct physical cores")
    expected_masks = json.loads((RAW/"protocol.json").read_text(encoding="utf-8"))["cpu_masks"]
    if masks != expected_masks:
        raise RuntimeError("CPU topology differs from the frozen execution plan")
    validate_masks(masks)
    LOGS.mkdir(parents=True, exist_ok=True)
    processes = []
    plan = dict(status="RUNNING", freeze_manifest_sha256=manifest, masks=masks, threads_per_worker=1,
                all_slots_pairwise_disjoint=True, slot_mapping="slot only; never seed plus slot", slots=[])
    # One worker owns one fixed core and one seed, completing all three models in order.
    for slot, (seed, mask) in enumerate(zip(SEEDS, masks)):
        label = f"slot_{slot}_seed{seed}"
        path = LOGS/f"{label}.log"
        stream = path.open("a", encoding="utf-8")
        command = [sys.executable, "-X", "utf8", "-u", "-m", "src.run_r9_wavepacket", "worker",
                   "--seed", str(seed), "--affinity", str(mask)]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        processes.append((proc, stream, seed))
        plan["slots"].append(dict(slot=slot, seed=seed, mask=mask, pid=proc.pid, command=command,
                                  started_utc=datetime.now(timezone.utc).isoformat(), log=str(path.relative_to(ROOT))))
    write_json(RAW/"runtime_plan.json", plan)
    failures = []
    while any(proc.poll() is None for proc, _, _ in processes):
        time.sleep(.25)
    for proc, stream, seed in processes:
        stream.close()
        if proc.returncode:
            failures.append(dict(seed=seed, exit_code=proc.returncode))
    plan.update(status="FAIL" if failures else "COMPLETE", failures=failures,
                ended_utc=datetime.now(timezone.utc).isoformat())
    write_json(RAW/"runtime_plan.json", plan)
    if failures:
        raise RuntimeError(f"Failed runs preserved; only exact recovery is allowed: {failures}")
    print("All 15 fixed-budget R9 runs complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "smoke", "freeze", "formal", "worker"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--affinity", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        from .r9_reference import prepare
        from .freeze_r9_wavepacket import capture_history
        print("Historical files protected:", len(capture_history()["files"]), flush=True)
        print(json.dumps(prepare(), indent=2), flush=True)
    elif args.action == "freeze":
        from .freeze_r9_wavepacket import freeze
        print(freeze(), flush=True)
    elif args.action == "formal":
        launch()
    elif args.action == "smoke":
        from .freeze_r9_wavepacket import selected_config
        from .train_r9_wavepacket import train_run
        cfg = replace(selected_config(), adam_steps=4, carrier_steps=2, history_every=1, recovery_every=2,
                      test_t_points=3, residual_time_points=21)
        records = [train_run(cfg, name, 299, run_directory(name, 299, smoke=True)/"chain_rule_kernel", smoke=True) for name in MODELS]
        write_json(RAW/"smoke_validation.json", dict(status="PASS", engineering_only=True, runs=3,
            dimensions_checked=True, initial_data_checked=True, finite_gradients=True, all_finite=True,
            checkpoint_write_checked=True, accuracy_not_used=True, no_hyperparameter_selection=True,
            total_smoke_optimizer_steps=sum(r["steps"] for r in records)))
    elif args.action == "worker":
        from .r7_runtime import set_cpu_affinity
        from .freeze_r9_wavepacket import selected_config
        from .train_r9_wavepacket import train_run
        set_cpu_affinity(args.affinity)
        # Independently verify the actual OS affinity after applying the requested mask.
        from .r9_runtime import actual_affinity_mask
        if actual_affinity_mask() != args.affinity:
            raise RuntimeError("Actual process affinity differs from registered allocation")
        for name in MODELS:
            train_run(selected_config(), name, args.seed, run_directory(name, args.seed))


if __name__ == "__main__":
    main()
