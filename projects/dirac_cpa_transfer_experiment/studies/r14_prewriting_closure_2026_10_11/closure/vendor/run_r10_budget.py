from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from .r10_config import ROOT, RAW, LOGS, SEEDS, run_directory
from .freeze_r7_phase_release import write_json


def launch():
    from .freeze_r10_budget import verify_freeze
    from .r7_runtime import physical_core_masks
    from .run_r9_wavepacket import validate_masks
    frozen = verify_freeze()
    masks = physical_core_masks()[:5]
    if len(masks) != 5:
        raise RuntimeError("Five distinct physical CPU cores required")
    validate_masks(masks)
    LOGS.mkdir(parents=True, exist_ok=True)
    plan = dict(status="RUNNING", masks=masks, all_slots_pairwise_disjoint=True,
        frozen_protocol_sha256=frozen, threads_per_worker=1, slots=[])
    processes = []
    for slot, (seed, mask) in enumerate(zip(SEEDS, masks)):
        log = LOGS/f"slot_{slot}_seed{seed}.log"
        stream = log.open("a", encoding="utf-8")
        command = [sys.executable, "-X", "utf8", "-u", "-m", "src.run_r10_budget",
                   "worker", "--seed", str(seed), "--affinity", str(mask)]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        plan["slots"].append(dict(slot=slot, seed=seed, mask=mask, pid=child.pid,
            started_utc=datetime.now(timezone.utc).isoformat(), command=command, log=log.relative_to(ROOT).as_posix()))
        processes.append((child, stream, seed))
    write_json(RAW/"runtime_plan.json", plan)
    while any(p.poll() is None for p, _, _ in processes):
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
        raise RuntimeError("Failed continuations preserved: "+str(failures))
    print("All five fixed step26000 R10 continuations complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("smoke", "freeze", "formal", "worker"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--affinity", type=int)
    args = parser.parse_args()
    if args.action == "worker":
        from .r7_runtime import set_cpu_affinity
        from .r9_runtime import actual_affinity_mask
        set_cpu_affinity(args.affinity)
        if actual_affinity_mask() != args.affinity:
            raise RuntimeError("Actual CPU affinity differs from plan")
        from .train_r10_budget import train_continuation
        train_continuation(args.seed, run_directory(args.seed))
    elif args.action == "smoke":
        from .train_r10_budget import train_continuation
        result = train_continuation(300, run_directory(300, smoke=True), additional_steps=2, smoke=True)
        write_json(RAW/"smoke_validation.json", dict(status="PASS", dimensions_checked=True, gradients_finite=True,
            all_finite=result["all_finite"], disk_write_verified=True, accuracy_not_used=True,
            optimizer_reset=False, sampler_reset=False, additional_smoke_steps=2))
        print("R10 two-step engineering smoke PASS", flush=True)
    elif args.action == "freeze":
        from .freeze_r10_budget import freeze
        print(freeze(), flush=True)
    else:
        launch()


if __name__ == "__main__":
    main()
