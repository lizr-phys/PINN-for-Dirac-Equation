from __future__ import annotations
import argparse
import json
import time
from datetime import datetime, timezone
from .r11_config import RAW, LOGS, GROUPS, SEEDS, run_directory

def snapshot():
    done=[]; active=[]; checkpoint_count=0
    for group in GROUPS:
        for seed in SEEDS:
            directory=run_directory(group,seed)
            if (directory/"metrics.json").exists():
                done.append(f"{group}/seed{seed}")
            checkpoint_count += len(list((directory/"snapshots").glob("coupled_*/metrics.json")))
    plan_path=RAW/"runtime_plan.json"
    plan=json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {}
    for slot in plan.get("slots",[]):
        path=LOGS/f"slot_{slot['slot']}_seed{slot['seed']}.log"
        lines=path.read_text(encoding="utf-8",errors="replace").splitlines() if path.exists() else []
        active.append(dict(seed=slot["seed"],mask=slot["mask"],last_line=lines[-1] if lines else "starting"))
    return dict(utc=datetime.now(timezone.utc).isoformat(),runtime_status=plan.get("status"),
        complete_runs=len(done),total_runs=25,completed_fixed_checkpoints=checkpoint_count,
        total_fixed_checkpoints=115,slots=active)

if __name__=="__main__":
    from .r7_runtime import set_cpu_affinity
    set_cpu_affinity(128)
    parser=argparse.ArgumentParser(); parser.add_argument("--watch",action="store_true")
    args=parser.parse_args()
    while True:
        value=snapshot(); print(json.dumps(value,ensure_ascii=False),flush=True)
        if not args.watch or value["runtime_status"] in ("COMPLETE","FAIL"):
            break
        time.sleep(45)
