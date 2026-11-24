"""Fixed 15-run process queue: no retries, no outcome-based scheduling."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from .periodic_potential_config import PERIODIC_POTENTIAL_MODELS, PERIODIC_POTENTIAL_SEEDS
from .train_periodic_potential_coupled_ltpb import ROOT, RAW, load_formal_config, write_json

LOGS = ROOT/"results/logs/periodic_potential_coupled_ltpb"


def worker(model, seed):
    path = LOGS/f"{model}__seed_{seed}.log"
    if path.exists():
        raise FileExistsError(f"Refusing a repeated formal attempt: {path}")
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    args = [sys.executable, "-m", "src.run_periodic_potential_coupled_ltpb", "--mode", "full", "--model", model, "--seed", str(seed)]
    with path.open("w", encoding="utf-8") as log:
        process = subprocess.run(args, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    result = {"model": model, "seed": seed, "returncode": process.returncode,
              "log": path.relative_to(ROOT).as_posix()}
    print(json.dumps(result), flush=True)
    return result


def main():
    from .freeze_periodic_potential_coupled_ltpb import verify_freeze
    load_formal_config()
    digest = verify_freeze()
    LOGS.mkdir(parents=True, exist_ok=True)
    dispatch = LOGS/"dispatch.json"
    if dispatch.exists():
        raise FileExistsError("Formal queue already dispatched; inspect evidence, do not retrain")
    jobs = [(model, seed) for model in PERIODIC_POTENTIAL_MODELS for seed in PERIODIC_POTENTIAL_SEEDS]
    write_json(dispatch, {"started_utc": datetime.now(timezone.utc).isoformat(), "workers": 5,
                          "cpu_threads_per_run": 1, "retries": 0, "freeze_manifest_sha256": digest,
                          "jobs": [{"model": m, "seed": s} for m, s in jobs]})
    results = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(worker, model, seed) for model, seed in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    write_json(LOGS/"completion.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "runs": results})
    if any(row["returncode"] for row in results):
        raise SystemExit("Formal queue has failures; all attempts retained")


if __name__ == "__main__":
    main()
