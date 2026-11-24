from __future__ import annotations
import argparse
import hashlib
import json
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
from .physics import ROOT,config,save_json

def freeze():
    names=["config.json","physics.py","model.py","optimizer.py","train.py","training_core.py","run.py","campaign.py"]
    snapshot={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    version=config()["version"]
    path=ROOT/f"frozen_sources_v{version}.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=snapshot:
            raise RuntimeError("Frozen scientific source/configuration changed. Preserve this campaign and use another version/group.")
    else:
        save_json(path,snapshot)
        out=ROOT/f"frozen_source_v{version}";out.mkdir(exist_ok=True)
        for n in names: (out/n).write_bytes((ROOT/n).read_bytes())

def train_campaign():
    from .train import train
    from .evaluate import evaluate_run
    freeze()
    from .campaign import plan,run_path
    rows=[]
    for item in plan():
        if item["protocol_version"]==6:
            out=run_path(item)
            print(f"RETAIN VERIFIED V6 {out}",flush=True)
        else:
            out=train(item["kind"],item["case"],item["state"],item["seed"],item["variant"],
                      item["group"],item["overrides"])
        rows.append(evaluate_run(out))
    return rows

def main(stage):
    if stage in ["reference","all"]:
        from .reference import main as reference
        reference()
    if stage in ["train","all"]: train_campaign()
    if stage in ["evaluate","all"]:
        from .campaign import evaluate_campaign
        evaluate_campaign()
    if stage in ["diagnostics","all"]:
        from .diagnostics import main as diagnostics
        diagnostics()
    if stage in ["figures","all"]:
        from .figures import main as figures
        figures()
    if stage in ["report","all"]:
        from .report import main as report
        report()
    if stage in ["audit","all"]:
        from .audit import main as audit
        audit()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["reference","train","evaluate","diagnostics","figures","report","audit","all"],default="all")
    main(p.parse_args().stage)
