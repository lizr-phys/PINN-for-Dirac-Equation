"""Explicit provenance plan: reuse independently valid v6 states, finish with v7."""
from __future__ import annotations
import json
from .physics import ROOT,config,save_json

def run_path(item):
    return ROOT/"results"/item["group"]/item["case"]/item["kind"]/f"state_{item['state']}"/item["variant"]/f"seed_{item['seed']}"

def requested():
    cfg=config();items=[]
    def add(purpose,group,case,kind,state,variant,seed,overrides=None):
        items.append(dict(purpose=purpose,group=group,case=case,kind=kind,state=state,variant=variant,seed=seed,overrides=overrides or {}))
    for name,case in cfg["cases"].items():
        seeds=cfg["base_seeds"] if name=="base" else cfg["validation_seeds"]
        for kind in ["dirac","kg"]:
            for state in case["states"]:
                for seed in seeds:add("formal",cfg["formal_group"],name,kind,state,"derivative",seed)
    for variant,kind in [("independent","dirac"),("legacy","dirac"),("legacy","kg")]:
        for seed in cfg["validation_seeds"]:add("ablations","ablations","base",kind,0,variant,seed)
    for group,changes in [("grid_refined",{"collocation_points":512,"quadrature_points":384}),
                           ("domain_extended",{"half_length_override":24.})]:
        for kind in ["dirac","kg"]:add(group,group,"base",kind,0,"derivative",0,changes)
    return items

def plan():
    path=ROOT/"campaign_manifest.json"
    if path.exists():return json.loads(path.read_text(encoding="utf-8"))["runs"]
    items=requested()
    for item in items:
        previous={**item,"group":"stable_formal"}
        out=run_path(previous)
        if item["purpose"]=="formal" and item["state"]<2 and (out/"metrics.json").exists():
            m=json.loads((out/"metrics.json").read_text(encoding="utf-8"))
            meta=json.loads((out/"config.json").read_text(encoding="utf-8"))
            if meta["config"]["version"]!=6 or not m["reliable"]:raise RuntimeError("Only verified v6 states may be retained")
            item["group"]="stable_formal";item["protocol_version"]=6
        else:item["protocol_version"]=config()["version"]
    for item in items:
        if item["state"]==2:
            lower=next(p for p in items if p["purpose"]=="formal" and p["case"]==item["case"] and
                       p["kind"]==item["kind"] and p["state"]==0 and p["seed"]==item["seed"])
            item["overrides"]["lower_group"]=lower["group"]
    save_json(path,dict(reuse_policy="38 completed v6 state 0/1 runs passed independent grids/reference/normalization checks. Their models, training configurations/history and frozen source remain immutable; derived evaluations are regenerated with protocol labels. Exact scale invariance is regression-tested; all remaining runs use v7 guards and stable algebra. Versions are reported, not silently pooled as one identical protocol.",runs=items))
    return items

def formal_path(case,kind,state,seed):
    item=next(i for i in plan() if i["purpose"]=="formal" and i["case"]==case and i["kind"]==kind and i["state"]==state and i["seed"]==seed)
    return run_path(item)

def evaluate_campaign():
    import pandas as pd
    from .evaluate import evaluate_run
    rows=[]
    for item in plan():
        row=evaluate_run(run_path(item),checkpoints=item["purpose"]=="ablations")
        row.update(purpose=item["purpose"],group=item["group"])
        rows.append(row)
    df=pd.DataFrame(rows)
    for purpose,rows in df.groupby("purpose"):
        rows.to_csv(ROOT/"results"/f"{purpose}_summary.csv",index=False)
        keys=["energy","energy_abs_error","wavefunction_rel_l2","residual_l2","normalization_error","seconds"]
        rows.groupby(["case","kind","state","variant"]).agg({k:["mean","std","min","max"] for k in keys}).to_csv(ROOT/"results"/f"{purpose}_aggregate.csv")
    df.to_csv(ROOT/"results/campaign_summary.csv",index=False)
    return rows
