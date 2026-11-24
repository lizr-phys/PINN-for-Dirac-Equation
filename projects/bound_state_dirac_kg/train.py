from __future__ import annotations
import argparse
import hashlib
import platform
import sys
import time
import numpy as np
import pandas as pd
import scipy
import torch
from .physics import ROOT,config,save_json
from .model import BoundPINN,Loss


def train(kind,seed,tag="formal",overrides=None):
    cfg=config()
    if overrides: cfg.update(overrides)
    if not cfg.get("parameters_frozen_after_reference"):
        raise RuntimeError("Run reference verification before training.")
    if (cfg["mass"],cfg["depth"],cfg["width"]) != (1.0,0.5,2.0):
        raise ValueError("Changing the physical model requires a new reference study.")
    out=ROOT/"results"/tag/kind/f"seed_{seed}"
    if (out/"training.json").exists():
        import json
        prior=json.loads((out/"config.json").read_text(encoding="utf-8"))
        if prior["config"] != cfg:
            raise RuntimeError("Cached run has a different configuration; use a new --tag.")
        for name in ["model.pt","history.csv"]:
            if not (out/name).is_file(): raise RuntimeError(f"Incomplete run: {out/name}")
        print(f"SKIP {out}",flush=True); return
    out.mkdir(parents=True,exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(cfg["cpu_threads"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model=BoundPINN(kind,cfg)
    loss_fn=Loss(model,cfg)
    np.savez_compressed(out/"training_points.npz",collocation_x=loss_fn.x.numpy(),
                        collocation_weights=loss_fn.w.numpy(),normalization_x=loss_fn.q.numpy(),
                        normalization_weights=loss_fn.qw.numpy())
    initial_energy=float(model.energy.detach())
    save_json(out/"config.json",dict(kind=kind,seed=seed,config=cfg,
        environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,
                         numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
                         cpu=platform.processor(),cuda=torch.cuda.is_available()),
        initialization="Xavier normal, output weights scaled by 0.1, biases zero; raw energy uniform(-1,1)",
        collocation="fixed half-domain Gauss-Legendre quadrature; independent dense off-grid evaluation",
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                       for name in ["model.py","train.py","physics.py","config.json"]}))
    history=[]; started=time.perf_counter(); evaluations=0

    def record(stage,step,total,parts):
        history.append(dict(stage=stage,step=step,evaluations=evaluations,
                            seconds=time.perf_counter()-started,energy=float(model.energy.detach()),
                            loss=float(total.detach()),**{k:float(v.detach()) for k,v in parts.items()}))

    adam=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
    for step in range(cfg["adam_steps"]+1):
        adam.zero_grad(); total,parts=loss_fn(); evaluations+=1
        if step%50==0: record("adam",step,total,parts)
        if step%500==0:
            print(f"{kind} seed={seed} Adam {step}: loss={total.item():.3e} E={model.energy.item():.10f}",flush=True)
        if step<cfg["adam_steps"]:
            total.backward(); adam.step()
    opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=cfg["lbfgs_steps"],
                         max_eval=cfg["lbfgs_steps"]*2,history_size=80,
                         tolerance_grad=1e-12,tolerance_change=1e-15,line_search_fn="strong_wolfe")
    calls=0
    objective_scale=cfg.get("lbfgs_objective_scale",1.0)
    def closure():
        nonlocal evaluations,calls
        opt.zero_grad(); total,parts=loss_fn(); (total*objective_scale).backward()
        evaluations+=1; calls+=1
        if calls%10==0: record("lbfgs",calls,total,parts)
        if calls%200==0:
            print(f"{kind} seed={seed} LBFGS eval {calls}: loss={total.item():.3e} E={model.energy.item():.10f}",flush=True)
        return total*objective_scale
    opt.step(closure)
    total,parts=loss_fn(); record("final",calls,total,parts)
    seconds=time.perf_counter()-started
    torch.save(model.state_dict(),out/"model.pt")
    pd.DataFrame(history).to_csv(out/"history.csv",index=False)
    save_json(out/"training.json",dict(kind=kind,seed=seed,energy=float(model.energy.detach()),
        initial_energy=initial_energy,seconds=seconds,parameters=sum(p.numel() for p in model.parameters()),
        loss=float(total.detach()),parts={k:float(v.detach()) for k,v in parts.items()},
        evaluations=evaluations,lbfgs_closure_calls=calls,
        lbfgs_iterations=opt.state[model.raw_energy].get("n_iter",0)))
    print(f"DONE {kind} seed={seed}: E={model.energy.item():.12f}, loss={total.item():.3e}, {seconds:.1f}s",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--kind",choices=["dirac","kg"],required=True)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--tag",default="formal")
    p.add_argument("--adam",type=int)
    p.add_argument("--lbfgs",type=int)
    a=p.parse_args()
    overrides={}
    if a.adam is not None: overrides["adam_steps"]=a.adam
    if a.lbfgs is not None: overrides["lbfgs_steps"]=a.lbfgs
    train(a.kind,a.seed,a.tag,overrides)
