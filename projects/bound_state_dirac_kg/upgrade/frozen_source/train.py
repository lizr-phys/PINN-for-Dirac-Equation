"""Train using only the equation, the potential and structural constraints."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
import time
import numpy as np
import pandas as pd
import torch
from .physics import ROOT,config,save_json
from .model import StructuredPINN,StructuredLoss
from .optimizer import physics_vector,polish
from ..model import BoundPINN,Loss

def train(kind,case_id="base",state=0,seed=0,variant="derivative",group="pilot",overrides=None):
    cfg=config()
    if overrides: cfg.update(overrides)
    case=cfg["cases"][case_id].copy()
    if "half_length_override" in cfg: case["half_length"]=cfg["half_length_override"]
    reference=ROOT/"results"/"reference"/case_id/"summary.json"
    if not reference.exists():
        raise RuntimeError("The physical case must be verified before training.")
    verification=json.loads(reference.read_text(encoding="utf-8"))
    if not verification["verified"] or any(verification["case"][k]!=case[k] for k in ["mass","depth","width"]):
        raise RuntimeError("Reference verification does not match these physical parameters.")
    if f"{kind}_{state}" not in verification["states"]:
        raise RuntimeError("The requested state has not been independently verified.")
    if case["half_length"]>verification["case"]["reference_length"]+8:
        raise RuntimeError("Extend the reference before enlarging the training domain further.")
    if state not in case["states"]:
        raise ValueError("This state has not been verified in the case.")
    out=ROOT/"results"/group/case_id/kind/f"state_{state}"/variant/f"seed_{seed}"
    meta=dict(kind=kind,case_id=case_id,case=case,state=state,seed=seed,variant=variant,config=cfg)
    if (out/"training.json").exists():
        cached=json.loads((out/"config.json").read_text(encoding="utf-8"))
        if any(cached[k]!=meta[k] for k in meta):
            raise RuntimeError(f"Cached configuration changed; choose a new group: {out}")
        if not all((out/n).exists() for n in ("model.pt","history.csv")):
            raise RuntimeError(f"Incomplete cached run: {out}")
        print(f"SKIP {out}",flush=True)
        return out
    out.mkdir(parents=True,exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(cfg["cpu_threads"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    fraction=cfg["initial_energy_fractions"][seed%len(cfg["initial_energy_fractions"])]
    if variant=="legacy":
        if state!=0: raise ValueError("Legacy ansatz has no excited-state support.")
        legacy_cfg={**cfg,**case,"energy_interval":[0.,case["mass"]]}
        model=BoundPINN(kind,legacy_cfg)
        with torch.no_grad(): model.raw_energy.copy_(torch.logit(torch.tensor(fraction)))
        loss_fn=Loss(model,legacy_cfg)
    else:
        model=StructuredPINN(kind,cfg,case,state,variant,fraction)
        lower_state=None
        if state==2:
            lower_out=ROOT/"results"/group/case_id/kind/"state_0"/"derivative"/f"seed_{seed}"
            if not (lower_out/"training.json").exists():
                raise RuntimeError("Train the same-seed ground PINN before its second excited state.")
            lm=json.loads((lower_out/"config.json").read_text(encoding="utf-8"))
            lower=StructuredPINN(kind,lm["config"],lm["case"],0,"derivative",fraction)
            lower.load_state_dict(torch.load(lower_out/"model.pt",weights_only=True))
            for p in lower.parameters(): p.requires_grad_(False)
            lower_state=(lower(model.q.detach().requires_grad_(True)).detach(),lower.energy.detach())
            meta["lower_pinn"]={"path":str(lower_out.relative_to(ROOT)).replace("\\","/"),
                                "sha256":hashlib.sha256((lower_out/"model.pt").read_bytes()).hexdigest(),
                                "constraint":"Dirac L2 inner product; KG (E_n+E_m-2V) charge inner product"}
        loss_fn=StructuredLoss(model,cfg,lower_state)
    meta["environment"]=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,
                              numpy=np.__version__,platform=platform.platform(),cpu=platform.processor())
    meta["source_sha256"]={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ["train.py","model.py","optimizer.py","physics.py","config.json"]}
    meta["initialization"]="Xavier normal, zero biases, output weights x0.1; specified E/m grid, independent network seeds"
    save_json(out/"config.json",meta)
    np.savez_compressed(out/"training_points.npz",x=loss_fn.x.numpy(),w=loss_fn.w.numpy(),
                        q=model.q.numpy() if variant!="legacy" else loss_fn.q.numpy(),
                        qw=model.qw.numpy() if variant!="legacy" else loss_fn.qw.numpy())
    initial=float(model.energy.detach())
    history=[]
    start=time.perf_counter()
    evaluations=0
    def record(stage,step,total,parts,checkpoint=False):
        history.append(dict(stage=stage,step=step,evaluations=evaluations,seconds=time.perf_counter()-start,
                            energy=float(model.energy.detach()),loss=float(total.detach()),
                            **{k:float(v.detach()) for k,v in parts.items()}))
        if checkpoint:
            path=out/"checkpoints"/f"eval_{evaluations}.pt"
            path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(),path)
    adam=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
    for step in range(cfg["adam_steps"]):
        adam.zero_grad()
        total,parts=loss_fn()
        evaluations+=1
        if step%100==0: record("adam",step,total,parts)
        total.backward()
        adam.step()
    calls=0
    opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=cfg["lbfgs_chunk"],max_eval=cfg["lbfgs_chunk"]*2,
                         history_size=80,tolerance_grad=1e-12,tolerance_change=1e-15,line_search_fn="strong_wolfe")
    def closure():
        nonlocal calls,evaluations
        opt.zero_grad()
        total,parts=loss_fn()
        (total*cfg["objective_scale"]).backward()
        evaluations+=1
        calls+=1
        if calls%20==0: record("lbfgs",calls,total,parts)
        return total*cfg["objective_scale"]
    stop="iteration_budget"
    iterations=0
    for block in range((cfg["lbfgs_steps"]+cfg["lbfgs_chunk"]-1)//cfg["lbfgs_chunk"]):
        opt.step(closure)
        total,parts=loss_fn()
        iterations=opt.state[model.raw_energy].get("n_iter",0)
        record("checkpoint",iterations,total,parts,True)
        print(f"{group} {case_id}/{kind}/{state}/{variant} seed={seed} LBFGS {iterations}: R2={parts['pde'].item():.3e} E={model.energy.item():.10f}",flush=True)
        norm_error=abs(float(parts["norm"].detach())-1)
        if variant!="legacy":
            boundary=float(torch.linalg.norm(model.boundary_residual()).detach())
        else:
            boundary=float(torch.sqrt(parts["boundary"]).detach())
        if float(total.detach())<=cfg["residual_stop_squared"] and norm_error<=cfg["normalization_stop"] and boundary<=cfg["boundary_stop"]:
            stop="physics_tolerance"
            break
    polish_result=None
    if cfg.get("gauss_newton_steps",0)>0 and stop!="physics_tolerance":
        previous_evaluations=evaluations
        def callback(step,polish_calls):
            nonlocal evaluations
            evaluations=previous_evaluations+polish_calls
            total,parts=loss_fn()
            record("gauss_newton",step,total,parts,True)
            if step%5==0:
                print(f"{group} {case_id}/{kind}/{state}/{variant} seed={seed} GN {step}: loss={total.item():.3e}",flush=True)
        polish_result=polish(model,lambda:physics_vector(loss_fn,variant),cfg["gauss_newton_steps"],
                             cfg["residual_stop_squared"],callback,cfg.get("jacobian_batch_size",16))
        evaluations=previous_evaluations+polish_result["evaluations"]
        if polish_result["stop_reason"]=="physics_tolerance": stop="physics_tolerance"
    total,parts=loss_fn()
    if float(total.detach())<=cfg["residual_stop_squared"] and abs(float(parts["norm"].detach())-1)<=cfg["normalization_stop"]:
        btest=float(torch.linalg.norm(model.boundary_residual()).detach()) if variant!="legacy" else float(torch.sqrt(parts["boundary"]).detach())
        if btest<=cfg["boundary_stop"]: stop="physics_tolerance"
    record("final",iterations,total,parts)
    seconds=time.perf_counter()-start
    torch.save(model.state_dict(),out/"model.pt")
    pd.DataFrame(history).to_csv(out/"history.csv",index=False)
    save_json(out/"training.json",dict(energy=float(model.energy.detach()),initial_energy=initial,seconds=seconds,
              parameters=sum(p.numel() for p in model.parameters()),loss=float(total.detach()),
              parts={k:float(v.detach()) for k,v in parts.items()},evaluations=evaluations,
              lbfgs_closure_calls=calls,lbfgs_iterations=iterations,stop_reason=stop,
              gauss_newton=polish_result,
              finite=bool(torch.isfinite(total))))
    print(f"DONE {case_id}/{kind}/{state}/{variant} seed={seed}: E={model.energy.item():.12f}, R={np.sqrt(parts['pde'].item()):.3e}, {seconds:.1f}s {stop}",flush=True)
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--kind",choices=["dirac","kg"],required=True)
    p.add_argument("--case",default="base",choices=list(config()["cases"]))
    p.add_argument("--state",type=int,default=0)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--variant",choices=["derivative","independent","legacy"],default="derivative")
    p.add_argument("--group",default="pilot")
    a=p.parse_args()
    train(a.kind,a.case,a.state,a.seed,a.variant,a.group)
