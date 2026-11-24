"""Independent grids and references are accessed only after training."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .physics import ROOT,config,save_json,potential,reference_on,aligned_l2,count_nodes
from .model import StructuredPINN,grad
from ..model import BoundPINN

def load_model(out,checkpoint=None):
    out=Path(out)
    meta=json.loads((out/"config.json").read_text(encoding="utf-8"))
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    if meta["variant"]=="legacy":
        model=BoundPINN(meta["kind"],{**meta["config"],**meta["case"],"energy_interval":[0.,meta["case"]["mass"]]})
    else:
        f=meta["config"]["initial_energy_fractions"][meta["seed"]%len(meta["config"]["initial_energy_fractions"])]
        model=StructuredPINN(meta["kind"],meta["config"],meta["case"],meta["state"],meta["variant"],f)
    model.load_state_dict(torch.load(checkpoint or out/"model.pt",weights_only=True))
    model.eval()
    return model,meta

def predictions(model,variant,x):
    scale=model.normalizer().detach() if variant!="legacy" else None
    ys,rs,dys=[],[],[]
    for chunk in np.array_split(x,max(1,len(x)//512)):
        xt=torch.tensor(chunk[:,None],requires_grad=True)
        y=model(xt,scale) if variant!="legacy" else model(xt)
        r=model.residual(xt,scale) if variant!="legacy" else model.residual(xt)
        dy=torch.cat([grad(y[:,j:j+1],xt) for j in range(y.shape[1])],dim=1)
        ys.append(y.detach().numpy());rs.append(r.detach().numpy());dys.append(dy.detach().numpy())
    return np.concatenate(ys),np.concatenate(rs),np.concatenate(dys)

def evaluate_run(out,checkpoints=False):
    out=Path(out)
    started=time.perf_counter()
    model,meta=load_model(out)
    t=json.loads((out/"training.json").read_text(encoding="utf-8"))
    case,kind,state=meta["case"],meta["kind"],meta["state"]
    L=case["half_length"]
    refdir=ROOT/"results"/"reference"/meta["case_id"]
    records=[]
    for n in [2001,4001,8001]:
        x=np.linspace(-L,L,n)
        y,r,dy=predictions(model,meta["variant"],x)
        ref,er=reference_on(refdir,kind,state,x)
        ref/=np.sqrt(np.trapezoid(np.sum(ref*ref,axis=1),x))
        l2,sign=aligned_l2(x,y,ref)
        e=float(model.energy.detach())
        m=case["mass"]
        rho=np.sum(y*y,axis=1)
        rn=float(np.sqrt(np.trapezoid(np.sum(r*r,axis=1),x)))
        charge_density=2*(e-potential(x,case))*y[:,0]**2 if kind=="kg" else rho
        k=np.sqrt(m*m-e*e)
        row=dict(case=meta["case_id"],kind=kind,state=state,variant=meta["variant"],seed=meta["seed"],
                 protocol_version=meta["config"]["version"],
                 test_points=n,energy=e,reference_energy=er,energy_abs_error=abs(e-er),
                 energy_rel_error=abs(e-er)/abs(er),binding_energy_rel_error=abs(e-er)/abs(er-m),
                 wavefunction_rel_l2=l2,normalization=float(np.trapezoid(rho,x)),
                 normalization_error=abs(float(np.trapezoid(rho,x))-1),residual_l2=rn,
                 residual_rms=rn/np.sqrt(2*L),residual_max=float(np.max(np.linalg.norm(r,axis=1))),
                 boundary_amplitude=float(np.linalg.norm(y[-1])),
                 boundary_to_peak=float(np.linalg.norm(y[-1])/np.max(np.linalg.norm(y,axis=1))),
                 upper_log_slope=float(-dy[-1,0]/y[-1,0]),expected_kappa=k,
                 boundary_log_slope=float(-np.dot(y[-1],dy[-1])/np.dot(y[-1],y[-1])),
                 tail_norm_estimate=float(np.sum(y[-1]**2)/k),nodes=count_nodes(y[:,0]),
                 parity_error=float(np.max(np.abs(y[:,0]-(-1)**state*y[::-1,0]))),
                 charge=float(np.trapezoid(charge_density,x)),charge_density_min=float(charge_density.min()),
                 alignment_sign=sign,seconds=t["seconds"],parameters=t["parameters"],
                 initial_energy=t["initial_energy"],stop_reason=t["stop_reason"],
                 effective_initial_energy=t.get("attempts",[{"initial_energy":t["initial_energy"]}])[-1]["initial_energy"],
                 restart_count=t.get("restart_count",0),rejected_trial_count=len(t.get("rejected_trials",[])),
                 teacher_training_seconds=json.loads((ROOT/meta["lower_pinn"]["path"]/"training.json").read_text(encoding="utf-8"))["seconds"] if "lower_pinn" in meta else 0.,
                 final_loss=t["loss"],lbfgs_iterations=t["lbfgs_iterations"],
                 relative_robin=abs(float((y[-1,1]-k/(e+m)*y[-1,0])/y[-1,0])) if kind=="dirac" else
                 abs(float((dy[-1,0]+k*y[-1,0])/y[-1,0])),
                 run=str(out.relative_to(ROOT)).replace("\\","/"))
        for j in range(y.shape[1]):
            row[f"component_{j}_rel_l2"]=float(np.sqrt(np.trapezoid((sign*y[:,j]-ref[:,j])**2,x)/np.trapezoid(ref[:,j]**2,x)))
        row["reliable"] = bool(np.isfinite(rn) and row["nodes"]==state and l2<5e-4 and rn<5e-4 and
                               row["energy_abs_error"]<1e-6 and row["normalization_error"]<1e-7 and row["charge"]>0)
        records.append(row)
    pd.DataFrame(records).to_csv(out/"test_grid_convergence.csv",index=False)
    np.savez_compressed(out/"evaluation.npz",x=x,y=sign*y,reference=ref,residual=sign*r,
                        dy=sign*dy,charge_density=charge_density,
                        reference_charge_density=2*(er-potential(x,case))*ref[:,0]**2 if kind=="kg" else np.sum(ref**2,axis=1))
    if checkpoints:
        curve=[]
        for cp in sorted((out/"checkpoints").glob("eval_*.pt"),key=lambda p:int(p.stem.split("_")[-1])):
            cm,_=load_model(out,cp)
            xx=np.linspace(-L,L,2001)
            cy,cr,_=predictions(cm,meta["variant"],xx)
            rf,ee=reference_on(refdir,kind,state,xx)
            cl2,_=aligned_l2(xx,cy,rf)
            curve.append(dict(evaluations=int(cp.stem.split("_")[-1]),energy_abs_error=abs(float(cm.energy.detach())-ee),
                              wavefunction_rel_l2=cl2,residual_l2=float(np.sqrt(np.trapezoid(np.sum(cr*cr,axis=1),xx)))))
        pd.DataFrame(curve).to_csv(out/"accuracy_history.csv",index=False)
    row["evaluation_seconds"]=time.perf_counter()-started
    save_json(out/"metrics.json",row)
    print(f"EVAL {meta['case_id']}/{kind}/{state}/{meta['variant']} seed={meta['seed']}: dE={row['energy_abs_error']:.2e}, L2={l2:.2e}, R={rn:.2e}, reliable={row['reliable']}",flush=True)
    return row

def main(group="formal",checkpoints=False):
    rows=[]
    for path in sorted((ROOT/"results"/group).rglob("training.json")):
        rows.append(evaluate_run(path.parent,checkpoints))
    if rows:
        df=pd.DataFrame(rows)
        df.to_csv(ROOT/"results"/f"{group}_summary.csv",index=False)
        keys=["energy","energy_abs_error","wavefunction_rel_l2","residual_l2","normalization_error","seconds"]
        df.groupby(["case","kind","state","variant"]).agg({k:["mean","std","min","max"] for k in keys}).to_csv(ROOT/"results"/f"{group}_aggregate.csv")
        if group==config().get("formal_group","formal"):
            df.to_csv(ROOT/"results"/"formal_summary.csv",index=False)
            (ROOT/"results"/"formal_aggregate.csv").write_bytes((ROOT/"results"/f"{group}_aggregate.csv").read_bytes())
    return rows

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--group",default="formal");p.add_argument("--checkpoints",action="store_true")
    a=p.parse_args();main(a.group,a.checkpoints)
