"""Post-training evaluation. Reference data are first accessed here."""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PPoly
from .physics import ROOT,save_json,aligned_l2,count_nodes,potential
from .model import BoundPINN,grad


def reference_on(kind,x):
    data=np.load(ROOT/"results"/"reference"/f"{kind}_bvp_polynomial.npz")
    poly=PPoly(data["coefficients"],data["knots"])
    y=poly(np.abs(x))/data["norm"]
    if kind=="dirac": y[:,1]*=np.sign(x)
    else: y=y[:,:1]
    return y,float(data["energy"])


def evaluate_run(out):
    meta=json.loads((out/"config.json").read_text(encoding="utf-8"))
    train=json.loads((out/"training.json").read_text(encoding="utf-8"))
    cfg=meta["config"]; kind=meta["kind"]
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    model=BoundPINN(kind,cfg)
    model.load_state_dict(torch.load(out/"model.pt",weights_only=True))
    model.eval()
    L=cfg["half_length"]
    metrics=[]
    for n in [2001,4001,8001]:
        x=np.linspace(-L,L,n)
        ys=[]; rs=[]; dys=[]
        for chunk in np.array_split(x,max(1,n//512)):
            xt=torch.tensor(chunk[:,None],requires_grad=True)
            yt=model(xt)
            ys.append(yt.detach().numpy())
            rs.append(model.residual(xt).detach().numpy())
            dys.append(torch.cat([grad(yt[:,j:j+1],xt) for j in range(yt.shape[1])],dim=1).detach().numpy())
        y=np.concatenate(ys); r=np.concatenate(rs); dy=np.concatenate(dys)
        yr,er=reference_on(kind,x)
        l2,sign=aligned_l2(x,y,yr)
        e=float(model.energy.detach())
        density=np.sum(y*y,axis=1)
        norm=float(np.trapezoid(density,x))
        rnorm=float(np.sqrt(np.trapezoid(np.sum(r*r,axis=1),x)))
        charge_density=2*(e-potential(x))*y[:,0]**2 if kind=="kg" else density
        row=dict(kind=kind,seed=meta["seed"],test_points=n,energy=e,
            reference_energy=er,energy_abs_error=abs(e-er),energy_rel_error=abs(e-er)/abs(er),
            wavefunction_rel_l2=l2,alignment_sign=sign,normalization=norm,
            normalization_error=abs(norm-1),residual_l2=rnorm,
            residual_rms=rnorm/np.sqrt(2*L),residual_max=float(np.max(np.linalg.norm(r,axis=1))),
            boundary_amplitude=float(max(np.linalg.norm(y[0]),np.linalg.norm(y[-1]))),
            boundary_to_peak=float(np.linalg.norm(y[-1])/np.max(np.linalg.norm(y,axis=1))),
            boundary_scaled_residual=float(torch.linalg.norm(model.boundary_residual()).detach()),
            tail_norm_estimate=float(np.sum(y[-1]**2)/np.sqrt(1-e*e)),
            measured_log_slope_at_boundary=float(-np.dot(y[-1],dy[-1])/np.dot(y[-1],y[-1])),
            expected_kappa=float(np.sqrt(1-e*e)),
            nodes=count_nodes(y[:,0]),
            parity_error=float(np.max(np.abs(y[:,0]-y[::-1,0]))),
            charge=float(np.trapezoid(charge_density,x)),charge_density_min=float(charge_density.min()),
            continuity_residual_max=0.0,seconds=train["seconds"],parameters=train["parameters"],
            initial_energy=train["initial_energy"],final_training_loss=train["loss"])
        if kind=="dirac":
            row["lower_parity_error"]=float(np.max(np.abs(y[:,1]+y[::-1,1])))
            for j in [0,1]:
                row[f"component_{j}_rel_l2"]=float(np.sqrt(np.trapezoid((sign*y[:,j]-yr[:,j])**2,x)/np.trapezoid(yr[:,j]**2,x)))
        metrics.append(row)
    np.savez_compressed(out/"evaluation.npz",x=x,y=sign*y,reference=yr,residual=sign*r,
                        point_error=sign*y-yr,charge_density=charge_density,
                        reference_charge_density=2*(er-potential(x))*yr[:,0]**2 if kind=="kg" else np.sum(yr*yr,axis=1))
    table={"x":x,"potential":potential(x),"charge_density":charge_density}
    for j in range(y.shape[1]):
        table[f"pinn_{j}"]=sign*y[:,j]; table[f"reference_{j}"]=yr[:,j]
        table[f"error_{j}"]=sign*y[:,j]-yr[:,j]
    for j in range(r.shape[1]): table[f"residual_{j}"]=sign*r[:,j]
    pd.DataFrame(table).to_csv(out/"evaluation.csv",index=False)
    pd.DataFrame(metrics).to_csv(out/"test_grid_convergence.csv",index=False)
    save_json(out/"metrics.json",row)
    return row


def main(tag="formal"):
    rows=[]
    for out in sorted((ROOT/"results"/tag).glob("*/seed_*")):
        if (out/"training.json").exists():
            row=evaluate_run(out);rows.append(row)
            print(f'{row["kind"]} seed={row["seed"]}: dE={row["energy_abs_error"]:.3e}, L2={row["wavefunction_rel_l2"]:.3e}, R={row["residual_l2"]:.3e}',flush=True)
    if not rows: raise RuntimeError("No completed runs")
    df=pd.DataFrame(rows)
    df.to_csv(ROOT/"results"/f"{tag}_summary.csv",index=False)
    df.groupby("kind").agg({k:["mean","std","min","max"] for k in ["energy","energy_abs_error","wavefunction_rel_l2","residual_l2","normalization_error","seconds"]}).to_csv(ROOT/"results"/f"{tag}_aggregate.csv")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--tag",default="formal")
    main(p.parse_args().tag)
