from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
from .physics import ROOT,config,potential,reference_on,save_json
from .reference import bvp

def overlaps():
    cfg=config();case=cfg["cases"]["deep_wide"]
    L=case["half_length"]
    x=np.linspace(-L,L,16001)
    rows=[]
    for kind in ["dirac","kg"]:
        for seed in [None]+cfg["validation_seeds"]:
            values=[];energies=[]
            for state in [0,1,2]:
                if seed is None:
                    y,e=reference_on(ROOT/"results"/"reference"/"deep_wide",kind,state,x)
                else:
                    from .evaluate import load_model,predictions
                    from .campaign import formal_path
                    out=formal_path("deep_wide",kind,state,seed)
                    model,meta=load_model(out)
                    y,_,_=predictions(model,meta["variant"],x)
                    e=float(model.energy.detach())
                    ref,_=reference_on(ROOT/"results"/"reference"/"deep_wide",kind,state,x)
                    if np.trapezoid(np.sum(y*ref,axis=1),x)<0: y=-y
                values.append(y);energies.append(e)
            for n in [0,1,2]:
                for m in range(n+1,3):
                    yn,ym=values[n],values[m];en,em=energies[n],energies[m]
                    ordinary=float(np.trapezoid(np.sum(yn*ym,axis=1),x))
                    kn,km=np.sqrt(1-en*en),np.sqrt(1-em*em)
                    if kind=="kg":
                        product=yn[:,0]*ym[:,0]
                        physical=float(np.trapezoid((en+em-2*potential(x,case))*product,x))
                        Qn=float(np.trapezoid(2*(en-potential(x,case))*yn[:,0]**2,x))
                        Qm=float(np.trapezoid(2*(em-potential(x,case))*ym[:,0]**2,x))
                        tail=(en+em)*(yn[-1,0]*ym[-1,0]+yn[0,0]*ym[0,0])/(kn+km)
                        normalized=(physical+tail)/np.sqrt(Qn*Qm)
                    else:
                        physical=ordinary
                        tail=(np.dot(yn[-1],ym[-1])+np.dot(yn[0],ym[0]))/(kn+km)
                        normalized=physical+tail
                    rows.append(dict(kind=kind,seed=-1 if seed is None else seed,source="reference" if seed is None else "pinn",
                                     state_n=n,state_m=m,ordinary_l2=ordinary,physical_overlap=physical,
                                     estimated_tail_correction=tail,normalized_physical_overlap=normalized))
    df=pd.DataFrame(rows)
    df.to_csv(ROOT/"results"/"orthogonality.csv",index=False)
    return df

def benchmark():
    case=config()["cases"]["base"]
    rows=[]
    for kind in ["dirac","kg"]:
        refdir=ROOT/"results"/"reference"/"base"
        x=np.linspace(-20,20,8001)
        from .physics import full_values,aligned_l2
        ref,e=reference_on(refdir,kind,0,x)
        bvp(kind,case,0,length=20,tol=1e-9)  # one warm-up, no reference initialization
        for tol in [1e-7,1e-9,2e-11]:
            for repeat in range(5):
                t0=time.perf_counter()
                sol=bvp(kind,case,0,length=20,tol=tol)
                seconds=time.perf_counter()-t0
                y=full_values(kind,sol.sol,x,0)
                if kind=="kg":y=y[:,:1]
                y/=np.sqrt(np.trapezoid(np.sum(y*y,axis=1),x))
                l2,_=aligned_l2(x,y,ref)
                rows.append(dict(kind=kind,tolerance=tol,repeat=repeat,seconds=seconds,energy=float(sol.p[0]),
                                 energy_abs_error=abs(float(sol.p[0])-e),wavefunction_rel_l2=l2,nodes=len(sol.x)))
    df=pd.DataFrame(rows)
    df.to_csv(ROOT/"results"/"bvp_benchmark.csv",index=False)
    return df

def main():
    matrix=overlaps()
    timing=benchmark()
    save_json(ROOT/"results"/"diagnostics.json",dict(
        kg_same_parity_reference=matrix[(matrix.kind=="kg")&(matrix.source=="reference")&(matrix.state_n==0)&(matrix.state_m==2)].to_dict("records"),
        bvp_timing=timing.groupby("kind").seconds.agg(["mean","min","max"]).to_dict("index"),
        timing_policy="Serial single CPU thread; warm-up excluded; BVP starts from analytic trial with a physics-only energy guess, not a stored reference wavefunction."))

if __name__=="__main__":main()
