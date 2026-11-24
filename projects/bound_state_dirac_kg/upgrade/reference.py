"""Generalized reference study. This module is never imported by training."""
from __future__ import annotations
import argparse
import time
import os
from contextlib import nullcontext
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp, solve_ivp
from scipy.linalg import eigh, eig
from scipy.optimize import brentq
from ..reference import fourier_matrices, fourier_eval
from .physics import ROOT,config,save_json,potential,rhs,full_values,aligned_l2,count_nodes

def spectrum(kind, case, n=513, length=None):
    length = case["reference_length"] if length is None else length
    x,d,d2 = fourier_matrices(n,length)
    v,m = potential(x,case),case["mass"]
    if kind == "dirac":
        h = np.block([[np.diag(m+v),d],[-d,np.diag(v-m)]])
        e,y = eigh(h,subset_by_value=(-m+1e-10,m-1e-10))
    else:
        k = -d2+np.diag(m*m-v*v)
        h = np.block([[np.zeros((n,n)),np.eye(n)],[k,np.diag(2*v)]])
        ev,vec = eig(h)
        keep = (np.abs(ev.imag)<1e-9)&(np.abs(ev.real)<m-1e-10)
        order = np.argsort(ev.real[keep])
        e,y = ev.real[keep][order],vec[:,keep][:,order].real
    keep = e>0
    return x,e[keep],y[:,keep]

def bvp(kind,case,state=0,length=None,tol=2e-11,guess=None,initial=None):
    length = case["reference_length"] if length is None else length
    x = np.linspace(0,length,321)
    m,a = case["mass"],case["width"]
    guess = m*(1-case["depth"]/m/2) if guess is None else guess
    if initial is None:
        f = np.exp(-x*x/(4*a*a))
        p = x/a if state%2 else np.ones_like(x)
        u = p*f
        du = ((1/a)*f if state%2 else 0)-x/(2*a*a)*u
        y = np.array([u,-du/(guess-potential(x,case)+m) if kind=="dirac" else du])
    else:
        xx,yy,L0 = initial
        u = fourier_eval(xx,yy,x,L0).T
        y = u if kind == "dirac" else np.vstack([u[0],fourier_eval(xx,yy,x,L0,1)[:,0]])
    target = -1.0 if state%2 and kind=="dirac" else 1.0
    anchor = y[1,0] if state%2 else y[0,0]
    y *= target/anchor

    def bc(ya,yb,p):
        e = p[0]
        k = np.sqrt(max(m*m-e*e,1e-18))
        origin = [ya[0],ya[1]-target] if state%2 else [ya[0]-1,ya[1]]
        tail = yb[1]-k/(e+m)*yb[0] if kind=="dirac" else yb[1]+k*yb[0]
        return np.r_[origin,tail]
    sol = solve_bvp(lambda z,y,p:rhs(kind,z,y,p[0],case),bc,x,y,p=[guess],
                    tol=tol,bc_tol=tol,max_nodes=80000)
    if not sol.success or not 0<sol.p[0]<m:
        raise RuntimeError(f"{kind} state {state}: {sol.message}, E={sol.p[0]}")
    return sol

def shooting(kind,case,state,energy,length):
    m = case["mass"]
    def integrate(e,dense=False):
        k = np.sqrt(m*m-e*e)
        tail = [1.,k/(e+m) if kind=="dirac" else -k]
        sol = solve_ivp(lambda z,y:rhs(kind,z,y,e,case),[length,0],tail,method="DOP853",
                        rtol=3e-13,atol=3e-14,max_step=.12,dense_output=dense)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol
    def mismatch(e):
        y = integrate(e).y[:,-1]
        return y[0 if state%2 else 1]/np.linalg.norm(y)
    delta = min(.01,energy/2,(m-energy)/2)
    e = brentq(mismatch,energy-delta,energy+delta,xtol=5e-15)
    return e,integrate(e,True)

def study(case_id):
    cfg = config()
    case = cfg["cases"][case_id]
    out = ROOT/"results"/"reference"/case_id
    if (out/"summary.json").exists():
        import json
        if json.loads((out/"summary.json").read_text(encoding="utf-8"))["case"] != case:
            raise RuntimeError("Reference cache differs from the requested case.")
        print(f"SKIP reference {case_id}",flush=True)
        return
    out.mkdir(parents=True,exist_ok=True)
    summary = {"case":case,"states":{},"verified":False}
    convergence = []
    with nullcontext():
        for kind in ("dirac","kg"):
            L = case["reference_length"]
            xx,ee,vectors = spectrum(kind,case,257,L)
            if len(ee)<=max(case["states"]):
                raise RuntimeError(f"Not enough positive bound states for {case_id}/{kind}: {ee}")
            for state in case["states"]:
                t0 = time.perf_counter()
                yy = vectors[:,state].reshape(2,len(xx)).T if kind=="dirac" else vectors[:len(xx),state,None]
                initial = (xx,yy,L)
                low = bvp(kind,case,state,L,2e-11,float(ee[state]),initial)
                high_tol=2e-12
                fallback=None
                try:
                    high = bvp(kind,case,state,L+8,high_tol,float(ee[state]),initial)
                except RuntimeError as exc:
                    fallback=str(exc)
                    high_tol=1e-11
                    high = bvp(kind,case,state,L+8,high_tol,float(ee[state]),initial)
                e = float(high.p[0])
                x = np.linspace(-case["half_length"],case["half_length"],12001)
                all_y = full_values(kind,high.sol,x,state)
                y = all_y if kind=="dirac" else all_y[:,:1]
                norm = np.sqrt(np.trapezoid(np.sum(y*y,axis=1),x))
                y /= norm
                dy = full_values(kind,high.sol,x,state,1)/norm
                r = dy-rhs(kind,x,y.T,e,case).T if kind=="dirac" else (dy[:,1]+((e-potential(x,case))**2-case["mass"]**2)*y[:,0])[:,None]
                es,ss = shooting(kind,case,state,e,L+8)
                sy = full_values(kind,ss.sol,x,state)
                if kind=="kg": sy = sy[:,:1]
                sy /= np.sqrt(np.trapezoid(np.sum(sy*sy,axis=1),x))
                sl2,_ = aligned_l2(x,sy,y)
                record = dict(energy=e,bvp_L=L+8,bvp_tol=high_tol,bvp_tolerance_fallback=fallback,bvp_nodes=len(high.x),shooting_energy=es,
                              shooting_energy_difference=abs(es-e),shooting_l2=sl2,
                              domain_energy_difference=abs(float(low.p[0])-e),
                              bvp_residual_l2=float(np.sqrt(np.trapezoid(np.sum(r*r,axis=1),x))),
                              nodes=count_nodes(y[:,0]),tail_amplitude=float(np.linalg.norm(y[-1])),
                              charge=float(np.trapezoid(2*(e-potential(x,case))*y[:,0]**2,x)) if kind=="kg" else 1.)
                for n,Lf in [(257,L),(513,L),(513,L+8)]:
                    xf,ef,vf = spectrum(kind,case,n,Lf)
                    idx = int(np.argmin(abs(ef-e)))
                    yf = vf[:,idx].reshape(2,n).T if kind=="dirac" else vf[:n,idx,None]
                    evaluated = fourier_eval(xf,yf,x,Lf)
                    evaluated /= np.sqrt(np.trapezoid(np.sum(evaluated**2,axis=1),x))
                    fl2,_ = aligned_l2(x,evaluated,y)
                    convergence.append(dict(case=case_id,kind=kind,state=state,L=Lf,N=n,energy=float(ef[idx]),
                                            energy_delta=abs(float(ef[idx])-e),wavefunction_l2=fl2))
                    if n==513 and Lf==L+8:
                        record.update(fourier_energy=float(ef[idx]),fourier_energy_difference=abs(float(ef[idx])-e),fourier_l2=fl2)
                record["seconds_with_crosschecks"] = time.perf_counter()-t0
                if not (record["nodes"]==state and record["shooting_energy_difference"]<1e-9 and
                        record["fourier_energy_difference"]<1e-9 and record["fourier_l2"]<2e-8 and sl2<1e-9):
                    save_json(out/"failed_reference.json",record)
                    raise RuntimeError(f"Reference validation failed: {case_id}/{kind}/{state}: {record}")
                summary["states"][f"{kind}_{state}"] = record
                np.savez_compressed(out/f"{kind}_state_{state}.npz",coefficients=high.sol.c,knots=high.sol.x,
                                    energy=e,norm=norm,x=x,y=y,residual=r,shooting_y=sy)
                print(f"REF {case_id}/{kind}/{state} E={e:.12f} nodes={record['nodes']} Fourier L2={record['fourier_l2']:.2e}",flush=True)
    pd.DataFrame(convergence).to_csv(out/"convergence.csv",index=False)
    summary["verified"] = True
    save_json(out/"summary.json",summary)

def main(case_id=None):
    for name in ([case_id] if case_id else config()["cases"]):
        study(name)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",choices=list(config()["cases"]))
    main(parser.parse_args().case)
