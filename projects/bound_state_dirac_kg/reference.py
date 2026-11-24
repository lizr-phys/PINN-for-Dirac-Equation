"""Adaptive BVP reference, reverse shooting, Fourier spectrum and doubler audit."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp, solve_ivp
from scipy.linalg import eigh, eig
from scipy.optimize import brentq
from contextlib import nullcontext
from .physics import ROOT, config, potential, rhs, save_json, count_nodes, aligned_l2


def bvp(kind, L=20.0, n=161, tol=1e-10, guess=0.65):
    x = np.linspace(0, L, n)
    f = np.exp(-x*x/8)
    y = np.array([f, x/4*f/(guess-potential(x)+1) if kind == "dirac" else -x/4*f])

    def bc(ya, yb, p):
        k = np.sqrt(1-p[0]**2)
        return np.array([ya[0]-1, ya[1], yb[1]-(k/(p[0]+1)*yb[0]) if kind == "dirac"
                         else yb[1]+k*yb[0]])

    sol = solve_bvp(lambda x,y,p: rhs(kind,x,y,p[0]), bc, x, y, p=[guess],
                    tol=tol, bc_tol=tol, max_nodes=60000)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def bvp_values(kind, sol, x, derivative=0):
    z = sol.sol(np.abs(x), derivative).T
    # Base first component even, second odd; derivatives reverse parity.
    if derivative % 2 == 0:
        z[:, 1] *= np.sign(x)
    else:
        z[:, 0] *= np.sign(x)
    return z


def shoot(kind, L, bracket, rtol=3e-13, max_step=0.12):
    def integrate(e, dense=False):
        k=np.sqrt(1-e*e)
        yl=[1., k/(e+1) if kind == "dirac" else -k]
        return solve_ivp(lambda x,y: rhs(kind,x,y,e), [L,0], yl,
                         method="DOP853", rtol=rtol, atol=rtol*.1,
                         max_step=max_step, dense_output=dense)
    def mismatch(e):
        y=integrate(e).y[:,-1]
        return y[1]/np.linalg.norm(y)
    e=brentq(mismatch,*bracket,xtol=5e-15)
    return e,integrate(e,True)


def fourier_matrices(n, L):
    if n % 2 != 1:
        raise ValueError("Odd N avoids an unpaired Nyquist mode in first differentiation.")
    x=-L+2*L*np.arange(n)/n
    k=2*np.pi*np.fft.fftfreq(n,d=2*L/n)
    eye=np.eye(n)
    f=np.fft.fft(eye,axis=0)
    d=np.fft.ifft(1j*k[:,None]*f,axis=0).real
    d2=np.fft.ifft(-k[:,None]**2*f,axis=0).real
    return x,d,d2


def spectrum(kind,n=257,L=20.,central=False):
    if central:
        x=-L+2*L*np.arange(n)/n
        d=(np.roll(np.eye(n),-1,axis=1)-np.roll(np.eye(n),1,axis=1))/(4*L/n)
        # Row j must have +1 at j+1.
        d=-d
    else:
        x,d,d2=fourier_matrices(n,L)
    v=potential(x)
    if kind == "dirac":
        h=np.block([[np.diag(1+v),d],[-d,np.diag(v-1)]])
        energies,vectors=eigh(h,subset_by_value=(-.999999999,.999999999))
    else:
        K=-d2+np.diag(1-v*v)
        h=np.block([[np.zeros((n,n)),np.eye(n)],[K,np.diag(2*v)]])
        ev,vec=eig(h)
        keep=(np.abs(ev.imag)<1e-8)&(np.abs(ev.real)<.999999999)
        order=np.argsort(ev.real[keep])
        energies=ev.real[keep][order]
        vectors=vec[:,keep][:,order].real
    return x,energies,vectors


def fourier_eval(x,y,z,L,derivative=0):
    n=len(x)
    coeff=np.fft.fft(y,axis=0)/n
    k=2*np.pi*np.fft.fftfreq(n,d=2*L/n)
    out=[]
    for zz in np.array_split(z,max(1,len(z)//512)):
        out.append((np.exp(1j*(zz[:,None]+L)*k) @ ((1j*k[:,None])**derivative*coeff)).real)
    return np.concatenate(out)


def main():
    cfg=config()
    if (cfg["mass"],cfg["depth"],cfg["width"]) != (1.0,0.5,2.0):
        raise ValueError("This validated reference study is fixed at m=1, V0=0.5, a=2.")
    out=ROOT/"results"/"reference"
    out.mkdir(parents=True,exist_ok=True)
    rows=[]; spectra=[]; summary={}; solutions={}
    started=time.perf_counter()
    with nullcontext():
        for kind in ["dirac","kg"]:
            kind_start=time.perf_counter()
            sol=bvp(kind,L=28,n=321,tol=2e-12)
            energy=float(sol.p[0]); solutions[kind]=sol
            x=np.linspace(-20,20,8001)
            full=bvp_values(kind,sol,x)
            z=full if kind=="dirac" else full[:,:1]
            norm=np.sqrt(np.trapezoid(np.sum(z*z,axis=1),x))
            z/=norm
            derivative=bvp_values(kind,sol,x,1)/norm
            if kind=="dirac":
                residual=derivative-rhs(kind,x,z.T,energy).T
            else:
                residual=(derivative[:,1]+((energy-potential(x))**2-1)*z[:,0])[:,None]
            es,ss=shoot(kind,28,(energy-.035,energy+.035))
            sy=ss.sol(np.abs(x)).T
            if kind=="dirac": sy[:,1]*=np.sign(x)
            else: sy=sy[:,:1]
            sy/=np.sqrt(np.trapezoid(np.sum(sy*sy,axis=1),x))
            shoot_l2,_=aligned_l2(x,sy,z)
            for L in [12.,16.,20.,24.,28.]:
                for n,tol in [(81,1e-7),(161,1e-9),(321,2e-12)]:
                    ts=time.perf_counter(); s=bvp(kind,L,n,tol)
                    rows.append(dict(kind=kind,method="bvp",L=L,N=n,tol=tol,
                                     adaptive_nodes=len(s.x),energy=float(s.p[0]),
                                     abs_delta=abs(s.p[0]-energy),seconds=time.perf_counter()-ts))
            for L,n in [(16.,129),(20.,33),(20.,65),(20.,129),(20.,257),(20.,513),(24.,257),(28.,257),(28.,513)]:
                ts=time.perf_counter(); xx,ee,vv=spectrum(kind,n,L)
                idx=np.where(ee>0)[0][0]; et=float(ee[idx])
                yy=vv[:,idx].reshape(2,n).T if kind=="dirac" else vv[:n,idx,None]
                test=np.linspace(-min(20,L),min(20,L),4002)[1:-1]
                yf=fourier_eval(xx,yy,test,L)
                yf/=np.sqrt(np.trapezoid(np.sum(yf*yf,axis=1),test))
                yr=bvp_values(kind,sol,test)
                if kind=="kg": yr=yr[:,:1]
                yr/=np.sqrt(np.trapezoid(np.sum(yr*yr,axis=1),test))
                l2,_=aligned_l2(test,yf,yr)
                rows.append(dict(kind=kind,method="fourier",L=L,N=n,tol=0,
                                 adaptive_nodes=n,energy=et,abs_delta=abs(et-energy),
                                 wavefunction_l2=l2,seconds=time.perf_counter()-ts))
                for e in ee: spectra.append(dict(kind=kind,L=L,N=n,energy=e))
                if L==28 and n==513:
                    yf0=fourier_eval(xx,yy,test,L)
                    if kind=="dirac":
                        rr=fourier_eval(xx,yy,test,L,1)-rhs(kind,test,yf0.T,et).T
                    else:
                        rr=fourier_eval(xx,yy,test,L,2)+(((et-potential(test))**2-1)[:,None])*yf0
                    fourier_res=np.sqrt(np.trapezoid(np.sum(rr*rr,axis=1),test)/np.trapezoid(np.sum(yf0*yf0,axis=1),test))
                    final_spectrum=ee.tolist(); fourier_l2=l2; fourier_e=et
            q=float(np.trapezoid(2*(energy-potential(x))*z[:,0]**2,x)) if kind=="kg" else 1.
            boundary=float(np.linalg.norm(z[-1]))
            summary[kind]=dict(energy=energy,L=28,tol=2e-12,adaptive_nodes=len(sol.x),
                shooting_energy=es,shooting_energy_difference=abs(es-energy),shooting_l2=shoot_l2,
                fourier_energy=fourier_e,fourier_energy_difference=abs(fourier_e-energy),
                fourier_l2=fourier_l2,fourier_continuous_residual_l2=float(fourier_res),
                positive_spectrum=[e for e in final_spectrum if e>0],
                negative_gap_spectrum=[e for e in final_spectrum if e<0],
                bvp_residual_l2=float(np.sqrt(np.trapezoid(np.sum(residual**2,axis=1),x))),
                boundary_amplitude_at_20=boundary,kappa=float(np.sqrt(1-energy**2)),
                tail_norm_estimate_beyond_20=boundary**2/np.sqrt(1-energy**2),
                upper_nodes=count_nodes(z[:,0]),charge=q,seconds=time.perf_counter()-kind_start)
            np.savez_compressed(out/f"{kind}.npz",x=x,y=z,residual=residual,energy=energy,
                                shooting_y=sy,charge_density=2*(energy-potential(x))*z[:,0]**2 if kind=="kg" else np.sum(z*z,axis=1))
            np.savez_compressed(out/f"{kind}_bvp_polynomial.npz",knots=sol.sol.x,coefficients=sol.sol.c,energy=energy,norm=norm)
            print(kind,summary[kind],flush=True)
        doubled=[]
        for n in [128,256,512]:
            xx,ee,vv=spectrum("dirac",n,20,central=True)
            for i,e in enumerate(ee):
                yy=vv[:,i].reshape(2,n).T
                power=np.sum(np.abs(np.fft.fft(yy,axis=0))**2,axis=1)
                freq=np.abs(np.fft.fftfreq(n))
                rr=fourier_eval(xx,yy,xx+0.0031,20,1)-rhs("dirac",xx+0.0031,fourier_eval(xx,yy,xx+0.0031,20).T,e).T
                doubled.append(dict(N=n,energy=e,high_frequency_fraction=float(power[freq>.25].sum()/power.sum()),
                                    continuum_residual_relative=float(np.linalg.norm(rr)/np.linalg.norm(yy)),
                                    upper_nodes=count_nodes(yy[:,0])))
        pd.DataFrame(doubled).to_csv(out/"naive_dirac_doubling.csv",index=False)
    pd.DataFrame(rows).to_csv(out/"convergence.csv",index=False)
    pd.DataFrame(spectra).to_csv(out/"spectra.csv",index=False)
    summary["total_seconds"]=time.perf_counter()-started
    save_json(out/"summary.json",summary)
    cfg=config()
    for kind in ["dirac","kg"]:
        e=summary[kind]["positive_spectrum"]
        assert .5<e[0]<.8 and (len(e)==1 or e[1]>.8)
        assert summary[kind]["upper_nodes"]==0
        assert summary[kind]["shooting_energy_difference"]<1e-9
        assert summary[kind]["fourier_energy_difference"]<1e-8
    cfg["energy_prior"]="Reference Fourier spectra and shooting confirm exactly one positive level in [0.5,0.8] for each equation. This reference-derived bracket and even/nodeless sector are used as priors, never as wavefunction or energy regression labels."
    cfg["parameters_frozen_after_reference"]=True
    save_json(ROOT/"config.json",cfg)


if __name__=="__main__":
    main()
