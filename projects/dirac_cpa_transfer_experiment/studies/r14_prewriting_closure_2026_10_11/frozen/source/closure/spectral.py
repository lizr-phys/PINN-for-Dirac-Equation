"""Strang time-splitting Fourier pseudospectral solver, independent of PINNs."""
from __future__ import annotations
import json,math,time
import numpy as np
import pandas as pd
import psutil
from scipy.fft import fft,ifft
from plan import BASE,CASES,write,protocol,THRESHOLDS
from .config import configuration,independent_initial

def solve(cfg,nx,dt,*,times=None):
    start=time.perf_counter()
    if times is None:times=np.linspace(0,cfg.t_max,201)
    n=np.rint(np.fft.fftfreq(nx)*nx).astype(int);k=n/8
    spectral=np.zeros((nx,2),dtype=np.complex128);spectral[np.asarray(cfg.indices)%nx]=independent_initial(cfg)
    x=np.arange(nx)*cfg.length/nx;potential=sum(a*np.cos(q*x) for q,a in cfg.potential_harmonics)
    half=np.exp(-.5j*dt*potential);energy=np.sqrt(k*k+cfg.mass**2)
    cos=np.cos(dt*energy);sin=-1j*np.sin(dt*energy)/energy
    out=[spectral.copy()];step_index=0;intervals=0;fft_calls=0
    # Every preset dt divides all test intervals, so no interpolation or shortened last step.
    for target in times[1:]:
        count=int(round(target/dt))-step_index
        if abs(target-(step_index+count)*dt)>2e-12:raise RuntimeError('Non-divisible frozen time step')
        for _ in range(count):
            physical=ifft(spectral,axis=0,norm='ortho',workers=1)*half[:,None]
            spectral=fft(physical,axis=0,norm='ortho',workers=1)
            h=np.stack((cfg.mass*spectral[:,0]+k*spectral[:,1],k*spectral[:,0]-cfg.mass*spectral[:,1]),-1)
            spectral=cos[:,None]*spectral+sin[:,None]*h
            physical=ifft(spectral,axis=0,norm='ortho',workers=1)*half[:,None]
            spectral=fft(physical,axis=0,norm='ortho',workers=1)
            fft_calls+=4;intervals+=1
        step_index+=count;out.append(spectral.copy())
    return np.array(out),dict(propagation_total_seconds=time.perf_counter()-start,integration_steps=intervals,FFT_calls=fft_calls,
        setup_and_all_free_phase_computation_included=True,known_free_operator_prior=True),n

def metrics(pred,exact,cfg,n):
    phase=float(np.angle(np.vdot(exact.ravel(),pred.ravel())));p=pred*np.exp(-1j*phase)
    def rel(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))
    def reconstruct(spec):
        grid=2048;full=np.zeros((len(spec),grid,2),dtype=np.complex128);full[:,n%grid]=spec
        return ifft(full,axis=1,norm='ortho',workers=1)*math.sqrt(grid/cfg.length)
    wp,wr=reconstruct(p),reconstruct(exact)
    rp,rr=np.sum(abs(wp)**2,-1),np.sum(abs(wr)**2,-1)
    jp,jr=2*(wp[...,0].conj()*wp[...,1]).real,2*(wr[...,0].conj()*wr[...,1]).real
    normp,normr=np.sum(abs(p)**2,(1,2)),np.sum(abs(exact)**2,(1,2))
    inner=np.sum(exact.conj()*pred,(1,2));fid=abs(inner)**2/(normp*normr)
    result=dict(spinor_rel_l2_gauge_aligned=rel(p,exact),spinor_rel_l2_raw=rel(pred,exact),density_rel_l2=rel(rp,rr),
        current_rel_l2=rel(jp,jr),fidelity_mean=float(fid.mean()),fidelity_min=float(fid.min()),
        modal_population_rel_l2=rel(np.sum(abs(p)**2,-1),np.sum(abs(exact)**2,-1)),max_norm_drift=float(abs(normp-1).max()),
        global_spacetime_constant_phase=phase,matched_metric_definitions=True,physical_quadrature_points=2048,
        full_FFT_spectrum_in_error=True,predicted_occupation_outside_reference_truncation_max=float(np.sum(abs(pred[:,(n<cfg.n_min)|(n>cfg.n_max)])**2,(1,2)).max()))
    result['seed_joint_pass']=all(result[k]>=v if k=='fidelity_mean' else result[k]<=v for k,v in THRESHOLDS.items())
    return result

def run():
    directory=BASE/'spectral_benchmark';directory.mkdir(exist_ok=True);rows=[]
    for case in CASES:
        cfg=configuration(case)
        with np.load(BASE/'references'/case/'test201.npz') as f:times,reference=f['t'],f['state']
        for nx in protocol()['spectral_integrator']['grid_points']:
            for dt in protocol()['spectral_integrator']['time_steps']:
                tag=f'{case}_N{nx}_dt{dt:.8f}';saved=directory/(tag+'.json')
                if saved.exists():rows.append(json.loads(saved.read_text(encoding='utf-8')));continue
                observed=[];cpus=[]
                for repeat in range(5):
                    cpu=time.process_time();pred,cost,n=solve(cfg,nx,dt,times=times);cpus.append(time.process_time()-cpu);observed.append(cost['propagation_total_seconds'])
                exact=np.zeros_like(pred);exact[:,np.asarray(cfg.indices)%nx]=reference
                met=metrics(pred,exact,cfg,n)
                record=dict(case=case,method='TSFP2',grid=nx,dt=dt,cost_repetitions=5,cost_median_seconds=float(np.median(observed)),
                    cost_seconds_samples=observed,cpu_seconds_samples=cpus,cpu_median_seconds=float(np.median(cpus)),
                    process_rss_bytes=psutil.Process().memory_info().rss,peak_rss_bytes=getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss),
                    training_seconds=0.,auxiliary_pretraining_seconds=0.,unit_spinor_residual_points=0,
                    full_grid_inference_includes_setup=True,reference_diagonalization_excluded=True,**cost,**met)
                np.savez_compressed(directory/(tag+'.npz'),t=times,n=n,prediction=pred)
                write(saved,record);rows.append(record)
                pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows]).to_csv(directory/'error_cost.csv',index=False)
                print('TSFP2',case,nx,dt,met['spinor_rel_l2_gauge_aligned'],flush=True)
    write(directory/'complete.json',dict(status='COMPLETE',configurations=len(rows),expected=108,cost_repetitions=5))

def self_test():
    from .reference import diagonalize,trajectory
    from dataclasses import replace
    cfg=configuration('m5T5',(0,32));times=np.linspace(0,1.,41)
    free=replace(cfg,potential_harmonics=((1,0.),))
    pred,_,n=solve(free,128,.025,times=times);ref=np.zeros_like(pred);ref[:,np.asarray(free.indices)%128]=trajectory(diagonalize(free),times)
    error=float(np.linalg.norm(pred-ref)/np.linalg.norm(ref));assert error<2e-13,error
    errors=[]
    # Wider independently diagonalized reference for a convergence test, not an experimental selection.
    wide=configuration('m5T5',(-48,80));diag=diagonalize(wide);exact=trajectory(diag,times)
    for dt in (.025,.0125,.00625):
        p,_,n=solve(wide,256,dt,times=times);e=np.zeros_like(p);e[:,np.asarray(wide.indices)%256]=exact
        errors.append(float(np.linalg.norm(p-e)/np.linalg.norm(e)))
    assert errors[0]/errors[1]>3.8 and errors[1]/errors[2]>3.8,errors
    return dict(free_exact_error=error,Strang_dt_errors=errors,second_order_verified=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    import sys
    with threadpool_limits(limits=1):
        if '--test' in sys.argv:write(BASE/'engineering/TSFP_tests.json',dict(status='PASS',**self_test()));print('TSFP_TEST_PASS')
        else:run()
