"""Independent references and fixed grid calibration. Never imported by training."""
from __future__ import annotations
import json,math,time
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.linalg import expm
from plan import BASE,CASES,protocol,write,sha
from .config import configuration,independent_initial

def hamiltonian(cfg):
    n=cfg.mode_count;h=np.zeros((2*n,2*n),dtype=np.complex128)
    for a,k in enumerate(cfg.wave_numbers):
        h[2*a:2*a+2,2*a:2*a+2]=[[cfg.mass,k],[k,-cfg.mass]]
        for q,amplitude in cfg.potential_harmonics:
            b=a+8*q
            if b<n:
                h[2*a:2*a+2,2*b:2*b+2]=amplitude/2*np.eye(2)
                h[2*b:2*b+2,2*a:2*a+2]=amplitude/2*np.eye(2)
    return h

def diagonalize(cfg):
    h=hamiltonian(cfg);e,u=np.linalg.eigh(h);c0=independent_initial(cfg).ravel();return h,e,u,c0,u.conj().T@c0

def trajectory(diag,times):
    h,e,u,c0,a0=diag;z=np.exp(-1j*np.asarray(times)[:,None]*e)*a0
    return (z@u.T).reshape(len(times),len(c0)//2,2)

def apply_independent(state,cfg):
    k=np.asarray(cfg.wave_numbers)[None,:,None]
    result=k*state[...,::-1]+cfg.mass*np.stack((state[...,0],-state[...,1]),-1)
    for q,a in cfg.potential_harmonics:
        shift=8*q;result[:,shift:]+=a/2*state[:,:-shift];result[:,:-shift]+=a/2*state[:,shift:]
    return result

def compare(cfg,big,times):
    da,db=diagonalize(cfg),diagonalize(big);a,b=trajectory(da,times),trajectory(db,times)
    padded=np.zeros_like(b);offset=cfg.n_min-big.n_min;padded[:,offset:offset+cfg.mode_count]=a
    h,e,u,c0,a0=da;z=np.exp(-1j*times[:,None]*e)*a0
    dt=((-1j*e*z)@u.T).reshape(a.shape)
    er=[]
    for tindex in (0,len(times)//4,len(times)//2,len(times)-1):er.append(np.linalg.norm(expm(-1j*times[tindex]*h)@c0-a[tindex].ravel()))
    row=dict(case=cfg.case,smaller=[cfg.n_min,cfg.n_max],larger=[big.n_min,big.n_max],
        trajectory_rel_difference=float(np.linalg.norm(padded-b)/np.linalg.norm(b)),
        lower_boundary_occupation=float(np.sum(abs(a[:,:16])**2,(1,2)).max()),
        upper_boundary_occupation=float(np.sum(abs(a[:,-16:])**2,(1,2)).max()),
        hermiticity_error=float(abs(h-h.conj().T).max()),norm_drift=float(abs(np.sum(abs(a)**2,(1,2))-1).max()),
        reverse_error=float(np.linalg.norm(expm(1j*times[-1]*h)@a[-1].ravel()-c0)),
        independent_expm_error=float(max(er)),pde_error_rms=float(np.linalg.norm(1j*dt-apply_independent(a,cfg))/np.sqrt(len(times))))
    row['pass']=bool(row['trajectory_rel_difference']<1e-6 and max(row['lower_boundary_occupation'],row['upper_boundary_occupation'])<1e-8
        and row['hermiticity_error']==0 and max(row[k] for k in ('norm_drift','reverse_error','independent_expm_error','pde_error_rms'))<1e-12)
    return row,a,b,da

def prepare():
    out=BASE/'references';out.mkdir(exist_ok=True)
    frozen=BASE/'frozen/design.json'
    if not frozen.is_file():raise RuntimeError('Design must precede references')
    existing=out/'calibration.json'
    if existing.exists():return json.loads(existing.read_text(encoding='utf-8'))
    start=time.perf_counter();candidates=protocol()['reference_policy']['candidate_bounds'];rows=[];selected=None
    for bounds,bigbounds in zip(candidates[:-1],candidates[1:]):
        qualified=True
        for case in CASES:
            cfg=configuration(case,bounds);big=configuration(case,bigbounds)
            times=np.linspace(0,cfg.t_max,cfg.residual_time_points)
            row,a,b,diag=compare(cfg,big,times);rows.append(row);qualified &= row['pass']
            d=out/case;d.mkdir(exist_ok=True)
            pd.DataFrame(rows).to_csv(out/'reference_convergence.csv',index=False)
            np.savez_compressed(d/f'reference_{bounds[0]}_{bounds[1]}.npz',t=times,state=a,n=np.array(cfg.indices))
            # Larger grid evidence is stored once; do not duplicate its trajectory again if promoted.
            np.savez_compressed(d/f'reference_{bigbounds[0]}_{bigbounds[1]}.npz',t=times,state=b,n=np.array(big.indices))
            print('REFERENCE',case,bounds,'rel=',row['trajectory_rel_difference'],'pass=',row['pass'],flush=True)
        if qualified:selected=bounds;break
    record={'status':'PASS' if selected else 'REFERENCE_CALIBRATION_FAILED','selected_common_truncation':selected,
        'comparisons':rows,'seconds':time.perf_counter()-start,'before_learning':True,'no_accuracy_selection':True,
        'design_sha256':sha(frozen),'grid_validation_complete':False}
    if not selected:write(existing,record);return record
    gridrows=[]
    from .vendor.r9_evaluate import field_metrics,reconstruct
    for case in CASES:
        cfg=configuration(case,selected);d=out/case;diag=diagonalize(cfg)
        times=np.linspace(0,cfg.t_max,201);state=trajectory(diag,times)
        times2=np.linspace(0,cfg.t_max,401);state2=trajectory(diag,times2)
        np.savez_compressed(d/'test201.npz',t=times,state=state,n=np.array(cfg.indices))
        np.savez_compressed(d/'test401.npz',t=times2,state=state2,n=np.array(cfg.indices))
        np.savez_compressed(d/'diagonalization_evaluation_only.npz',hamiltonian=diag[0],eigenvalues=diag[1],eigenvectors=diag[2],initial=diag[3],coordinate=diag[4])
        norms=[];j2=[];rho2=[]
        for nx in (512,1024,2048):
            x=np.linspace(cfg.x_min,cfg.x_max,nx,endpoint=False);psi=reconstruct(state,cfg,x)
            density=abs(psi).square().sum(-1) if hasattr(psi,'square') else np.sum(abs(psi)**2,-1)
            current=2*(psi[...,0].conj()*psi[...,1]).real
            norms.append(cfg.length*density.mean(1));j2.append(cfg.length*(current**2).mean(1));rho2.append(cfg.length*(density**2).mean(1))
        gridrow=dict(case=case,Nx_primary=1024,Nx_check=2048,Nt_primary=201,Nt_check=401,
            norm_parseval_max=float(abs(norms[1]-np.sum(abs(state)**2,(1,2))).max()),
            norm_double_grid_max=float(abs(norms[1]-norms[2]).max()),current_squared_integral_double_grid_max=float(abs(j2[1]-j2[2]).max()),
            density_squared_integral_double_grid_max=float(abs(rho2[1]-rho2[2]).max()),
            test201_in_test401_max=float(abs(state-state2[::2]).max()),
            maximum_reference_bilinear_angular_step=float(2*abs(diag[1]).max()*cfg.t_max/200),
            time_Nyquist_reference_resolved=bool(2*abs(diag[1]).max()*cfg.t_max/200<math.pi),
            physical_grid_no_alias=bool(1024>2*(selected[1]-selected[0])))
        free_diag=diagonalize(replace(cfg,potential_harmonics=((1,0.),)))
        diagnostics=[]
        for ts,ref in ((times,state),(times2,state2)):
            met,_,_=field_metrics(trajectory(free_diag,ts),ref,cfg,ts);diagnostics.append(met)
        for k in ('spinor_rel_l2_gauge_aligned','current_rel_l2','modal_population_rel_l2','fidelity_mean'):
            gridrow[k+'_201_401_abs_difference']=abs(diagnostics[0][k]-diagnostics[1][k])
        gridrow['grid_pass']=bool(max(gridrow[k] for k in ('norm_parseval_max','norm_double_grid_max','current_squared_integral_double_grid_max','density_squared_integral_double_grid_max','test201_in_test401_max'))<1e-12 and gridrow['physical_grid_no_alias'] and gridrow['time_Nyquist_reference_resolved'])
        gridrows.append(gridrow);write(d/'grid_validation.json',gridrow)
    pd.DataFrame(gridrows).to_csv(out/'evaluation_grid_calibration.csv',index=False)
    record.update(grid_validation_complete=True,grid_checks=gridrows,seconds=time.perf_counter()-start)
    if not all(r['grid_pass'] for r in gridrows):record['status']='EVALUATION_GRID_CALIBRATION_FAILED'
    write(existing,record);print('REFERENCE_CALIBRATION',record['status'],flush=True);return record

if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result=prepare()
    if result['status']!='PASS':raise SystemExit(2)
