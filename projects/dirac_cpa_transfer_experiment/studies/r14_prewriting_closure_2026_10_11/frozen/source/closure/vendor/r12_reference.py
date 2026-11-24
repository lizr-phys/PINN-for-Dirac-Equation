from __future__ import annotations
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from scipy.linalg import expm
from .r12_config import RAW,ROOT,CASES,TRUNCATIONS,configuration,reference_policy
from .r12_hamiltonian import build_hamiltonian,hamiltonian_apply
from .r9_reference import initial_numpy
from .freeze_r7_phase_release import write_json,sha256

def solve(cfg,times):
    h=build_hamiltonian(cfg)
    values,vectors=np.linalg.eigh(h)
    initial=initial_numpy(cfg)
    coordinate=vectors.conj().T@initial.ravel()
    evolved=np.exp(-1j*np.asarray(times)[:,None]*values)*coordinate
    state=(evolved@vectors.T).reshape(-1,cfg.mode_count,2)
    dt=((-1j*values*evolved)@vectors.T).reshape(state.shape)
    return dict(t=np.asarray(times),n=np.asarray(cfg.indices),k=np.asarray(cfg.wave_numbers),state=state,dt=dt,
        hamiltonian=h,eigenvalues=values,eigenvectors=vectors,initial=initial)

def compare_pair(cfg,big_cfg,times):
    a,b=solve(cfg,times),solve(big_cfg,times)
    padded=np.zeros_like(b['state']);left=cfg.n_min-big_cfg.n_min
    padded[:,left:left+cfg.mode_count]=a['state']
    difference=float(np.linalg.norm(padded-b['state'])/np.linalg.norm(b['state']))
    pop=np.sum(abs(a['state'])**2,axis=2)
    h,initial=a['hamiltonian'],a['initial'].ravel()
    checks=[0,len(times)//4,len(times)//2,len(times)-1]
    errors=[np.linalg.norm(expm(-1j*times[i]*h)@initial-a['state'][i].ravel()) for i in checks]
    reverse=np.linalg.norm(expm(1j*times[-1]*h)@a['state'][-1].ravel()-initial)
    row=dict(case=cfg.case,smaller=[cfg.n_min,cfg.n_max],larger=[big_cfg.n_min,big_cfg.n_max],
        full_complex_zero_padded_relative_difference=difference,
        lower_sixteen_layers_max_total_occupation=float(pop[:,:16].sum(1).max()),
        upper_sixteen_layers_max_total_occupation=float(pop[:,-16:].sum(1).max()),
        hermiticity_max_abs=float(abs(h-h.conj().T).max()),
        norm_max_drift=float(abs(np.sum(abs(a['state'])**2,(1,2))-1).max()),
        reversal_relative_error=float(reverse),independent_expm_max_relative_error=float(max(errors)),
        spectral_pde_residual_l2_rms=float(np.linalg.norm(1j*a['dt']-hamiltonian_apply(a['state'],cfg))/np.sqrt(len(times))),
        points=len(times),time_spacing=float(times[1]-times[0]))
    row['convergence_pass']=difference<1e-6 and max(row['lower_sixteen_layers_max_total_occupation'],row['upper_sixteen_layers_max_total_occupation'])<1e-8
    row['validation_pass']=row['hermiticity_max_abs']==0 and max(row[k] for k in
        ('norm_max_drift','reversal_relative_error','independent_expm_max_relative_error','spectral_pde_residual_l2_rms'))<1e-12
    row['pass_check']=bool(row['convergence_pass'] and row['validation_pass'])
    return row,a,b

def prepare():
    RAW.mkdir(parents=True,exist_ok=True)
    policy=RAW/'reference_selection_policy.json'
    if not policy.exists():write_json(policy,reference_policy())
    if (RAW/'reference_validation.json').exists():
        return json.loads((RAW/'reference_validation.json').read_text(encoding='utf-8'))
    rows=[];selected=None
    for bounds,larger in zip(TRUNCATIONS[:-1],TRUNCATIONS[1:]):
        qualified=True
        for case in CASES:
            cfg=configuration(case,bounds);times=np.linspace(0,cfg.t_max,cfg.residual_time_points)
            row,a,b=compare_pair(cfg,configuration(case,larger),times);rows.append(row)
            directory=RAW/'references'/case;directory.mkdir(parents=True,exist_ok=True)
            for data,cut in ((a,bounds),(b,larger)):
                path=directory/f'reference_n{cut[0]}_{cut[1]}.npz'
                if not path.exists():np.savez_compressed(path,**data)
            qualified &= row['pass_check']
            print('REFERENCE',case,bounds,'difference=',row['full_complex_zero_padded_relative_difference'],'pass=',row['pass_check'],flush=True)
        if qualified:selected=bounds;break
    result=dict(status='PASS' if selected else 'REFERENCE_CALIBRATION_FAILED',
        selected_common_truncation=selected,comparisons=rows,policy_sha256=sha256(policy),
        before_learning=True,no_pinn_based_selection=True,original_gaussian_unchanged=True,
        no_renormalization=True,boundary_layers=16)
    write_json(RAW/'reference_validation.json',result)
    pd.DataFrame(rows).to_csv(RAW/'reference_convergence.csv',index=False)
    if selected:
        from .r9_evaluate import field_metrics
        for case in CASES:
            cfg=configuration(case);directory=RAW/'references'/case
            fine=directory/f'reference_n{selected[0]}_{selected[1]}.npz'
            # A named selected file avoids retaining a second identical fine-grid archive.
            write_json(directory/'selected_reference.json',dict(fine_grid_path=fine.relative_to(ROOT).as_posix(),sha256=sha256(fine)))
            reference=solve(cfg,np.linspace(0,cfg.t_max,201))
            np.savez_compressed(directory/'reference_test_grid.npz',**reference)
            free=solve(replace(cfg,potential_harmonics=((1,0.),)),reference['t'])
            np.savez_compressed(directory/'diagnostic_ignoring_potential.npz',**free)
            metrics,_,_=field_metrics(free['state'],reference['state'],cfg,reference['t'])
            write_json(directory/'diagnostic_ignoring_potential.json',metrics|dict(diagnostic_only=True,no_learning_or_selection=True))
    return result
