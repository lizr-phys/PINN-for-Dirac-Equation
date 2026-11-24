"""Independent evaluation subprocess; reference arrays never enter the trainer."""
from __future__ import annotations
import argparse,json,math,time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.integrate import cumulative_trapezoid
from plan import BASE,write,sha,THRESHOLDS
from .config import configuration
from .models import Model,StaticCarrier,residual
from .vendor.r13_models import LearnedRecoveryCarrier
from .vendor.r9_evaluate import field_metrics,spectral_grid,physical_quadrature,rel
from .vendor.r13_residual import free_carrier_residual
from .vendor.r12_hamiltonian import exterior_residual

def inference_cost(model,cfg,times):
    spectral_grid(model,cfg,times)
    observations=[]
    for _ in range(7):
        tick=time.perf_counter();spectral_grid(model,cfg,times);observations.append(time.perf_counter()-tick)
    return dict(inference_seconds_median=float(np.median(observations)),inference_seconds_samples=observations,
        inference_spinor_points=len(times)*cfg.mode_count,inference_includes_all_carrier_calls=True)

def evaluate_coupled(model,cfg,checkpoint,*,full):
    directory=BASE/'references'/cfg.case
    with np.load(directory/'test201.npz') as f:times,exact=f['t'],f['state']
    predicted=spectral_grid(model,cfg,times)
    metrics,series,dense=field_metrics(predicted,exact,cfg,times)
    metrics.update(initial_condition_max_abs_error=float(abs(predicted[0]-exact[0]).max()),
        evaluation_coupled_residual_points=0,structural_total_norm=model.envelope_kind=='hard_total_norm',
        no_posthoc_normalization=True,global_constant_phase_only=True,
        analytic_prior=model.carrier_kind=='exact',reference_sha256=sha(directory/'test201.npz'))
    # Keep compact raw spectrum, not repeated copies of physical/reference arrays.
    archive=dict(t=times,n=np.asarray(cfg.indices),prediction=predicted)
    tables={}
    if full:
        q=physical_quadrature(dense['spectral_predicted_aligned'],exact,replace(cfg,test_x_points=2048))
        metrics.update(Parseval_max_error=float(abs(cfg.length*dense['density_predicted'].mean(1)-np.sum(abs(predicted)**2,(1,2))).max()),
            density_1024_2048_difference=abs(metrics['density_rel_l2']-rel(q[3],q[4])),
            current_1024_2048_difference=abs(metrics['current_rel_l2']-rel(q[5],q[6])),fourier_aliasing_absent=cfg.test_x_points>2*(cfg.n_max-cfg.n_min))
        with np.load(directory/'test401.npz') as f:ts2,ref2=f['t'],f['state']
        pred2=spectral_grid(model,cfg,ts2);m2,_,_=field_metrics(pred2,ref2,cfg,ts2)
        for k in THRESHOLDS:metrics[k+'_401']=m2[k];metrics[k+'_201_401_difference']=abs(metrics[k]-m2[k])
        archive.update(t401=ts2,prediction401=pred2)
        fine_times=np.linspace(0,cfg.t_max,cfg.residual_time_points)
        ks=torch.tensor(cfg.wave_numbers,dtype=torch.float64).reshape(-1,1)
        norms=[];normspectrum=[];power=np.zeros(cfg.mode_count);samples=[];sample_times=[];exterior_norm=[];balance=[]
        for start in range(0,len(fine_times),32):
            ts=fine_times[start:start+32];t=torch.tensor(ts).repeat_interleave(cfg.mode_count).reshape(-1,1)
            k=ks.repeat(len(ts),1)
            res=residual(model,cfg,k,t).detach().numpy()
            with torch.no_grad():raw=model(k,t).numpy().reshape(len(ts),cfg.mode_count,2)
            norms.extend(np.linalg.norm(res,axis=(1,2)));normspectrum.extend(np.sum(abs(raw)**2,(1,2)))
            ext,_,_=exterior_residual(raw,cfg);exterior_norm.extend(np.linalg.norm(ext,axis=(1,2)))
            balance.extend(2*np.sum(raw.conj()*res,(1,2)).imag)
            power+=np.sum(abs(res)**2,(0,2))/(2*cfg.length)
            stride=(len(fine_times)-1)//200
            for local,j in enumerate(range(start,start+len(ts))):
                if j%stride==0:samples.append(res[local]);sample_times.append(fine_times[j])
        norms=np.asarray(norms);exterior_norm=np.asarray(exterior_norm);fine_norm=np.asarray(normspectrum)
        integral=cumulative_trapezoid(norms,fine_times,initial=0);coarse=float(np.trapezoid(norms[::2],fine_times[::2]))
        metrics.update(fine_unweighted_pde_loss=float(np.mean(norms**2)/(2*cfg.length)),residual_integral_fine=float(integral[-1]),
            residual_integral_coarse=coarse,residual_quadrature_relative_difference=abs(float(integral[-1])-coarse)/max(float(integral[-1]),1e-300),
            exterior_residual_leakage_max_l2_norm=float(exterior_norm.max()),exterior_residual_integral=float(np.trapezoid(exterior_norm,fine_times)),
            full_physical_residual_integral=float(np.trapezoid(np.hypot(norms,exterior_norm),fine_times)),
            max_norm_drift_fine=float(abs(fine_norm-1).max()),norm_residual_projection_max=float(abs(np.asarray(balance)).max()),
            evaluation_coupled_residual_points=len(fine_times)*cfg.mode_count,residual_integral_is_continuous_certificate=False)
        tables['residual_integral.csv']=pd.DataFrame(dict(t=fine_times,residual_l2=norms,residual_integral=integral,
            exterior_residual_l2=exterior_norm,norm=fine_norm,norm_rate_from_residual=balance))
        tables['residual_by_mode.csv']=pd.DataFrame(dict(n=cfg.indices,mean_unweighted_component_power=power/len(fine_times)))
        archive.update(residual_sample_t=np.array(sample_times),residual_sample_complex=np.array(samples))
        metrics.update(inference_cost(model,cfg,times))
        if model.carrier_kind not in ('exact','none'):
            cp=[];defects=[]
            for start in range(0,len(times),32):
                ts=torch.tensor(times[start:start+32]).repeat_interleave(cfg.mode_count).reshape(-1,1);k=ks.repeat(len(ts)//cfg.mode_count,1)
                cp.append(model.carrier(k,ts).detach().numpy().reshape(-1,cfg.mode_count,2))
                defects.append(free_carrier_residual(model.carrier,cfg,k,ts).detach().numpy())
            cp=np.concatenate(cp);defect=np.concatenate(defects)
            energy=np.sqrt(np.asarray(cfg.wave_numbers)**2+cfg.mass**2);rates=np.stack((-energy,energy),-1)
            # Free phases are diagnostic; in joint training envelope-carrier gauge is not unique.
            if model.carrier_kind=='static_frequency':
                omega=model.carrier.mlp((ks-cfg.k0)/cfg.ladder_half_width).detach().numpy()
                phase=-times[:,None,None]*omega[None]-times[:,None,None]*rates[None]
            else:phase=np.unwrap(np.angle(cp),axis=0)-times[:,None,None]*rates[None]
            weights=abs(model.initial_coeff.detach().numpy())**2
            metrics.update(carrier_free_residual_rms=float(np.sqrt(np.mean(abs(defect)**2))),
                initial_weighted_free_phase_rms=float(np.sqrt(np.mean(np.sum(weights[None]*phase**2,(1,2))))),
                evaluation_carrier_unit_spinor_residual_points=2*len(times)*cfg.mode_count)
            archive.update(carrier=cp,carrier_phase_diagnostic=phase)
    metrics['seed_joint_pass']=all(metrics[k]>=v if k=='fidelity_mean' else metrics[k]<=v for k,v in THRESHOLDS.items())
    return metrics,series,archive,tables

def evaluate_carrier(carrier,cfg):
    times=np.linspace(0,cfg.t_max,201);ks=torch.tensor(cfg.wave_numbers).reshape(-1,1);ps=[];rs=[]
    for start in range(0,len(times),32):
        ts=torch.tensor(times[start:start+32]).repeat_interleave(cfg.mode_count).reshape(-1,1);k=ks.repeat(len(ts)//cfg.mode_count,1)
        ps.append(carrier(k,ts).detach().numpy().reshape(-1,cfg.mode_count,2));rs.append(free_carrier_residual(carrier,cfg,k,ts).detach().numpy())
    p=np.concatenate(ps);r=np.concatenate(rs);energy=np.sqrt(np.array(cfg.wave_numbers)**2+cfg.mass**2);rate=np.stack((-energy,energy),-1)
    # Autonomous carriers expose their phase directly, so evaluation has no unwrapping alias risk.
    if carrier.kind=='static_frequency':
        omega=carrier.mlp((ks-cfg.k0)/cfg.ladder_half_width).detach().numpy();phase=-times[:,None,None]*omega[None]-times[:,None,None]*rate[None]
    else:phase=np.unwrap(np.angle(p),axis=0)-times[:,None,None]*rate[None]
    weights=abs(__import__('closure.config',fromlist=['initial_coefficients']).initial_coefficients(cfg).numpy())**2
    met=dict(carrier_free_residual_rms=float(np.sqrt(np.mean(abs(r)**2))),initial_weighted_free_phase_rms=float(np.sqrt(np.mean(np.sum(weights[None]*phase**2,(1,2))))),
        evaluation_carrier_unit_spinor_residual_points=2*len(times)*cfg.mode_count,analytic_prior=False)
    return met,pd.DataFrame(dict(t=times,initial_weighted_phase_rmse=np.sqrt(np.sum(weights[None]*phase**2,(1,2))))),dict(t=times,carrier=p,phase_error=phase),{}

def main(directory,step):
    directory=Path(directory);checkpoint=directory/'snapshots'/f'{step:05d}'/'model_state.pt'
    saved=torch.load(checkpoint,weights_only=False);job=saved['job'];cfg=configuration(job['case'],saved['bounds'])
    start=time.perf_counter()
    if job['type']=='carrier':
        model=StaticCarrier(cfg,job['seed']) if job['kind']=='static_frequency' else LearnedRecoveryCarrier(cfg,job['kind'],job['seed'])
        model.load_state_dict(saved['state_dict']);metrics,series,archive,tables=evaluate_carrier(model,cfg)
    else:
        model=Model(cfg,job['group'],job['seed'],job.get('options'));model.load_state_dict(saved['state_dict'])
        metrics,series,archive,tables=evaluate_coupled(model,cfg,checkpoint,full=step in (20000,26000,job['budget']))
    target=checkpoint.parent;series.to_csv(target/'time_series.csv',index=False)
    for name,table in tables.items():table.to_csv(target/name,index=False)
    np.savez_compressed(target/'prediction.npz',**archive)
    metrics.update(job=job,step=step,checkpoint_sha256=sha(checkpoint),evaluation_seconds=time.perf_counter()-start)
    write(target/'metrics.json',metrics)
    print('EVALUATED',job['case'],job.get('group',job.get('kind')),job['seed'],step,flush=True)

if __name__=='__main__':
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('step',type=int);a=parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main(a.directory,a.step)
