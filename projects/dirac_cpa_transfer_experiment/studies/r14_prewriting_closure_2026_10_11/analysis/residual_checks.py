"""Trained-state JVP plus independent spectral convolution and residual-table audit."""
from __future__ import annotations
import json,time,hashlib
import numpy as np
import pandas as pd
import torch
from plan import BASE,write,sha

def digest_state(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):digest.update(name.encode());digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()

def independent_hamiltonian(state,cfg):
    k=np.asarray(cfg.wave_numbers);result=np.empty_like(state)
    result[...,0]=cfg.mass*state[...,0]+k[None]*state[...,1]
    result[...,1]=k[None]*state[...,0]-cfg.mass*state[...,1]
    ghosts={};n=np.asarray(cfg.indices)
    for harmonic,amplitude in cfg.potential_harmonics:
        shift=int(round(harmonic*cfg.coupling_wave_number*cfg.length/(2*np.pi)));coefficient=amplitude/2
        result[:,:-shift]+=coefficient*state[:,shift:];result[:,shift:]+=coefficient*state[:,:-shift]
        for sign in (-1,1):
            for index,destination in enumerate(n+sign*shift):
                if destination<n[0] or destination>n[-1]:
                    if int(destination) not in ghosts:ghosts[int(destination)]=np.zeros((len(state),2),dtype=np.complex128)
                    ghosts[int(destination)]+=coefficient*state[:,index]
    outside=np.stack([ghosts[key] for key in sorted(ghosts)],axis=1)
    return result,np.linalg.norm(outside,axis=(1,2))

def close(a,b):return bool(np.isclose(a,b,rtol=2e-10,atol=2e-11))

def inspect(metric_path):
    from closure.config import configuration
    from closure.models import Model
    target=metric_path.parent;checkpoint=target/'model_state.pt';raw_path=target/'prediction.npz';tick=time.perf_counter()
    saved=torch.load(checkpoint,weights_only=False);job=saved['job'];cfg=configuration(job['case'],saved['bounds'])
    with np.load(raw_path) as raw:times,archived=raw['residual_sample_t'],raw['residual_sample_complex']
    model=Model(cfg,job['group'],job['seed'],saved['job'].get('options'));model.load_state_dict(saved['state_dict']);model.eval()
    before=digest_state(model);residuals=[];norm_rate=[];exterior=[];predictions=[]
    for start in range(0,len(times),32):
        values=times[start:start+32];k=torch.tensor(cfg.wave_numbers).repeat(len(values)).reshape(-1,1)
        t=torch.tensor(values).repeat_interleave(cfg.mode_count).reshape(-1,1)
        value,tangent=torch.autograd.functional.jvp(lambda clock:model(k,clock),t,torch.ones_like(t),create_graph=False)
        state=value.detach().numpy().reshape(len(values),cfg.mode_count,2);dt=tangent.detach().numpy().reshape(state.shape)
        h,outside=independent_hamiltonian(state,cfg);res=1j*dt-h
        residuals.append(res);exterior.extend(outside);norm_rate.extend(2*np.sum(state.conj()*res,(1,2)).imag);predictions.append(state)
    independent=np.concatenate(residuals);prediction=np.concatenate(predictions)
    maximum=float(np.max(abs(independent-archived)));scale=float(np.max(abs(archived)))
    table=pd.read_csv(target/'residual_integral.csv');by_mode=pd.read_csv(target/'residual_by_mode.csv');native=json.loads(metric_path.read_text(encoding='utf-8'))
    t=table.t.to_numpy();norm=table.residual_l2.to_numpy();ext=table.exterior_residual_l2.to_numpy()
    increments=(norm[1:]+norm[:-1])*np.diff(t)/2;integral=np.r_[0.,np.cumsum(increments)]
    coarse_t=t[::2];coarse_n=norm[::2];coarse=float(np.sum((coarse_n[1:]+coarse_n[:-1])*np.diff(coarse_t)/2))
    area=lambda values:float(np.sum((values[1:]+values[:-1])*np.diff(t)/2))
    summaries=dict(fine_unweighted_pde_loss=float(np.mean(norm**2)/(2*cfg.length)),residual_integral_fine=float(integral[-1]),
        residual_integral_coarse=coarse,residual_quadrature_relative_difference=abs(float(integral[-1])-coarse)/max(float(integral[-1]),1e-300),
        exterior_residual_leakage_max_l2_norm=float(ext.max()),exterior_residual_integral=area(ext),full_physical_residual_integral=area(np.hypot(norm,ext)),
        max_norm_drift_fine=float(abs(table['norm'].to_numpy()-1).max()),norm_residual_projection_max=float(abs(table.norm_rate_from_residual.to_numpy()).max()))
    stride=(len(table)-1)//200;samples=table.iloc[::stride]
    if len(samples)!=len(times) or np.max(abs(samples.t.to_numpy()-times))>2e-12:raise RuntimeError('Residual sample grid does not match its full table')
    checks={key:close(value,native[key]) for key,value in summaries.items()}
    checks.update(independent_JVP_and_convolution=maximum<2e-11+2e-10*scale,
        independent_exterior_convolution=np.allclose(exterior,samples.exterior_residual_l2.to_numpy(),rtol=2e-10,atol=2e-11),
        independent_norm_projection=np.allclose(norm_rate,samples.norm_rate_from_residual.to_numpy(),rtol=2e-10,atol=2e-11),
        residual_norm_samples=np.allclose(np.linalg.norm(independent,axis=(1,2)),samples.residual_l2.to_numpy(),rtol=2e-10,atol=2e-11),
        cumulative_integral_table=np.allclose(integral,table.residual_integral.to_numpy(),rtol=2e-10,atol=2e-11),
        mode_power_sum=close(by_mode.mean_unweighted_component_power.sum(),summaries['fine_unweighted_pde_loss']),
        mode_indices=np.array_equal(by_mode.n.to_numpy(),cfg.indices),model_state_unchanged=digest_state(model)==before,
        checkpoint_source_matches_native=sha(checkpoint)==native['checkpoint_sha256'],finite_residual=np.isfinite(independent).all())
    return dict(path=metric_path.relative_to(BASE).as_posix(),case=job['case'],group=job['group'],seed=job['seed'],step=saved['step'],
        pass_check=bool(all(checks.values())),failed_checks=[key for key,value in checks.items() if not value],
        checkpoint_sha256=sha(checkpoint),prediction_sha256=sha(raw_path),metric_sha256=sha(metric_path),
        residual_table_sha256=sha(target/'residual_integral.csv'),by_mode_table_sha256=sha(target/'residual_by_mode.csv'),script_sha256=sha(__file__),
        independent_sample_times=len(times),independent_unit_spinor_residual_points=len(times)*cfg.mode_count,
        full_table_time_points=len(table),maximum_residual_abs_difference=maximum,maximum_archived_residual=scale,
        independent_exterior_max_difference=float(np.max(abs(np.array(exterior)-samples.exterior_residual_l2.to_numpy()))),
        metric_differences={key:abs(value-native[key]) for key,value in summaries.items()},evaluation_wall_seconds=time.perf_counter()-tick,
        no_reference_trajectory_loaded=True,no_optimizer_or_weight_update=True,discrete_residual_audit_is_not_continuous_error_certificate=True,
        norm_projection_is_not_dynamics_accuracy=True)

def main():
    from closure.train import verify_freeze
    verify_freeze();out=BASE/'analysis_outputs/residual_checks';out.mkdir(exist_ok=True);rows=[];missing=[]
    for directory in (BASE/'formal/coupled',BASE/'development'):
        for metric in sorted(directory.rglob('metrics.json')):
            native=json.loads(metric.read_text(encoding='utf-8'));job=native['job'];step=native['step']
            if step not in (20000,26000,job['budget']):continue
            if not (metric.parent/'residual_integral.csv').exists():missing.append(metric.relative_to(BASE).as_posix());continue
            key=metric.parent.relative_to(BASE).as_posix().replace('/','__');target=out/f'{key}.json'
            if target.exists():
                result=json.loads(target.read_text(encoding='utf-8'))
                if result['script_sha256']!=sha(__file__) or result['checkpoint_sha256']!=sha(metric.parent/'model_state.pt') or result['metric_sha256']!=sha(metric):
                    raise RuntimeError('Existing residual audit provenance changed; preserve and investigate '+str(target))
            else:result=inspect(metric);write(target,result)
            rows.append(result)
            print('RESIDUAL_JVP_AUDIT',job['case'],job['group'],job['seed'],step,result['pass_check'],flush=True)
            if not result['pass_check']:raise RuntimeError('Independent residual mismatch; preserve '+str(target))
    pd.DataFrame([{k:v for k,v in row.items() if k not in ('metric_differences','failed_checks')} for row in rows]).to_csv(BASE/'analysis_outputs/independent_residual_checks.csv',index=False)
    report=dict(updated_utc=time.time(),status='PASS_AVAILABLE_FIXED_ENDPOINTS' if rows and not missing else 'MISSING_OR_LIMITATION',
        trained_endpoints_checked=len(rows),missing=missing,independent_JVP_and_hamiltonian_convolution=True,
        model_states_unchanged=True,stored_residual_samples_per_endpoint=201,all_full_tables_independently_summed=True,
        no_training_budget_or_success_rule_changes=True,whole_study_complete=False,
        norm_projection_agreement_is_structural_not_dynamics_accuracy=True,discrete_checks_not_continuous_certificates=True)
    write(BASE/'analysis_outputs/residual_audit.json',report);print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    import psutil
    psutil.Process().cpu_affinity([6]);torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
