"""Direct latent-phase/autograd audit; independent diagnostics, never training labels."""
import json,time
import numpy as np
import pandas as pd
import torch
from plan import BASE,write,sha
from closure.config import configuration,initial_coefficients
from closure.models import StaticCarrier,Model
from closure.vendor.r13_models import LearnedRecoveryCarrier

def inspect(checkpoint):
    saved=torch.load(checkpoint,weights_only=False);job=saved['job'];cfg=configuration(job['case'],saved['bounds'])
    if job['type']=='carrier':
        carrier=StaticCarrier(cfg,job['seed']) if job['kind']=='static_frequency' else LearnedRecoveryCarrier(cfg,job['kind'],job['seed'])
        carrier.load_state_dict(saved['state_dict'])
    else:
        if saved['step'] not in (20000,26000):return None
        model=Model(cfg,job['group'],job['seed'],job.get('options'));model.load_state_dict(saved['state_dict']);carrier=model.carrier
        if job['group'] not in ('original_pde','original_hard','static_pde','learned_hard','joint_hard'):return None
    with np.load(checkpoint.parent/'prediction.npz') as f:times=f['t'];archived_carrier=f['carrier']
    phases=[];slopes=[]
    for start in range(0,len(times),32):
        ts=times[start:start+32];k=torch.tensor(cfg.wave_numbers).repeat(len(ts)).reshape(-1,1)
        t=torch.tensor(ts).repeat_interleave(cfg.mode_count).reshape(-1,1).requires_grad_(True)
        phase=carrier.phase(k,t)
        derivatives=torch.stack([torch.autograd.grad(phase[:,branch].sum(),t,retain_graph=branch==0)[0].reshape(-1) for branch in (0,1)],-1)
        phases.append(phase.detach().numpy().reshape(len(ts),cfg.mode_count,2));slopes.append(derivatives.detach().numpy().reshape(len(ts),cfg.mode_count,2))
    phase=np.concatenate(phases);slope=np.concatenate(slopes);energy=np.sqrt(np.asarray(cfg.wave_numbers)**2+cfg.mass**2)
    target=np.stack((-energy,energy),-1);error=phase-times[:,None,None]*target[None]
    weights=abs(initial_coefficients(cfg).numpy())**2
    weighted=float(np.sqrt(np.mean(np.sum(weights[None]*error**2,(1,2)))))
    unit_residual=float(np.sqrt(.5*np.mean((slope-target[None])**2)))
    metric=json.loads((checkpoint.parent/'metrics.json').read_text(encoding='utf-8'))
    old_phase=metric.get('initial_weighted_free_phase_rms');old_residual=metric['carrier_free_residual_rms']
    direct_vs_carrier=float(np.max(abs(np.exp(1j*phase)-archived_carrier)))
    residual_difference=abs(unit_residual-old_residual)
    return dict(path=checkpoint.relative_to(BASE).as_posix(),case=job['case'],group=job.get('group',job.get('kind')),seed=job['seed'],step=saved['step'],
        checkpoint_sha256=sha(checkpoint),direct_phase_carrier_max_abs_difference=direct_vs_carrier,
        independent_unit_free_residual_rms=unit_residual,unit_free_residual_abs_difference=residual_difference,
        direct_initial_weighted_unwrapped_phase_rms=weighted,archived_weighted_phase_rms=old_phase,
        direct_vs_archived_phase_diagnostic_difference=abs(weighted-old_phase) if old_phase is not None else None,
        archived_phase_diagnostic_may_alias=bool(old_phase is not None and abs(weighted-old_phase)>1e-8),
        pass_physical_consistency=direct_vs_carrier<2e-11 and residual_difference<2e-11,
        joint_latent_phase_is_not_a_unique_physical_gauge=job.get('group')=='joint_hard',
        exact_free_phase_used_only_for_independent_evaluation=True,primary_dynamics_metrics_unchanged=True)

def main():
    rows=[]
    for root in (BASE/'formal/carriers',BASE/'formal/coupled'):
        for cp in sorted(root.glob('*/*/seed_*/snapshots/*/model_state.pt')):
            if not (cp.parent/'metrics.json').exists():continue
            row=inspect(cp)
            if row is not None:rows.append(row)
    out=BASE/'analysis_outputs';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'independent_direct_carrier_phase.csv',index=False)
    report=dict(updated_utc=time.time(),status='PASS_AVAILABLE_PHASE_DIAGNOSTICS' if rows and all(r['pass_physical_consistency'] for r in rows) else 'MISMATCH_OR_MISSING',
        snapshots_checked=len(rows),physical_consistency_mismatches=[r['path'] for r in rows if not r['pass_physical_consistency']],
        unwrap_alias_diagnostics=[r['path'] for r in rows if r['archived_phase_diagnostic_may_alias']],
        exact_unwrapped_phase_from_raw_real_network_output=True,slope_from_independent_autograd=True,
        archived_phase_metrics_preserved=True,agreement_is_not_carrier_or_coupled_dynamics_accuracy=True,
        whole_study_complete=False,no_training_or_budget_changed=True)
    write(out/'carrier_phase_audit.json',report);print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    import psutil
    psutil.Process().cpu_affinity([6]);torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
