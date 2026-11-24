from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .r12_config import RAW,ROOT,CASES,GROUPS,SEEDS,ARTIFACTS,THRESHOLDS,budget,configuration,run_directory,snapshot_directory
from .r12_decision import decision,comparisons
from .freeze_r7_phase_release import write_json

def original_metrics():
    rows=[]
    for case in CASES:
        for endpoint in ('final','coupled_20000'):
            for group in GROUPS:
                for seed in SEEDS:
                    directory=run_directory(case,group,seed) if endpoint=='final' else snapshot_directory(case,group,seed,20000)
                    if not all((directory/name).is_file() for name in ARTIFACTS+('recovery_state.pt',)):
                        raise RuntimeError('Missing fixed endpoint: '+str(directory))
                    m=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
                    rows.append(m|dict(endpoint=endpoint,snapshot_identity=f'{case}/{group}/seed_{seed}/coupled_{m["coupled_steps"]:05d}'))
    return pd.DataFrame(rows)

def aggregate():
    frame=original_metrics();frame.to_csv(RAW/'per_seed_metrics.csv',index=False)
    summaries=[]
    keys=[k for k in frame.select_dtypes(include=[np.number]).columns if k!='seed']
    for (case,endpoint,group),part in frame.groupby(['case','endpoint','model'],sort=False):
        for key in keys:
            v=part[key].dropna()
            summaries.append(dict(case=case,endpoint=endpoint,model=group,metric=key,n=5,n_valid=len(v),
                applicable=bool(len(v)),mean=float(v.mean()) if len(v) else None,
                population_sd=float(v.std(ddof=0)) if len(v) else None,
                minimum=float(v.min()) if len(v) else None,maximum=float(v.max()) if len(v) else None))
    pd.DataFrame(summaries).to_csv(RAW/'summary_metrics.csv',index=False)
    pairs,comparison=comparisons(frame);pairs.to_csv(RAW/'paired_comparisons.csv',index=False)
    comparison.to_csv(RAW/'comparison_summary.csv',index=False)
    budget_rows=[];budget_summary=[]
    for case in CASES:
        for group in GROUPS[1:]:
            a=frame[(frame.case==case)&(frame.model==group)&(frame.endpoint=='coupled_20000')].set_index('seed')
            b=frame[(frame.case==case)&(frame.model==group)&(frame.endpoint=='final')].set_index('seed')
            for key in tuple(THRESHOLDS)+('density_rel_l2','active_coherence_rel_l2','branch_complex_coefficient_rel_l2','residual_integral_fine'):
                delta=b[key]-a[key];ratio=b[key]/a[key]
                for seed in SEEDS:budget_rows.append(dict(case=case,model=group,seed=seed,metric=key,
                    before_coupled_steps=20000,after_coupled_steps=26000,before=float(a.loc[seed,key]),
                    after=float(b.loc[seed,key]),paired_difference=float(delta[seed]),actual_ratio=float(ratio[seed])))
                budget_summary.append(dict(case=case,model=group,metric=key,n=5,
                    paired_difference_mean=float(delta.mean()),paired_difference_population_sd=float(delta.std(ddof=0)),
                    ratio_of_means=float(b[key].mean()/a[key].mean()),mean_of_paired_ratios=float(ratio.mean()),
                    favorable_pair_count=int((delta>0 if key=='fidelity_mean' else delta<0).sum())))
    pd.DataFrame(budget_rows).to_csv(RAW/'budget_changes.csv',index=False)
    pd.DataFrame(budget_summary).to_csv(RAW/'budget_change_summary.csv',index=False)
    write_json(RAW/'decision.json',decision(frame))
    costs=[];curves=[];artifacts=[];index=[]
    for case in CASES:
        points=configuration(case).spectral_collocation_points
        for group in GROUPS:
            for seed in SEEDS:
                directory=run_directory(case,group,seed)
                execution=json.loads((directory/'execution_cost.json').read_text(encoding='utf-8'))
                final=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
                curve=pd.read_csv(directory/'cost_curve.csv',float_precision='round_trip')
                curve['case'],curve['model'],curve['seed']=case,group,seed;curves.append(curve)
                eval_points=carrier_points=batches=full_count=0
                for snap in sorted((directory/'snapshots').iterdir()):
                    m=json.loads((snap/'metrics.json').read_text(encoding='utf-8'))
                    eval_points+=m.get('evaluation_coupled_spinor_residual_points',0)
                    carrier_points+=m.get('evaluation_carrier_unit_spinor_residual_points',0)
                    batches+=m.get('evaluation_coupled_residual_batches',0)+m.get('evaluation_carrier_residual_batches',0)
                    full_count+=int(m['evaluation_full_residual'])
                    index.append(dict(case=case,model=group,seed=seed,coupled_steps=m['coupled_steps'],
                        total_steps=m['steps'],full_residual=m['evaluation_full_residual'],directory=snap.relative_to(ROOT).as_posix()))
                    if m['evaluation_full_residual']:
                        for name in ARTIFACTS+('recovery_state.pt',):
                            artifacts.append(dict(case=case,model=group,seed=seed,role='full_snapshot',
                                coupled_steps=m['coupled_steps'],artifact=name,path=str(snap/name)))
                for name in ARTIFACTS+('recovery_state.pt','initial_state.pt','execution_cost.json','execution_events.jsonl','cost_curve.csv'):
                    artifacts.append(dict(case=case,model=group,seed=seed,role='final_run',coupled_steps=final['coupled_steps'],artifact=name,path=str(directory/name)))
                pre,coupled=budget(group)
                costs.append(dict(case=case,model=group,seed=seed,analytic_prior=group==GROUPS[-1],**execution,
                    total_optimizer_steps=pre+coupled,carrier_steps=pre,coupled_steps=coupled,
                    training_residual_batches=pre+coupled,training_carrier_unit_spinor_residual_points=2*pre*points,
                    training_coupled_spinor_residual_points=coupled*points,training_spinor_residual_points=(2*pre+coupled)*points,
                    training_complex_scalar_residual_evaluations=2*(2*pre+coupled)*points,
                    evaluation_coupled_spinor_residual_points=eval_points,evaluation_carrier_unit_spinor_residual_points=carrier_points,
                    evaluation_residual_batches=batches,evaluation_forward_only_checkpoint_count=len(curve)-full_count,
                    cpu_affinity_mask=final['cpu_affinity_mask']))
    pd.DataFrame(costs).to_csv(RAW/'cost_and_residual_work.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(RAW/'error_cost_curves.csv',index=False)
    pd.DataFrame(artifacts).to_csv(RAW/'run_artifact_index.csv',index=False)
    pd.DataFrame(index).to_csv(RAW/'snapshot_index.csv',index=False)
    for case in CASES:
        directory=RAW/'case_statistics'/case;directory.mkdir(parents=True,exist_ok=True)
        frame[frame.case==case].to_csv(directory/'per_seed_metrics.csv',index=False)
        comparison[comparison.case==case].to_csv(directory/'comparison_summary.csv',index=False)
    return frame,decision(frame)

if __name__=='__main__':print(json.dumps(aggregate()[1],indent=2))
