from __future__ import annotations

import json
import itertools
import numpy as np
import pandas as pd
from .r13_config import RAW,ROOT,THRESHOLDS,EXPLORATION_SEEDS,VALIDATION_SEEDS
from .r13_plan import STAGES,jobs_for,directory_for
from .r13_selection import absolute_success
from .freeze_r7_phase_release import write_json,sha256

METRICS=tuple(THRESHOLDS)+('density_rel_l2','active_coherence_rel_l2','branch_complex_coefficient_rel_l2',
    'predicted_max_occupation_outside_initial_support','plus_center_rmse','minus_center_rmse',
    'plus_width_relative_l2','minus_width_relative_l2','residual_integral_fine',
    'carrier_free_pde_residual_rms','initial_weighted_unwrapped_phase_rms','initial_weighted_unit_spinor_residual_rms',
    'coupled_reference_weighted_free_phase_rms','training_seconds','coupled_training_seconds','carrier_training_seconds',
    'evaluation_seconds','peak_rss_mib','training_spinor_residual_points','full_attributed_pipeline_seconds',
    'unique_incurred_pipeline_seconds','setup_seconds','io_seconds')

def all_jobs():
    return [j for stage in STAGES for seed in (VALIDATION_SEEDS if stage=='independent_validation' else EXPLORATION_SEEDS) for j in jobs_for(stage,seed)]

def collect():
    coupled=[];carriers=[];snapshots=[];ledger=[];attempts=[]
    for job in all_jobs():
        directory=directory_for(job)
        path=directory/'metrics.json'
        status='COMPLETE' if path.exists() else ('NUMERICAL_FAILURE' if (directory/'failure.json').exists() else
            ('DEPENDENCY_FAILED' if (directory/'dependency_failure.json').exists() else 'INCOMPLETE'))
        attempts.append(job|dict(status=status,directory=directory.relative_to(ROOT).as_posix()))
        if status!='COMPLETE':continue
        record=json.loads(path.read_text(encoding='utf-8'))
        # Immutable endpoint metrics are written before their CSV/NPZ export.
        # The separate execution receipt includes that export. Use the receipt
        # for cost statistics without altering scientific endpoint measurements.
        receipt=json.loads((directory/'execution_cost.json').read_text(encoding='utf-8'))
        pre_receipt=None
        if job['type']=='coupled' and job['carrier_kind']!='exact':
            pre_receipt=json.loads((ROOT/job['pretrain_source']/'execution_cost.json').read_text(encoding='utf-8'))
        pre_pipeline=pre_receipt['unique_incurred_pipeline_seconds'] if pre_receipt else 0.
        record['pipeline_cost_endpoint_metrics_before_export_seconds']=record['full_attributed_pipeline_seconds']
        record.update(unique_incurred_pipeline_seconds=receipt['unique_incurred_pipeline_seconds'],
            full_attributed_pipeline_seconds=receipt['unique_incurred_pipeline_seconds']+pre_pipeline,
            io_seconds=receipt['io_seconds'],
            pipeline_cost_observation_boundary='after final snapshot export; before root artifact copies and module startup')
        record.update(alias=False,directory=directory.relative_to(ROOT).as_posix(),metrics_sha256=sha256(path))
        (carriers if job['type']=='carrier' else coupled).append(record)
        ledger.append({k:v for k,v in record.items() if isinstance(v,(int,float,str,bool)) or v is None}|
            dict(job_type=job['type'],unique_incurred_training_seconds=record['incurred_this_job_training_seconds']))
        for p in sorted((directory/'snapshots').glob('*/metrics.json')):
            m=json.loads(p.read_text(encoding='utf-8'));m.update(alias=False,job_type=job['type'],directory=p.parent.relative_to(ROOT).as_posix())
            m['full_attributed_pipeline_seconds']=m['unique_incurred_pipeline_seconds']+pre_pipeline
            m['pipeline_cost_observation_boundary']='after evaluation before current snapshot export; previous exports included'
            if m['executed_steps']==record['executed_steps']:
                m['full_attributed_pipeline_seconds']=record['full_attributed_pipeline_seconds']
                m['pipeline_cost_observation_boundary']=record['pipeline_cost_observation_boundary']
            snapshots.append(m)
    # The E factorial comparator is the same prescribed endpoint, not a new attempt.
    for m in list(coupled):
        if m['stage']=='exact_envelopes' and m['case']=='T10' and m['group']=='pde_only':
            coupled.append(m|dict(stage='T10_factorial',group='E_pde_only_alias',model='E_pde_only_alias',alias=True,
                alias_source_directory=m['directory']))
        if (m['stage']=='exact_envelopes' and m['case']=='T10' and m['envelope_kind']==
            json.loads((RAW/'envelope_selection.json').read_text(encoding='utf-8'))['kind']):
            coupled.append(m|dict(stage='T10_factorial',group='E_repaired_envelope_alias',model='E_repaired_envelope_alias',
                alias=True,alias_source_directory=m['directory']))
    return pd.DataFrame(coupled),pd.DataFrame(carriers),pd.DataFrame(snapshots),pd.DataFrame(ledger),attempts

def summaries(frame,keys):
    records=[]
    for (stage,case,group),f in frame.groupby(['stage','case','group'],sort=False):
        row=dict(stage=stage,case=case,group=group,n=len(f),alias=bool(f.alias.any()))
        for key in keys:
            if key in f and f[key].notna().any():
                row[key+'_mean']=float(f[key].mean());row[key+'_population_sd']=float(f[key].std(ddof=0))
        seeds=VALIDATION_SEEDS if stage=='independent_validation' else EXPLORATION_SEEDS
        if all(k in f for k in THRESHOLDS):
            check=absolute_success(f,seeds=seeds,required_joint=4 if stage=='independent_validation' else 2)
            row.update(joint_pass_count=check['joint_pass_count'],complete=check['complete'],
                all_mean_criteria=all(check['mean_checks'].values()),
                absolute_success=check['pass_check'] if stage=='independent_validation' else None,
                exploratory_gate=check['pass_check'] if stage!='independent_validation' else None)
        records.append(row)
    return pd.DataFrame(records)

def paired_comparisons(frame,carrier=False):
    individual=[];pooled=[]
    for (stage,case),f in frame.groupby(['stage','case'],sort=False):
        groups=list(f.group.unique());baselines=[]
        if carrier:baselines=['original']
        elif stage=='exact_envelopes':baselines=['pde_only']
        else:baselines=[g for g in ('A_original','E_pde_only','E_pde_only_alias','E_repaired_envelope_alias') if g in groups]
        for target,baseline in itertools.product(groups,baselines):
            if target==baseline:continue
            a=f[f.group==target].set_index('seed');b=f[f.group==baseline].set_index('seed')
            common=sorted(set(a.index)&set(b.index))
            if not common:continue
            for key in METRICS:
                if key not in a or key not in b or not a.loc[common,key].notna().all() or not b.loc[common,key].notna().all():continue
                x,y=a.loc[common,key],b.loc[common,key];delta=x-y
                ratios=np.divide(x.to_numpy(),y.to_numpy(),out=np.full(len(common),np.nan),where=y.to_numpy()!=0)
                for seed,d,r in zip(common,delta,ratios):
                    individual.append(dict(stage=stage,case=case,target=target,baseline=baseline,metric=key,seed=seed,
                        paired_difference=float(d),actual_ratio=float(r) if np.isfinite(r) else None))
                pooled.append(dict(stage=stage,case=case,target=target,baseline=baseline,metric=key,n=len(common),
                    paired_difference_mean=float(delta.mean()),paired_difference_population_sd=float(delta.std(ddof=0)),
                    ratio_of_means=float(x.mean()/y.mean()) if y.mean()!=0 else None,
                    mean_of_paired_ratios=float(np.mean(ratios)) if np.isfinite(ratios).all() else None,
                    favorable_pair_count=int((delta>0 if key=='fidelity_mean' else delta<0).sum())))
    return pd.DataFrame(individual),pd.DataFrame(pooled)

def aggregate():
    coupled,carriers,snapshots,ledger,attempts=collect()
    coupled.to_csv(RAW/'per_seed_metrics.csv',index=False);carriers.to_csv(RAW/'carrier_per_seed_metrics.csv',index=False)
    snapshots.to_csv(RAW/'snapshot_index.csv',index=False);ledger.to_csv(RAW/'unique_cost_ledger.csv',index=False)
    summary=summaries(coupled,METRICS);free_summary=summaries(carriers,METRICS)
    summary.to_csv(RAW/'summary_metrics.csv',index=False);free_summary.to_csv(RAW/'carrier_summary_metrics.csv',index=False)
    pairs,ratios=paired_comparisons(coupled);fp,fr=paired_comparisons(carriers,carrier=True)
    pd.concat((pairs,fp),ignore_index=True).to_csv(RAW/'paired_differences.csv',index=False)
    pd.concat((ratios,fr),ignore_index=True).to_csv(RAW/'paired_summary.csv',index=False)
    results={}
    for case in ('T10','m10'):
        f=coupled[(coupled.stage=='independent_validation')&(coupled.case==case)]
        selection=json.loads((RAW/'combination_selection.json').read_text(encoding='utf-8'))
        groups=['A_original','E_pde_only']+(['selected_combination'] if selection['selected'] else [])
        results[case]={g:absolute_success(f[f.group==g]) for g in groups}
    incomplete=[j for j in attempts if j['status']=='INCOMPLETE']
    failures=[j for j in attempts if j['status'] in ('NUMERICAL_FAILURE','DEPENDENCY_FAILED')]
    decision=dict(experiment='r13_long_time_recovery',status='INCOMPLETE' if incomplete else 'ALL_FIXED_ATTEMPTS_RECORDED',
        per_case_independent_validation=results,combination_selection=selection,
        thresholds=THRESHOLDS,population_sd_ddof=0,required_joint_pass_count=4,
        unique_carrier_jobs=sum(j['type']=='carrier' for j in attempts),unique_coupled_jobs=sum(j['type']=='coupled' for j in attempts),
        complete_jobs=sum(j['status']=='COMPLETE' for j in attempts),failed_jobs=failures,incomplete_jobs=incomplete,
        independent_validation_is_new_seeds=True,learned_success_excludes_E=True,
        unique_incurred_training_hours=float(ledger.unique_incurred_training_seconds.sum()/3600),
        unique_incurred_pipeline_hours=float(ledger.unique_incurred_pipeline_seconds.sum()/3600),
        no_posthoc_advantage_factor=True,no_cross_case_average=True,originality_claim=False,
        scope='known finite Gaussian Fourier initial data, specified weak periodic potential; no general Dirac solver claim')
    write_json(RAW/'decision.json',decision);write_json(RAW/'attempt_index.json',dict(attempts=attempts))
    return decision
