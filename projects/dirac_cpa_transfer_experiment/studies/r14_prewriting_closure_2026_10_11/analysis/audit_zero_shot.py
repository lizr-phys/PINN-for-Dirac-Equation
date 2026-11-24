"""Direct-space recalculation of all 45 unchanged-model diagnostics; missing stays missing."""
from __future__ import annotations
import json,time
import numpy as np
import pandas as pd
from plan import BASE,SEEDS,EXTENSION,THRESHOLDS,write,sha

def main():
    from closure.config import configuration
    from closure.vendor.r9_evaluate import field_metrics
    selected=json.loads((BASE/'frozen/baseline_selection.json').read_text(encoding='utf-8'))
    groups=('learned_hard','analytic_hard',selected['strongest_family']);out=BASE/'zero_shot'
    out.mkdir(exist_ok=True);rows=[];bad=[];series=[]
    for case in EXTENSION:
        cfg=configuration(case);reference_path=BASE/'references'/case/'test201.npz'
        with np.load(reference_path) as reference_file:reference,times=reference_file['state'],reference_file['t']
        for group in groups:
            for seed in SEEDS:
                path=out/f'{case}_{group}_seed{seed}.json';raw_path=path.with_suffix('.npz')
                row=dict(case=case,group=group,seed=seed,pass_check=False,diagnostic_executed=False,
                    status='INCOMPLETE',source_accuracy_failure=False,seed_joint_pass=False)
                if not path.exists():rows.append(row);continue
                try:
                    record=json.loads(path.read_text(encoding='utf-8'))
                    row['metric_record_sha256']=sha(path)
                    if record['status']=='SOURCE_NUMERICAL_OR_DEPENDENCY_FAILURE':
                        source_failure=BASE/record['source_failure']
                        expected_source=BASE/'formal/coupled/m5T5'/group/f'seed_{seed}'/'failure.json'
                        failure=json.loads(source_failure.read_text(encoding='utf-8'))
                        if source_failure!=expected_source or sha(source_failure)!=record['source_failure_sha256'] or failure['status'] not in ('NUMERICAL_FAILURE','DEPENDENCY_FAILED'):
                            raise RuntimeError('Invalid source failure provenance')
                        if record.get('inference_executed') or record.get('new_training_performed'):
                            raise RuntimeError('Failed source was incorrectly marked as evaluated or retrained')
                        row.update(status='PRESERVED_SOURCE_FAILURE',pass_check=True,source_accuracy_failure=True,
                            source_failure_sha256=sha(source_failure));rows.append(row);continue
                    if record['status']!='EVALUATED_UNCHANGED_MODEL':raise RuntimeError('Unknown zero-shot result status')
                    source_checkpoint=BASE/record['source_checkpoint'];budget=26000 if group in ('siren','fourier') else 20000
                    expected_checkpoint=BASE/'formal/coupled/m5T5'/group/f'seed_{seed}'/'model_state.pt'
                    if source_checkpoint!=expected_checkpoint or record['source_step']!=budget or sha(source_checkpoint)!=record['source_sha256']:
                        raise RuntimeError('Wrong terminal checkpoint or source SHA')
                    if sha(raw_path)!=record['prediction_sha256']:raise RuntimeError('Changed raw diagnostic output')
                    with np.load(raw_path) as raw:pred,n,t=raw['prediction'],raw['n'],raw['t']
                    if not np.array_equal(t,times) or not np.array_equal(n,cfg.indices) or not np.isfinite(pred).all() or pred.shape!=reference.shape:
                        raise RuntimeError('Invalid raw diagnostic grid or finite values')
                    with np.load(source_checkpoint.parent/'snapshots'/f'{budget:05d}'/'prediction.npz') as archived:
                        original,source_times=archived['prediction'],archived['t']
                    unchanged_difference=float(np.max(abs(pred-original)))
                    if not np.array_equal(t,source_times) or unchanged_difference>=2e-12:
                        raise RuntimeError('Zero-shot predictions differ from the unchanged source-model grid')
                    # Production zero-shot metrics use FFT. This audit uses the separately
                    # frozen direct Fourier sum and eigenspinor projection implementation.
                    metrics,per_time,_=field_metrics(pred,reference,cfg,t)
                    differences={k:abs(v-metrics[k]) for k,v in record.items() if k in metrics and isinstance(v,(int,float)) and v is not None and metrics[k] is not None}
                    initial_error=float(abs(pred[0]-reference[0]).max())
                    differences['initial_condition_error_max']=abs(initial_error-record['initial_condition_error_max'])
                    joint=all(np.isfinite(metrics[k]) and (metrics[k]>=limit if k=='fidelity_mean' else metrics[k]<=limit) for k,limit in THRESHOLDS.items())
                    passed=max(differences.values())<2e-11 and joint==record['seed_joint_pass'] and record['weights_and_initial_buffers_unchanged'] and not record['retrained']
                    row.update(status='PASS_INDEPENDENT_RECALCULATION' if passed else 'MISMATCH',pass_check=bool(passed),
                        diagnostic_executed=True,seed_joint_pass=bool(joint),source_accuracy_failure=not joint,
                        maximum_metric_abs_difference=max(differences.values()),unchanged_source_prediction_max_difference=unchanged_difference,
                        initial_condition_error_max=initial_error,source_sha256=sha(source_checkpoint),prediction_sha256=sha(raw_path),
                        reference_sha256=sha(reference_path),**{k:metrics[k] for k in (*THRESHOLDS,'active_coherence_rel_l2')})
                    per_time.insert(0,'case',case);per_time.insert(1,'group',group);per_time.insert(2,'seed',seed);series.append(per_time)
                    if not passed:bad.append(dict(case=case,group=group,seed=seed,differences=differences))
                except Exception as error:
                    row.update(status='ENGINEERING_OR_PROVENANCE_ERROR',error=repr(error));bad.append(dict(case=case,group=group,seed=seed,error=repr(error)))
                rows.append(row)
    result=pd.DataFrame(rows);result.to_csv(out/'independent_per_seed_metrics.csv',index=False)
    if series:pd.concat(series,ignore_index=True).to_csv(out/'independent_per_time_metrics.csv',index=False)
    summaries=[]
    for case in EXTENSION:
        for group in groups:
            d=result[(result.case==case)&(result.group==group)];evaluated=d[d.diagnostic_executed & d.pass_check]
            accounted=int(d.pass_check.sum());failed=int(sum(d.status=='PRESERVED_SOURCE_FAILURE'));all_finite=len(evaluated)==5
            s=dict(case=case,group=group,n_planned=5,n_evaluated=len(evaluated),attempts_accounted=accounted,
                source_failures=failed,missing_or_invalid=5-accounted,joint_pass_count=int(evaluated.seed_joint_pass.sum()),
                joint_success_rate=float(evaluated.seed_joint_pass.sum()/5) if accounted==5 else None,
                comparison='UNCHANGED_COMPLETE_M5T5_MODEL_NO_INITIAL_CORRECTION',zero_shot_diagnostic_not_retraining=True)
            conditions=[]
            for metric,limit in THRESHOLDS.items():
                mean=float(evaluated[metric].mean()) if all_finite else None
                s[metric+'_mean']=mean;s[metric+'_population_sd']=float(evaluated[metric].std(ddof=0)) if all_finite else None
                conditions.append(mean is not None and (mean>=limit if metric=='fidelity_mean' else mean<=limit))
            s['status']='INCOMPLETE' if accounted<5 else ('PASS' if all(conditions) and s['joint_pass_count']>=4 else 'FAIL')
            summaries.append(s)
    pd.DataFrame(summaries).to_csv(out/'summary_metrics.csv',index=False)
    assert len(rows)==45 and len(summaries)==9
    complete=all(r['pass_check'] for r in rows)
    report=dict(updated_utc=time.time(),status='PASS_ALL_45_ACCOUNTED' if complete else ('FAIL' if bad else 'WAITING_FOR_PRESCRIBED_DIAGNOSTICS'),
        prescribed=45,actual_diagnostics_recomputed=sum(r['diagnostic_executed'] and r['pass_check'] for r in rows),
        preserved_source_failures=sum(r['status']=='PRESERVED_SOURCE_FAILURE' for r in rows),
        missing_or_invalid=sum(not r['pass_check'] for r in rows),errors=bad,entire_zero_shot_validation_complete=complete,
        direct_space_audit_vs_FFT_production=True,weights_no_training_reference_is_evaluation_only=True,
        initial_condition_mismatch_preserved=True,no_survivor_mean_replaces_five_seeds=True,whole_study_complete=False)
    write(out/'independent_audit.json',report);print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    import psutil,torch
    psutil.Process().cpu_affinity([6]);torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
