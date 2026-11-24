"""Audit saved inference repetitions and their exact source checkpoints.

Recomputes costs, not predictions; the latter are independently checked by the
metric audit and by the inference child. Separate processes need not mean an
otherwise idle machine. No measurements are overwritten or selected by speed.
"""
from __future__ import annotations
import json,math,statistics,time
import pandas as pd
import numpy as np
from plan import BASE,sha,write
from .expected_jobs import expected_jobs


def main():
    output=BASE/'isolated_inference'; rows=[]; missing=[]; failures=[]; errors=[]; expected_paths=set()
    script_sha=sha(BASE/'analysis/inference_benchmark.py')
    for job in expected_jobs():
        if job['type']!='coupled' or job['stage']=='development':continue
        for step in sorted({20000,job['budget']}):
            cp=BASE/job['directory']/'snapshots'/f'{step:05d}'/'model_state.pt'
            target=output/job['case']/job['group']/f"seed_{job['seed']}"/f'step_{step}.json'
            expected_paths.add(target.relative_to(BASE).as_posix())
            if not target.exists():
                fpath=BASE/job['directory']/'failure.json'
                if fpath.exists():
                    f=json.loads(fpath.read_text(encoding='utf-8'))
                    if f['status']=='DEPENDENCY_FAILED' or (f['status']=='NUMERICAL_FAILURE' and f.get('occurred_step',0)<=step):
                        failures.append(dict(case=job['case'],group=job['group'],seed=job['seed'],step=step,
                            failure=fpath.relative_to(BASE).as_posix(),failure_sha256=sha(fpath),
                            checkpoint=cp.relative_to(BASE).as_posix(),valid_inference_not_executed=True,cost_is_missing_not_zero=True))
                        continue
                missing.append(target.relative_to(BASE).as_posix());continue
            record=json.loads(target.read_text(encoding='utf-8')); checks={}
            checks['identity']=all(record[k]==v for k,v in dict(case=job['case'],group=job['group'],stage=job['stage'],seed=job['seed'],step=step).items())
            checks['status']=record['status']=='COMPLETE_VERIFIED_INFERENCE'
            checks['checkpoint_SHA']=cp.exists() and record['checkpoint']==cp.relative_to(BASE).as_posix() and record['checkpoint_sha256']==sha(cp)
            checks['measurement_source_SHA']=record['script_sha256']==script_sha
            wall=record['inference_seconds_samples'];cpu=record['inference_cpu_seconds_samples']
            checks['seven_actual_repetitions']=len(wall)==len(cpu)==7 and record['timed_calls']==7 and record['warmup_calls']==1
            checks['valid_observed_costs']=all(math.isfinite(v) and v>0 for v in wall) and all(math.isfinite(v) and v>=0 for v in cpu)
            checks['independent_wall_median']=math.isclose(statistics.median(wall),record['inference_seconds_median'],abs_tol=1e-12,rel_tol=1e-12)
            checks['independent_CPU_median']=math.isclose(statistics.median(cpu),record['inference_cpu_seconds_median'],abs_tol=1e-12,rel_tol=1e-12)
            checks['memory_range']=record['process_peak_rss_bytes']>=record['process_rss_after_model_setup_bytes']>0
            checks['whole_process_memory_scope']=record['memory_includes_interpreter_model_loading_and_caches'] is True
            checks['model_setup_and_process_span']=math.isfinite(record['model_setup_seconds']) and record['model_setup_seconds']>0 and \
                record['parent_observed_process_wall_seconds']>=record['model_setup_seconds']+math.fsum(wall)-1e-6
            with np.load(cp.parent/'prediction.npz') as raw:
                t=raw['t'];n=raw['n'];shape=raw['prediction'].shape
            checks['full_call_grid']=len(t)==201 and shape==(201,len(n),2) and record['points_per_call']==201*len(n)
            metric=json.loads((cp.parent/'metrics.json').read_text(encoding='utf-8'))
            checks['source_metric_checkpoint']=metric['checkpoint_sha256']==record['checkpoint_sha256'] and metric['step']==step and metric['job']==job
            checks['prediction_agreement']=math.isfinite(record['maximum_difference_from_archived_prediction']) and record['maximum_difference_from_archived_prediction']<2e-12
            checks['all_carrier_cost_included']=record['all_carrier_network_calls_included'] is True and record['no_precomputed_carrier'] is True
            checks['analytic_prior_label']=record['analytic_free_prior']==(job['group']=='analytic_hard')
            checks['single_bound_core']=record['cpu_affinity']==[6] and record['threads']==1
            checks['reference_separation']=record['evaluation_reference_not_loaded'] is True
            passed=all(checks.values())
            row=dict(case=job['case'],group=job['group'],seed=job['seed'],step=step,
                path=target.relative_to(BASE).as_posix(),record_sha256=sha(target),checkpoint_sha256=record['checkpoint_sha256'],
                archived_prediction_sha256=sha(cp.parent/'prediction.npz'),metric_record_sha256=sha(cp.parent/'metrics.json'),
                inference_seconds_median=statistics.median(wall),inference_cpu_seconds_median=statistics.median(cpu),
                inference_wall_min_seconds=min(wall),inference_wall_max_seconds=max(wall),
                process_peak_rss_bytes=record['process_peak_rss_bytes'],model_setup_seconds=record['model_setup_seconds'],
                parent_observed_process_wall_seconds=record['parent_observed_process_wall_seconds'],
                measurement_file_modified_utc=target.stat().st_mtime,machine_idle_is_not_established=True,pass_check=passed,checks=checks)
            rows.append(row)
            if not passed:errors.append(dict(path=row['path'],failed_checks=[k for k,v in checks.items() if not v]))
    actual_paths={p.relative_to(BASE).as_posix() for p in output.glob('*/*/seed_*/step_*.json')}
    unexpected=sorted(actual_paths-expected_paths)
    if unexpected:errors.append(dict(unexpected_measurements=unexpected))
    all_verified=len(rows)+len(failures)==len(expected_paths)==255 and not errors
    report=dict(updated_utc=time.time(),status='FAIL' if errors else ('ALL_PRESCRIBED_INFERENCE_RECORDS_VERIFIED' if all_verified else 'PASS_AVAILABLE_INFERENCE_RECORDS'),
        prescribed_endpoints=len(expected_paths),actual_measurements_checked=len(rows),preserved_source_failures=failures,
        missing=missing,errors=errors,all_prescribed_inference_records_verified=all_verified,
        original_measurements_not_overwritten=True,separate_process_is_not_machine_idle=True,
        whole_goal_complete=False,rows=rows)
    write(output/'independent_audit.json',report)
    pd.DataFrame([{k:v for k,v in r.items() if k!='checks'} for r in rows]).to_csv(output/'independent_costs.csv',index=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','missing','preserved_source_failures')},ensure_ascii=False))
    if errors:raise RuntimeError('Independent inference cost/provenance audit failed')


if __name__=='__main__':main()
