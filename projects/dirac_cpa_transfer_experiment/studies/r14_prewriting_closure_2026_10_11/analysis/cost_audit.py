"""Recompute workload and auxiliary attribution from configs and unique source records."""
from __future__ import annotations
import json,time
import numpy as np
import pandas as pd
from plan import BASE,write,sha
from .expected_jobs import expected_jobs

def close(a,b):return bool(np.isclose(a,b,rtol=2e-12,atol=2e-9))

def inspect_cost(path,job):
    record=json.loads(path.read_text(encoding='utf-8'));conf=json.loads((BASE/job['directory']/'config.json').read_text(encoding='utf-8'))
    step=record['step'];modes=conf['bounds'][1]-conf['bounds'][0]+1;free=job['type']=='carrier'
    checks={};points=(2 if free else 1)*step*32*modes;source={}
    if job.get('source'):
        source=json.loads((BASE/job['source']/'complete.json').read_text(encoding='utf-8'))
        checks['source_completed_6000']=source['step']==6000 and source['fixed_endpoint_reached']
        checks['source_checkpoint_SHA']=conf['source_sha256']==sha(BASE/job['source']/'model_state.pt')
        checks['source_seed']=source['job']['seed']==job['seed']
        checks['source_reuse_label']=bool(job.get('reuse_source'))==bool(record['carrier_pretraining_reused'])
    prepoints=source.get('unit_spinor_residual_points',0)
    checks['own_residual_work']=record['unit_spinor_residual_points']==points
    checks['attributed_residual_work']=record['attributed_unit_spinor_residual_points']==points+prepoints
    checks['complex_scalar_work']=record['complex_scalar_residual_components']==2*(points+prepoints)
    checks['free_steps']=record['free_steps']==(step if free else (6000 if source else 0))
    checks['coupled_steps']=record['coupled_steps']==(0 if free else step)
    for unit in ('wall','cpu'):
        own=record[f'training_{unit}_seconds'];pre=source.get(f'training_{unit}_seconds',0.)
        checks[f'{unit}_source_attribution']=close(record[f'source_pretraining_{unit}_seconds'],pre)
        checks[f'{unit}_complete_attribution']=close(record[f'attributed_training_{unit}_seconds'],own+pre)
    own_pipeline=sum(record[k] for k in ('training_wall_seconds','evaluation_seconds','checkpoint_io_seconds','setup_seconds'))
    checks['own_pipeline_sum']=close(record['incurred_pipeline_seconds'],own_pipeline)
    checks['attributed_pipeline_sum']=close(record['attributed_pipeline_seconds'],own_pipeline+source.get('incurred_pipeline_seconds',0.))
    checks['parameter_partition']=record['total_parameters']==record['active_parameters']+record['frozen_parameters']
    checks['finite_positive_costs']=all(np.isfinite(record[k]) and record[k]>=0 for k in
        ('training_wall_seconds','training_cpu_seconds','evaluation_seconds','checkpoint_io_seconds','setup_seconds','peak_rss_bytes'))
    # Full endpoint evaluation workload is outside training cost and counted separately.
    metric_path=path.parent/'metrics.json';evaluation_points=None;inference_points=None
    if metric_path.exists():
        metric=json.loads(metric_path.read_text(encoding='utf-8'))
        if 'inference_seconds_samples' in metric:
            checks['seven_inference_samples']=len(metric['inference_seconds_samples'])==7
            checks['inference_median']=close(metric['inference_seconds_median'],np.median(metric['inference_seconds_samples']))
            inference_points=201*modes;checks['inference_work']=metric['inference_spinor_points']==inference_points
        if 'evaluation_coupled_residual_points' in metric:
            expected=(8001 if job['case']=='T10' else 4001)*modes if step in (20000,26000,job['budget']) else 0
            # Development endpoints also use the full independent residual grid.
            checks['evaluation_coupled_work']=metric['evaluation_coupled_residual_points']==expected
        evaluation_points=metric.get('evaluation_coupled_residual_points',0)+metric.get('evaluation_carrier_unit_spinor_residual_points',0)
    return dict(path=path.relative_to(BASE).as_posix(),case=job['case'],group=job.get('group',job.get('kind')),stage=job['stage'],
        seed=job['seed'],step=step,pass_check=all(checks.values()),failed_checks=';'.join(k for k,v in checks.items() if not v),
        independently_computed_own_residual_points=points,independently_computed_auxiliary_points=prepoints,
        evaluation_unit_spinor_residual_points=evaluation_points,inference_points_per_call=inference_points,
        source_sha256=conf.get('source_sha256'),cost_record_sha256=sha(path))

def tuning_audit(jobs):
    selection=json.loads((BASE/'frozen/baseline_selection.json').read_text(encoding='utf-8'));rows=[];costrows=[]
    for j in jobs:
        if j['stage']!='development':continue
        p=BASE/j['directory'];m=p/'snapshots/04000/metrics.json';f=p/'failure.json';c=p/'complete.json'
        if m.exists():
            a=json.loads(m.read_text(encoding='utf-8'));score=a['spinor_rel_l2_gauge_aligned']/.10+a['current_rel_l2']/.15+a['modal_population_rel_l2']/.10+(1-a['fidelity_mean'])/.02
        elif f.exists():score=None
        else:raise RuntimeError('Development attempt is not complete: '+j['directory'])
        rows.append(dict(case=j['case'],family=j['group'],candidate=j['candidate'],seed=j['seed'],score=score))
        if c.exists():
            x=json.loads(c.read_text(encoding='utf-8'));costrows.append(dict(case=j['case'],family=j['group'],candidate=j['candidate'],seed=j['seed'],
                **{k:x[k] for k in ('training_wall_seconds','training_cpu_seconds','incurred_pipeline_seconds','unit_spinor_residual_points','peak_rss_bytes')}))
        elif f.exists():
            x=json.loads(f.read_text(encoding='utf-8'));costrows.append(dict(case=j['case'],family=j['group'],candidate=j['candidate'],seed=j['seed'],
                training_wall_seconds=x.get('observed_training_wall_seconds'),training_cpu_seconds=x.get('observed_training_cpu_seconds'),
                incurred_pipeline_seconds=None,unit_spinor_residual_points=None,peak_rss_bytes=None))
    families=('siren','fourier')
    def rank(family,case=None):
        result=[]
        for i in range(3):
            x=[r for r in rows if r['family']==family and r['candidate']==i and (case is None or r['case']==case)]
            finite=[r['score'] for r in x if r['score'] is not None]
            result.append((sum(r['score'] is None for r in x),np.mean(finite) if finite else float('inf'),i))
        return min(result)
    percase={c:{f:rank(f,c)[2] for f in families} for c in ('T10','m10')}
    globalpick={f:rank(f)[2] for f in families};strongest=min(families,key=lambda f:(rank(f)[0],rank(f)[1],families.index(f)))
    passed=percase==selection['per_case_candidate'] and globalpick==selection['global_candidate'] and strongest==selection['strongest_family']
    return dict(status='PASS' if passed else 'FAIL',development_attempts=24,recomputed_per_case_candidate=percase,
        recomputed_global_candidate=globalpick,recomputed_strongest_family=strongest,selection_sha256=sha(BASE/'frozen/baseline_selection.json'),
        no_formal_outcomes_used=True,no_extra_candidates=True),pd.DataFrame(costrows)

def main():
    jobs=expected_jobs();rows=[];missing=[]
    for j in jobs:
        directory=BASE/j['directory']
        for p in sorted(directory.glob('snapshots/*/cost.json')):rows.append(inspect_cost(p,j))
        c=directory/'complete.json'
        if c.exists():rows.append(inspect_cost(c,j))
        elif not (directory/'failure.json').exists():missing.append(j['directory'])
    tuning,tuningcost=tuning_audit(jobs);out=BASE/'analysis_outputs';out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'independent_cost_workload_checks.csv',index=False)
    tuningcost.to_csv(out/'development_costs_all_candidates.csv',index=False);write(out/'baseline_selection_audit.json',tuning)
    report=dict(updated_utc=time.time(),status='PASS_AVAILABLE_COST_RECORDS' if all(r['pass_check'] for r in rows) and tuning['status']=='PASS' else 'FAIL',
        cost_records_checked=len(rows),mismatches=[r for r in rows if not r['pass_check']],missing_attempts=missing,
        all_prescribed_cost_records_present=not missing,entire_study_complete=False,
        attribution_does_not_replace_unique_incurred_ledger=True,concurrent_wall_is_not_isolated_speedup=True,
        possible_uncheckpointed_replay_work_not_assumed_zero=True)
    write(out/'independent_cost_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='missing_attempts'},ensure_ascii=False))

if __name__=='__main__':main()
