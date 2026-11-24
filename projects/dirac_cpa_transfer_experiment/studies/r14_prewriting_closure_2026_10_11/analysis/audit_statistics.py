"""Recalculate tables from source records, independently of aggregate.py.

The frozen scientific rules are read rather than changed. Missing endpoints and
scientific failures remain explicit. No new training or checkpoint selection.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

from plan import BASE, sha, write
from .expected_jobs import expected_jobs

T975_DF4 = 2.7764451051977987


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value in (None, '', 'nan', 'NaN'): return None
    return float(value)


def boolean(value):
    return value is True or str(value).lower() == 'true'


def mean(values):
    return math.fsum(values) / len(values)


def population_sd(values):
    center = mean(values)
    return math.sqrt(math.fsum((v - center) ** 2 for v in values) / len(values))


def pair_estimates(left, right, metric):
    differences = [a - b for a, b in zip(left, right)]
    center = mean(differences)
    sd = population_sd(differences)
    radius = T975_DF4 * sd / 2.0  # sqrt(n/(n-1))/sqrt(n), n=5
    extreme = 0
    for bits in range(32):
        value = mean([d if bits & (1 << i) else -d for i, d in enumerate(differences)])
        extreme += abs(value) >= abs(center) - 1e-15
    return dict(paired_difference_mean=center, population_sd=sd,
        CI95_lower=center-radius, CI95_upper=center+radius,
        exact_sign_flip_two_sided_p=extreme/32,
        ratio_of_means=mean(left)/mean(right) if mean(right) != 0 else None,
        mean_paired_ratio=mean([a/b for a,b in zip(left,right)]) if all(right) else None,
        favorable_pairs=sum(d>0 if metric=='fidelity_mean' else d<0 for d in differences))


def main():
    out=BASE/'analysis_outputs'
    design=json.loads((BASE/'frozen/design.json').read_text(encoding='utf-8'))
    seeds=design['formal_seeds']; thresholds=design['thresholds']
    jobs=[j for j in expected_jobs() if j['type']=='coupled' and j['stage']!='development']
    planned={(j['case'],j['group'],s) for j in jobs for s in j['snapshots']}
    metric_audit=read_csv(out/'independent_metrics_per_snapshot.csv')
    audited={Path(r['path']).as_posix():r for r in metric_audit if boolean(r['pass_check'])}
    snapshots=read_csv(out/'all_snapshots_metrics.csv')
    sources={}; pending=[];errors=[]
    def check(context, field, actual, expected):
        actual=number(actual) if not isinstance(expected,(str,bool)) else actual
        if expected is None: valid=actual is None
        elif isinstance(expected,bool): valid=boolean(actual)==expected
        elif isinstance(expected,str): valid=actual==expected
        else: valid=actual is not None and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=2e-10,abs_tol=2e-11)
        if not valid: errors.append(dict(context=context,field=field,actual=actual,expected=expected))
    for row in snapshots:
        if row['stage']=='development': continue
        path=Path(row['path']); native_path=BASE/path
        native=json.loads(native_path.read_text(encoding='utf-8'))
        cost_path=native_path.parent/'cost.json';cost=json.loads(cost_path.read_text(encoding='utf-8'))
        j=native['job']; key=(j['case'],j['group'],int(j['seed']),int(native['step']))
        if key in sources: errors.append(dict(context=str(key),field='duplicate_source'));continue
        independent=audited.get(path.as_posix())
        fresh=independent is not None and independent['metric_record_sha256']==sha(native_path)
        if not fresh: pending.append(path.as_posix())
        for metric, value in native.items():
            if metric in row and metric not in cost and isinstance(value,(float,int)) and not isinstance(value,bool):
                check(path.as_posix(),metric,row[metric],value)
        check(path.as_posix(),'snapshot_evaluator_computation_seconds',row.get('snapshot_evaluator_computation_seconds'),native.get('evaluation_seconds'))
        check(path.as_posix(),'training_cumulative_evaluator_process_wall_seconds',row.get('training_cumulative_evaluator_process_wall_seconds'),cost.get('evaluation_seconds'))
        for metric,value in cost.items():
            if metric in row and isinstance(value,(float,int)) and not isinstance(value,bool):
                check(path.as_posix(),metric,row[metric],value)
        # Keep frozen observations as the statistical inputs. Independently
        # validate them first; re-rounding machine-precision norm differences
        # would change arbitrary pair signs without changing dynamics accuracy.
        values={k:v for k,v in native.items() if isinstance(v,(float,int)) and not isinstance(v,bool)}
        if fresh:
            for metric in list(thresholds)+['active_coherence_rel_l2']:
                check(path.as_posix(),'independent_'+metric,native[metric],float(independent[metric]))
        values.update({k:v for k,v in cost.items() if isinstance(v,(float,int)) and not isinstance(v,bool)})
        native_joint=all(native[k]>=v if k=='fidelity_mean' else native[k]<=v for k,v in thresholds.items())
        check(path.as_posix(),'seed_joint_pass',row['seed_joint_pass'],native_joint)
        values.update(seed_joint_pass=native_joint,audited=fresh)
        sources[key]=values
    for j in jobs:
        for step in j['snapshots']:
            p=BASE/j['directory']/'snapshots'/f'{step:05d}'/'metrics.json'
            key=(j['case'],j['group'],j['seed'],step)
            if p.exists() and key not in sources:
                pending.append(p.relative_to(BASE).as_posix())
    failures=[]
    for j in jobs:
        f=BASE/j['directory']/'failure.json'
        if f.exists(): failures.append(json.loads(f.read_text(encoding='utf-8')))
    summary=read_csv(out/'summary_metrics.csv')
    actual_keys={(r['case'],r['group'],int(r['step'])) for r in summary}
    if actual_keys!=planned or len(summary)!=len(planned):
        errors.append(dict(context='summary',field='expected_endpoint_inventory',expected_count=len(planned),actual_count=len(summary)))
    complete_summaries=0
    for row in summary:
        case,group,step=row['case'],row['group'],int(row['step']); context=f'{case}/{group}/{step}'
        available={s:sources[(case,group,s,step)] for s in seeds if (case,group,s,step) in sources}
        affected=[f for f in failures if f['job']['case']==case and f['job'].get('group')==group and
            (f['status']=='DEPENDENCY_FAILED' or f.get('occurred_step',0)<=step)]
        failed={f['job']['seed'] for f in affected}; missing=set(seeds)-set(available)-failed
        complete=len(available)==5 and not affected
        finite=all(math.isfinite(v[k]) for v in available.values() for k in thresholds)
        audited_complete=complete and all(v['audited'] for v in available.values())
        passes=sum(v['seed_joint_pass'] for v in available.values());checks=[]
        for metric,threshold in thresholds.items():
            values=[available[s][metric] for s in seeds if s in available]
            group_mean=mean(values) if complete and finite else None
            check(context,metric+'_mean',row[metric+'_mean'],group_mean)
            check(context,metric+'_population_sd',row[metric+'_population_sd'],population_sd(values) if complete and finite else None)
            check(context,metric+'_available_finite_mean',row[metric+'_available_finite_mean'],mean(values) if values and finite else None)
            checks.append(group_mean is not None and (group_mean>=threshold if metric=='fidelity_mean' else group_mean<=threshold))
        passed=complete and finite and all(checks) and passes>=design['required_joint_successes']
        expected_status='FAIL' if affected else ('INCOMPLETE' if not complete else ('INVALID_METRICS' if not finite else
            ('PASS' if passed else 'FAIL') if audited_complete else ('PASS_PENDING_AUDIT' if passed else 'FAIL_PENDING_AUDIT')))
        check(context,'status',row['status'],expected_status)
        for field,value in dict(n=len(available),joint_pass_count=passes,attempts_accounted_at_endpoint=len(set(available)|failed),
            all_five_present=complete,independent_metric_audit_complete=audited_complete,
            absolute_success=passed and audited_complete,computed_absolute_success_before_independent_audit=passed,
            joint_success_rate=passes/5 if len(set(available)|failed)==5 else None,
            success_rate_denominator=5 if len(set(available)|failed)==5 else None,
            missing_seeds=','.join(map(str,sorted(missing))),failed_seeds=','.join(map(str,sorted(failed)))).items():
            check(context,field,row[field],value)
        complete_summaries+=complete
    pairs=read_csv(out/'paired_statistics.csv');seedpairs=read_csv(out/'paired_per_seed.csv') if (out/'paired_per_seed.csv').stat().st_size>3 else []
    selection=json.loads((BASE/'frozen/baseline_selection.json').read_text(encoding='utf-8'))
    compared_metrics=list(thresholds)+['active_coherence_rel_l2','attributed_training_wall_seconds',
        'attributed_training_cpu_seconds','inference_seconds_median','peak_rss_bytes','attributed_unit_spinor_residual_points']
    controls=[('learned_hard','analytic_hard'),('static_pde','original_pde'),('learned_hard','original_hard'),
        ('original_hard','original_pde'),('learned_hard','static_pde'),('joint_hard','learned_hard'),
        ('no_carrier_hard','learned_hard'),('siren','learned_hard'),('fourier','learned_hard')]
    expected_pair_keys=set()
    for case in design['core_cases']+design['extension_cases']:
        comparisons=[(a,b,20000,20000) for a,b in controls] if case in design['core_cases'] else [
            ('learned_hard','analytic_hard',20000,20000),(selection['strongest_family'],'learned_hard',20000,20000)]
        terminal=('joint_hard','no_carrier_hard','siren','fourier') if case in design['core_cases'] else (selection['strongest_family'],)
        comparisons.extend((a,'learned_hard',26000,20000) for a in terminal)
        comparisons.extend((a,'analytic_hard',26000,20000) for a in terminal if a in ('siren','fourier'))
        expected_pair_keys.update((case,a,b,ts,bs,k) for a,b,ts,bs in comparisons for k in compared_metrics)
    actual_pair_keys={(r['case'],r['target'],r['baseline'],int(r['target_step']),int(r['baseline_step']),r['metric']) for r in pairs}
    if actual_pair_keys!=expected_pair_keys or len(pairs)!=len(expected_pair_keys):
        errors.append(dict(context='paired_statistics',field='exact_prescribed_contrast_inventory'))
    pair_count=0; matched_seed_rows=set()
    for row in pairs:
        case,target,baseline,ts,bs,metric=row['case'],row['target'],row['baseline'],int(row['target_step']),int(row['baseline_step']),row['metric']
        context=f'{case}/{target}@{ts}-{baseline}@{bs}/{metric}'
        check(context,'step',row['step'],ts)
        common=[s for s in seeds if all((case,g,s,k) in sources and metric in sources[(case,g,s,k)] and
            math.isfinite(sources[(case,g,s,k)][metric]) for g,k in ((target,ts),(baseline,bs)))]
        check(context,'n',row['n'],len(common))
        check(context,'status',row['status'],'ALL_FIVE_FINITE_PAIRS' if len(common)==5 else 'INCOMPLETE_OR_FAILURE')
        if len(common)!=5:
            for name in ('paired_difference_mean','population_sd','CI95_lower','CI95_upper','exact_sign_flip_two_sided_p','ratio_of_means','mean_paired_ratio'):
                if name in row:check(context,name,row[name],None)
            continue
        left=[sources[(case,target,s,ts)][metric] for s in seeds];right=[sources[(case,baseline,s,bs)][metric] for s in seeds]
        for field,value in pair_estimates(left,right,metric).items():check(context,field,row.get(field),value)
        children=[(index,r) for index,r in enumerate(seedpairs) if r['case']==case and r['target']==target and r['baseline']==baseline and
            int(r['target_step'])==ts and int(r['baseline_step'])==bs and r['metric']==metric]
        if len(children)!=5 or {int(r['seed']) for _,r in children}!=set(seeds):errors.append(dict(context=context,field='paired_seed_inventory'))
        for index,r in children:
            s=int(r['seed']);matched_seed_rows.add(index)
            a=sources[(case,target,s,ts)][metric];b=sources[(case,baseline,s,bs)][metric]
            for field,value in dict(target_value=a,baseline_value=b,difference=a-b).items():check(context,field,r[field],value)
        pair_count+=1
    if len(matched_seed_rows)!=len(seedpairs):errors.append(dict(context='paired_per_seed',field='unexpected_or_duplicate_seed_rows'))
    interactions=read_csv(out/'factor_interactions.csv');interactionseeds=read_csv(out/'factor_interactions_per_seed.csv')
    expected_interaction_keys={(case,k) for case in design['core_cases'] for k in list(thresholds)+['active_coherence_rel_l2']}
    if {(r['case'],r['metric']) for r in interactions}!=expected_interaction_keys or len(interactions)!=len(expected_interaction_keys):
        errors.append(dict(context='factor_interactions',field='exact_prescribed_inventory'))
    factors=('original_pde','static_pde','original_hard','learned_hard');interactions_complete=0
    for row in interactions:
        case,metric=row['case'],row['metric'];context=f'{case}/interaction/{metric}';deltas=[]
        for s in seeds:
            keys=[(case,g,s,20000) for g in factors]
            children=[r for r in interactionseeds if r['case']==case and r['metric']==metric and int(r['seed'])==s]
            if len(children)!=1:errors.append(dict(context=context,field='factor_seed_inventory',seed=s));continue
            child=children[0];valid=all(k in sources and metric in sources[k] and math.isfinite(sources[k][metric]) for k in keys)
            check(context+f'/{s}','status',child['status'],'AVAILABLE' if valid else 'INCOMPLETE_OR_FAILURE')
            if not valid:continue
            a,b,c,d=[sources[k][metric] for k in keys];delta=(d-b)-(c-a);deltas.append(delta)
            for field,value in dict(original_pde=a,static_pde=b,original_hard=c,learned_hard=d,
                hard_norm_effect_original=c-a,hard_norm_effect_static=d-b,interaction_difference=delta).items():check(context+f'/{s}',field,child[field],value)
        check(context,'n',row['n'],len(deltas));check(context,'status',row['status'],'ALL_FIVE_PRESENT' if len(deltas)==5 else 'INCOMPLETE_OR_FAILURE')
        if len(deltas)==5:
            center=mean(deltas);sd=population_sd(deltas);radius=T975_DF4*sd/2
            for field,value in dict(interaction_difference_mean=center,population_sd=sd,CI95_lower=center-radius,CI95_upper=center+radius).items():check(context,field,row[field],value)
            interactions_complete+=1
    attempt=json.loads((out/'attempt_completion_audit.json').read_text(encoding='utf-8'))
    expected_pairs=627; expected_interactions=18
    if len(pairs)!=expected_pairs or len(interactions)!=expected_interactions or len(interactionseeds)!=90:
        errors.append(dict(context='statistics',field='full_inventory',comparisons=len(pairs),interactions=len(interactions),interaction_seed_rows=len(interactionseeds)))
    all_verified=attempt['all_training_attempts_verified'] and not errors and not pending
    source_names=['all_snapshots_metrics.csv','summary_metrics.csv','paired_statistics.csv','paired_per_seed.csv',
        'factor_interactions.csv','factor_interactions_per_seed.csv','independent_metrics_per_snapshot.csv','attempt_completion_audit.json']
    report=dict(updated_utc=time.time(),status='FAIL' if errors else ('ALL_PRESCRIBED_STATISTICS_VERIFIED' if all_verified else 'PASS_AVAILABLE_STATISTICAL_TABLES'),
        group_endpoints_checked=len(summary),prescribed_pair_contrasts_checked=len(pairs),full_five_pair_estimates_checked=pair_count,
        factor_interactions_checked=len(interactions),complete_factor_estimates_checked=interactions_complete,
        source_snapshots_checked=len(sources),pending_independent_source_audits=sorted(set(pending)),errors=errors,
        new_live_snapshots_require_reaggregation_not_false_completion=True,
        source_table_sha256={name:sha(out/name) for name in source_names},audit_code_sha256=sha(__file__),
        separate_csv_math_implementation=True,archived_dynamics_inputs_independently_FFT_validated=True,
        student_t_df4_critical_constant=T975_DF4,no_survivor_five_seed_mean=True,
        exact_sign_flip_minimum_two_sided_p=2/32,all_prescribed_statistics_verified=bool(all_verified),whole_goal_complete=False)
    write(out/'independent_statistical_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_table_sha256','errors','pending_independent_source_audits')},ensure_ascii=False))
    if errors:
        print(json.dumps(errors[:12],ensure_ascii=False));raise RuntimeError('Independent statistical audit mismatch')


if __name__=='__main__':main()
