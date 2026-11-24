from __future__ import annotations
import itertools,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from plan import BASE,SEEDS,CORE,EXTENSION,THRESHOLDS,write,sha
from .expected_jobs import expected_jobs

def joint(metrics):return all(metrics[k]>=v if k=='fidelity_mean' else metrics[k]<=v for k,v in THRESHOLDS.items())

def collect():
    rows=[];costs=[];failures=[]
    for root in ('formal','development'):
        directory=BASE/root
        if not directory.exists():continue
        for p in sorted(directory.rglob('complete.json')):
            c=json.loads(p.read_text(encoding='utf-8'));j=c['job'];costs.append({k:v for k,v in c.items() if isinstance(v,(str,int,float,bool))}|j|{'path':str(p.relative_to(BASE))})
        for p in sorted(directory.rglob('failure.json')):
            f=json.loads(p.read_text(encoding='utf-8'));failures.append(f|{'path':str(p.relative_to(BASE))})
        for p in sorted(directory.rglob('metrics.json')):
            m=json.loads(p.read_text(encoding='utf-8'));j=m['job']
            if j['type']!='coupled':continue
            c=json.loads((p.parent/'cost.json').read_text(encoding='utf-8'));row={k:v for k,v in m.items() if isinstance(v,(str,int,float,bool)) or v is None}
            row['snapshot_evaluator_computation_seconds']=m.get('evaluation_seconds')
            row['training_cumulative_evaluator_process_wall_seconds']=c.get('evaluation_seconds')
            row.update(c);row.update({k:v for k,v in j.items() if isinstance(v,(str,int,float,bool))});row['path']=str(p.relative_to(BASE));row['metric_sha256']=sha(p)
            row['stage']=j['stage'];row['seed_joint_pass']=joint(m);rows.append(row)
    return pd.DataFrame(rows),pd.DataFrame(costs),failures

def paired_frame(a,b,case,target,baseline,step,metric):
    pair=a[['seed',metric]].merge(b[['seed',metric]],on='seed',suffixes=('_target','_baseline'))
    if len(pair)!=5 or set(pair['seed'])!=set(SEEDS):return None,[]
    av,bv=pair[metric+'_target'].to_numpy(),pair[metric+'_baseline'].to_numpy();delta=av-bv
    mean=float(delta.mean());sem=float(delta.std(ddof=1)/math.sqrt(5));radius=float(student_t.ppf(.975,4)*sem)
    distribution=np.array([np.mean(delta*np.array(signs)) for signs in itertools.product((-1.,1.),repeat=5)])
    pvalue=float(np.mean(abs(distribution)>=abs(mean)-1e-15))
    favorable=delta>0 if metric=='fidelity_mean' else delta<0
    summary=dict(case=case,target=target,baseline=baseline,step=step,metric=metric,n=5,paired_difference_mean=mean,
        population_sd=float(delta.std(ddof=0)),CI95_lower=mean-radius,CI95_upper=mean+radius,
        exact_sign_flip_two_sided_p=pvalue,ratio_of_means=float(av.mean()/bv.mean()) if bv.mean()!=0 else None,
        mean_paired_ratio=float(np.mean(av/bv)) if np.all(bv!=0) else None,favorable_pairs=int(favorable.sum()),
        descriptive_small_n=True)
    seeds=[dict(case=case,target=target,baseline=baseline,step=step,metric=metric,seed=int(s),target_value=float(x),baseline_value=float(y),difference=float(x-y)) for s,x,y in zip(pair['seed'],av,bv)]
    return summary,seeds

def summaries_with_missing(formal,jobs,failures,audits):
    expected={}
    for j in jobs:
        if j['type']=='coupled' and j['stage']!='development':
            for step in j['snapshots']:expected.setdefault((j['case'],j['group'],step),[]).append(j)
    rows=[]
    for (case,group,step),planned in expected.items():
        table=formal[(formal.case==case)&(formal.group==group)&(formal.step==step)]
        affected=[f for f in failures if f['job'].get('case')==case and f['job'].get('group')==group and
            (f.get('status')=='DEPENDENCY_FAILED' or f.get('occurred_step',0)<=step)]
        present=set(table.seed);failed_seeds={f['job']['seed'] for f in affected};missing=set(SEEDS)-present-failed_seeds
        complete=len(table)==5 and present==set(SEEDS) and not affected
        finite=all(np.isfinite(table[k].to_numpy(dtype=float)).all() for k in THRESHOLDS)
        mean_checks={};row=dict(case=case,group=group,step=int(step),n=len(table),n_planned=5,
            all_five_present=complete,missing_seeds=','.join(str(s) for s in sorted(missing)),failed_seeds=','.join(str(s) for s in sorted(failed_seeds)),
            endpoint='primary' if step==20000 else ('terminal' if step==planned[0]['budget'] else 'fixed_snapshot'))
        for metric,threshold in THRESHOLDS.items():
            # A surviving-seed mean cannot stand in for the prespecified five-seed mean.
            row[metric+'_available_finite_mean']=float(table[metric].mean()) if len(table) and finite else None
            row[metric+'_mean']=float(table[metric].mean()) if complete and finite else None
            row[metric+'_population_sd']=float(table[metric].std(ddof=0)) if complete and finite else None
            mean_checks[metric]=(row[metric+'_mean']>=threshold if metric=='fidelity_mean' else row[metric+'_mean']<=threshold) if row[metric+'_mean'] is not None else False
        passed=bool(complete and finite and all(mean_checks.values()) and int(table.seed_joint_pass.sum())>=4)
        audited=complete and all(r['path'] in audits and audits[r['path']]==r['metric_sha256'] for _,r in table.iterrows())
        status='FAIL' if affected else ('INCOMPLETE' if not complete else ('INVALID_METRICS' if not finite else
            ('PASS' if passed else 'FAIL') if audited else ('PASS_PENDING_AUDIT' if passed else 'FAIL_PENDING_AUDIT')))
        accounted=len(present|failed_seeds)
        row.update(joint_pass_count=int(table.seed_joint_pass.sum()),joint_success_rate=float(table.seed_joint_pass.sum()/5) if accounted==5 else None,
            success_rate_denominator=5 if accounted==5 else None,attempts_accounted_at_endpoint=accounted,
            absolute_success=passed and audited,
            computed_absolute_success_before_independent_audit=passed,independent_metric_audit_complete=bool(audited),
            numerical_or_dependency_failures=len(affected),status=status)
        rows.append(row)
    return pd.DataFrame(rows)

def comparisons_with_missing(formal,selection):
    pairs=[];seedpairs=[];metrics=list(THRESHOLDS)+['active_coherence_rel_l2','attributed_training_wall_seconds',
        'attributed_training_cpu_seconds','inference_seconds_median','peak_rss_bytes','attributed_unit_spinor_residual_points']
    controls=[('learned_hard','analytic_hard'),('static_pde','original_pde'),('learned_hard','original_hard'),
        ('original_hard','original_pde'),('learned_hard','static_pde'),('joint_hard','learned_hard'),
        ('no_carrier_hard','learned_hard'),('siren','learned_hard'),('fourier','learned_hard')]
    for case in CORE+EXTENSION:
        comparisons=[(a,b,20000,20000) for a,b in controls] if case in CORE else [
            ('learned_hard','analytic_hard',20000,20000),(selection['strongest_family'],'learned_hard',20000,20000)]
        terminal=('joint_hard','no_carrier_hard','siren','fourier') if case in CORE else (selection['strongest_family'],)
        comparisons.extend((a,'learned_hard',26000,20000) for a in terminal)
        comparisons.extend((a,'analytic_hard',26000,20000) for a in terminal if a in ('siren','fourier'))
        for target,baseline,ts,bs in comparisons:
            a=formal[(formal.case==case)&(formal.group==target)&(formal.step==ts)]
            b=formal[(formal.case==case)&(formal.group==baseline)&(formal.step==bs)]
            for metric in metrics:
                if metric not in formal:continue
                aa=a[np.isfinite(a[metric])];bb=b[np.isfinite(b[metric])]
                s,p=paired_frame(aa,bb,case,target,baseline,ts,metric)
                if s:
                    s.update(target_step=ts,baseline_step=bs,status='ALL_FIVE_FINITE_PAIRS',
                        endpoint_comparison='equal_coupled_budget' if ts==bs else 'prescribed_terminal_vs_primary')
                    for r in p:r.update(target_step=ts,baseline_step=bs)
                    pairs.append(s);seedpairs.extend(p)
                else:
                    pairs.append(dict(case=case,target=target,baseline=baseline,step=ts,target_step=ts,baseline_step=bs,metric=metric,
                        n=len(set(aa.seed)&set(bb.seed)),n_planned=5,status='INCOMPLETE_OR_FAILURE',descriptive_small_n=True))
    return pd.DataFrame(pairs),pd.DataFrame(seedpairs)

def factor_interactions(formal):
    rows=[];seeds=[]
    for case in CORE:
        table=formal[(formal.case==case)&(formal.step==20000)]
        for metric in list(THRESHOLDS)+['active_coherence_rel_l2']:
            if metric not in table:continue
            values=[]
            for seed in SEEDS:
                v={g:table[(table.seed==seed)&(table.group==g)][metric].tolist() for g in ('original_pde','static_pde','original_hard','learned_hard')}
                valid=all(len(x)==1 and np.isfinite(x[0]) for x in v.values())
                row=dict(case=case,metric=metric,seed=seed,step=20000,status='AVAILABLE' if valid else 'INCOMPLETE_OR_FAILURE')
                if valid:
                    v={g:x[0] for g,x in v.items()};interaction=(v['learned_hard']-v['static_pde'])-(v['original_hard']-v['original_pde'])
                    row.update(v,hard_norm_effect_original=v['original_hard']-v['original_pde'],
                        hard_norm_effect_static=v['learned_hard']-v['static_pde'],interaction_difference=interaction);values.append(interaction)
                seeds.append(row)
            r=dict(case=case,metric=metric,step=20000,n=len(values),n_planned=5,
                interaction_definition='(static_hard-static_PDE)-(original_hard-original_PDE)',status='ALL_FIVE_PRESENT' if len(values)==5 else 'INCOMPLETE_OR_FAILURE')
            if len(values)==5:
                v=np.asarray(values);mean=float(v.mean());radius=float(student_t.ppf(.975,4)*v.std(ddof=1)/np.sqrt(5))
                r.update(interaction_difference_mean=mean,population_sd=float(v.std(ddof=0)),CI95_lower=mean-radius,CI95_upper=mean+radius,descriptive_small_n=True)
            rows.append(r)
    return pd.DataFrame(rows),pd.DataFrame(seeds)

def main():
    out=BASE/'analysis_outputs';out.mkdir(exist_ok=True);d,cost,failures=collect();jobs=expected_jobs()
    d.to_csv(out/'all_snapshots_metrics.csv',index=False);cost.to_csv(out/'unique_cost_ledger.csv',index=False);write(out/'preserved_failures.json',failures)
    if d.empty:
        columns=['stage','case','group','step','seed','path','metric_sha256','seed_joint_pass',*THRESHOLDS]
        d=pd.DataFrame(columns=columns)
    formal=d[d.stage!='development'];primary=formal[formal.step==20000];primary.to_csv(out/'per_seed_metrics.csv',index=False)
    auditpath=out/'independent_metrics_per_snapshot.csv';audits={}
    if auditpath.exists():
        audit=pd.read_csv(auditpath)
        if 'metric_record_sha256' in audit:
            audits={r['path']:r['metric_record_sha256'] for _,r in audit.iterrows() if bool(r['pass_check'])}
    summaries=summaries_with_missing(formal,jobs,failures,audits);summaries.to_csv(out/'summary_metrics.csv',index=False)
    selection=json.loads((BASE/'frozen/baseline_selection.json').read_text(encoding='utf-8'))
    pairs,seedpairs=comparisons_with_missing(formal,selection)
    pairs.to_csv(out/'paired_statistics.csv',index=False);seedpairs.to_csv(out/'paired_per_seed.csv',index=False)
    interactions,seedinteractions=factor_interactions(formal)
    interactions.to_csv(out/'factor_interactions.csv',index=False);seedinteractions.to_csv(out/'factor_interactions_per_seed.csv',index=False)
    complete_count=len(list((BASE/'formal').rglob('complete.json')))
    write(out/'analysis_status.json',dict(updated_utc=time.time(),formal_complete=complete_count,formal_planned=210,entire_study_complete=False,
        results_are_available_snapshots_only=True,missing_not_success=True,all_failed_groups_remain_in_expected_table=True,
        no_survivor_mean_replaces_five_seed_mean=True,scientific_failure_does_not_trigger_extra_budget=True))
    print('AGGREGATED',len(d),'snapshots',len(summaries),'expected group endpoints',len(pairs),'prescribed metric comparisons')

if __name__=='__main__':main()
