"""Unique experiment cost vs standalone attribution and explicit search/amortization examples."""
from __future__ import annotations
import json,time
import numpy as np
import pandas as pd
from plan import BASE,write
from .expected_jobs import expected_jobs
from .aggregate import collect

def lifecycle(job,complete_path):
    log=BASE/'logs'/(job['directory'].replace('/','__')+'.log')
    if not log.exists():return None
    stat=log.stat();birth=getattr(stat,'st_birthtime',None)
    return complete_path.stat().st_mtime-birth if birth is not None else None

def main():
    out=BASE/'analysis_outputs';out.mkdir(exist_ok=True);snapshots,_,failures=collect();rows=[];replay=[]
    for job in expected_jobs():
        directory=BASE/job['directory'];path=directory/'complete.json';fpath=directory/'failure.json'
        if path.exists():
            c=json.loads(path.read_text(encoding='utf-8'));elapsed=lifecycle(job,path)
            r=dict(directory=job['directory'],type=job['type'],stage=job['stage'],case=job['case'],group=job.get('group',job.get('kind')),seed=job['seed'],
                state='COMPLETE',**{k:c[k] for k in ('training_wall_seconds','training_cpu_seconds','evaluation_seconds','checkpoint_io_seconds','setup_seconds',
                'incurred_pipeline_seconds','unit_spinor_residual_points','peak_rss_bytes')},launch_to_completion_wall_span_seconds=elapsed,
                unclassified_launch_overhead_or_downtime_seconds=elapsed-c['incurred_pipeline_seconds'] if elapsed is not None else None,
                startup_span_is_metadata_derived=True)
            rows.append(r)
        elif fpath.exists():
            f=json.loads(fpath.read_text(encoding='utf-8'));rows.append(dict(directory=job['directory'],type=job['type'],stage=job['stage'],case=job['case'],
                group=job.get('group',job.get('kind')),seed=job['seed'],state=f['status'],training_wall_seconds=f.get('observed_training_wall_seconds'),
                training_cpu_seconds=f.get('observed_training_cpu_seconds'),incurred_pipeline_seconds=None,unit_spinor_residual_points=None,
                failure_cost_fields_not_recorded_remain_missing=True))
        for p in sorted((directory/'interruptions').glob('*.json')):
            f=json.loads(p.read_text(encoding='utf-8'));history=directory/'loss_history.csv';lost_wall=lost_cpu=None
            if history.exists():
                h=pd.read_csv(history);h=h[h.step==f.get('last_saved_step',-1)]
                if len(h)==1:
                    lost_wall=max(0.,f['observed_training_wall_seconds']-float(h.training_wall_seconds.iloc[0]))
                    lost_cpu=max(0.,f['observed_training_cpu_seconds']-float(h.training_cpu_seconds.iloc[0]))
            replay.append(dict(directory=job['directory'],interruption=p.relative_to(BASE).as_posix(),last_saved_step=f.get('last_saved_step'),
                occurred_step=f.get('occurred_step'),uncheckpointed_training_wall_seconds=lost_wall,uncheckpointed_training_cpu_seconds=lost_cpu,
                replay_cost_unknown=lost_wall is None))
    unique=pd.DataFrame(rows);unique.to_csv(out/'unique_incurred_costs_completed_and_failures.csv',index=False)
    pd.DataFrame(replay).to_csv(out/'engineering_replay_costs.csv',index=False)
    formal=snapshots[snapshots.stage!='development'].copy() if not snapshots.empty else pd.DataFrame()
    dev=unique[unique.stage=='development'];algorithm=[];amortized=[]
    if not formal.empty:
        for _,r in formal.iterrows():
            case,family=r['case'],r['group'];isbaseline=family in ('siren','fourier')
            candidate_search=dev[(dev.group==family)&(dev.case==case)] if case in ('T10','m10') else dev[dev.group==family]
            selected_search=float(candidate_search.training_wall_seconds.sum(min_count=1)) if isbaseline else 0.
            selected_search_cpu=float(candidate_search.training_cpu_seconds.sum(min_count=1)) if isbaseline else 0.
            selected_search_pipeline=float(candidate_search.incurred_pipeline_seconds.sum(min_count=1)) if isbaseline else 0.
            selected_search_work=float(candidate_search.unit_spinor_residual_points.sum(min_count=1)) if isbaseline else 0.
            global_family_search=float(dev[dev.group==family].training_wall_seconds.sum(min_count=1)) if isbaseline else 0.
            all_search=float(dev.training_wall_seconds.sum(min_count=1))
            all_search_pipeline=float(dev.incurred_pipeline_seconds.sum(min_count=1))
            all_search_cpu=float(dev.training_cpu_seconds.sum(min_count=1))
            all_search_work=float(dev.unit_spinor_residual_points.sum(min_count=1))
            a=dict(case=case,group=family,seed=int(r['seed']),step=int(r['step']),stage=r['stage'],
                standalone_optimization_wall_seconds=float(r['attributed_training_wall_seconds']),standalone_recorded_pipeline_seconds=float(r['attributed_pipeline_seconds']),
                standalone_optimization_cpu_seconds=float(r['attributed_training_cpu_seconds']),
                standalone_unit_spinor_residual_points=int(r['attributed_unit_spinor_residual_points']),
                auxiliary_optimization_wall_seconds=float(r['source_pretraining_wall_seconds']),auxiliary_optimization_cpu_seconds=float(r['source_pretraining_cpu_seconds']),
                relevant_family_candidate_search_wall_seconds=selected_search,global_family_search_wall_seconds=global_family_search,
                relevant_family_candidate_search_cpu_seconds=selected_search_cpu,
                relevant_family_candidate_search_pipeline_seconds=selected_search_pipeline,
                relevant_family_candidate_search_unit_spinor_residual_points=selected_search_work,
                all_baseline_development_wall_seconds=all_search,
                all_baseline_development_pipeline_seconds=all_search_pipeline,
                standalone_optimization_plus_relevant_search_seconds=float(r['attributed_training_wall_seconds'])+selected_search,
                standalone_optimization_cpu_plus_relevant_search_seconds=float(r['attributed_training_cpu_seconds'])+selected_search_cpu,
                standalone_recorded_pipeline_plus_relevant_search_seconds=float(r['attributed_pipeline_seconds'])+selected_search_pipeline,
                standalone_unit_spinor_residual_points_plus_relevant_search=int(r['attributed_unit_spinor_residual_points'])+selected_search_work,
                five_seed_ensemble_mean_optimization_plus_once_search_seconds=float(r['attributed_training_wall_seconds'])+selected_search/5,
                five_seed_ensemble_mean_pipeline_plus_once_search_seconds=float(r['attributed_pipeline_seconds'])+selected_search_pipeline/5,
                standalone_optimization_plus_all_baseline_search_seconds=float(r['attributed_training_wall_seconds'])+all_search if isbaseline else float(r['attributed_training_wall_seconds']),
                standalone_recorded_pipeline_plus_all_baseline_search_seconds=float(r['attributed_pipeline_seconds'])+(all_search_pipeline if isbaseline else 0.),
                standalone_optimization_cpu_plus_all_baseline_search_seconds=float(r['attributed_training_cpu_seconds'])+(all_search_cpu if isbaseline else 0.),
                standalone_unit_spinor_residual_points_plus_all_baseline_search=int(r['attributed_unit_spinor_residual_points'])+(all_search_work if isbaseline else 0.),
                search_allocation_scope='six case/family development attempts' if case in ('T10','m10') else 'all12 family development attempts',
                source_reused=bool(r['carrier_pretraining_reused']),incurred_ledger_does_not_sum_standalone_attribution=True)
            group_directory=BASE/str(r['directory']);config=json.loads((group_directory/'config.json').read_text(encoding='utf-8'))
            sourcejob=config['job'].get('source');source_peak=None
            if sourcejob:source_peak=json.loads((BASE/sourcejob/'complete.json').read_text(encoding='utf-8'))['peak_rss_bytes']
            a['standalone_phasewise_training_peak_rss_bytes']=max(float(r['peak_rss_bytes']),source_peak or 0)
            inf=BASE/'isolated_inference'/case/family/f"seed_{int(r['seed'])}"/f"step_{int(r['step'])}.json"
            if inf.exists():
                m=json.loads(inf.read_text(encoding='utf-8'));a.update(isolated_inference_seconds_median=m['inference_seconds_median'],
                    isolated_inference_cpu_seconds_median=m['inference_cpu_seconds_median'],isolated_inference_peak_rss_bytes=m['process_peak_rss_bytes'])
            algorithm.append(a)
            if sourcejob:
                for uses in (1,5,10,100):amortized.append(dict(case=case,group=family,seed=int(r['seed']),step=int(r['step']),hypothetical_source_uses=uses,
                    attributed_optimization_seconds_per_solve=float(r['training_wall_seconds'])+float(r['source_pretraining_wall_seconds'])/uses,
                    auxiliary_seconds_per_solve=float(r['source_pretraining_wall_seconds'])/uses,
                    hypothetical_not_observed_efficiency_gain=True,actual_same_seed_source_reuse=bool(r['carrier_pretraining_reused'])))
    pd.DataFrame(algorithm).to_csv(out/'algorithm_costs_with_search_overhead.csv',index=False)
    pd.DataFrame(amortized).to_csv(out/'carrier_amortization_examples.csv',index=False)
    summary=dict(updated_utc=time.time(),status='AVAILABLE_ATTEMPTS_ONLY',unique_complete=int(sum(unique.state=='COMPLETE')),
        unique_preserved_failure=int(sum(unique.state!='COMPLETE')),unique_planned=234,whole_study_complete=False,
        unique_optimization_wall_seconds_recorded=float(unique.training_wall_seconds.sum(min_count=1)),
        unique_optimization_cpu_seconds_recorded=float(unique.training_cpu_seconds.sum(min_count=1)),
        recorded_unique_pipeline_seconds=float(unique.incurred_pipeline_seconds.sum(min_count=1)),
        retained_replay_records=len(replay),uncheckpointed_replay_wall_seconds_known=sum(r['uncheckpointed_training_wall_seconds'] for r in replay if r['uncheckpointed_training_wall_seconds'] is not None),
        replay_missing_fields_not_assumed_zero=any(r['replay_cost_unknown'] for r in replay),
        dev_all24_optimization_wall_seconds=float(dev.training_wall_seconds.sum(min_count=1)),
        unknown_failed_costs_not_filled_with_zero=True,optimization_CPU_excludes_imports_IO_and_evaluation_child_CPU=True,
        launch_span_from_log_creation_and_complete_mtime_is_observed_lifecycle_not_CPU_consumption=True,
        concurrent_training_costs_are_not_isolated_hardware_speedup=True)
    write(out/'cost_summary.json',summary);print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
