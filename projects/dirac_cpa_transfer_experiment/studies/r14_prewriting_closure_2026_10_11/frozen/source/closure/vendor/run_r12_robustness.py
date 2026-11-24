from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from .r12_config import ROOT,RAW,LOGS,DOCS,CASES,GROUPS,SEEDS,ARTIFACTS,configuration,run_directory,group_order,protocol
from .freeze_r7_phase_release import write_json,sha256

def prepare():
    from .r12_reference import prepare as reference
    from .freeze_r12_robustness import preserve_sources
    record=reference()
    if record['status']!='PASS':raise RuntimeError('Reference calibration failed')
    write_json(RAW/'protocol.json',protocol())
    print('Protected files:',len(preserve_sources()['files']),flush=True)
    import pandas as pd
    reviews={}
    for tag,id_,filename in (
        ('R8','r8_coupling_recovery','R8_COUPLING_RECOVERY_REPORT.md'),
        ('R9','r9_wavepacket_periodic_potential','R9_WAVEPACKET_PERIODIC_POTENTIAL_REPORT.md'),
        ('R10','r10_wavepacket_budget_diagnosis','R10_WAVEPACKET_BUDGET_DIAGNOSIS_REPORT.md'),
        ('R11','r11_independent_replication_ablation','R11_INDEPENDENT_REPLICATION_ABLATION_REPORT.md')):
        f=ROOT/filename;directory=ROOT/'results/raw'/id_
        reviews[tag]=dict(report_sha256=sha256(f),report_characters=len(f.read_text(encoding='utf-8')),
            raw_csv_tables={p.name:dict(rows=len(pd.read_csv(p)),sha256=sha256(p)) for p in directory.glob('*.csv')},
            original_decision=json.loads((directory/'decision.json').read_text(encoding='utf-8')))
    write_json(RAW/'history_review.json',reviews)
    costs=pd.read_csv(ROOT/'results/raw/r11_independent_replication_ablation/cost_and_residual_work.csv')
    n=configuration(CASES[0]).mode_count;means=costs.groupby('model').training_seconds.mean()
    groups={g:float(means[g]*(1.3 if g==GROUPS[-1] else 1)*n/129) for g in GROUPS}
    seconds=20*sum(groups.values())
    write_json(RAW/'cost_estimate.json',dict(estimated_training_seconds_by_group_per_run=groups,
        estimated_total_core_hours=seconds/3600,estimated_five_worker_wall_hours=seconds/5/3600,
        assumptions=['R11 measured concurrent one-core times','linear mode-count scaling','E 20k to26k linear estimate',
            'shared memory/cache/power; no isolated speed claim','evaluation and I/O extra'],
        estimated_optimizer_steps=80*26000,training_unit_spinor_residual_points=20*(32000+3*26000)*32*n,
        formal_run_count=80,estimated_full_snapshots=140,estimated_forward_snapshots=240,
        expected_cost_curve_points=380,estimate_accuracy_not_a_budget_decision=True))

def smoke():
    from .train_r12_robustness import train_run
    records=[]
    for case in CASES:
        for group in GROUPS:
            directory=run_directory(case,group,500,smoke=True)
            result=train_run(case,group,500,directory,smoke=True)
            records.append(dict(case=case,group=group,all_finite=result['all_finite'],
                hard_initial_error=result['initial_condition_max_abs_error'],
                all_artifacts=all((directory/p).is_file() for p in ARTIFACTS+('recovery_state.pt',))))
    write_json(RAW/'smoke_validation.json',dict(status='PASS' if all(r['all_finite'] and r['all_artifacts'] and
        r['hard_initial_error']<1e-14 for r in records) else 'FAIL',records=records,
        engineering_only=True,accuracy_not_used=True))

def launch():
    from .freeze_r12_robustness import verify_freeze
    from .r7_runtime import physical_core_masks,set_cpu_affinity
    from .run_r9_wavepacket import validate_masks
    frozen=verify_freeze();available=physical_core_masks();masks=available[:5]
    if len(available)<6:raise RuntimeError('Five distinct workers plus control core required')
    validate_masks(masks+[available[-1]]);set_cpu_affinity(available[-1]);LOGS.mkdir(parents=True,exist_ok=True)
    plan=dict(status='RUNNING',masks=masks,control_mask=available[-1],all_slots_pairwise_disjoint=True,
        freeze_manifest_sha256=frozen,threads_per_worker=1,batch_order=CASES,batches=[],
        estimated_cost=json.loads((RAW/'cost_estimate.json').read_text(encoding='utf-8')))
    write_json(RAW/'runtime_plan.json',plan)
    failures=[]
    for case in CASES:
        batch=dict(case=case,status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),slots=[])
        plan['batches'].append(batch);processes=[]
        for slot,(seed,mask) in enumerate(zip(SEEDS,masks)):
            log=LOGS/f'{case}_slot_{slot}_seed{seed}.log';stream=log.open('a',encoding='utf-8')
            command=[sys.executable,'-X','utf8','-u','-m','src.run_r12_robustness','worker',
                '--case',case,'--seed',str(seed),'--affinity',str(mask)]
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            batch['slots'].append(dict(slot=slot,seed=seed,mask=mask,pid=child.pid,group_order=group_order(case,seed),
                log=log.relative_to(ROOT).as_posix()))
            processes.append((child,stream,seed))
        write_json(RAW/'runtime_plan.json',plan)
        while any(p.poll() is None for p,_,_ in processes):time.sleep(.25)
        errors=[]
        for p,stream,seed in processes:
            stream.close()
            if p.returncode:errors.append(dict(case=case,seed=seed,exit_code=p.returncode))
        failures.extend(errors);batch.update(status='FAIL' if errors else 'COMPLETE',failures=errors,
            ended_utc=datetime.now(timezone.utc).isoformat())
        write_json(RAW/'runtime_plan.json',plan)
        print('BATCH COMPLETE',case,'failures=',errors,flush=True)
    plan.update(status='FAIL' if failures else 'COMPLETE',failures=failures,ended_utc=datetime.now(timezone.utc).isoformat())
    write_json(RAW/'runtime_plan.json',plan)
    if failures:raise RuntimeError('All prescribed attempts retained: '+str(failures))
    print('All 80 formal R12 runs complete',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','smoke','freeze','formal','worker'))
    parser.add_argument('--case',choices=CASES);parser.add_argument('--seed',type=int);parser.add_argument('--affinity',type=int)
    args=parser.parse_args()
    from .r7_runtime import set_cpu_affinity
    set_cpu_affinity(args.affinity if args.action=='worker' else 128)
    if args.action=='worker':
        from .r9_runtime import actual_affinity_mask
        os.environ['R7_CPU_AFFINITY_MASK']=str(args.affinity)
        if actual_affinity_mask()!=args.affinity:raise RuntimeError('Actual affinity differs from disjoint allocation')
        from .train_r12_robustness import train_run
        failures=[]
        for group in group_order(args.case,args.seed):
            try:train_run(args.case,group,args.seed,run_directory(args.case,group,args.seed))
            except Exception as error:failures.append(dict(case=args.case,group=group,error=repr(error)))
        write_json(LOGS/f'{args.case}_seed{args.seed}_worker_result.json',dict(status='FAIL' if failures else 'PASS',
            actual_affinity_mask=actual_affinity_mask(),seed=args.seed,case=args.case,failures=failures))
        if failures:raise RuntimeError(str(failures))
    elif args.action=='prepare':prepare()
    elif args.action=='smoke':smoke()
    elif args.action=='freeze':
        from .freeze_r12_robustness import freeze
        print(freeze(),flush=True)
    else:launch()

if __name__=='__main__':main()
