from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from .r13_config import ROOT,RAW,LOGS,EXPLORATION_SEEDS,VALIDATION_SEEDS,ARTIFACTS,configuration,protocol,carrier_directory,run_directory,CARRIER_KINDS,ENVELOPE_KINDS
from .r13_plan import STAGES,jobs_for,directory_for
from .freeze_r7_phase_release import write_json,sha256

def utc():return datetime.now(timezone.utc).isoformat()

def alive(pid):
    if not pid:return False
    if os.name=='nt':
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_bool,ctypes.c_ulong]
        kernel.OpenProcess.restype=ctypes.c_void_p
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:return False
        code=ctypes.c_ulong()
        kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        ok=kernel.GetExitCodeProcess(handle,ctypes.byref(code));kernel.CloseHandle(handle)
        return bool(ok and code.value==259)
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False

def prepare():
    RAW.mkdir(parents=True,exist_ok=True)
    from .freeze_r13_recovery import preserve_sources
    protected=preserve_sources();print('Protected historical files:',len(protected['files']),flush=True)
    import pandas as pd
    reviews={}
    for tag,exp,report in (
        ('R8','r8_coupling_recovery','R8_COUPLING_RECOVERY_REPORT.md'),
        ('R9','r9_wavepacket_periodic_potential','R9_WAVEPACKET_PERIODIC_POTENTIAL_REPORT.md'),
        ('R10','r10_wavepacket_budget_diagnosis','R10_WAVEPACKET_BUDGET_DIAGNOSIS_REPORT.md'),
        ('R11','r11_independent_replication_ablation','R11_INDEPENDENT_REPLICATION_ABLATION_REPORT.md'),
        ('R12','r12_cross_condition_robustness','R12_ROBUSTNESS_REPORT.md')):
        directory=ROOT/'results/raw'/exp
        reviews[tag]=dict(report_sha256=sha256(ROOT/report),report_characters=len((ROOT/report).read_text(encoding='utf-8')),
            raw_csv_tables={p.name:dict(rows=len(pd.read_csv(p)),sha256=sha256(p)) for p in directory.glob('*.csv')},
            original_decision=json.loads((directory/'decision.json').read_text(encoding='utf-8')))
    write_json(RAW/'history_review.json',reviews)
    write_json(RAW/'protocol.json',protocol())
    costs=pd.read_csv(ROOT/'results/raw/r12_cross_condition_robustness/per_seed_metrics.csv')
    costs=costs[costs.endpoint=='coupled_20000']
    rates={}
    for case,old in (('bridge','V04'),('T10','T10'),('m10','m10')):
        a=costs[(costs.case==old)&(costs.model=='A_pretrained_frozen')]
        e=costs[(costs.case==old)&(costs.model=='E_exact_carrier_diagnostic')]
        rates[case]=dict(pre=float(a.carrier_training_seconds.mean()),
            learned_coupled=float(a.coupled_training_seconds.mean()),exact_coupled=float(e.coupled_training_seconds.mean()))
    overhead=1.25
    stages={
        'bridge':3*(rates['bridge']['pre']+rates['bridge']['learned_coupled']+rates['bridge']['exact_coupled']),
        'free_carriers':9*sum(rates[c]['pre'] for c in ('T10','m10')),
        'exact_envelopes':3*sum(rates[c]['exact_coupled']*(1+2*overhead) for c in ('T10','m10')),
        'T10_factorial':3*rates['T10']['learned_coupled']*(1+3*overhead),
        'independent_validation_max':5*sum(2*rates[c]['pre']+rates[c]['learned_coupled']*(1+overhead)+rates[c]['exact_coupled'] for c in ('T10','m10'))}
    seconds=sum(stages.values());estimated_wall=sum(v/(5 if k.startswith('independent') else 3) for k,v in stages.items())
    write_json(RAW/'cost_estimate.json',dict(r12_mean_seconds_20000_coupled=rates,repair_coupled_overhead_assumption=overhead,
        estimated_stage_accumulated_core_hours={k:v/3600 for k,v in stages.items()},
        estimated_total_accumulated_bound_core_hours=seconds/3600,estimated_batch_wall_hours=estimated_wall/3600,
        maximum_unique_carrier_jobs=41,maximum_unique_coupled_jobs=66,maximum_unique_optimizer_steps=1566000,
        maximum_unique_unit_spinor_residual_points=(2*41*6000+66*20000)*32*161,
        storage_reserve_gib=25,estimated_peak_memory_all_workers_gib=5,
        assumptions=['R12 measured concurrent bound-core training times; bridge uses m5T5 V04 workload proxy',
            'no isolated speed claim','repair overhead estimate only, not measured result','evaluation/I/O and interruptions extra',
            'reuse charged once to incurred experiment cost, fully charged to standalone method cost'],
        budget_not_selected_by_accuracy=True))
    from .r13_reference import prepare as reference
    record=reference()
    if record['status']!='PASS':raise RuntimeError('Reference calibration failed; formal training forbidden')

def smoke():
    from .train_r13_recovery import train_job
    records=[]
    for case in ('bridge','T10','m10'):
        sources={}
        for kind in CARRIER_KINDS:
            job=dict(type='carrier',stage='smoke',case=case,carrier_kind=kind,seed=600)
            target=carrier_directory('smoke',case,kind,600,smoke=True)
            result=train_job(job,target,smoke=True);sources[kind]=target
            records.append(dict(case=case,type='carrier',kind=kind,finite=result['all_finite'],
                initial_error=result['initial_condition_max_abs_error'],artifacts=all((target/n).exists() for n in ARTIFACTS+('recovery_state.pt',))))
        for kind in CARRIER_KINDS+('exact',):
            for envelope in ENVELOPE_KINDS:
                job=dict(type='coupled',stage='smoke',case=case,carrier_kind=kind,envelope_kind=envelope,seed=600,group=f'{kind}__{envelope}',
                    **({'pretrain_source':str(sources[kind])} if kind!='exact' else {}))
                target=run_directory('smoke',case,job['group'],600,smoke=True)
                result=train_job(job,target,smoke=True)
                records.append(dict(case=case,type='coupled',kind=kind,envelope=envelope,finite=result['all_finite'],
                    initial_error=result['initial_condition_max_abs_error'],artifacts=all((target/n).exists() for n in ARTIFACTS+('recovery_state.pt',))))
    write_json(RAW/'smoke_validation.json',dict(status='PASS' if all(r['finite'] and r['artifacts'] and r['initial_error']<1e-14 for r in records) else 'FAIL',
        engineering_only=True,accuracy_not_used=True,records=records))

def status():
    path=RAW/'runtime_plan.json'
    plan=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    completed=list((RAW/'formal').rglob('execution_cost.json'))
    failures=list((RAW/'formal').rglob('failure.json'))
    live=[s for b in plan.get('batches',[]) for s in b.get('slots',[]) if alive(s.get('pid'))]
    driver_alive=alive(plan.get('driver_pid'))
    state=plan.get('status','NOT_STARTED')
    if state=='RUNNING' and not live and not driver_alive:state='INTERRUPTED_NO_LIVE_PROCESS'
    progress=[]
    for slot in live:
        path=ROOT/slot['log']
        lines=path.read_text(encoding='utf-8',errors='replace').splitlines() if path.exists() else []
        progress.append(dict(seed=slot['seed'],stage=slot['stage'],pid=slot['pid'],mask=slot['mask'],recent=lines[-3:]))
    result=dict(status=state,driver_alive=driver_alive,complete_unique_jobs=len(completed),
        completed_carrier_jobs=sum('/carriers/' in p.as_posix() for p in completed),
        completed_coupled_jobs=sum('/coupled/' in p.as_posix() for p in completed),
        numerical_failures=len(failures),live_workers=progress)
    print(json.dumps(result,ensure_ascii=False,indent=2));return result

def launch():
    from .freeze_r13_recovery import verify_freeze
    from .r7_runtime import physical_core_masks,set_cpu_affinity
    from .run_r9_wavepacket import validate_masks
    from .r13_selection import select_carrier,select_envelope,select_combination
    frozen=verify_freeze();cores=physical_core_masks()
    import shutil
    if shutil.disk_usage(ROOT).free < 25*2**30:raise RuntimeError('Less than the preregistered 25 GiB storage reserve available')
    if len(cores)<6:raise RuntimeError('Five physical workers plus control core required')
    masks=cores[:5];control=cores[-1];validate_masks(masks+[control]);set_cpu_affinity(control)
    LOGS.mkdir(parents=True,exist_ok=True)
    old_path=RAW/'runtime_plan.json'
    if old_path.exists():
        old=json.loads(old_path.read_text(encoding='utf-8'))
        if alive(old.get('driver_pid')) or any(alive(s.get('pid')) for b in old.get('batches',[]) for s in b.get('slots',[])):
            raise RuntimeError('Existing R13 process alive; no overlapping launch')
        archive=RAW/'runtime_history';archive.mkdir(exist_ok=True)
        import shutil
        shutil.copyfile(old_path,archive/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'))
    plan=dict(status='RUNNING',driver_pid=os.getpid(),started_utc=utc(),masks=masks,control_mask=control,
        available_distinct_physical_core_masks=cores,all_slots_pairwise_disjoint=True,threads_per_worker=1,
        freeze_manifest_sha256=frozen,batches=[],exact_recovery=True,no_selective_new_initialization=True)
    write_json(old_path,plan)
    for stage in STAGES:
        if stage=='T10_factorial':
            select_carrier();select_envelope()
        if stage=='independent_validation':select_combination()
        seeds=VALIDATION_SEEDS if stage=='independent_validation' else EXPLORATION_SEEDS
        batch=dict(stage=stage,status='RUNNING',started_utc=utc(),slots=[]);plan['batches'].append(batch)
        processes=[]
        for slot,(seed,mask) in enumerate(zip(seeds,masks)):
            log=LOGS/f'{stage}_slot_{slot}_seed{seed}.log';stream=log.open('a',encoding='utf-8')
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            command=[sys.executable,'-X','utf8','-u','-m','src.run_r13_recovery','worker','--stage',stage,'--seed',str(seed),'--affinity',str(mask)]
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            batch['slots'].append(dict(stage=stage,slot=slot,seed=seed,mask=mask,pid=child.pid,
                log=log.relative_to(ROOT).as_posix(),jobs=jobs_for(stage,seed)))
            processes.append((child,stream,seed))
        write_json(old_path,plan)
        while any(p.poll() is None for p,_,_ in processes):time.sleep(.25)
        errors=[]
        for p,stream,seed in processes:
            stream.close()
            if p.returncode:errors.append(dict(stage=stage,seed=seed,exit_code=p.returncode))
        batch.update(status='FAIL' if errors else 'COMPLETE',errors=errors,ended_utc=utc());write_json(old_path,plan)
        print('BATCH COMPLETE',stage,'worker_errors',errors,flush=True)
        if errors:
            plan.update(status='INTERRUPTED_OR_IMPLEMENTATION_FAILURE',ended_utc=utc());write_json(old_path,plan)
            raise RuntimeError('Worker failed; preserve output and diagnose without selective retraining')
    plan.update(status='TRAINING_COMPLETE',ended_utc=utc());write_json(old_path,plan)
    from .finalize_r13_recovery import finalize
    try:
        result=finalize()
    except Exception as error:
        plan.update(status='POSTPROCESSING_NEEDS_ATTENTION',error=repr(error),ended_utc=utc());write_json(old_path,plan)
        raise
    plan.update(status='COMPLETE' if result=='PASS' else 'TRAINING_COMPLETE_VISUAL_REVIEW_PENDING',ended_utc=utc());write_json(old_path,plan)

def worker(stage,seed,mask):
    from .r9_runtime import actual_affinity_mask
    from .train_r13_recovery import train_job
    if actual_affinity_mask()!=mask:raise RuntimeError('Actual CPU affinity differs from assigned disjoint core')
    jobs=jobs_for(stage,seed);failures=[];engineering=[]
    for job in jobs:
        target=directory_for(job)
        if (target/'failure.json').exists():
            failures.append(dict(job=job,status='RETAINED_NUMERICAL_FAILURE'));continue
        if job['type']=='coupled' and job['carrier_kind']!='exact' and not (ROOT/job['pretrain_source']/'metrics.json').exists():
            failures.append(dict(job=job,status='DEPENDENCY_FAILED'));write_json(target/'dependency_failure.json',failures[-1]);continue
        try:train_job(job,target)
        except FloatingPointError as error:
            failures.append(dict(job=job,status='NUMERICAL_FAILURE',error=repr(error)));traceback.print_exc()
        except Exception as error:
            engineering.append(dict(job=job,status='ENGINEERING_OR_INTERRUPTION',error=repr(error)));traceback.print_exc();break
    write_json(LOGS/f'{stage}_seed{seed}_worker_result.json',dict(status='ENGINEERING_FAIL' if engineering else ('RETAINED_FAILURES' if failures else 'PASS'),
        actual_affinity_mask=actual_affinity_mask(),stage=stage,seed=seed,failures=failures,engineering_errors=engineering))
    if engineering:raise RuntimeError(str(engineering))

def main():
    for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[variable]='1'
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','smoke','freeze','formal','worker','status','finalize'))
    parser.add_argument('--stage',choices=STAGES);parser.add_argument('--seed',type=int);parser.add_argument('--affinity',type=int)
    args=parser.parse_args()
    if args.action=='status':status();return
    from .r7_runtime import physical_core_masks,set_cpu_affinity
    set_cpu_affinity(args.affinity if args.action=='worker' else physical_core_masks()[-1])
    if args.action=='prepare':prepare()
    elif args.action=='smoke':smoke()
    elif args.action=='freeze':
        from .freeze_r13_recovery import freeze
        print(freeze(),flush=True)
    elif args.action=='worker':worker(args.stage,args.seed,args.affinity)
    elif args.action=='finalize':
        from .finalize_r13_recovery import finalize
        finalize()
    else:launch()

if __name__=='__main__':main()
