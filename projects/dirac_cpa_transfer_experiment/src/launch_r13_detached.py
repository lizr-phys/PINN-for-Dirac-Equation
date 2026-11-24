from __future__ import annotations

"""Operational single-run launcher. Does not change the frozen scientific plan."""
import ctypes
import json
import os
import shutil
import sys
from datetime import datetime,timezone

for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[variable]='1'

from .r13_config import ROOT,RAW,LOGS
from .freeze_r7_phase_release import write_json,sha256
from .run_r13_recovery import alive
from .r7_runtime import physical_core_masks,set_cpu_affinity

def current_in_job():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.IsProcessInJob.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
    flag=ctypes.c_int()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(),None,ctypes.byref(flag)):
        raise ctypes.WinError(ctypes.get_last_error())
    return bool(flag.value)

def main():
    os.chdir(ROOT)
    set_cpu_affinity(physical_core_masks()[-1])
    LOGS.mkdir(parents=True,exist_ok=True)
    sys.stdout=(LOGS/'controller_stdout.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=(LOGS/'controller_stderr.log').open('a',encoding='utf-8',buffering=1)
    from .freeze_r13_recovery import verify_freeze
    frozen=verify_freeze()
    old=json.loads((RAW/'runtime_plan.json').read_text(encoding='utf-8'))
    if alive(old.get('driver_pid')) or any(alive(s.get('pid')) for b in old.get('batches',[]) for s in b.get('slots',[])):
        raise RuntimeError('An existing R13 process is alive; overlapping launch refused')
    import torch
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory=RAW/'interruption_history'/stamp;directory.mkdir(parents=True)
    archived=[]
    for p in (RAW/'formal').rglob('recovery_state.pt'):
        if 'snapshots' in p.parts or (p.parent/'metrics.json').exists():continue
        state=torch.load(p,weights_only=False)
        target=directory/f"seed{state['job']['seed']}_{state['job'].get('group',state['job']['carrier_kind'])}_step{state['step']}.pt"
        shutil.copyfile(p,target)
        archived.append(dict(original=p.relative_to(ROOT).as_posix(),archive=target.relative_to(ROOT).as_posix(),
            sha256=sha256(target),step=state['step'],job=state['job']))
    record=dict(launched_utc=datetime.now(timezone.utc).isoformat(),driver_pid=os.getpid(),
        launcher='Windows WMI single hidden process',observed_driver_is_in_job=current_in_job(),
        frozen_code_sha256=frozen,recovery_archives=archived,
        no_scientific_code_change=True,no_budget_extension=True,no_fresh_initialization=True,
        previous_processes_confirmed_dead=True,termination_reason_not_assumed=True)
    write_json(directory/'resume_record.json',record)
    write_json(RAW/'detached_launch_status.json',record)
    print('R13 detached exact-resume launch',json.dumps(record,ensure_ascii=False),flush=True)
    from .run_r13_recovery import launch
    launch()

if __name__=='__main__':main()
