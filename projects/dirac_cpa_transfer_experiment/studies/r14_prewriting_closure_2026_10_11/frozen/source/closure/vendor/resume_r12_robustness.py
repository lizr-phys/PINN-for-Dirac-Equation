"""Operational recovery only: never duplicate live workers or change scientific budgets."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from datetime import datetime,timezone
from .r12_config import RAW,ROOT,LOGS

def live_processes():
    command="Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    result=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    records=json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(records,dict):records=[records]
    return [dict(pid=r['ProcessId'],arguments=r['CommandLine']) for r in records
        if 'src.run_r12_robustness' in (r['CommandLine'] or '') and
        (' formal' in r['CommandLine'] or ' worker' in r['CommandLine'])]

def resume():
    existing=live_processes()
    if existing:
        print(json.dumps(dict(status='ALREADY_RUNNING_NO_DUPLICATE_LAUNCH',processes=existing),ensure_ascii=False,indent=2))
        return
    plan_path=RAW/'runtime_plan.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text(encoding='utf-8'))
        if plan['status']=='COMPLETE':
            print('All fixed training completed; use recorded postprocessing status. No retraining.')
            return
        directory=RAW/'runtime_history';directory.mkdir(exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        (directory/(stamp+'_interrupted_runtime_plan.json')).write_text(plan_path.read_text(encoding='utf-8'),encoding='utf-8')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result=subprocess.run([sys.executable,'-X','utf8','-u','-m','src.run_r12_robustness','formal'],cwd=ROOT,env=env,check=False)
    if result.returncode:raise RuntimeError('Recovery retained failed attempts; see logs')

if __name__=='__main__':resume()
