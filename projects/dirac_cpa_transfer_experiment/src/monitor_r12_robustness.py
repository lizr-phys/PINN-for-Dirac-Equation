from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime,timezone
from .r12_config import RAW,LOGS,CASES,GROUPS,SEEDS
from .freeze_r7_phase_release import write_json

def snapshot():
    from .resume_r12_robustness import live_processes
    live = live_processes()
    rows=[]
    for f in sorted(LOGS.glob('*_slot_*_seed*.log')):
        lines=f.read_text(encoding='utf-8').splitlines()
        rows.append(dict(log=f.name,log_updated_utc=datetime.fromtimestamp(f.stat().st_mtime,timezone.utc).isoformat(),
            last_progress=next((s for s in reversed(lines) if 'step=' in s or 'CHECKPOINT' in s),''),
            error_tail='\n'.join(lines[-8:]) if any('Traceback' in s for s in lines[-10:]) else None))
    complete=list((RAW/'formal').glob('*/*/seed_*/metrics.json'))
    plan=json.loads((RAW/'runtime_plan.json').read_text(encoding='utf-8'))
    observed_status=plan['status'] if plan['status']!='RUNNING' or live else 'INTERRUPTED_NO_LIVE_PROCESS'
    record=dict(utc=datetime.now(timezone.utc).isoformat(),runtime_status=observed_status,
        recorded_runtime_status=plan['status'],active_training_processes=live,
        completed_runs=len(complete),total_runs=80,
        saved_fixed_snapshots=len(list((RAW/'formal').glob('*/*/seed_*/snapshots/*/metrics.json'))),
        expected_fixed_snapshots=380,
        batch_statuses=[dict(case=b['case'],status=b['status']) for b in plan['batches']],worker_logs=rows,
        postprocessing=json.loads((RAW/'postprocessing_status.json').read_text(encoding='utf-8'))
            if (RAW/'postprocessing_status.json').exists() else None)
    write_json(RAW/'progress_snapshot.json',record);return record

if __name__=='__main__':print(json.dumps(snapshot(),ensure_ascii=False,indent=2))
