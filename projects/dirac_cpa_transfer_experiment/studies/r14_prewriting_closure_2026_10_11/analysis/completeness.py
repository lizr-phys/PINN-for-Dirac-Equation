"""Check every prescribed attempt and endpoint, without declaring the study complete."""
from __future__ import annotations
import json,time
import pandas as pd
from plan import BASE,write,sha
from .expected_jobs import expected_jobs

def inspect_job(job,check_hashes=True):
    directory=BASE/job['directory'];complete=directory/'complete.json';failure=directory/'failure.json'
    issues=[];snapshots=[];terminal=False
    if complete.exists() and failure.exists():issues.append('BOTH_COMPLETE_AND_FAILURE')
    for step in job['snapshots']:
        s=directory/'snapshots'/f'{step:05d}';m=s/'metrics.json';cp=s/'model_state.pt'
        ready=all((s/name).exists() for name in ('metrics.json','cost.json','model_state.pt','prediction.npz','time_series.csv'))
        if ready:
            metric=json.loads(m.read_text(encoding='utf-8'));cost=json.loads((s/'cost.json').read_text(encoding='utf-8'))
            valid=metric['job']==job and metric['step']==step and cost['step']==step
            if check_hashes:valid=valid and metric['checkpoint_sha256']==sha(cp)
            if not valid:issues.append(f'SNAPSHOT_PROVENANCE_{step}')
            ready=ready and valid
        snapshots.append(dict(step=step,artifacts_verified=ready))
        if step==job['budget']:terminal=ready
    state='INCOMPLETE'
    if complete.exists():
        record=json.loads(complete.read_text(encoding='utf-8'))
        if record['job']!=job or record.get('step')!=job['budget'] or record.get('fixed_endpoint_reached') is not True:
            issues.append('COMPLETION_NOT_PRESCRIBED_ENDPOINT')
        if not terminal:issues.append('COMPLETION_MISSING_VERIFIED_ENDPOINT')
        for s in snapshots:
            if not s['artifacts_verified']:issues.append(f"COMPLETION_MISSING_PRESCRIBED_SNAPSHOT_{s['step']}")
        if not (directory/'model_state.pt').exists():issues.append('MISSING_TERMINAL_CHECKPOINT')
        elif terminal and check_hashes and sha(directory/'model_state.pt')!=sha(directory/'snapshots'/f"{job['budget']:05d}"/'model_state.pt'):
            issues.append('TERMINAL_CHECKPOINT_DIFFERS_FROM_ENDPOINT')
        state='COMPLETE_VERIFIED' if not issues else 'INVALID_COMPLETION'
    elif failure.exists():
        record=json.loads(failure.read_text(encoding='utf-8'))
        if record['job']!=job:issues.append('FAILURE_JOB_DIFFERS')
        kind=record.get('status')
        if kind=='NUMERICAL_FAILURE':
            if not (directory/'failed_state.pt').exists():issues.append('MISSING_FAILED_STATE')
            for s in snapshots:
                if s['step']<record.get('occurred_step',0) and not s['artifacts_verified']:
                    issues.append(f"FAILURE_MISSING_EARLIER_PRESCRIBED_SNAPSHOT_{s['step']}")
        elif kind=='DEPENDENCY_FAILED':
            source=BASE/record['source']/'failure.json'
            if not source.exists():issues.append('DEPENDENCY_WITHOUT_SOURCE_FAILURE')
        else:issues.append('UNRECOGNIZED_FAILURE')
        state='FAILURE_PRESERVED' if not issues else 'INVALID_FAILURE'
    return dict(**{k:v for k,v in job.items() if isinstance(v,(str,int,bool))},state=state,
        terminal_artifacts_verified=terminal,available_snapshots=sum(s['artifacts_verified'] for s in snapshots),
        prescribed_snapshots=len(snapshots),snapshot_checks=snapshots,issues=issues)

def main():
    jobs=expected_jobs();rows=[inspect_job(j) for j in jobs]
    prescribed={j['directory'] for j in jobs};unexpected=[]
    for root in ('formal','development'):
        for name in ('complete.json','failure.json'):
            for p in (BASE/root).rglob(name):
                if p.parent.relative_to(BASE).as_posix() not in prescribed:unexpected.append(p.relative_to(BASE).as_posix())
    counts={stage:{state:sum(r['stage']==stage and r['state']==state for r in rows) for state in
        ('COMPLETE_VERIFIED','FAILURE_PRESERVED','INCOMPLETE','INVALID_COMPLETION','INVALID_FAILURE')}
        for stage in ('development','core','extension')}
    all_attempts=all(r['state'] in ('COMPLETE_VERIFIED','FAILURE_PRESERVED') for r in rows) and not unexpected
    report=dict(updated_utc=time.time(),status='ALL_PRESCRIBED_ATTEMPTS_VERIFIED' if all_attempts else 'INCOMPLETE',
        planned_development=24,planned_formal=210,counts=counts,unexpected_attempts=unexpected,
        all_training_attempts_verified=all_attempts,entire_evidence_package_complete=False,
        failure_is_not_success=True,missing_endpoint_is_not_complete=True,rows=rows)
    out=BASE/'analysis_outputs';out.mkdir(exist_ok=True);write(out/'attempt_completion_audit.json',report)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ('snapshot_checks','issues')}|
        {'issues':';'.join(r['issues'])} for r in rows]).to_csv(out/'prescribed_attempt_status.csv',index=False)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False))

if __name__=='__main__':main()
