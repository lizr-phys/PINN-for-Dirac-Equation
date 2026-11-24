"""Fresh file checks for every cached independent residual observation."""
import json,time
from plan import BASE,sha,write

def main():
    rows=[];errors=[]
    for path in sorted((BASE/'analysis_outputs/residual_checks').glob('*.json')):
        record=json.loads(path.read_text(encoding='utf-8'));metric=BASE/record['path'];directory=metric.parent
        sources=((metric,'metric_sha256'),(directory/'model_state.pt','checkpoint_sha256'),
            (directory/'prediction.npz','prediction_sha256'),(directory/'residual_integral.csv','residual_table_sha256'),
            (directory/'residual_by_mode.csv','by_mode_table_sha256'),(BASE/'analysis/residual_checks.py','script_sha256'))
        changed=[p.relative_to(BASE).as_posix() for p,key in sources if not p.exists() or sha(p)!=record[key]]
        row=dict(audit_record=path.relative_to(BASE).as_posix(),audit_record_sha256=sha(path),
            source_metric=record['path'],pass_check=not changed and record['pass_check'],changed=changed)
        rows.append(row)
        if not row['pass_check']:errors.append(row)
    report=dict(updated_utc=time.time(),status='PASS_CURRENT_RESIDUAL_SOURCE_CHAIN' if rows and not errors else 'MISMATCH_OR_MISSING',
        independent_cached_records_checked=len(rows),errors=errors,checks=rows,current_raw_tables_and_checkpoint_SHA_verified=True,
        cached_observations_not_recomputed_or_overwritten=True,whole_study_complete=False)
    write(BASE/'analysis_outputs/residual_provenance_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False))
    if errors:raise RuntimeError('Cached residual evidence source changed; preserve and investigate')

if __name__=='__main__':main()
