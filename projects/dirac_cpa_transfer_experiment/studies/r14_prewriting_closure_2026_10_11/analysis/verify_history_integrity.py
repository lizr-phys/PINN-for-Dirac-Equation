"""Final read-only content check against the original protection manifest; never rewrite it."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from plan import BASE,write,sha

def main(inventory_only=False):
    workspace=BASE.parents[3];manifest=BASE/'history/protected_sha256.jsonl'
    initial=json.loads((BASE/'history/audit_complete.json').read_text(encoding='utf-8'))
    if sha(manifest)!=initial['manifest_sha256']:raise RuntimeError('Original historical protection manifest changed')
    records=[json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(records)!=initial['files']:raise RuntimeError('Historical protection inventory count changed')
    if inventory_only:
        print(json.dumps(dict(status='BASELINE_MANIFEST_VALIDATED_ONLY',protected_paths=len(records),
            baseline_manifest_sha256=sha(manifest),historical_content_rehash_executed=False),ensure_ascii=False));return
    run=BASE/'history/final_integrity_runs'/str(time.time_ns());run.mkdir(parents=True)
    output=run/'checked_sha256.jsonl';start=time.time();checked={};errors=[];metadata_changes=0
    # Re-open every canonical file for this final run. Alias hashes may share a physical path.
    with output.open('w',encoding='utf-8') as stream:
        for index,record in enumerate(records,1):
            path=workspace/record['path'];row=dict(original_path=record['path'],expected_sha256=record['sha256'])
            try:
                canonical=path.resolve().as_posix().casefold();stat=path.stat()
                if canonical not in checked:checked[canonical]=sha(path)
                digest=checked[canonical];passed=stat.st_size==record['bytes'] and digest==record['sha256']
                row.update(current_sha256=digest,pass_check=passed,current_bytes=stat.st_size,
                    current_mtime_ns=stat.st_mtime_ns,original_mtime_ns=record['mtime_ns'],canonical_path=canonical)
                metadata_changes+=int(stat.st_mtime_ns!=record['mtime_ns'])
                if not passed:errors.append(dict(path=record['path'],reason='CONTENT_OR_SIZE_CHANGED'))
            except Exception as error:
                row.update(pass_check=False,error=repr(error));errors.append(dict(path=record['path'],error=repr(error)))
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            if index%500==0:
                stream.flush();write(BASE/'history/final_integrity_progress.json',dict(status='REHASH_RUNNING',
                    checked=index,total=len(records),content_errors=len(errors),elapsed_seconds=time.time()-start))
                print('FINAL_HISTORY_REHASH',index,'/',len(records),flush=True)
    report=dict(updated_utc=time.time(),status='PASS' if not errors else 'FAIL',protected_paths_checked=len(records),
        canonical_files_rehashed=len(checked),errors=errors,metadata_changed_paths=metadata_changes,
        baseline_manifest_sha256=sha(manifest),final_check_manifest_sha256=sha(output),elapsed_wall_seconds=time.time()-start,
        final_check_manifest_path=output.relative_to(BASE).as_posix(),
        original_content_matches_before_R14=not errors,no_historical_files_written_by_this_verifier=True,
        metadata_change_does_not_override_content_SHA=True)
    write(run/'report.json',report)
    write(BASE/'history/final_integrity_report.json',report);print(json.dumps(report,ensure_ascii=False))
    if errors:raise RuntimeError('Historical content integrity failures require investigation')

if __name__=='__main__':
    import psutil
    psutil.Process().cpu_affinity([6])
    parser=argparse.ArgumentParser();parser.add_argument('--inventory-only',action='store_true');args=parser.parse_args();main(args.inventory_only)
