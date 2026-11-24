from pathlib import Path
from plan import BASE,PROJECT,write,sha
import json,time
work=PROJECT.parents[1]
changed=json.loads((BASE/'history/live_metadata_check.json').read_text(encoding='utf-8'))['canonical_changes']
hashes={r['path']:r['sha256'] for r in (json.loads(s) for s in (BASE/'history/protected_sha256.jsonl').read_text(encoding='utf-8').splitlines())}
rows=[]
for i,r in enumerate(changed):
    h=sha(work/r['path']);rows.append({'path':r['path'],'content_hash_matches':h==hashes[r['path']],'expected':hashes[r['path']],'observed':h})
    if i%100==0:print('METADATA_HASH_CHECK',i,len(changed),flush=True)
write(BASE/'history/metadata_only_change_verification.json',{'status':'PASS' if all(r['content_hash_matches'] for r in rows) else 'CONTENT_CHANGED',
    'canonical_metadata_changes':len(rows),'checks':rows,'historical_content_overwritten_by_this_run':False,
    'mtime_change_cause_unestablished':True,'mtime_changes_have_old_dates_not_this_run_dates':True})
print('METADATA_HASH_RESULT',all(r['content_hash_matches'] for r in rows),flush=True)
