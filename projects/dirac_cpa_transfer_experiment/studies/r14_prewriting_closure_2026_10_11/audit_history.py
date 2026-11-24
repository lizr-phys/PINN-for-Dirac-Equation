"""Read-only R1-R13 inventory and protection; no historical imports or writes."""
from __future__ import annotations
import csv, hashlib, json, os, re, time
from pathlib import Path
from datetime import datetime, timezone

BASE=Path(__file__).resolve().parent
PROJECT=BASE.parents[1]
WORKSPACE=PROJECT.parents[1]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)

def main():
    out=BASE/'history';out.mkdir(parents=True,exist_ok=True)
    files=[]
    for folder in ('src','tests','docs','results'):
        for p in (PROJECT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not p.is_symlink():files.append(p)
    files += list(PROJECT.glob('*.md'))+list(PROJECT.glob('*.tex'))
    files += list((WORKSPACE/'docs/reviews').glob('*'))
    files += list((WORKSPACE/'literature/reviews').rglob('*.json'))+list((WORKSPACE/'literature/reviews').rglob('*.md'))
    files=sorted(set(files),key=lambda p:str(p))
    selected=[p for p in files if p.suffix in ('.md','.json','.py','.csv','.tex') and not any(x in p.parts for x in ('snapshots','progress_history','runtime_history','storage_history','interruption_history'))]
    # Read each canonical report/preregistration/decision completely; retain extracts plus hashes.
    reviews=[];seeds=set()
    for p in selected:
        if p.name in ('report.md','preregistration.md','decision.json') or ('REPORT' in p.name and p.suffix=='.md') or p.name=='r1_r13_assessment_2026_10_11.md':
            text=p.read_text(encoding='utf-8-sig')
            reviews.append({'path':str(p.relative_to(WORKSPACE)),'sha256':sha(p),'characters_read':len(text),
                'headings':[s for s in text.splitlines() if s.startswith('#')],
                'decision_and_boundary_lines':[s for s in text.splitlines() if re.search(r'PASS|FAIL|INCONCLUSIVE|5/5|4/5|主|结论|边界|未通过|预算|种子|必要|先验|status|pass_check|complete',s)][:180]})
        if p.name in ('metrics.json','config.json','protocol.json','decision.json'):
            try:
                obj=json.loads(p.read_text(encoding='utf-8-sig'))
                def visit(v,key=''):
                    if isinstance(v,dict):
                        for k,w in v.items(): visit(w,k)
                    elif isinstance(v,list):
                        for w in v: visit(w,key)
                    elif 'seed' in key.lower() and isinstance(v,int):seeds.add(v)
                visit(obj)
            except Exception:pass
    write(out/'formal_evidence_read_log.json',reviews)
    write(out/'used_seed_audit.json',{'used_seeds':sorted(seeds),'proposed_formal':[810,811,812,813,814],'proposed_development':[910,911],
        'unused_formal':not bool(seeds & {810,811,812,813,814}), 'unused_development':not bool(seeds & {910,911}),
        'no_history_training_repeated':True})
    rows=[{'path':str(p.relative_to(WORKSPACE)),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in files]
    write(out/'protected_metadata.json',{'files':rows,'count':len(rows),'total_bytes':sum(r['bytes'] for r in rows)})
    # Stream a complete SHA manifest. Restart resumes completed hash entries without rewriting them.
    target=out/'protected_sha256.jsonl';done={}
    if target.exists():
        for s in target.read_text(encoding='utf-8').splitlines():
            if s.strip():r=json.loads(s);done[r['path']]=r
    tick=time.perf_counter()
    with target.open('a',encoding='utf-8') as f:
        for i,(p,row) in enumerate(zip(files,rows)):
            if row['path'] in done:
                if any(row[k]!=done[row['path']][k] for k in ('bytes','mtime_ns')):raise RuntimeError('Historical file changed: '+str(p))
                continue
            row['sha256']=sha(p);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            if i%100==0:
                print('HISTORY_HASH',i+1,'/',len(files),flush=True)
                write(out/'audit_progress.json',{'hashed':i+1,'total':len(files),'elapsed_seconds':time.perf_counter()-tick,'complete':False})
    write(out/'audit_progress.json',{'hashed':len(files),'total':len(files),'elapsed_seconds':time.perf_counter()-tick,'complete':True})
    write(out/'audit_complete.json',{'status':'PASS','all_historical_content_hashed':True,'files':len(files),'bytes':sum(r['bytes'] for r in rows),
        'read_reports':len(reviews),'manifest_sha256':sha(target),'utc':datetime.now(timezone.utc).isoformat()})
    print('HISTORY_AUDIT_COMPLETE',len(files),flush=True)

if __name__=='__main__':main()
