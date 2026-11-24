from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime,timezone
from .r13_config import ROOT,RAW,LOGS,FIGURES,DOCS
from .freeze_r7_phase_release import write_json,sha256

def finalize(*,reuse_existing_figures=False,reuse_verified_tests=False):
    from .aggregate_r13_recovery import aggregate
    decision=aggregate()
    if decision['incomplete_jobs']:raise RuntimeError('Prescribed attempts remain incomplete; cannot finalize')
    from .plot_r13_recovery import plot_all
    from .qa_r13_recovery_figures import qa
    from .build_r13_recovery_report import build
    from .audit_r13_recovery import audit
    if not (RAW/'audit.json').exists():write_json(RAW/'audit.json',dict(status='PENDING'))
    build()
    if reuse_existing_figures:
        if not (FIGURES/'figure_registry.json').is_file():raise RuntimeError('Existing figure registry absent')
    else:plot_all()
    figure_qa=qa()
    # The report links the audit file; create the placeholder before checking links.
    if not (RAW/'audit.json').exists():write_json(RAW/'audit.json',dict(status='PENDING'))
    build()
    log=LOGS/'full_pytest_final.log';xml=LOGS/'full_pytest_final.xml'
    if reuse_verified_tests:
        from .freeze_r13_recovery import verify_freeze
        tests=json.loads((RAW/'final_test_result.json').read_text(encoding='utf-8'))
        if (tests['status']!='PASS' or tests['exit_code']!=0 or
            tests['verified_scientific_sha256']!=verify_freeze() or
            tests['log_sha256']!=sha256(log) or tests['xml_sha256']!=sha256(xml)):
            raise RuntimeError('Completed full tests do not match the immutable scientific sources and saved outputs')
        test_exit_code=0
    else:
        with log.open('w',encoding='utf-8') as output:
            result=subprocess.run([sys.executable,'-X','utf8','-m','pytest','-q','--disable-warnings','--junitxml='+str(xml)],
                cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        test_exit_code=result.returncode
        write_json(RAW/'final_test_result.json',dict(status='PASS' if test_exit_code==0 else 'FAIL',
            exit_code=test_exit_code,log_sha256=sha256(log),xml_sha256=sha256(xml)))
    audited=audit();build()
    status='PASS' if test_exit_code==0 and audited['status']=='PASS' and figure_qa['status']=='PASS' else (
        'VISUAL_REVIEW_PENDING' if test_exit_code==0 and audited['status']=='PASS' and figure_qa['status']=='AUTOMATED_PASS_VISUAL_REVIEW_PENDING' else 'FAIL')
    write_json(RAW/'deliverable_status.json',dict(status=status,training_complete=True,decision_status=decision['status'],
        audit_status=audited['status'],figure_qa_status=figure_qa['status'],full_pytest_exit_code=test_exit_code,
        reused_existing_figures=reuse_existing_figures,reused_verified_final_tests=reuse_verified_tests))
    files={}
    for base in (RAW,FIGURES,LOGS,DOCS):
        for p in base.rglob('*'):
            if p.is_file() and p.name not in ('artifact_manifest.json','runtime_plan.json'):
                files[p.relative_to(ROOT).as_posix()]=sha256(p)
    report=ROOT/'R13_LONG_TIME_RECOVERY_REPORT.md';files[report.relative_to(ROOT).as_posix()]=sha256(report)
    for p in (ROOT/'src').glob('*r13*.py'):
        files[p.relative_to(ROOT).as_posix()]=sha256(p)
    write_json(RAW/'artifact_manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),files=files))
    if status=='FAIL':raise RuntimeError('Postprocessing checks need attention; training results preserved')
    return status

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-existing-figures',action='store_true');parser.add_argument('--reuse-verified-tests',action='store_true')
    args=parser.parse_args()
    print(finalize(reuse_existing_figures=args.reuse_existing_figures,reuse_verified_tests=args.reuse_verified_tests))
