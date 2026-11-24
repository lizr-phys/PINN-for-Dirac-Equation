from __future__ import annotations
import json
import os
import time
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime,timezone
from .r12_config import ROOT,RAW,FIGURES,LOGS
from .freeze_r7_phase_release import write_json

def status(stage,state='RUNNING',**extra):
    write_json(RAW/'postprocessing_status.json',dict(stage=stage,status=state,utc=datetime.now(timezone.utc).isoformat(),**extra))

def finalize():
    from .r7_runtime import set_cpu_affinity
    set_cpu_affinity(128)
    while True:
        plan=json.loads((RAW/'runtime_plan.json').read_text(encoding='utf-8'))
        if plan['status']=='COMPLETE':break
        if plan['status']=='FAIL':
            status('training','FAIL',reason='Fixed formal attempts failed; retained for diagnosis')
            raise RuntimeError('Formal training failures retained')
        time.sleep(20)
    status('aggregate')
    from .aggregate_r12_robustness import aggregate
    aggregate()
    status('audit_before_figures')
    from .audit_r12_robustness import audit
    audit()
    status('figure_export')
    from .plot_r12_robustness import plot
    plot()
    status('pdf_render_for_visual_QA')
    from .qa_r12_robustness_figures import prepare_visual_qa
    qa=prepare_visual_qa()
    from .build_r12_robustness_report import build
    build()
    status('await_actual_visual_inspection','NEEDS_VISUAL_QA',figure_count=len(qa['figures']))
    print('R12 all numerical outputs and 12 figures ready for actual PNG/PDF visual inspection',flush=True)

def verify():
    from .r7_runtime import set_cpu_affinity
    set_cpu_affinity(128)
    qa=json.loads((FIGURES/'FIGURE_QA.json').read_text(encoding='utf-8'))
    if qa['status']!='PASS' or qa['visual_inspection_pending']:raise RuntimeError('Actual visual QA required first')
    if (RAW/'pytest_final_result.json').exists():raise RuntimeError('Preserve previous full-test result')
    status('full_repository_pytest')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    xml=RAW/'pytest_final.xml';log=LOGS/'pytest_final.log';started=datetime.now(timezone.utc)
    with log.open('w',encoding='utf-8') as stream:
        done=subprocess.run([sys.executable,'-X','utf8','-m','pytest','-q','--junitxml='+str(xml)],cwd=ROOT,
            env=env,stdout=stream,stderr=subprocess.STDOUT,check=False)
    suites=list(ET.parse(xml).getroot().iter('testsuite')) if xml.exists() else []
    totals={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','errors','failures','skipped')}
    result=dict(status='PASS' if done.returncode==0 else 'FAIL',**totals,exit_code=done.returncode,
        started_utc=started.isoformat(),ended_utc=datetime.now(timezone.utc).isoformat(),
        scope='complete repository pytest',cpu_affinity_mask=128,log=log.relative_to(ROOT).as_posix())
    write_json(RAW/'pytest_final_result.json',result)
    if done.returncode:
        status('full_repository_pytest','FAIL');raise RuntimeError('Full pytest failed; preserve evidence')
    from .build_r12_robustness_report import build
    build();status('final_audit')
    from .audit_r12_robustness import audit,artifact_manifest,check_links
    from .r12_config import DOCS
    audit(final=True);build()
    links=check_links([ROOT/'R12_ROBUSTNESS_REPORT.md']+list(DOCS.glob('*.md'))+[FIGURES/'FIGURE_QA.md'])
    write_json(RAW/'link_check.json',links)
    if links['status']!='PASS':raise RuntimeError('Final links failed')
    build();status('complete','COMPLETE',artifact_files=artifact_manifest(),full_pytest=result)
    print('R12 final verification COMPLETE',flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    verify() if args.verify else finalize()
