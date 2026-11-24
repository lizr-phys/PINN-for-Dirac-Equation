"""Local verification ledger, separate from the live training driver's progress."""
from __future__ import annotations
import json,time,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
import psutil
from plan import BASE,write,sha

def main():
    from closure.train import verify_freeze
    science=verify_freeze();references=json.loads((BASE/'frozen/reference_manifest.json').read_text(encoding='utf-8'))
    changed=[r['path'] for r in references if sha(BASE/r['path'])!=r['sha256']]
    training=json.loads((BASE/'progress.json').read_text(encoding='utf-8'))
    def read(name):
        p=BASE/name
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else dict(status='NOT_EXECUTED')
    files={'theoretical_closure':'engineering/theory_closure_audit.json',
        'attempts':'analysis_outputs/attempt_completion_audit.json','metrics':'analysis_outputs/independent_metric_audit.json',
        'residuals':'analysis_outputs/residual_audit.json','residual_sources':'analysis_outputs/residual_provenance_audit.json',
        'postprocessing':'runtime/postprocessing_progress.json',
        'costs':'analysis_outputs/independent_cost_audit.json','cost_summary':'analysis_outputs/cost_summary.json',
        'statistics':'analysis_outputs/independent_statistical_audit.json',
        'inference_cost_provenance':'isolated_inference/independent_audit.json',
        'parameter_counts':'analysis_outputs/independent_parameter_audit.json',
        'spectral_benchmark':'analysis_outputs/independent_TSFP_audit.json',
        'pairing':'analysis_outputs/pairing_audit.json','phase':'analysis_outputs/carrier_phase_audit.json',
        'carrier_initial_controls':'analysis_outputs/carrier_initialization_audit.json',
        'learning_advantage':'analysis_outputs/learning_advantage_assessment.json','isolated_inference':'isolated_inference/status.json',
        'history_early':'history/independent_R1_R8_audit.json','history_late':'history/independent_R9_R13_audit.json',
        'history_snapshots':'history/independent_canonical_snapshot_audit.json','history_row_links':'history/canonical_row_traceability_audit.json',
        'zero_shot_audit':'zero_shot/independent_audit.json',
        'links':'link_check.json','zero_shot':'zero_shot/complete.json','visual':'figures/actual_visual_QA.json'}
    def compact(value,source):
        if isinstance(value,dict):return {k:compact(v,source) for k,v in value.items()}
        if isinstance(value,list):
            return [compact(v,source) for v in value] if len(value)<=10 else dict(record_count=len(value),full_entries_in=source)
        return value
    records={k:compact(read(v),v) for k,v in files.items()}
    driver_alive=False
    try:
        driver_process=psutil.Process(training['driver_pid'])
        driver_alive=any(s.endswith('driver.py') for s in driver_process.cmdline()) and any(str(BASE).lower() in s.lower() for s in driver_process.cmdline())
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    workers=[]
    for child in training['active_children']:
        job=child['job'];p=BASE/job['directory']/'progress.json';progress=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
        alive=False
        try:
            worker=psutil.Process(child['pid']);cmd=worker.cmdline()
            alive='closure.train' in cmd and worker.ppid()==training['driver_pid'] and any(str(BASE).lower() in s.lower() for s in cmd)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        workers.append(dict(pid=child['pid'],alive=alive,job=job['directory'],last_saved_step=progress.get('step',0)))
    actual=[]
    for p in psutil.process_iter(['pid','ppid','name','cmdline','create_time']):
        cmd=p.info['cmdline'] or [];name=p.info['name'] or ''
        if name.lower() in ('python.exe','pythonw.exe') and any(str(BASE).lower() in s.lower() for s in cmd) and ('closure.train' in cmd or any(s.endswith('driver.py') for s in cmd)):
            actual.append(p.info)
    actual_jobs=[p['cmdline'][-1] for p in actual if 'closure.train' in p['cmdline']]
    duplicate=len(actual_jobs)!=len(set(actual_jobs))
    report=dict(updated_utc=time.time(),status='WORK_IN_PROGRESS',scientific_source_sha256=science,frozen_reference_changes=changed,
        driver_alive=driver_alive,observed_training_stage=training['stage'],formal_training_endpoints=training['formal_complete'],formal_planned=210,
        development_endpoints=training['development_complete'],development_planned=24,actual_processes=actual,duplicate_active_training_jobs=duplicate,
        workers=workers,available_RAM_bytes=psutil.virtual_memory().available,free_E_bytes=shutil.disk_usage(BASE).free,
        numerical_or_dependency_failures=training['preserved_failures'],completed_training_is_not_method_accuracy_success=True,
        entire_goal_accepted_complete=False,verification=records,verification_sources=files)
    write(BASE/'verification_progress.json',report)
    now=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    text=f'# R14 独立验证进度\n\n更新时间：{now}。整体证据目标尚未验收。\n\n'
    text+=f"正式固定终点 {training['formal_complete']}/210；开发 {training['development_complete']}/24；数值/依赖失败 {training['preserved_failures']}。完成训练不代表准确度通过，完整五种子判定见独立汇总。\n\n"
    text+=f"驱动器真实存活：{driver_alive}；活跃重复训练：{duplicate}；冻结参考改动：{len(changed)}。科学源码SHA：{science}。\n\n"
    text+='| 验证项 | 当前记录 |\n| --- | --- |\n'
    for label,record in records.items():
        status=record.get('status','ACTUAL_VISUAL_CHECK_RECORDS' if label=='visual' else 'NOT_EXECUTED')
        display=f'[{status}]({files[label]})' if (BASE/files[label]).exists() else f'{status}（尚无产物）'
        text+=f'| {label} | {display} |\n'
    text+='\n未完成：全部预定训练与终点验证、15个核心配对块、45项零样本、全部模型推理/内存测量、正式图组、最终主张证据表及投稿定位。数字以各项最近运行时间为准；它们不会随训练自动被标为完成。\n'
    (BASE/'verification_progress.md').write_text(text,encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('verification','verification_sources','actual_processes','workers')},ensure_ascii=False))

if __name__=='__main__':main()

