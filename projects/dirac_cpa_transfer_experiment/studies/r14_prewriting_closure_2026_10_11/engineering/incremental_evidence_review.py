"""Operational review: run python -m engineering.incremental_evidence_review from the study root.

Frozen science and measurement sources are unchanged; this entrypoint never schedules training.
"""
from pathlib import Path
import argparse,contextlib,csv,hashlib,importlib,json,re,shutil,subprocess,sys,time,traceback
from datetime import datetime,timezone,timedelta
import psutil,torch
from threadpoolctl import threadpool_limits
from plan import BASE,sha

PHASES=("audit_new_metrics","completeness","cost_audit","aggregate","audit_statistics",
        "residual_checks","residual_provenance","parameter_audit","carrier_initialization_audit",
        "pairing_audit","carrier_phase_audit","cost_summary","learning_advantage","archive_cohort_receipts")
POST=("audit_inference","residual_checks","residual_provenance","cost_audit","parameter_audit","cost_summary")

def read(name):
    return json.loads((BASE/name).read_text(encoding="utf-8"))

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")

def verify_receipts():
    receipts={}
    for old in read("runtime/continuation_observation.json").get("source_preserved_receipts",[]):
        p=BASE/old["path"]
        assert sha(p)==old["receipt_sha256"]
        receipts[old["path"]]=p
    for p in (BASE/"analysis_outputs/cohort_receipts").glob("*/receipt.json"):
        receipts[p.relative_to(BASE).as_posix()]=p
    rows=[]
    for name,p in sorted(receipts.items()):
        d=read(name);mapping=d.get("preserved_file_sha256",d.get("preserved_audit_files",{}))
        checks={f:sha(p.parent/f)==digest for f,digest in mapping.items()}
        assert all(checks.values()),"Preserved receipt content changed"
        rows.append(dict(path=name,receipt_sha256=sha(p),archived_file_checks=checks,
                         actual_content_checked_this_goal_turn=True))
    return rows

def observe(directory,args):
    previous=read("runtime/continuation_observation.json")
    metric=read("analysis_outputs/independent_metric_audit.json")
    residual=read("analysis_outputs/residual_audit.json")
    provenance=read("analysis_outputs/residual_provenance_audit.json")
    costs=read("analysis_outputs/independent_cost_audit.json")
    params=read("analysis_outputs/independent_parameter_audit.json")
    phase=read("analysis_outputs/carrier_phase_audit.json")
    pairing=read("analysis_outputs/pairing_audit.json")
    carrier=read("analysis_outputs/carrier_initialization_audit.json")
    inference=read("isolated_inference/independent_audit.json")
    ledger=read("verification_progress.json")
    assert not metric["metric_mismatches"] and not costs["mismatches"]
    assert not provenance["errors"] and not params["errors"] and not inference["errors"]
    assert not ledger["frozen_reference_changes"] and not ledger["duplicate_active_training_jobs"]
    assert ledger["driver_alive"]
    # The ledger checks actual process commands, not only RUNNING text.
    with (BASE/"analysis_outputs/independent_metrics_per_snapshot.csv").open(encoding="utf-8-sig",newline="") as f:
        checked={r["path"].replace("\\","/") for r in csv.DictReader(f)}
    present={p.relative_to(BASE).as_posix() for folder in ("formal/coupled","development")
             for p in (BASE/folder).rglob("metrics.json")}
    receipts=verify_receipts()
    if (BASE/"delivery_exports/verified_static_2026_10_12/receipt.json").exists():
        static=read("delivery_exports/verified_static_2026_10_12/receipt.json")
        for x in static["exports"]:
            assert sha(BASE/x["target"])==sha(BASE/x["source"])==x["sha256"]
    now=time.time();local=datetime.fromtimestamp(now,timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    obs=dict(previous)
    obs.update(observed_utc=now,local_time=local,status="RUNNING" if ledger["driver_alive"] else "REQUIRES_REVIEW",
        formal_complete=ledger["formal_training_endpoints"],development_complete=ledger["development_endpoints"],
        driver_pid=read("progress.json")["driver_pid"],workers=ledger["workers"],
        actual_task_processes=ledger["actual_processes"],source_preserved_receipts=receipts,
        audit_run_directory=directory.relative_to(BASE).as_posix(),currently_audited_neural_snapshots=metric["snapshots_audited"],
        cost_records_independently_checked=costs["cost_records_checked"],
        JVP_endpoints_independently_checked=residual["trained_endpoints_checked"],
        inference_verified_endpoints=inference["actual_measurements_checked"],
        actual_initial_states_checked=params["actual_initial_states_checked"],
        direct_phase_diagnostics_checked=phase["snapshots_checked"],
        actual_mechanism_initializers_checked=pairing["actual_initializers_checked"],
        actual_terminal_trace_prefixes_checked=pairing["actual_terminal_trace_prefixes_checked"],
        full_mechanism_seed_blocks_checked=pairing["seed_blocks"],
        carrier_initial_pairs_checked=carrier["actual_carrier_pairs_checked"],
        joint_initializers_checked=carrier["actual_joint_initializers_checked"],
        new_snapshot_paths_pending_audit=sorted(present-checked),app_goal_status_last_observed=args.goal_status,
        app_goal_status_updated_utc=args.goal_updated_utc,previous_goal_turn_classification="progress",
        free_disk_bytes=ledger["free_E_bytes"],available_RAM_bytes=ledger["available_RAM_bytes"],
        whole_study_complete=False,goal_complete=False,real_external_blocker_present=False,
        no_early_stop_or_budget_change=True,
        next_step="Continue every frozen training attempt and remaining verification; new live snapshots remain pending, not completed. Scope, zero-shot45, actual visual QA, final historical integrity and final reports remain required.")
    save(directory/"continuation_observation_before.json",previous)
    p=BASE/"CONTINUE.md";text=p.read_text(encoding="utf-8")
    shutil.copy2(p,directory/"CONTINUE_before_incremental_review.md")
    number=max(map(int,re.findall(r"(?m)^(\d+)\. ",text)))+1
    created=json.loads((directory/"archive_cohort_receipts.log").read_text(encoding="utf-8"))["created"]
    record=(f"\n{number}. 按原冻结方案增量复核：{metric['snapshots_audited']}指标、"
        f"{residual['trained_endpoints_checked']} JVP/来源、{costs['cost_records_checked']}成本、"
        f"{params['actual_initial_states_checked']}初态、{phase['snapshots_checked']}相位记录通过；"
        f"独立推理实测及七次中位数/来源复核{inference['actual_measurements_checked']}/255。"
        f"机制初态{pairing['actual_initializers_checked']}、终止前缀{pairing['actual_terminal_trace_prefixes_checked']}，"
        f"完整块{pairing['seed_blocks']}/15。新完整组首次收据{len(created)}份，已有收据各份归档SHA再次一致，不覆盖。"
        f"正式{ledger['formal_training_endpoints']}/210、开发{ledger['development_endpoints']}/24为{local}清点；"
        f"{len(present-checked)}份新生快照待重算。操作包装归并为engineering/incremental_evidence_review.py，"
        "仍调用原测量模块，默认float64/单线程、逐阶段日志，科学源/种子/预算/判据不改；不自动验收或关闭目标。"
        f"日志在{directory.relative_to(BASE).as_posix()}。\n")
    assert "\n## 接下来必须完成\n" in text
    text=text.replace("\n## 接下来必须完成\n",record+"\n## 接下来必须完成\n",1)
    p.write_text(text,encoding="utf-8");obs["CONTINUE_sha256"]=sha(p)
    save(directory/"continuation_observation.json",obs)
    save(BASE/"runtime/continuation_observation.json",obs)
    return dict(observed=local,formal=obs["formal_complete"],metrics=metric["snapshots_audited"],
                residuals=residual["trained_endpoints_checked"],inference=inference["actual_measurements_checked"],
                pending_new_snapshots=sorted(present-checked),new_cohorts=created,whole_goal_complete=False)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--goal-status",required=True,choices=("active","blocked"))
    parser.add_argument("--goal-updated-utc",required=True,type=int)
    args=parser.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);psutil.Process().cpu_affinity([6])
    for p in psutil.process_iter(["pid","name","cmdline"]):
        cmd=p.info.get("cmdline") or []
        if p.pid!=psutil.Process().pid and (p.info.get("name") or "").lower().startswith("python"):
            file_entry=any(str(BASE).lower() in x.lower() and x.endswith("incremental_evidence_review.py") for x in cmd)
            module_entry="engineering.incremental_evidence_review" in cmd and Path(p.cwd()).resolve()==BASE.resolve()
            if file_entry or module_entry:
                raise RuntimeError("An operational incremental review is already active")
    directory=BASE/"runtime/incremental_audit_runs"/str(time.time_ns());directory.mkdir(parents=True)
    print("RUN_DIRECTORY",directory.relative_to(BASE).as_posix(),flush=True)
    before=directory/"before";before.mkdir()
    for p in (BASE/"analysis_outputs").glob("*.json"):shutil.copy2(p,before/p.name)
    for name in ("progress.json","progress.md","CONTINUE.md","runtime/continuation_observation.json"):
        shutil.copy2(BASE/name,before/Path(name).name)
    result=dict(modules=[],whole_goal_complete=False,operational_wrapper_sha256=sha(__file__),
                original_measurement_sources_not_edited=True)
    def execute(module,suffix=""):
        log=directory/(module.rsplit(".",1)[-1]+suffix+".log")
        row=dict(module=module,started_utc=time.time(),log=log.relative_to(BASE).as_posix())
        print("START",module,flush=True)
        try:
            with log.open("w",encoding="utf-8") as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
                importlib.import_module(module).main()
            row.update(status="EXECUTED",finished_utc=time.time())
        except Exception:
            row.update(status="ENGINEERING_OR_VALIDATION_ERROR",finished_utc=time.time(),traceback=traceback.format_exc())
            result["modules"].append(row);save(directory/"result.json",result);raise
        result["modules"].append(row);save(directory/"result.json",result);print("DONE",module,flush=True)
    with threadpool_limits(limits=1):
        for name in PHASES:execute("analysis."+name)
        # Inference has its own PID/creation-time lock and source-SHA checks.
        for p in psutil.process_iter(["pid","name","cmdline"]):
            if (p.info.get("name") or "").lower().startswith("python") and "analysis.inference_benchmark" in (p.info.get("cmdline") or []):
                raise RuntimeError("An inference benchmark is already active")
        psutil.Process().cpu_affinity([7])
        log=directory/"inference_benchmark.log";started=time.time()
        with log.open("w",encoding="utf-8") as f:
            proc=subprocess.run([sys.executable,"-X","utf8","-u","-m","analysis.inference_benchmark"],cwd=BASE,stdout=f,stderr=subprocess.STDOUT)
        result["modules"].append(dict(module="analysis.inference_benchmark",started_utc=started,finished_utc=time.time(),
            log=log.relative_to(BASE).as_posix(),status="EXECUTED" if proc.returncode==0 else "ENGINEERING_OR_VALIDATION_ERROR",
            returncode=proc.returncode))
        save(directory/"result.json",result)
        if proc.returncode:raise RuntimeError("Inference benchmark failed; preserve its log and resolve engineering issue")
        psutil.Process().cpu_affinity([6])
        for name in POST:execute("analysis."+name,"_after_inference")
        execute("verification_progress")
    summary=observe(directory,args);save(directory/"review_summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__":main()

