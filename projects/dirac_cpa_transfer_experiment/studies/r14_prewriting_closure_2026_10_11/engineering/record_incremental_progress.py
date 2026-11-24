"""Record an already executed review; never schedule or alter scientific work."""
import argparse
import csv
import hashlib
import json
import re
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil

BASE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exclusive(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--initial-check", required=True)
    parser.add_argument("--goal-status", choices=("active", "blocked"), required=True)
    parser.add_argument("--goal-updated-utc", type=int, required=True)
    args = parser.parse_args()
    assert args.run.isdecimal() and args.previous.isdecimal()
    run = BASE / "runtime/incremental_audit_runs" / args.run
    classification_path = run / "goal_turn_progress_classification.json"
    assert not classification_path.exists(), "Existing first classification must be preserved"
    result = read(run / "result.json")
    assert len(result["modules"]) == 22
    assert all(row["status"] == "EXECUTED" for row in result["modules"])
    assert result["operational_wrapper_sha256"] == sha(BASE / "engineering/incremental_evidence_review.py")
    assert result["whole_goal_complete"] is False
    summary = read(run / "review_summary.json")
    previous = read(BASE / "runtime/incremental_audit_runs" / args.previous / "review_summary.json")
    observation_path = BASE / "runtime/continuation_observation.json"
    observation = read(observation_path)
    assert observation["audit_run_directory"] == run.relative_to(BASE).as_posix()
    progress = read(BASE / "progress.json")
    ledger = read(BASE / "verification_progress.json")
    assert progress["status"] == "RUNNING" and ledger["driver_alive"]
    assert not ledger["frozen_reference_changes"] and not ledger["duplicate_active_training_jobs"]
    source_manifest = BASE / "frozen/source_manifest.json"
    reference_manifest = BASE / "frozen/reference_manifest.json"
    assert sha(source_manifest) == "0c2bf352f6bfd12ad106b75ae0814e09abf9af51ac75b59dbc3e6665d595f0c9"
    assert sha(reference_manifest) == "0a2e2e35819bcf403ec28fd3799b9cd727cafc10e6b4092ccfaf2fb3c2b3d32d"
    frozen = read(source_manifest)
    assert sha(BASE / "frozen/design.json") == frozen["design_sha256"]
    references = read(reference_manifest)
    for row in frozen["files"] + references:
        assert sha(BASE / row["path"]) == row["sha256"], row["path"]
    initial_path = BASE / args.initial_check
    assert initial_path.resolve().is_relative_to(BASE.resolve())
    initial = read(initial_path)
    initial_processes = {row["pid"]: row for row in initial["actual_processes"]}
    processes, workers = [], []
    for process in psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time"]):
        try:
            if not (process.info.get("name") or "").lower().startswith("python"):
                continue
            command = process.info.get("cmdline") or []
            if str(BASE).lower() not in " ".join(command).lower():
                continue
            row = dict(process.info)
            cpu = process.cpu_times()
            row.update(cpu_seconds=cpu.user + cpu.system, rss_bytes=process.memory_info().rss)
            processes.append(row)
            if "closure.train" in command:
                job = read(Path(command[-1]))
                state_path = BASE / job["directory"] / "progress.json"
                state = read(state_path) if state_path.exists() else {}
                earlier = initial_processes.get(process.pid)
                delta = None
                if earlier and abs(earlier["create_time"] - row["create_time"]) < .001:
                    delta = row["cpu_seconds"] - earlier["cpu_seconds"]
                    assert delta >= 0
                workers.append(dict(pid=process.pid, created_utc=row["create_time"], job=job,
                    directory=job["directory"], saved_step=state.get("step"),
                    fixed_endpoint_reached=state.get("fixed_endpoint_reached"),
                    cpu_seconds=row["cpu_seconds"], cpu_progress_since_initial_check=delta,
                    rss_bytes=row["rss_bytes"]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    driver = [p for p in processes if p["pid"] == progress["driver_pid"]
              and any(x.endswith("driver.py") for x in p["cmdline"])]
    assert len(driver) == 1 and abs(driver[0]["create_time"] - observation["driver_created_utc"]) < .001
    assert len({w["directory"] for w in workers}) == len(workers)
    assert workers, "No live training workers; investigate before using the healthy-progress recorder"
    with (BASE / "analysis_outputs/independent_metrics_per_snapshot.csv").open(encoding="utf-8-sig", newline="") as handle:
        checked = {row["path"].replace("\\", "/") for row in csv.DictReader(handle)}
    present = {p.relative_to(BASE).as_posix() for folder in ("formal/coupled", "development")
               for p in (BASE / folder).rglob("metrics.json")}
    pending = sorted(present - checked)
    guard_path = max((BASE / "runtime/archive_integrity_checks").glob("*/verification.json"), key=lambda p: int(p.parent.name))
    guard = read(guard_path)
    assert guard["status"] == "PASS_ALL_CURRENT_PRESERVED_RECEIPT_CONTENTS"
    assert 0 <= time.time() - guard["verified_utc"] < 1800, "Run the archive verifier for this review first"
    now = time.time()
    local = datetime.fromtimestamp(now, timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    pulse = dict(observed_utc=now, observed_local=local, actual_progress=progress,
        actual_processes=processes, workers=workers, no_duplicate_active_jobs=True,
        pending_new_neural_metrics=pending, free_disk_bytes=shutil.disk_usage(BASE).free,
        available_RAM_bytes=psutil.virtual_memory().available, whole_goal_complete=False)
    classification = dict(recorded_utc=now, classification="PROGRESS", previous_turn_classification="PROGRESS",
        recorder_sha256=sha(Path(__file__)), current_audit=summary, prior_audit=previous,
        delta={k: summary[k] - previous[k] for k in ("formal", "metrics", "residuals", "inference")},
        actual_liveness=(run / "actual_liveness_after_review.json").relative_to(BASE).as_posix(),
        initial_check=args.initial_check, initial_check_sha256=sha(initial_path),
        all_archive_guard=guard_path.relative_to(BASE).as_posix(), all_archive_guard_sha256=sha(guard_path),
        source_preserved_receipts=guard["receipt_count"], archive_file_checks=guard["archive_file_checks"],
        frozen_source_files_actual_sha_checks=len(frozen["files"]), frozen_reference_files_actual_sha_checks=len(references),
        pending_new_neural_metrics=pending, application_goal_status_observed=args.goal_status,
        application_goal_updated_utc_observed=args.goal_updated_utc, real_external_blocker=False,
        whole_goal_complete=False, scientific_design_sources_budget_seeds_methods_unchanged=True,
        no_extra_training_for_accuracy=True,
        next_step="继续冻结训练与新生证据复核；扩展、45零样本、正式图实际视觉检查、最终文档及18项验收仍须完成。")
    continuation = BASE / "CONTINUE.md"
    content = continuation.read_bytes().decode("utf-8-sig").replace("\r", "")
    assert "\n## 接下来必须完成\n" in content
    # All assertions precede mutations; originals and measurement outputs are preserved.
    exclusive(run / "actual_liveness_after_review.json", pulse)
    classification["actual_liveness_sha256"] = sha(run / "actual_liveness_after_review.json")
    exclusive(classification_path, classification)
    shutil.copy2(observation_path, run / "continuation_observation_before_local_classification.json")
    shutil.copy2(continuation, run / "CONTINUE_before_local_classification.md")
    number = max(map(int, re.findall(r"(?m)^(\d+)\. ", content))) + 1
    record = (f"\n{number}. {local}实际进程复核：同一驱动器及{len(workers)}个固定队列worker正常，无重复。"
        f"本轮22阶段实际执行，独立指标{summary['metrics']}、JVP{summary['residuals']}、推理{summary['inference']}；"
        f"固定正式终点{progress['formal_complete']}/210、开发{progress['development_complete']}/24不代表整轮验收。"
        f"归档守护{guard['receipt_count']}份收据/{guard['archive_file_checks']}项SHA全部一致；"
        f"科学源{len(frozen['files'])}与参考{len(references)}文件实际SHA再次核对，冻结方案未变。"
        f"清点后{len(pending)}份新生指标仍待复核；本轮新完整五种子收据{len(summary['new_cohorts'])}份。"
        f"操作记录在{classification_path.relative_to(BASE).as_posix()}。不重启、不改种子或预算，未完成训练/验证不标完成；"
        "扩展、45零样本、正式图、最终文档及18项整体验收仍待。\n")
    content = content.replace("\n## 接下来必须完成\n", record + "\n## 接下来必须完成\n", 1)
    continuation.write_bytes(content.encode("utf-8"))
    observation.update(observed_utc=now, local_time=local,
        formal_complete=progress["formal_complete"], development_complete=progress["development_complete"],
        workers=workers, actual_task_processes=processes, new_snapshot_paths_pending_audit=pending,
        free_disk_bytes=pulse["free_disk_bytes"], available_RAM_bytes=pulse["available_RAM_bytes"],
        supplemental_all_schema_archive_integrity_record=guard_path.relative_to(BASE).as_posix(),
        supplemental_all_schema_archive_integrity_sha256=sha(guard_path),
        latest_actual_heartbeat_check=classification["actual_liveness"],
        last_goal_turn_progress_classification=classification_path.relative_to(BASE).as_posix(),
        app_goal_status_last_observed=args.goal_status, app_goal_status_updated_utc=args.goal_updated_utc,
        whole_study_complete=False, goal_complete=False, real_external_blocker_present=False,
        CONTINUE_sha256=sha(continuation))
    observation_path.write_bytes(json.dumps(observation, ensure_ascii=False, indent=2).encode("utf-8"))
    exclusive(run / "continuation_observation_after_local_classification.json", observation)
    print(json.dumps(dict(observed=local, classification="PROGRESS", formal=progress["formal_complete"],
        metrics=summary["metrics"], pending_new_metrics=len(pending), archive_receipts=guard["receipt_count"],
        workers=[dict(pid=w["pid"], group=w["job"]["group"], seed=w["job"]["seed"],
                      step=w["saved_step"], cpu_delta=w["cpu_progress_since_initial_check"]) for w in workers],
        whole_goal_complete=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
