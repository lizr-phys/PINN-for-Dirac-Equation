from __future__ import annotations
import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .physics import ROOT, read_json, save_json


def verify_preservation():
    path = ROOT/"results"/"preservation_before.json"
    if not path.exists():
        raise RuntimeError("Original project preservation baseline is missing")
    original = read_json(path)
    project_root = ROOT.parent
    changed, missing = [], []
    total = 0
    start = time.perf_counter()
    for name, before in original.items():
        p = project_root/name
        if not p.exists():
            missing.append(name)
            continue
        now = hashlib.sha256(p.read_bytes()).hexdigest()
        total += before["bytes"]
        if now!=before["sha256"]:
            changed.append(dict(path=name, before=before["sha256"], after=now))
    result = dict(verified=not changed and not missing, checked_files=len(original),
                  checked_bytes=total, changed=changed, missing=missing,
                  seconds=time.perf_counter()-start,
                  statement="All pre-existing non-cache files in the four projects; read-only full SHA256 comparison.")
    save_json(ROOT/"results"/"preservation_after.json", result)
    return result


def freeze_sources():
    out = ROOT/"frozen_source"/"formal_v1"
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    for p in ROOT.glob("*.py"):
        target = out/p.name
        target.write_bytes(p.read_bytes())
        records[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    target = out/"config.json"
    target.write_bytes((ROOT/"config.json").read_bytes())
    records["config.json"] = hashlib.sha256(target.read_bytes()).hexdigest()
    save_json(ROOT/"frozen_sources.json", records)
    return records


def artifact_hashes():
    result = {}
    for p in ROOT.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts or ".pytest_cache" in p.parts:
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel=="results/artifact_sha256.json":
            continue
        result[rel] = dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    save_json(ROOT/"results"/"artifact_sha256.json", result)
    return result


def audit(preservation=True):
    fixed = pd.read_csv(ROOT/"results"/"formal_v1_summary.csv")
    baseline = pd.read_csv(ROOT/"results"/"mlp_baseline_summary.csv")
    conditional = pd.read_csv(ROOT/"results"/"conditional_v1_summary.csv")
    retrain = pd.read_csv(ROOT/"results"/"retraining_scan_v1_summary.csv")
    runs = []
    required_main = ("config.json", "environment.json", "model.npz", "training.json",
                     "training_points.npz", "history.csv", "linear_diagnostics.npz")
    complete = True
    for p in (ROOT/"results").rglob("training.json"):
        directory = p.parent
        cfg = read_json(directory/"config.json")
        required = ("config.json", "model.pt", "training.json", "history.csv", "training_points.npz") if "mlp" in cfg["group"] else required_main
        exists = all((directory/n).exists() for n in required)
        complete &= exists
        runs.append(dict(path=directory.relative_to(ROOT).as_posix(), complete=exists,
                         config=cfg, training=read_json(p)))
    save_json(ROOT/"campaign_manifest.json", runs)
    preserved = verify_preservation() if preservation else read_json(ROOT/"results"/"preservation_after.json")
    assessment_path = ROOT/"results"/"preservation_assessment.json"
    assessment = read_json(assessment_path) if assessment_path.exists() else {}
    known_dynamic = all(item["path"].startswith("dirac_cpa_transfer_experiment/")
                        and "/results/" in item["path"] and "r12_cross_condition_robustness/" in item["path"]
                        for item in preserved["changed"])
    live_evidence = bool(assessment.get("active_r12_process_count", 0))
    assessment.update(all_original_bytes_unchanged=preserved["verified"],
        static_files_preserved=known_dynamic and not preserved["missing"],
        changes_explained_by_preexisting_runs=not preserved["changed"] or (known_dynamic and live_evidence),
        dynamic_training_output_changes=len(preserved["changed"]),
        unchanged_original_file_count=preserved["checked_files"]-len(preserved["changed"]),
        missing_original_files=preserved["missing"])
    save_json(assessment_path, assessment)
    references = read_json(ROOT/"results"/"reference"/"verification.json")
    high_precision = read_json(ROOT/"results"/"reference"/"high_precision"/"verification.json")
    gate = read_json(ROOT/"results"/"fixed_validation_gate.json")
    # Every recorded science-source hash must be available in current or frozen
    # source, including pre-fix pilot source. No need to execute historical code.
    source_files = list(ROOT.glob("*.py"))+list((ROOT/"frozen_source").rglob("*.py"))
    available_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    unmatched = []
    for run in runs:
        hashes = run["config"].get("source_sha256", {})
        hashes = hashes.values() if isinstance(hashes, dict) else [hashes]
        if any(h not in available_hashes for h in hashes):
            unmatched.append(run["path"])
    tests_path = ROOT/"results"/"tests.txt"
    tests = tests_path.read_text(encoding="utf-8") if tests_path.exists() else ""
    checks = dict(original_static_files_preserved=assessment["static_files_preserved"],
        preexisting_dynamic_changes_explained=assessment["changes_explained_by_preexisting_runs"],
        no_original_files_missing=not preserved["missing"], all_run_files_complete=bool(complete),
        fixed_reference_verified=references["verified"], high_precision_verified=high_precision["verified"],
        fixed_gate_passed=gate["passed"], fixed_runs=len(fixed)==24,
        baseline_runs=len(baseline)==24, all_fixed_seeds_passed=bool(fixed.passed.all()),
        conditional_runs=len(conditional)==606, retraining_scan_runs=len(retrain)==202,
        all_conditional_seeds_preserved=all(conditional[conditional.kind==k].seed.nunique()==3 for k in ("dirac","kg")),
        source_hashes_available=not unmatched, tests_passed="passed" in tests and "FAILED" not in tests,
        figure_files_complete=all((ROOT/p).exists() for f in read_json(ROOT/"figures"/"manifest.json") for p in f["files"]))
    summary = dict(verified=all(checks.values()), checks=checks, unmatched_source_hashes=unmatched,
        preserved_files=preserved["checked_files"], formal_accepted=int(fixed.passed.sum()),
        all_original_bytes_unchanged=preserved["verified"], preservation_assessment=assessment,
        baseline_accepted=int(baseline.passed.sum()), conditional_accepted=int(conditional.passed.sum()),
        total_training_runs=len(runs), failures_preserved=int((~baseline.passed).sum()),
        test_log="results/tests.txt", statement="Failed scientific attempts are retained and do not cause artifact completeness to fail.")
    save_json(ROOT/"results"/"audit.json", summary)
    freeze_sources()
    artifact_hashes()
    print(f"AUDIT verified={summary['verified']} full_byte_equality={preserved['verified']} runs={len(runs)}", flush=True)
    return summary


if __name__ == "__main__":
    audit()
