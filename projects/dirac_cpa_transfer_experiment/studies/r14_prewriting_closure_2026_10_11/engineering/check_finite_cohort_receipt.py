"""Read-only checks of a full finite formal cohort against immutable raw sources.

This checks existing independent FFT results and the frozen decision rule. It
does not train, benchmark, alter scientific sources, or accept the whole study.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics
import time

BASE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(receipt_path):
    receipt_path = receipt_path.resolve()
    receipt_path.relative_to(BASE)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    design_path = BASE / "frozen/design.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    assert sha(design_path) == receipt["design_sha256"]
    seeds = design["formal_seeds"]
    assert receipt["all_five_seeds"] == seeds and len(seeds) == 5
    case, group, step = receipt["case"], receipt["group"], receipt["coupled_steps"]
    archive_hashes = {}
    for name, expected in receipt["preserved_audit_files"].items():
        actual = sha(receipt_path.parent / name)
        assert actual == expected, name
        archive_hashes[name] = actual
    with (receipt_path.parent / "independent_five_seed_rows.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    by_seed = {int(row["seed"]): row for row in rows}
    assert sorted(by_seed) == seeds
    verified = []
    budgets = set()
    for source in receipt["per_seed_sources"]:
        seed = source["seed"]
        row = by_seed[seed]
        metric_path = BASE / source["metric_path"]
        assert source["metric_path"].startswith("formal/coupled/")
        assert row["path"].replace("\\", "/") == source["metric_path"]
        assert (row["case"], row["group"], int(row["step"])) == (case, group, step)
        assert row["pass_check"] == row["raw_and_checkpoint_provenance_pass"] == "True"
        hashes = {
            "metric": sha(metric_path),
            "checkpoint": sha(metric_path.parent / "model_state.pt"),
            "prediction": sha(metric_path.parent / "prediction.npz"),
        }
        assert hashes["metric"] == source["metric_sha256"] == row["metric_record_sha256"]
        assert hashes["checkpoint"] == source["checkpoint_sha256"] == row["checkpoint_sha256"]
        assert hashes["prediction"] == source["prediction_sha256"] == row["prediction_sha256"]
        native = json.loads(metric_path.read_text(encoding="utf-8"))
        job = native["job"]
        assert (job["case"], job["group"], job["seed"]) == (case, group, seed)
        assert native["step"] == step and step in job["snapshots"]
        budgets.add(job["budget"])
        values, checks = {}, {}
        for key, threshold in design["thresholds"].items():
            value = float(row[key])
            assert math.isfinite(value), "Finite-cohort checker cannot drop nonfinite seeds"
            assert abs(value - source["independent_values"][key]) < 1e-12
            assert abs(value - float(native[key])) < 1e-11
            values[key] = value
            checks[key] = value >= threshold if key == "fidelity_mean" else value <= threshold
        passed = all(checks.values())
        assert passed == source["seed_joint_pass"] == native["seed_joint_pass"]
        verified.append({"seed": seed, "source_hashes": hashes, "values": values,
                         "threshold_checks": checks, "joint_pass": passed})
    assert len(verified) == 5 and sorted(item["seed"] for item in verified) == seeds
    assert len(budgets) == 1
    budget = budgets.pop()
    means = {key: statistics.fmean(item["values"][key] for item in verified)
             for key in design["thresholds"]}
    for key, value in means.items():
        assert abs(value - receipt["independent_means"][key]) < 1e-12
    mean_checks = {key: value >= design["thresholds"][key] if key == "fidelity_mean"
                   else value <= design["thresholds"][key] for key, value in means.items()}
    successes = sum(item["joint_pass"] for item in verified)
    group_pass = all(mean_checks.values()) and successes >= design["required_joint_successes"]
    expected_status = ("PASS" if group_pass else "FAIL") + "_FIXED_CASE_GROUP_ENDPOINT"
    assert receipt["status"] == expected_status
    assert int(receipt["summary"]["joint_pass_count"]) == successes
    terminal_rows = []
    completion_path = BASE / "analysis_outputs/prescribed_attempt_status.csv"
    if step == budget:
        with completion_path.open(encoding="utf-8-sig", newline="") as stream:
            terminal_rows = [row for row in csv.DictReader(stream)
                             if row["case"] == case and row["group"] == group
                             and row["stage"] in ("core", "extension")]
        assert len(terminal_rows) == 5
        assert sorted(int(row["seed"]) for row in terminal_rows) == seeds
        assert all(row["state"] == "COMPLETE_VERIFIED"
                   and row["terminal_artifacts_verified"] == "True"
                   and int(row["budget"]) == budget and not row["issues"] for row in terminal_rows)
    return {"status": "PASS_SOURCE_AND_FROZEN_DECISION_CHECK", "verified_utc": time.time(),
            "checker_sha256": sha(Path(__file__)), "receipt_path": receipt_path.relative_to(BASE).as_posix(),
            "receipt_sha256": sha(receipt_path), "case": case, "group": group, "step": step,
            "endpoint_label_from_receipt": receipt["endpoint"], "prescribed_training_budget": budget,
            "this_is_prescribed_training_terminal": step == budget,
            "terminal_attempt_rows": terminal_rows,
            "completion_table_sha256": sha(completion_path) if terminal_rows else None,
            "preserved_audit_hashes": archive_hashes, "per_seed": verified,
            "means": means, "mean_threshold_checks": mean_checks, "joint_successes": successes,
            "scientific_status": expected_status, "finite_five_seed_scope_only": True,
            "no_training_or_measurement_writes": True, "whole_goal_complete": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", help="Existing first receipt, relative to the study or absolute")
    args = parser.parse_args()
    path = Path(args.receipt)
    record = check(path if path.is_absolute() else BASE / path)
    folder = BASE / "runtime/cohort_manual_checks" / str(time.time_ns())
    folder.mkdir(parents=True, exist_ok=False)
    output = folder / "verification.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"record": str(output), "status": record["status"],
                      "scientific_status": record["scientific_status"],
                      "joint_successes": record["joint_successes"], "means": record["means"],
                      "training_terminal": record["this_is_prescribed_training_terminal"],
                      "whole_goal_complete": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
