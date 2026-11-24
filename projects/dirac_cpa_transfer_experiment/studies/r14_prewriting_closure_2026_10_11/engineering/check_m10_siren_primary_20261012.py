"""Read-only source and decision check of the first m10 SIREN primary receipt.

This operational checker does not run or alter any training or measurement.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import time

BASE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = BASE / "analysis_outputs/cohort_receipts/m10_siren_20000_first"
    receipt_path = folder / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    design_path = BASE / "frozen/design.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    assert sha(design_path) == receipt["design_sha256"]
    assert receipt["case"] == "m10" and receipt["group"] == "siren"
    assert receipt["coupled_steps"] == 20000 and receipt["endpoint"] == "primary"
    assert receipt["all_five_seeds"] == design["formal_seeds"] == [810, 811, 812, 813, 814]
    archive_hashes = {}
    for name, expected in receipt["preserved_audit_files"].items():
        actual = sha(folder / name)
        assert actual == expected, name
        archive_hashes[name] = actual
    with (folder / "independent_five_seed_rows.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    by_seed = {int(row["seed"]): row for row in rows}
    assert sorted(by_seed) == design["formal_seeds"]
    thresholds = design["thresholds"]
    verification = []
    for source in receipt["per_seed_sources"]:
        seed = source["seed"]
        row = by_seed[seed]
        path = BASE / source["metric_path"]
        assert row["path"].replace("\\", "/") == source["metric_path"]
        assert row["case"] == "m10" and row["group"] == "siren"
        assert int(row["step"]) == 20000
        assert row["pass_check"] == row["raw_and_checkpoint_provenance_pass"] == "True"
        hashes = {
            "metric": sha(path),
            "checkpoint": sha(path.parent / "model_state.pt"),
            "prediction": sha(path.parent / "prediction.npz"),
        }
        assert hashes["metric"] == source["metric_sha256"] == row["metric_record_sha256"]
        assert hashes["checkpoint"] == source["checkpoint_sha256"] == row["checkpoint_sha256"]
        assert hashes["prediction"] == source["prediction_sha256"] == row["prediction_sha256"]
        native = json.loads(path.read_text(encoding="utf-8"))
        checks = {}
        values = {}
        for metric, threshold in thresholds.items():
            value = float(row[metric])
            assert math.isfinite(value)
            assert abs(value - source["independent_values"][metric]) < 1e-12
            assert abs(value - float(native[metric])) < 1e-11
            values[metric] = value
            checks[metric] = value >= threshold if metric == "fidelity_mean" else value <= threshold
        passed = all(checks.values())
        assert passed == source["seed_joint_pass"]
        verification.append({"seed": seed, "source_hashes": hashes, "values": values,
                             "threshold_checks": checks, "joint_pass": passed})
    means = {metric: statistics.fmean(float(row[metric]) for row in rows) for metric in thresholds}
    for metric, value in means.items():
        assert abs(value - receipt["independent_means"][metric]) < 1e-12
    successes = sum(item["joint_pass"] for item in verification)
    mean_checks = {metric: value >= thresholds[metric] if metric == "fidelity_mean"
                   else value <= thresholds[metric] for metric, value in means.items()}
    group_pass = all(mean_checks.values()) and successes >= design["required_joint_successes"]
    assert successes == int(receipt["summary"]["joint_pass_count"]) == 0
    assert not group_pass and receipt["status"] == "FAIL_FIXED_CASE_GROUP_ENDPOINT"
    output = BASE / "runtime/cohort_manual_checks" / str(time.time_ns())
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "PASS_SOURCE_AND_DECISION_CHECK", "verified_utc": time.time(),
              "checker_sha256": sha(Path(__file__)), "receipt_sha256": sha(receipt_path),
              "receipt_path": receipt_path.relative_to(BASE).as_posix(),
              "preserved_audit_hashes": archive_hashes, "per_seed": verification,
              "means": means, "mean_threshold_checks": mean_checks,
              "joint_successes": successes, "scientific_group_pass": group_pass,
              "scientific_status": receipt["status"], "endpoint_steps": 20000,
              "prescribed_training_budget": 26000,
              "primary_endpoint_is_not_training_completion": True,
              "no_training_or_measurement_writes": True, "whole_goal_complete": False}
    with (output / "verification.json").open("x", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"record": str(output / "verification.json"), "status": record["status"],
                      "scientific_status": record["scientific_status"], "joint_successes": successes,
                      "means": means}, ensure_ascii=False))


if __name__ == "__main__":
    main()
