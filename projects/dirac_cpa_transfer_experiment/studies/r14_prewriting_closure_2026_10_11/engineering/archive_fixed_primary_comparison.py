"""Preserve a verified fixed primary comparison, without training or selection.

The earlier T10 archive and the partial m10 bound remain unchanged. This reader
checks saved weights, stable terminal sampling traces, independent metrics and
existing statistics before creating a new first-complete receipt.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import time

import psutil
import torch

BASE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def archive(case):
    out = BASE / "analysis_outputs"
    destination = out / "primary_comparison_receipts" / f"{case}_20000_first"
    assert not destination.exists(), "Existing first evidence must not be overwritten"
    design_path = BASE / "frozen/design.json"
    design = read(design_path)
    seeds = design["formal_seeds"]
    assert seeds == [810, 811, 812, 813, 814]
    statistics = read(out / "independent_statistical_audit.json")
    assert statistics["status"] in (
        "PASS_AVAILABLE_STATISTICAL_TABLES", "ALL_PRESCRIBED_STATISTICS_VERIFIED"
    ) and not statistics["errors"]
    for name, expected in statistics["source_table_sha256"].items():
        assert sha(out / name) == expected, name
    assessment = read(out / "learning_advantage_assessment.json")
    decision = next(row for row in assessment["cases"] if row["case"] == case)
    assert decision["all_five_paired_and_independently_audited"]
    assert decision["prescribed_primary_attempts_accounted"]
    assert decision["accuracy_advantage_qualified"] in (True, False)
    pairing = read(out / "pairing_audit.json")
    assert not pairing["mismatches"]
    independent = table(out / "independent_metrics_per_snapshot.csv")
    controls = []
    for seed in seeds:
        states, traces, sources = [], [], []
        for group in ("learned_hard", "analytic_hard"):
            folder = BASE / "formal/coupled" / case / group / f"seed_{seed}"
            assert (folder / "complete.json").is_file()
            metric_path = folder / "snapshots/20000/metrics.json"
            row = next(r for r in independent if r["case"] == case
                       and r["group"] == group and int(r["seed"]) == seed
                       and int(r["step"]) == 20000)
            assert row["pass_check"] == row["raw_and_checkpoint_provenance_pass"] == "True"
            assert sha(metric_path) == row["metric_record_sha256"]
            assert sha(metric_path.parent / "model_state.pt") == row["checkpoint_sha256"]
            assert sha(metric_path.parent / "prediction.npz") == row["prediction_sha256"]
            assert sha(BASE / "references" / case / "test201.npz") == row["reference_sha256"]
            assert sha(BASE / "references" / case / "test401.npz") == row["reference401_sha256"]
            initial = next(r for r in pairing["initializer_checks"]
                           if (r["case"], r["group"], r["seed"]) == (case, group, seed))
            trace = next(r for r in pairing["stream_replay_checks"]
                         if (r["case"], r["group"], r["seed"]) == (case, group, seed))
            assert initial["pass_check"] and trace["pass_check"]
            assert trace["actual_saved_trace_matches_independent_replay"]
            assert sha(BASE / initial["path"]) == initial["initial_file_sha256"]
            assert sha(BASE / trace["checkpoint"]) == trace["checkpoint_sha256"] == row["checkpoint_sha256"]
            assert sha(BASE / trace["terminal_recovery"]) == trace["recovery_sha256"]
            raw = torch.load(BASE / initial["path"], weights_only=False)["state_dict"]
            states.append({key: value for key, value in raw.items()
                           if not key.startswith("carrier.")})
            traces.append(trace)
            sources.append({"group": group, "independent_metric_row": row,
                            "actual_initializer": initial, "terminal_trace": trace})
        assert states[0].keys() == states[1].keys()
        assert all(torch.equal(states[0][key], states[1][key])
                   and states[0][key].dtype == states[1][key].dtype for key in states[0])
        assert traces[0]["actual_normalized_prefix_sha256"] == traces[1]["actual_normalized_prefix_sha256"]
        assert traces[0]["actual_physical_prefix_sha256"] == traces[1]["actual_physical_prefix_sha256"]
        controls.append({"seed": seed, "actual_noncarrier_weights_and_buffers_bitwise_equal": True,
                         "private_20000_time_streams_bitwise_equal": True, "sources": sources})
    pairs = [row for row in table(out / "paired_statistics.csv")
             if (row["case"], row["target"], row["baseline"], row["target_step"], row["baseline_step"])
             == (case, "learned_hard", "analytic_hard", "20000", "20000")]
    assert pairs and all(row["status"] == "ALL_FIVE_FINITE_PAIRS" and int(row["n"]) == 5
                         for row in pairs)
    spinor = next(row for row in pairs if row["metric"] == "spinor_rel_l2_gauge_aligned")
    assert math.isclose(decision["spinor_paired_difference_mean"],
                        float(spinor["paired_difference_mean"]), abs_tol=2e-14)
    assert math.isclose(decision["spinor_CI95_upper"], float(spinor["CI95_upper"]), abs_tol=2e-14)
    names = ["independent_metric_audit.json", "independent_metrics_per_snapshot.csv",
             "independent_cost_audit.json", "independent_statistical_audit.json",
             "paired_statistics.csv", "paired_per_seed.csv", "summary_metrics.csv",
             "pairing_audit.json", "learning_advantage_assessment.json",
             "algorithm_costs_with_search_overhead.csv", "residual_provenance_audit.json"]
    source_bytes = {name: (out / name).read_bytes() for name in names}
    inference_path = BASE / "isolated_inference/independent_audit.json"
    source_bytes["independent_isolated_inference_audit.json"] = inference_path.read_bytes()
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, data in source_bytes.items():
        with (destination / name).open("xb") as stream:
            stream.write(data)
        hashes[name] = sha(destination / name)
        assert hashes[name] == hashlib.sha256(data).hexdigest()
    record = {"assessed_utc": time.time(),
              "status": f"FIRST_FULL_{case}_PRIMARY_COMPARISON_INDEPENDENTLY_VERIFIED",
              "case": case, "coupled_steps": 20000, "formal_seeds": seeds,
              "assessment": decision, "paired_rows": pairs, "strict_pairing_controls": controls,
              "preserved_source_sha256": hashes, "design_sha256": sha(design_path),
              "source_manifest_sha256": sha(BASE / "frozen/source_manifest.json"),
              "archive_source": Path(__file__).relative_to(BASE).as_posix(),
              "archive_source_sha256": sha(Path(__file__)),
              "optimization_cost_includes_full_6000_auxiliary_steps": True,
              "optimization_wall_excludes_setup_evaluation_and_IO": True,
              "paired_table_inference_and_RSS_scope": "native snapshot observations; isolated observations kept separately",
              "exact_sign_flip_two_sided_p": float(spinor["exact_sign_flip_two_sided_p"]),
              "n5_t_interval_descriptive_not_general_significance": True,
              "other_case_results_not_pooled": True,
              "partial_bound_and_old_first_receipts_not_overwritten": True,
              "all_scope_evidence_pending": True, "whole_goal_complete": False}
    with (destination / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"receipt": (destination / "receipt.json").relative_to(BASE).as_posix(),
                      "strict_five_pairs_checked": len(controls), "assessment": decision,
                      "whole_goal_complete": False}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["T10", "m10"])
    args = parser.parse_args()
    psutil.Process().cpu_affinity([6])
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    archive(args.case)


if __name__ == "__main__":
    main()
