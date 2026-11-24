"""Read-only verification of the preregistered m5T5 applicability comparison."""
import itertools
import json
import math
import statistics
import time
from pathlib import Path
import psutil
import torch
from archive_fixed_primary_comparison import BASE, sha, read, table


def main():
    psutil.Process().cpu_affinity([6])
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    out = BASE / "analysis_outputs"
    destination = out / "primary_comparison_receipts/m5T5_20000_first"
    assert not (destination / "receipt.json").exists(), "Preserve an existing first receipt"
    audit = read(out / "independent_statistical_audit.json")
    assert not audit["errors"] and audit["status"] in ("PASS_AVAILABLE_STATISTICAL_TABLES", "ALL_PRESCRIBED_STATISTICS_VERIFIED")
    for name, expected in audit["source_table_sha256"].items():
        assert sha(out / name) == expected
    design = read(BASE / "frozen/design.json")
    assert design["formal_seeds"] == list(range(810, 815))
    pairing = read(out / "pairing_audit.json")
    assert not pairing["mismatches"]
    independent = table(out / "independent_metrics_per_snapshot.csv")
    controls, differences, values = [], [], []
    for seed in range(810, 815):
        states, traces, sources, metrics = [], [], [], []
        for group in ("learned_hard", "analytic_hard"):
            folder = BASE / f"formal/coupled/m5T5/{group}/seed_{seed}"
            completion = read(folder / "complete.json")
            assert completion["status"] == "COMPLETE" and completion["step"] == 20000
            row = next(r for r in independent if (r["case"], r["group"], r["seed"], r["step"]) == ("m5T5", group, str(seed), "20000"))
            assert row["pass_check"] == row["raw_and_checkpoint_provenance_pass"] == "True"
            mp = folder / "snapshots/20000/metrics.json"
            assert sha(mp) == row["metric_record_sha256"]
            assert sha(mp.parent / "model_state.pt") == row["checkpoint_sha256"]
            assert sha(mp.parent / "prediction.npz") == row["prediction_sha256"]
            assert sha(BASE / "references/m5T5/test201.npz") == row["reference_sha256"]
            initial = next(r for r in pairing["initializer_checks"] if (r["case"], r["group"], r["seed"]) == ("m5T5", group, seed))
            trace = next(r for r in pairing["stream_replay_checks"] if (r["case"], r["group"], r["seed"]) == ("m5T5", group, seed))
            assert initial["pass_check"] and trace["pass_check"] and trace["actual_saved_trace_matches_independent_replay"]
            assert sha(BASE / initial["path"]) == initial["initial_file_sha256"]
            assert sha(BASE / trace["checkpoint"]) == trace["checkpoint_sha256"] == row["checkpoint_sha256"]
            assert sha(BASE / trace["terminal_recovery"]) == trace["recovery_sha256"]
            raw = torch.load(BASE / initial["path"], weights_only=False)["state_dict"]
            cost_path = folder / "snapshots/20000/cost.json"
            cost = read(cost_path)
            if group == "learned_hard":
                config = read(folder / "config.json")
                auxiliary_path = BASE / config["job"]["source"] / "complete.json"
                auxiliary = read(auxiliary_path)
                assert auxiliary["step"] == cost["free_steps"] == 6000
                assert abs(cost["attributed_training_wall_seconds"] - cost["training_wall_seconds"] - auxiliary["training_wall_seconds"]) < 1e-7
            else:
                auxiliary_path, auxiliary = None, None
                assert cost["free_steps"] == 0 and cost["source_pretraining_wall_seconds"] == 0
            states.append({k: v for k, v in raw.items() if not k.startswith("carrier.")})
            traces.append(trace)
            sources.append(dict(group=group, metric=row, actual_initializer=initial, terminal_trace=trace,
                                native_cost=cost, cost_sha256=sha(cost_path), auxiliary_cost=auxiliary,
                                auxiliary_cost_sha256=sha(auxiliary_path) if auxiliary_path else None))
            metrics.append(float(row["spinor_rel_l2_gauge_aligned"]))
        assert states[0].keys() == states[1].keys()
        assert all(torch.equal(states[0][k], states[1][k]) and states[0][k].dtype == states[1][k].dtype for k in states[0])
        assert traces[0]["actual_normalized_prefix_sha256"] == traces[1]["actual_normalized_prefix_sha256"]
        assert traces[0]["actual_physical_prefix_sha256"] == traces[1]["actual_physical_prefix_sha256"]
        differences.append(metrics[0] - metrics[1])
        values.append(dict(seed=seed, learned=metrics[0], analytic=metrics[1], difference=metrics[0] - metrics[1]))
        controls.append(dict(seed=seed, actual_noncarrier_weights_and_buffers_bitwise_equal=True,
                             private_20000_time_streams_bitwise_equal=True, sources=sources))
    pairs = [r for r in table(out / "paired_statistics.csv") if (r["case"], r["target"], r["baseline"], r["target_step"], r["baseline_step"]) == ("m5T5", "learned_hard", "analytic_hard", "20000", "20000")]
    assert len(pairs) == 11 and all(r["status"] == "ALL_FIVE_FINITE_PAIRS" and int(r["n"]) == 5 for r in pairs)
    spinor = next(r for r in pairs if r["metric"] == "spinor_rel_l2_gauge_aligned")
    mean, sd = statistics.fmean(differences), statistics.pstdev(differences)
    half = 2.7764451051977987 * sd / math.sqrt(4)
    p = sum(abs(statistics.fmean(s * d for s, d in zip(signs, differences))) >= abs(mean) - 1e-14
            for signs in itertools.product((-1, 1), repeat=5)) / 32
    for key, actual in (("paired_difference_mean", mean), ("population_sd", sd), ("CI95_lower", mean - half),
                        ("CI95_upper", mean + half), ("exact_sign_flip_two_sided_p", p)):
        assert abs(float(spinor[key]) - actual) < 1e-12
    names = ["independent_metric_audit.json", "independent_metrics_per_snapshot.csv", "independent_cost_audit.json",
             "independent_statistical_audit.json", "paired_statistics.csv", "paired_per_seed.csv", "summary_metrics.csv",
             "pairing_audit.json", "algorithm_costs_with_search_overhead.csv", "residual_provenance_audit.json"]
    data = {n: (out / n).read_bytes() for n in names}
    data["independent_isolated_inference_audit.json"] = (BASE / "isolated_inference/independent_audit.json").read_bytes()
    if destination.exists():
        assert {p.name for p in destination.iterdir()} == set(data)
        assert all((destination / name).read_bytes() == raw for name, raw in data.items())
    else:
        destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, raw in data.items():
        if not (destination / name).exists():
            with (destination / name).open("xb") as stream:
                stream.write(raw)
        hashes[name] = sha(destination / name)
    record = dict(assessed_utc=time.time(), status="FIRST_FULL_M5T5_APPLICABILITY_COMPARISON_INDEPENDENTLY_VERIFIED",
                  case="m5T5", coupled_steps=20000, formal_seeds=list(range(810, 815)), paired_rows=pairs,
                  independent_spinor_pair_math=dict(mean=mean, population_sd=sd, CI95_lower=mean-half, CI95_upper=mean+half,
                                                    exact_sign_flip_two_sided_p=p, favorable_pairs=sum(d < 0 for d in differences)),
                  per_seed_spinor=values, strict_pairing_controls=controls, preserved_source_sha256=hashes,
                  design_sha256=sha(BASE / "frozen/design.json"), source_manifest_sha256=sha(BASE / "frozen/source_manifest.json"),
                  archive_source=Path(__file__).relative_to(BASE).as_posix(), archive_source_sha256=sha(Path(__file__)),
                  scope="Frozen applicability comparison, not an added primary learning-advantage hypothesis",
                  primary_learning_advantage_cases_remain_T10_and_m10=True,
                  n5_interval_and_sign_flip_are_descriptive=True, optimization_cost_includes_full_6000_auxiliary_steps=True,
                  other_case_results_not_pooled=True, whole_goal_complete=False)
    with (destination / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(receipt=(destination / "receipt.json").relative_to(BASE).as_posix(), math=record["independent_spinor_pair_math"], whole_goal_complete=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
