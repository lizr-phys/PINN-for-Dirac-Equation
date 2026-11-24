from __future__ import annotations
import inspect
import json
import re
from pathlib import Path
from urllib.parse import unquote
import numpy as np
import pandas as pd
import torch
from .r10_config import ROOT, RAW, DOCS, FIGURES, LOGS, SEEDS, BASE_MODEL, run_directory, source_directory
from .r9_config import ARTIFACTS, RAW as R9_RAW
from .r9_models import R9Model, tensor_digest, parameter_counts, initial_coefficients
from .r9_decision import absolute_success
from .r10_norm_diagnostics import norm_balance, carrier_envelope_decomposition
from .freeze_r9_wavepacket import selected_config
from .freeze_r10_budget import verify_freeze, verify_prior_results
from .freeze_r7_phase_release import write_json, sha256
from .audit_r9_wavepacket import independent_primary_metrics, recursive_keys
from .aggregate_r10_budget import STAGES


def link_check():
    files = [ROOT/"R10_WAVEPACKET_BUDGET_DIAGNOSIS_REPORT.md", ROOT/"NEXT_STEP_PLAN_AFTER_R9.md"]
    files += list(DOCS.glob("*.md"))+[FIGURES/"FIGURE_QA.md"]
    missing, count = [], 0
    for file in files:
        if not file.is_file():
            missing.append(dict(source=str(file), target="missing document"))
            continue
        for target in re.findall(r"\]\((<[^>]+>|[^)]+)\)", file.read_text(encoding="utf-8")):
            target = unquote(target.strip("<>"))
            if target.startswith(("http:", "https:", "#")):
                continue
            target = re.sub(r":\d+$", "", target.split("#")[0])
            path = Path(target)
            if not path.is_absolute():
                path = file.parent/path
            count += 1
            if not path.exists():
                missing.append(dict(source=file.relative_to(ROOT).as_posix(), target=target))
    result = dict(status="PASS" if not missing else "FAIL", links_checked=count, missing=missing)
    write_json(RAW/"link_check.json", result)
    return result


def audit():
    cfg = selected_config()
    for name in ("audit.json", "artifact_sha256.json", "link_check.json"):
        if not (RAW/name).exists():
            write_json(RAW/name, dict(status="IN_PROGRESS", placeholder=True))
    checks, failed, runs = [], [], []
    def check(label, passed, detail=None):
        checks.append(dict(check=label, pass_check=bool(passed), detail=detail))
        if not passed:
            failed.append(label)
    try:
        frozen = verify_freeze()
        check("frozen_scientific_code_unchanged", True, frozen)
    except Exception as error:
        check("frozen_scientific_code_unchanged", False, str(error))
    prior = verify_prior_results()
    check("all_r1_r9_sources_unchanged", prior["status"] == "PASS", prior)
    source_decision = json.loads((R9_RAW/"decision.json").read_text(encoding="utf-8"))
    check("r9_original_failed_classification_preserved", source_decision["status"] == "FULL_DYNAMICS_NOT_ESTABLISHED"
        and source_decision["absolute_success"][BASE_MODEL]["joint_pass_count"] == 3)
    plan = json.loads((RAW/"runtime_plan.json").read_text(encoding="utf-8"))
    masks = [s["mask"] for s in plan["slots"]]
    check("formal_runtime_complete", plan["status"] == "COMPLETE" and not plan["failures"])
    check("five_disjoint_training_physical_cores", len(masks) == 5 and not any(
        a & b for i, a in enumerate(masks) for b in masks[i+1:]))
    for module_name in ("r9_models", "r9_loss", "r9_time_derivative"):
        import importlib
        text = inspect.getsource(importlib.import_module("src."+module_name))
        check("no_exact_frequency_or_reference_in_"+module_name,
            all(s not in text for s in ("np.linalg.eigh", "matrix_exp", "r9_reference", "r9_evaluate", "sqrt(k**2", "sqrt(k*k")))
    from . import train_r10_budget
    training_text = inspect.getsource(train_r10_budget.train_continuation)
    check("evaluation_only_after_fixed_checkpoint", training_text.index("fixed_final_checkpoint_saved") <
          training_text.index("from .r10_evaluate"))
    for seed, mask in zip(SEEDS, masks):
        directory, old_directory = run_directory(seed), source_directory(seed)
        required = ARTIFACTS+("recovery_state.pt", "execution_events.jsonl", "residual_by_mode.csv", "residual_integral.csv",
            "carrier_diagnostics.csv", "norm_balance.csv", "residual_decomposition.csv")
        complete = all((directory/name).is_file() for name in required)
        check(f"all_artifacts_seed{seed}", complete)
        if not complete:
            continue
        config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
        metrics = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
        cp = torch.load(directory/"model_state.pt", weights_only=False, map_location="cpu")
        recovery = torch.load(directory/"recovery_state.pt", weights_only=False, map_location="cpu")
        old = torch.load(old_directory/"recovery_state.pt", weights_only=False, map_location="cpu")
        control = torch.load(R9_RAW/"formal/C_double_complex"/f"seed_{seed}"/"recovery_state.pt", weights_only=False, map_location="cpu")
        check(f"fixed26000_no_reset_seed{seed}", cp["steps"] == recovery["global_step"] == metrics["fixed_final_step"] == 26000
            and recovery["additional_step"] == metrics["additional_optimizer_steps"] == 6000
            and metrics["carrier_steps"] == 6000 and metrics["coupled_steps"] == 20000 and metrics["new_carrier_steps"] == 0
            and config["optimizer_reset"] is False and config["sampler_reset"] is False
            and config["loss_terms"] == ["original_full_coupled_spinor_pde"] and config["fresh_independent_seed"] is False
            and not config["smoke"] and cp["final_checkpoint_only"])
        check(f"same_physics_and_hard_initial_configuration_seed{seed}",
            config["source_configuration"] == json.loads(json.dumps(cfg.as_dict()))
            and cp["physical_configuration"] == cfg.as_dict())
        check(f"checkpoint_equals_recovery_seed{seed}", all(torch.equal(v, recovery["state_dict"][k]) for k, v in cp["state_dict"].items()))
        carrier_keys = [k for k in old["state_dict"] if k.startswith("carrier.")]
        check(f"carrier_bitwise_unchanged_seed{seed}", all(torch.equal(cp["state_dict"][k], old["state_dict"][k]) for k in carrier_keys)
            and tensor_digest([cp["state_dict"][k] for k in carrier_keys]) == metrics["ancestry"]["frozen_carrier_tensor_sha256"])
        check(f"source_sha_and_adam_ancestry_seed{seed}", config["ancestry"]["source_checkpoint_sha256"] == sha256(old_directory/"model_state.pt")
            and config["ancestry"]["source_recovery_sha256"] == sha256(old_directory/"recovery_state.pt")
            and config["ancestry"]["inherited_optimizer_parameter_steps"] == [14000.]
            and all(float(v["step"]) == 20000 for v in recovery["optimizer"]["state"].values()))
        time = recovery["coupled_sampler"]["trace"]
        old_time = old["coupled_sampler"]["trace"]
        control_time = control["coupled_sampler"]["trace"]
        check(f"complete_paired_time_stream_seed{seed}", time.shape == (20000, 32, 1)
            and torch.equal(time[:14000], old_time) and torch.equal(time, control_time)
            and torch.equal(recovery["coupled_sampler"]["normalized_trace"], control["coupled_sampler"]["normalized_trace"])
            and torch.equal(recovery["coupled_sampler"]["generator_state"], control["coupled_sampler"]["generator_state"]))
        check(f"free_sampler_not_extended_seed{seed}", torch.equal(recovery["free_sampler"]["trace"],
            old["free_sampler"]["trace"]) and len(recovery["free_sampler"]["trace"]) == 6000
            and torch.equal(recovery["free_sampler"]["generator_state"], old["free_sampler"]["generator_state"]))
        check(f"checkpoint_precision_finite_initial_seed{seed}",
            all(v.dtype in (torch.float64, torch.complex128) and torch.isfinite(v).all() for v in cp["state_dict"].values())
            and torch.equal(cp["state_dict"]["initial_coeff"], initial_coefficients(cfg)))
        forbidden = [k for k in recursive_keys(cp) if any(s in k.lower() for s in ("frequency", "eigenvalue", "reference", "target", "energy"))]
        check(f"no_target_frequency_checkpoint_seed{seed}", not forbidden, forbidden)
        model = R9Model(cfg, BASE_MODEL, seed)
        model.load_state_dict(cp["state_dict"])
        model.carrier.freeze()
        check(f"parameters_frozen_and_active_seed{seed}", all(metrics[k] == v for k, v in parameter_counts(model).items())
            and all(not p.requires_grad for p in model.carrier.parameters()))
        check(f"actual_os_affinity_seed{seed}", config["cpu_affinity_mask"] == metrics["cpu_affinity_mask"] == mask
            and config["environment"]["torch_num_threads"] == 1)
        check(f"work_and_cost_accounting_seed{seed}", metrics["additional_coupled_spinor_residual_points"] == 24768000
            and metrics["additional_complex_scalar_residual_evaluations"] == 49536000
            and metrics["cumulative_training_spinor_residual_points"] == 132096000
            and abs(metrics["cumulative_training_seconds"]-metrics["inherited_training_seconds"]-metrics["additional_training_seconds"]) < 1e-8
            and metrics["norm_diagnostic_additional_residual_calls"] == 0)
        with np.load(directory/"dense_grid.npz") as dense:
            values, evidence = independent_primary_metrics(dense, cfg)
            check(f"five_primary_metrics_recomputed_seed{seed}", all(abs(values[k]-metrics[k]) < 5e-12 for k in values), values)
            check(f"one_constant_phase_and_physical_archive_seed{seed}", all(abs(v) < 5e-12 for k, v in evidence.items() if k != "single_constant_phase"), evidence)
            check(f"fixed_unaliased_test_grid_seed{seed}", dense["x"].shape == (1024,) and dense["t"].shape == (201,)
                and metrics["fourier_integral_no_aliasing"])
            active = dense["branch_occupations_reference"] >= 1e-6
            check(f"reference_active_masks_saved_seed{seed}", np.array_equal(active, dense["phase_reference_active_mask"])
                and np.array_equal(active.reshape(201, -1), dense["coherence_reference_active_channel_mask"]))
            signed = dense["exterior_residual_complex"]
            positive = dense["exterior_positive_coupling_source_complex"]
            expected = .1*np.concatenate((dense["spectral_prediction_raw_fine"][:, :8],
                                          dense["spectral_prediction_raw_fine"][:, -8:]), axis=1)
            check(f"signed_exterior_full_residual_seed{seed}", np.allclose(signed, -expected, rtol=0, atol=2e-13)
                and np.array_equal(signed, -positive)
                and dense["full_physical_residual_complex"].shape == (4001, 145, 2)
                and np.array_equal(dense["full_physical_residual_complex"][:, 8:-8], dense["residual_complex_unweighted"])
                and np.array_equal(dense["full_physical_residual_complex"][:, :8], signed[:, :8])
                and np.array_equal(dense["full_physical_residual_complex"][:, -8:], signed[:, 8:]))
            balance, btable = norm_balance(cfg, dense["residual_t"], dense["spectral_prediction_raw_fine"], dense["residual_complex_unweighted"])
            components, ctable = carrier_envelope_decomposition(cfg, dense["t"], dense["spectral_predicted_raw"],
                dense["residual_complex_unweighted"][::20], dense["learned_carrier"], dense["carrier_unit_spinor_defect"])
            check(f"norm_diagnostics_independently_recomputed_seed{seed}", all(
                metrics[k] == v if isinstance(v, bool) else abs(metrics[k]-v) < 2e-13 for k, v in (balance|components).items()))
            stored = pd.read_csv(directory/"norm_balance.csv")
            check(f"norm_balance_table_seed{seed}", np.allclose(stored.to_numpy(), btable.to_numpy(), rtol=1e-10, atol=1e-14))
            check(f"no_false_continuous_or_causal_certificate_seed{seed}", metrics["norm_balance_is_continuous_certificate"] is False
                and metrics["residual_numerical_estimate_is_continuous_certificate"] is False
                and metrics["decomposition_not_a_pretraining_causal_ablation"] is True)
            check(f"unit_carrier_and_hermitian_global_projection_seed{seed}",
                components["carrier_unit_modulus_max_deviation"] < 1e-12
                and components["carrier_defect_direct_norm_projection_max_abs"] < 1e-12
                and components["hermitian_coupling_global_norm_projection_max_abs"] < 1e-12
                and components["envelope_vs_complete_norm_projection_max_difference"] < 1e-12)
        runs.append(dict(seed=seed, fixed_final_step=26000, carrier_unchanged=True, source_sha256=config["ancestry"]["source_checkpoint_sha256"]))
    check("all_five_ancestries_retained", len(runs) == 5)
    frame = pd.read_csv(RAW/"per_seed_metrics.csv", float_precision="round_trip")
    decision = json.loads((RAW/"decision.json").read_text(encoding="utf-8"))
    for stage in STAGES:
        getter = source_directory if stage == STAGES[0] else run_directory
        original_metrics = pd.DataFrame([json.loads((getter(seed)/"metrics.json").read_text(encoding="utf-8")) for seed in SEEDS])
        recalculated = absolute_success(original_metrics)
        check("absolute_decision_recomputed_"+stage, recalculated == decision["absolute_success"][stage],
            dict(source="original full-precision per-run metrics.json", recalculated=recalculated))
        check("CSV_decision_agrees_"+stage, absolute_success(frame[frame.stage == stage]) == recalculated,
            dict(csv_float_parser="round_trip"))
    check("scientific_scope_and_unequal_total_budget_acknowledged", decision["fresh_independent_seed_replication"] is False
        and decision["total_budget_matched_comparison_to_r9_C"] is False
        and decision["pretraining_causal_effect_isolated"] is False and decision["r9_original_classification_unchanged"])
    qa = json.loads((FIGURES/"FIGURE_QA.json").read_text(encoding="utf-8"))
    check("six_figure_layout_and_visual_QA", qa["status"] == "PASS" and len(qa["figures"]) == 6
        and qa["visual_inspection_pending"] is False and qa["visual_inspection_record"]["passed"])
    for item in qa["figures"]:
        name = item["figure"]
        expected_seeds = [300] if name == "dynamics_seed300" else list(SEEDS)
        check("figure_all_ancestries_"+name, item["seeds"] == expected_seeds)
        check("figure_exports_and_source_"+name,
            all((FIGURES/p).is_file() for p in item["source_csv"]) and all(
            (FIGURES/f"{name}.{suffix}").is_file() and sha256(FIGURES/f"{name}.{suffix}") == item["exports"][suffix]["sha256"] for suffix in ("png", "svg", "pdf")))
    tests = json.loads((RAW/"pytest_final_result.json").read_text(encoding="utf-8"))
    check("full_final_pytest_pass_no_skips", tests["status"] == "PASS" and tests["skipped"] == 0, tests)
    reporting = json.loads((RAW/"reporting_code_sha256.json").read_text(encoding="utf-8"))["files"]
    check("reporting_code_sha256", all(sha256(ROOT/p) == digest for p, digest in reporting.items()))
    links = link_check()
    check("all_r10_plan_report_and_figure_links", links["status"] == "PASS", links)
    paths = list(RAW.rglob("*"))+list(FIGURES.rglob("*"))+list(DOCS.rglob("*"))+list(LOGS.rglob("*"))
    paths += [ROOT/"R10_WAVEPACKET_BUDGET_DIAGNOSIS_REPORT.md", ROOT/"NEXT_STEP_PLAN_AFTER_R9.md"]+[ROOT/p for p in reporting]
    exclude = {RAW/"audit.json", RAW/"artifact_sha256.json"}
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(set(paths)) if p.is_file() and p not in exclude}
    write_json(RAW/"artifact_sha256.json", dict(files=hashes, self_and_audit_excluded_to_avoid_hash_cycle=True))
    result = dict(status="PASS" if not failed else "FAIL", checks=checks, failed_checks=failed, runs=runs,
        engineering_audit_is_scientific_success=False, prior_results=prior, artifact_files=len(hashes))
    write_json(RAW/"audit.json", result)
    return result


if __name__ == "__main__":
    result = audit()
    print("R10 audit:", result["status"], "failed:", result["failed_checks"])
    if result["status"] != "PASS":
        raise SystemExit(1)
