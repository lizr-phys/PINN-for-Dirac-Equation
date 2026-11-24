from __future__ import annotations
import inspect
import json
import re
from pathlib import Path
from urllib.parse import unquote
import numpy as np
import pandas as pd
import torch
from .r9_config import ROOT, RAW, DOCS, FIGURES, MODELS, SEEDS, ARTIFACTS, run_directory
from .r9_models import R9Model, initial_coefficients, tensor_digest, parameter_counts
from .freeze_r9_wavepacket import verify_freeze, verify_history, selected_config
from .freeze_r7_phase_release import write_json, sha256


def recursive_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from recursive_keys(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from recursive_keys(child)


def independent_primary_metrics(dense, cfg):
    """Recompute the five fixed criteria directly from archived complex fields."""
    raw, exact = dense["spectral_predicted_raw"], dense["spectral_reference"]
    phase = np.angle(np.vdot(exact.ravel(), raw.ravel()))
    aligned = raw*np.exp(-1j*phase)
    spatial = np.exp(1j*dense["x"][:, None]*np.asarray(cfg.wave_numbers)[None, :])/np.sqrt(cfg.length)
    predicted_field = np.einsum("xk,tkc->txc", spatial, aligned)
    reference_field = np.einsum("xk,tkc->txc", spatial, exact)
    predicted_current = 2*np.real(predicted_field[..., 0].conj()*predicted_field[..., 1])
    reference_current = 2*np.real(reference_field[..., 0].conj()*reference_field[..., 1])
    norm = np.sum(np.abs(raw)**2, axis=(1, 2))
    reference_norm = np.sum(np.abs(exact)**2, axis=(1, 2))
    overlap = np.sum(exact.conj()*raw, axis=(1, 2))
    population = np.sum(np.abs(raw)**2, axis=-1)
    reference_population = np.sum(np.abs(exact)**2, axis=-1)
    values = dict(
        spinor_rel_l2_gauge_aligned=float(np.linalg.norm(aligned-exact)/np.linalg.norm(exact)),
        current_rel_l2=float(np.linalg.norm(predicted_current-reference_current)/np.linalg.norm(reference_current)),
        fidelity_mean=float(np.mean(np.abs(overlap)**2/np.maximum(norm*reference_norm, 1e-300))),
        modal_population_rel_l2=float(np.linalg.norm(population-reference_population)/np.linalg.norm(reference_population)),
        max_norm_drift=float(np.max(np.abs(norm-1))))
    evidence = dict(single_constant_phase=float(phase),
        aligned_spectrum_max_abs_difference=float(np.max(np.abs(aligned-dense["spectral_predicted_aligned"]))),
        predicted_physical_field_max_abs_difference=float(np.max(np.abs(predicted_field-dense["psi_predicted_aligned"]))),
        reference_physical_field_max_abs_difference=float(np.max(np.abs(reference_field-dense["psi_reference"]))),
        total_modal_occupation_max_abs_difference=float(np.max(np.abs(population-dense["modal_occupations_predicted"]))))
    return values, evidence


def link_check():
    files = [ROOT/"R9_WAVEPACKET_PERIODIC_POTENTIAL_REPORT.md"]+list(DOCS.glob("*.md"))+[FIGURES/"FIGURE_QA.md"]
    missing, checked = [], 0
    for file in files:
        if not file.exists():
            missing.append(dict(source=str(file), target="missing document"))
            continue
        content = file.read_text(encoding="utf-8")
        for target in re.findall(r"\]\((<[^>]+>|[^)]+)\)", content):
            target = unquote(target.strip("<>"))
            if target.startswith(("http:", "https:", "#")):
                continue
            target = target.split("#")[0]
            target = re.sub(r":\d+$", "", target)
            path = Path(target)
            if not path.is_absolute():
                path = file.parent/path
            checked += 1
            if not path.exists():
                missing.append(dict(source=str(file.relative_to(ROOT)), target=target))
    result = dict(status="PASS" if not missing else "FAIL", links_checked=checked, missing=missing,
                  scope="all R9 report, preregistration, theory, figure QA and experiment documents")
    write_json(RAW/"link_check.json", result)
    return result


def audit(*, require_pytest=True, history=True, require_visual=True):
    cfg = selected_config()
    # The report links to these outputs; create explicit pending placeholders
    # before checking links, then replace them with the complete audit below.
    for name in ("audit.json", "artifact_sha256.json", "link_check.json"):
        if not (RAW/name).exists():
            write_json(RAW/name, dict(status="IN_PROGRESS", placeholder=True))
    checks, runs, failures, file_hashes = [], [], [], {}
    def check(label, passed, detail=None):
        checks.append(dict(check=label, pass_check=bool(passed), detail=detail))
        if not passed:
            failures.append(label)
    try:
        manifest = verify_freeze()
        check("frozen_scientific_code_sha256", True, manifest)
    except Exception as error:
        check("frozen_scientific_code_sha256", False, str(error))
    if history:
        old = verify_history()
        check("historical_files_unchanged", old["pass_check"], old)
    reference = json.loads((RAW/"reference_validation.json").read_text(encoding="utf-8"))
    check("reference_prescribed_selection", reference["status"] == "PASS" and reference["selected_truncation"] == [-48, 80]
          and len(reference["comparisons"]) == 2 and not reference["comparisons"][0]["pass_check"]
          and reference["comparisons"][1]["pass_check"])
    import src.r9_models as models
    import src.r9_loss as loss
    import src.r9_time_derivative as tangent
    for module in (models, loss, tangent):
        text = inspect.getsource(module)
        check("no_frequency_or_reference_in_"+module.__name__,
              all(token not in text for token in ("np.linalg.eigh", "matrix_exp", "r9_reference", "r9_evaluate", "sqrt(k**2", "sqrt(k*k")))
    plan = json.loads((RAW/"runtime_plan.json").read_text(encoding="utf-8"))
    masks = [slot["mask"] for slot in plan["slots"]]
    check("runtime_complete", plan["status"] == "COMPLETE" and not plan["failures"])
    check("all_training_cpu_slots_disjoint", len(set(masks)) == 5 and not any(a & b for i, a in enumerate(masks) for b in masks[i+1:]))
    expected_masks = dict(zip(SEEDS, masks))
    states = {}
    metrics_rows = []
    for model in MODELS:
        for seed in SEEDS:
            directory = run_directory(model, seed)
            complete = all((directory/p).is_file() for p in ARTIFACTS+("recovery_state.pt", "residual_integral.csv", "residual_by_mode.csv"))
            check(f"six_artifacts_and_recovery_{model}_{seed}", complete)
            if not complete:
                continue
            m = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
            config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
            checkpoint = torch.load(directory/"model_state.pt", map_location="cpu", weights_only=False)
            recovery = torch.load(directory/"recovery_state.pt", map_location="cpu", weights_only=False)
            pre = 6000 if model == MODELS[0] else 0
            check(f"frozen_configuration_{model}_{seed}", config["configuration"] == json.loads(json.dumps(cfg.as_dict())) and not config["smoke"])
            check(f"budget_and_final_only_{model}_{seed}", checkpoint["steps"] == recovery["step"] == m["fixed_final_step"] == m["steps"] == 20000
                  and checkpoint["final_checkpoint_only"] and m["carrier_steps"] == pre and m["coupled_steps"] == 20000-pre)
            same = checkpoint["state_dict"].keys() == recovery["state_dict"].keys() and all(
                torch.equal(v, recovery["state_dict"][key]) for key, v in checkpoint["state_dict"].items())
            check(f"checkpoint_recovery_identical_{model}_{seed}", same)
            forbidden = [key for key in recursive_keys(checkpoint) if any(word in key.lower() for word in ("frequency", "eigenvalue", "reference", "target", "energy"))]
            check(f"checkpoint_no_exact_frequency_or_reference_{model}_{seed}", not forbidden, forbidden)
            check(f"checkpoint_double_precision_finite_{model}_{seed}", all(v.dtype in (torch.float64, torch.complex128)
                  and torch.isfinite(v).all() for v in checkpoint["state_dict"].values()))
            net = R9Model(cfg, model, seed)
            net.load_state_dict(checkpoint["state_dict"])
            if net.carrier:
                net.carrier.freeze()
            check(f"parameter_inventory_{model}_{seed}", all(m[key] == value for key, value in parameter_counts(net).items()))
            if seed == 300:
                # Independent derivative validation of the actual final learned state.
                from .r9_time_derivative import spinor_value_and_time_derivative
                from .r8_loss import derivative
                k_test = torch.tensor(cfg.wave_numbers, dtype=torch.float64).reshape(-1, 1).repeat(3, 1)
                t_test = torch.tensor([0., 1.234, 5.], dtype=torch.float64).repeat_interleave(cfg.mode_count).reshape(-1, 1).requires_grad_()
                slow_state = net(k_test, t_test)
                slow_dt = derivative(slow_state, t_test)
                fast_state, fast_dt = spinor_value_and_time_derivative(net, cfg, k_test, t_test)
                dt_error = float((fast_dt-slow_dt).detach().abs().max())
                state_error = float((fast_state-slow_state).detach().abs().max())
                check(f"final_checkpoint_chain_rule_vs_autograd_{model}", dt_error < 1e-11 and state_error < 1e-13,
                      dict(time_derivative_max_abs_error=dt_error, state_max_abs_error=state_error, times=[0., 1.234, 5.]))
            check(f"hard_initial_data_unchanged_{model}_{seed}", torch.equal(checkpoint["state_dict"]["initial_coeff"], initial_coefficients(cfg)))
            actual = config["cpu_affinity_mask"]
            check(f"actual_affinity_{model}_{seed}", int(actual) == expected_masks[seed] and int(m["cpu_affinity_mask"]) == expected_masks[seed]
                  and config["environment"]["torch_num_threads"] == 1)
            work = m["training_coupled_spinor_residual_points"]+m["training_carrier_unit_spinor_residual_points"]
            check(f"residual_work_counts_{model}_{seed}", work == (20000+pre)*32*129
                  and m["training_complex_scalar_residual_evaluations"] == 2*work and m["training_residual_batches"] == 20000
                  and m["evaluation_coupled_spinor_residual_points"] == 4001*129
                  and m["evaluation_carrier_unit_spinor_residual_points"] == (2*201*129 if pre else 0))
            dense = np.load(directory/"dense_grid.npz")
            required = ["residual_complex_unweighted", "exterior_residual_complex", "phase_reference_active_mask",
                        "coherence_reference_active_channel_mask", "phase_predicted_defined_mask", "raw_error_fine", "residual_error_estimate"]
            check(f"dense_physical_grid_and_diagnostics_{model}_{seed}", all(key in dense for key in required)
                  and len(dense["x"]) == 1024 and len(dense["t"]) == 201 and dense["residual_complex_unweighted"].shape == (4001, 129, 2)
                  and dense["exterior_residual_complex"].shape == (4001, 16, 2))
            ref_mask = dense["branch_occupations_reference"] >= 1e-6
            check(f"saved_reference_coherence_mask_{model}_{seed}", np.array_equal(dense["phase_reference_active_mask"], ref_mask)
                  and np.array_equal(dense["coherence_reference_active_channel_mask"], ref_mask.reshape(201, -1)))
            check(f"quadrature_unaliased_{model}_{seed}", m["fourier_integral_no_aliasing"]
                  and m["fourier_norm_parseval_max_error"] < 1e-11
                  and m["density_quadrature_double_grid_difference"] < 1e-11
                  and m["current_quadrature_double_grid_difference"] < 1e-11)
            independent, independent_evidence = independent_primary_metrics(dense, cfg)
            metric_differences = {key: abs(value-m[key]) for key, value in independent.items()}
            check(f"independent_five_primary_metrics_{model}_{seed}",
                  max(metric_differences.values()) < 1e-10
                  and max(value for key, value in independent_evidence.items()
                          if key.endswith("difference")) < 1e-11,
                  dict(recomputed=independent, saved_metric_absolute_differences=metric_differences,
                       evidence=independent_evidence))
            check(f"no_false_continuous_certificate_{model}_{seed}", m["residual_numerical_estimate_is_continuous_certificate"] is False)
            supplement_path = directory/"full_physical_residual_archival_record.json"
            check(f"signed_full_residual_supplement_exists_{model}_{seed}", supplement_path.is_file())
            if supplement_path.is_file():
                archival = json.loads(supplement_path.read_text(encoding="utf-8"))
                check(f"supplement_preserves_original_results_{model}_{seed}",
                      archival["status"] == "PASS" and archival["added_optimizer_steps"] == 0
                      and archival["primary_metrics_changed"] is False
                      and archival["supplement_sha256"] == sha256(directory/"full_physical_residual.npz")
                      and archival["supplement_code_sha256"] == sha256(ROOT/"src/r9_full_residual_archive.py")
                      and archival["archived_source_vs_final_prediction_max_abs_error"] < 1e-12
                      and archival["full_residual_norm_vs_original_max_abs_error"] < 1e-12
                      and archival["additional_coupled_residual_calls"] == 0
                      and all(sha256(directory/name) == digest
                              for name, digest in archival["original_files_unchanged"].items()))
                with np.load(directory/"full_physical_residual.npz") as full:
                    check(f"full_residual_exterior_sign_{model}_{seed}",
                          full["residual_complex_unweighted"].shape == (4001, 145, 2)
                          and np.array_equal(full["n"], np.arange(-56, 89))
                          and np.array_equal(full["residual_complex_unweighted"][:, 8:-8],
                                             dense["residual_complex_unweighted"])
                          and np.array_equal(full["exterior_residual_complex"],
                                             -dense["exterior_residual_complex"]))
            if pre:
                pretrained = torch.load(directory/"pretrained_carrier_state.pt", weights_only=False)
                carrier_state = {key.removeprefix("carrier."): value for key, value in checkpoint["state_dict"].items() if key.startswith("carrier.")}
                check(f"carrier_frozen_after_pretraining_{seed}", pretrained["steps"] == 6000 and all(
                    torch.equal(value, carrier_state[key]) for key, value in pretrained["state_dict"].items()))
                check(f"carrier_defect_and_slopes_saved_{seed}", "carrier_unit_spinor_defect" in dense and "learned_carrier_phase_slope" in dense)
            states[(model, seed)] = dict(time=recovery["coupled_sampler"]["trace"].numpy(),
                initial_sha=config["initial_complex_network_sha256"], free_time=recovery["free_sampler"]["trace"].numpy(),
                final_model_sha=tensor_digest(checkpoint["state_dict"].values()))
            metrics_rows.append(m)
            runs.append(dict(model=model, seed=seed, final_model_tensor_sha256=states[(model, seed)]["final_model_sha"],
                             cpu_mask=int(actual), no_best_checkpoint=True, complete=complete))
            for p in directory.rglob("*"):
                if p.is_file():
                    file_hashes[p.relative_to(ROOT).as_posix()] = sha256(p)
            dense.close()
    for seed in SEEDS:
        a, b, c = [states.get((model, seed)) for model in MODELS]
        if all(entry is not None for entry in (a, b, c)):
            check(f"paired_coupled_time_prefix_{seed}", a["time"].shape == (14000, 32, 1) and b["time"].shape == c["time"].shape == (20000, 32, 1)
                  and np.array_equal(a["time"], b["time"][:14000]) and np.array_equal(b["time"], c["time"]))
            check(f"paired_same_shape_network_weights_{seed}", a["initial_sha"][0] == b["initial_sha"][0] == c["initial_sha"][0]
                  and c["initial_sha"][0] != c["initial_sha"][1])
            check(f"pretrain_sampling_separate_{seed}", a["free_time"].shape == (6000, 32, 1)
                  and not np.array_equal(a["free_time"], a["time"][:6000]))
    check("all_15_formal_runs", len(runs) == 15)
    decision = json.loads((RAW/"decision.json").read_text(encoding="utf-8"))
    from .r9_decision import absolute_success
    frame = pd.DataFrame(metrics_rows)
    for name in MODELS:
        if len(frame):
            recalculated = absolute_success(frame[frame.model == name])
            check("decision_recalculated_"+name, recalculated == decision["absolute_success"][name])
    qa = json.loads((FIGURES/"FIGURE_QA.json").read_text(encoding="utf-8"))
    check("figure_exports_source_and_layout_QA", qa["status"] == "PASS" and len(qa["figures"]) == 9)
    if require_visual:
        check("figure_visual_QA", qa.get("visual_inspection_pending") is False and bool(qa.get("visual_inspection_record")))
    for figure in qa["figures"]:
        expected = [300] if "seed300" in figure["figure"] else ([] if figure["figure"] == "reference_calibration" else list(SEEDS))
        check("figure_seed_coverage_"+figure["figure"], figure["seeds"] == expected)
        check("figure_source_csv_"+figure["figure"], all((FIGURES/p).is_file() for p in figure["source_csv"]))
    if require_pytest:
        test = json.loads((RAW/"pytest_final_result.json").read_text(encoding="utf-8"))
        check("full_pytest_final_pass", test["status"] == "PASS", test)
    links = link_check()
    check("R9_report_links", links["status"] == "PASS", links)
    for folder in (DOCS, FIGURES):
        for p in folder.rglob("*"):
            if p.is_file():
                file_hashes[p.relative_to(ROOT).as_posix()] = sha256(p)
    # Cover reference archives, random-state evidence, smoke, tests and logs too.
    for folder in (RAW, ROOT/"results/logs/r9_wavepacket_periodic_potential"):
        for p in folder.rglob("*"):
            if p.is_file() and p.name not in ("artifact_sha256.json", "audit.json"):
                file_hashes[p.relative_to(ROOT).as_posix()] = sha256(p)
    report = ROOT/"R9_WAVEPACKET_PERIODIC_POTENTIAL_REPORT.md"
    file_hashes[report.relative_to(ROOT).as_posix()] = sha256(report)
    write_json(RAW/"artifact_sha256.json", dict(files=file_hashes))
    result = dict(status="PASS" if not failures else "FAIL", checks=checks, failed_checks=failures, runs=runs,
                  engineering_integrity_separate_from_scientific_success=True, preserved_formal_runs=len(runs),
                  history_checked=history, visual_checked=require_visual, full_pytest_required=require_pytest,
                  artifact_manifest_sha256=sha256(RAW/"artifact_sha256.json"))
    write_json(RAW/"audit.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
