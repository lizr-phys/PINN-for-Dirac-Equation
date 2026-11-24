from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pypdf import PdfReader
from .r8_config import (ROOT, RAW, DOCS, FIGURES, LOGS, CANDIDATES, CONTROLS,
                        THRESHOLDS, VALIDATION_SEEDS, config_for, allocation)
from .r8_decision import read_stage
from .r8_models import R8Model, parameter_counts
from .r8_reference import reference_state
from .freeze_r8_coupling_recovery import verify_freeze, verify_history
from .freeze_r7_phase_release import write_json, sha256
from .audit_r7_phase_release import independent_metrics, link_check
from .train_r8_coupling_recovery import ARTIFACTS


def tensor_state_digest(state):
    """Content hash independent of torch archive timestamps or serialization."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def carrier_unchanged(final_state, pretrained_state):
    final = {key.removeprefix("carrier."): value for key, value in final_state.items()
             if key.startswith("carrier.")}
    return bool(final) and set(final) == set(pretrained_state) and all(
        torch.equal(final[key], pretrained_state[key]) for key in final)


def independent_absolute(group, seeds, minimum):
    if tuple(sorted(group.seed)) != tuple(seeds):
        return dict(pass_check=False, complete=False, joint_pass_count=0)
    flags = np.column_stack([group[key].to_numpy() >= value if key == "fidelity_mean"
                             else group[key].to_numpy() <= value for key, value in THRESHOLDS.items()])
    means = {key: bool(group[key].mean() >= value if key == "fidelity_mean"
                       else group[key].mean() <= value) for key, value in THRESHOLDS.items()}
    joint = flags.all(axis=1)
    return dict(pass_check=all(means.values()) and int(joint.sum()) >= minimum,
        complete=True, mean_checks=means, joint_pass_count=int(joint.sum()),
        per_seed_joint_pass={str(seed): bool(flag) for seed, flag in zip(group.seed, joint)})


def independent_score(group):
    return float(np.mean(np.column_stack([group.spinor_rel_l2_gauge_aligned/.10,
        group.current_rel_l2/.15, group.sideband_complex_coefficient_rel_l2/.10])))


def independently_decide(selection, validation):
    names = selection["selected_candidates"]+list(CONTROLS)
    groups = {name: independent_absolute(validation[validation.model==name], VALIDATION_SEEDS, 4) for name in names}
    successes = [name for name in selection["selected_candidates"] if groups[name]["pass_check"]]
    successes.sort(key=lambda name: (independent_score(validation[validation.model==name]), CANDIDATES.index(name)))
    return dict(stage="INDEPENDENT_VALIDATION", seeds=list(VALIDATION_SEEDS),
        classification="COUPLING_RECOVERY_SUCCESS" if successes else "NO_VALIDATED_RECOVERY",
        groups=groups, successful_candidates=successes, exploratory_selection_does_not_imply_validation=True)


def physical_sampled_spinor_metrics(dense, length):
    raw, exact = dense["psi_predicted_raw"], dense["psi_exact"]
    phase = np.angle(np.vdot(exact.ravel(), raw.ravel()))
    aligned = raw*np.exp(-1j*phase)
    pn, en = [length*np.mean(np.sum(abs(value)**2, axis=2), axis=1) for value in (raw, exact)]
    overlap = length*np.mean(np.sum(exact.conj()*raw, axis=2), axis=1)
    fidelity = np.clip(abs(overlap)**2/np.maximum(pn*en, 1e-30), 0, 1)
    return dict(spinor_rel_l2_gauge_aligned=float(np.linalg.norm(aligned-exact)/np.linalg.norm(exact)),
        spinor_rel_l2_raw=float(np.linalg.norm(raw-exact)/np.linalg.norm(exact)),
        fidelity_mean=float(fidelity.mean()), fidelity_min=float(fidelity.min()),
        max_norm_drift=float(np.max(abs(pn-1))))


def dynamics_series_matches(series, dense, physical_grid_length=None):
    predicted, exact = dense["spectral_state_predicted_raw"], dense["spectral_state_exact"]
    pn, en = np.sum(abs(predicted)**2, axis=(1, 2)), np.sum(abs(exact)**2, axis=(1, 2))
    fidelity = np.clip(abs(np.sum(exact.conj()*predicted, axis=(1, 2)))**2/np.maximum(pn*en, 1e-30), 0, 1)
    if physical_grid_length is not None:
        fields = [dense["psi_predicted_raw"], dense["psi_exact"]]
        pn, en = [physical_grid_length*np.mean(np.sum(abs(value)**2, axis=2), axis=1) for value in fields]
        overlap = physical_grid_length*np.mean(np.sum(fields[1].conj()*fields[0], axis=2), axis=1)
        fidelity = np.clip(abs(overlap)**2/np.maximum(pn*en, 1e-30), 0, 1)
    checks = {"t": dense["t"], "norm_predicted": pn, "norm_exact": en,
              "norm_drift": pn-1, "fidelity": fidelity}
    for index, offset in enumerate(dense["mode_offsets"]):
        for tag in ("predicted", "exact"):
            checks[f"mode_{offset:+d}_occupation_{tag}"] = dense[f"modal_occupations_{tag}"][:, index]
            for branch, label in ((0, "plus"), (1, "minus")):
                checks[f"mode_{offset:+d}_{label}_occupation_{tag}"] = dense[f"branch_occupations_{tag}"][:, index, branch]
    if len(series) != len(dense["t"]):
        return False
    return all(key in series and np.allclose(series[key], values, rtol=2e-12, atol=2e-12)
               for key, values in checks.items())


def affinity_masks_overlap(masks):
    return any(left & right for i, left in enumerate(masks) for right in masks[i+1:])


def runtime_affinity_diagnostic(tasks):
    # run_tasks batches these four queues separately. Conditional stages share a queue.
    queues = [[task for task in tasks if task[0] == stage] for stage in ("smoke", "exploration", "validation")]
    queues.append([task for task in tasks if task[0] in ("ablation", "extension", "rescue")])
    rows, groups, metadata_ok = [], [], True
    for queue_index, queue in enumerate(queues):
        for start in range(0, len(queue), 3):
            cohort = queue[start:start+3]
            commands = []
            for stage, case, name, seed, balanced in cohort:
                label = f"{stage}_{case}_{name}_seed{seed}"+("_balanced" if balanced else "")
                commands.append(json.loads((LOGS/(label+"_command.json")).read_text(encoding="utf-8")))
            masks = [int(command["affinity"]) for command in commands]
            overlap = affinity_masks_overlap(masks)
            group_id = f"queue{queue_index}_cohort{start//3}"
            groups.append(dict(cohort=group_id, affinity_masks=masks, shared_core=overlap,
                               tasks=[list(task) for task in cohort]))
            for task, command, mask in zip(cohort, commands, masks):
                stage, case, name, seed, balanced = task
                from .r8_config import run_directory
                config = json.loads((run_directory(stage, name, seed, case=case, balanced=balanced)/"config.json").read_text(encoding="utf-8"))
                metadata_ok &= (command["task"] == list(task) and command["threads"] == 1
                                and int(config["cpu_affinity_mask"]) == mask)
                rows.append(dict(stage=stage, case=case, model=name, seed=seed, balanced=balanced,
                    cohort=group_id, cpu_affinity_mask=mask, cohort_shared_cpu_core=overlap,
                    timing_is_observed_wall_cost_not_speedup_evidence=True))
    affected = sum(row["cohort_shared_cpu_core"] for row in rows if row["stage"] != "smoke")
    return dict(status="TIMING_COMPARISON_LIMITED" if affected else "PASS", metadata_pass=bool(metadata_ok),
        non_smoke_runs_in_shared_core_cohorts=affected, cohorts=groups, runs=rows,
        deviation="Frozen scheduler rotates affinity using individual seed and slot; mixed-seed cohorts can collide.",
        scientific_steps_samples_and_updates_not_modified=True,
        no_selective_rerun_or_postfreeze_science_edit=True)


def expected_runs():
    selection = json.loads((RAW/"exploration_selection.json").read_text(encoding="utf-8"))
    plan = json.loads((RAW/"conditional_execution_plan.json").read_text(encoding="utf-8"))
    tasks = [("smoke", "main", n, 0, False) for n in CANDIDATES+CONTROLS]
    tasks += [("exploration", "main", n, s, False) for s in range(3) for n in CANDIDATES+CONTROLS]
    tasks += [("validation", "main", n, s, False) for s in range(100, 105) for n in selection["selected_candidates"]+list(CONTROLS)]
    tasks += [("ablation", "main", n, s, False) for s in range(100, 105) for n in plan["ablations"]]
    tasks += [("extension", case, n, s, False) for case in plan["extension_cases"] for s in range(200, 203) for n in plan["extension_models"]]
    if plan["rescue_model"]:
        tasks += [("rescue", "main", plan["rescue_model"], s, True) for s in range(3)]
    return tasks


def reload_model(run_path):
    states = torch.load(run_path/"model_state.pt", weights_only=False)
    values = {k: v for k, v in states["configuration"].items() if k in config_for("smoke").__dataclass_fields__}
    from .r8_config import R8Config
    cfg = R8Config(**values)
    model = R8Model(cfg, states["model"], states["seed"])
    for _ in range(states["window_count"]-1):
        model.add_window()
    model.load_state_dict(states["state_dict"])
    if model.carrier is not None:
        model.carrier.freeze()
    return model, cfg, states


def figure_qa():
    manifest = json.loads((FIGURES/"figure_manifest.json").read_text(encoding="utf-8"))
    renders = json.loads((FIGURES/"qa/pdf_render_manifest.json").read_text(encoding="utf-8"))
    rendered = {r["figure_id"]: r for r in renders["figures"]}
    review = (FIGURES/"visual_review.md").read_text(encoding="utf-8")
    rows = []
    for figure in manifest["figures"]:
        name = figure["figure_id"]
        checks = [sha256(ROOT/item["path"]) == item["sha256"] for item in figure["exports"]+figure["source_data"]]
        with Image.open(FIGURES/(name+".png")) as im:
            dimensions = im.size
        svg_text = (FIGURES/(name+".svg")).read_text(encoding="utf-8").count("<text")
        pdf = PdfReader(FIGURES/(name+".pdf"))
        pdf_text = pdf.pages[0].extract_text()
        r = rendered[name]
        checks += [min(dimensions)>250, svg_text>0, len(pdf.pages)==1, len(pdf_text)>40,
                   r["pdf_sha256"] == sha256(FIGURES/(name+".pdf")),
                   r["rendered_sha256"] == sha256(ROOT/r["rendered_path"]),
                   f"| {name} | PASS |" in review]
        rows.append(dict(figure=name, pass_check=all(checks), dimensions=dimensions, svg_editable_text_count=svg_text))
    notes = ["# R8 FIGURE QA", "", "统计图展示全部seed；中心为均值，离散度为population SD(ddof=0)，不是SEM。正式动力学示例固定seed100。",
             "文件、源数据与导出哈希检查不替代目视检查；最终PNG和PDF渲染记录见visual_review.md及qa/pdf_render_manifest.json。", "",
             "| Figure | Integrity and actual visual review | PNG dimensions | Editable SVG texts |",
             "| --- | --- | --- | ---: |"]
    notes += [f"| {r['figure']} | {'PASS' if r['pass_check'] else 'FAIL'} | {r['dimensions']} | {r['svg_editable_text_count']} |" for r in rows]
    (FIGURES/"FIGURE_QA.md").write_text("\n".join(notes)+"\n", encoding="utf-8")
    return dict(pass_check=all(r["pass_check"] for r in rows), figures=rows)


def audit(*, final=True):
    frozen = verify_freeze()
    history = verify_history()
    checks = dict(frozen_science=True, previous_work_unchanged=history["pass_check"])
    errors, rows, streams, init_hashes = [], [], {}, {}
    carrier_initial_hashes, pretrained_hashes = {}, {}
    evaluation_adapters = []
    max_difference = 0.
    tasks = expected_runs()
    reference_info = json.loads((RAW/"reference_validation.json").read_text(encoding="utf-8"))
    checks["reference_validation"] = reference_info["status"] == "PASS"
    for stage, case, name, seed, balanced in tasks:
        from .r8_config import run_directory
        folder = run_directory(stage, name, seed, case=case, balanced=balanced)
        if any(not (folder/f).is_file() for f in ARTIFACTS+("recovery_state.pt", "execution_events.jsonl")):
            errors.append(f"Incomplete artifacts: {folder}")
            continue
        config = json.loads((folder/"config.json").read_text(encoding="utf-8"))
        metrics = json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
        states = torch.load(folder/"model_state.pt", weights_only=False)
        saved = torch.load(folder/"recovery_state.pt", weights_only=False)
        if case == "free_packet":
            adapter_file = folder/"evaluation_adapter_record.json"
            if not adapter_file.exists():
                errors.append(f"Missing V=0 evaluation compatibility record: {folder}")
            else:
                adapter = json.loads(adapter_file.read_text(encoding="utf-8"))
                adapter_ok = (adapter["status"] == "PASS" and adapter["extra_optimizer_steps"] == 0
                    and adapter["primary_metric_definitions_unchanged"]
                    and adapter["final_step"] == states["steps"]
                    and adapter["final_tensor_state_sha256"] == tensor_state_digest(states["state_dict"]))
                previous = adapter["original_final_checkpoint"]
                if previous is not None:
                    preserved = folder/"evaluation_failure_preserved/model_state.pt"
                    before = torch.load(preserved, weights_only=False)
                    adapter_ok &= (previous["tensor_state_sha256"] == tensor_state_digest(before["state_dict"])
                        == tensor_state_digest(states["state_dict"])
                        and previous["step"] == states["steps"]
                        and previous["training_seconds"] == metrics["training_seconds"])
                if not adapter_ok:
                    errors.append(f"Evaluation-only recovery changed weights, budget or metrics: {folder}")
                evaluation_adapters.append(dict(run=folder.relative_to(RAW).as_posix(),
                    checks_pass=bool(adapter_ok), recovered_final_checkpoint=previous is not None))
        cfg = config_for(stage, case, reference_info["cases"][case]["selected_half_width"])
        pre, windows, per = allocation(cfg, name)
        adam_steps = [int(record["step"].item()) for record in saved["optimizer"]["state"].values()]
        if not adam_steps or set(adam_steps) != {per} or saved["phase"] != "coupled":
            errors.append(f"Final Adam update count differs from frozen active-window budget: {folder}")
        if any(record.get("configuration") != cfg.as_dict() for record in (config, states, saved)):
            errors.append(f"Configuration does not match frozen physics and stage: {folder}")
        pcounts = {"parameter_count": 12932*windows+(12802 if pre else 0),
                   "stored_window_network_count": windows}
        weight_count = sum(v.numel() for k, v in states["state_dict"].items() if k.endswith(("weight", "bias")))
        reloaded, _, _ = reload_model(folder)
        if parameter_counts(reloaded)["parameter_count"] != weight_count:
            errors.append(f"Reconstructed model parameter count mismatch: {folder}")
        with np.load(folder/"dense_grid.npz") as dense:
            rebuilt, mask_ok = independent_metrics(dense, cfg.as_dict())
            if stage == "smoke":
                # Nx=32 is a dimension/gradient/I-O smoke, not a Parseval accuracy grid.
                rebuilt.update(physical_sampled_spinor_metrics(dense, cfg.length))
            side = np.abs(dense["mode_offsets"]) == 1
            pred, exact = dense["spectral_state_predicted_gauge_aligned"], dense["spectral_state_exact"]
            rebuilt["sideband_complex_coefficient_rel_l2"] = float(np.linalg.norm((pred-exact)[:, side])/np.linalg.norm(exact[:, side]))
            original_ref = reference_state(cfg, dense["t"])
            ref_error = float(abs(original_ref-exact).max())
            fourier = np.exp(1j*np.outer(dense["x"], dense["wave_numbers"]))/np.sqrt(cfg.length)
            reconstruction = np.einsum("xk,tks->txs", fourier, dense["spectral_state_predicted_raw"])
            reconstruction_error = float(abs(reconstruction-dense["psi_predicted_raw"]).max())
            series_ok = dynamics_series_matches(pd.read_csv(folder/"time_series.csv"), dense,
                physical_grid_length=cfg.length if stage == "smoke" else None)
            phase = np.angle(np.vdot(dense["psi_exact"].ravel(), dense["psi_predicted_raw"].ravel()))
            gauge_difference = float(max(
                abs(dense["psi_predicted_raw"]*np.exp(-1j*phase)-dense["psi_predicted_gauge_aligned"]).max(),
                abs(dense["spectral_state_predicted_raw"]*np.exp(-1j*phase)-dense["spectral_state_predicted_gauge_aligned"]).max()))
            indices = np.unique(np.linspace(0, len(dense["t"])-1, 7).astype(int))
            tt, kk = torch.meshgrid(torch.tensor(dense["t"][indices], dtype=torch.float64),
                torch.tensor(dense["wave_numbers"], dtype=torch.float64), indexing="ij")
            with torch.no_grad():
                checkpoint_predictions = reloaded(kk.reshape(-1, 1), tt.reshape(-1, 1)).reshape(len(indices), cfg.mode_count, 2).numpy()
            checkpoint_difference = float(abs(checkpoint_predictions-dense["spectral_state_predicted_raw"][indices]).max())
            if stage != "smoke":
                from scipy.integrate import cumulative_trapezoid
                norms = dense["residual_fine_norm"]
                integral = cumulative_trapezoid(norms, dense["residual_fine_t"], initial=0.)
                estimate_difference = float(abs(integral-dense["residual_fine_integral"]).max())
                residual_violation = float(np.maximum(dense["raw_spinor_error_norm_fine"]-dense["residual_error_estimate"], 0).max())
            else:
                estimate_difference, residual_violation = 0., 0.
        difference = max(abs(float(metrics[k])-v) for k, v in rebuilt.items())
        max_difference = max(max_difference, difference)
        finite_metrics = all(np.isfinite(float(metrics[key])) and np.isfinite(value) for key, value in rebuilt.items())
        if not finite_metrics or difference>3e-10 or not mask_ok or not series_ok or gauge_difference>2e-12 or ref_error>2e-12 or reconstruction_error>2e-12 or estimate_difference>2e-12 or checkpoint_difference>2e-11:
            errors.append(f"Independent metric/reference/mask mismatch: {folder}")
        if residual_violation>5e-5:
            errors.append(f"Numerical integral estimate violated (not a certificate): {folder}: {residual_violation}")
        trace = saved["coupled_sampler"]["trace"]
        normalized = saved["coupled_sampler"]["normalized_trace"]
        free = saved["free_sampler"]["trace"]
        digest = hashlib.sha256(trace.numpy().tobytes()).hexdigest()
        if (len(trace) != cfg.adam_steps-pre or len(free) != pre or metrics["steps"] != cfg.adam_steps
            or states["steps"] != cfg.adam_steps or saved["step"] != cfg.adam_steps
            or metrics["residual_training_spinor_points"] != (cfg.adam_steps+pre)*cfg.spectral_collocation_points
            or digest != metrics["coupled_time_stream_sha256"] or states["coupled_time_stream_sha256"] != digest
            or weight_count != pcounts["parameter_count"] or any(metrics[key] != value for key, value in pcounts.items())
            or (stage != "smoke" and config["freeze_manifest_sha256"] != frozen)):
            errors.append(f"Budget/parameter/RNG/freeze mismatch: {folder}")
        if any("reference" in k or "target" in k or "eigenvalue" in k or "frequency" in k for k in states["state_dict"]):
            errors.append(f"Forbidden label in checkpoint: {folder}")
        if pre and not (folder/"pretrained_carrier_state.pt").is_file():
            errors.append(f"Missing separately stored pretrained carrier: {folder}")
        elif pre:
            pretrained = torch.load(folder/"pretrained_carrier_state.pt", weights_only=False)
            if not carrier_unchanged(states["state_dict"], pretrained["state_dict"]):
                errors.append(f"Frozen carrier changed during coupled training: {folder}")
            carrier_initial_hashes.setdefault(seed, set()).add(config["carrier_initial_sha256"])
            pretrained_hashes.setdefault((case, seed, pre), set()).add(tensor_state_digest(pretrained["state_dict"]))
        hist = pd.read_csv(folder/"loss_history.csv")
        if int(hist.step.iloc[-1]) != cfg.adam_steps or not np.isfinite(hist.objective).all():
            errors.append(f"Incomplete/nonfinite history: {folder}")
        if windows>1 and stage != "smoke":
            table = pd.read_csv(folder/"window_handoffs.csv")
            if len(table) != 9 or table.hard_handoff_max_abs_error.max() > 2e-12:
                errors.append(f"Incorrect complex hard handoff: {folder}")
        streams.setdefault((stage, case, seed), []).append(normalized)
        init_hashes.setdefault(seed, set()).add(config["initial_envelope_sha256"])
        rows.append(dict(stage=stage, case=case, model=name, seed=seed, balanced=balanced,
            max_independent_difference=difference, masks_pass=mask_ok, reference_max_abs=ref_error,
            time_series_recomputed=series_ok,
            reloaded_final_checkpoint_prediction_max_abs=checkpoint_difference,
            single_global_gauge_max_abs_difference=gauge_difference,
            smoke_grid_not_used_for_accuracy_evidence=stage == "smoke",
            reconstruction_max_abs=reconstruction_error, numerical_integral_max_violation=residual_violation,
            actual_parameter_count=weight_count, steps=cfg.adam_steps, carrier_steps=pre,
            final_active_window_adam_steps=min(adam_steps) if adam_steps else None,
            residual_training_spinor_points=metrics["residual_training_spinor_points"]))
    checks["all_runs_and_metrics"] = len(rows) == len(tasks) and not errors
    checks["evaluation_only_compatibility_repair"] = all(row["checks_pass"] for row in evaluation_adapters)
    paired = []
    for key, traces in streams.items():
        for a in traces:
            for b in traces:
                length = min(len(a), len(b))
                paired.append(torch.equal(a[:length], b[:length]))
    checks["paired_normalized_sampling_prefixes"] = all(paired)
    checks["paired_initial_envelopes"] = all(len(v)==1 for v in init_hashes.values())
    checks["paired_initial_carriers"] = all(len(v)==1 for v in carrier_initial_hashes.values())
    checks["paired_pretrained_carriers"] = all(len(v)==1 for v in pretrained_hashes.values())
    selection = json.loads((RAW/"exploration_selection.json").read_text(encoding="utf-8"))
    exploration = read_stage("exploration")
    ranked = sorted(CANDIDATES, key=lambda name: (independent_score(exploration[exploration.model==name]), CANDIDATES.index(name)))
    checks["predeclared_exploratory_selection"] = (
        selection["selected_candidates"] == ranked[:2]
        and [row["model"] for row in selection["ranking"]] == ranked
        and all(abs(row["score"]-independent_score(exploration[exploration.model==row["model"]])) < 1e-12 for row in selection["ranking"]))
    recorded = json.loads((RAW/"decision.json").read_text(encoding="utf-8"))
    checks["independent_decision_recomputed"] = independently_decide(selection, read_stage("validation")) == recorded
    runtime = runtime_affinity_diagnostic(tasks)
    checks["runtime_metadata_truthful"] = runtime["metadata_pass"]
    if final:
        recorded_runtime = json.loads((RAW/"runtime_affinity_audit.json").read_text(encoding="utf-8"))
        checks["runtime_deviation_disclosed"] = recorded_runtime == runtime
        work=pd.read_csv(RAW/"residual_evaluation_work.csv")
        work_ok=len(work)==len(tasks)
        for stage,case,name,seed,balanced in tasks:
            selected=work[(work.stage==stage)&(work.case==case)&(work.model==name)&(work.seed==seed)]
            cfg=config_for(stage,case,reference_info["cases"][case]["selected_half_width"])
            pre,_,_=allocation(cfg,name)
            coupled=(cfg.test_t_points+(cfg.residual_time_points if stage!="smoke" else 0))*cfg.mode_count
            free=2*cfg.test_t_points*cfg.mode_count if pre else 0
            tangents=0 if case=="free_packet" else (1+int(stage!="smoke"))*cfg.mode_count
            work_ok &= len(selected)==1
            if len(selected)==1:
                row=selected.iloc[0]
                work_ok &= (row.evaluation_full_coupled_residual_points==coupled
                    and row.evaluation_carrier_unit_spinor_residual_points==free
                    and row.initial_derivative_only_diagnostic_points==tangents
                    and row.full_spinor_residual_points_training_and_evaluation
                        ==(cfg.adam_steps+pre)*cfg.spectral_collocation_points+coupled+free)
        checks["residual_evaluation_work_recomputed"]=bool(work_ok)
    if final:
        tests = ET.parse(LOGS/"final_pytest.xml").getroot()
        suites = [tests] if tests.tag=="testsuite" else list(tests.findall("testsuite"))
        test_count = sum(int(s.get("tests", 0)) for s in suites)
        checks["complete_old_and_new_pytest"] = test_count>=208 and all(sum(int(s.get(k, 0)) for k in ("errors", "failures", "skipped"))==0 for s in suites)
        qa = figure_qa()
        checks["figures_and_actual_visual_review"] = qa["pass_check"]
        report = ROOT/"R8_COUPLING_RECOVERY_REPORT.md"
        links = link_check(report)
        checks["all_report_links"] = links["pass_check"]
        checks["categorized_report_identical"] = sha256(report)==sha256(DOCS/"report.md")
        reporting = json.loads((RAW/"reporting_code_sha256.json").read_text(encoding="utf-8"))
        checks["reporting_code_hashes"] = all(sha256(ROOT/path)==digest for path, digest in reporting["files"].items())
    else:
        test_count, qa, links = None, None, None
    status = ("PASS_WITH_TIMING_DEVIATION" if runtime["status"] != "PASS" else "PASS") if all(checks.values()) else "FAIL"
    result = dict(status=status, final=final, checks=checks, runtime_affinity=runtime,
        expected_runs=len(tasks), checked_runs=len(rows), maximum_independent_metric_difference=max_difference,
        freeze_manifest_sha256=frozen, protected_history=history, errors=errors, per_run=rows,
        pytest_count=test_count, figure_qa=qa, report_links=links,
        evaluation_adapters=evaluation_adapters,
        engineering_audit_does_not_imply_scientific_success=True)
    write_json(RAW/("audit.json" if final else "data_audit.json"), result)
    print(json.dumps(dict(status=result["status"], runs=len(rows), errors=errors, checks=checks), ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-only", action="store_true")
    args = p.parse_args()
    raise SystemExit(0 if audit(final=not args.data_only)["status"].startswith("PASS") else 1)
