from __future__ import annotations

import json
from dataclasses import replace
import numpy as np
import pandas as pd
from .r8_config import RAW, CANDIDATES, CONTROLS, THRESHOLDS, EXTENSION_SEEDS
from .r8_decision import read_stage, validation_decision, absolute_success, joint_mask
from .freeze_r7_phase_release import write_json

CORE = ("spinor_rel_l2_gauge_aligned", "current_rel_l2", "fidelity_mean",
        "modal_population_rel_l2", "max_norm_drift", "sideband_complex_coefficient_rel_l2",
        "active_coherence_rel_l2")


def bilinear_spectral_coefficients(state):
    """Exact finite-Fourier coefficients of density and current, without 1/L."""
    count = state.shape[1]
    rho, current = [], []
    for offset in range(-(count-1), count):
        if offset >= 0:
            left, right = state[:, :count-offset], state[:, offset:]
        else:
            left, right = state[:, -offset:], state[:, :count+offset]
        rho.append(np.sum(left.conj()*right, axis=(1, 2)))
        current.append(np.sum(left.conj()*right[..., ::-1], axis=(1, 2)))
    return np.stack(rho, axis=1), np.stack(current, axis=1)


def exact_bilinear_relative_errors(predicted, exact):
    observables_p = bilinear_spectral_coefficients(predicted)
    observables_e = bilinear_spectral_coefficients(exact)
    return tuple(float(np.linalg.norm(p-e)/np.linalg.norm(e)) for p, e in zip(observables_p, observables_e))


def global_spinor_distance(predicted, exact):
    phase = np.angle(np.vdot(exact.ravel(), predicted.ravel()))
    return float(np.linalg.norm(predicted*np.exp(-1j*phase)-exact)/np.linalg.norm(exact))


def uncoupled_free_diagnostic(validation):
    """Explicit evaluation oracle; it never enters a model or optimization."""
    from .r8_config import config_for
    from .r8_reference import reference_state
    from .periodic_potential_physics import build_coupled_hamiltonian
    cfg = config_for("validation")
    times = np.linspace(0, cfg.t_max, cfg.test_t_points)
    free_cfg = replace(cfg, potential_strength=0.)
    free, exact = reference_state(free_cfg, times), reference_state(cfg, times)
    rho_error, current_error = exact_bilinear_relative_errors(free, exact)
    modal_free, modal_exact = np.sum(abs(free)**2, axis=2), np.sum(abs(exact)**2, axis=2)
    fidelity = abs(np.sum(exact.conj()*free, axis=(1, 2)))**2/(
        np.sum(abs(exact)**2, axis=(1, 2))*np.sum(abs(free)**2, axis=(1, 2)))
    h0 = build_coupled_hamiltonian(free_cfg)
    h = build_coupled_hamiltonian(cfg)
    residual = free.reshape(len(times), -1)@(h0-h).T
    norms = np.linalg.norm(residual, axis=1)
    result = dict(diagnostic_oracle_not_a_learned_ranking_entry=True,
        spinor_rel_l2_gauge_aligned=global_spinor_distance(free, exact), density_rel_l2=rho_error,
        current_rel_l2=current_error, fidelity_mean=float(fidelity.mean()),
        modal_population_rel_l2=float(np.linalg.norm(modal_free-modal_exact)/np.linalg.norm(modal_exact)),
        max_norm_drift=float(np.max(abs(np.sum(abs(free)**2, axis=(1, 2))-1))),
        max_outside_center_occupation=float(modal_free[:, np.asarray(cfg.ladder_indices)!=0].sum(1).max()),
        coupled_residual_l2_mean=float(norms.mean()),
        expected_residual_norm_analytic=cfg.potential_strength/np.sqrt(2))
    write_json(RAW/"diagnostic_uncoupled_free.json", result)
    np.savez_compressed(RAW/"diagnostic_uncoupled_free.npz", t=times,
        spectral_state_free=free, modal_occupations_free=modal_free, coupled_residual_norm=norms)
    rows = []
    for _, run in validation.iterrows():
        with np.load(RAW/run.run_path/"dense_grid.npz") as data:
            distance = global_spinor_distance(data["spectral_state_predicted_raw"], free)
        rows.append(dict(model=run.model, seed=int(run.seed), spinor_distance_to_uncoupled_free=distance,
            spinor_distance_to_coupled_reference=run.spinor_rel_l2_gauge_aligned,
            diagnostic_only_no_independent_per_mode_alignment=True))
    pd.DataFrame(rows).to_csv(RAW/"validation_distance_to_free.csv", index=False)
    return result


def occupied_polar_required_outputs(initial, velocity, duration):
    radius = abs(initial)
    if np.any(radius == 0):
        raise ValueError("Occupied polar coordinates require a strictly nonzero initial coefficient")
    rotated = velocity*np.exp(-1j*np.angle(initial))
    return duration*rotated.real, duration*rotated.imag/radius


def window_coordinate_diagnostic(validation):
    """Local-output conditioning, not physical phase errors or training targets."""
    import torch
    from .audit_r8_coupling_recovery import reload_model
    from .r8_config import config_for
    from .r8_reference import reference_state
    from .periodic_potential_physics import build_coupled_hamiltonian, energy_basis_numpy
    group = validation[validation.model == "C_release_windows"]
    if group.empty:
        return
    cfg = config_for("validation")
    starts = np.arange(cfg.windows)*cfg.t_max/cfg.windows
    reference = reference_state(cfg, starts)
    matrix = build_coupled_hamiltonian(cfg)
    up, um = energy_basis_numpy(np.asarray(cfg.wave_numbers), cfg.mass)
    project = lambda state: np.stack(((state*up).sum(-1), (state*um).sum(-1)), axis=-1)
    reference_coeff = project(reference)
    rows = []
    for _, run in group.iterrows():
        model, _, _ = reload_model(RAW/run.run_path)
        k = torch.tensor(cfg.wave_numbers, dtype=torch.float64).reshape(-1, 1)
        for window, field in enumerate(model.fields):
            initial = field.initial_coeff.numpy()
            psi = initial[:, :1]*up+initial[:, 1:]*um
            required = project((-1j*(matrix@psi.ravel())).reshape(cfg.mode_count, 2))
            with torch.no_grad():
                features = torch.cat(((k-cfg.k0)/(cfg.ladder_half_width*cfg.coupling_wave_number), torch.zeros_like(k)), dim=1)
                output = field.mlp(features).numpy()
            radius, argument = abs(initial), np.angle(initial)
            r, q = output[:, :2], output[:, 2:]
            occupied = radius != 0
            actual = np.where(occupied, np.exp(1j*argument)*(r+1j*radius*q)/field.duration,
                              r*np.exp(1j*(argument+q))/field.duration)
            qscale = np.where(occupied, radius, abs(r))/field.duration
            rscale = np.full_like(radius, 1/field.duration)
            with np.errstate(divide="ignore", invalid="ignore"):
                condition = np.maximum(qscale, rscale)/np.minimum(qscale, rscale)
            ref_active = abs(reference_coeff[window])**2 >= 1e-6
            valid = occupied & ref_active
            required_q = np.full_like(radius, np.nan)
            if valid.any():
                _, required_q[valid] = occupied_polar_required_outputs(initial[valid], required[valid], field.duration)
            for index, offset in enumerate(cfg.ladder_indices):
                for branch in range(2):
                    rows.append(dict(model=run.model, seed=int(run.seed), window=window+1, t=field.start,
                        mode_offset=int(offset), branch=branch,
                        predicted_initial_population=float(radius[index, branch]**2),
                        reference_initial_population=float(abs(reference_coeff[window, index, branch])**2),
                        reference_active=bool(ref_active[index, branch]),
                        initial_nonzero=bool(occupied[index, branch]),
                        phase_output_tangent_scale=float(qscale[index, branch]),
                        amplitude_output_tangent_scale=float(rscale[index, branch]),
                        local_velocity_coordinate_condition=float(condition[index, branch]),
                        required_q_output_by_pde=float(required_q[index, branch]),
                        actual_q_output=float(q[index, branch]) if valid[index, branch] else np.nan,
                        local_initial_derivative_abs_error=float(abs(actual[index, branch]-required[index, branch])),
                        q_comparison_valid=bool(valid[index, branch]),
                        coordinate_diagnostic_not_physical_phase_error=True))
    table = pd.DataFrame(rows)
    table.to_csv(RAW/"window_coordinate_conditioning.csv", index=False)
    side = table[(table.mode_offset.abs()==1)&table.reference_active&table.initial_nonzero]
    summaries = []
    for seed, subset in side.groupby("seed"):
        summaries.append(dict(model="C_release_windows", seed=int(seed), valid_channel_handoffs=len(subset),
            median_local_velocity_condition=float(subset.local_velocity_coordinate_condition.median()),
            maximum_required_absolute_q=float(subset.required_q_output_by_pde.abs().max()),
            maximum_actual_absolute_q=float(subset.actual_q_output.abs().max()),
            mean_initial_derivative_abs_error=float(subset.local_initial_derivative_abs_error.mean())))
    pd.DataFrame(summaries).to_csv(RAW/"window_coordinate_summary.csv", index=False)


def summary_table(frame):
    numeric = frame.select_dtypes(include="number").columns.difference(["seed"])
    rows = []
    for keys, group in frame.groupby(["stage", "case", "model", "balanced"], dropna=False):
        row = dict(zip(("stage", "case", "model", "balanced"), keys))
        row["n_seeds"] = len(group)
        row["joint_absolute_pass_count"] = int(joint_mask(group).sum())
        for key in numeric:
            values = group[key].dropna().to_numpy(dtype=float)
            if len(values):
                row[key+"_mean"] = float(values.mean())
                row[key+"_population_sd"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_rows(target, baseline, label):
    rows = []
    joined = target.merge(baseline, on=["seed", "case"], suffixes=("_target", "_baseline"))
    for _, record in joined.iterrows():
        row = dict(comparison=label, seed=int(record.seed), case=record.case,
                   target_model=record.model_target, baseline_model=record.model_baseline)
        for metric in CORE:
            a, b = float(record[metric+"_target"]), float(record[metric+"_baseline"])
            row[metric+"_difference"] = a-b
            row[metric+"_ratio"] = a/b if abs(b)>1e-30 else np.nan
        row["spinor_current_joint_win"] = bool(record.spinor_rel_l2_gauge_aligned_target < record.spinor_rel_l2_gauge_aligned_baseline
                                              and record.current_rel_l2_target < record.current_rel_l2_baseline)
        rows.append(row)
    return rows


def residual_work_record(run, cfg):
    """Count full spinor-residual points; distinguish derivative-only diagnostics.

    The frozen legacy auxiliary evaluation counter includes one initial-tangent
    grid. It is not an instrumentation count and is retained verbatim in metrics.
    Here we separately enumerate the actual frozen evaluator call sites, including
    the disclosed free-packet skip, without revising primary metrics or budgets.
    """
    modes, times = cfg.mode_count, cfg.test_t_points
    fine = cfg.residual_time_points if run.stage != "smoke" else 0
    carrier = int(run.carrier_steps)>0
    coupled = (times+fine)*modes
    free = 2*times*modes if carrier else 0
    tangents = 0 if run.case == "free_packet" else (1+int(run.stage != "smoke"))*modes
    return dict(stage=run.stage, case=run.case, model=run.model, seed=int(run.seed),
        training_residual_batches=int(run.residual_training_batches),
        training_spinor_residual_points=int(run.residual_training_spinor_points),
        evaluation_full_coupled_residual_points=coupled,
        evaluation_carrier_unit_spinor_residual_points=free,
        initial_derivative_only_diagnostic_points=tangents,
        carrier_scalar_phase_derivative_points=2*times*modes if carrier else 0,
        legacy_auxiliary_evaluation_spinor_points=int(run.evaluation_coupled_spinor_points),
        full_spinor_residual_points_training_and_evaluation=int(run.residual_training_spinor_points)+coupled+free,
        pure_forward_calls_and_derivative_only_diagnostics_not_counted_as_residuals=True)


def aggregate():
    stages, combined = {}, []
    for stage in ("smoke", "exploration", "validation", "ablation", "extension", "rescue"):
        frame = read_stage(stage)
        stages[stage] = frame
        if frame.empty:
            continue
        frame.to_csv(RAW/(stage+"_runs.csv"), index=False)
        summary_table(frame).to_csv(RAW/(stage+"_summary.csv"), index=False)
        combined.append(frame)
    all_runs = pd.concat(combined, ignore_index=True)
    all_runs.to_csv(RAW/"all_runs.csv", index=False)
    summary_table(all_runs).to_csv(RAW/"all_summary.csv", index=False)
    from .r8_config import config_for
    reference_counts=json.loads((RAW/"reference_validation.json").read_text(encoding="utf-8"))
    work=[residual_work_record(run,config_for(run.stage,run.case,
        reference_counts["cases"][run.case]["selected_half_width"])) for _,run in all_runs.iterrows()]
    pd.DataFrame(work).to_csv(RAW/"residual_evaluation_work.csv",index=False)
    from .audit_r8_coupling_recovery import expected_runs, runtime_affinity_diagnostic
    runtime = runtime_affinity_diagnostic(expected_runs())
    write_json(RAW/"runtime_affinity_audit.json", runtime)
    pd.DataFrame(runtime["runs"]).to_csv(RAW/"cost_runtime_context.csv", index=False)
    quadrature = []
    for _, run in all_runs.iterrows():
        with np.load(RAW/run.run_path/"dense_grid.npz") as data:
            rho, current = exact_bilinear_relative_errors(data["spectral_state_predicted_raw"], data["spectral_state_exact"])
        quadrature.append(dict(stage=run.stage, case=run.case, model=run.model, seed=int(run.seed),
            density_error_fixed_grid=float(run.density_rel_l2), density_error_exact_fourier=rho,
            current_error_fixed_grid=float(run.current_rel_l2), current_error_exact_fourier=current,
            density_absolute_metric_difference=abs(rho-run.density_rel_l2),
            current_absolute_metric_difference=abs(current-run.current_rel_l2),
            diagnostic_only_primary_metrics_unchanged=True))
    pd.DataFrame(quadrature).to_csv(RAW/"physical_quadrature_check.csv", index=False)
    decision = validation_decision()
    validation = stages["validation"]
    uncoupled_free_diagnostic(validation)
    window_coordinate_diagnostic(validation)
    paired = []
    for model in validation.model.unique():
        if model in CONTROLS:
            continue
        for baseline in CONTROLS:
            paired += paired_rows(validation[validation.model == model], validation[validation.model == baseline],
                                   f"validation/{model}_minus_{baseline}")
    ablation = stages["ablation"]
    if not ablation.empty:
        counterparts = {"A_identity_carrier": "A_carrier_complex", "B_static_free": "B_free_correction",
                        "window_complex": "C_release_windows", "D_identity_carrier": "D_carrier_windows",
                        "A_carrier_complex": "D_carrier_windows"}
        for name in ablation.model.unique():
            target = validation[validation.model == counterparts[name]]
            paired += paired_rows(target, ablation[ablation.model == name], f"ablation/{counterparts[name]}_minus_{name}")
    rescue = stages["rescue"]
    if not rescue.empty:
        for name in rescue.model.unique():
            paired += paired_rows(rescue[rescue.model == name], stages["exploration"][stages["exploration"].model == name],
                                   f"exploratory_rescue/{name}_balanced_minus_original")
    pairs = pd.DataFrame(paired)
    pairs.to_csv(RAW/"paired_comparisons.csv", index=False)
    comparison_summary = []
    if not pairs.empty:
        for (comparison, case), group in pairs.groupby(["comparison", "case"]):
            row = dict(comparison=comparison, case=case, n_seeds=len(group),
                       spinor_current_joint_wins=int(group.spinor_current_joint_win.sum()))
            for key in pairs.select_dtypes(include="number").columns.difference(["seed"]):
                values = group[key].dropna().to_numpy(float)
                row[key+"_mean"] = float(values.mean()) if len(values) else None
                row[key+"_population_sd"] = float(values.std(ddof=0)) if len(values) else None
            comparison_summary.append(row)
    pd.DataFrame(comparison_summary).to_csv(RAW/"comparison_summary.csv", index=False)
    mechanism = []
    for name in decision["groups"]:
        group = validation[validation.model == name]
        item = dict(model=name, independently_validated_absolute_success=decision["groups"][name]["pass_check"],
            joint_absolute_pass_count=decision["groups"][name]["joint_pass_count"],
            mean_first_sideband_complex_error=float(group.sideband_complex_coefficient_rel_l2.mean()),
            mean_max_outside_center_occupation=float(group.max_outside_center_occupation.mean()),
            reference_max_outside_center_occupation=float(group.reference_max_outside_center_occupation.mean()),
            comparisons={})
        for baseline in CONTROLS:
            if baseline == name:
                continue
            control = validation[validation.model == baseline]
            paired_part = pd.DataFrame(paired_rows(group, control, "mechanism"))
            item["comparisons"][baseline] = dict(
                spinor_mean_error_ratio=float(group.spinor_rel_l2_gauge_aligned.mean()/control.spinor_rel_l2_gauge_aligned.mean()),
                current_mean_error_ratio=float(group.current_rel_l2.mean()/control.current_rel_l2.mean()),
                spinor_current_joint_wins=int(paired_part.spinor_current_joint_win.sum()))
        mechanism.append(item)
    write_json(RAW/"mechanism_summary.json", dict(stage="INDEPENDENT_VALIDATION",
        no_additional_posthoc_success_threshold=True,
        windowing_changes_stored_capacity_and_optimization=True, models=mechanism))
    extension_decisions = []
    ext = stages["extension"]
    if not ext.empty:
        for (case, name), group in ext.groupby(["case", "model"]):
            extension_decisions.append(dict(case=case, model=name,
                absolute_scope_diagnostic=absolute_success(group, EXTENSION_SEEDS, 2),
                not_five_seed_independent_main_validation=True))
    write_json(RAW/"extension_decision.json", dict(status="COMPLETED" if not ext.empty else "NOT_TRIGGERED",
        scope="Three new seeds per case; no extrapolation to arbitrary potentials", results=extension_decisions))
    packet = ext[ext.case == "free_packet"] if not ext.empty else pd.DataFrame()
    packet_metrics, packet_series = [], []
    if not packet.empty:
        import torch
        from .wavepacket_evaluate import _branch_diagnostics
        from .r8_config import config_for
        cfg = config_for("extension", "free_packet", 16)
        for _, run in packet.iterrows():
            with np.load(RAW/run.run_path/"dense_grid.npz") as data:
                extra, trajectories = _branch_diagnostics(data["branch_coefficients_predicted_gauge_aligned"],
                    data["branch_coefficients_exact"], torch.tensor(data["wave_numbers"], dtype=torch.float64),
                    torch.tensor(data["x"], dtype=torch.float64), data["t"], cfg)
                packet_metrics.append(dict(model=run.model, seed=int(run.seed), **extra,
                    posttraining_regression_diagnostic_only=True,
                    center_width_population_reliable=extra["branch_minimum_norm"] >= 1e-6))
                table = pd.DataFrame({key: values for key, values in trajectories.items() if values.ndim == 1})
                table["t"], table["model"], table["seed"] = data["t"], run.model, int(run.seed)
                packet_series.append(table)
        pd.DataFrame(packet_metrics).to_csv(RAW/"free_packet_regression_metrics.csv", index=False)
        pd.concat(packet_series, ignore_index=True).to_csv(RAW/"free_packet_regression_series.csv", index=False)
    write_json(RAW/"free_packet_regression_diagnostics.json", dict(status="COMPLETED" if packet_metrics else "NOT_TRIGGERED",
        inherited_R5_circular_center_width_definition=True, not_an_additional_success_gate=True,
        runs=len(packet_metrics), minimum_population_reliability_flag=1e-6))
    write_json(RAW/"rescue_status.json", dict(status="EXPLORATORY_COMPLETED" if not rescue.empty else "NOT_TRIGGERED",
        result_is_not_independent_validation=True, runs=len(rescue),
        reason="No rescue is allowed when a primary candidate passed independent validation" if rescue.empty else "All primary candidates failed; one fixed residual-balancing factor only"))
    paths = []
    for _, row in all_runs.iterrows():
        folder = RAW/row.run_path
        paths.append(dict(stage=row.stage, case=row.case, model=row.model, seed=int(row.seed),
            balanced=bool(row.balanced), run_path=row.run_path,
            **{file.replace(".", "_"): str(folder/file) for file in
               ("config.json", "metrics.json", "loss_history.csv", "time_series.csv", "dense_grid.npz", "model_state.pt")}))
    pd.DataFrame(paths).to_csv(RAW/"run_artifact_index.csv", index=False)
    print(json.dumps(dict(classification=decision["classification"], completed_runs=len(all_runs),
                         successful=decision["successful_candidates"], extension_runs=len(ext), rescue_runs=len(rescue))), flush=True)
    return all_runs


if __name__ == "__main__":
    aggregate()
