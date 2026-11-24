from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from scipy.integrate import cumulative_trapezoid

from .periodic_potential_evaluate import (
    compute_periodic_potential_evaluation, _spectral_model_residual_grid,
    _reference_spectral_state, _project_branches, _batched_model,
)
from .periodic_potential_models import trainable_state_sha256
from .r7_config import EXPERIMENT


def spectral_grid(model, cfg, times):
    k = torch.tensor(cfg.wave_numbers, dtype=torch.float64)
    ts = torch.tensor(times, dtype=torch.float64)
    tm, km = torch.meshgrid(ts, k, indexing="ij")
    state = _batched_model(model, km.reshape(-1, 1), tm.reshape(-1, 1)).reshape(len(ts), len(k), 2)
    return state.numpy()


def evaluate_run(model, cfg, model_name, seed, *, steps, training_seconds,
                 final_loss, peak_rss_mb, peak_increment_mb, full_diagnostics=True):
    evaluation_cfg = cfg.evaluation_config()
    legacy_name = ("coupled_shared_spectral_complex" if model_name == "complex"
                   else "coupled_shared_spectral_ltpb")
    weight_hash = trainable_state_sha256(model)
    metrics, series, dense = compute_periodic_potential_evaluation(
        model, evaluation_cfg, model_name=legacy_name, seed=seed, parameter_count=12932,
        training_seconds=training_seconds, final_pde_loss=final_loss,
        peak_rss_mb=peak_rss_mb, peak_rss_increment_mb=peak_increment_mb,
        steps=steps, experiment=EXPERIMENT,
    )
    metrics.update(model=model_name, feature_and_gate_time_scale=cfg.t_max,
                   evaluation_time_horizon=cfg.training_horizon)
    centers = np.abs(dense["mode_offsets"]) > 0
    outside = np.sum(dense["modal_occupations_predicted"][:, centers], axis=1)
    outside_ref = np.sum(dense["modal_occupations_exact"][:, centers], axis=1)
    metrics.update(max_outside_center_occupation=float(np.max(outside)),
                   reference_max_outside_center_occupation=float(np.max(outside_ref)))
    series["outside_center_occupation_predicted"] = outside
    series["outside_center_occupation_exact"] = outside_ref
    extras = {}
    if full_diagnostics:
        diagnostic_metrics, arrays, tables = fine_residual_diagnostics(model, cfg)
        metrics.update(diagnostic_metrics)
        dense.update(arrays)
        extras.update(tables)
        early_metrics, early_arrays, early_table = early_diagnostics(model, cfg)
        metrics.update(early_metrics)
        dense.update(early_arrays)
        extras["early_time_series.csv"] = early_table
    if weight_hash != trainable_state_sha256(model):
        raise AssertionError("Post-step evaluation mutated trained parameters")
    return metrics, series, dense, extras


def fine_residual_diagnostics(model, cfg):
    times = np.linspace(0, cfg.training_horizon, cfg.residual_time_points)
    ks = torch.tensor(cfg.wave_numbers, dtype=torch.float64)
    residual = _spectral_model_residual_grid(model, ks, torch.tensor(times),
                                            mass=cfg.mass, potential_strength=cfg.potential_strength).numpy()
    norms = np.sqrt(np.sum(np.abs(residual)**2, axis=(1, 2)))
    integral = cumulative_trapezoid(norms, times, initial=0.0)
    coarse = np.trapezoid(norms[::2], times[::2])
    raw = spectral_grid(model, cfg, times)
    exact = _reference_spectral_state(cfg, times, ks.numpy())
    raw_error = np.sqrt(np.sum(np.abs(raw-exact)**2, axis=(1, 2)))
    initial_error = float(raw_error[0])
    bound = initial_error + integral
    outer = np.sqrt(np.sum(np.abs(raw[:, [0, -1]])**2, axis=(1, 2))) * cfg.potential_strength/2
    chunks = np.minimum((times / cfg.training_horizon * cfg.time_bins).astype(int), cfg.time_bins-1)
    residual_by_mode = np.mean(np.abs(residual)**2, axis=2) / cfg.length
    mode_rows = []
    for segment in range(cfg.time_bins):
        values = residual_by_mode[chunks == segment]
        for mode, k in enumerate(cfg.wave_numbers):
            mode_rows.append(dict(segment=segment+1, mode_offset=mode-cfg.ladder_half_width,
                                  k=k, unweighted_component_mean_power=float(values[:, mode].mean())))
    table = pd.DataFrame(dict(t=times, residual_l2_norm=norms, residual_integral=integral,
                              error_bound=bound, raw_spinor_error_norm=raw_error,
                              estimate_slack=bound-raw_error, exterior_leakage_l2_norm=outer))
    metrics = {
        "residual_integral_fine": float(integral[-1]),
        "residual_integral_coarse": float(coarse),
        "residual_integral_quadrature_abs_difference": float(abs(integral[-1]-coarse)),
        "residual_integral_quadrature_relative_difference": float(abs(integral[-1]-coarse)/max(integral[-1], 1e-30)),
        "raw_final_spinor_error_norm": float(raw_error[-1]),
        "residual_estimate_max_numerical_violation": float(np.max(np.maximum(raw_error-bound, 0))),
        "fine_unweighted_pde_loss": float(np.mean(norms**2)/(2*cfg.length)),
        "exterior_residual_leakage_max_l2_norm": float(np.max(outer)),
    }
    arrays = dict(residual_fine_t=times, residual_fine_norm=norms,
                  residual_fine_integral=integral, residual_error_estimate=bound,
                  raw_spinor_error_norm_fine=raw_error, residual_by_mode_fine=residual_by_mode,
                  exterior_residual_leakage_norm_fine=outer)
    return metrics, arrays, {"residual_integral.csv": table,
                              "residual_by_segment_mode.csv": pd.DataFrame(mode_rows)}


def early_diagnostics(model, cfg):
    horizon = min(0.5, cfg.training_horizon)
    times = np.unique(np.concatenate(([0.0], np.geomspace(1e-10, 1e-2, 33),
                                      np.linspace(0, horizon, cfg.early_time_points))))
    k = np.asarray(cfg.wave_numbers)
    raw = spectral_grid(model, cfg, times)
    exact = _reference_spectral_state(cfg, times, k)
    predicted = _project_branches(raw, k, cfg.mass)
    reference = _project_branches(exact, k, cfg.mass)
    rows = []
    for side in (-1, 1):
        index = cfg.ladder_half_width + side
        for branch in range(2):
            for n, time in enumerate(times):
                p, r = predicted[n, index, branch], reference[n, index, branch]
                rows.append(dict(t=time, sideband=side, branch=branch,
                                 predicted_real=p.real, predicted_imag=p.imag,
                                 reference_real=r.real, reference_imag=r.imag,
                                 predicted_occupation=abs(p)**2, reference_occupation=abs(r)**2,
                                 reference_phase_active=bool(abs(r)**2 >= cfg.active_occupation_threshold)))
    k_all = torch.tensor(cfg.wave_numbers, dtype=torch.float64).reshape(-1, 1)
    t0 = torch.zeros_like(k_all)
    with torch.no_grad():
        fields = model.mlp(model.features(k_all, t0))
    output_jacobians = []
    for mode in (cfg.ladder_half_width-1, cfg.ladder_half_width+1):
        for branch in range(2):
            r, b = (float(fields[mode, branch]), float(fields[mode, branch+2]))
            scale = cfg.t_max
            if model.model_name == "complex":
                jac = np.eye(2)/scale
            elif model.model_name == "original_ltpb":
                jac = np.array([[1.0/scale, 0], [0, 0]])
            else:
                jac = np.array([[math.cos(b), -r*math.sin(b)],
                                [math.sin(b), r*math.cos(b)]])/scale
            output_jacobians.append(jac)
    jacs = np.asarray(output_jacobians)
    minimum_singular = np.linalg.svd(jacs, compute_uv=False)[:, -1]
    masks = np.abs(reference)**2 >= cfg.active_occupation_threshold
    metrics = {"initial_output_tangent_jacobian_min_singular_value": float(minimum_singular.min()),
               "initial_output_tangent_jacobian_rank_two_count": int(np.sum(minimum_singular > 1e-12))}
    return metrics, dict(early_t=times, early_branch_coefficients_predicted_raw=predicted,
                         early_branch_coefficients_exact=reference, early_reference_phase_active_mask=masks,
                         initial_output_tangent_jacobians=jacs), pd.DataFrame(rows)
