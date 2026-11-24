from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.integrate import cumulative_trapezoid
from .periodic_potential_evaluate import (compute_periodic_potential_evaluation, _batched_model,
    _spectral_model_residual_grid, _project_branches, _relative_l2)
from .r8_reference import reference_state, initial_numpy
from .r8_models import parameter_counts, basis
from .r8_loss import derivative, free_carrier_residual


class EvaluationProxy(nn.Module):
    """Black-box adapter: legacy diagnostics must not differentiate frozen weights."""
    def __init__(self, model):
        super().__init__()
        self.original = model

    def parameters(self, recurse=True):
        return iter(())

    def forward(self, k, t):
        return self.original(k, t)


def spectral_grid(model, cfg, times):
    ts, ks = torch.tensor(times, dtype=torch.float64), torch.tensor(cfg.wave_numbers, dtype=torch.float64)
    tm, km = torch.meshgrid(ts, ks, indexing="ij")
    return _batched_model(model, km.reshape(-1, 1), tm.reshape(-1, 1)).reshape(len(ts), len(ks), 2).numpy()


def evaluate_run(model, cfg, *, seed, seconds, final_loss, rss, rss_increment, full=True):
    exact = reference_state(cfg, np.linspace(0, cfg.t_max, cfg.test_t_points))
    metrics, series, dense = compute_periodic_potential_evaluation(
        EvaluationProxy(model), cfg, model_name="coupled_shared_spectral_complex", seed=seed,
        parameter_count=parameter_counts(model)["parameter_count"], training_seconds=seconds,
        final_pde_loss=final_loss, peak_rss_mb=rss, peak_rss_increment_mb=rss_increment,
        steps=cfg.adam_steps, experiment="r8_coupling_recovery", reference_spectral=exact)
    metrics.update(model=model.model_name, case=cfg.case, **parameter_counts(model))
    # Legacy sensitivity is intentionally replaced by a directly differentiated local field.
    metrics.pop("initial_sideband_imaginary_tangent_parameter_sensitivity", None)
    if cfg.case == "free_packet":
        for name in list(metrics):
            if name.startswith("initial_sideband_"):
                metrics.pop(name)
        for name in list(dense):
            if name.startswith("initial_"):
                dense.pop(name)
    c = dense["branch_coefficients_predicted_gauge_aligned"]
    ce = dense["branch_coefficients_exact"]
    side = np.abs(dense["mode_offsets"]) == 1
    metrics["sideband_complex_coefficient_rel_l2"] = _relative_l2(c[:, side], ce[:, side])
    outside = dense["mode_offsets"] != 0
    for tag in ("predicted", "exact"):
        population = dense[f"modal_occupations_{tag}"][:, outside].sum(1)
        series[f"outside_center_occupation_{tag}"] = population
        metrics[("reference_" if tag == "exact" else "")+"max_outside_center_occupation"] = float(population.max())
    tables = {}
    if full:
        times = np.linspace(0, cfg.t_max, cfg.residual_time_points)
        ks = torch.tensor(cfg.wave_numbers, dtype=torch.float64)
        residual = _spectral_model_residual_grid(model, ks, torch.tensor(times), mass=cfg.mass,
                                                potential_strength=cfg.potential_strength).numpy()
        raw = spectral_grid(model, cfg, times)
        ref = reference_state(cfg, times)
        norms = np.sqrt(np.sum(abs(residual)**2, axis=(1, 2)))
        integral = cumulative_trapezoid(norms, times, initial=0.)
        error = np.sqrt(np.sum(abs(raw-ref)**2, axis=(1, 2)))
        bound = error[0]+integral
        coarse = np.trapezoid(norms[::2], times[::2])
        by_mode = np.mean(abs(residual)**2, axis=2)/cfg.length
        outer = np.linalg.norm(raw[:, [0, -1]], axis=(1, 2))*cfg.potential_strength/2
        dense.update(residual_fine_t=times, residual_fine_norm=norms, residual_fine_integral=integral,
                     residual_error_estimate=bound, raw_spinor_error_norm_fine=error,
                     residual_by_mode_fine=by_mode, exterior_residual_leakage_norm_fine=outer)
        metrics.update(fine_unweighted_pde_loss=float(np.mean(norms**2)/(2*cfg.length)),
                       residual_integral_fine=float(integral[-1]), residual_integral_coarse=float(coarse),
                       residual_integral_quadrature_relative_difference=float(abs(integral[-1]-coarse)/max(integral[-1], 1e-30)),
                       residual_estimate_max_numerical_violation=float(np.maximum(error-bound, 0).max()),
                       raw_final_spinor_error_norm=float(error[-1]),
                       exterior_residual_leakage_max_l2_norm=float(outer.max()))
        tables["residual_integral.csv"] = pd.DataFrame(dict(t=times, residual_l2_norm=norms,
            residual_integral=integral, raw_spinor_error_norm=error, error_bound=bound, exterior_leakage=outer))
        chunks = np.minimum((times/cfg.t_max*10).astype(int), 9)
        tables["residual_by_segment_mode.csv"] = pd.DataFrame([
            dict(segment=w+1, mode_offset=j-cfg.ladder_half_width, k=k,
                 unweighted_component_mean_power=float(by_mode[chunks == w, j].mean()))
            for w in range(10) for j, k in enumerate(cfg.wave_numbers)])
        if cfg.case != "free_packet":
            field = model.fields[0]
            old_flags = [p.requires_grad for p in field.parameters()]
            for p in field.parameters():
                p.requires_grad_(True)
            k = ks.reshape(-1, 1)
            t = torch.zeros_like(k, requires_grad=True)
            state = field(k, t, model.carrier)
            dt = derivative(state, t)
            sensitivity_grads = torch.autograd.grad(dt[side].imag.sum(), tuple(field.parameters()), allow_unused=True)
            metrics["initial_sideband_imaginary_tangent_parameter_sensitivity"] = float(np.sqrt(sum(float((g*g).sum()) for g in sensitivity_grads if g is not None)))
            for p, flag in zip(field.parameters(), old_flags):
                p.requires_grad_(flag)
        joins = []
        for w in range(1, len(model.fields)):
            k = ks.reshape(-1, 1)
            t = torch.full_like(k, model.fields[w].start)
            with torch.no_grad():
                left = model.fields[w-1](k, t, model.carrier).numpy()
                right = model.fields[w](k, t, model.carrier).numpy()
            joins.append(dict(window=w+1, t=model.fields[w].start,
                              hard_handoff_max_abs_error=float(abs(left-right).max())))
        if joins:
            tables["window_handoffs.csv"] = pd.DataFrame(joins)
            metrics["window_handoff_max_abs_error"] = max(r["hard_handoff_max_abs_error"] for r in joins)
    if model.carrier is not None:
        ks = torch.tensor(cfg.wave_numbers, dtype=torch.float64)
        ts = torch.tensor(dense["t"], dtype=torch.float64)
        tm, km = torch.meshgrid(ts, ks, indexing="ij")
        t = tm.reshape(-1, 1).clone().requires_grad_(True)
        k = km.reshape(-1, 1)
        p = model.carrier(k, t)
        dt = derivative(p, t, False)
        slope = (p.conj()*dt).imag.detach().reshape(len(ts), len(ks), 2).numpy()
        # Exact dispersion appears only in this independent post-training evaluator.
        target = np.stack((-np.sqrt(ks.numpy()**2+cfg.mass**2), np.sqrt(ks.numpy()**2+cfg.mass**2)), -1)
        residual = free_carrier_residual(model.carrier, cfg, k, t, False).detach().numpy()
        dense.update(learned_carrier=p.detach().reshape(len(ts), len(ks), 2).numpy(),
                     learned_carrier_phase_slope=slope, carrier_target_phase_slope_evaluation_only=target)
        metrics.update(carrier_free_pde_residual_rms=float(np.sqrt(np.mean(abs(residual)**2))),
                       carrier_phase_slope_mean_relative_error=float(np.mean(abs(slope-target[None])/abs(target[None]))),
                       carrier_phase_slope_max_relative_error=float(np.max(abs(slope-target[None])/abs(target[None]))))
        tables["carrier_diagnostics.csv"] = pd.DataFrame([dict(t=float(time), k=float(wave), branch=b,
            learned_slope=float(slope[i, j, b]), target_slope=float(target[j, b]),
            relative_error=float(abs(slope[i, j, b]-target[j, b])/abs(target[j, b])))
            for i, time in enumerate(ts) for j, wave in enumerate(ks) for b in range(2)])
    return metrics, series, dense, tables
