from __future__ import annotations
import math
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
from scipy.integrate import cumulative_trapezoid
from .r9_evaluate import field_metrics, spectral_grid, physical_quadrature, rel
from .r12_norm_diagnostics import norm_balance, carrier_envelope_decomposition
from .r12_residual import coupled_residual, free_carrier_residual, carrier_value_and_time_derivative
from .r12_config import RAW,ROOT
from .r12_hamiltonian import exterior_residual

def evaluate(model, cfg, *, full):
    directory=RAW/"references"/cfg.case
    with np.load(directory/"reference_test_grid.npz") as ref:
        times, exact = ref["t"], ref["state"]
    predicted = spectral_grid(model, cfg, times)
    metrics, series, dense = field_metrics(predicted, exact, cfg, times)
    metrics["case"] = cfg.case
    metrics["predicted_max_boundary_sixteen_layer_occupation"] = float(max(np.sum(abs(predicted[:,:16])**2,(1,2)).max(),np.sum(abs(predicted[:,-16:])**2,(1,2)).max()))
    metrics["initial_condition_max_abs_error"] = float(abs(predicted[0]-exact[0]).max())
    doubled = physical_quadrature(dense["spectral_predicted_aligned"], exact,
                                 replace(cfg, test_x_points=2*cfg.test_x_points))[3:]
    metrics.update(fourier_integral_no_aliasing=cfg.test_x_points > 2*(cfg.n_max-cfg.n_min),
        max_squared_bilinear_fourier_index=2*(cfg.n_max-cfg.n_min),
        fourier_norm_parseval_max_error=float(abs(cfg.length*dense["density_predicted"].mean(1)-
                                                 np.sum(abs(predicted)**2, (1, 2))).max()),
        density_quadrature_double_grid_difference=abs(metrics["density_rel_l2"]-rel(doubled[0], doubled[1])),
        current_quadrature_double_grid_difference=abs(metrics["current_rel_l2"]-rel(doubled[2], doubled[3])),
        evaluation_coupled_spinor_residual_points=0, evaluation_coupled_residual_batches=0,
        evaluation_carrier_unit_spinor_residual_points=0, evaluation_carrier_residual_batches=0,
        evaluation_full_residual=full)
    tables = {}
    if not full:
        return metrics, series, dense, tables
    import json
    selected=json.loads((directory/"selected_reference.json").read_text(encoding="utf-8"))
    with np.load(ROOT/selected["fine_grid_path"]) as ref:
        residual_times, reference = ref["t"], ref["state"]
    ks = torch.tensor(cfg.wave_numbers, dtype=torch.float64).reshape(-1, 1)
    residuals = []
    for start in range(0, len(residual_times), 32):
        ts = torch.tensor(residual_times[start:start+32], dtype=torch.float64).reshape(-1, 1)
        t = ts.repeat_interleave(cfg.mode_count, 0).requires_grad_()
        residuals.append(coupled_residual(model, cfg, ks.repeat(len(ts), 1), t).detach().numpy())
    residual = np.concatenate(residuals)
    raw = spectral_grid(model, cfg, residual_times)
    norm = np.linalg.norm(residual, axis=(1, 2))
    integral = cumulative_trapezoid(norm, residual_times, initial=0)
    error = np.linalg.norm(raw-reference, axis=(1, 2))
    exterior, source, exterior_n = exterior_residual(raw,cfg)
    edge = cfg.max_coupling_shift
    exterior_norm = np.linalg.norm(exterior, axis=(1, 2))
    full_residual = np.concatenate((exterior[:, :edge], residual, exterior[:, edge:]), axis=1)
    coarse = np.trapezoid(norm[::2], residual_times[::2])
    metrics.update(fine_unweighted_pde_loss=float(np.mean(norm**2)/(2*cfg.length)),
        residual_integral_fine=float(integral[-1]), residual_integral_coarse=float(coarse),
        residual_integral_quadrature_relative_difference=float(abs(integral[-1]-coarse)/max(integral[-1], 1e-300)),
        residual_estimate_max_numerical_violation=float(np.maximum(error-error[0]-integral, 0).max()),
        exterior_residual_leakage_max_l2_norm=float(exterior_norm.max()),
        exterior_residual_integral=float(np.trapezoid(exterior_norm, residual_times)),
        full_physical_residual_integral=float(np.trapezoid(np.linalg.norm(full_residual, axis=(1, 2)), residual_times)),
        residual_numerical_estimate_is_continuous_certificate=False,
        evaluation_coupled_spinor_residual_points=len(residual_times)*cfg.mode_count,
        evaluation_coupled_residual_batches=math.ceil(len(residual_times)/32),
        signed_exterior_archive_correct=True)
    dense.update(residual_t=residual_times, residual_complex_unweighted=residual,
        residual_unweighted_component_mean_power=np.mean(abs(residual)**2, -1)/cfg.length,
        exterior_residual_complex=exterior, exterior_positive_coupling_source_complex=source,
        exterior_residual_n=exterior_n,
        full_physical_residual_n=np.arange(cfg.n_min-edge, cfg.n_max+edge+1),
        full_physical_residual_complex=full_residual,
        residual_l2_norm=norm, residual_integral=integral, raw_error_fine=error,
        residual_error_estimate=error[0]+integral, exterior_residual_l2_norm=exterior_norm,
        spectral_prediction_raw_fine=raw)
    tables["residual_integral.csv"] = pd.DataFrame(dict(t=residual_times, residual_l2_norm=norm,
        integral=integral, raw_error=error, numerical_error_estimate=error[0]+integral, exterior_leakage=exterior_norm))
    tables["residual_by_mode.csv"] = pd.DataFrame(dict(n=cfg.indices, k=cfg.wave_numbers,
        unweighted_component_mean_power=np.mean(abs(residual)**2, (0, 2))/cfg.length))
    balance_metrics, balance_table = norm_balance(cfg, residual_times, raw, residual)
    metrics.update(balance_metrics, norm_diagnostic_additional_residual_calls=0,
                   norm_diagnostic_additional_forward_spinor_points=0)
    tables["norm_balance.csv"] = balance_table
    if model.carrier is not None:
        ps, slopes, defects = [], [], []
        for start in range(0, len(times), 32):
            ts = torch.tensor(times[start:start+32], dtype=torch.float64).reshape(-1, 1)
            k, t = ks.repeat(len(ts), 1), ts.repeat_interleave(cfg.mode_count, 0).requires_grad_()
            p, pt = carrier_value_and_time_derivative(model.carrier, cfg, k, t)
            defect = free_carrier_residual(model.carrier, cfg, k, t)
            ps.append(p.detach().numpy().reshape(len(ts), cfg.mode_count, 2))
            slopes.append((p.conj()*pt).imag.detach().numpy().reshape(len(ts), cfg.mode_count, 2))
            defects.append(defect.detach().numpy().reshape(len(ts), cfg.mode_count, 2, 2))
        p, slope, defect = map(np.concatenate, (ps, slopes, defects))
        energy = np.sqrt(np.asarray(cfg.wave_numbers)**2+cfg.mass**2)
        target = np.stack((-energy, energy), -1)
        metrics.update(carrier_free_pde_residual_rms=float(np.sqrt(np.mean(abs(defect)**2))),
            carrier_phase_slope_mean_relative_error=float(np.mean(abs(slope-target[None])/energy[None, :, None])),
            carrier_phase_slope_max_relative_error=float(np.max(abs(slope-target[None])/energy[None, :, None])),
            evaluation_carrier_unit_spinor_residual_points=2*len(times)*cfg.mode_count,
            evaluation_carrier_residual_batches=math.ceil(len(times)/32))
        dense.update(learned_carrier=p, learned_carrier_phase_slope=slope,
            carrier_target_phase_slope_evaluation_only=target, carrier_unit_spinor_defect=defect)
        ti, ni, bi = np.indices(slope.shape)
        tables["carrier_diagnostics.csv"] = pd.DataFrame(dict(t=times[ti.ravel()],
            n=np.asarray(cfg.indices)[ni.ravel()], branch=bi.ravel(), slope=slope.ravel(),
            target=np.broadcast_to(target, slope.shape).ravel(),
            unit_spinor_residual_norm=np.linalg.norm(defect, axis=-1).ravel()))
        dm, dt = carrier_envelope_decomposition(cfg, times, predicted, residual[::((len(residual_times)-1)//(len(times)-1))], p, defect)
        metrics.update(dm)
        tables["residual_decomposition.csv"] = dt
    return metrics, series, dense, tables
