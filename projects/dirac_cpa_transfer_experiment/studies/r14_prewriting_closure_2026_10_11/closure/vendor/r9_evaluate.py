from __future__ import annotations

"""Independent fixed-endpoint evaluator; exact evolution stays in this module."""
import math
import numpy as np
import pandas as pd
import torch
from scipy.integrate import cumulative_trapezoid
from .r8_models import basis
from .r8_loss import derivative, free_carrier_residual
from .r9_loss import coupled_residual


def rel(predicted, exact):
    return float(np.linalg.norm(predicted-exact)/max(np.linalg.norm(exact), 1e-300))


def reconstruct(state, cfg, x):
    spatial = np.exp(1j*np.asarray(x)[:, None]*np.asarray(cfg.wave_numbers)[None])/math.sqrt(cfg.length)
    return np.einsum("xk,tkc->txc", spatial, state, optimize=True)


def branches(state, cfg):
    k = torch.tensor(cfg.wave_numbers, dtype=torch.float64)
    up, um = (v.numpy() for v in basis(k, cfg.mass))
    return np.stack((np.einsum("kc,tkc->tk", up, state), np.einsum("kc,tkc->tk", um, state)), -1)


def physical_quadrature(pred, exact, cfg):
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.test_x_points, endpoint=False)
    p, r = reconstruct(pred, cfg, x), reconstruct(exact, cfg, x)
    pop_p, pop_r = np.sum(abs(p)**2, -1), np.sum(abs(r)**2, -1)
    j_p, j_r = 2*(p[..., 0].conj()*p[..., 1]).real, 2*(r[..., 0].conj()*r[..., 1]).real
    return x, p, r, pop_p, pop_r, j_p, j_r


def branch_trajectories(coeff, cfg, x):
    up, um = (v.numpy() for v in basis(torch.tensor(cfg.wave_numbers, dtype=torch.float64), cfg.mass))
    result = []
    circular = np.exp(2j*np.pi*x/cfg.length)
    for b, u in enumerate((up, um)):
        psi = reconstruct(coeff[..., b, None]*u[None], cfg, x)
        rho = np.sum(abs(psi)**2, -1)
        norm = cfg.length*rho.mean(1)
        z = np.sum(rho*circular[None], 1)/np.maximum(rho.sum(1), 1e-300)
        center = cfg.length*np.unwrap(np.angle(z))/(2*np.pi)
        width = cfg.length*np.sqrt(-2*np.log(np.clip(abs(z), 1e-300, 1)))/(2*np.pi)
        result.append((center, width, norm))
    return result


def field_metrics(pred, exact, cfg, times):
    phase = float(np.angle(np.vdot(exact.ravel(), pred.ravel())))
    aligned = pred*np.exp(-1j*phase)
    x, psi_p, psi_r, rho_p, rho_r, j_p, j_r = physical_quadrature(aligned, exact, cfg)
    norm_p = np.sum(abs(pred)**2, (1, 2))
    norm_r = np.sum(abs(exact)**2, (1, 2))
    inner = np.sum(exact.conj()*pred, (1, 2))
    fidelity = abs(inner)**2/np.maximum(norm_p*norm_r, 1e-300)
    c_p, c_r = branches(aligned, cfg), branches(exact, cfg)
    occ_p, occ_r = abs(c_p)**2, abs(c_r)**2
    modal_p, modal_r = np.sum(occ_p, -1), np.sum(occ_r, -1)
    active = occ_r >= cfg.active_occupation_threshold
    phase_defined = active & (occ_p >= 1e-12)
    phase_error = np.angle(c_p*np.conj(c_r))
    flat_p, flat_r = c_p.reshape(len(times), -1), c_r.reshape(len(times), -1)
    flat_mask = active.reshape(len(times), -1)
    coherence_num, coherence_den, pairs = 0., 0., 0
    coherence_time = []
    for p, r, mask in zip(flat_p, flat_r, flat_mask):
        valid = mask[:, None] & mask[None, :] & ~np.eye(len(mask), dtype=bool)
        cp, cr = p[:, None]*p[None, :].conj(), r[:, None]*r[None, :].conj()
        numerator = float(np.sum(abs(cp[valid]-cr[valid])**2))
        denominator = float(np.sum(abs(cr[valid])**2))
        coherence_num += numerator
        coherence_den += denominator
        pairs += int(valid.sum())
        coherence_time.append(math.sqrt(numerator/max(denominator, 1e-300)))
    outside = (np.asarray(cfg.indices)<0) | (np.asarray(cfg.indices)>32)
    boundary = np.r_[np.arange(8), np.arange(cfg.mode_count-8, cfg.mode_count)]
    metrics = dict(spinor_rel_l2_raw=rel(pred, exact), spinor_rel_l2_gauge_aligned=rel(aligned, exact),
        spinor_final_rel_l2=rel(aligned[-1], exact[-1]), density_rel_l2=rel(rho_p, rho_r),
        current_rel_l2=rel(j_p, j_r), fidelity_mean=float(fidelity.mean()),
        fidelity_min=float(fidelity.min()), fidelity_final=float(fidelity[-1]),
        max_norm_drift=float(abs(norm_p-1).max()), modal_population_rel_l2=rel(modal_p, modal_r),
        branch_population_rel_l2=rel(occ_p, occ_r), complex_spectral_coefficient_rel_l2=rel(aligned, exact),
        branch_complex_coefficient_rel_l2=rel(c_p, c_r), global_spacetime_constant_phase=phase,
        active_coherence_rel_l2=math.sqrt(coherence_num/max(coherence_den, 1e-300)),
        active_coherence_pair_count=pairs, active_channel_sample_count=int(active.sum()),
        phase_defined_reference_active_coverage=float(phase_defined.sum()/max(active.sum(), 1)),
        active_phase_circular_rmse=float(np.sqrt(np.mean(phase_error[phase_defined]**2))) if phase_defined.any() else None,
        predicted_max_occupation_outside_initial_support=float(modal_p[:, outside].sum(1).max()),
        reference_max_occupation_outside_initial_support=float(modal_r[:, outside].sum(1).max()),
        predicted_max_boundary_eight_layer_occupation=float(modal_p[:, boundary].sum(1).max()),
        lower_boundary_max_occupation=float(modal_p[:, :8].sum(1).max()),
        upper_boundary_max_occupation=float(modal_p[:, -8:].sum(1).max()))
    series = dict(t=times, norm_predicted=norm_p, norm_reference=norm_r, fidelity=fidelity,
                  spinor_raw_error=np.sqrt(np.sum(abs(pred-exact)**2, (1, 2))),
                  spinor_aligned_error=np.sqrt(np.sum(abs(aligned-exact)**2, (1, 2))),
                  coherence_relative_error=coherence_time,
                  occupation_outside_initial_support_predicted=modal_p[:, outside].sum(1),
                  occupation_outside_initial_support_reference=modal_r[:, outside].sum(1))
    trajectories_p, trajectories_r = branch_trajectories(c_p, cfg, x), branch_trajectories(c_r, cfg, x)
    for b, (p, r) in enumerate(zip(trajectories_p, trajectories_r)):
        label = ("plus", "minus")[b]
        for index, measure in enumerate(("center", "width", "norm")):
            series[f"{label}_{measure}_predicted"] = p[index]
            series[f"{label}_{measure}_reference"] = r[index]
        metrics[f"{label}_center_rmse"] = float(np.sqrt(np.mean((p[0]-r[0])**2)))
        metrics[f"{label}_center_trajectory_relative_l2"] = rel(p[0], r[0])
        metrics[f"{label}_width_relative_l2"] = rel(p[1], r[1])
        metrics[f"{label}_min_predicted_norm"] = float(p[2].min())
    dense = dict(t=np.asarray(times), x=x, n=np.asarray(cfg.indices), k=np.asarray(cfg.wave_numbers),
                 spectral_predicted_raw=pred, spectral_predicted_aligned=aligned, spectral_reference=exact,
                 psi_predicted_aligned=psi_p, psi_reference=psi_r,
                 density_predicted=rho_p, density_reference=rho_r, current_predicted=j_p, current_reference=j_r,
                 modal_occupations_predicted=modal_p, modal_occupations_reference=modal_r,
                 branch_coefficients_predicted_aligned=c_p, branch_coefficients_reference=c_r,
                 branch_occupations_predicted=occ_p, branch_occupations_reference=occ_r,
                 phase_reference_active_mask=active, coherence_reference_active_channel_mask=flat_mask,
                 phase_predicted_defined_mask=phase_defined, active_phase_error=phase_error,
                 initial_support_mask=~outside)
    return metrics, pd.DataFrame(series), dense


def spectral_grid(model, cfg, times):
    result = []
    ks = torch.tensor(cfg.wave_numbers, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        for start in range(0, len(times), 32):
            ts = torch.tensor(times[start:start+32], dtype=torch.float64).reshape(-1, 1)
            result.append(model(ks.repeat(len(ts), 1), ts.repeat_interleave(cfg.mode_count, 0)).numpy()
                          .reshape(len(ts), cfg.mode_count, 2))
    return np.concatenate(result)


def evaluate(model, cfg, *, full=True):
    from .r9_reference import solve
    times = np.linspace(0, cfg.t_max, cfg.test_t_points)
    exact = solve(cfg, times)["state"]
    predicted = spectral_grid(model, cfg, times)
    metrics, series, dense = field_metrics(predicted, exact, cfg, times)
    metrics["initial_condition_max_abs_error"] = float(abs(predicted[0]-exact[0]).max())
    # Squared bilinear errors have index range +/-2*(nmax-nmin).
    span = cfg.n_max-cfg.n_min
    x2cfg = cfg.__class__(**(vars(cfg) | {"test_x_points": 2*cfg.test_x_points}))
    doubled = physical_quadrature(dense["spectral_predicted_aligned"], exact, x2cfg)[3:]
    metrics.update(fourier_integral_no_aliasing=bool(cfg.test_x_points > 2*span),
                   max_squared_bilinear_fourier_index=2*span,
                   fourier_norm_parseval_max_error=float(abs(cfg.length*dense["density_predicted"].mean(1)
                                                             -np.sum(abs(predicted)**2, (1, 2))).max()),
                   density_quadrature_double_grid_difference=abs(metrics["density_rel_l2"]-rel(doubled[0], doubled[1])),
                   current_quadrature_double_grid_difference=abs(metrics["current_rel_l2"]-rel(doubled[2], doubled[3])))
    tables = {}
    residual_times = np.linspace(0, cfg.t_max, cfg.residual_time_points if full else 21)
    ks = torch.tensor(cfg.wave_numbers, dtype=torch.float64).reshape(-1, 1)
    residuals = []
    for start in range(0, len(residual_times), 32):
        ts = torch.tensor(residual_times[start:start+32], dtype=torch.float64).reshape(-1, 1)
        t = ts.repeat_interleave(cfg.mode_count, 0).requires_grad_()
        residuals.append(coupled_residual(model, cfg, ks.repeat(len(ts), 1), t, False).detach().numpy())
    residual = np.concatenate(residuals)
    raw = spectral_grid(model, cfg, residual_times)
    reference = solve(cfg, residual_times)["state"]
    norm = np.sqrt(np.sum(abs(residual)**2, (1, 2)))
    integral = cumulative_trapezoid(norm, residual_times, initial=0)
    error = np.sqrt(np.sum(abs(raw-reference)**2, (1, 2)))
    # Every outer eight-index block sources one block just outside the retained grid.
    exterior = cfg.potential_strength/2*np.concatenate((raw[:, :8], raw[:, -8:]), axis=1)
    exterior_norm = np.linalg.norm(exterior, axis=(1, 2))
    full_norm = np.sqrt(norm**2+exterior_norm**2)
    coarse = np.trapezoid(norm[::2], residual_times[::2])
    metrics.update(fine_unweighted_pde_loss=float(np.mean(norm**2)/(2*cfg.length)),
        residual_integral_fine=float(integral[-1]), residual_integral_coarse=float(coarse),
        residual_integral_quadrature_relative_difference=float(abs(integral[-1]-coarse)/max(integral[-1], 1e-300)),
        residual_estimate_max_numerical_violation=float(np.maximum(error-error[0]-integral, 0).max()),
        exterior_residual_leakage_max_l2_norm=float(exterior_norm.max()),
        exterior_residual_integral=float(np.trapezoid(exterior_norm, residual_times)),
        full_physical_residual_integral=float(np.trapezoid(full_norm, residual_times)),
        residual_numerical_estimate_is_continuous_certificate=False,
        evaluation_coupled_spinor_residual_points=len(residual_times)*cfg.mode_count,
        evaluation_coupled_residual_batches=math.ceil(len(residual_times)/32),
        evaluation_carrier_unit_spinor_residual_points=0, evaluation_carrier_residual_batches=0)
    dense.update(residual_t=residual_times, residual_complex_unweighted=residual,
                 residual_unweighted_component_mean_power=np.mean(abs(residual)**2, -1)/cfg.length,
                 exterior_residual_complex=exterior, exterior_residual_n=np.r_[np.arange(cfg.n_min-8, cfg.n_min),
                       np.arange(cfg.n_max+1, cfg.n_max+9)],
                 residual_l2_norm=norm, residual_integral=integral, raw_error_fine=error,
                 residual_error_estimate=error[0]+integral, exterior_residual_l2_norm=exterior_norm)
    tables["residual_integral.csv"] = pd.DataFrame(dict(t=residual_times, residual_l2_norm=norm,
        integral=integral, raw_error=error, numerical_error_estimate=error[0]+integral, exterior_leakage=exterior_norm))
    tables["residual_by_mode.csv"] = pd.DataFrame(dict(n=cfg.indices, k=cfg.wave_numbers,
        unweighted_component_mean_power=np.mean(abs(residual)**2, (0, 2))/cfg.length))
    if model.carrier is not None:
        ps, slopes, defects = [], [], []
        for start in range(0, len(times), 32):
            ts = torch.tensor(times[start:start+32], dtype=torch.float64).reshape(-1, 1)
            t = ts.repeat_interleave(cfg.mode_count, 0).requires_grad_()
            k = ks.repeat(len(ts), 1)
            p = model.carrier(k, t)
            dt = derivative(p, t, False)
            defect = free_carrier_residual(model.carrier, cfg, k, t, False).detach().numpy()
            ps.append(p.detach().numpy().reshape(len(ts), cfg.mode_count, 2))
            slopes.append((p.conj()*dt).imag.detach().numpy().reshape(len(ts), cfg.mode_count, 2))
            defects.append(defect.reshape(len(ts), cfg.mode_count, 2, 2))
        p, slope, defect = map(np.concatenate, (ps, slopes, defects))
        # Evaluation only: the target rate is never serialized to learned checkpoints.
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
        tables["carrier_diagnostics.csv"] = pd.DataFrame(dict(t=times[ti.ravel()], n=np.asarray(cfg.indices)[ni.ravel()],
            branch=bi.ravel(), slope=slope.ravel(), target=np.broadcast_to(target, slope.shape).ravel(),
            unit_spinor_residual_norm=np.linalg.norm(defect, axis=-1).ravel()))
    return metrics, series, dense, tables
