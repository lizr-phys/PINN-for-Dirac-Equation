from __future__ import annotations

import math
import numpy as np
import pandas as pd
import torch
from .r9_models import initial_coefficients
from .r13_residual import free_carrier_residual

def evaluate_carrier(carrier, cfg):
    """Analytic frequencies are diagnostic labels, never training targets."""
    times = np.linspace(0, cfg.t_max, cfg.test_t_points)
    energy = np.sqrt(np.asarray(cfg.wave_numbers)**2+cfg.mass**2)
    target_slope = np.stack((-energy, energy), -1)
    target_phase = times[:, None, None]*target_slope[None]
    predicted, slopes, phases, defects = [], [], [], []
    ks = torch.as_tensor(cfg.wave_numbers, dtype=torch.float64).reshape(-1,1)
    with torch.no_grad():
        for start in range(0, len(times),32):
            ts = torch.as_tensor(times[start:start+32], dtype=torch.float64).reshape(-1,1)
            k, t = ks.repeat(len(ts),1), ts.repeat_interleave(cfg.mode_count,0)
            p, pt = carrier.value_and_time_derivative(k,t)
            if getattr(carrier, 'analytic_prior', False):
                phase = target_phase[start:start+len(ts)]
            else:
                phase = carrier.phase(k,t).numpy().reshape(len(ts),cfg.mode_count,2)
            predicted.append(p.numpy().reshape(len(ts),cfg.mode_count,2))
            slopes.append((p.conj()*pt).imag.numpy().reshape(len(ts),cfg.mode_count,2))
            phases.append(phase)
            defects.append(free_carrier_residual(carrier,cfg,k,t).numpy().reshape(len(ts),cfg.mode_count,2,2))
    p, slope, phase, defect = map(np.concatenate,(predicted,slopes,phases,defects))
    weight = initial_coefficients(cfg).abs().square().numpy()
    error = phase-target_phase
    wrapped = np.angle(np.exp(1j*error))
    chord = abs(p-np.exp(1j*target_phase))
    defect_square = np.sum(abs(defect)**2, axis=-1)
    def weighted_rms(square):
        return np.sqrt(np.sum(weight[None]*square, axis=(1,2)))
    unwrapped_t = weighted_rms(error**2)
    wrapped_t = weighted_rms(wrapped**2)
    chord_t = weighted_rms(chord**2)
    weighted_defect_t = weighted_rms(defect_square)
    full_defect_t = np.sqrt(np.mean(abs(defect)**2, axis=(1,2,3)))
    metrics = dict(
        initial_weighted_unwrapped_phase_rms=float(np.sqrt(np.mean(unwrapped_t**2))),
        initial_weighted_final_unwrapped_phase_rms=float(unwrapped_t[-1]),
        initial_weighted_wrapped_phase_rms=float(np.sqrt(np.mean(wrapped_t**2))),
        initial_weighted_carrier_chord_rms=float(np.sqrt(np.mean(chord_t**2))),
        initial_weighted_unit_spinor_residual_rms=float(np.sqrt(np.mean(weighted_defect_t**2))),
        carrier_free_pde_residual_rms=float(np.sqrt(np.mean(abs(defect)**2))),
        carrier_phase_slope_mean_relative_error=float(np.mean(abs(slope-target_slope[None])/energy[None,:,None])),
        carrier_unit_norm_max_drift=float(abs(abs(p)**2-1).max()),
        exact_labels_independent_evaluation_only=True, no_phase_alignment=True,
        evaluation_carrier_unit_spinor_residual_points=2*len(times)*cfg.mode_count,
        evaluation_carrier_residual_batches=math.ceil(len(times)/32))
    series = pd.DataFrame(dict(t=times, initial_weighted_unwrapped_phase_rmse=unwrapped_t,
        initial_weighted_wrapped_phase_rmse=wrapped_t, initial_weighted_carrier_chord_rmse=chord_t,
        initial_weighted_unit_spinor_residual_rms=weighted_defect_t,
        full_spectrum_unit_spinor_residual_rms=full_defect_t))
    dense = dict(t=times,n=np.asarray(cfg.indices),k=np.asarray(cfg.wave_numbers), carrier=p,
        carrier_phase=phase, carrier_phase_slope=slope, carrier_unit_spinor_defect=defect,
        target_phase_evaluation_only=target_phase, unwrapped_phase_error=error,
        wrapped_phase_error=wrapped, initial_branch_occupation_weight=weight,
        phase_initial_reference_active_mask=weight>=cfg.active_occupation_threshold)
    ti, ni, bi = np.indices(error.shape)
    table = pd.DataFrame(dict(t=times[ti.ravel()], n=np.asarray(cfg.indices)[ni.ravel()],
        branch=bi.ravel(),unwrapped_phase_error=error.ravel(),wrapped_phase_error=wrapped.ravel(),
        slope_error=(slope-target_slope[None]).ravel(),
        initial_weight=np.broadcast_to(weight,error.shape).ravel(),
        unit_spinor_residual_norm=np.sqrt(defect_square).ravel()))
    return metrics, series, dense, table
