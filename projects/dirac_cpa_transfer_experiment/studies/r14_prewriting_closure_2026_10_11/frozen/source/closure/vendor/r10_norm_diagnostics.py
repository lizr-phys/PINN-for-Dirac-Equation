from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid


def branch_projection(state, cfg):
    angle = np.arctan2(np.asarray(cfg.wave_numbers), cfg.mass)/2
    up = np.stack((np.cos(angle), np.sin(angle)), -1)
    down = np.stack((-np.sin(angle), np.cos(angle)), -1)
    return np.stack((np.einsum("kc,tkc->tk", up, state),
                     np.einsum("kc,tkc->tk", down, state)), -1)


def potential_apply(state, cfg):
    result = np.zeros_like(state)
    result[:, 8:] = state[:, :-8]
    result[:, :-8] += state[:, 8:]
    return cfg.potential_strength/2*result


def norm_balance(cfg, times, raw, residual):
    """Exact residual identity followed by explicitly numerical quadrature."""
    if raw.shape != residual.shape or raw.shape != (len(times), cfg.mode_count, 2):
        raise ValueError("Ordered complete retained spinor arrays required")
    norm = np.sum(np.abs(raw)**2, axis=(1, 2))
    rate = 2*np.imag(np.sum(raw.conj()*residual, axis=(1, 2)))
    integral = cumulative_trapezoid(rate, times, initial=0)
    coarse = cumulative_trapezoid(rate[::2], times[::2], initial=0)
    coarse_interpolated = np.interp(times, times[::2], coarse)
    discrepancy = norm-norm[0]-integral
    metrics = dict(
        fine_grid_max_norm_drift=float(np.max(np.abs(norm-1))),
        norm_balance_integration_max_abs_discrepancy=float(np.max(np.abs(discrepancy))),
        norm_balance_4001_2001_integral_max_difference=float(np.max(np.abs(integral-coarse_interpolated))),
        norm_residual_projection_max_abs=float(np.max(np.abs(rate))),
        norm_balance_is_continuous_certificate=False)
    table = pd.DataFrame(dict(t=times, norm_predicted=norm, norm_change_from_initial=norm-norm[0],
        norm_rate_from_residual=rate, residual_projected_integral=integral,
        coarse_projected_integral=coarse_interpolated, numerical_integration_discrepancy=discrepancy))
    return metrics, table


def carrier_envelope_decomposition(cfg, times, raw, residual, carrier, unit_defect):
    if carrier.shape != raw.shape or unit_defect.shape != (len(times), cfg.mode_count, 2, 2):
        raise ValueError("Both unit-branch defects on the registered test grid required")
    envelope = branch_projection(raw, cfg)/carrier
    defect_term = np.einsum("tkb,tkbc->tkc", envelope, unit_defect)
    potential_term = potential_apply(raw, cfg)
    envelope_term = residual-defect_term+potential_term
    def projection(term):
        return 2*np.imag(np.sum(raw.conj()*term, axis=(1, 2)))
    defect_rate, potential_rate = projection(defect_term), -projection(potential_term)
    envelope_rate, residual_rate = projection(envelope_term), projection(residual)
    metrics = dict(
        carrier_defect_direct_norm_projection_max_abs=float(np.max(np.abs(defect_rate))),
        hermitian_coupling_global_norm_projection_max_abs=float(np.max(np.abs(potential_rate))),
        envelope_vs_complete_norm_projection_max_difference=float(np.max(np.abs(envelope_rate-residual_rate))),
        carrier_unit_modulus_max_deviation=float(np.max(np.abs(np.abs(carrier)-1))),
        weighted_carrier_defect_spinor_rms=float(np.sqrt(np.mean(np.abs(defect_term)**2))),
        envelope_tangent_spinor_rms=float(np.sqrt(np.mean(np.abs(envelope_term)**2))),
        decomposition_not_a_pretraining_causal_ablation=True)
    table = pd.DataFrame(dict(t=times, residual_norm_rate=residual_rate,
        envelope_tangent_norm_rate=envelope_rate, carrier_defect_norm_rate=defect_rate,
        coupling_global_norm_rate=potential_rate,
        carrier_defect_l2_norm=np.linalg.norm(defect_term, axis=(1, 2)),
        envelope_tangent_l2_norm=np.linalg.norm(envelope_term, axis=(1, 2))))
    return metrics, table


def source_diagnostics(seed):
    import json
    from .r10_config import RAW, source_directory
    from .freeze_r9_wavepacket import selected_config
    from .freeze_r7_phase_release import write_json
    cfg, directory = selected_config(), source_directory(seed)
    output = RAW/"r9_source_diagnostics"/f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    with np.load(directory/"dense_grid.npz") as dense, np.load(directory/"full_physical_residual.npz") as full:
        metrics, balance = norm_balance(cfg, full["t"], full["spectral_prediction_raw_fine"],
                                       full["retained_residual_complex"])
        coarse = full["retained_residual_complex"][::20]
        terms, components = carrier_envelope_decomposition(cfg, dense["t"], dense["spectral_predicted_raw"],
            coarse, dense["learned_carrier"], dense["carrier_unit_spinor_defect"])
    balance.to_csv(output/"norm_balance.csv", index=False)
    components.to_csv(output/"residual_decomposition.csv", index=False)
    record = dict(seed=seed, source_stage="R9 fixed step20000", **metrics, **terms)
    write_json(output/"diagnostics.json", record)
    return record
