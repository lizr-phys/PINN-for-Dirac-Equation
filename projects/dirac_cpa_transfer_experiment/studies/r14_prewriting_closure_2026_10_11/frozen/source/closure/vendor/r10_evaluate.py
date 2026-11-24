from __future__ import annotations
import numpy as np
from .r9_evaluate import evaluate, spectral_grid
from .r10_norm_diagnostics import norm_balance, carrier_envelope_decomposition


def evaluate_continuation(model, cfg):
    metrics, series, dense, tables = evaluate(model, cfg)
    # Correct exterior signed archival convention before R10's scientific freeze.
    source = dense["exterior_residual_complex"]
    dense["exterior_positive_coupling_source_complex"] = source
    dense["exterior_residual_complex"] = -source
    dense["full_physical_residual_n"] = np.arange(cfg.n_min-8, cfg.n_max+9)
    dense["full_physical_residual_complex"] = np.concatenate(
        (-source[:, :8], dense["residual_complex_unweighted"], -source[:, 8:]), axis=1)
    raw = spectral_grid(model, cfg, dense["residual_t"])
    dense["spectral_prediction_raw_fine"] = raw
    balance_metrics, balance_table = norm_balance(
        cfg, dense["residual_t"], raw, dense["residual_complex_unweighted"])
    decomposition_metrics, decomposition_table = carrier_envelope_decomposition(
        cfg, dense["t"], dense["spectral_predicted_raw"],
        dense["residual_complex_unweighted"][::20], dense["learned_carrier"],
        dense["carrier_unit_spinor_defect"])
    metrics.update(balance_metrics, **decomposition_metrics,
        norm_diagnostic_additional_forward_spinor_points=len(dense["residual_t"])*cfg.mode_count,
        norm_diagnostic_additional_residual_calls=0,
        signed_exterior_archive_corrected_before_r10_freeze=True)
    tables["norm_balance.csv"] = balance_table
    tables["residual_decomposition.csv"] = decomposition_table
    return metrics, series, dense, tables
