"""Independent reference diagnostics; this module never selects learned physics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from .calibrate_periodic_potential import calibrate_reference_problem
from .periodic_potential_config import PeriodicPotentialConfig, ReferenceCalibrationConfig
from .periodic_potential_physics import (
    build_coupled_hamiltonian,
    central_initial_state,
    compare_reference_truncations,
    solve_reference_eigendecomposition,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/raw/periodic_potential_coupled_ltpb/reference_verification.json"


def verify() -> dict:
    cfg = PeriodicPotentialConfig()
    checks = []
    for horizon in (5.0, 8.0):
        for strength in (0.1, 0.2, 0.5):
            sample_times = np.array([0.0, 0.13, 0.71, horizon/2, horizon])
            h = build_coupled_hamiltonian(cfg, half_width=6, potential_strength=strength)
            initial = central_initial_state(cfg, half_width=6).ravel()
            ref = solve_reference_eigendecomposition(cfg, sample_times, half_width=6, potential_strength=strength)
            independent = np.stack([expm(-1j*t*h) @ initial for t in sample_times])
            expm_error = float(np.linalg.norm(independent-ref.state.reshape(len(sample_times), -1))/np.linalg.norm(independent))
            times = np.linspace(0, horizon, 8001)
            ref6 = solve_reference_eigendecomposition(cfg, times, half_width=6, potential_strength=strength)
            ref8 = solve_reference_eigendecomposition(cfg, times, half_width=8, potential_strength=strength)
            checks.append({
                "t_max": horizon, "potential_strength": strength,
                "eigendecomposition_vs_scipy_expm_relative_l2": expm_error,
                "J6_vs_J8_diagnostic_only": compare_reference_truncations(ref6, ref8),
                "not_authorized_as_a_replacement_selection_gate": True,
            })
    denser, _ = calibrate_reference_problem(calibration_cfg=ReferenceCalibrationConfig(time_points=8001))
    result = {
        "purpose": "Verify reference numerics and expose an unselected possible protocol amendment",
        "independent_matrix_exponential_checks": checks,
        "denser_time_grid_calibration": denser,
        "max_expm_relative_l2": max(row["eigendecomposition_vs_scipy_expm_relative_l2"] for row in checks),
        "formal_physics_selected": False,
    }
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = verify()
    print(json.dumps({"max_expm_relative_l2": result["max_expm_relative_l2"],
                      "dense_calibration_status": result["denser_time_grid_calibration"]["calibration_status"]}))
