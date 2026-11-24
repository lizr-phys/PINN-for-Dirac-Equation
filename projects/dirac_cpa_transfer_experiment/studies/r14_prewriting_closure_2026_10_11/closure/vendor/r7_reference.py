from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from .periodic_potential_physics import (
    solve_reference_eigendecomposition, compare_reference_truncations,
    reference_solution_checks, calibration_observables, build_coupled_hamiltonian,
)
from .r7_config import R7Config


def reference_pair(cfg, half_width, *, points=8001):
    times = np.linspace(0, cfg.training_horizon, points)
    small = solve_reference_eigendecomposition(cfg, times, half_width=half_width)
    large = solve_reference_eigendecomposition(cfg, times, half_width=half_width+2)
    comparison = compare_reference_truncations(small, large)
    checks = reference_solution_checks(small)
    observables = calibration_observables(small)
    matrix = build_coupled_hamiltonian(cfg, half_width=half_width)
    errors = []
    for index in (0, points//4, points//2, points-1):
        alternative = expm(-1j * times[index] * matrix) @ small.initial_state.ravel()
        errors.append(float(np.linalg.norm(alternative-small.state[index].ravel())))
    gates = {
        "outer_total_occupation": observables["max_outer_layer_total_occupation"] < 1e-8,
        "padded_complex_trajectory": comparison["padded_full_trajectory_spinor_relative_l2"] < 1e-6,
        "hermiticity": checks["hermiticity_max_abs"] <= 2e-15,
        "norm": checks["norm_max_drift"] <= 2e-13,
        "reversibility": checks["time_reversal_relative_l2"] <= 2e-13,
        "expm_agreement": max(errors) < 2e-13,
    }
    return {"half_width": half_width, "comparison_half_width": half_width+2,
            "time_points": points, "time_horizon": cfg.training_horizon,
            "checks": checks, "observables": observables, "comparison": comparison,
            "expm_max_absolute_error": max(errors), "gates": gates,
            "pass": all(gates.values())}, small


def validate_references(root: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "reference_validation.json"
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    prior = root / "results/raw/periodic_potential_coupled_ltpb/formal_selection.json"
    selection = json.loads(prior.read_text(encoding="utf-8"))
    if selection["status"] != "SELECTED" or selection["revised_truncation_pair"] != [6, 8]:
        raise ValueError("R6 authorized selection is inconsistent")
    if selection["selected"] != dict(potential_strength=0.2, t_max=5.0, ladder_half_width=6):
        raise ValueError("R7 must preserve the R6 selected physical problem")
    main, main_solution = reference_pair(R7Config(), 6)
    agreement = abs(main["comparison"]["padded_full_trajectory_spinor_relative_l2"]
                    - selection["truncation_relative_l2"])
    if not main["pass"] or agreement > 1e-12:
        raise ValueError("Main reference validation failed")
    trials, selected_long = [], None
    for half_width in (8, 10, 12):
        cfg = replace(R7Config(), t_max=8.0, training_horizon=8.0, ladder_half_width=half_width)
        entry, solution = reference_pair(cfg, half_width)
        trials.append(entry)
        if entry["pass"]:
            selected_long = half_width
            np.savez_compressed(destination / f"reference_T8_J{half_width}.npz",
                                t=solution.times, k=solution.wave_numbers, state=solution.state,
                                state_t=solution.time_derivative)
            break
    np.savez_compressed(destination / "reference_T5_J6.npz", t=main_solution.times,
                        k=main_solution.wave_numbers, state=main_solution.state,
                        state_t=main_solution.time_derivative)
    result = {"status": "PASS", "reference_only": True,
              "prior_authorization": selection["authorization"], "main": main,
              "prior_truncation_value_absolute_difference": agreement,
              "long_trials": trials, "selected_long_half_width": selected_long,
              "long_reference_pass": selected_long is not None}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return result
