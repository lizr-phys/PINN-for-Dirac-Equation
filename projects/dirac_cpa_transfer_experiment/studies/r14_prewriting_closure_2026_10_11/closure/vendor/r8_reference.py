from __future__ import annotations

"""Independent reference and calibration; never imported by learned loss/models."""
from dataclasses import replace
import numpy as np
import torch
from scipy.linalg import expm
from .periodic_potential_physics import (build_coupled_hamiltonian, energy_basis_numpy,
    solve_reference_eigendecomposition, compare_reference_truncations,
    reference_solution_checks, calibration_observables)
from .r8_config import config_for, RAW
from .freeze_r7_phase_release import write_json


def initial_numpy(cfg):
    k = np.asarray(cfg.wave_numbers)
    if cfg.case == "free_packet":
        g = np.exp(-(k-cfg.k0)**2/(4*.5**2))
        g /= np.linalg.norm(g)
    else:
        g = (k == cfg.k0).astype(float)
    up, um = energy_basis_numpy(k, cfg.mass)
    return g[:, None]*(up+um)/np.sqrt(2)


def reference_state(cfg, times):
    matrix = build_coupled_hamiltonian(cfg)
    vals, vecs = np.linalg.eigh(matrix)
    coordinates = vecs.conj().T@initial_numpy(cfg).ravel()
    field = (np.exp(-1j*np.asarray(times)[:, None]*vals)*coordinates)@vecs.T
    return field.reshape(-1, cfg.mode_count, 2)


def reference_pair(cfg, half_width, points=8001):
    times = np.linspace(0, cfg.t_max, points)
    small = solve_reference_eigendecomposition(cfg, times, half_width=half_width)
    large = solve_reference_eigendecomposition(cfg, times, half_width=half_width+2)
    comparison = compare_reference_truncations(small, large)
    checks = reference_solution_checks(small)
    observables = calibration_observables(small)
    matrix = build_coupled_hamiltonian(cfg, half_width=half_width)
    alternatives = [float(np.linalg.norm(expm(-1j*times[i]*matrix)@small.initial_state.ravel()-small.state[i].ravel()))
                    for i in (0, points//4, points//2, points-1)]
    gates = dict(outer_total=observables["max_outer_layer_total_occupation"] < 1e-8,
                 padded_complex=comparison["padded_full_trajectory_spinor_relative_l2"] < 1e-6,
                 hermiticity=checks["hermiticity_max_abs"] <= 2e-15,
                 norm=checks["norm_max_drift"] <= 2e-13,
                 reversibility=checks["time_reversal_relative_l2"] <= 2e-13,
                 expm=max(alternatives) < 2e-13)
    return dict(half_width=half_width, comparison_half_width=half_width+2,
                gates=gates, pass_check=all(gates.values()), checks=checks,
                observables=observables, comparison=comparison, expm_max_abs=max(alternatives)), small


def prepare_references():
    path = RAW/"reference_validation.json"
    if path.exists():
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    cases = {}
    for case in ("main", "T8", "V01", "V04"):
        cfg = config_for("validation", case)
        trials, selected = [], None
        for j in (6, 8, 10, 12, 14):
            entry, solution = reference_pair(replace(cfg, ladder_half_width=j), j)
            trials.append(entry)
            if entry["pass_check"]:
                selected = j
                (RAW/"reference").mkdir(parents=True, exist_ok=True)
                np.savez_compressed(RAW/"reference"/f"{case}_J{j}.npz", t=solution.times,
                                    k=solution.wave_numbers, state=solution.state, state_t=solution.time_derivative)
                break
        if selected is None or (case == "main" and selected != 6):
            raise RuntimeError(f"Reference truncation failed: {case}")
        cases[case] = dict(selected_half_width=selected, trials=trials)
    cfg = config_for("extension", "free_packet")
    times = np.linspace(0, 5, 8001)
    state = reference_state(cfg, times)
    h = build_coupled_hamiltonian(cfg)
    bigger = build_coupled_hamiltonian(replace(cfg, ladder_half_width=18))
    padded_initial = np.pad(initial_numpy(cfg), ((2, 2), (0, 0)))
    vals, vecs = np.linalg.eigh(bigger)
    padded = ((np.exp(-1j*times[:, None]*vals)*(vecs.conj().T@padded_initial.ravel()))@vecs.T).reshape(-1, 37, 2)
    padded_small = np.pad(state, ((0, 0), (2, 2), (0, 0)))
    difference = np.linalg.norm(padded-padded_small)/np.linalg.norm(padded)
    norm_drift = np.max(np.abs(np.sum(abs(state)**2, axis=(1, 2))-1))
    inverse = expm(1j*5*h)@state[-1].ravel()
    reversal = np.linalg.norm(inverse-initial_numpy(cfg).ravel())
    cases["free_packet"] = dict(selected_half_width=16, initial_support="same 33 Gaussian coefficients, zero-padded outside",
                                padded_trajectory_relative_l2=float(difference), max_norm_drift=float(norm_drift),
                                time_reversal_relative_l2=float(reversal), pass_check=bool(difference<1e-12 and norm_drift<2e-13 and reversal<2e-13))
    result = dict(status="PASS", reference_only=True, cases=cases)
    write_json(path, result)
    return result
