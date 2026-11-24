from __future__ import annotations

"""Reference only. No reference evolution is imported by learned training code."""
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.linalg import expm
from .r9_config import R9Config, RAW, TRUNCATIONS
from .r9_models import initial_coefficients, compose
from .freeze_r7_phase_release import write_json
import torch


def initial_numpy(cfg):
    k = torch.tensor(cfg.wave_numbers, dtype=torch.float64)
    return compose(initial_coefficients(cfg), k, cfg.mass).numpy()


def build_hamiltonian(cfg):
    nk = cfg.mode_count
    h = np.zeros((2*nk, 2*nk), dtype=np.complex128)
    for j, k in enumerate(cfg.wave_numbers):
        h[2*j:2*j+2, 2*j:2*j+2] = [[cfg.mass, k], [k, -cfg.mass]]
        if j+8 < nk:
            h[2*j:2*j+2, 2*(j+8):2*(j+8)+2] = cfg.potential_strength/2*np.eye(2)
            h[2*(j+8):2*(j+8)+2, 2*j:2*j+2] = cfg.potential_strength/2*np.eye(2)
    return h


def solve(cfg, times):
    h = build_hamiltonian(cfg)
    values, vectors = np.linalg.eigh(h)
    coordinate = vectors.conj().T@initial_numpy(cfg).ravel()
    evolved = np.exp(-1j*np.asarray(times)[:, None]*values)*coordinate
    state = (evolved@vectors.T).reshape(-1, cfg.mode_count, 2)
    dt = ((-1j*values*evolved)@vectors.T).reshape(state.shape)
    return dict(t=np.asarray(times), n=np.asarray(cfg.indices), k=np.asarray(cfg.wave_numbers),
                state=state, dt=dt, hamiltonian=h, eigenvalues=values, eigenvectors=vectors,
                initial=initial_numpy(cfg))


def compare_pair(small_cfg, big_cfg, times):
    a, b = solve(small_cfg, times), solve(big_cfg, times)
    padded = np.zeros_like(b["state"])
    left = small_cfg.n_min-big_cfg.n_min
    padded[:, left:left+small_cfg.mode_count] = a["state"]
    difference = np.linalg.norm(padded-b["state"])/np.linalg.norm(b["state"])
    pop = np.sum(abs(a["state"])**2, axis=2)
    low, high = float(pop[:, :8].sum(1).max()), float(pop[:, -8:].sum(1).max())
    h, initial = a["hamiltonian"], a["initial"].ravel()
    norms = np.sum(abs(a["state"])**2, axis=(1, 2))
    indices = [0, len(times)//4, len(times)//2, len(times)-1]
    errors = [np.linalg.norm(expm(-1j*times[i]*h)@initial-a["state"][i].ravel()) for i in indices]
    reverse = np.linalg.norm(expm(1j*times[-1]*h)@a["state"][-1].ravel()-initial)
    pde = np.linalg.norm(1j*a["dt"].reshape(len(times), -1)-a["state"].reshape(len(times), -1)@h.T)/np.sqrt(len(times))
    row = dict(smaller=[small_cfg.n_min, small_cfg.n_max], larger=[big_cfg.n_min, big_cfg.n_max],
               full_complex_zero_padded_relative_difference=float(difference),
               lower_eight_layers_max_total_occupation=low, upper_eight_layers_max_total_occupation=high,
               hermiticity_max_abs=float(abs(h-h.conj().T).max()), norm_max_drift=float(abs(norms-1).max()),
               reversal_relative_error=float(reverse), independent_expm_max_relative_error=float(max(errors)),
               spectral_pde_residual_l2_rms=float(pde), points=len(times))
    row["convergence_pass"] = bool(difference < 1e-6 and low < 1e-8 and high < 1e-8)
    row["validation_pass"] = bool(row["hermiticity_max_abs"] == 0 and row["norm_max_drift"] < 1e-12
                                  and reverse < 1e-12 and max(errors) < 1e-12 and pde < 1e-12)
    row["pass_check"] = row["convergence_pass"] and row["validation_pass"]
    return row, a, b


def prepare():
    RAW.mkdir(parents=True, exist_ok=True)
    record_path = RAW/"reference_validation.json"
    if record_path.exists():
        import json
        return json.loads(record_path.read_text(encoding="utf-8"))
    times = np.linspace(0, 5, 4001)
    rows, selected = [], None
    for bounds, next_bounds in zip(TRUNCATIONS[:-1], TRUNCATIONS[1:]):
        cfg = R9Config(n_min=bounds[0], n_max=bounds[1])
        big = replace(cfg, n_min=next_bounds[0], n_max=next_bounds[1])
        row, small_solution, big_solution = compare_pair(cfg, big, times)
        rows.append(row)
        np.savez_compressed(RAW/f"reference_n{bounds[0]}_{bounds[1]}.npz", **small_solution)
        np.savez_compressed(RAW/f"reference_n{next_bounds[0]}_{next_bounds[1]}.npz", **big_solution)
        if row["pass_check"]:
            selected = bounds
            break
    record = dict(status="PASS" if selected else "REFERENCE_CALIBRATION_FAILED", comparisons=rows,
                  selected_truncation=selected, selection_before_pinn=True,
                  fixed_initial_support=[0, 32], only_zero_padding=True, no_renormalization=True,
                  boundary_definition="separate total occupation of each outer eight-index block",
                  physical_parameters_changed=False)
    write_json(record_path, record)
    pd.DataFrame(rows).to_csv(RAW/"reference_convergence.csv", index=False)
    if selected:
        cfg = R9Config(n_min=selected[0], n_max=selected[1])
        coupled = solve(cfg, np.linspace(0, 5, 201))
        free = solve(replace(cfg, potential_strength=0), coupled["t"])
        np.savez_compressed(RAW/"reference_test_grid.npz", **coupled)
        np.savez_compressed(RAW/"diagnostic_ignoring_potential.npz", **free)
        from .r9_evaluate import field_metrics
        metrics, _, _ = field_metrics(free["state"], coupled["state"], cfg, coupled["t"])
        write_json(RAW/"diagnostic_ignoring_potential.json", metrics | dict(diagnostic_only=True,
                   potential_measurable=metrics["spinor_rel_l2_gauge_aligned"] > 1e-6,
                   no_learning_or_selection=True))
    return record
