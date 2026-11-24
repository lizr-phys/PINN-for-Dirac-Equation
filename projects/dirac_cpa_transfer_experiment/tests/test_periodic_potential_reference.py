from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from src.calibrate_periodic_potential import (
    STRICT_CONVERGENCE_METRIC,
    calibrate_reference_problem,
    write_calibration_outputs,
)
from src.periodic_potential_config import PeriodicPotentialConfig, ReferenceCalibrationConfig
from src.periodic_potential_physics import (
    apply_coupled_hamiltonian_numpy,
    apply_coupled_hamiltonian_torch,
    build_coupled_hamiltonian,
    central_initial_state,
    compare_reference_truncations,
    energy_basis_numpy,
    reconstruct_physical_numpy,
    reference_solution_checks,
    solve_reference_eigendecomposition,
)


def test_reference_hamiltonian_blocks_hermiticity_initial_norm_and_basis() -> None:
    cfg = PeriodicPotentialConfig()
    h = build_coupled_hamiltonian(cfg)
    np.testing.assert_array_equal(h, h.conj().T)
    for index, k in enumerate(cfg.wave_numbers):
        np.testing.assert_array_equal(h[2*index:2*index+2, 2*index:2*index+2],
                                      [[cfg.mass, k], [k, -cfg.mass]])
        if index + 1 < cfg.mode_count:
            np.testing.assert_array_equal(h[2*index:2*index+2, 2*index+2:2*index+4],
                                          cfg.potential_strength * np.eye(2) / 2)
    initial = central_initial_state(cfg)
    assert abs(np.linalg.norm(initial) - 1) < 3e-16
    assert np.count_nonzero(initial) == 2
    plus, minus = energy_basis_numpy(cfg.wave_numbers, cfg.mass)
    np.testing.assert_allclose(np.sum(plus * minus, axis=1), 0, atol=2e-16)
    np.testing.assert_allclose(np.sum(plus**2, axis=1), 1, atol=3e-16)


@pytest.mark.parametrize("v0,t_max", [(0.1, 5.0), (0.2, 5.0), (0.5, 8.0)])
def test_eigen_reference_matches_independent_matrix_exponential(v0: float, t_max: float) -> None:
    cfg = replace(PeriodicPotentialConfig(), potential_strength=v0, t_max=t_max)
    times = np.array([0.0, 0.17, 1.23, t_max])
    ref = solve_reference_eigendecomposition(cfg, times)
    h = build_coupled_hamiltonian(cfg)
    expected = np.stack([expm(-1j*t*h) @ central_initial_state(cfg).ravel() for t in times])
    np.testing.assert_allclose(ref.state.reshape(len(times), -1), expected, rtol=2e-13, atol=2e-14)
    checks = reference_solution_checks(ref)
    assert checks["norm_max_drift"] < 1e-14
    assert checks["time_reversal_relative_l2"] < 1e-14
    assert checks["spectral_pde_residual_rms"] < 1e-14


def test_torch_coupled_operator_and_parseval_include_exterior_residual() -> None:
    cfg = PeriodicPotentialConfig()
    rng = np.random.default_rng(710)
    a = rng.normal(size=(3, cfg.mode_count, 2)) + 1j*rng.normal(size=(3, cfg.mode_count, 2))
    h = build_coupled_hamiltonian(cfg)
    direct = (a.reshape(3, -1) @ h.T).reshape(a.shape)
    applied = apply_coupled_hamiltonian_numpy(a, np.asarray(cfg.wave_numbers), cfg.mass, cfg.potential_strength)
    np.testing.assert_allclose(direct, applied, atol=1e-14)
    ta = torch.tensor(a, requires_grad=True)
    torch_applied = apply_coupled_hamiltonian_torch(ta, torch.tensor(cfg.wave_numbers), cfg.mass, cfg.potential_strength)
    np.testing.assert_allclose(torch_applied.detach().numpy(), direct, atol=1e-14)
    torch.sum(torch.abs(torch_applied)**2).backward()
    assert torch.isfinite(ta.grad).all()

    x = cfg.x_min + cfg.length * np.arange(256) / 256
    k = np.asarray(cfg.wave_numbers)
    physical = reconstruct_physical_numpy(a, k, x, length=cfg.length)
    np.testing.assert_allclose(cfg.length*np.mean(np.abs(physical)**2, axis=1),
                               np.sum(np.abs(a)**2, axis=1), rtol=2e-14)
    physical_h = reconstruct_physical_numpy(
        apply_coupled_hamiltonian_numpy(a, k, cfg.mass, 0.0), k, x, length=cfg.length
    ) + cfg.potential_strength * np.cos(cfg.coupling_wave_number*x)[None, :, None] * physical
    extended = np.zeros((3, cfg.mode_count + 2, 2), complex)
    extended[:, 1:-1] = direct
    extended[:, 0] = cfg.potential_strength * a[:, 0] / 2
    extended[:, -1] = cfg.potential_strength * a[:, -1] / 2
    extended_k = np.r_[k[0]-cfg.coupling_wave_number, k, k[-1]+cfg.coupling_wave_number]
    np.testing.assert_allclose(physical_h, reconstruct_physical_numpy(extended, extended_k, x, length=cfg.length),
                               atol=2e-14, rtol=2e-13)


def test_zero_padding_cannot_hide_unrepresented_modes() -> None:
    cfg = PeriodicPotentialConfig()
    times = np.linspace(0, 5, 201)
    small = solve_reference_eigendecomposition(cfg, times, half_width=4)
    large = solve_reference_eigendecomposition(cfg, times, half_width=6)
    comparison = compare_reference_truncations(small, large)
    assert comparison["padded_full_trajectory_spinor_relative_l2"] > 5e-6
    assert comparison["common_full_trajectory_spinor_relative_l2"] < comparison["padded_full_trajectory_spinor_relative_l2"]
    assert comparison["common_modal_population_relative_l2"] < 1e-6
    assert STRICT_CONVERGENCE_METRIC == "padded_full_trajectory_spinor_relative_l2"


def test_fixed_calibration_has_no_candidate_and_writer_protects_evidence(tmp_path) -> None:
    protocol = replace(ReferenceCalibrationConfig(), time_points=401)
    payload, refs = calibrate_reference_problem(calibration_cfg=protocol)
    assert payload["horizons_evaluated"] == [5.0, 8.0]
    assert len(payload["candidates"]) == 6
    assert payload["selected"] is None
    assert payload["calibration_status"] == "NO_STRICT_CANDIDATE"
    assert not any(row["candidate_pass"] for row in payload["candidates"])
    destination = tmp_path / "calibration"
    write_calibration_outputs(destination, payload, refs)
    assert len(list(destination.glob("*.npz"))) == 6
    with pytest.raises(FileExistsError):
        write_calibration_outputs(destination, payload, refs)
