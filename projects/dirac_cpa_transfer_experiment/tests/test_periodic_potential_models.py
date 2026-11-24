from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from src.periodic_potential_models import (
    PERIODIC_POTENTIAL_MODELS,
    CoupledSharedSpectralComplex,
    CoupledSharedSpectralLTPB,
    PhysicalPotentialVanilla,
    build_periodic_potential_model,
    central_initial_spinor,
    count_trainable_parameters,
    trainable_state_sha256,
)
from src.periodic_potential_config import PeriodicPotentialConfig
from src.wavepacket_physics import compute_spectral_basis


@dataclass(frozen=True)
class _Config:
    x_min: float = -8.0 * math.pi
    x_max: float = 8.0 * math.pi
    t_max: float = 5.0
    mass: float = 5.0
    k0: float = 2.0
    q: float = 1.0
    truncation_j: int = 4
    hidden_layers: int = 4
    hidden_width: int = 64

    @property
    def length(self) -> float:
        return self.x_max - self.x_min

    @property
    def wave_number_min(self) -> float:
        return self.k0 - self.truncation_j * self.q

    @property
    def wave_number_max(self) -> float:
        return self.k0 + self.truncation_j * self.q

    @property
    def wave_numbers(self) -> tuple[float, ...]:
        return tuple(
            self.k0 + index * self.q
            for index in range(-self.truncation_j, self.truncation_j + 1)
        )


def _expected_center_state(cfg: _Config) -> torch.Tensor:
    basis = compute_spectral_basis(
        torch.tensor([cfg.k0], dtype=torch.float64), cfg.mass
    )
    return (basis.u_plus[0] + basis.u_minus[0]).to(torch.complex128) / math.sqrt(2.0)


def test_model_inventory_is_exactly_the_preregistered_three() -> None:
    assert PERIODIC_POTENTIAL_MODELS == (
        "physical_vanilla",
        "coupled_shared_spectral_complex",
        "coupled_shared_spectral_ltpb",
    )


@pytest.mark.parametrize("model_name", PERIODIC_POTENTIAL_MODELS)
def test_builders_return_four_output_float64_networks(model_name: str) -> None:
    model = build_periodic_potential_model(model_name, _Config(), 200_000)
    assert model.output_dim == 4
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())


def test_physical_model_hard_initial_condition_and_normalization() -> None:
    cfg = _Config()
    model = PhysicalPotentialVanilla(cfg, 200_000)
    x = cfg.x_min + cfg.length * torch.arange(256, dtype=torch.float64) / 256
    t = torch.zeros_like(x)
    predicted = model(x, t)
    spinor = _expected_center_state(cfg)
    expected = (
        torch.exp(1j * cfg.k0 * x)[:, None]
        * spinor[None, :]
        / math.sqrt(cfg.length)
    )
    torch.testing.assert_close(predicted, expected, rtol=2e-15, atol=2e-15)
    norm = cfg.length * torch.mean(torch.sum(torch.abs(predicted) ** 2, dim=1))
    torch.testing.assert_close(
        norm,
        torch.tensor(1.0, dtype=torch.float64),
        rtol=2e-15,
        atol=2e-15,
    )
    assert predicted.shape == (256, 2)
    assert predicted.dtype == torch.complex128


@pytest.mark.parametrize(
    "constructor", (CoupledSharedSpectralComplex, CoupledSharedSpectralLTPB)
)
def test_shared_models_hard_initial_condition_including_zero_channels(
    constructor: type[torch.nn.Module],
) -> None:
    cfg = _Config()
    model = constructor(cfg, 200_001)
    k = torch.tensor(cfg.wave_numbers, dtype=torch.float64)
    predicted = model(k, torch.zeros_like(k))
    expected = torch.zeros((len(k), 2), dtype=torch.complex128)
    center = cfg.truncation_j
    expected[center] = _expected_center_state(cfg)
    torch.testing.assert_close(predicted, expected, rtol=2e-15, atol=2e-15)
    assert predicted.dtype == torch.complex128


def test_ltpb_zero_channel_amplitudes_and_all_initial_phases_are_exact() -> None:
    cfg = _Config()
    model = CoupledSharedSpectralLTPB(cfg, 200_002)
    k = torch.tensor(cfg.wave_numbers, dtype=torch.float64)
    amplitudes, phases = model.branch_fields(k, torch.zeros_like(k))
    expected_amplitude = torch.zeros_like(amplitudes)
    expected_amplitude[cfg.truncation_j] = 1.0 / math.sqrt(2.0)
    torch.testing.assert_close(amplitudes, expected_amplitude, rtol=0.0, atol=0.0)
    torch.testing.assert_close(phases, torch.zeros_like(phases), rtol=0.0, atol=0.0)


def test_shared_models_have_identical_parameters_and_initial_weights() -> None:
    cfg = _Config()
    complex_model = CoupledSharedSpectralComplex(cfg, 200_003)
    ltpb_model = CoupledSharedSpectralLTPB(cfg, 200_003)
    assert count_trainable_parameters(complex_model) == 12_932
    assert count_trainable_parameters(ltpb_model) == 12_932
    assert trainable_state_sha256(complex_model) == trainable_state_sha256(ltpb_model)
    assert complex_model.state_dict().keys() == ltpb_model.state_dict().keys()
    for name, tensor in complex_model.state_dict().items():
        torch.testing.assert_close(tensor, ltpb_model.state_dict()[name], rtol=0.0, atol=0.0)


def test_parameter_budget_matches_existing_four_output_baseline() -> None:
    cfg = _Config()
    physical = PhysicalPotentialVanilla(cfg, 200_004)
    spectral = CoupledSharedSpectralLTPB(cfg, 200_004)
    assert count_trainable_parameters(physical) == 12_996
    relative_spread = abs(
        count_trainable_parameters(physical) - count_trainable_parameters(spectral)
    ) / count_trainable_parameters(physical)
    assert relative_spread < 0.02


def test_forward_paths_and_state_have_no_forbidden_solution_objects() -> None:
    classes = (
        PhysicalPotentialVanilla,
        CoupledSharedSpectralComplex,
        CoupledSharedSpectralLTPB,
    )
    forbidden = (
        "reference_energies",
        "exact_spectral_state",
        "matrix_exp",
        "matrix_power",
        "phase_slope_target",
        "time_carrier",
    )
    for cls in classes:
        source = inspect.getsource(cls).lower()
        assert not any(token in source for token in forbidden)
        state_keys = build_periodic_potential_model(
            cls.model_name, _Config(), 200_000
        ).state_dict().keys()
        assert not any(
            token in key.lower()
            for key in state_keys
            for token in ("energy", "frequency", "carrier", "phase_slope")
        )


def test_models_accept_scalar_time_and_have_finite_time_gradients() -> None:
    cfg = _Config()
    k = torch.tensor(cfg.wave_numbers, dtype=torch.float64)
    for constructor in (CoupledSharedSpectralComplex, CoupledSharedSpectralLTPB):
        model = constructor(cfg, 200_004)
        t = torch.tensor([0.37], dtype=torch.float64, requires_grad=True)
        state = model(k, t)
        loss = torch.sum(torch.abs(state) ** 2)
        gradient = torch.autograd.grad(loss, t)[0]
        assert state.shape == (len(k), 2)
        assert state.dtype == torch.complex128
        assert torch.all(torch.isfinite(gradient))


def test_central_initial_spinor_is_normalized() -> None:
    spinor = central_initial_spinor(_Config())
    torch.testing.assert_close(
        torch.sum(torch.abs(spinor) ** 2),
        torch.tensor(1.0, dtype=torch.float64),
        rtol=2e-15,
        atol=2e-15,
    )


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown periodic-potential model"):
        build_periodic_potential_model("remedial_hybrid", _Config(), 200_000)


def test_models_accept_the_formal_periodic_potential_config() -> None:
    cfg = PeriodicPotentialConfig()
    k = torch.as_tensor(cfg.wave_numbers, dtype=torch.float64)
    t = torch.zeros_like(k)
    for model_name in PERIODIC_POTENTIAL_MODELS:
        model = build_periodic_potential_model(model_name, cfg, 500_000)
        if model_name == "physical_vanilla":
            state = model(torch.zeros_like(t), t)
        else:
            state = model(k, t)
        assert state.shape == (cfg.mode_count, 2)
        assert state.dtype == torch.complex128
