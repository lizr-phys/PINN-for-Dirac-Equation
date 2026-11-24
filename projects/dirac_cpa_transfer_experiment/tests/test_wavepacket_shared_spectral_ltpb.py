from __future__ import annotations

import inspect
import math
from dataclasses import replace

import pytest
import torch

import src.wavepacket_physics as wavepacket_physics
from src.physics import dirac_residual, spinor_derivatives
from src.wavepacket_config import (
    WAVEPACKET_MODELS,
    WavepacketConfig,
    paired_collocation_seed,
    paired_initialization_seed,
)
from src.wavepacket_models import (
    PhysicalComponentPolar,
    PhysicalSinglePhaseCPA,
    PhysicalVanilla,
    PhysicalWavepacketModel,
    SharedSpectralComplex,
    SharedSpectralLTPB,
    SharedSpectralModel,
    SpectralOracle,
    build_wavepacket_model,
    count_trainable_parameters,
    trainable_state_sha256,
)
from src.wavepacket_physics import (
    compute_spectral_basis,
    exact_spectral_state,
    gaussian_spectral_envelope,
    project_energy_branches,
    reconstruct_physical_grid,
    reference_energies,
    spatial_fourier_coefficients,
    spectral_dirac_residual,
    spectral_initial_state,
    training_wave_numbers,
    wavepacket_exact_solution,
    wavepacket_initial_solution,
)


torch.set_default_dtype(torch.float64)
CFG = WavepacketConfig()
EXPECTED_PARAMETERS = {
    "physical_vanilla": 12_996,
    "physical_sp_cpa": 13_061,
    "physical_component_polar": 12_996,
    "shared_spectral_complex": 12_932,
    "shared_spectral_ltpb": 12_932,
}


def periodic_grid(n: int) -> torch.Tensor:
    return CFG.x_min + CFG.length * torch.arange(n).view(-1, 1) / n


def sample_physical_points(n: int = 47) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(314_159)
    x = CFG.x_min + CFG.length * torch.rand((n, 1), generator=generator)
    t = CFG.t_max * torch.rand((n, 1), generator=generator)
    return x, t


def complex_time_derivative(
    state: torch.Tensor,
    t: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    components: list[torch.Tensor] = []
    for component in range(state.shape[1]):
        real = torch.autograd.grad(
            state[:, component].real,
            t,
            grad_outputs=torch.ones_like(state[:, component].real),
            create_graph=create_graph,
            retain_graph=True,
        )[0].reshape(-1)
        imaginary = torch.autograd.grad(
            state[:, component].imag,
            t,
            grad_outputs=torch.ones_like(state[:, component].imag),
            create_graph=create_graph,
            retain_graph=True,
        )[0].reshape(-1)
        components.append(torch.complex(real, imaginary))
    return torch.stack(components, dim=1)


def test_wavepacket_config_matches_preregistered_contract() -> None:
    assert CFG.x_min == -8.0 * math.pi
    assert CFG.x_max == 8.0 * math.pi
    assert CFG.length == 16.0 * math.pi
    assert CFG.mass == 5.0 and CFG.t_max == 5.0
    assert CFG.n_min == 0 and CFG.n_max == 32
    assert CFG.mode_count == 33
    assert CFG.wave_numbers == tuple(index / 8.0 for index in range(33))
    assert CFG.k0 == 2.0 and CFG.sigma_k == 0.5
    assert CFG.hidden_layers == 4 and CFG.hidden_width == 64
    assert CFG.physical_collocation_points == 1024
    assert CFG.spectral_time_points == 32
    assert CFG.spectral_collocation_points == 1056
    assert CFG.adam_steps == 5000 and CFG.learning_rate == 1.0e-3
    assert CFG.test_x_points == 256 and CFG.test_t_points == 201


def test_gaussian_spectrum_and_full_initial_state_are_normalized() -> None:
    k = training_wave_numbers(CFG)
    envelope = gaussian_spectral_envelope(k, CFG)
    initial = spectral_initial_state(k, CFG)
    branches = project_energy_branches(initial, k, CFG.mass)
    expected = (envelope / math.sqrt(2.0))[:, None].expand(-1, 2)
    assert torch.all(envelope > 0.0)
    assert abs(float(torch.sum(envelope**2)) - 1.0) < 2.0e-15
    assert torch.max(torch.abs(branches - expected)).item() < 5.0e-15
    assert abs(float(torch.sum(torch.abs(initial) ** 2)) - 1.0) < 3.0e-15


def test_continuous_real_basis_is_orthonormal_and_diagonalizes_symbol() -> None:
    k = torch.linspace(CFG.wave_number_min, CFG.wave_number_max, 257)
    basis = compute_spectral_basis(k, CFG.mass)
    identity = torch.eye(2).expand(k.numel(), -1, -1)
    stacked = torch.stack((basis.u_plus, basis.u_minus), dim=2)
    gram = torch.einsum("nci,ncj->nij", stacked, stacked)
    hamiltonian = torch.zeros((k.numel(), 2, 2))
    hamiltonian[:, 0, 0] = CFG.mass
    hamiltonian[:, 0, 1] = k
    hamiltonian[:, 1, 0] = k
    hamiltonian[:, 1, 1] = -CFG.mass
    energies = reference_energies(k, CFG.mass)
    assert isinstance(energies, torch.Tensor)
    plus_error = torch.einsum("nij,nj->ni", hamiltonian, basis.u_plus) - (
        energies[:, None] * basis.u_plus
    )
    minus_error = torch.einsum("nij,nj->ni", hamiltonian, basis.u_minus) + (
        energies[:, None] * basis.u_minus
    )
    assert torch.max(torch.abs(gram - identity)).item() < 5.0e-15
    assert torch.max(torch.abs(plus_error)).item() < 4.0e-15
    assert torch.max(torch.abs(minus_error)).item() < 4.0e-15
    assert torch.min(basis.u_plus[:, 0]).item() > 0.0
    assert torch.min(basis.u_minus[:, 1]).item() > 0.0
    assert torch.max(torch.abs(torch.diff(basis.u_plus, dim=0))).item() < 0.01


def test_exact_spectral_state_satisfies_nondiagonal_dirac_residual() -> None:
    generator = torch.Generator(device="cpu").manual_seed(271_828)
    k = CFG.wave_number_max * torch.rand((101, 1), generator=generator)
    t = (CFG.t_max * torch.rand((101, 1), generator=generator)).requires_grad_(True)
    state = exact_spectral_state(k, t, CFG)
    state_t = complex_time_derivative(state, t, create_graph=True)
    residual = spectral_dirac_residual(state, state_t, k, CFG.mass)
    assert state.dtype == torch.complex128
    assert torch.max(torch.abs(residual)).item() < 2.0e-14


def test_exact_physical_wavepacket_satisfies_dirac_pde() -> None:
    x, t = sample_physical_points(89)
    x.requires_grad_(True)
    t.requires_grad_(True)
    exact = wavepacket_exact_solution(x, t, CFG)
    exact_x, exact_t = spinor_derivatives(exact, x, t, create_graph=True)
    residual = dirac_residual(exact, exact_x, exact_t, mass=CFG.mass)
    assert exact.shape == (len(x), 2)
    assert exact.dtype == torch.complex128
    assert torch.max(torch.abs(residual)).item() < 1.0e-10


def test_exact_physical_wavepacket_preserves_total_probability() -> None:
    x = periodic_grid(512)
    for time in torch.linspace(0.0, CFG.t_max, 7):
        exact = wavepacket_exact_solution(x, torch.full_like(x, time), CFG)
        norm = CFG.length * torch.mean(torch.sum(torch.abs(exact) ** 2, dim=1))
        assert abs(float(norm) - 1.0) < 3.0e-13


def test_fourier_round_trip_and_parseval_norm() -> None:
    generator = torch.Generator(device="cpu").manual_seed(161_803)
    k = training_wave_numbers(CFG)
    x = periodic_grid(256)
    coefficients = torch.complex(
        torch.randn((4, CFG.mode_count, 2), generator=generator),
        torch.randn((4, CFG.mode_count, 2), generator=generator),
    )
    physical = reconstruct_physical_grid(coefficients, k, x, length=CFG.length)
    recovered = spatial_fourier_coefficients(
        physical, x, k, length=CFG.length
    )
    physical_norm = CFG.length * torch.mean(
        torch.sum(torch.abs(physical) ** 2, dim=-1), dim=-1
    )
    spectral_norm = torch.sum(torch.abs(coefficients) ** 2, dim=(-2, -1))
    assert torch.max(torch.abs(recovered - coefficients)).item() < 2.0e-13
    assert torch.max(torch.abs(physical_norm - spectral_norm)).item() < 2.0e-12


def test_spectral_and_physical_strong_residual_norms_obey_parseval() -> None:
    generator = torch.Generator(device="cpu").manual_seed(577_215)
    k = training_wave_numbers(CFG)
    x = periodic_grid(256)
    state = torch.complex(
        torch.randn((CFG.mode_count, 2), generator=generator),
        torch.randn((CFG.mode_count, 2), generator=generator),
    )
    state_t = torch.complex(
        torch.randn((CFG.mode_count, 2), generator=generator),
        torch.randn((CFG.mode_count, 2), generator=generator),
    )
    spectral_residual = spectral_dirac_residual(state, state_t, k, CFG.mass)

    psi = reconstruct_physical_grid(state, k, x, length=CFG.length)
    psi_t = reconstruct_physical_grid(state_t, k, x, length=CFG.length)
    psi_x = reconstruct_physical_grid(
        1j * k[:, None] * state, k, x, length=CFG.length
    )
    physical_residual = dirac_residual(psi, psi_x, psi_t, mass=CFG.mass)
    physical_squared_norm = CFG.length * torch.mean(
        torch.sum(torch.abs(physical_residual) ** 2, dim=1)
    )
    spectral_squared_norm = torch.sum(torch.abs(spectral_residual) ** 2)
    assert abs(float(physical_squared_norm - spectral_squared_norm)) < 2.0e-10


@pytest.mark.parametrize("model_name", WAVEPACKET_MODELS)
@torch.no_grad()
def test_all_learned_models_have_hard_initial_conditions_and_complex128(
    model_name: str,
) -> None:
    model = build_wavepacket_model(model_name, CFG, initialization_seed=200_007)
    if model_name.startswith("physical_"):
        x, t = sample_physical_points()
        predicted = model(x, t)
        initial = model(x, torch.zeros_like(t))
        expected = wavepacket_initial_solution(x, CFG)
        left = model(torch.full_like(t, CFG.x_min), t)
        right = model(torch.full_like(t, CFG.x_max), t)
        assert torch.max(torch.abs(left - right)).item() < 2.0e-10
    else:
        k = training_wave_numbers(CFG).view(-1, 1)
        t = torch.linspace(0.0, CFG.t_max, CFG.mode_count).view(-1, 1)
        predicted = model(k, t)
        initial = model(k, torch.zeros_like(t))
        expected = spectral_initial_state(k, CFG)
    assert predicted.shape == expected.shape
    assert predicted.dtype == torch.complex128
    assert torch.max(torch.abs(initial - expected)).item() < 3.0e-13


@torch.no_grad()
def test_shared_models_hard_satisfy_held_out_midpoint_initial_state() -> None:
    training_k = training_wave_numbers(CFG)
    midpoints = 0.5 * (training_k[:-1] + training_k[1:])
    zeros = torch.zeros_like(midpoints)
    expected = spectral_initial_state(midpoints, CFG)
    for model_type in (SharedSpectralComplex, SharedSpectralLTPB):
        model = model_type(CFG, initialization_seed=200_011)
        assert torch.max(torch.abs(model(midpoints, zeros) - expected)).item() < 3.0e-15


def test_physical_features_use_full_domain_period_and_shared_features_are_normalized() -> None:
    physical = PhysicalVanilla(CFG, initialization_seed=200_000)
    zero = torch.zeros((1, 1))
    left = physical.features(torch.tensor([[CFG.x_min]]), zero)
    right = physical.features(torch.tensor([[CFG.x_max]]), zero)
    shifted = physical.features(torch.tensor([[CFG.x_min + 2.0 * math.pi]]), zero)
    assert torch.max(torch.abs(left - right)).item() < 3.0e-15
    assert torch.max(torch.abs(left - shifted)).item() > 0.1

    shared = SharedSpectralLTPB(CFG, initialization_seed=200_000)
    features = shared.features(
        torch.tensor([[CFG.wave_number_min], [CFG.wave_number_max]]),
        torch.tensor([[0.0], [CFG.t_max]]),
    )
    assert torch.equal(features, torch.tensor([[-1.0, 0.0], [1.0, 1.0]]))


def test_model_parameter_counts_are_matched_and_shared_count_is_mode_invariant() -> None:
    observed = {
        name: count_trainable_parameters(
            build_wavepacket_model(name, CFG, initialization_seed=200_000)
        )
        for name in WAVEPACKET_MODELS
    }
    assert observed == EXPECTED_PARAMETERS
    assert (max(observed.values()) - min(observed.values())) / max(
        observed.values()
    ) < 0.02

    for mode_count in (17, 33, 65):
        variant = replace(CFG, n_max=mode_count - 1)
        for model_type in (SharedSpectralComplex, SharedSpectralLTPB):
            model = model_type(variant, initialization_seed=200_003)
            assert count_trainable_parameters(model) == 12_932


def test_shared_spectral_pair_has_identical_initial_weights_and_seed_protocol() -> None:
    seed = 4
    initialization_seed = paired_initialization_seed(seed)
    complex_model = SharedSpectralComplex(CFG, initialization_seed)
    ltpb_model = SharedSpectralLTPB(CFG, initialization_seed)
    complex_state = complex_model.mlp.state_dict()
    ltpb_state = ltpb_model.mlp.state_dict()
    assert complex_state.keys() == ltpb_state.keys()
    assert all(
        torch.equal(complex_state[key], ltpb_state[key]) for key in complex_state
    )
    assert trainable_state_sha256(complex_model) == trainable_state_sha256(ltpb_model)
    assert paired_initialization_seed(seed) == 200_000 + seed
    assert paired_collocation_seed(seed) == 300_000 + seed
    assert paired_initialization_seed(seed) != paired_collocation_seed(seed)


def test_learned_models_and_config_do_not_expose_reference_time_rates() -> None:
    learned_classes = (
        PhysicalWavepacketModel,
        PhysicalVanilla,
        PhysicalSinglePhaseCPA,
        PhysicalComponentPolar,
        SharedSpectralModel,
        SharedSpectralComplex,
        SharedSpectralLTPB,
    )
    forbidden = (
        "reference_energies",
        "exact_spectral_state",
        "wavepacket_exact_solution",
        "target_slope",
        "exact_carrier",
    )
    for cls in learned_classes:
        source = inspect.getsource(cls).lower()
        for token in forbidden:
            assert token not in source, (cls.__name__, token)

    config_keys = {str(key).lower() for key in CFG.as_dict()}
    assert not any(
        token in key
        for key in config_keys
        for token in ("energy", "frequency", "carrier", "phase_slope")
    )
    for name in WAVEPACKET_MODELS:
        model = build_wavepacket_model(name, CFG, initialization_seed=200_002)
        assert not any(
            token in key.lower()
            for key in model.state_dict()
            for token in ("energy", "frequency", "carrier", "phase_slope")
        )


@pytest.mark.parametrize("model_name", WAVEPACKET_MODELS)
def test_learned_forward_and_gradients_do_not_call_reference_dispersion(
    model_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_reference(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("learned code accessed analytic time rates")

    monkeypatch.setattr(wavepacket_physics, "reference_energies", forbidden_reference)
    model = build_wavepacket_model(model_name, CFG, initialization_seed=200_005)
    if model_name.startswith("physical_"):
        first, second = sample_physical_points(13)
    else:
        first = training_wave_numbers(CFG)[:13].view(-1, 1)
        second = torch.linspace(0.0, CFG.t_max, 13).view(-1, 1)
    predicted = model(first, second)
    objective = torch.mean(torch.abs(predicted) ** 2)
    gradients = torch.autograd.grad(objective, tuple(model.parameters()))
    assert torch.isfinite(objective)
    assert gradients and all(torch.all(torch.isfinite(value)) for value in gradients)


@torch.no_grad()
def test_spectral_oracle_is_exact_and_excluded_from_trainable_models() -> None:
    oracle = SpectralOracle(CFG)
    k = training_wave_numbers(CFG).view(-1, 1)
    t = torch.linspace(0.0, CFG.t_max, CFG.mode_count).view(-1, 1)
    assert count_trainable_parameters(oracle) == 0
    assert torch.equal(oracle(k, t), exact_spectral_state(k, t, CFG))
    assert oracle.model_name not in WAVEPACKET_MODELS
