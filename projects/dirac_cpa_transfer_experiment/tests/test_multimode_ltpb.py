from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import torch

import src.multimode_models as multimode_models
from src.multimode_config import MULTIMODE_MODELS, MultimodeConfig
from src.multimode_models import (
    FourCarrierOracle,
    MultiBranchComplex,
    MultiLTPB,
    MultimodeComponentPolar,
    MultimodeDiracModel,
    MultimodeSinglePhaseCPA,
    MultimodeVanilla,
    build_multimode_model,
    count_trainable_parameters,
    trainable_state_sha256,
)
from src.multimode_physics import (
    compute_multimode_basis,
    modal_basis_functions,
    multimode_exact_solution,
    multimode_initial_solution,
    project_modal_coefficients,
    reference_energies,
)
from src.physics import dirac_residual, spinor_derivatives


torch.set_default_dtype(torch.float64)
CFG = MultimodeConfig()
EXPECTED_PARAMETERS = {
    "vanilla": 12_996,
    "sp_cpa": 13_061,
    "component_polar": 12_996,
    "multi_branch_complex": 12_860,
    "multi_ltpb": 12_860,
}


def periodic_grid(n: int) -> torch.Tensor:
    return CFG.x_min + CFG.length * torch.arange(n).view(-1, 1) / n


def sample_points(n: int = 47) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(314_159)
    x = CFG.x_min + CFG.length * torch.rand((n, 1), generator=generator)
    t = CFG.t_max * torch.rand((n, 1), generator=generator)
    return x, t


def test_multimode_exact_solution_satisfies_dirac_pde_to_machine_precision() -> None:
    x, t = sample_points(97)
    x.requires_grad_(True)
    t.requires_grad_(True)
    exact = multimode_exact_solution(
        x,
        t,
        mass=CFG.mass,
        wave_numbers=CFG.wave_numbers,
        length=CFG.length,
    )
    exact_x, exact_t = spinor_derivatives(exact, x, t, create_graph=True)
    residual = dirac_residual(exact, exact_x, exact_t, mass=CFG.mass)
    assert torch.max(torch.abs(residual)).item() < 1.0e-10


def test_multimode_exact_solution_is_normalized_for_all_tested_times() -> None:
    x = periodic_grid(512)
    for time in torch.linspace(0.0, CFG.t_max, 11):
        t = torch.full_like(x, time)
        exact = multimode_exact_solution(
            x,
            t,
            mass=CFG.mass,
            wave_numbers=CFG.wave_numbers,
            length=CFG.length,
        )
        norm = CFG.length * torch.mean(torch.sum(torch.abs(exact) ** 2, dim=1))
        assert abs(float(norm) - 1.0) < 2.0e-12


def test_four_spatial_modal_channels_are_orthonormal_on_periodic_grid() -> None:
    x = periodic_grid(128)
    basis = compute_multimode_basis(CFG.mass, CFG.wave_numbers)
    functions = modal_basis_functions(x, basis)
    gram = torch.mean(
        torch.sum(torch.conj(functions[:, :, None, :]) * functions[:, None, :, :], dim=-1),
        dim=0,
    )
    assert functions.shape == (len(x), 4, 2)
    assert torch.max(torch.abs(gram - torch.eye(4, dtype=torch.complex128))).item() < 1.0e-12


def test_modal_projection_recovers_all_four_exact_coefficients() -> None:
    x = periodic_grid(128)
    basis = compute_multimode_basis(CFG.mass, CFG.wave_numbers)
    energies = torch.as_tensor(reference_energies(CFG.mass, CFG.wave_numbers))
    times = torch.tensor((0.0, 0.37, 1.41, CFG.t_max))
    fields = torch.stack(
        [
            multimode_exact_solution(
                x,
                torch.full_like(x, time),
                mass=CFG.mass,
                wave_numbers=CFG.wave_numbers,
                length=CFG.length,
            )
            for time in times
        ],
        dim=0,
    )
    projected = project_modal_coefficients(fields, x, basis)
    slopes = torch.tensor((-energies[0], energies[0], -energies[1], energies[1]))
    expected = CFG.initial_mode_coefficient * torch.exp(1j * times[:, None] * slopes)
    assert projected.shape == (len(times), 4)
    assert torch.max(torch.abs(projected - expected)).item() < 2.0e-12


def test_initial_spinor_components_and_all_modal_coefficients_are_nonzero() -> None:
    x = periodic_grid(4096)
    initial = multimode_initial_solution(
        x,
        mass=CFG.mass,
        wave_numbers=CFG.wave_numbers,
        length=CFG.length,
    )
    basis = compute_multimode_basis(CFG.mass, CFG.wave_numbers)
    coefficients = project_modal_coefficients(initial, x, basis)
    assert torch.min(torch.abs(initial)).item() > 1.0e-3
    assert torch.all(torch.abs(coefficients) > 0.0)
    assert torch.max(
        torch.abs(coefficients - CFG.initial_mode_coefficient)
    ).item() < 2.0e-12


@pytest.mark.parametrize("model_name", MULTIMODE_MODELS)
@torch.no_grad()
def test_all_learned_models_have_hard_initial_condition_shape_dtype_and_periodicity(
    model_name: str,
) -> None:
    model = build_multimode_model(model_name, CFG, initialization_seed=200_007)
    x, t = sample_points()
    predicted = model(x, t)
    initial = model(x, torch.zeros_like(t))
    expected_initial = multimode_initial_solution(
        x,
        mass=CFG.mass,
        wave_numbers=CFG.wave_numbers,
        length=CFG.length,
    )
    left = model(torch.full_like(t, CFG.x_min), t)
    right = model(torch.full_like(t, CFG.x_max), t)
    assert predicted.shape == (len(x), 2)
    assert predicted.dtype == torch.complex128
    assert torch.max(torch.abs(initial - expected_initial)).item() < 1.0e-12
    assert torch.max(torch.abs(left - right)).item() < 1.0e-10


def test_model_parameter_counts_are_exact_and_within_two_percent() -> None:
    observed = {
        name: count_trainable_parameters(
            build_multimode_model(name, CFG, initialization_seed=200_000)
        )
        for name in MULTIMODE_MODELS
    }
    assert observed == EXPECTED_PARAMETERS
    assert (max(observed.values()) - min(observed.values())) / max(observed.values()) < 0.02


def test_paired_initialization_is_deterministic_for_matching_architectures() -> None:
    models = {
        name: build_multimode_model(name, CFG, initialization_seed=200_013)
        for name in MULTIMODE_MODELS
    }
    repeated = {
        name: build_multimode_model(name, CFG, initialization_seed=200_013)
        for name in MULTIMODE_MODELS
    }
    for name in MULTIMODE_MODELS:
        assert trainable_state_sha256(models[name]) == trainable_state_sha256(repeated[name])

    for first, second in (
        ("vanilla", "component_polar"),
        ("multi_branch_complex", "multi_ltpb"),
    ):
        first_state = models[first].mlp.state_dict()
        second_state = models[second].mlp.state_dict()
        assert first_state.keys() == second_state.keys()
        assert all(torch.equal(first_state[key], second_state[key]) for key in first_state)


@torch.no_grad()
def test_multi_ltpb_can_exactly_represent_all_four_analytic_time_carriers() -> None:
    model = MultiLTPB(CFG, initialization_seed=200_000)
    for parameter in model.parameters():
        parameter.zero_()
    output = model.mlp.net[-1]
    assert isinstance(output, torch.nn.Linear)
    energies = torch.as_tensor(reference_energies(CFG.mass, CFG.wave_numbers))
    phase_slopes = torch.tensor((-energies[0], energies[0], -energies[1], energies[1]))
    output.bias[4:].copy_(CFG.t_max * phase_slopes)

    x, t = sample_points(83)
    predicted = model(x, t)
    exact = multimode_exact_solution(
        x,
        t,
        mass=CFG.mass,
        wave_numbers=CFG.wave_numbers,
        length=CFG.length,
    )
    assert torch.max(torch.abs(predicted - exact)).item() < 3.0e-12


@torch.no_grad()
def test_four_carrier_oracle_is_exact_and_has_no_trainable_parameters() -> None:
    oracle = FourCarrierOracle(CFG)
    x, t = sample_points(59)
    predicted = oracle(x, t)
    exact = multimode_exact_solution(
        x,
        t,
        mass=CFG.mass,
        wave_numbers=CFG.wave_numbers,
        length=CFG.length,
    )
    assert predicted.shape == (len(x), 2)
    assert predicted.dtype == torch.complex128
    assert count_trainable_parameters(oracle) == 0
    assert torch.equal(predicted, exact)


def test_learned_forward_source_has_no_exact_frequency_or_reference_carrier() -> None:
    learned_classes = (
        MultimodeDiracModel,
        MultimodeVanilla,
        MultimodeSinglePhaseCPA,
        MultimodeComponentPolar,
        MultiBranchComplex,
        MultiLTPB,
    )
    forbidden = ("multimode_exact_solution", "reference_energies", "exact_carrier")
    for cls in learned_classes:
        source = inspect.getsource(cls).lower()
        for token in forbidden:
            assert token not in source, (cls.__name__, token)

    config_keys = {key.lower() for key in CFG.as_dict()}
    assert not any(
        token in key
        for key in config_keys
        for token in ("energy", "frequency", "carrier", "phase_slope")
    )


@pytest.mark.parametrize("model_name", MULTIMODE_MODELS)
def test_learned_forward_and_gradients_do_not_access_exact_solution(
    model_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = build_multimode_model(model_name, CFG, initialization_seed=200_005)
    x, t = sample_points(17)

    def forbidden_reference(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("learned model accessed analytic time carrier")

    # The oracle owns the only exact-solution reference in this module.  A
    # learned forward, its PDE loss, and its gradients must remain executable
    # if that reference is made unavailable.
    monkeypatch.setattr(multimode_models, "multimode_exact_solution", forbidden_reference)
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    predicted = model(x, t)
    from src.train_multimode_ltpb import multimode_pde_loss

    loss = multimode_pde_loss(model, CFG, x, t)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    assert torch.isfinite(loss)
    assert gradients and all(torch.all(torch.isfinite(value)) for value in gradients)
    assert not any(
        token in key.lower()
        for key in model.state_dict()
        for token in ("energy", "frequency", "carrier", "phase_slope")
    )


def test_collocation_random_stream_is_paired_and_separate() -> None:
    from src.train_multimode_ltpb import (
        PairedCollocationSampler,
        paired_collocation_seed,
        paired_initialization_seed,
    )

    seed = 3
    assert paired_initialization_seed(seed) == 200_000 + seed
    assert paired_collocation_seed(seed) == 300_000 + seed
    batches_by_model: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    for model_name in MULTIMODE_MODELS:
        # Constructing different networks must not alter the explicit sampling
        # stream.  Verify two consecutive batches, not just the first draw.
        build_multimode_model(
            model_name, CFG, initialization_seed=paired_initialization_seed(seed)
        )
        sampler = PairedCollocationSampler(CFG, seed)
        batches_by_model.append([sampler.sample() for _ in range(2)])
        assert sampler.batch_index == 2
    reference = batches_by_model[0]
    for batches in batches_by_model[1:]:
        for (x, t), (x_ref, t_ref) in zip(batches, reference, strict=True):
            assert torch.equal(x, x_ref)
            assert torch.equal(t, t_ref)

    x_other, t_other = PairedCollocationSampler(CFG, seed + 1).sample()
    assert not torch.equal(x_other, reference[0][0])
    assert not torch.equal(t_other, reference[0][1])
    assert paired_initialization_seed(seed) != paired_collocation_seed(seed)
