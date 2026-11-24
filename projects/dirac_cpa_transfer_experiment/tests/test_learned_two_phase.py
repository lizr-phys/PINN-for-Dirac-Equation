from __future__ import annotations

import torch
import pytest

from src.config import ExperimentConfig
from src.models import LearnedTwoPhaseBranchCPA, build_model, count_parameters
from src.physics import dirac_residual, exact_solution, spinor_derivatives


torch.set_default_dtype(torch.float64)
CFG = ExperimentConfig()


def sample_points(n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(314159)
    x = CFG.x_min + CFG.length * torch.rand((n, 1), generator=generator)
    t = CFG.t_max * torch.rand((n, 1), generator=generator)
    return x, t


def set_exact_branch_outputs(model: LearnedTwoPhaseBranchCPA) -> None:
    for parameter in model.parameters():
        parameter.data.zero_()
    output_layer = model.mlp.net[-1]
    assert isinstance(output_layer, torch.nn.Linear)
    with torch.no_grad():
        output_layer.bias.copy_(
            torch.tensor(
                [0.0, 0.0, -CFG.energy * CFG.t_max, CFG.energy * CFG.t_max],
                dtype=torch.float64,
            )
        )


@pytest.mark.parametrize("eta", [0.0, 1.0])
@torch.no_grad()
def test_ltpb_output_shape_complex_dtype_and_hard_initial_condition(eta: float) -> None:
    model = build_model("ltpb_cpa", CFG, eta=eta).to(dtype=torch.float64)
    assert isinstance(model, LearnedTwoPhaseBranchCPA)
    x, t = sample_points(33)
    predicted = model(x, t)
    initial = model(x, torch.zeros_like(t))
    exact_initial = exact_solution(
        x,
        torch.zeros_like(t),
        mass=CFG.mass,
        wave_number=CFG.wave_number,
        eta=eta,
    )
    assert predicted.shape == (33, 2)
    assert predicted.dtype == torch.complex128
    assert torch.max(torch.abs(initial - exact_initial)).item() < 1.0e-12


@torch.no_grad()
def test_ltpb_forward_does_not_use_stored_spectral_energy() -> None:
    model = build_model("ltpb_cpa", CFG, eta=1.0).to(dtype=torch.float64)
    assert isinstance(model, LearnedTwoPhaseBranchCPA)
    x, t = sample_points(17)
    before = model(x, t)
    model.energy = 12345.0
    after = model(x, t)
    assert torch.equal(before, after)


@torch.no_grad()
def test_ltpb_hard_initial_branch_fields_are_exact() -> None:
    model = LearnedTwoPhaseBranchCPA(CFG, eta=1.0).to(dtype=torch.float64)
    x, _ = sample_points(29)
    zero = torch.zeros_like(x)
    a_plus, a_minus, phase_plus, phase_minus = model.branch_fields(x, zero)
    a_plus_initial, a_minus_initial = model.initial_branch_amplitudes
    assert torch.equal(a_plus, torch.full_like(a_plus, a_plus_initial))
    assert torch.equal(a_minus, torch.full_like(a_minus, a_minus_initial))
    assert torch.equal(phase_plus, torch.zeros_like(phase_plus))
    assert torch.equal(phase_minus, torch.zeros_like(phase_minus))


@torch.no_grad()
def test_ltpb_can_represent_exact_learned_two_phase_solution() -> None:
    model = LearnedTwoPhaseBranchCPA(CFG, eta=1.0).to(dtype=torch.float64)
    set_exact_branch_outputs(model)
    x, t = sample_points(96)
    predicted = model(x, t)
    exact = exact_solution(
        x, t, mass=CFG.mass, wave_number=CFG.wave_number, eta=1.0
    )
    assert torch.max(torch.abs(predicted - exact)).item() < 1.0e-12


def test_ltpb_exact_parameterization_has_machine_precision_dirac_residual() -> None:
    model = LearnedTwoPhaseBranchCPA(CFG, eta=1.0).to(dtype=torch.float64)
    set_exact_branch_outputs(model)
    x, t = sample_points(96)
    x.requires_grad_(True)
    t.requires_grad_(True)
    predicted = model(x, t)
    predicted_x, predicted_t = spinor_derivatives(
        predicted, x, t, create_graph=True
    )
    residual = dirac_residual(
        predicted, predicted_x, predicted_t, mass=CFG.mass
    )
    assert torch.max(torch.abs(residual)).item() < 1.0e-10


def test_ltpb_parameter_count_matches_vanilla_and_oracle() -> None:
    ltpb = build_model("ltpb_cpa", CFG, eta=1.0)
    vanilla = build_model("vanilla", CFG, eta=1.0)
    oracle = build_model("two_branch", CFG, eta=1.0)
    assert count_parameters(ltpb) == count_parameters(vanilla) == count_parameters(oracle)
    assert count_parameters(ltpb) == 12_996
