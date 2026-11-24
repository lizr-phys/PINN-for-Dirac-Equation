from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import ExperimentConfig
from src.evaluate import _internal_channel_diagnostics
from src.build_ltpb_mechanism_ablation_report import (
    assert_close as report_assert_close,
    validate_loss_history,
)
from src.audit_ltpb_mechanism_ablation import (
    _close as audit_close,
    _compare_source_values,
    _expected_metric_source,
)
from src.losses import LossComputer
from src.ltpb_mechanism_decision import (
    ABLATION_MODELS,
    ABLATION_SEEDS,
    CORE_METRICS,
    compute_mechanism_decision,
)
from src.models import (
    ComponentPolarCPA,
    RotatedLearnedTwoPhaseBranchCPA,
    TwoChannelDiracModel,
    build_model,
    count_parameters,
)
from src.physics import dirac_residual, exact_solution, spinor_derivatives


torch.set_default_dtype(torch.float64)
CFG = ExperimentConfig(x_min=0.0, x_max=2.0 * math.pi)
MODEL_NAMES = (
    "ltpb_cpa",
    "branch_complex",
    "component_polar",
    "rotated_ltpb_15",
    "rotated_ltpb_30",
)


def sample_points(n: int = 41) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(271828)
    x = CFG.x_min + CFG.length * torch.rand((n, 1), generator=generator)
    t = CFG.t_max * torch.rand((n, 1), generator=generator)
    return x, t


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@torch.no_grad()
def test_ablation_models_have_hard_initial_condition_shape_dtype_and_periodicity(
    model_name: str,
) -> None:
    torch.manual_seed(17)
    model = build_model(model_name, CFG, eta=1.0).to(dtype=torch.float64)
    assert isinstance(model, TwoChannelDiracModel)
    x, t = sample_points()
    predicted = model(x, t)
    initial = model(x, torch.zeros_like(t))
    exact_initial = exact_solution(
        x,
        torch.zeros_like(t),
        mass=CFG.mass,
        wave_number=CFG.wave_number,
        eta=1.0,
    )
    left = model(torch.full_like(t, CFG.x_min), t)
    right = model(torch.full_like(t, CFG.x_max), t)
    assert predicted.shape == (len(x), 2)
    assert predicted.dtype == torch.complex128
    assert torch.max(torch.abs(initial - exact_initial)).item() < 1.0e-12
    assert torch.max(torch.abs(left - right)).item() < 1.0e-10


def test_ablation_models_match_parameter_count_and_paired_initialization() -> None:
    state_by_model: dict[str, dict[str, torch.Tensor]] = {}
    for model_name in MODEL_NAMES:
        torch.manual_seed(123)
        model = build_model(model_name, CFG, eta=1.0)
        assert count_parameters(model) == 12_996
        state_by_model[model_name] = {
            name: value.detach().clone() for name, value in model.mlp.state_dict().items()
        }
    reference = state_by_model["ltpb_cpa"]
    for model_name, state in state_by_model.items():
        assert state.keys() == reference.keys(), model_name
        for name in reference:
            assert torch.equal(state[name], reference[name]), (model_name, name)


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("rotated_ltpb_15", (0.34549414947133544, 0.19947114020071635)),
        ("rotated_ltpb_30", (0.38534865183839900, 0.10325386006452092)),
    ],
)
def test_rotated_basis_is_orthogonal_and_initial_transform_is_exact(
    model_name: str, expected: tuple[float, float]
) -> None:
    model = build_model(model_name, CFG, eta=1.0)
    assert isinstance(model, TwoChannelDiracModel)
    basis = model.channel_basis_matrix
    gram = basis.T @ basis
    assert torch.max(torch.abs(gram - torch.eye(2))).item() < 1.0e-12
    assert torch.all(model.initial_channel_coefficients.abs() > 0.0)
    assert torch.max(
        torch.abs(
            model.initial_channel_coefficients
            - torch.tensor(expected, dtype=torch.float64)
        )
    ).item() < 1.0e-12
    energy_basis = torch.stack((model.u_plus, model.u_minus), dim=1)
    energy_initial = torch.full((2,), 1.0 / math.sqrt(4.0 * math.pi))
    assert torch.max(
        torch.abs(
            basis @ model.initial_channel_coefficients
            - energy_basis @ energy_initial
        )
    ).item() < 1.0e-12


def test_component_polar_is_negative_eigenspinor_rotation_special_case() -> None:
    angle = -0.5 * math.degrees(math.atan2(CFG.wave_number, CFG.mass))
    torch.manual_seed(99)
    component = ComponentPolarCPA(CFG, eta=1.0)
    torch.manual_seed(99)
    rotated = RotatedLearnedTwoPhaseBranchCPA(CFG, eta=1.0, angle_degrees=angle)
    assert torch.max(
        torch.abs(component.channel_basis_matrix - rotated.channel_basis_matrix)
    ).item() < 1.0e-12
    assert torch.max(
        torch.abs(
            component.initial_channel_coefficients
            - rotated.initial_channel_coefficients
        )
    ).item() < 1.0e-12
    x, t = sample_points(19)
    assert torch.max(torch.abs(component(x, t) - rotated(x, t))).item() < 1.0e-12


def test_zero_degree_rotated_model_matches_ltpb_with_same_weights() -> None:
    torch.manual_seed(88)
    ltpb = build_model("ltpb_cpa", CFG, eta=1.0)
    torch.manual_seed(88)
    rotated = RotatedLearnedTwoPhaseBranchCPA(CFG, eta=1.0, angle_degrees=0.0)
    x, t = sample_points(23)
    assert torch.max(torch.abs(ltpb(x, t) - rotated(x, t))).item() < 1.0e-12


def test_branch_complex_output_order_is_real_channels_then_imaginary_channels() -> None:
    model = build_model("branch_complex", CFG, eta=1.0)
    for parameter in model.parameters():
        parameter.data.zero_()
    output = model.mlp.net[-1]
    assert isinstance(output, torch.nn.Linear)
    output.bias.data.copy_(torch.tensor((1.0, 2.0, 3.0, 4.0)))
    x = torch.zeros((1, 1))
    t = torch.full((1, 1), CFG.t_max)
    correction = model.channel_coefficients(x, t) - model.initial_channel_coefficients.to(
        torch.complex128
    )[None, :]
    expected = torch.tensor((1.0 + 3.0j, 2.0 + 4.0j), dtype=torch.complex128)
    assert torch.equal(correction[0], expected)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_ablation_forward_pde_loss_and_gradients_do_not_use_stored_energy(
    model_name: str,
) -> None:
    torch.manual_seed(7)
    model = build_model(model_name, CFG, eta=1.0)
    x, t = sample_points(13)

    def loss_and_gradients() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x_local = x.clone().requires_grad_(True)
        t_local = t.clone().requires_grad_(True)
        psi = model(x_local, t_local)
        psi_x, psi_t = spinor_derivatives(psi, x_local, t_local, create_graph=True)
        residual = dirac_residual(psi, psi_x, psi_t, mass=CFG.mass)
        loss = torch.mean(torch.abs(residual) ** 2)
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        return loss.detach(), tuple(value.detach().clone() for value in gradients)

    before_output = model(x, t).detach().clone()
    before_loss, before_gradients = loss_and_gradients()
    model.energy = 12345.0
    after_output = model(x, t).detach().clone()
    after_loss, after_gradients = loss_and_gradients()
    assert torch.equal(before_output, after_output)
    assert torch.equal(before_loss, after_loss)
    assert all(
        torch.equal(before, after)
        for before, after in zip(before_gradients, after_gradients, strict=True)
    )


def test_pde_only_loss_computer_does_not_access_config_energy() -> None:
    class EnergyForbiddenConfig(ExperimentConfig):
        @property
        def energy(self) -> float:
            raise AssertionError("PDE-only training accessed exact spectral energy")

    cfg = EnergyForbiddenConfig(x_min=0.0, x_max=2.0 * math.pi, collocation_points=11)
    model = build_model("ltpb_cpa", cfg, eta=1.0)
    loss = LossComputer(
        cfg,
        model,
        conservative=False,
        lambda_struct=1.0,
        device=torch.device("cpu"),
        collocation_generator=torch.Generator().manual_seed(101),
        structure_generator=torch.Generator().manual_seed(202),
    )()
    assert torch.isfinite(loss["pde"])
    assert torch.equal(loss["total"], loss["pde"])


def test_collocation_sequence_is_model_independent_for_a_fixed_seed() -> None:
    samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    for model_name in MODEL_NAMES:
        torch.manual_seed(0)
        model = build_model(model_name, CFG, eta=1.0)
        computer = LossComputer(
            CFG,
            model,
            conservative=False,
            lambda_struct=1.0,
            device=torch.device("cpu"),
            collocation_generator=torch.Generator().manual_seed(100_000),
            structure_generator=torch.Generator().manual_seed(200_000),
        )
        x, t = computer._sample_collocation()
        samples.append((x.detach(), t.detach()))
    for (x, t) in samples[1:]:
        assert torch.equal(x, samples[0][0])
        assert torch.equal(t, samples[0][1])


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_exact_native_channel_projection_reconstructs_solution_and_residual(
    model_name: str,
) -> None:
    model = build_model(model_name, CFG, eta=1.0)
    x, t = sample_points(37)
    x.requires_grad_(True)
    t.requires_grad_(True)
    exact = exact_solution(
        x, t, mass=CFG.mass, wave_number=CFG.wave_number, eta=1.0
    )
    demodulated = torch.exp(-1j * CFG.wave_number * x) * exact
    basis = model.channel_basis_matrix.to(torch.complex128)
    coefficients = demodulated @ torch.conj(basis)
    reconstructed = model.compose_channels(x, coefficients)
    psi_x, psi_t = spinor_derivatives(reconstructed, x, t, create_graph=True)
    residual = dirac_residual(reconstructed, psi_x, psi_t, mass=CFG.mass)
    assert torch.max(torch.abs(reconstructed - exact)).item() < 1.0e-12
    assert torch.max(torch.abs(residual)).item() < 1.0e-10


def test_internal_phase_complexity_is_finite_and_linear_carriers_are_simple() -> None:
    t = np.linspace(0.0, 5.0, 201)
    x_count = 5
    initial = np.array((0.3, 0.2))
    coefficients = np.empty((len(t), x_count, 2), dtype=np.complex128)
    coefficients[:, :, 0] = initial[0] * np.exp(-1j * 5.0 * t)[:, None]
    coefficients[:, :, 1] = initial[1] * np.exp(+1j * 5.0 * t)[:, None]
    metrics, fields = _internal_channel_diagnostics(coefficients, t, initial)
    assert metrics["amplitude_drift_rel_rms"] < 1.0e-12
    assert metrics["phase_velocity_fluctuation_weighted_rms"] < 1.0e-10
    assert metrics["phase_curvature_weighted_rms"] < 1.0e-9
    assert metrics["channel_1_phase_valid_link_fraction"] == 1.0
    assert metrics["channel_2_phase_valid_link_fraction"] == 1.0
    assert np.all(np.isfinite(fields["phase_unwrapped"]))


def test_pre_registered_mechanism_decision_is_executable_and_paired() -> None:
    levels = {
        "ltpb_cpa": 1.0,
        "branch_complex": 6.0,
        "component_polar": 5.0,
        "rotated_ltpb_15": 2.0,
        "rotated_ltpb_30": 7.0,
    }
    rows: list[dict[str, float | int | str]] = []
    for model in ABLATION_MODELS:
        for seed in ABLATION_SEEDS:
            row: dict[str, float | int | str] = {
                "model": model,
                "seed": seed,
                "fidelity_mean": 0.99 if model == "ltpb_cpa" else 0.98,
            }
            for metric in CORE_METRICS:
                row[metric] = levels[model] * (1.0 + 0.01 * seed)
            rows.append(row)
    runs = pd.DataFrame(rows)
    grouped = runs.groupby("model", sort=False)
    summary = grouped[[*CORE_METRICS, "fidelity_mean"]].mean().add_suffix(
        "_mean"
    ).reset_index()
    decision = compute_mechanism_decision(runs, summary)
    assert decision["ltpb_dominance_passes"] is True
    assert decision["rotation_trend_passing_metric_count"] == 3
    assert decision["rotation_30_passing_metric_count"] == 3
    assert decision["strong_mechanism_support"] is True
    for metric in CORE_METRICS:
        assert decision["rotation_trend"][metric][
            "paired_seed_increases_0_to_15"
        ] == 3
        assert decision["rotation_trend"][metric][
            "paired_seed_increases_15_to_30"
        ] == 3


def test_audit_recomputes_and_rejects_tampered_source_labels() -> None:
    rows: list[dict[str, float | int | str]] = []
    for model_index, model in enumerate(ABLATION_MODELS):
        for seed in ABLATION_SEEDS:
            row: dict[str, float | int | str] = {
                "model": model,
                "seed": seed,
            }
            for metric_index, metric in enumerate(CORE_METRICS):
                row[metric] = 1.0 + model_index + metric_index + 0.01 * seed
            rows.append(row)

    expected, keys, numeric, text = _expected_metric_source(
        "01_core_error_comparison", pd.DataFrame(rows)
    )
    assert text == ("model_label", "metric_label")
    _compare_source_values(expected, expected, keys, numeric, text)

    tampered = expected.copy()
    tampered.loc[0, "model_label"] = "tampered label"
    with pytest.raises(ValueError, match="source evidence text mismatch: model_label"):
        _compare_source_values(tampered, expected, keys, numeric, text)


def test_report_validator_accepts_a_valid_pde_only_history(tmp_path: Path) -> None:
    steps = np.concatenate(([1], np.arange(10, 5001, 10)))
    pde = np.linspace(1.0, 1.0e-6, len(steps))
    history = pd.DataFrame(
        {
            "step": steps,
            "total": pde,
            "pde": pde,
            "continuity": 0.0,
            "norm": 0.0,
            "energy": 0.0,
            "structure": 0.0,
            "gradient_cosine": np.nan,
            "elapsed_seconds": np.linspace(0.01, 1.0, len(steps)),
        }
    )
    path = tmp_path / "loss_history.csv"
    history.to_csv(path, index=False)
    validate_loss_history(path)


def test_aggregate_validators_accept_matching_structural_nan_only() -> None:
    assert audit_close(float("nan"), float("nan"))
    assert not audit_close(float("nan"), 1.0)
    assert not audit_close(1.0, float("nan"))
    report_assert_close(float("nan"), float("nan"), "structural missingness")
    with pytest.raises(RuntimeError, match="not finite"):
        report_assert_close(float("nan"), 1.0, "unexpected missingness")
