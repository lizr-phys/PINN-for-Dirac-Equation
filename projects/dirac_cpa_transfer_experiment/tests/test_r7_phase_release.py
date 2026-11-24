from dataclasses import replace
import inspect
import math
import types

import numpy as np
import pandas as pd
import pytest
import torch

from src.r7_config import R7Config, MODELS, TRAINING_MODES, stage_config
from src.r7_models import build_model
from src.r7_sampling import StratifiedTimeSampler
from src.r7_loss import causal_weights, training_loss, coupled_residual, epsilon_for_step
from src.r7_decision import decide
from src.physics import complex_derivative
from src.periodic_potential_models import count_trainable_parameters, trainable_state_sha256
from src.periodic_potential_physics import central_initial_state, build_coupled_hamiltonian, apply_coupled_hamiltonian_numpy
from src.train_r7_phase_release import recovery_payload, restore_recovery

torch.set_num_threads(1)
CFG = R7Config()


@pytest.mark.parametrize("name", MODELS)
def test_r7_hard_initial_state_parameter_count_and_precision(name):
    model = build_model(name, CFG, 500000)
    k = torch.tensor(CFG.wave_numbers, dtype=torch.float64)
    state = model(k, torch.zeros_like(k))
    assert state.dtype == torch.complex128 and state.shape == (13, 2)
    np.testing.assert_allclose(state.detach(), central_initial_state(CFG), atol=3e-16, rtol=0)
    assert count_trainable_parameters(model) == 12932
    assert all(p.dtype == torch.float64 for p in model.parameters())


def test_r7_identical_initial_weights_and_mode_independent_parameter_count():
    hashes = [trainable_state_sha256(build_model(n, CFG, 500004)) for n in MODELS]
    assert len(set(hashes)) == 1
    for width in (4, 6, 8, 10):
        assert count_trainable_parameters(build_model(MODELS[-1], replace(CFG, ladder_half_width=width), 1)) == 12932


def constant_output(model, values):
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.mlp.net[-1].bias.copy_(torch.tensor(values, dtype=torch.float64))


@pytest.mark.parametrize("name", MODELS)
def test_r7_initial_sideband_tangent_formula(name):
    model = build_model(name, CFG, 11)
    constant_output(model, (0.7, -0.4, 0.9, -0.6))
    k = torch.tensor([1.0, 3.0], dtype=torch.float64)
    t = torch.zeros(2, 1, requires_grad=True, dtype=torch.float64)
    coefficients = model.branch_coefficients(k, t)
    derivative = torch.stack([complex_derivative(coefficients[:, i], t, create_graph=True) for i in range(2)], dim=1)
    r, b = np.array([0.7, -0.4]), np.array([0.9, -0.6])
    expected = (r+1j*b)/5 if name == "complex" else (r/5 if name == "original_ltpb" else r*np.exp(1j*b)/5)
    np.testing.assert_allclose(derivative.detach().numpy(), np.broadcast_to(expected, (2, 2)), atol=2e-16, rtol=0)
    gradients = torch.autograd.grad(derivative.imag.sum(), tuple(model.parameters()), allow_unused=True)
    sensitivity = sum(float((g*g).sum()) for g in gradients if g is not None)
    assert sensitivity == 0 if name == "original_ltpb" else sensitivity > 0


def test_r7_coordinate_jacobian_rank_and_zero_amplitude_degeneracy():
    def mapping(z):
        r, b = z
        return torch.stack((r*torch.cos(b)/5, r*torch.sin(b)/5))
    for r in (0.3, -0.2):
        jac = torch.autograd.functional.jacobian(mapping, torch.tensor([r, 0.8], dtype=torch.float64))
        assert torch.linalg.matrix_rank(jac) == 2
        assert abs(float(torch.linalg.det(jac))-r/25) < 1e-17
    jac0 = torch.autograd.functional.jacobian(mapping, torch.tensor([0.0, 0.8], dtype=torch.float64))
    assert torch.linalg.matrix_rank(jac0) == 1


def test_r7_release_can_represent_required_pure_imaginary_sideband_tangent():
    from src.wavepacket_physics import compute_spectral_basis
    cfg = CFG
    target = -1j*apply_coupled_hamiltonian_numpy(central_initial_state(cfg), np.array(cfg.wave_numbers), cfg.mass, cfg.potential_strength)
    ks = torch.tensor([1.0, 3.0], dtype=torch.float64)
    basis = compute_spectral_basis(ks, cfg.mass)
    target_branch = torch.stack((torch.sum(torch.tensor(target[[5, 7]])*basis.u_plus, dim=1),
                                 torch.sum(torch.tensor(target[[5, 7]])*basis.u_minus, dim=1)), dim=1)
    r = -cfg.t_max*target_branch.imag
    b = torch.full_like(r, -math.pi/2)
    represented = torch.complex(r, torch.zeros_like(r))*torch.exp(1j*b)/cfg.t_max
    torch.testing.assert_close(represented, target_branch, atol=2e-17, rtol=0)


def test_r7_all_occupied_free_regression_equal_outputs_and_derivatives():
    cfg = replace(CFG, potential_strength=0.0)
    original = build_model("original_ltpb", cfg, 17)
    released = build_model("phase_release_ltpb", cfg, 17)
    def occupied(self, k):
        amplitudes = torch.exp(-0.1*(k.reshape(-1)-2)**2)/math.sqrt(2)
        return amplitudes[:, None].expand(-1, 2)
    for model in (original, released):
        model.initial_branch_coefficients = types.MethodType(occupied, model)
    k = torch.linspace(-4, 8, 19, dtype=torch.float64)
    t1 = torch.linspace(0, 5, 19, dtype=torch.float64).requires_grad_()
    t2 = t1.detach().clone().requires_grad_()
    y1, y2 = original(k, t1), released(k, t2)
    torch.testing.assert_close(y1, y2, rtol=0, atol=0)
    for component in range(2):
        torch.testing.assert_close(complex_derivative(y1[:, component], t1, create_graph=True),
                                   complex_derivative(y2[:, component], t2, create_graph=True), rtol=0, atol=0)


def test_r7_sampler_pairing_bin_coverage_and_exact_resume():
    a, b = StratifiedTimeSampler(CFG, 4), StratifiedTimeSampler(CFG, 4)
    for _ in range(3):
        k, t = a.sample()
        torch.randn(71)
        k2, t2 = b.sample()
        assert torch.equal(k, k2) and torch.equal(t, t2)
        times = t.reshape(32, 13)[:, 0].detach().reshape(8, 4)
        low = torch.arange(8)[:, None]*5/8
        assert ((times >= low) & (times < low+5/8)).all()
        assert (torch.diff(times.reshape(-1)) >= 0).all()
    restored = StratifiedTimeSampler(CFG, 0)
    restored.load_state_dict(a.state_dict())
    for _ in range(3):
        assert torch.equal(a.sample()[1], restored.sample()[1])
    assert a.digest() == restored.digest()


def test_r7_short_preserves_gate_scale_and_schedule_is_fixed():
    short = stage_config("short")
    assert short.t_max == 5 and short.training_horizon == 0.5 and short.adam_steps == 1000
    assert short.evaluation_config().t_max == 0.5
    assert [epsilon_for_step(CFG, s) for s in (1, 1000, 1001, 2001, 3001, 4001, 5000)] == [0.01, 0.01, 0.1, 1, 10, 100, 100]


def test_r7_causal_weights_stop_gradient_and_formula():
    losses = torch.tensor([0.2, 0.4, 0.1], dtype=torch.float64, requires_grad=True)
    weights = causal_weights(losses, 2.0, 3.0)
    assert not weights.requires_grad
    torch.testing.assert_close(weights, torch.exp(-3*torch.tensor([0, 0.1, 0.3], dtype=torch.float64)))
    loss = (weights*losses).mean()
    torch.testing.assert_close(torch.autograd.grad(loss, losses)[0], weights/3)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("mode", TRAINING_MODES)
def test_r7_training_residual_gradients_no_reference_leak(name, mode, monkeypatch):
    import src.periodic_potential_physics as physics
    import src.periodic_potential_evaluate as evaluate
    import src.wavepacket_physics as wavepacket
    def forbidden(*args, **kwargs):
        raise AssertionError("Reference accessed in training")
    monkeypatch.setattr(physics, "solve_reference_eigendecomposition", forbidden)
    monkeypatch.setattr(evaluate, "_reference_spectral_state", forbidden)
    monkeypatch.setattr(wavepacket, "reference_energies", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    model = build_model(name, CFG, 3)
    k, t = StratifiedTimeSampler(CFG, 2).sample()
    objective, losses, weights = training_loss(model, CFG, k, t, mode, 1001)
    objective.backward()
    assert torch.isfinite(objective) and losses.shape == weights.shape == (8,)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not any(any(word in key for word in ("energy", "carrier", "eigenvalue")) for key in model.state_dict())
    assert "reference" not in inspect.getsource(training_loss)


def test_r7_complete_coupled_residual_matches_dense_hamiltonian():
    model = build_model("phase_release_ltpb", CFG, 1)
    k, t = StratifiedTimeSampler(CFG, 1).sample()
    residual = coupled_residual(model, CFG, k, t)
    state = model(k, t)
    derivative = torch.stack([complex_derivative(state[:, s], t, create_graph=True) for s in range(2)], dim=1)
    matrix = torch.tensor(build_coupled_hamiltonian(CFG))
    dense = 1j*derivative.reshape(32, 26)-state.reshape(32, 26)@matrix.T
    torch.testing.assert_close(residual.reshape(32, 26), dense, rtol=2e-14, atol=5e-16)
    loss, segments, _ = training_loss(model, CFG, k, t, "ordinary", 1)
    torch.testing.assert_close(loss, torch.sum(torch.abs(residual)**2, dim=1).mean()/CFG.length)


def test_r7_hermitian_unweighted_residual_bound_manufactured_error():
    from scipy.integrate import cumulative_trapezoid
    matrix = build_coupled_hamiltonian(CFG)
    vector = np.linspace(0.1, 0.5, matrix.shape[0]).astype(complex)
    times = np.linspace(0, 5, 4001)
    error = times[:, None]*vector
    residual = 1j*vector[None, :]-error@matrix.T
    integral = cumulative_trapezoid(np.linalg.norm(residual, axis=1), times, initial=0)
    assert np.max(np.linalg.norm(error, axis=1)-integral) < 1e-13


def test_r7_optimizer_rng_exact_recovery():
    cfg = CFG
    model = build_model("phase_release_ltpb", cfg, 23)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sampler = StratifiedTimeSampler(cfg, 1)
    def one_step(m, o, s, step):
        o.zero_grad(set_to_none=True)
        k, t = s.sample()
        training_loss(m, cfg, k, t, "causal", step)[0].backward()
        o.step()
    one_step(model, optimizer, sampler, 1)
    import copy
    saved = copy.deepcopy(recovery_payload(model, optimizer, sampler, cfg, model_name=model.model_name,
                                           mode="causal", seed=1, step=1, history=[], training_seconds=0,
                                           peak_bytes=0, initial_bytes=0))
    one_step(model, optimizer, sampler, 2)
    recovered = build_model("phase_release_ltpb", cfg, 42)
    recovered_optimizer = torch.optim.Adam(recovered.parameters(), lr=1e-3)
    recovered_sampler = StratifiedTimeSampler(cfg, 0)
    restore_recovery(saved, recovered, recovered_optimizer, recovered_sampler)
    one_step(recovered, recovered_optimizer, recovered_sampler, 2)
    assert sampler.digest() == recovered_sampler.digest()
    for key in model.state_dict():
        torch.testing.assert_close(model.state_dict()[key], recovered.state_dict()[key], rtol=0, atol=0)


def synthetic_inventory(success=True):
    rows = []
    for model in MODELS:
        for mode in TRAINING_MODES:
            for seed in range(5):
                error = 0.03 if model == "phase_release_ltpb" and success else 0.4
                rows.append(dict(model=model, training_mode=mode, seed=seed,
                                 spinor_rel_l2_gauge_aligned=error, current_rel_l2=error,
                                 fidelity_mean=0.995 if error < 0.1 else 0.5,
                                 modal_population_rel_l2=error, max_norm_drift=0.001 if error < 0.1 else 0.5))
    return pd.DataFrame(rows)


def test_r7_decision_long_trigger_and_no_success_by_relative_advantage_alone():
    result = decide(synthetic_inventory())
    assert result["long_training_mode_selected"] == "ordinary"
    assert result["comparisons"]["release_vs_complex/ordinary"]["comparative_advantage"]
    failed = decide(synthetic_inventory(False))
    assert failed["long_training_mode_selected"] is None
    assert failed["classification"] == "INCONCLUSIVE_FULL_DYNAMICS"
    with pytest.raises(ValueError):
        decide(synthetic_inventory().iloc[:-1])
