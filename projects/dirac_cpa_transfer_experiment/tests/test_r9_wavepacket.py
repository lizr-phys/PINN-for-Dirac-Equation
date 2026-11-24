from dataclasses import replace
import inspect
import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from src.r9_config import R9Config, MODELS, SEEDS, ARTIFACTS, RAW
from src.r9_models import R9Model, initial_coefficients, original_gaussian, parameter_counts, tensor_digest
from src.r9_loss import hamiltonian_apply, coupled_residual, training_loss, free_carrier_residual
from src.r9_reference import initial_numpy, build_hamiltonian, solve, compare_pair
from src.r8_loss import derivative
from src.r8_sampling import TimeSampler
from src.r8_models import LearnedCarrier, basis
from src.r9_decision import absolute_success
from src.r9_evaluate import field_metrics, reconstruct
from src.run_r9_wavepacket import validate_masks

torch.set_num_threads(1)
CFG = R9Config(n_min=-48, n_max=80)


def test_shift_eight_coupling_original_spinor_hamiltonian():
    state = torch.zeros((1, CFG.mode_count, 2), dtype=torch.complex128)
    middle = -CFG.n_min+16
    state[0, middle] = torch.tensor([1+2j, 3-4j], dtype=torch.complex128)
    applied = hamiltonian_apply(state, CFG)
    torch.testing.assert_close(applied[0, middle-8], .1*state[0, middle])
    torch.testing.assert_close(applied[0, middle+8], .1*state[0, middle])
    assert torch.equal(applied[0, middle-1], torch.zeros(2, dtype=torch.complex128))
    assert torch.equal(applied[0, middle+1], torch.zeros(2, dtype=torch.complex128))
    np.testing.assert_allclose(applied.numpy().ravel(), build_hamiltonian(CFG)@state.numpy().ravel(), atol=1e-15)


def test_original_gaussian_unchanged_and_zero_padding_no_renormalization():
    from src.wavepacket_physics import gaussian_spectral_envelope, spectral_initial_state
    from src.wavepacket_config import WavepacketConfig
    r5 = WavepacketConfig()
    k = torch.arange(33, dtype=torch.float64)/8
    torch.testing.assert_close(original_gaussian(), gaussian_spectral_envelope(k, r5), atol=0, rtol=0)
    for low, high in ((-32, 64), (-48, 80), (-64, 96), (-80, 112)):
        cfg = R9Config(n_min=low, n_max=high)
        coefficients = initial_coefficients(cfg)
        assert float(coefficients.abs().square().sum()) == pytest.approx(1, abs=5e-16)
        assert torch.count_nonzero(coefficients[: -low]) == 0
        assert torch.count_nonzero(coefficients[33-low:]) == 0
        torch.testing.assert_close(coefficients[-low:33-low, 0], original_gaussian().to(torch.complex128)/np.sqrt(2), atol=0, rtol=0)
        np.testing.assert_allclose(initial_numpy(cfg)[-low:33-low], spectral_initial_state(k, r5).numpy(), atol=2e-16)


def test_reference_predetermined_convergence_and_matrix_checks():
    times = np.linspace(0, 5, 401)
    first, _, _ = compare_pair(R9Config(), CFG, times)
    assert not first["convergence_pass"] and first["validation_pass"]
    second, _, _ = compare_pair(CFG, R9Config(n_min=-64, n_max=96), times)
    assert second["pass_check"]
    assert second["full_complex_zero_padded_relative_difference"] < 1e-6
    assert second["lower_eight_layers_max_total_occupation"] < 1e-8
    assert second["upper_eight_layers_max_total_occupation"] < 1e-8


@pytest.mark.parametrize("name", MODELS)
def test_double_precision_hard_initial_and_finite_gradients(name):
    model = R9Model(CFG, name, 300)
    if model.carrier:
        model.carrier.freeze()
    k = torch.tensor(CFG.wave_numbers, dtype=torch.float64)
    state = model(k, torch.zeros_like(k))
    assert state.dtype == torch.complex128 and state.shape == (129, 2)
    np.testing.assert_allclose(state.detach().numpy(), initial_numpy(CFG), atol=2e-16)
    assert all(p.dtype == torch.float64 for p in model.parameters())
    k, t = TimeSampler(CFG, 300).sample()
    loss, power = training_loss(model, CFG, k, t)
    assert power.shape == (129,) and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.networks.parameters())


def test_parameter_capacity_matches_and_only_same_shape_weights_are_paired():
    a, b, c = [R9Model(CFG, name, 300) for name in MODELS]
    a.carrier.freeze()
    assert parameter_counts(a) == dict(total_parameters=25734, active_parameters=12932, frozen_parameters=12802,
                                     carrier_parameters=12802, complex_network_parameters=12932)
    assert parameter_counts(b)["total_parameters"] == 12932
    assert parameter_counts(c)["total_parameters"] == 25864
    assert abs(25864/25734-1) < .0051
    assert tensor_digest(a.networks[0].parameters()) == tensor_digest(b.networks[0].parameters())
    assert tensor_digest(b.networks[0].parameters()) == tensor_digest(c.networks[0].parameters())
    assert tensor_digest(c.networks[0].parameters()) != tensor_digest(c.networks[1].parameters())
    assert tensor_digest(a.carrier.parameters()) == tensor_digest(LearnedCarrier(CFG, 300).parameters())


def test_frozen_carrier_time_derivative_and_defect_retained():
    a = R9Model(CFG, MODELS[0], 300)
    with torch.no_grad():
        for p in a.parameters():
            p.zero_()
        a.carrier.mlp.net[-1].bias.copy_(torch.tensor([-7.3, 4.1], dtype=torch.float64))
    a.carrier.freeze()
    k, t = TimeSampler(CFG, 300).sample()
    carrier = a.carrier(k, t)
    dt = derivative(carrier, t)
    torch.testing.assert_close(dt, 1j*torch.tensor([-7.3, 4.1], dtype=torch.float64)[None]/5*carrier, atol=2e-15, rtol=0)
    residual = coupled_residual(a, CFG, k, t)
    assert residual.abs().max() > .1
    residual.abs().square().mean().backward()
    assert all(p.grad is None for p in a.carrier.parameters())
    assert all(p.grad is not None for p in a.networks.parameters())


def test_carrier_separate_nonzero_unit_initials_on_all_retained_modes():
    carrier = LearnedCarrier(CFG, 300)
    k = torch.tensor(CFG.wave_numbers, dtype=torch.float64)
    p = carrier(k, torch.zeros_like(k))
    torch.testing.assert_close(p, torch.ones((129, 2), dtype=torch.complex128), atol=0, rtol=0)
    up, um = basis(k, CFG.mass)
    torch.testing.assert_close(up.square().sum(1), torch.ones(129, dtype=torch.float64))
    torch.testing.assert_close(um.square().sum(1), torch.ones(129, dtype=torch.float64))


@pytest.mark.parametrize("name", MODELS)
def test_exact_chain_rule_matches_autograd_values_derivatives_and_parameter_gradients(name):
    from src.r9_time_derivative import spinor_value_and_time_derivative, unit_carrier_spinor_residual
    from src.r8_loss import free_carrier_residual as original_free_residual
    model = R9Model(CFG, name, 300)
    k, t = TimeSampler(CFG, 300).sample()
    k, t = k[:129], t[:129].detach().clone().requires_grad_()
    if model.carrier:
        original_free = original_free_residual(model.carrier, CFG, k, t)
        fast_free = unit_carrier_spinor_residual(model.carrier, CFG, k, t)
        torch.testing.assert_close(fast_free, original_free, atol=3e-14, rtol=3e-14)
        model.carrier.freeze()
    state = model(k, t)
    dt = derivative(state, t)
    fast_state, fast_dt = spinor_value_and_time_derivative(model, CFG, k, t)
    torch.testing.assert_close(fast_state, state, atol=0, rtol=0)
    torch.testing.assert_close(fast_dt, dt, atol=2e-14, rtol=2e-14)
    slow_residual = 1j*dt.reshape(1, 129, 2)-hamiltonian_apply(state.reshape(1, 129, 2), CFG)
    fast_residual = coupled_residual(model, CFG, k, t)
    params = tuple(model.networks.parameters())
    slow_grad = torch.autograd.grad(slow_residual.abs().square().mean(), params, retain_graph=True)
    fast_grad = torch.autograd.grad(fast_residual.abs().square().mean(), params)
    for a, b in zip(slow_grad, fast_grad):
        torch.testing.assert_close(a, b, atol=1e-13, rtol=1e-12)


def test_exact_reference_autograd_full_original_residual():
    class TestOnlyReference(nn.Module):
        def __init__(self):
            super().__init__()
            vals, vec = torch.linalg.eigh(torch.tensor(build_hamiltonian(CFG)))
            self.vals, self.vec, self.coord = vals, vec, vec.conj().T@torch.tensor(initial_numpy(CFG)).reshape(-1)
        def forward(self, k, t):
            all_states = ((torch.exp(-1j*t*self.vals[None])*self.coord[None])@self.vec.T).reshape(-1, CFG.mode_count, 2)
            index = torch.round(8*k.reshape(-1)).long()-CFG.n_min
            return all_states[torch.arange(len(k)), index]
    # Three times are sufficient to check the full derivative and all mode couplings.
    ks = torch.tensor(CFG.wave_numbers, dtype=torch.float64).reshape(-1, 1).repeat(3, 1)
    t = torch.tensor([0., 1.3, 5.], dtype=torch.float64).repeat_interleave(129).reshape(-1, 1).requires_grad_()
    assert coupled_residual(TestOnlyReference(), CFG, ks, t, False).abs().max() < 1e-13


def test_no_frequency_reference_dependency_in_learned_execution_or_state(monkeypatch):
    import src.r9_models as models
    import src.r9_loss as loss
    import src.r9_reference as ref
    for module in (models, loss):
        source = inspect.getsource(module)
        assert "np.linalg.eigh" not in source and "matrix_exp" not in source
        assert "r9_reference" not in source and "r9_evaluate" not in source
        assert "sqrt(k**2" not in source and "sqrt(k*k" not in source
    monkeypatch.setattr(ref, "solve", lambda *args: (_ for _ in ()).throw(AssertionError("Reference leakage")))
    a = R9Model(CFG, MODELS[0], 300)
    k, t = TimeSampler(CFG, 300).sample()
    assert torch.isfinite(free_carrier_residual(a.carrier, CFG, k, t)).all()
    a.carrier.freeze()
    objective, _ = training_loss(a, CFG, k, t)
    objective.backward()
    for key in list(a.state_dict())+list(CFG.as_dict()):
        assert not any(word in key for word in ("energy", "frequency", "eigenvalue", "target", "reference"))


def test_random_streams_separate_common_prefix_and_exact_restore():
    a, b, pre = TimeSampler(CFG, 300), TimeSampler(CFG, 300), TimeSampler(CFG, 300, carrier=True)
    for _ in range(5):
        torch.randn(37)
        ka, ta = a.sample()
        R9Model(CFG, MODELS[2], 301)
        pre.sample()
        kb, tb = b.sample()
        torch.testing.assert_close(ta, tb, atol=0, rtol=0)
        assert torch.equal(ka, kb)
    assert a.digest() == b.digest() != pre.digest()
    restored = TimeSampler(CFG, 999)
    restored.load_state_dict(a.state_dict())
    torch.testing.assert_close(a.sample()[1], restored.sample()[1], atol=0, rtol=0)


def test_fourier_integration_aliasing_parseval_and_bilinear_errors():
    rng = np.random.default_rng(31)
    state = rng.normal(size=(3, 129, 2))+1j*rng.normal(size=(3, 129, 2))
    x = np.linspace(CFG.x_min, CFG.x_max, 1024, endpoint=False)
    psi = reconstruct(state, CFG, x)
    np.testing.assert_allclose(CFG.length*np.sum(abs(psi)**2, (1, 2))/1024, np.sum(abs(state)**2, (1, 2)), rtol=1e-14)
    psi2 = reconstruct(state, CFG, np.linspace(CFG.x_min, CFG.x_max, 2048, endpoint=False))
    rho, rho2 = np.sum(abs(psi)**2, -1), np.sum(abs(psi2)**2, -1)
    np.testing.assert_allclose(np.mean(rho**2, 1), np.mean(rho2**2, 1), rtol=1e-14)
    assert 1024 > 2*(CFG.n_max-CFG.n_min)


def test_single_constant_gauge_and_collapse_not_excluded_from_coherence():
    cfg = replace(CFG, test_x_points=64)
    times = np.linspace(0, 5, 11)
    exact = solve(cfg, times)["state"]
    m, _, d = field_metrics(exact*np.exp(.7j), exact, cfg, times)
    assert m["spinor_rel_l2_gauge_aligned"] < 1e-14
    varying = exact*np.exp(1j*times[:, None, None])
    m, _, _ = field_metrics(varying, exact, cfg, times)
    assert m["spinor_rel_l2_gauge_aligned"] > .5
    collapsed = np.zeros_like(exact)
    m, _, d = field_metrics(collapsed, exact, cfg, times)
    assert m["active_coherence_rel_l2"] == pytest.approx(1, abs=2e-15)
    assert m["phase_defined_reference_active_coverage"] == 0
    assert d["phase_reference_active_mask"].any()
    assert np.array_equal(d["initial_support_mask"], (np.asarray(cfg.indices)>=0)&(np.asarray(cfg.indices)<=32))


def test_success_uses_all_five_means_and_four_joint_seeds():
    frame = pd.DataFrame([dict(seed=s, spinor_rel_l2_gauge_aligned=.05, current_rel_l2=.06,
        fidelity_mean=.99, modal_population_rel_l2=.01, max_norm_drift=.01) for s in SEEDS])
    assert absolute_success(frame)["pass_check"]
    frame.loc[0, "max_norm_drift"] = .025
    assert absolute_success(frame)["pass_check"]
    frame.loc[1, "max_norm_drift"] = .025
    assert not absolute_success(frame)["pass_check"]
    assert not absolute_success(frame.iloc[:4])["complete"]


def test_cpu_slot_allocations_must_not_overlap():
    assert validate_masks([1, 2, 4, 8, 16])
    with pytest.raises(RuntimeError):
        validate_masks([1, 2, 1])
    with pytest.raises(RuntimeError):
        validate_masks([3, 4])


@pytest.mark.parametrize("name", MODELS)
def test_exact_optimizer_sampler_recovery_and_smoke_six_artifacts(tmp_path, name):
    from src.train_r9_wavepacket import train_run
    cfg = replace(R9Config(), adam_steps=4, carrier_steps=2, history_every=1, recovery_every=2)
    whole, resumed = tmp_path/"whole", tmp_path/"resumed"
    train_run(cfg, name, 299, whole, smoke=True)
    train_run(cfg, name, 299, resumed, smoke=True, stop_after=2)
    train_run(cfg, name, 299, resumed, smoke=True)
    a = torch.load(whole/"model_state.pt", weights_only=False)
    b = torch.load(resumed/"model_state.pt", weights_only=False)
    assert a["coupled_time_stream_sha256"] == b["coupled_time_stream_sha256"]
    for key in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], atol=0, rtol=0)
    assert all((resumed/p).is_file() for p in ARTIFACTS+("recovery_state.pt",))
    assert not any(word in key for key in a for word in ("frequency", "energy", "reference", "target"))


def test_formal_artifact_completeness_if_training_exists():
    directories = list((RAW/"formal").glob("*/seed_*"))
    if not directories:
        pytest.skip("Formal training has not started")
    if not all((p/"metrics.json").exists() for p in directories) or len(directories) < 15:
        pytest.skip("Formal runs still active; final audit enforces completion")
    assert len(directories) == 15
    for directory in directories:
        assert all((directory/p).is_file() for p in ARTIFACTS+("recovery_state.pt", "residual_by_mode.csv", "residual_integral.csv"))
        metrics = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
        assert metrics["steps"] == metrics["fixed_final_step"] == 20000
