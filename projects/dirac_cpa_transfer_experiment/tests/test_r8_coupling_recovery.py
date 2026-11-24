from dataclasses import replace
import inspect
import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from src.r8_config import (R8Config, config_for, allocation, CANDIDATES, CONTROLS, VALIDATION_SEEDS)
from src.r8_models import (R8Model, LearnedCarrier, LocalField, basis, initial_coefficients,
                           compose, parameter_counts)
from src.r8_sampling import TimeSampler
from src.r8_loss import coupled_residual, free_carrier_residual, derivative, training_loss, ResidualBalance
from src.r8_reference import reference_state, initial_numpy, reference_pair
from src.r8_decision import absolute_success, score
from src.periodic_potential_physics import build_coupled_hamiltonian, reconstruct_physical_numpy
from src.periodic_potential_models import trainable_state_sha256

torch.set_num_threads(1)
CFG = R8Config()


@pytest.mark.parametrize("name", CANDIDATES+CONTROLS)
def test_all_hard_initial_state_type_and_same_envelope(name):
    model = R8Model(CFG, name, 0)
    k = torch.tensor(CFG.wave_numbers, dtype=torch.float64)
    psi = model(k, torch.zeros_like(k))
    assert psi.dtype == torch.complex128 and psi.shape == (13, 2)
    np.testing.assert_allclose(psi.detach(), initial_numpy(CFG), atol=4e-16, rtol=0)
    assert parameter_counts(model)["envelope_parameter_count_per_window"] == 12932
    assert trainable_state_sha256(model.fields[0]) == trainable_state_sha256(R8Model(CFG, "complex", 0).fields[0])


def test_unit_nonzero_carrier_initials_both_branches_all_modes():
    carrier = LearnedCarrier(CFG, 1)
    k = torch.tensor(CFG.wave_numbers, dtype=torch.float64)
    p = carrier(k, torch.zeros_like(k))
    torch.testing.assert_close(p, torch.ones((13, 2), dtype=torch.complex128), atol=0, rtol=0)
    up, um = basis(k, CFG.mass)
    torch.testing.assert_close(up.square().sum(1), torch.ones(13, dtype=torch.float64))
    torch.testing.assert_close(um.square().sum(1), torch.ones(13, dtype=torch.float64))
    assert (p.abs() == 1).all()


def test_frozen_carrier_time_derivative_and_full_defect_are_retained():
    model = R8Model(CFG, "A_carrier_complex", 0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.carrier.mlp.net[-1].bias.copy_(torch.tensor([-7.3, 4.1], dtype=torch.float64))
    model.carrier.freeze()
    sampler = TimeSampler(CFG, 2)
    k, t = sampler.sample()
    p = model.carrier(k, t)
    dt = derivative(p, t)
    expected = 1j*torch.tensor([-7.3, 4.1], dtype=torch.float64)[None]/5*p
    torch.testing.assert_close(dt, expected, atol=2e-15, rtol=0)
    assert all(not v.requires_grad for v in model.carrier.parameters())
    residual = coupled_residual(model, CFG, k, t)
    assert residual.abs().max() > 1
    objective = residual.abs().square().mean()
    objective.backward()
    assert all(p.grad is None for p in model.carrier.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.active_field.parameters())


@pytest.mark.parametrize("name", ("A_carrier_complex", "B_free_correction", "complex", "phase_release"))
def test_empty_sideband_has_available_imaginary_tangent_gradient(name):
    model = R8Model(CFG, name, 1)
    if model.carrier is not None:
        model.carrier.freeze()
    k = torch.tensor([1., 3.], dtype=torch.float64)
    t = torch.zeros((2, 1), dtype=torch.float64, requires_grad=True)
    dt = derivative(model(k, t), t)
    grad = torch.autograd.grad(dt.imag.sum(), tuple(model.fields[0].parameters()), allow_unused=True)
    assert sum(float((g*g).sum()) for g in grad if g is not None) > 0


@pytest.mark.parametrize("name", ("C_release_windows", "D_carrier_windows", "window_complex"))
def test_complex_predicted_handoff_is_hard_and_no_reference_reset(name):
    model = R8Model(CFG, name, 3)
    if model.carrier is not None:
        model.carrier.freeze()
    k = torch.tensor(CFG.wave_numbers, dtype=torch.float64)
    t = torch.full_like(k, .5)
    before = model(k, t).detach()
    copied = model.add_window()
    after = model(k, t).detach()
    torch.testing.assert_close(after, before, atol=8e-16, rtol=0)
    assert not copied.requires_grad
    assert (copied.imag.abs() > 1e-5).any()
    assert not any(p.requires_grad for p in model.fields[0].parameters())
    assert all(p.requires_grad for p in model.fields[1].parameters())
    assert len(model.fields) == 2


def test_release_window_complex_initial_and_exact_empty_support():
    initial = initial_coefficients(CFG).clone()
    initial[5] = torch.tensor([.3+.2j, -.4+.7j], dtype=torch.complex128)
    field = LocalField(CFG, 0, "release", initial, start=1.5, duration=.5)
    k = torch.tensor(CFG.wave_numbers, dtype=torch.float64)
    c = field.branch_coefficients(k, torch.full_like(k, 1.5))
    torch.testing.assert_close(c, initial, atol=4e-16, rtol=0)
    assert torch.equal(field.initial_occupied, initial.abs() != 0)


def test_reference_full_pde_hermiticity_norm_reversibility_and_convergence():
    entry, solution = reference_pair(CFG, 6, points=401)
    assert entry["pass_check"]
    assert entry["checks"]["spectral_pde_residual_rms"] < 2e-14
    assert entry["checks"]["norm_max_drift"] < 2e-13
    assert entry["comparison"]["padded_full_trajectory_spinor_relative_l2"] < 1e-6


def test_autograd_full_coupled_residual_machine_precision_on_manufactured_reference():
    class AnalyticTestOnly(nn.Module):
        def __init__(self):
            super().__init__()
            h = torch.tensor(build_coupled_hamiltonian(CFG))
            self.vals, self.vecs = torch.linalg.eigh(h)
            self.coord = self.vecs.conj().T@torch.tensor(initial_numpy(CFG), dtype=torch.complex128).reshape(-1)
        def forward(self, k, t):
            whole = (torch.exp(-1j*t*self.vals[None])*self.coord[None])@self.vecs.T
            ix = torch.round(k.reshape(-1)-CFG.wave_numbers[0]).long()
            whole = whole.reshape(-1, 13, 2)
            return whole[torch.arange(len(k)), ix]
    k, t = TimeSampler(CFG, 5).sample()
    residual = coupled_residual(AnalyticTestOnly(), CFG, k, t, False)
    assert residual.abs().max() < 3e-14


def test_free_unit_carrier_residual_is_original_spinor_equation():
    class CarrierTestOnly(nn.Module):
        def forward(self, k, t):
            w = torch.sqrt(k*k+CFG.mass**2)
            return torch.cat((torch.exp(-1j*w*t), torch.exp(1j*w*t)), 1)
    k, t = TimeSampler(CFG, 1).sample()
    res = free_carrier_residual(CarrierTestOnly(), CFG, k, t, False)
    assert res.abs().max() < 3e-15


def test_parseval_full_residual_including_exterior():
    rng = np.random.default_rng(9)
    r = rng.normal(size=(3, 15, 2))+1j*rng.normal(size=(3, 15, 2))
    k = np.arange(-5, 10)
    x = np.linspace(-8*np.pi, 8*np.pi, 256, endpoint=False)
    field = reconstruct_physical_numpy(r, k, x, length=CFG.length)
    assert np.mean(abs(field)**2) == pytest.approx(np.mean(np.sum(abs(r)**2, axis=1))/CFG.length, rel=1e-14)


def test_paired_streams_independent_model_rng_and_window_mappings():
    a, b = TimeSampler(CFG, 0), TimeSampler(CFG, 0)
    for _ in range(5):
        torch.randn(111)
        k1, t1 = a.sample()
        k2, t2 = b.sample(1., .5)
        assert torch.equal(k1, k2)
        torch.testing.assert_close((t2-1.)/.5, t1/5, atol=5e-16, rtol=0)
    assert a.digest(True) == b.digest(True)
    assert a.digest() != b.digest()
    c = TimeSampler(CFG, 999)
    c.load_state_dict(a.state_dict())
    torch.testing.assert_close(a.sample()[1], c.sample()[1], atol=0, rtol=0)
    assert TimeSampler(CFG, 0, carrier=True).generator.initial_seed() != TimeSampler(CFG, 0).generator.initial_seed()


def test_total_budgets_include_pretraining_all_windows_and_parameters():
    for stage, budget, pre in (("exploration", 10000, 3000), ("validation", 20000, 6000)):
        cfg = config_for(stage)
        for name in CANDIDATES+CONTROLS:
            p, n, length = allocation(cfg, name)
            assert p+n*length == budget
            if name in ("A_carrier_complex", "B_free_correction", "D_carrier_windows"):
                assert p == pre
    counts = []
    for half in (4, 6, 8):
        counts.append(parameter_counts(R8Model(replace(CFG, ladder_half_width=half), "A_carrier_complex", 0))["parameter_count"])
    assert counts == [25734]*3


def test_no_reference_or_exact_frequency_in_learned_modules_or_configuration():
    import src.r8_models as models
    import src.r8_loss as loss
    import src.r8_sampling as sampling
    import src.r8_config as config
    for module in (models, loss, sampling, config):
        text = inspect.getsource(module).lower()
        assert "np.linalg.eigh" not in text and "matrix_exp" not in text and "reference_state" not in text
        assert "sqrt(k*k" not in text and "sqrt(k**2" not in text and "spectral_energy" not in text
    state = R8Model(CFG, "A_carrier_complex", 0).state_dict()
    assert not any("frequency" in k or "eigenvalue" in k or "target" in k or "reference" in k for k in state)
    assert not any("energy" in k or "frequency" in k for k in CFG.as_dict())


def test_success_requires_all_means_and_four_independent_joint_passes():
    frame = pd.DataFrame([dict(seed=s, spinor_rel_l2_gauge_aligned=.05, current_rel_l2=.05,
        fidelity_mean=.99, modal_population_rel_l2=.05, max_norm_drift=.01,
        sideband_complex_coefficient_rel_l2=.07) for s in VALIDATION_SEEDS])
    assert absolute_success(frame)["pass_check"]
    frame.loc[:1, "max_norm_drift"] = .021
    assert not absolute_success(frame)["pass_check"]
    assert not absolute_success(frame.iloc[:4])["complete"]
    assert score(frame) == pytest.approx((.05/.10+.05/.15+.07/.10)/3)


def test_rescue_only_changes_detached_weights_and_keeps_unweighted_diagnostics():
    model = R8Model(CFG, "complex", 0)
    k, t = TimeSampler(CFG, 0).sample()
    balance = ResidualBalance(CFG.mode_count)
    loss, power, weights = training_loss(model, CFG, k, t, balance)
    assert not weights.requires_grad and weights.mean() == pytest.approx(1.)
    assert power.shape == (13,)
    loss.backward()
    assert all(p.grad is not None for p in model.active_field.parameters())


def test_free_packet_normalization_and_hard_initial_state():
    cfg = config_for("extension", "free_packet")
    assert cfg.mode_count == 33 and np.sum(abs(initial_numpy(cfg))**2) == pytest.approx(1., abs=5e-16)
    for name in CANDIDATES+CONTROLS:
        model = R8Model(cfg, name, 0)
        k = torch.tensor(cfg.wave_numbers, dtype=torch.float64)
        np.testing.assert_allclose(model(k, torch.zeros_like(k)).detach(), initial_numpy(cfg), atol=3e-16, rtol=0)


def test_exact_adam_rng_resume_across_carrier_and_window_boundaries(tmp_path):
    from src.train_r8_coupling_recovery import train_run
    cfg = replace(config_for("smoke"), adam_steps=30, carrier_steps=10, windows=10, recovery_every=5)
    for name, interrupt in (("A_carrier_complex", 10), ("D_carrier_windows", 14)):
        whole, restored = tmp_path/(name+"whole"), tmp_path/(name+"resume")
        train_run(cfg, name, 0, whole, stage="smoke")
        train_run(cfg, name, 0, restored, stage="smoke", stop_after=interrupt)
        train_run(cfg, name, 0, restored, stage="smoke")
        a = torch.load(whole/"model_state.pt", weights_only=False)
        b = torch.load(restored/"model_state.pt", weights_only=False)
        for key in a["state_dict"]:
            torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], atol=0, rtol=0)
        assert a["coupled_time_stream_sha256"] == b["coupled_time_stream_sha256"]
        assert a["free_time_stream_sha256"] == b["free_time_stream_sha256"]
        assert all((restored/f).is_file() for f in ("config.json", "metrics.json", "loss_history.csv", "time_series.csv", "dense_grid.npz", "model_state.pt"))
