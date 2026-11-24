from dataclasses import replace
import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from src.periodic_potential_config import PeriodicPotentialConfig, PERIODIC_POTENTIAL_MODELS, paired_initialization_seed
from src.periodic_potential_models import build_periodic_potential_model
from src.periodic_potential_physics import build_coupled_hamiltonian, central_initial_state, solve_reference_eigendecomposition
from src.periodic_potential_evaluate import (
    _branch_phase_and_coherence, _global_gauge_align, _initial_tangent_diagnostics,
    compute_periodic_potential_evaluation,
)
from src.periodic_potential_coupled_ltpb_decision import TARGET, CONTROL, evaluate_periodic_potential_decision
from src.train_periodic_potential_coupled_ltpb import PairedSampler, pde_loss, physical_pde_loss, spectral_pde_loss

torch.set_num_threads(1)
CFG = PeriodicPotentialConfig()


def test_paired_time_streams_are_independent_of_initialization_and_positions():
    a, b = PairedSampler(CFG, 3, physical=False), PairedSampler(CFG, 3, physical=False)
    c = PairedSampler(CFG, 3, physical=True)
    for _ in range(4):
        k1, t1 = a.sample()
        torch.randn(177)
        k2, t2 = b.sample()
        x, tx = c.sample()
        assert torch.equal(k1, k2) and torch.equal(t1, t2)
        assert torch.equal(t1.reshape(32, -1)[:, 0], tx.reshape(32, -1)[:, 0])
        assert len(x) == 1024 and len(k1) == 416
    assert a.time_digest.hexdigest() == b.time_digest.hexdigest() == c.time_digest.hexdigest()


@pytest.mark.parametrize("name", PERIODIC_POTENTIAL_MODELS)
def test_loss_gradients_and_forward_cannot_access_reference(name, monkeypatch):
    import src.periodic_potential_physics as physics
    import src.periodic_potential_evaluate as evaluation
    import src.wavepacket_physics as free_physics
    def forbidden(*args, **kwargs):
        raise AssertionError("Reference accessed during learned training")
    monkeypatch.setattr(physics, "solve_reference_eigendecomposition", forbidden)
    monkeypatch.setattr(evaluation, "_reference_spectral_state", forbidden)
    monkeypatch.setattr(free_physics, "reference_energies", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    model = build_periodic_potential_model(name, CFG, paired_initialization_seed(0))
    coordinates, time = PairedSampler(CFG, 0, physical=name == "physical_vanilla").sample()
    loss = pde_loss(name, model, CFG, coordinates, time)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for function in (physical_pde_loss, spectral_pde_loss):
        assert "reference" not in inspect.getsource(function)


def test_zero_initial_ltpb_tangent_is_real_but_required_sideband_tangent_is_imaginary():
    model = build_periodic_potential_model(TARGET, CFG, paired_initialization_seed(1))
    metrics, fields = _initial_tangent_diagnostics(model, CFG)
    mask = fields["initial_first_sideband_mask"]
    assert np.max(np.abs(fields["initial_time_derivative_predicted"][mask].imag)) == 0
    assert np.linalg.norm(fields["initial_time_derivative_required_by_pde"][mask].imag) > 0.1
    assert metrics["initial_sideband_time_derivative_rel_l2"] >= 1
    assert metrics["initial_sideband_imaginary_tangent_parameter_sensitivity"] == 0
    complex_model = build_periodic_potential_model(CONTROL, CFG, paired_initialization_seed(1))
    complex_metrics, _ = _initial_tangent_diagnostics(complex_model, CFG)
    assert complex_metrics["initial_sideband_imaginary_tangent_parameter_sensitivity"] > 0


def test_global_phase_and_active_coherence_masks_do_not_hide_wrong_relative_phase():
    t = np.linspace(0, 1, 10)
    exact = np.zeros((10, 3, 2), complex)
    exact[:, 1] = np.exp(1j*t)[:, None]/np.sqrt(2)
    exact[:, 2, 0] = np.linspace(0, 0.1, 10)
    aligned, _ = _global_gauge_align(exact*np.exp(0.71j), exact)
    np.testing.assert_allclose(aligned, exact, atol=3e-16)
    predicted = exact.copy()
    predicted[:, 2, 0] *= 1j
    aligned, _ = _global_gauge_align(predicted, exact)
    metrics, dense = _branch_phase_and_coherence(aligned, exact, active_threshold=1e-6)
    assert not dense["reference_active_branch_mask"][:, 0].any()
    assert np.isnan(dense["branch_phase_error_rad_masked_nan"][:, 0]).all()
    assert metrics["active_coherence_rel_l2"] > 0.05
    collapsed, masks = _branch_phase_and_coherence(np.zeros_like(exact), exact, active_threshold=1e-6)
    assert collapsed["active_coherence_rel_l2"] == 1
    assert collapsed["active_branch_phase_coverage"] == 0
    assert not masks["branch_phase_evaluation_mask"].any()


class ReferenceOracle(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        values, vectors = torch.linalg.eigh(torch.tensor(build_coupled_hamiltonian(cfg)))
        self.register_buffer("values", values)
        self.register_buffer("vectors", vectors)
        self.register_buffer("coordinates", vectors.conj().T @ torch.tensor(central_initial_state(cfg).ravel()))

    def forward(self, k, t):
        all_states = (torch.exp(-1j*t.reshape(-1, 1)*self.values[None]) * self.coordinates[None]) @ self.vectors.T
        mode = torch.round((k.reshape(-1)-self.cfg.k0)/self.cfg.coupling_wave_number).long()+self.cfg.ladder_half_width
        return all_states.reshape(-1, self.cfg.mode_count, 2)[torch.arange(len(mode)), mode]


def test_reference_oracle_metrics_are_exact_and_physical_residual_retains_boundary_terms():
    cfg = replace(CFG, test_x_points=128, test_t_points=21)
    model = ReferenceOracle(cfg)
    metrics, series, dense = compute_periodic_potential_evaluation(
        model, cfg, model_name=CONTROL, seed=0, parameter_count=0,
        training_seconds=0, final_pde_loss=0, peak_rss_mb=0, peak_rss_increment_mb=0,
    )
    for key in ("spinor_rel_l2_gauge_aligned", "current_rel_l2", "modal_population_rel_l2", "max_norm_drift",
                "retained_coupled_spectral_pde_residual_rms", "parseval_dense_rms_absolute_difference"):
        assert metrics[key] < 1e-12, (key, metrics[key])
    assert abs(metrics["fidelity_min"]-1) < 3e-14
    assert metrics["dense_physical_pde_residual_rms"] > 0
    assert len(series) == cfg.test_t_points
    assert dense["reference_active_branch_mask"].shape == (21, 13, 2)


def synthetic_runs():
    return pd.DataFrame([dict(model=model, seed=seed, spinor_rel_l2_gauge_aligned=0.03 if model == TARGET else 0.06,
                             current_rel_l2=0.04 if model == TARGET else 0.08, fidelity_mean=0.99,
                             modal_population_rel_l2=0.02, max_norm_drift=0.01)
                         for model in PERIODIC_POTENTIAL_MODELS for seed in range(5)])


def test_decision_classes_and_joint_seed_condition():
    data = synthetic_runs()
    assert evaluate_periodic_potential_decision(data)["classification"] == "COMPARATIVE_ADVANTAGE"
    data.loc[data.model == CONTROL, ["spinor_rel_l2_gauge_aligned", "current_rel_l2"]] = 0.01
    assert evaluate_periodic_potential_decision(data)["classification"] == "TRANSFER_SUCCESS"
    data.loc[data.model == TARGET, "max_norm_drift"] = 0.021
    assert evaluate_periodic_potential_decision(data)["classification"] == "REPRESENTATION_FAILURE"
    data.loc[data.model == CONTROL, "current_rel_l2"] = 0.151
    assert evaluate_periodic_potential_decision(data)["classification"] == "INCONCLUSIVE"
    data = synthetic_runs()
    data.loc[(data.model == TARGET) & data.seed.isin([0, 1]), "spinor_rel_l2_gauge_aligned"] = 0.11
    decision = evaluate_periodic_potential_decision(data)
    assert decision["absolute_results"][TARGET]["mean_checks"]["spinor_rel_l2_gauge_aligned"]["pass"]
    assert decision["absolute_results"][TARGET]["joint_absolute_pass_count"] == 3
    assert not decision["transfer_success"]
