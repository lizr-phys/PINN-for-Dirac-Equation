from __future__ import annotations
import ast
import json
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch
from src.r11_config import configuration, GROUPS, SEEDS, budget, protocol, ROOT, RAW
from src.r11_models import R11Model, hashes, coupled_parameters, parameter_counts
from src.r11_residual import (training_loss, coupled_residual, spinor_value_and_time_derivative,
    carrier_value_and_time_derivative, free_carrier_residual)
from src.r9_loss import hamiltonian_apply
from src.r9_reference import initial_numpy, build_hamiltonian, solve
from src.r9_models import original_gaussian, initial_coefficients
from src.r9_time_derivative import spinor_value_and_time_derivative as r9_tangent
from src.r8_loss import derivative
from src.r8_sampling import TimeSampler
from src.train_wavepacket_shared_spectral_ltpb import _set_runtime

_set_runtime(1, 1)

def points(cfg, count=2):
    k = torch.tensor(cfg.wave_numbers).reshape(-1, 1).repeat(count, 1)
    t = torch.linspace(.15, 4.8, count).reshape(-1, 1).repeat_interleave(cfg.mode_count, 0).requires_grad_()
    return k, t

@pytest.mark.parametrize("group", GROUPS)
def test_shapes_hard_initial_and_finite_full_parameter_gradients(group):
    cfg, model = configuration(), R11Model(configuration(), group, 400)
    if group == GROUPS[0]:
        model.carrier.freeze()
    k = torch.tensor(cfg.wave_numbers)
    initial = model(k, torch.zeros_like(k)).detach().numpy()
    assert np.max(abs(initial-initial_numpy(cfg))) < 1e-15
    k, t = points(cfg)
    state, dt = spinor_value_and_time_derivative(model, cfg, k, t)
    assert state.dtype == dt.dtype == torch.complex128
    assert state.shape == dt.shape == (2*129, 2)
    assert torch.allclose(state, model(k,t), atol=3e-15, rtol=3e-15)
    loss, power = training_loss(model, cfg, k, t)
    loss.backward()
    assert power.shape == (129,) and torch.isfinite(loss)
    for p in coupled_parameters(model):
        assert p.grad is not None and torch.isfinite(p.grad).all()
    if group == GROUPS[2]:
        assert sum(p.grad.abs().sum() for p in model.carrier.parameters()) > 0
    if group in GROUPS[:2]:
        assert all(p.grad is None and not p.requires_grad for p in model.carrier.parameters())

@pytest.mark.parametrize("group", GROUPS)
def test_chain_rule_state_time_residual_and_all_parameter_gradients_match_ad(group):
    cfg, model = configuration(), R11Model(configuration(), group, 402)
    if group == GROUPS[0]:
        model.carrier.freeze()
    k, t = points(cfg)
    value, tangent = spinor_value_and_time_derivative(model, cfg, k, t)
    automatic = derivative(model(k, t), t, True)
    assert torch.allclose(tangent, automatic, atol=2e-13, rtol=2e-13)
    residual = coupled_residual(model, cfg, k, t)
    ad_residual = 1j*automatic.reshape(2,129,2)-hamiltonian_apply(model(k,t).reshape(2,129,2), cfg)
    assert torch.allclose(residual, ad_residual, atol=3e-13, rtol=3e-13)
    params = coupled_parameters(model)
    objective = residual.abs().square().mean(-1).sum(-1).mean()/cfg.length
    ad_objective = ad_residual.abs().square().mean(-1).sum(-1).mean()/cfg.length
    ga = torch.autograd.grad(objective, params)
    gb = torch.autograd.grad(ad_objective, params)
    assert all(torch.allclose(a,b,atol=2e-11,rtol=2e-11) for a,b in zip(ga,gb))

@pytest.mark.parametrize("group", GROUPS[:4])
def test_learned_tangent_exactly_inherits_r9_formula(group):
    cfg, model = configuration(), R11Model(configuration(), group, 401)
    k, t = points(cfg)
    a, b = spinor_value_and_time_derivative(model,cfg,k,t), r9_tangent(model,cfg,k,t)
    assert all(torch.equal(x,y) for x,y in zip(a,b))

def test_random_frozen_time_derivative_and_carrier_defect_not_dropped():
    cfg, model = configuration(), R11Model(configuration(), GROUPS[1], 400)
    k, t = points(cfg)
    p, pt = carrier_value_and_time_derivative(model.carrier,cfg,k,t)
    assert all(not v.requires_grad for v in model.carrier.parameters())
    assert pt.abs().max() > 1e-5
    assert torch.allclose(pt, derivative(p,t,True), atol=1e-13,rtol=1e-13)
    assert free_carrier_residual(model.carrier,cfg,k,t).abs().max() > 1
    # Omitting the carrier derivative gives a measurably different full residual.
    state, dt = spinor_value_and_time_derivative(model,cfg,k,t)
    assert not torch.allclose(1j*dt.reshape(2,129,2)-hamiltonian_apply(state.reshape(2,129,2),cfg),
                              -hamiltonian_apply(state.reshape(2,129,2),cfg))

def test_exact_diagnostic_is_explicit_zero_parameter_free_operator_solution():
    cfg, model = configuration(), R11Model(configuration(), GROUPS[4],400)
    k, t = points(cfg)
    assert model.carrier.analytic_prior
    assert len(list(model.carrier.parameters())) == 0
    assert free_carrier_residual(model.carrier,cfg,k,t).abs().max() < 2e-14
    assert all("energy" not in key and "frequency" not in key for key in model.state_dict())

def test_shift_eight_matches_independent_hermitian_matrix():
    cfg = configuration()
    h = build_hamiltonian(cfg)
    generator = torch.Generator().manual_seed(777)
    raw = torch.complex(torch.randn(3,129,2,generator=generator), torch.randn(3,129,2,generator=generator))
    applied = hamiltonian_apply(raw,cfg).numpy()
    np.testing.assert_allclose(applied, (raw.numpy().reshape(3,-1)@h.T).reshape(3,129,2),atol=2e-14,rtol=2e-14)
    assert np.array_equal(h,h.conj().T)
    assert h[0,16] == .1 and h[0,2] == 0

def test_original_gaussian_zero_padding_without_renormalization():
    cfg = configuration()
    coeff = initial_coefficients(cfg)
    original = original_gaussian()
    assert abs(original.square().sum()-1) < 3e-16
    assert abs(coeff.abs().square().sum()-1) < 3e-16
    assert torch.equal(coeff[48:81,0], original.to(torch.complex128)/2**.5)
    assert not torch.count_nonzero(coeff[:48]) and not torch.count_nonzero(coeff[81:])
    larger = initial_coefficients(replace(cfg,n_min=-64,n_max=96))
    assert torch.equal(coeff,larger[16:145])

def test_param_counts_and_elementwise_pairing_only_same_shape():
    cfg = configuration()
    models = [R11Model(cfg,g,400) for g in GROUPS]
    models[0].carrier.freeze()
    counts = [parameter_counts(m) for m in models]
    assert [c["total_parameters"] for c in counts] == [25734,25734,25734,25864,12932]
    assert [c["active_parameters"] for c in counts] == [12932,12932,25734,25864,12932]
    assert abs(counts[3]["total_parameters"]/counts[0]["total_parameters"]-1) < .006
    init = [hashes(m) for m in models]
    assert len({r["envelopes"][0] for r in init}) == 1
    assert len({r["carrier"] for r in init[:3]}) == 1
    assert init[3]["envelopes"][1] != init[3]["envelopes"][0]
    assert all(p.dtype == torch.float64 for m in models for p in m.parameters())

def test_independent_initialization_sampling_and_full_coupled_prefix():
    cfg = configuration()
    first = R11Model(cfg,GROUPS[0],404)
    free, coupled = TimeSampler(cfg,404,carrier=True), TimeSampler(cfg,404)
    original = TimeSampler(cfg,404)
    for _ in range(3):
        free.sample()
    assert torch.equal(coupled.sample()[1], original.sample()[1])
    after_sampling = R11Model(cfg,GROUPS[2],404)
    assert hashes(first) == hashes(after_sampling)
    fresh = [TimeSampler(cfg,404) for _ in GROUPS]
    for _ in range(5):
        traces = [s.sample()[1] for s in fresh]
        assert all(torch.equal(traces[0], v) for v in traces)
        assert traces[0].shape == (32*129,1)
    saved = fresh[0].state_dict()
    restored = TimeSampler(cfg,404); restored.load_state_dict(saved)
    assert torch.equal(restored.sample()[1],fresh[0].sample()[1])
    assert not torch.equal(saved["trace"][:3],free.state_dict()["trace"])

def test_registered_budgets_criteria_and_new_seed_completeness():
    assert [budget(g) for g in GROUPS] == [(6000,20000),(0,26000),(0,26000),(0,26000),(0,20000)]
    assert protocol()["required_joint_pass_count"] == 4
    from src.r11_decision import absolute_success
    data = pd.DataFrame(dict(seed=SEEDS,spinor_rel_l2_gauge_aligned=[.01]*5,
        current_rel_l2=[.02]*5,fidelity_mean=[.999]*5,modal_population_rel_l2=[.01]*5,max_norm_drift=[.001]*5))
    assert absolute_success(data)["pass_check"]
    data.loc[0,"max_norm_drift"] = .03
    assert absolute_success(data)["pass_check"]
    data.loc[1,"max_norm_drift"] = .03
    assert not absolute_success(data)["pass_check"]
    data.seed = [300,301,302,303,304]
    assert not absolute_success(data)["complete"]

def test_no_reference_or_frequency_in_learned_path_with_explicit_e_exception():
    for name in ("r11_models.py","r11_residual.py","train_r11_ablation.py"):
        tree = ast.parse((ROOT/"src"/name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in ("r9_reference","r11_reference")
    # Independent evaluation is imported only after fixed weights have been saved.
    text = (ROOT/"src/train_r11_ablation.py").read_text(encoding="utf-8")
    assert "if smoke:\n            from .r9_evaluate" in text
    for group in GROUPS[:4]:
        model = R11Model(configuration(),group,403)
        assert model.carrier is None or not getattr(model.carrier,"analytic_prior",False)
        assert not any(any(word in key for word in ("energy","frequency","eigenvalue","reference","target"))
                       for key in model.state_dict())

def test_reference_revalidation_saved_before_training():
    record = json.loads((RAW/"reference_validation.json").read_text(encoding="utf-8"))
    assert record["status"] == "PASS" and record["selected_truncation"] == [-48,80]
    assert record["relative_difference_from_r9"] < 1e-12
    assert record["revalidation"]["full_complex_zero_padded_relative_difference"] < 1e-6
    assert record["revalidation"]["lower_eight_layers_max_total_occupation"] < 1e-8
    assert record["revalidation"]["upper_eight_layers_max_total_occupation"] < 1e-8
    with np.load(RAW/"reference_test_grid.npz") as ref:
        assert ref["state"].shape == (201,129,2)
        assert abs(np.sum(abs(ref["state"])**2,(1,2))-1).max() < 1e-12

@pytest.mark.parametrize("group", GROUPS)
def test_complete_evaluation_archive_dimensions_masks_finite_and_exterior_sign(group):
    from src.r11_evaluate import evaluate
    cfg, model = configuration(), R11Model(configuration(), group, 404)
    if group == GROUPS[0]:
        model.carrier.freeze()
    metrics, series, dense, tables = evaluate(model,cfg,full=True)
    assert metrics["fourier_integral_no_aliasing"]
    assert metrics["initial_condition_max_abs_error"] < 1e-14
    assert dense["spectral_predicted_raw"].shape == (201,129,2)
    assert dense["residual_complex_unweighted"].shape == (4001,129,2)
    assert dense["full_physical_residual_complex"].shape == (4001,145,2)
    assert np.array_equal(dense["phase_reference_active_mask"],
                          dense["coherence_reference_active_channel_mask"].reshape(201,129,2))
    assert np.array_equal(dense["exterior_residual_complex"],
                          -dense["exterior_positive_coupling_source_complex"])
    assert np.array_equal(dense["full_physical_residual_complex"][:,8:137],
                          dense["residual_complex_unweighted"])
    assert len(series) == 201 and len(tables["residual_by_mode.csv"]) == 129
    assert len(tables["norm_balance.csv"]) == 4001
    assert not metrics["residual_numerical_estimate_is_continuous_certificate"]
    assert all(np.isfinite(a).all() for a in dense.values())
    if group == GROUPS[3]:
        assert "carrier_diagnostics.csv" not in tables
    else:
        assert dense["carrier_unit_spinor_defect"].shape == (201,129,2,2)
    # Engineering-only test: no accuracy thresholds or settings chosen from these predictions.

@pytest.mark.parametrize("group", (GROUPS[0],GROUPS[2]))
def test_exact_optimizer_rng_resume_matches_uninterrupted_smoke(tmp_path, group):
    from src.train_r11_ablation import train_run
    direct, resumed = tmp_path/"direct", tmp_path/"resumed"
    train_run(group,404,direct,smoke=True)
    train_run(group,404,resumed,smoke=True,stop_after=1)
    train_run(group,404,resumed,smoke=True)
    a,b = (torch.load(p/"recovery_state.pt",weights_only=False) for p in (direct,resumed))
    assert a["step"] == b["step"]
    assert all(torch.equal(a["state_dict"][k],b["state_dict"][k]) for k in a["state_dict"])
    assert all(torch.equal(a["coupled_sampler"][k],b["coupled_sampler"][k]) for k in a["coupled_sampler"])
    assert all(torch.equal(a["free_sampler"][k],b["free_sampler"][k]) for k in a["free_sampler"])
    assert len(a["optimizer"]["state"]) == len(b["optimizer"]["state"])
    for k in a["optimizer"]["state"]:
        assert all(torch.equal(a["optimizer"]["state"][k][v],b["optimizer"]["state"][k][v])
                   for v in a["optimizer"]["state"][k])
