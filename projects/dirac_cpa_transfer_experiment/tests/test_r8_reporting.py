"""Independent aggregation/audit checks; these do not alter training science."""
import hashlib
import numpy as np
import pandas as pd
import pytest
import torch
from src.aggregate_r8_coupling_recovery import (summary_table, paired_rows, CORE,
                                              exact_bilinear_relative_errors,
                                              occupied_polar_required_outputs)
from src.audit_r8_coupling_recovery import (carrier_unchanged, tensor_state_digest,
                                          independent_absolute, independent_score,
                                          dynamics_series_matches, independently_decide,
                                          affinity_masks_overlap, physical_sampled_spinor_metrics)
from src.build_r8_coupling_recovery_report import table, number, link
from src.r8_decision import score, joint_mask, absolute_success
from src.r8_config import THRESHOLDS


def frame(seeds=(100, 101, 102, 103, 104), model="A_carrier_complex"):
    records = []
    for seed in seeds:
        records.append(dict(stage="validation", case="main", model=model, balanced=False,
            seed=seed, spinor_rel_l2_gauge_aligned=.04, current_rel_l2=.07,
            fidelity_mean=.995, modal_population_rel_l2=.03, max_norm_drift=.008,
            sideband_complex_coefficient_rel_l2=.06, active_coherence_rel_l2=.07))
    return pd.DataFrame(records)


def test_population_sd_and_all_seeds_are_retained():
    values = frame()
    values["current_rel_l2"] = [.03, .04, .05, .06, .07]
    result = summary_table(values).iloc[0]
    assert result.n_seeds == 5 and result.joint_absolute_pass_count == 5
    assert result.current_rel_l2_mean == pytest.approx(.05)
    assert result.current_rel_l2_population_sd == pytest.approx(np.std(values.current_rel_l2, ddof=0))
    assert result.current_rel_l2_population_sd != pytest.approx(np.std(values.current_rel_l2, ddof=1))


def test_paired_effects_join_by_seed_not_row_order():
    target, baseline = frame(), frame(model="complex")
    baseline["spinor_rel_l2_gauge_aligned"] = [.10, .20, .30, .40, .50]
    baseline["current_rel_l2"] = .21
    baseline = baseline.iloc[::-1]
    result = pd.DataFrame(paired_rows(target, baseline, "test")).set_index("seed")
    assert len(result) == 5 and result.spinor_current_joint_win.all()
    assert result.loc[100, "spinor_rel_l2_gauge_aligned_difference"] == pytest.approx(-.06)
    assert result.loc[104, "spinor_rel_l2_gauge_aligned_ratio"] == pytest.approx(.08)
    assert set(CORE).issubset({key.removesuffix("_difference") for key in result.columns})


def test_selection_uses_sideband_and_does_not_change_success_thresholds():
    values = frame((0, 1, 2))
    expected = (.04/.1 + .07/.15 + .06/.1)/3
    assert score(values) == pytest.approx(expected)
    higher_sideband = values.copy()
    higher_sideband["sideband_complex_coefficient_rel_l2"] = .96
    assert score(higher_sideband) > score(values)
    assert absolute_success(higher_sideband, (0, 1, 2), 2)["pass_check"]
    assert THRESHOLDS["spinor_rel_l2_gauge_aligned"] == .10
    assert independent_score(values) == pytest.approx(score(values), abs=2e-15)


def test_success_requires_every_primary_metric_and_four_new_seeds():
    values = frame()
    values.loc[0, "max_norm_drift"] = .03
    assert int(joint_mask(values).sum()) == 4
    assert absolute_success(values)["pass_check"]
    values.loc[1, "modal_population_rel_l2"] = .11
    assert not absolute_success(values)["pass_check"]
    assert not absolute_success(frame((0, 1, 2)))["complete"]
    assert independent_absolute(values, (100, 101, 102, 103, 104), 4) == absolute_success(values)


def test_tensor_hash_and_frozen_carrier_check_are_content_based():
    original = {"mlp.weight": torch.tensor([[1., 2.]], dtype=torch.float64),
                "mlp.bias": torch.zeros(1, dtype=torch.float64)}
    reverse_order = dict(reversed(list(original.items())))
    assert tensor_state_digest(original) == tensor_state_digest(reverse_order)
    final = {"carrier."+key: value.clone() for key, value in original.items()}
    final["fields.0.weight"] = torch.ones(2, dtype=torch.float64)
    assert carrier_unchanged(final, original)
    final["carrier.mlp.bias"][0] = 1e-10
    assert not carrier_unchanged(final, original)
    assert tensor_state_digest(original) != tensor_state_digest({k.removeprefix("carrier."): v for k, v in final.items() if k.startswith("carrier.")})
    assert not carrier_unchanged({}, original)


def test_report_table_format_missing_values_and_absolute_links(tmp_path):
    rendered = table(["seed", "error"], [[100, number(.03)], [101, number(np.nan)]])
    assert "| --- | --- |" in rendered and "| 101 | — |" in rendered
    path = tmp_path/"report with spaces.md"
    assert path.resolve().as_posix() in link(path)
    assert "(<" in link(path) and ">)" in link(path)


def test_time_series_must_match_dense_states_not_only_file_existence():
    times = np.linspace(0, 1, 3)
    exact = np.zeros((3, 3, 2), dtype=np.complex128)
    exact[:, 1, 0] = 1
    predicted = exact.copy()
    predicted[:, 0, 0] = .1*times
    population_p, population_e = np.sum(abs(predicted)**2, axis=2), np.sum(abs(exact)**2, axis=2)
    pn, en = population_p.sum(1), population_e.sum(1)
    dense = dict(t=times, mode_offsets=np.arange(-1, 2), spectral_state_predicted_raw=predicted,
        spectral_state_exact=exact, modal_occupations_predicted=population_p, modal_occupations_exact=population_e,
        branch_occupations_predicted=abs(predicted)**2, branch_occupations_exact=abs(exact)**2)
    data = dict(t=times, norm_predicted=pn, norm_exact=en, norm_drift=pn-1, fidelity=1/pn)
    for index, offset in enumerate(dense["mode_offsets"]):
        for tag in ("predicted", "exact"):
            data[f"mode_{offset:+d}_occupation_{tag}"] = dense[f"modal_occupations_{tag}"][:, index]
            for branch, label in ((0, "plus"), (1, "minus")):
                data[f"mode_{offset:+d}_{label}_occupation_{tag}"] = dense[f"branch_occupations_{tag}"][:, index, branch]
    series = pd.DataFrame(data)
    assert dynamics_series_matches(series, dense)
    series.loc[1, "mode_-1_plus_occupation_predicted"] = .9
    assert not dynamics_series_matches(series, dense)


def test_independent_decision_has_no_exploration_success_substitution():
    selection = dict(selected_candidates=["A_carrier_complex", "C_release_windows"])
    values = pd.concat([frame(model=name) for name in selection["selected_candidates"]+["complex", "phase_release"]])
    values.loc[values.model != "A_carrier_complex", "current_rel_l2"] = .456
    result = independently_decide(selection, values)
    assert result["successful_candidates"] == ["A_carrier_complex"]
    assert result["classification"] == "COUPLING_RECOVERY_SUCCESS"
    assert result["exploratory_selection_does_not_imply_validation"]


def test_exact_bilinear_fourier_check_and_global_gauge_invariance():
    rng = np.random.default_rng(7)
    exact = rng.normal(size=(3, 5, 2))+1j*rng.normal(size=(3, 5, 2))
    predicted = exact+.1*(rng.normal(size=exact.shape)+1j*rng.normal(size=exact.shape))
    x = np.linspace(0, 2*np.pi, 64, endpoint=False)
    fourier = np.exp(1j*np.outer(x, np.arange(-2, 3)))/np.sqrt(2*np.pi)
    fields = [np.einsum("xk,tks->txs", fourier, state) for state in (predicted, exact)]
    density = [np.sum(abs(field)**2, axis=2) for field in fields]
    current = [2*np.real(field[..., 0].conj()*field[..., 1]) for field in fields]
    reference = [np.linalg.norm(pair[0]-pair[1])/np.linalg.norm(pair[1]) for pair in (density, current)]
    np.testing.assert_allclose(exact_bilinear_relative_errors(predicted, exact), reference, atol=4e-16, rtol=2e-15)
    np.testing.assert_allclose(exact_bilinear_relative_errors(predicted*np.exp(1j*.37), exact), reference, atol=4e-16, rtol=2e-15)


def test_shared_cpu_core_is_disclosed_not_declared_distinct():
    assert not affinity_masks_overlap([1, 2, 4])
    assert affinity_masks_overlap([2, 1, 2])
    assert affinity_masks_overlap([3, 4, 2])


def test_smoke_grid_aliasing_is_not_certified_as_spectral_accuracy():
    times = np.linspace(0, 1, 3)
    exact = np.zeros((3, 5, 2), dtype=np.complex128)
    exact[:, 2, 0] = 1
    predicted = exact.copy()
    predicted[:, [0, 4], 0] = .2*times[:, None]
    x = np.linspace(0, 2*np.pi, 4, endpoint=False)
    fourier = np.exp(1j*np.outer(x, np.arange(-2, 3)))/np.sqrt(2*np.pi)
    fields = [np.einsum("xk,tks->txs", fourier, state) for state in (predicted, exact)]
    result = physical_sampled_spinor_metrics(dict(psi_predicted_raw=fields[0], psi_exact=fields[1]), 2*np.pi)
    assert result["max_norm_drift"] == pytest.approx(.16)
    assert abs(np.sum(abs(predicted[-1])**2)-1) == pytest.approx(.08)
    assert result["spinor_rel_l2_raw"] == pytest.approx(np.linalg.norm(fields[0]-fields[1])/np.linalg.norm(fields[1]))


def test_small_nonzero_polar_handoff_can_require_large_phase_coordinate():
    initial = np.asarray([1e-8*np.exp(1j*.3)], dtype=np.complex128)
    velocity = np.asarray([.1j*np.exp(1j*.3)], dtype=np.complex128)
    r, q = occupied_polar_required_outputs(initial, velocity, .5)
    assert q[0] == pytest.approx(5e6)
    rebuilt = np.exp(1j*np.angle(initial))*(r+1j*abs(initial)*q)/.5
    np.testing.assert_allclose(rebuilt, velocity, rtol=2e-15, atol=2e-17)
    with pytest.raises(ValueError, match="nonzero"):
        occupied_polar_required_outputs(np.zeros(1, dtype=complex), velocity, .5)


def test_free_packet_evaluation_adapter_skips_only_undefined_diagnostic():
    from types import SimpleNamespace
    from src.r8_free_packet_evaluation_adapter import initial_tangent_adapter
    calls = []
    def old(model, cfg):
        calls.append(cfg)
        return {"kept": True}, {"diagnostic": 1}
    adapted = initial_tangent_adapter(old)
    assert adapted(None, SimpleNamespace(case="free_packet", potential_strength=0.)) == ({}, {})
    assert not calls
    for case, strength in (("main", .2), ("main", 0.), ("free_packet", .2)):
        assert adapted(None, SimpleNamespace(case=case, potential_strength=strength))[0] == {"kept": True}
    assert len(calls) == 3


def test_evaluation_recovery_tensor_hash_is_order_independent_and_sensitive():
    import torch
    from src.r8_free_packet_evaluation_adapter import tensor_state_digest
    a = torch.tensor([1., 2.], dtype=torch.float64)
    b = torch.tensor([3j], dtype=torch.complex128)
    original = tensor_state_digest(dict(a=a, b=b))
    assert original == tensor_state_digest(dict(b=b.clone(), a=a.clone()))
    assert original != tensor_state_digest(dict(a=a+1e-8, b=b))


def test_residual_work_distinguishes_free_regression_skip_and_tangent_only_calls():
    from types import SimpleNamespace
    from src.aggregate_r8_coupling_recovery import residual_work_record
    cfg=SimpleNamespace(mode_count=33,test_t_points=201,residual_time_points=4001)
    run=SimpleNamespace(stage="extension",case="free_packet",model="A_carrier_complex",seed=200,
        carrier_steps=6000,residual_training_batches=20000,residual_training_spinor_points=27456000,
        evaluation_coupled_spinor_points=138699)
    record=residual_work_record(run,cfg)
    assert record["evaluation_full_coupled_residual_points"]==138666
    assert record["evaluation_carrier_unit_spinor_residual_points"]==13266
    assert record["initial_derivative_only_diagnostic_points"]==0
    assert record["full_spinor_residual_points_training_and_evaluation"]==27607932
    run.case="main"
    assert residual_work_record(run,cfg)["initial_derivative_only_diagnostic_points"]==66
