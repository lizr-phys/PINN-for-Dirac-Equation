"""Post-training evidence checks; not changes to the frozen numerical method."""
import json

import numpy as np
import pandas as pd
import pytest

from src.aggregate_r7_phase_release import stage_runs, summary_table
from src.audit_r7_phase_release import independent_metrics, link_check
from src.freeze_r7_phase_release import RAW, verify_freeze, verify_history
from src.periodic_potential_physics import energy_basis_numpy
from src.r7_decision import decide


def synthetic_data():
    rng = np.random.default_rng(104)
    k, x = np.array([1., 2., 3.]), np.linspace(-np.pi, np.pi, 16, endpoint=False)
    exact = rng.normal(size=(4, 3, 2))+1j*rng.normal(size=(4, 3, 2))
    exact /= np.linalg.norm(exact, axis=(1, 2), keepdims=True)
    raw = .9*np.exp(.31j)*exact
    phase = np.angle(np.vdot(exact, raw))
    pred = raw*np.exp(-1j*phase)
    basis_p, basis_m = energy_basis_numpy(k, 5.)
    ce = np.stack((np.sum(exact*basis_p, axis=2), np.sum(exact*basis_m, axis=2)), axis=2)
    cp = np.stack((np.sum(pred*basis_p, axis=2), np.sum(pred*basis_m, axis=2)), axis=2)
    active, valid = np.abs(ce)**2 >= 1e-6, (np.abs(ce)**2 >= 1e-6)&(np.abs(cp)**2 >= 1e-12)
    flat = active.reshape(4, -1)
    pairs = flat[:, :, None]&flat[:, None, :]&~np.eye(6, dtype=bool)[None]
    phase_errors = np.full(ce.shape, np.nan)
    phase_errors[valid] = np.angle(cp[valid]*ce[valid].conj())
    fourier = np.exp(1j*np.outer(x, k))/np.sqrt(2*np.pi)
    return dict(spectral_state_predicted_raw=raw, spectral_state_exact=exact, wave_numbers=k,
                psi_predicted_gauge_aligned=np.einsum("xk,tks->txs", fourier, pred),
                psi_exact=np.einsum("xk,tks->txs", fourier, exact), reference_active_branch_mask=active,
                branch_phase_evaluation_mask=valid, reference_active_pair_mask=pairs,
                branch_phase_error_rad_masked_nan=phase_errors)


def test_independent_audit_gauge_and_populations():
    metrics, masks_ok = independent_metrics(synthetic_data(), {"mass": 5.})
    assert isinstance(masks_ok, bool)
    assert masks_ok
    assert metrics["spinor_rel_l2_gauge_aligned"] == pytest.approx(.1, abs=2e-15)
    for metric in ("density_rel_l2", "current_rel_l2", "modal_population_rel_l2", "branch_population_rel_l2", "active_coherence_rel_l2"):
        assert metrics[metric] == pytest.approx(.19, abs=2e-15)
    assert metrics["fidelity_mean"] == pytest.approx(1., abs=2e-15)


def test_audit_does_not_mask_collapsed_predictions_out_of_coherence():
    data = synthetic_data()
    data["spectral_state_predicted_raw"][:] = 0
    data["psi_predicted_gauge_aligned"][:] = 0
    data["branch_phase_evaluation_mask"][:] = False
    data["branch_phase_error_rad_masked_nan"][:] = np.nan
    metrics, masks_ok = independent_metrics(data, {"mass": 5.})
    assert masks_ok
    assert metrics["active_coherence_rel_l2"] == pytest.approx(1.)
    assert metrics["active_branch_phase_coverage"] == 0
    assert metrics["modal_population_rel_l2"] == pytest.approx(1.)


def test_markdown_link_check_handles_absolute_paths_with_spaces(tmp_path):
    target = tmp_path/"evidence with spaces.md"
    target.write_text("evidence", encoding="utf-8")
    report = tmp_path/"report.md"
    report.write_text(f"[local](<{target.as_posix()}>) [web](https://example.org/) [section](#scope)", encoding="utf-8")
    assert link_check(report) == dict(pass_check=True, checked_links=1, missing=[])


def test_frozen_science_and_previous_results_unchanged():
    assert len(verify_freeze()) == 64
    history = verify_history()
    assert history["pass"] and history["checked_files"] > 100


def test_short_summary_uses_all_seeds_and_population_sd():
    short = stage_runs("short")
    assert len(short) == 9
    summary = summary_table(short)
    assert summary.n_seeds.tolist() == [3, 3, 3]
    for _, row in summary.iterrows():
        values = short[short.model == row.model].spinor_rel_l2_gauge_aligned
        assert row.spinor_rel_l2_gauge_aligned_population_sd == pytest.approx(values.std(ddof=0), abs=1e-16)


def test_final_report_refuses_partial_cohort(monkeypatch):
    from src import build_r7_phase_release_report as builder
    monkeypatch.setattr(builder, "stage_runs", lambda stage: pd.DataFrame())
    with pytest.raises(ValueError, match="all 30 main and 9 short"):
        builder.build_report()


def test_posthoc_uncoupled_surrogate_norm_and_expected_coupling_defect():
    from src.r7_config import stage_config
    from src.r7_failure_diagnostics import central_free_surrogate
    from src.periodic_potential_physics import central_initial_state, apply_coupled_hamiltonian_numpy
    cfg = stage_config("main")
    initial = central_initial_state(cfg)
    state = central_free_surrogate(np.linspace(0, 5, 21), cfg.as_dict(), initial[cfg.ladder_half_width])
    assert np.sum(np.abs(state)**2, axis=(1, 2)) == pytest.approx(np.ones(21), abs=3e-15)
    free_h = np.array([[cfg.mass, cfg.k0], [cfg.k0, -cfg.mass]])
    derivative = np.zeros_like(state)
    derivative[:, cfg.ladder_half_width] = -1j*np.einsum("ab,tb->ta", free_h, state[:, cfg.ladder_half_width])
    residual = 1j*derivative-apply_coupled_hamiltonian_numpy(state, cfg.wave_numbers, cfg.mass, cfg.potential_strength)
    assert np.mean(np.sum(abs(residual)**2, axis=(1, 2)))/(2*cfg.length) == pytest.approx(cfg.potential_strength**2/(4*cfg.length), abs=1e-17)


def test_final_r7_decision_matches_complete_run_json():
    main = stage_runs("main")
    if len(main) != 30:
        pytest.skip("Formal cohort still running; must run again after all 30 finish")
    stored = json.loads((RAW/"decision.json").read_text(encoding="utf-8"))
    assert decide(main) == stored
    for key, result in stored["comparisons"].items():
        if result["comparative_advantage"]:
            assert result["target_absolute_success"]
            assert result["paired_joint_win_count"] >= 4


def test_completed_r7_runs_preserve_early_and_residual_evidence():
    paths = list((RAW/"main").glob("*/*/seed_*/metrics.json"))
    if len(paths) != 30:
        pytest.skip("Formal cohort still running; must run again after all 30 finish")
    for path in paths:
        for artifact in ("early_time_series.csv", "residual_integral.csv", "residual_by_segment_mode.csv", "recovery_state.pt", "execution_events.jsonl"):
            assert (path.parent/artifact).is_file()
        with np.load(path.parent/"dense_grid.npz") as dense:
            assert len(dense["residual_fine_t"]) == 4001
            assert (np.diff(dense["early_t"]) > 0).all()
            assert 0 < dense["early_t"][1] < 1e-8
