import json
import numpy as np
import pandas as pd
import pytest
from src.r9_config import R9Config, RAW, MODELS, SEEDS, run_directory
from src.r9_full_residual_archive import assemble_full_residual
from src.r9_loss import hamiltonian_apply
from src.r9_decision import comparisons
import torch


def test_signed_full_residual_equals_direct_physical_operator():
    cfg = R9Config(n_min=-48, n_max=80)
    rng = np.random.default_rng(901)
    state = rng.normal(size=(3, 129, 2))+1j*rng.normal(size=(3, 129, 2))
    retained = -hamiltonian_apply(torch.tensor(state), cfg).numpy()
    positive_source = cfg.potential_strength/2*np.concatenate((state[:, :8], state[:, -8:]), 1)
    n, full = assemble_full_residual(retained, positive_source, cfg)
    x = np.linspace(cfg.x_min, cfg.x_max, 1024, endpoint=False)
    spatial = np.exp(1j*x[:, None]*np.asarray(cfg.wave_numbers)[None])/np.sqrt(cfg.length)
    psi = np.einsum("xk,tkc->txc", spatial, state)
    free_apply = np.asarray(cfg.wave_numbers)[None, :, None]*state[..., ::-1]
    free_apply += cfg.mass*np.stack((state[..., 0], -state[..., 1]), -1)
    direct = -np.einsum("xk,tkc->txc", spatial, free_apply)-cfg.potential_strength*np.cos(x)[None, :, None]*psi
    full_spatial = np.exp(1j*x[:, None]*(n/8)[None])/np.sqrt(cfg.length)
    reconstructed = np.einsum("xk,tkc->txc", full_spatial, full)
    np.testing.assert_allclose(reconstructed, direct, atol=8e-14, rtol=2e-13)
    assert np.array_equal(n, np.arange(-56, 89))
    np.testing.assert_array_equal(full[:, :8], -positive_source[:, :8])
    np.testing.assert_array_equal(full[:, -8:], -positive_source[:, -8:])


def test_paired_statistics_use_actual_seed_pairing_and_population_sd():
    records = []
    for model_index, name in enumerate(MODELS):
        for seed in SEEDS:
            values = seed-300
            records.append(dict(model=name, seed=seed, spinor_rel_l2_gauge_aligned=.02+.1*model_index+.001*values,
                current_rel_l2=.03+.2*model_index+.002*values, fidelity_mean=.99-.1*model_index,
                modal_population_rel_l2=.03+.05*model_index, max_norm_drift=.01+.1*model_index,
                density_rel_l2=.04+.2*model_index, active_coherence_rel_l2=.05+.1*model_index,
                branch_complex_coefficient_rel_l2=.02+.2*model_index))
    frame = pd.DataFrame(records).sample(frac=1, random_state=8)
    paired, summary = comparisons(frame)
    rows = paired[(paired.target == MODELS[0])&(paired.baseline == MODELS[2])&(paired.metric == "spinor_rel_l2_gauge_aligned")]
    np.testing.assert_allclose(rows.paired_difference, -.2, atol=5e-17)
    row = summary[(summary.target == MODELS[0])&(summary.baseline == MODELS[2])&(summary.metric == "spinor_rel_l2_gauge_aligned")].iloc[0]
    assert row.paired_difference_population_sd < 5e-17
    assert row.ratio_of_means == pytest.approx(.022/.222)


def test_full_signed_residual_supplements_preserve_formal_originals_if_present():
    records = list((RAW/"formal").glob("*/seed_*/full_physical_residual_archival_record.json"))
    if not records:
        pytest.skip("Final archival supplement has not run")
    assert len(records) == 15
    from src.freeze_r7_phase_release import sha256
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["status"] == "PASS" and record["added_optimizer_steps"] == 0
        assert record["primary_metrics_changed"] is False
        for name, digest in record["original_files_unchanged"].items():
            assert sha256(path.parent/name) == digest
        dense = np.load(path.parent/"dense_grid.npz")
        full = np.load(path.parent/"full_physical_residual.npz")
        np.testing.assert_array_equal(full["exterior_residual_complex"], -dense["exterior_residual_complex"])
        assert full["residual_complex_unweighted"].shape == (4001, 145, 2)
