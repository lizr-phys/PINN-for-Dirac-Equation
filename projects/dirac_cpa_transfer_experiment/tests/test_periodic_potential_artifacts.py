import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.periodic_potential_config import PERIODIC_POTENTIAL_MODELS
from src.train_periodic_potential_coupled_ltpb import RAW, ROOT, RUN_ARTIFACTS, load_formal_config
from src.periodic_potential_physics import solve_reference_eigendecomposition, reference_solution_checks, compare_reference_truncations
from src.freeze_periodic_potential_coupled_ltpb import verify_freeze


def test_authorized_amended_reference_passes_without_changing_original_failure():
    if not (RAW/"formal_selection.json").exists():
        pytest.skip("Formal selection not created")
    cfg = load_formal_config()
    assert (cfg.potential_strength, cfg.t_max, cfg.ladder_half_width) == (0.2, 5.0, 6)
    t = np.linspace(0, cfg.t_max, 4001)
    ref6 = solve_reference_eigendecomposition(cfg, t, half_width=6)
    ref8 = solve_reference_eigendecomposition(cfg, t, half_width=8)
    diff = compare_reference_truncations(ref6, ref8)
    assert diff["padded_full_trajectory_spinor_relative_l2"] < 1e-6
    assert diff["padded_max_time_spinor_relative_l2"] < 1e-6
    for ref in (ref6, ref8):
        diagnostics = reference_solution_checks(ref)
        assert diagnostics["hermiticity_max_abs"] == 0
        assert diagnostics["norm_max_drift"] < 2e-13
        assert diagnostics["time_reversal_relative_l2"] < 2e-13
    original = json.loads((RAW/"calibration/calibration.json").read_text())
    assert original["calibration_status"] == "NO_STRICT_CANDIDATE"


def test_smoke_complete_finite_and_hard_initial():
    base = ROOT/"results/raw/periodic_potential_coupled_ltpb_smoke"
    if not base.exists():
        pytest.skip("Smoke not run yet")
    for model in PERIODIC_POTENTIAL_MODELS:
        run = base/model/"seed_0"
        assert {p.name for p in run.iterdir()} == set(RUN_ARTIFACTS)
        metrics = json.loads((run/"metrics.json").read_text())
        assert metrics["steps"] == 20 and not metrics["formal"]
        assert metrics["initial_condition_max_abs_error"] < 1e-12
        h = pd.read_csv(run/"loss_history.csv")
        assert np.isfinite(h.to_numpy()).all()


def test_formal_output_inventory_and_artifact_pairing():
    if not (RAW/"decision.json").exists():
        pytest.skip("Formal results have not been aggregated")
    assert verify_freeze()
    collected = []
    for model in PERIODIC_POTENTIAL_MODELS:
        for seed in range(5):
            run = RAW/"learned"/model/f"seed_{seed}"
            assert {p.name for p in run.iterdir()} == set(RUN_ARTIFACTS)
            metric = json.loads((run/"metrics.json").read_text())
            assert metric["steps"] == 5000 and metric["formal"]
            assert len(pd.read_csv(run/"loss_history.csv")) == 501
            assert len(pd.read_csv(run/"time_series.csv")) == 201
            with np.load(run/"dense_grid.npz") as data:
                assert data["reference_active_branch_mask"].shape == (201, 13, 2)
                assert data["psi_predicted_raw"].dtype == np.complex128
            collected.append(metric)
    frame = pd.DataFrame(collected)
    for seed in range(5):
        paired = frame[frame.seed == seed]
        assert paired.time_stream_sha256.nunique() == 1
        assert paired[paired.model != "physical_vanilla"].initial_trainable_sha256.nunique() == 1
