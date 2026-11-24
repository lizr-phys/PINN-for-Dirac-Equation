"""Validate delivered physical evidence, not exact implementation internals."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scattering_dirac_kg.physics import ROOT, read_json


def require(path):
    if not path.exists():
        pytest.skip("Full campaign artifacts not present in a fresh checkout")
    return path


def test_complete_fixed_energy_seed_coverage_and_reference_gate():
    frame=pd.read_csv(require(ROOT/"results"/"formal_v1_summary.csv"))
    assert len(frame)==24
    assert frame.groupby(["kind","energy"]).size().eq(3).all()
    assert frame.passed.all()
    gate=read_json(ROOT/"results"/"fixed_validation_gate.json")
    assert gate["passed"] and gate["reference_separated"]


def test_conditional_split_not_mislabeled_training_grid():
    frame=pd.read_csv(require(ROOT/"results"/"conditional_v1_summary.csv"))
    nodes=np.array(read_json(ROOT/"results"/"conditional_validation.json")["training_energy_nodes"])
    assert len(frame)==606
    for energy, subset in frame.groupby("energy"):
        is_node=np.min(abs(nodes-energy))<1e-12
        assert subset.training_energy.eq(is_node).all()
    heldout=frame[~frame.training_energy]
    assert heldout.energy.nunique()>90
    assert heldout.passed.all()


def test_all_baseline_seeds_and_failures_remain_available():
    frame=pd.read_csv(require(ROOT/"results"/"mlp_baseline_summary.csv"))
    assert len(frame)==24
    assert frame.groupby(["kind","energy"]).size().eq(3).all()
    for directory in (ROOT/"results"/"mlp_baseline").glob("*/E_*/seed_*"):
        assert all((directory/name).exists() for name in ["model.pt","history.csv","training.json","metrics.json","evaluation.npz"])


def test_small_reflection_phases_are_not_interpreted():
    frame=pd.read_csv(require(ROOT/"results"/"formal_v1_summary.csv"))
    high=frame[frame.energy==2.]
    assert high.r_phase_error.isna().all()


def test_actual_retraining_scan_is_complete():
    frame=pd.read_csv(require(ROOT/"results"/"retraining_scan_v1_summary.csv"))
    assert len(frame)==202
    assert frame.groupby("kind").energy.nunique().eq(101).all()
    assert frame.passed.all()
