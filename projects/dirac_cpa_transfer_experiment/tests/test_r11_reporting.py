from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
from src.r11_config import ROOT,RAW,FIGURES,GROUPS,SEEDS,ARTIFACTS,budget,run_directory
from src.r11_decision import decision,comparisons
from src.aggregate_r11_ablation import original_metrics
from src.freeze_r11_ablation import verify_freeze,verify_prior_results

def test_population_sd_and_actual_paired_ratios_not_posthoc_thresholds():
    rows=[]
    for endpoint in ("final","coupled_20000"):
        for i,g in enumerate(GROUPS):
            for j,s in enumerate(SEEDS):
                v=.01*(i+1)*(j+1)
                rows.append(dict(endpoint=endpoint,model=g,seed=s,spinor_rel_l2_gauge_aligned=v,
                    current_rel_l2=2*v,fidelity_mean=.99,modal_population_rel_l2=v,max_norm_drift=.001,
                    density_rel_l2=v,active_coherence_rel_l2=v,branch_complex_coefficient_rel_l2=v,
                    residual_integral_fine=v))
    frame=pd.DataFrame(rows)
    pairs,summary=comparisons(frame)
    selected=summary[(summary.endpoint=="final")&(summary.target==GROUPS[0])&
        (summary.baseline==GROUPS[1])&(summary.metric=="spinor_rel_l2_gauge_aligned")].iloc[0]
    assert selected.ratio_of_means == .5 and selected.mean_of_paired_ratios == .5
    np.testing.assert_allclose(selected.paired_difference_population_sd,np.std(-.01*np.arange(1,6),ddof=0))
    assert selected.favorable_pair_count == 5
    assert not decision(frame)["additional_advantage_threshold"]
    assert decision(frame)["analytic_diagnostic_excluded_from_learned_success"]

def test_all_final_runs_and_full_evaluation_restore_artifacts_exist():
    unique=0
    for g in GROUPS:
        for s in SEEDS:
            directory=run_directory(g,s)
            for name in ARTIFACTS+("recovery_state.pt","initial_state.pt","execution_cost.json"):
                assert (directory/name).is_file()
            for stage in sorted({20000,budget(g)[1]}):
                snap=directory/"snapshots"/f"coupled_{stage:05d}"
                for name in ARTIFACTS+("recovery_state.pt",):
                    assert (snap/name).is_file()
                cp=torch.load(snap/"model_state.pt",weights_only=False)
                state=torch.load(snap/"recovery_state.pt",weights_only=False)
                assert cp["steps"]==state["step"]==sum((budget(g)[0],stage))
                assert all(torch.equal(v,state["state_dict"][k]) for k,v in cp["state_dict"].items())
                unique+=1
    assert unique==40

def test_original_json_decision_exactly_matches_roundtrip_csv():
    expected=json.loads((RAW/"decision.json").read_text(encoding="utf-8"))
    assert decision(original_metrics())==expected
    assert decision(pd.read_csv(RAW/"per_seed_metrics.csv",float_precision="round_trip"))==expected
    assert len(expected["absolute_success"]["final"])==5
    assert not expected["pretraining_vs_freezing_single_factor_isolated"]

def test_115_cost_curve_points_and_distinct_workload_costs():
    costs=pd.read_csv(RAW/"cost_and_residual_work.csv")
    snapshots=pd.read_csv(RAW/"snapshot_index.csv")
    curves=pd.read_csv(RAW/"error_cost_curves.csv")
    assert len(costs)==25 and len(curves)==len(snapshots)==115
    assert snapshots.full_residual.sum()==40
    assert int(costs.training_spinor_residual_points.sum())==2683200000
    assert int(costs.evaluation_coupled_spinor_residual_points.sum())==4001*129*40
    assert int(costs.evaluation_carrier_unit_spinor_residual_points.sum())==201*129*2*30
    assert costs.evaluation_forward_only_checkpoint_count.sum()==75
    for g in GROUPS:
        part=costs[costs.model==g]
        pre,coupled=budget(g)
        assert (part.training_spinor_residual_points==(2*pre+coupled)*4128).all()
        assert (part.training_complex_scalar_residual_evaluations==2*(2*pre+coupled)*4128).all()
        assert (part.training_seconds>0).all() and (part.evaluation_seconds>0).all()

def test_full_reference_masks_and_no_predicted_collapse_exclusion():
    for g in GROUPS:
        for s in SEEDS:
            with np.load(run_directory(g,s)/"dense_grid.npz") as dense:
                valid=dense["branch_occupations_reference"]>=1e-6
                assert np.array_equal(valid,dense["phase_reference_active_mask"])
                assert np.array_equal(valid.reshape(201,258),dense["coherence_reference_active_channel_mask"])
                assert dense["full_physical_residual_complex"].shape==(4001,145,2)
                assert np.array_equal(dense["exterior_residual_complex"],
                    -dense["exterior_positive_coupling_source_complex"])

def test_freeze_prior_history_and_data_audit_pass():
    assert len(verify_freeze())==64
    assert verify_prior_results()["status"]=="PASS"
    assert json.loads((RAW/"data_audit.json").read_text(encoding="utf-8"))["status"]=="PASS"

def test_all_figure_bundles_and_fixed_seed_scope():
    qa=json.loads((FIGURES/"FIGURE_QA.json").read_text(encoding="utf-8"))
    assert qa["status"]=="PASS" and not qa["visual_inspection_pending"]
    assert len(qa["figures"])==9
    for item in qa["figures"]:
        assert item["bounding_box_pass"] and item["svg_editable_text"]
        assert item["pdf_valid_and_text_selectable"]
        assert item["seeds"]==([400] if item["figure"]=="dynamics_seed400" else list(SEEDS))
        for suffix in ("png","svg","pdf"):
            assert (FIGURES/(item["figure"]+"."+suffix)).is_file()
        assert all((FIGURES/p).is_file() for p in item["source_csv"])
    text=(ROOT/"R11_INDEPENDENT_REPLICATION_ABLATION_REPORT.md").read_text(encoding="utf-8")
    assert "不能宣称未知连续谱发现" in text and "不能声称单因素冻结" in text
