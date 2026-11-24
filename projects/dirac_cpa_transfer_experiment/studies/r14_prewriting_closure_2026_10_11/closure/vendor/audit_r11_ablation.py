from __future__ import annotations
import ast
import json
import re
from pathlib import Path
from urllib.parse import unquote
import numpy as np
import pandas as pd
import torch
from .r11_config import (ROOT,RAW,DOCS,FIGURES,LOGS,GROUPS,SEEDS,ARTIFACTS,
    CURVE_STEPS,configuration,budget,run_directory,snapshot_directory)
from .freeze_r11_ablation import verify_freeze,verify_prior_results
from .freeze_r7_phase_release import write_json,sha256
from .r11_models import R11Model,hashes,parameter_counts
from .r11_decision import decision
from .aggregate_r11_ablation import original_metrics

def forbidden_keys(value):
    found=[]
    if isinstance(value,dict):
        for key,item in value.items():
            if any(w in str(key).lower() for w in ("frequency","eigenvalue","reference","target","energy")):
                found.append(str(key))
            found.extend(forbidden_keys(item))
    elif isinstance(value,(list,tuple)):
        for item in value:
            found.extend(forbidden_keys(item))
    return found

def check_links(paths):
    broken=[]; checked=0
    for path in paths:
        text=path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\((?:<([^>]+)>|([^)\s]+))\)",text):
            link=unquote(target[0] or target[1])
            if link.startswith(("https://","http://","app://","#")):
                continue
            link=re.sub(r":\d+$","",link)
            destination=Path(link) if Path(link).is_absolute() else path.parent/link
            checked+=1
            if not destination.exists():
                broken.append(dict(document=str(path),target=link))
    return dict(status="PASS" if not broken else "FAIL",checked_links=checked,broken=broken)

def audit(*,final=False):
    cfg=configuration(); checks=[]
    def add(name,condition,**details):
        checks.append(dict(check=name,pass_check=bool(condition),**details))
    frozen=verify_freeze()
    add("scientific_freeze_unchanged",True,sha256=frozen)
    history=verify_prior_results()
    add("r1_through_r10_unchanged",history["status"]=="PASS",**history)
    plan=json.loads((RAW/"runtime_plan.json").read_text(encoding="utf-8"))
    masks=plan["masks"]+[plan["control_mask"]]
    add("pairwise_disjoint_physical_core_allocation",
        len(set(masks))==6 and all(not a&b for i,a in enumerate(masks) for b in masks[i+1:]),
        masks=masks)
    add("all_25_formal_runs_complete",plan["status"]=="COMPLETE" and not plan.get("failures"),
        runtime_status=plan["status"])
    add("five_seed_group_order_counterbalanced",
        all(sorted(s["group_order"])==sorted(GROUPS) for s in plan["slots"]) and
        all(len({s["group_order"][i] for s in plan["slots"]})==5 for i in range(5)))
    observed=json.loads((RAW/"runtime_affinity_observed.json").read_text(encoding="utf-8-sig"))
    add("actual_os_affinity_observation_matches_plan",observed["all_match"] and
        all(r["expected_mask"]==r["observed_mask"] for r in observed["records"]),
        observation_utc=observed["observed_utc"])
    for source in ("r11_models.py","r11_residual.py","train_r11_ablation.py"):
        tree=ast.parse((ROOT/"src"/source).read_text(encoding="utf-8"))
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        add("matrix_reference_excluded_from_training_"+source,
            "r9_reference" not in imports and "r11_reference" not in imports)
    frame=original_metrics()
    recorded=json.loads((RAW/"decision.json").read_text(encoding="utf-8"))
    add("decision_recomputed_from_original_json",recorded==decision(frame))
    csv=pd.read_csv(RAW/"per_seed_metrics.csv",float_precision="round_trip")
    add("decision_csv_roundtrip_consistency",decision(csv)==recorded)
    add("both_fixed_endpoint_cohorts_present",len(frame)==50 and frame.snapshot_identity.nunique()==40)
    costs=pd.read_csv(RAW/"cost_and_residual_work.csv",float_precision="round_trip")
    snapshots=pd.read_csv(RAW/"snapshot_index.csv")
    curves=pd.read_csv(RAW/"error_cost_curves.csv",float_precision="round_trip")
    add("115_fixed_cost_points_and_40_full_evaluations",
        len(snapshots)==115 and snapshots.full_residual.sum()==40 and len(curves)==115)
    slots={s["seed"]:s for s in plan["slots"]}
    total_counter=0
    for seed in SEEDS:
        initial_records={}; traces={}
        for group in GROUPS:
            directory=run_directory(group,seed); pre,coupled=budget(group)
            config=json.loads((directory/"config.json").read_text(encoding="utf-8"))
            metrics=json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
            state=torch.load(directory/"model_state.pt",weights_only=False)
            recovery=torch.load(directory/"recovery_state.pt",weights_only=False)
            initial=torch.load(directory/"initial_state.pt",weights_only=False)["state_dict"]
            initial_records[group]=initial
            prefix=f"{group}/seed{seed}"
            add(prefix+"/final_six_pack_and_recovery",all((directory/p).is_file() for p in ARTIFACTS+("recovery_state.pt",)))
            add(prefix+"/fixed_budget_and_optimizer_count",state["steps"]==metrics["steps"]==recovery["step"]==pre+coupled and
                {int(v["step"]) for v in recovery["optimizer"]["state"].values()}=={coupled})
            add(prefix+"/checkpoint_equals_recovery",all(torch.equal(v,recovery["state_dict"][k])
                for k,v in state["state_dict"].items()))
            add(prefix+"/new_independent_initialization",config["fresh_initialization"] and
                config["seed"] in SEEDS and config["random_streams"]["carrier_init"]==900000+seed and
                config["random_streams"]["coupled_time"]==920000+seed)
            add(prefix+"/science_and_affinity",config["freeze_manifest_sha256"]==frozen and
                int(config["cpu_affinity_mask"])==slots[seed]["mask"] and
                config["environment"]["torch_num_threads"]==1)
            free=recovery["free_sampler"]; coupled_state=recovery["coupled_sampler"]
            add(prefix+"/sampling_dimensions_and_unit_auxiliary_work",tuple(free["trace"].shape)==(pre,32,1) and
                tuple(coupled_state["trace"].shape)==(coupled,32,1) and
                torch.equal(coupled_state["trace"],5*coupled_state["normalized_trace"]))
            traces[group]=coupled_state["trace"]
            expected=(2*pre+coupled)*4128
            add(prefix+"/training_residual_work",metrics["training_spinor_residual_points"]==expected and
                metrics["training_complex_scalar_residual_evaluations"]==2*expected)
            total_counter+=expected
            model=R11Model(cfg,group,seed); model.load_state_dict(state["state_dict"])
            if group==GROUPS[0]: model.carrier.freeze()
            counts=parameter_counts(model)
            add(prefix+"/parameters_and_float64",all(metrics[k]==v for k,v in counts.items()) and
                all(p.dtype==torch.float64 for p in model.parameters()))
            if group in GROUPS[:4]:
                add(prefix+"/no_exact_labels_in_learned_checkpoint_or_optimizer",
                    not forbidden_keys(state) and not forbidden_keys(recovery["optimizer"]) and
                    not state["analytic_prior"],forbidden=forbidden_keys(state))
            else:
                add(prefix+"/exact_diagnostic_explicitly_identified",state["analytic_prior"] and
                    config["analytic_prior"] and counts["carrier_parameters"]==0)
            if group==GROUPS[0]:
                pretrained=torch.load(directory/"pretrained_carrier_state.pt",weights_only=False)
                add(prefix+"/frozen_pretrained_carrier_bitwise",pretrained["steps"]==6000 and all(
                    torch.equal(v,state["state_dict"]["carrier."+k]) for k,v in pretrained["state_dict"].items()))
            elif group==GROUPS[1]:
                add(prefix+"/random_carrier_frozen_bitwise",all(torch.equal(initial[k],v)
                    for k,v in state["state_dict"].items() if k.startswith("carrier.")))
            elif group==GROUPS[2]:
                add(prefix+"/joint_carrier_updated",any(not torch.equal(initial[k],v)
                    for k,v in state["state_dict"].items() if k.startswith("carrier.")))
            scheduled=[s for s in CURVE_STEPS if s<=coupled]
            own=snapshots[(snapshots.model==group)&(snapshots.seed==seed)]
            add(prefix+"/all_preregistered_snapshots",sorted(own.coupled_steps.tolist())==scheduled)
            with np.load(directory/"dense_grid.npz") as dense:
                retained=dense["residual_complex_unweighted"]
                full_residual=dense["full_physical_residual_complex"]
                raw=dense["spectral_prediction_raw_fine"]
                source=dense["exterior_positive_coupling_source_complex"]
                active=dense["phase_reference_active_mask"]
                mask_expected=dense["branch_occupations_reference"]>=1e-6
                add(prefix+"/all_residual_modes_and_correct_exterior_sign",
                    retained.shape==(4001,129,2) and full_residual.shape==(4001,145,2) and
                    np.array_equal(full_residual[:,8:137],retained) and
                    np.array_equal(dense["exterior_residual_complex"],-source) and
                    np.array_equal(source,.1*np.concatenate((raw[:,:8],raw[:,-8:]),axis=1)))
                add(prefix+"/reference_only_active_masks_and_full_support",
                    np.array_equal(active,mask_expected) and
                    np.array_equal(active.reshape(201,-1),dense["coherence_reference_active_channel_mask"]) and
                    np.array_equal(dense["initial_support_mask"],(dense["n"]>=0)&(dense["n"]<=32)))
                add(prefix+"/full_archive_finite",all(np.isfinite(v).all() for v in dense.values()))
                numerical=float(np.trapezoid(np.linalg.norm(retained,axis=(1,2)),dense["residual_t"]))
                add(prefix+"/residual_integral_recomputed",abs(numerical-metrics["residual_integral_fine"])<1e-12 and
                    not metrics["residual_numerical_estimate_is_continuous_certificate"])
            add(prefix+"/hard_initial_and_fourier_rule",metrics["initial_condition_max_abs_error"]<1e-14 and
                metrics["fourier_integral_no_aliasing"] and metrics["fourier_norm_parseval_max_error"]<1e-11 and
                metrics["current_quadrature_double_grid_difference"]<1e-11 and
                metrics["density_quadrature_double_grid_difference"]<1e-11)
            add(prefix+"/evaluation_independent_rng_and_cost_finite",metrics["evaluation_does_not_change_training_rng"] and
                metrics["training_seconds"]>0 and metrics["peak_rss_mib"]>0 and
                (directory/"execution_cost.json").is_file())
            for checkpoint in sorted({20000,coupled}):
                target=snapshot_directory(group,seed,checkpoint)
                cp=torch.load(target/"model_state.pt",weights_only=False)
                restore=torch.load(target/"recovery_state.pt",weights_only=False)
                add(prefix+f"/coupled_{checkpoint}_six_pack_and_exact_restore",
                    all((target/p).is_file() for p in ARTIFACTS+("recovery_state.pt",)) and
                    cp["coupled_steps"]==checkpoint and restore["step"]==pre+checkpoint and
                    all(torch.equal(v,restore["state_dict"][k]) for k,v in cp["state_dict"].items()))
        first=initial_records[GROUPS[0]]
        add(f"seed{seed}/same_shape_envelope_pairing_all_groups",all(all(
            torch.equal(v,record[k]) for k,v in first.items() if k.startswith("networks.0."))
            for record in initial_records.values()))
        add(f"seed{seed}/same_initial_carrier_A_B_C",all(all(torch.equal(v,initial_records[group][k])
            for k,v in first.items() if k.startswith("carrier.")) for group in GROUPS[:3]))
        add(f"seed{seed}/coupled_20000_prefix_pairing_all_groups",
            all(torch.equal(traces[GROUPS[0]],v[:20000]) for v in traces.values()))
        add(f"seed{seed}/full_26000_pairing_B_C_D",all(torch.equal(traces[GROUPS[1]],traces[g])
            for g in GROUPS[1:4]))
    add("training_point_cost_sum_consistent",int(costs.training_spinor_residual_points.sum())==total_counter,
        training_spinor_residual_points=total_counter)
    add("fixed_seed400_dynamics_preregistered",json.loads((RAW/"protocol.json").read_text(encoding="utf-8"))["dynamics_seed"]==400)
    if final:
        test=json.loads((RAW/"pytest_final_result.json").read_text(encoding="utf-8"))
        qa=json.loads((FIGURES/"FIGURE_QA.json").read_text(encoding="utf-8"))
        add("complete_pytest_pass",test["status"]=="PASS",**test)
        add("all_figures_pdf_svg_source_and_manual_qa_pass",qa["status"]=="PASS" and
            not qa["visual_inspection_pending"])
        if not (RAW/"audit.json").exists():
            write_json(RAW/"audit.json",dict(status="RUNNING",audit_stage="final"))
        if not (RAW/"link_check.json").exists():
            write_json(RAW/"link_check.json",dict(status="RUNNING"))
        paths=[ROOT/"R11_INDEPENDENT_REPLICATION_ABLATION_REPORT.md"]+list(DOCS.glob("*.md"))
        links=check_links(paths); write_json(RAW/"link_check.json",links)
        add("report_links_resolve",links["status"]=="PASS",**links)
    result=dict(status="PASS" if all(c["pass_check"] for c in checks) else "FAIL",
        audit_stage="final" if final else "data",checks=checks,check_count=len(checks),
        failed_checks=[c for c in checks if not c["pass_check"]],scientific_decision_separate=True)
    write_json(RAW/("audit.json" if final else "data_audit.json"),result)
    return result

def artifact_manifest():
    files={}
    for base in (RAW,FIGURES,DOCS,LOGS):
        for p in base.rglob("*"):
            if p.is_file() and p.name not in ("artifact_sha256.json","audit.json","data_audit.json"):
                files[p.relative_to(ROOT).as_posix()]=sha256(p)
    sources=[p for p in (ROOT/"src").glob("*r11*.py")]+list((ROOT/"tests").glob("*r11*.py"))
    sources += [ROOT/"R11_INDEPENDENT_REPLICATION_ABLATION_REPORT.md"]
    for p in sources:
        files[p.relative_to(ROOT).as_posix()]=sha256(p)
    write_json(RAW/"artifact_sha256.json",dict(files=files,file_count=len(files)))
    return len(files)

if __name__=="__main__":
    import argparse
    from .r7_runtime import set_cpu_affinity
    from .train_wavepacket_shared_spectral_ltpb import _set_runtime
    set_cpu_affinity(128); _set_runtime(0,1)
    parser=argparse.ArgumentParser();parser.add_argument("--final",action="store_true")
    args=parser.parse_args()
    if args.final:
        artifact_manifest()
    result=audit(final=args.final)
    print(json.dumps(dict(status=result["status"],check_count=result["check_count"],
                         failed_checks=result["failed_checks"]),indent=2))
    if args.final:
        from datetime import datetime,timezone
        write_json(RAW/"postprocessing_status.json",dict(
            stage="complete" if result["status"]=="PASS" else "final_audit",
            status="COMPLETE" if result["status"]=="PASS" else "FAIL",
            utc=datetime.now(timezone.utc).isoformat(),check_count=result["check_count"],
            evidence=str(RAW/"audit.json")))
        print("Artifacts: "+str(artifact_manifest()))
    if result["status"]!="PASS":
        raise SystemExit(1)
