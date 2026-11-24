"""Archive a necessary-condition bound; never replace the complete-cohort assessment."""
from pathlib import Path
import csv, json, math, re, shutil, time
from plan import BASE, sha

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def main():
    out=BASE/"analysis_outputs"
    target=out/"primary_comparison_receipts/m10_favorable_pair_bound_first_2"
    if target.exists():
        receipt=json.loads((target/"receipt.json").read_text(encoding="utf-8"))
        assert all(sha(target/name)==digest for name,digest in receipt["preserved_file_sha256"].items())
        print(json.dumps({"status":"ALREADY_PRESERVED","path":str(target.relative_to(BASE))}))
        return
    design=json.loads((BASE/"frozen/design.json").read_text(encoding="utf-8"))
    rule=design["statistics"]["learning_advantage_gate"]
    match=re.search(r">=(\d+)/(\d+) lower error",rule)
    required,total=map(int,match.groups())
    assert (required,total)==(4,5) and len(design["formal_seeds"])==total
    metric="spinor_rel_l2_gauge_aligned"
    assert metric in design["thresholds"]
    metric_report=json.loads((out/"independent_metric_audit.json").read_text(encoding="utf-8"))
    assert metric_report["status"]=="PASS_AVAILABLE_SNAPSHOTS" and not metric_report["metric_mismatches"]
    pairing=json.loads((out/"pairing_audit.json").read_text(encoding="utf-8"))
    assert not pairing["mismatches"]
    provenance=json.loads((out/"residual_provenance_audit.json").read_text(encoding="utf-8"))
    assert provenance["status"]=="PASS_CURRENT_RESIDUAL_SOURCE_CHAIN" and not provenance["errors"]
    allrows=read_csv(out/"independent_metrics_per_snapshot.csv")
    pairs=[]; sources=[]
    for seed in design["formal_seeds"]:
        rows={r["group"]:r for r in allrows if r["case"]=="m10" and int(r["seed"])==seed and int(r["step"])==20000 and r["group"] in ("learned_hard","analytic_hard")}
        if len(rows)!=2: continue
        for group,row in rows.items():
            assert row["pass_check"]=="True" and row["raw_and_checkpoint_provenance_pass"]=="True"
            p=BASE/row["path"].replace("\\","/")
            assert sha(p)==row["metric_record_sha256"]
            assert sha(p.parent/"model_state.pt")==row["checkpoint_sha256"]
            assert sha(p.parent/"prediction.npz")==row["prediction_sha256"]
            assert sha(BASE/"references/m10/test201.npz")==row["reference_sha256"]
            assert sha(BASE/"references/m10/test401.npz")==row["reference401_sha256"]
            assert (p.parents[2]/"complete.json").exists()
            initial=next(x for x in pairing["initializer_checks"] if x["case"]=="m10" and x["seed"]==seed and x["group"]==group)
            stream=next(x for x in pairing["stream_replay_checks"] if x["case"]=="m10" and x["seed"]==seed and x["group"]==group)
            assert initial["pass_check"] and stream["pass_check"] and stream["actual_saved_trace_matches_independent_replay"]
            assert sha(BASE/initial["path"])==initial["initial_file_sha256"]
            assert sha(BASE/stream["checkpoint"])==stream["checkpoint_sha256"]==row["checkpoint_sha256"]
            assert sha(BASE/stream["terminal_recovery"])==stream["recovery_sha256"]
            sources.append({"seed":seed,"group":group,"independent_metric_row":row,"actual_initializer":initial,"actual_terminal_sampling_prefix":stream})
        a,c=rows["learned_hard"],rows["analytic_hard"]
        initials=[x["actual_initializer"] for x in sources if x["seed"]==seed]
        streams=[x["actual_terminal_sampling_prefix"] for x in sources if x["seed"]==seed]
        assert len({x["computed_envelope_sha256"] for x in initials})==1
        assert len({x["actual_normalized_prefix_sha256"] for x in streams})==1
        assert len({x["actual_physical_prefix_sha256"] for x in streams})==1
        learned,analytic=float(a[metric]),float(c[metric])
        assert math.isfinite(learned) and math.isfinite(analytic)
        pairs.append({"seed":seed,"learned_primary_error":learned,"analytic_primary_error":analytic,"learned_minus_analytic":learned-analytic,"learning_lower_error":learned<analytic})
    assert len(pairs)==2 and all(p["learned_minus_analytic"]>0 for p in pairs)
    favorable=sum(p["learning_lower_error"] for p in pairs)
    maximum=favorable+total-len(pairs)
    assert maximum==3 and maximum<required
    files=["independent_metric_audit.json","independent_metrics_per_snapshot.csv","pairing_audit.json","residual_audit.json","residual_provenance_audit.json","independent_statistical_audit.json","learning_advantage_assessment.json"]
    hashes={name:sha(out/name) for name in files}
    target.mkdir(parents=True)
    for name in files:
        shutil.copy2(out/name,target/name)
        assert sha(target/name)==hashes[name]==sha(out/name)
    columns=list(pairs[0])
    with (target/"verified_pairs.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(pairs)
    hashes["verified_pairs.csv"]=sha(target/"verified_pairs.csv")
    receipt={"assessed_utc":time.time(),"status":"NECESSARY_CONDITION_UNATTAINABLE_COMPLETE_COHORT_PENDING","case":"m10","primary_step":20000,"primary_metric":metric,"constant_phase_scope":"one phase over the whole spacetime; raw errors separately retained","all_prescribed_seeds":design["formal_seeds"],"all_currently_available_verified_pairs":pairs,"pairs_checked":len(pairs),"minimum_favorable_pairs_required":required,"maximum_favorable_pairs_after_all_remaining_seeds":maximum,"frozen_necessary_condition_can_still_be_met":False,"full_cohort_accuracy_verdict":"INCOMPLETE","full_five_pair_mean_and_CI_computed":False,"does_not_claim_population_significance_or_all_seed_mean_worse":True,"does_not_claim_absolute_dynamics_failure_of_learned_model":True,"untrained_or_unverified_seeds_not_marked_complete":True,"remaining_prescribed_jobs_continue_unchanged":True,"no_early_stopping_or_extra_budget":True,"formal_learning_advantage_assessment_not_modified":True,"frozen_rule":rule,"design_sha256":sha(BASE/"frozen/design.json"),"source_manifest_sha256":sha(BASE/"frozen/source_manifest.json"),"archive_code_sha256":sha(__file__),"per_seed_sources":sources,"preserved_file_sha256":hashes,"whole_goal_complete":False}
    lines=["# m10：冻结优势必要条件的部分证据","", "20000耦合步的主旋量误差移除一个全时空常数相位。当前全部可用且独立核验的两对如下；未对齐误差另存。","", "| 种子 | 学习载波主误差 | 解析载波主误差 | 学习减解析 |","| --- | --- | --- | --- |"]
    lines.extend(f"| {p['seed']} | {p['learned_primary_error']:.12f} | {p['analytic_primary_error']:.12f} | {p['learned_minus_analytic']:+.12f} |" for p in pairs)
    lines+=["","冻结规则要求至少4/5配对误差降低。已验证两对均未改善，因此即使其余三对全部改善，最多也只有3/5；这一必要条件已不可满足。","", "这不是完整五种子的组判定，不计算不完整组的均值或置信区间，不表示学习模型绝对准确度失败，也不推断总体显著性。正式优势评估仍为INCOMPLETE。其余种子、条件及验证继续执行原预算。","", "[来源与逐项SHA](receipt.json) · [实际配对数据](verified_pairs.csv)",""]
    (target/"assessment.md").write_text("\n".join(lines),encoding="utf-8")
    hashes["assessment.md"]=sha(target/"assessment.md")
    (target/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"path":str(target.relative_to(BASE)),"maximum_favorable_pairs":maximum,"required":required,"full_cohort_verdict":"INCOMPLETE","whole_goal_complete":False},ensure_ascii=False))

if __name__=="__main__":main()

