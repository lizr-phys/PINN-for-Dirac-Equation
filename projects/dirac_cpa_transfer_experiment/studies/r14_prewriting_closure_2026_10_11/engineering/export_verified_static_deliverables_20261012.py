"""Export three completed static sources under the frozen delivery names."""
from pathlib import Path
import csv,json,shutil,time
from plan import BASE,sha

def read(name):
    return json.loads((BASE/name).read_text(encoding="utf-8"))

def rows(name):
    with (BASE/name).open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))

def main():
    design=read("frozen/design.json")
    history=read("history/unified_history_status.json")
    trace=read("history/canonical_row_traceability_audit.json")
    calibration=read("references/calibration.json")
    tests=read("engineering/tests_and_recovery.json")
    assert history["status"]=="ARCHIVED_CANONICAL_ROWS" and history["rounds"]==13 and history["rows"]==528
    assert trace["status"]=="PASS_CANONICAL_ROW_LINKS" and trace["matched_rows"]==528 and not trace["unresolved_rows"]
    historical=rows("history/historical_results_all_native_fields.csv")
    assert len(historical)==528 and {int(r["historical_round"]) for r in historical}==set(range(1,14))
    reference=rows("references/reference_convergence.csv")
    assert len(reference)==6 and {r["case"] for r in reference}==set(design["core_cases"]+design["extension_cases"])
    assert all(r["pass"]=="True" for r in reference)
    assert calibration["status"]=="PASS" and calibration["before_learning"] and calibration["grid_validation_complete"]
    assert tests["status"]=="PASS" and all(tests["exact_recovery"]["bitwise_equal"].values())
    pairs=[
        ("history/historical_results_all_native_fields.csv","historical_results.csv","528 native rows, aliases and native definitions retained; heterogeneous physics not pooled"),
        ("references/reference_convergence.csv","reference_convergence.csv","Six frozen main-reference truncation checks; additional wide-reference diagnostics remain separately archived"),
        ("engineering/tests_and_recovery.json","tests_and_recovery.json","Specified implementation/gradient/R13-equivalence and short exact-recovery checks; not proof that234 planned training attempts are finished"),
    ]
    records=[]
    for source,target,scope in pairs:
        assert target in design["deliverables"]
        s=BASE/source;t=BASE/target
        digest=sha(s)
        if t.exists():assert sha(t)==digest,"Existing delivery evidence must not be overwritten"
        records.append(dict(source=source,target=target,sha256=digest,bytes=s.stat().st_size,scope=scope))
    directory=BASE/"delivery_exports/verified_static_2026_10_12"
    if (directory/"receipt.json").exists():
        receipt=read((directory/"receipt.json").relative_to(BASE))
        assert receipt["exports"]==records
        assert all(sha(BASE/r["target"])==r["sha256"] and sha(directory/r["target"])==r["sha256"] for r in records)
        print(json.dumps(dict(status="ALREADY_EXPORTED_SOURCES_UNCHANGED",files=3,whole_goal_complete=False)))
        return
    directory.mkdir(parents=True,exist_ok=True)
    for r in records:
        s=BASE/r["source"];t=BASE/r["target"]
        if not t.exists():shutil.copy2(s,t)
        shutil.copy2(s,directory/r["target"])
        assert sha(s)==sha(t)==sha(directory/r["target"])==r["sha256"]
    proof_names=["history/unified_history_status.json","history/canonical_row_traceability_audit.json","references/calibration.json","frozen/reference_manifest.json","engineering/tests_and_recovery.json"]
    receipt=dict(exported_utc=time.time(),status="THREE_VERIFIED_STATIC_DELIVERY_NAMES_EXPORTED",exports=records,
        source_proof_files=[dict(path=n,sha256=sha(BASE/n)) for n in proof_names],
        export_code_sha256=sha(__file__),design_sha256=sha(BASE/"frozen/design.json"),
        original_sources_and_R1_R13_not_written=True,no_training_or_scientific_change=True,
        final_history_content_rehash_still_required=True,full_training_and_remaining_validations_still_required=True,
        historical_rows_are_not_new_independent_trials=True,mixed_historical_cost_definitions_not_assumed_uniform=True,
        literature_and_live_result_exports_wait_for_final_stable_sources=True,whole_goal_complete=False)
    (directory/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(status=receipt["status"],targets=[r["target"] for r in records],receipt=(directory/"receipt.json").relative_to(BASE).as_posix(),whole_goal_complete=False),ensure_ascii=False))

if __name__=="__main__":main()

