"""Preservation, provenance, complete cohorts and numerical evidence audit."""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .physics import ROOT,config,save_json

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def expected():
    from .campaign import plan
    return plan()

def main():
    cfg=config();checks=[]
    def check(name,condition,detail=None):
        checks.append(dict(name=name,passed=bool(condition),detail=detail))
    original=json.loads((ROOT/"original_snapshot.json").read_text(encoding="utf-8"))
    changed=[name for name,sha in original.items() if not (ROOT.parent/name).exists() or digest(ROOT.parent/name)!=sha]
    check("original_files_preserved",not changed,dict(count=len(original),changed=changed))
    frozen=json.loads((ROOT/f"frozen_sources_v{cfg['version']}.json").read_text(encoding="utf-8"))
    drift=[name for name,sha in frozen.items() if digest(ROOT/name)!=sha or digest(ROOT/f"frozen_source_v{cfg['version']}"/name)!=sha]
    check("scientific_source_matches_freeze",not drift,drift)
    reference_sources=json.loads((ROOT/"reference_sources.json").read_text(encoding="utf-8"))["source_sha256"]
    reference_paths={"upgrade/reference.py":ROOT/"reference.py","upgrade/physics.py":ROOT/"physics.py",
                     "baseline/reference.py":ROOT.parent/"reference.py"}
    check("reference_source_record_matches",all(digest(reference_paths[n])==sha for n,sha in reference_sources.items()))
    requirements=["config.json","training.json","model.pt","history.csv","training_points.npz",
                  "evaluation.npz","metrics.json","test_grid_convergence.csv"]
    records=[];missing=[];provenance=[];nonfinite=[];teacher_drift=[];grid_bad=[]
    from .campaign import requested
    key=lambda i:(i["purpose"],i["case"],i["kind"],i["state"],i["variant"],i["seed"])
    check("plan_matches_requested_cohorts",sorted(map(key,expected()))==sorted(map(key,requested())))
    for item in expected():
        group,case,kind,state,variant,seed=[item[k] for k in ["group","case","kind","state","variant","seed"]]
        out=ROOT/"results"/group/case/kind/f"state_{state}"/variant/f"seed_{seed}"
        absent=[n for n in requirements if not (out/n).exists()]
        if absent:
            missing.append(dict(run=str(out.relative_to(ROOT)),files=absent));continue
        meta=json.loads((out/"config.json").read_text(encoding="utf-8"))
        m=json.loads((out/"metrics.json").read_text(encoding="utf-8"))
        t=json.loads((out/"training.json").read_text(encoding="utf-8"))
        records.append(dict(group=group,purpose=item["purpose"],**m))
        version=meta["config"]["version"]
        own_frozen=json.loads((ROOT/f"frozen_sources_v{version}.json").read_text(encoding="utf-8"))
        if version!=item["protocol_version"] or meta["source_sha256"]!=own_frozen or any(
            digest(ROOT/f"frozen_source_v{version}"/n)!=sha for n,sha in own_frozen.items()):
            provenance.append(m["run"])
        sd=torch.load(out/"model.pt",weights_only=True)
        if not all(bool(torch.isfinite(v).all()) for v in sd.values()):nonfinite.append(m["run"]+"/model")
        for name in ["training_points.npz","evaluation.npz"]:
            with np.load(out/name) as data:
                if not all(np.isfinite(data[k]).all() for k in data.files):nonfinite.append(m["run"]+"/"+name)
        h=pd.read_csv(out/"history.csv")
        if not np.isfinite(h.select_dtypes(include=["number"]).to_numpy()).all():nonfinite.append(m["run"]+"/history")
        if t["restart_count"]!=len(t["attempts"])-1 or t["seconds"]+1e-5<sum(a["seconds"] for a in t["attempts"]):
            provenance.append(m["run"]+"/attempt_time")
        if "lower_pinn" in meta and digest(ROOT/meta["lower_pinn"]["path"]/"model.pt")!=meta["lower_pinn"]["sha256"]:
            teacher_drift.append(m["run"])
        g=pd.read_csv(out/"test_grid_convergence.csv")
        if list(g.test_points)!=[2001,4001,8001]:grid_bad.append(m["run"])
        if not list((out/"checkpoints").glob("*.pt")):missing.append(dict(run=m["run"],files=["checkpoints"]))
    check("all_63_planned_runs_retained",len(records)==63 and not missing,dict(count=len(records),missing=missing))
    check("all_run_source_and_attempt_provenance",not provenance,provenance)
    check("all_final_data_finite",not nonfinite,nonfinite)
    check("excited_state_teacher_unchanged",not teacher_drift,teacher_drift)
    check("independent_test_grids",not grid_bad,grid_bad)
    df=pd.DataFrame(records)
    formal=df[df.purpose=="formal"] if len(records) else pd.DataFrame()
    check("formal_cohort_50_complete",len(formal)==50)
    failed=formal.loc[~formal.reliable,"run"].tolist() if len(formal) else ["no formal records"]
    mandatory=formal[formal.state==0]
    check("all_lowest_states_pass_joint_accuracy",bool(mandatory.reliable.all()),mandatory.loc[~mandatory.reliable,"run"].tolist())
    check("each_extended_target_has_verified_pinn",bool(formal.groupby(["case","kind","state"]).reliable.any().all()))
    check("failed_exploratory_runs_explicitly_retained",all((ROOT/r/"metrics.json").exists() for r in failed),failed)
    resolutions=[]
    for case,case_cfg in cfg["cases"].items():
        summary=json.loads((ROOT/"results/reference"/case/"summary.json").read_text(encoding="utf-8"))
        check("verified_reference_"+case,summary["verified"])
        for name,s in summary["states"].items():
            kind,index=name.split("_");index=int(index)
            energy=max(2e-12,10*max(s["shooting_energy_difference"],s["fourier_energy_difference"],s["domain_energy_difference"]))
            waveform=max(1e-10,10*max(s["shooting_l2"],s["fourier_l2"]))
            rows=formal[(formal["case"]==case)&(formal.kind==kind)&(formal.state==index)]
            check(f"reference_waveform_resolves_{case}_{name}",len(rows)>0 and waveform<rows.wavefunction_rel_l2.min()/10)
            resolutions.append(dict(case=case,kind=kind,state=index,conservative_energy_scale=energy,
                                    conservative_waveform_scale=waveform,
                                    minimum_pinn_energy_difference=float(rows.energy_abs_error.min()) if len(rows) else None,
                                    energy_resolution_limited_runs=int((rows.energy_abs_error<energy).sum())))
    ortho=pd.read_csv(ROOT/"results/orthogonality.csv")
    q=ortho[(ortho.kind=="kg")&(ortho.state_n==0)&(ortho.state_m==2)]
    ref=q[q.source=="reference"]
    check("kg_nontrivial_weighted_orthogonality",len(ref)==1 and ref.ordinary_l2.abs().iloc[0]>1e-3 and
          ref.normalized_physical_overlap.abs().iloc[0]<1e-9,
          q.to_dict("records"))
    bench=pd.read_csv(ROOT/"results/bvp_benchmark.csv")
    check("serial_bvp_repetitions",len(bench)==30 and (bench.groupby(["kind","tolerance"]).size()==5).all())
    manifest=json.loads((ROOT/"figures/manifest.json").read_text(encoding="utf-8"))
    figure_missing=[]
    for name,entry in manifest["figures"].items():
        for fmt in ["png","svg","pdf"]:
            if not (ROOT/"figures"/f"{name}.{fmt}").exists():figure_missing.append(name+"."+fmt)
        for src in entry["sources"]:
            if not (ROOT/src).exists():figure_missing.append(src)
    check("eight_source_grounded_figure_sets",len(manifest["figures"])==8 and not figure_missing,figure_missing)
    qa_path=ROOT/"figures/visual_qa.json"
    qa=json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.exists() else {}
    check("figure_preview_inspected",bool(qa.get("passed")),qa)
    check("failed_v5_campaign_retained",(ROOT/"results/campaign_v5_failure.json").exists() and
          (ROOT/"results/formal/base/dirac/state_0/derivative/seed_8/history.csv").exists())
    check("failed_v6_zero_wave_retained",(ROOT/"results/campaign_v6_failure.json").exists() and
          (ROOT/"results/stable_formal/deep_wide/dirac/state_2/derivative/seed_0/model.pt").exists())
    completed=subprocess.run([sys.executable,"-m","pytest",str(ROOT.parent/"tests"),str(ROOT/"tests"),"-q"],
                             cwd=ROOT.parent.parent.parent,text=True,capture_output=True)
    output=completed.stdout+completed.stderr
    (ROOT/"results/tests.txt").write_text(output,encoding="utf-8")
    summary_line=next((s for s in reversed(output.splitlines()) if " passed" in s),output[-500:])
    save_json(ROOT/"results/test_run.json",dict(exit_code=completed.returncode,summary=summary_line,
                                               command="python -m pytest bound_state_dirac_kg/tests bound_state_dirac_kg/upgrade/tests -q"))
    check("physics_and_implementation_tests",completed.returncode==0,summary_line)
    result=dict(passed=all(c["passed"] for c in checks),protocol_version=cfg["version"],
                formal_reliable_runs=int(formal.reliable.sum()),formal_total_runs=len(formal),
                failed_exploratory_runs=failed,
                checks=checks,reference_resolution=resolutions,
                limitation="Numerical agreement and declared empirical thresholds are not rigorous error bounds; archived v5 failed models may be nonfinite.")
    save_json(ROOT/"results/audit.json",result)
    # Refresh the report after final test evidence is available.
    from .report import main as report
    report()
    excluded={"results/sha256_manifest.json"}
    hashes={str(p.relative_to(ROOT)).replace("\\","/"):digest(p) for p in ROOT.rglob("*") if p.is_file() and
            not any(c in ["__pycache__",".pytest_cache"] for c in p.parts) and
            str(p.relative_to(ROOT)).replace("\\","/") not in excluded}
    save_json(ROOT/"results/sha256_manifest.json",hashes)
    print(json.dumps(dict(passed=result["passed"],checks=len(checks),failed=[c["name"] for c in checks if not c["passed"]]),ensure_ascii=False),flush=True)
    if not result["passed"]:raise RuntimeError("Delivery audit failed; see results/audit.json")
    return result

if __name__=="__main__":main()
