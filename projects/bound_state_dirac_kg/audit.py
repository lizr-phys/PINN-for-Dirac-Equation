"""Deliverable and numerical-consistency audit; does not retrain models."""
from __future__ import annotations
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .physics import ROOT,save_json,config
from .model import gauss_grid


def main():
    result=ROOT/"results"
    ref=json.loads((result/"reference"/"summary.json").read_text())
    df=pd.read_csv(result/"formal_summary.csv")
    checks={}
    checks["six_formal_runs"]=set(zip(df.kind,df.seed))=={(k,s) for k in ["dirac","kg"] for s in [0,1,2]}
    checks["nonzero_normalized"]=bool((df.normalization_error<1e-5).all())
    checks["positive_nodeless_targets"]=bool(((df.energy>.5)&(df.energy<.8)&(df.nodes==0)).all())
    checks["energy_absolute_error_below_1e-5"]=bool((df.energy_abs_error<1e-5).all())
    checks["wavefunction_l2_below_1e-3"]=bool((df.wavefunction_rel_l2<1e-3).all())
    checks["continuous_residual_l2_below_1e-3"]=bool((df.residual_l2<1e-3).all())
    checks["positive_kg_charge"]=bool((df.query("kind == 'kg'").charge_density_min>0).all())
    checks["small_boundary_amplitude"]=bool((df.boundary_amplitude<1e-6).all())
    checks["reference_energy_at_least_1000_times_smaller"]=bool(df.energy_abs_error.min()>1000*2e-12)
    checks["reference_wavefunction_at_least_1000_times_smaller"]=bool(df.wavefunction_rel_l2.min()>1000*1e-9)
    for k in ["dirac","kg"]:
        checks[k+"_independent_reference"]=bool(ref[k]["fourier_energy_difference"]<2e-12 and ref[k]["shooting_energy_difference"]<2e-12 and ref[k]["fourier_l2"]<1e-9)
    checks["sensitivity_complete"]=all(len(pd.read_csv(result/f"{t}_summary.csv"))==2 for t in ["grid_refined","domain_extended"])
    missing=[];grid_stability=[];regenerated=[]
    if (result/"audit.json").exists():
        regenerated=json.loads((result/"audit.json").read_text()).get("deterministic_quadrature_reconstructed_for_earlier_runs",[])
    torch.set_default_dtype(torch.float64)
    for tag in ["formal","grid_refined","domain_extended"]:
        for path in sorted((result/tag).glob("*/seed_*")):
            for f in ["model.pt","config.json","history.csv","training.json","metrics.json","evaluation.npz","evaluation.csv","test_grid_convergence.csv"]:
                if not (path/f).is_file():missing.append(str((path/f).relative_to(ROOT)))
            cfg=json.loads((path/"config.json").read_text())["config"]
            if not (path/"training_points.npz").exists():
                x,w=gauss_grid(cfg["collocation_points"],cfg["half_length"])
                q,qw=gauss_grid(cfg["quadrature_points"],cfg["half_length"])
                np.savez_compressed(path/"training_points.npz",collocation_x=x.numpy(),collocation_weights=w.numpy(),normalization_x=q.numpy(),normalization_weights=qw.numpy())
                regenerated.append(str(path.relative_to(ROOT)))
            grids=pd.read_csv(path/"test_grid_convergence.csv")
            grid_stability.append(float(abs(grids.residual_l2.iloc[-1]-grids.residual_l2.iloc[-2])))
            hist=pd.read_csv(path/"history.csv")
            checks[str(path.relative_to(result))+"_finite_history"]=bool(np.isfinite(hist.select_dtypes("number")).all().all())
    checks["artifacts_complete"]=not missing
    checks["test_grid_refinement_stable"]=max(grid_stability)<1e-8
    checks["training_has_no_reference_import"]=not any(re.search(r"from\s+\.reference|from\s+\.evaluate|np\.load",(ROOT/f).read_text()) for f in ["model.py","train.py"])
    checks["six_vector_and_raster_figure_groups"]=all((result/"figures"/f"{name}.{ext}").is_file() for name in ["01_potential_energy","02_dirac_components","03_kg_wavefunction_charge","04_residuals_tails","05_training","06_reference_convergence"] for ext in ["png","pdf","svg"])
    checks["svg_editable_text"]=all("<text" in p.read_text(encoding="utf-8") for p in (result/"figures").glob("*.svg"))
    checks["report_present"]=(ROOT/"REPORT.md").is_file()
    # These are delivery checks selected after the pilot, not preregistered research endpoints.
    audit=dict(passed=all(checks.values()),checks=checks,missing_artifacts=missing,
        threshold_status="Delivery checks chosen after pilot; not a preregistered hypothesis test.",
        reference_empirical_energy_scale=2e-12,reference_empirical_wavefunction_scale=1e-9,
        reference_wavefunction_error_domain="[-20,20]; tail_norm_estimate denotes the exterior normalization integral, not its square root",
        max_test_residual_change_4001_to_8001=max(grid_stability),
        deterministic_quadrature_reconstructed_for_earlier_runs=regenerated,
        scope="New stationary experiment only; existing time-evolution experiments were not imported or written.")
    test_path=result/"tests.txt"
    if not test_path.exists():
        test=subprocess.run([sys.executable,"-m","pytest",str(ROOT/"tests"),"-q"],
                            cwd=ROOT,capture_output=True,text=True)
        test_path.write_text(test.stdout+test.stderr,encoding="utf-8")
    checks["test_log_present"]=test_path.exists()
    if test_path.exists():
        audit["test_summary"]=test_path.read_text(encoding="utf-8").strip()
        checks["physics_tests_passed"]="11 passed" in audit["test_summary"] and "failed" not in audit["test_summary"]
    audit["passed"]=all(checks.values())
    save_json(result/"audit.json",audit)
    hashes={}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name=="artifact_sha256.json" or any(p in {"__pycache__",".pytest_cache"} for p in path.parts):continue
        hashes[str(path.relative_to(ROOT)).replace("\\","/")]=hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(result/"artifact_sha256.json",hashes)
    print(json.dumps(dict(passed=audit["passed"],failed=[k for k,v in checks.items() if not v],files_hashed=len(hashes)),indent=2))
    if not audit["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
