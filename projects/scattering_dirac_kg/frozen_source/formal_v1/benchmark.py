from __future__ import annotations
import numpy as np
import pandas as pd
from .physics import ROOT, KINDS, read_json, save_json


def traditional_practical_scan():
    """An actual looser-tolerance scan, evaluated against the fine reference."""
    import time
    from .reference import jost
    from .physics import relative_l2
    rows = []
    fine_all = pd.read_csv(ROOT/"results"/"reference"/"scan.csv")
    for kind in KINDS:
        reference = np.load(ROOT/"results"/"reference"/f"{kind}_scan.npz")
        fine = fine_all[fine_all.kind==kind].reset_index(drop=True)
        for i, e in enumerate(reference["energies"]):
            sol = jost(kind, float(e), rtol=1e-8, atol=1e-9, max_step=.6)
            start = time.perf_counter()
            state = sol.value(reference["x"])
            inference = time.perf_counter()-start
            wave = state if kind=="dirac" else state[:,:1]
            refwave = reference["state"][i] if kind=="dirac" else reference["state"][i,:,:1]
            rows.append(dict(**sol.summary(), inference_seconds=inference,
                wave_relative_l2=relative_l2(reference["x"], wave, refwave),
                T_absolute_error=abs(sol.T-fine["T"].iloc[i]), R_absolute_error=abs(sol.R-fine["R"].iloc[i]),
                t_complex_error=abs(sol.t-complex(fine.t_real.iloc[i],fine.t_imag.iloc[i]))))
    pd.DataFrame(rows).to_csv(ROOT/"results"/"traditional_practical_scan.csv", index=False)


def summarize_scan_costs():
    reference = pd.read_csv(ROOT/"results"/"reference"/"scan.csv")
    retrain = pd.read_csv(ROOT/"results"/"retraining_scan_v1_summary.csv")
    conditional = pd.read_csv(ROOT/"results"/"conditional_v1_summary.csv")
    rows = []
    practical_path = ROOT/"results"/"traditional_practical_scan.csv"
    if not practical_path.exists():
        traditional_practical_scan()
    practical = pd.read_csv(practical_path)
    for kind in KINDS:
        r = reference[reference.kind==kind]
        p = retrain[retrain.kind==kind]
        c = conditional[conditional.kind==kind]
        rows.append(dict(kind=kind, method="Jost", seeds=1, scan_energies=len(r),
            training_seconds=float(r.seconds.sum()), inference_seconds=float(r.inference_seconds.sum()),
            total_seconds=float((r.seconds+r.inference_seconds).sum()),
            mean_wave_relative_l2=None, maximum_wave_relative_l2=None,
            maximum_T_error=None, interpretation="Independent traditional solutions at all energies, high-accuracy DOP853."))
        a = practical[practical.kind==kind]
        rows.append(dict(kind=kind, method="Jost rtol 1e-8", seeds=1, scan_energies=len(a),
            training_seconds=float(a.seconds.sum()), inference_seconds=float(a.inference_seconds.sum()),
            total_seconds=float((a.seconds+a.inference_seconds).sum()),
            mean_wave_relative_l2=float(a.wave_relative_l2.mean()), maximum_wave_relative_l2=float(a.wave_relative_l2.max()),
            maximum_T_error=float(a.T_absolute_error.max()), interpretation="Actual looser-tolerance traditional solve; error evaluated against fine reference."))
        rows.append(dict(kind=kind, method="Retrain RBF", seeds=1, scan_energies=len(p),
            training_seconds=float(p.training_seconds.sum()), inference_seconds=float(p.inference_seconds.sum()),
            total_seconds=float((p.training_seconds+p.inference_seconds).sum()),
            mean_wave_relative_l2=float(p.wave_relative_l2.mean()), maximum_wave_relative_l2=float(p.wave_relative_l2.max()),
            maximum_T_error=float(p.T_absolute_error.max()), interpretation="Actual independent training at each scan energy; seed 0."))
        per_seed = []
        for seed in (0,1,2):
            s = c[c.seed==seed]
            training = read_json(ROOT/"results"/"conditional_v1"/kind/"conditional"/f"seed_{seed}"/"training.json")
            per_seed.append(dict(kind=kind, seed=seed, training_seconds=training["seconds"],
                                inference_seconds=float(s.inference_seconds.sum()),
                                total_seconds=float(training["seconds"]+s.inference_seconds.sum())))
        pd.DataFrame(per_seed).to_csv(ROOT/"results"/f"{kind}_conditional_scan_costs.csv", index=False)
        rows.append(dict(kind=kind, method="Conditional RBF", seeds=3, scan_energies=len(c)//3,
            training_seconds=float(np.mean([s["training_seconds"] for s in per_seed])),
            inference_seconds=float(np.mean([s["inference_seconds"] for s in per_seed])),
            total_seconds=float(np.mean([s["total_seconds"] for s in per_seed])),
            mean_wave_relative_l2=float(c.wave_relative_l2.mean()), maximum_wave_relative_l2=float(c.wave_relative_l2.max()),
            maximum_T_error=float(c.T_absolute_error.max()), interpretation="Mean of three separately trained conditional networks; one training plus all-energy inference each."))
    pd.DataFrame(rows).to_csv(ROOT/"results"/"scan_costs.csv", index=False)
    dense = pd.read_csv(ROOT/"results"/"reference"/"dense_scan.csv")
    extrema = read_json(ROOT/"results"/"reference"/"transmission_extrema.json")
    save_json(ROOT/"results"/"cost_scope.json", dict(
        reference_dense_extrema_grid_solve_seconds=float(dense.seconds.sum()),
        reference_extrema_refinement_seconds=float(sum(r["seconds"] for r in extrema)),
        reference_scan_grid_solve_seconds=float(reference.seconds.sum()),
        fixed_training_seconds=float(pd.read_csv(ROOT/"results"/"formal_v1_summary.csv").training_seconds.sum()),
        mlp_baseline_training_seconds=float(pd.read_csv(ROOT/"results"/"mlp_baseline_summary.csv").training_seconds.sum()),
        scan_points=101, dense_extrema_points_per_model=1001,
        background_load="Pre-existing R12 training controller, five workers and finalizer were active before and during this scattering task. See results/preexisting_active_processes.json. Costs are actual wall times under background load, not isolated CPU benchmarks.",
        statement="Timings exclude library import, file writes, development and evaluation against reference; dense-extremum verification is separate from a 101-energy solve cost. MLP baseline overlapped some development reference computation; all speed comparisons need idle-machine repetition."))


if __name__ == "__main__":
    summarize_scan_costs()
