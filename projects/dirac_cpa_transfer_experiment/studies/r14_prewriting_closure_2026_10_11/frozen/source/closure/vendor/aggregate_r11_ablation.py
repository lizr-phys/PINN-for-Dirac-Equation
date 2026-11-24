from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .r11_config import RAW, ROOT, GROUPS, SEEDS, ARTIFACTS, budget, run_directory, snapshot_directory
from .r11_decision import decision, comparisons
from .freeze_r7_phase_release import write_json

def original_metrics():
    records = []
    for endpoint in ("final","coupled_20000"):
        for group in GROUPS:
            for seed in SEEDS:
                directory = run_directory(group,seed) if endpoint == "final" else snapshot_directory(group,seed,20000)
                if not all((directory/name).is_file() for name in ARTIFACTS+("recovery_state.pt",)):
                    raise RuntimeError("Missing fixed endpoint artifacts: "+str(directory))
                row = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
                records.append(row | dict(endpoint=endpoint,
                    snapshot_identity=f"{group}/seed_{seed}/coupled_{row['coupled_steps']:05d}"))
    return pd.DataFrame(records)

def aggregate():
    frame = original_metrics()
    frame.to_csv(RAW/"per_seed_metrics.csv",index=False)
    summaries = []
    keys = [k for k in frame.select_dtypes(include=[np.number]).columns if k != "seed"]
    for (endpoint,group), part in frame.groupby(["endpoint","model"],sort=False):
        for key in keys:
            values = part[key].dropna()
            summaries.append(dict(endpoint=endpoint,model=group,metric=key,n=5,n_valid=len(values),
                applicable=bool(len(values)),mean=float(values.mean()) if len(values) else None,
                population_sd=float(values.std(ddof=0)) if len(values) else None,
                minimum=float(values.min()) if len(values) else None, maximum=float(values.max()) if len(values) else None))
    pd.DataFrame(summaries).to_csv(RAW/"summary_metrics.csv",index=False)
    pairs, comparison = comparisons(frame)
    pairs.to_csv(RAW/"paired_comparisons.csv",index=False)
    comparison.to_csv(RAW/"comparison_summary.csv",index=False)
    budget_rows,budget_summary=[],[]
    for group in GROUPS[1:4]:
        a=frame[(frame.model==group)&(frame.endpoint=="coupled_20000")].set_index("seed")
        b=frame[(frame.model==group)&(frame.endpoint=="final")].set_index("seed")
        for key in tuple(decision(frame)["thresholds"])+("density_rel_l2","active_coherence_rel_l2",
                "branch_complex_coefficient_rel_l2","residual_integral_fine"):
            difference,ratio=b[key]-a[key],b[key]/a[key]
            for seed in SEEDS:
                budget_rows.append(dict(model=group,seed=seed,metric=key,
                    before_coupled_steps=20000,after_coupled_steps=26000,
                    before=float(a.loc[seed,key]),after=float(b.loc[seed,key]),
                    paired_difference=float(difference[seed]),actual_ratio=float(ratio[seed])))
            budget_summary.append(dict(model=group,metric=key,n=5,
                paired_difference_mean=float(difference.mean()),
                paired_difference_population_sd=float(difference.std(ddof=0)),
                ratio_of_means=float(b[key].mean()/a[key].mean()),
                mean_of_paired_ratios=float(ratio.mean()),
                favorable_pair_count=int((difference>0 if key=="fidelity_mean" else difference<0).sum())))
    pd.DataFrame(budget_rows).to_csv(RAW/"budget_changes.csv",index=False)
    pd.DataFrame(budget_summary).to_csv(RAW/"budget_change_summary.csv",index=False)
    write_json(RAW/"decision.json",decision(frame))
    costs, curves, artifacts, snapshot_index = [], [], [], []
    for group in GROUPS:
        for seed in SEEDS:
            directory = run_directory(group,seed)
            execution = json.loads((directory/"execution_cost.json").read_text(encoding="utf-8"))
            final = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
            curve = pd.read_csv(directory/"cost_curve.csv",float_precision="round_trip")
            curve["model"], curve["seed"] = group, seed
            curves.append(curve)
            evaluation_points = evaluation_carrier_points = evaluation_batches = 0
            for snapshot in sorted((directory/"snapshots").iterdir()):
                metrics = json.loads((snapshot/"metrics.json").read_text(encoding="utf-8"))
                evaluation_points += metrics.get("evaluation_coupled_spinor_residual_points",0)
                evaluation_carrier_points += metrics.get("evaluation_carrier_unit_spinor_residual_points",0)
                evaluation_batches += metrics.get("evaluation_coupled_residual_batches",0)+metrics.get("evaluation_carrier_residual_batches",0)
                snapshot_index.append(dict(model=group,seed=seed,coupled_steps=metrics["coupled_steps"],
                    total_steps=metrics["steps"],full_residual=metrics["evaluation_full_residual"],
                    directory=snapshot.relative_to(ROOT).as_posix()))
                if metrics["evaluation_full_residual"]:
                    for name in ARTIFACTS+("recovery_state.pt",):
                        artifacts.append(dict(model=group,seed=seed,role="full_snapshot",
                            coupled_steps=metrics["coupled_steps"],artifact=name,path=str(snapshot/name)))
            for name in ARTIFACTS+("recovery_state.pt","initial_state.pt","execution_cost.json",
                                  "execution_events.jsonl","cost_curve.csv"):
                artifacts.append(dict(model=group,seed=seed,role="final_run",
                    coupled_steps=final["coupled_steps"],artifact=name,path=str(directory/name)))
            pre,coupled = budget(group)
            costs.append(dict(model=group,seed=seed,analytic_prior=group == GROUPS[4],**execution,
                total_optimizer_steps=pre+coupled,carrier_steps=pre,coupled_steps=coupled,
                training_residual_batches=pre+coupled,
                training_carrier_unit_spinor_residual_points=2*pre*4128,
                training_coupled_spinor_residual_points=coupled*4128,
                training_spinor_residual_points=(2*pre+coupled)*4128,
                training_complex_scalar_residual_evaluations=2*(2*pre+coupled)*4128,
                evaluation_coupled_spinor_residual_points=evaluation_points,
                evaluation_carrier_unit_spinor_residual_points=evaluation_carrier_points,
                evaluation_residual_batches=evaluation_batches,
                evaluation_forward_only_checkpoint_count=len(curve)-sum(
                    s["full_residual"] for s in snapshot_index if s["model"]==group and s["seed"]==seed),
                cpu_affinity_mask=final["cpu_affinity_mask"]))
    pd.DataFrame(costs).to_csv(RAW/"cost_and_residual_work.csv",index=False)
    pd.concat(curves,ignore_index=True).to_csv(RAW/"error_cost_curves.csv",index=False)
    pd.DataFrame(artifacts).to_csv(RAW/"run_artifact_index.csv",index=False)
    pd.DataFrame(snapshot_index).to_csv(RAW/"snapshot_index.csv",index=False)
    return frame,decision(frame)

if __name__ == "__main__":
    print(json.dumps(aggregate()[1],indent=2))
