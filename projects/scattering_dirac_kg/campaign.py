from __future__ import annotations
import time
import numpy as np
import pandas as pd
from .physics import ROOT, CASE, KINDS, ENERGIES, read_json, save_json
from .train import train
from .evaluate import evaluate_run, aggregate


def fixed_campaign(group="formal_v1"):
    cfg = read_json(ROOT/"config.json")
    verification = read_json(ROOT/"results"/"reference"/"verification.json")
    if not verification["verified"] or verification["case"]!=cfg["case"]:
        raise RuntimeError("Reference gate not verified for current physics.")
    records = []
    for kind in KINDS:
        for energy in ENERGIES:
            for seed in cfg["seeds"]:
                out = train(kind, energy, seed, group, **cfg["fixed"])
                m = evaluate_run(out)
                records.append(m)
                print(f"FIXED {kind} E={energy:g} seed={seed} wave={m['wave_relative_l2']:.2e} residual={m['residual_rms']:.2e} passed={m['passed']}", flush=True)
    df = aggregate(records, ROOT/"results"/f"{group}_summary.csv")
    # Quantitative reference separation is checked for wave and both amplitudes
    # per model/energy; tiny R has squared errors and is not used as a ratio test.
    convergence = pd.read_csv(ROOT/"results"/"reference"/"convergence.csv")
    fine = convergence[(convergence.variant.isin(["bvp", "domain_20", "tolerance_3e-13"])) & (convergence.grid_points==4001)]
    hp_path = ROOT/"results"/"reference"/"high_precision"/"verification.json"
    hp_ok = hp_path.exists() and read_json(hp_path)["verified"]
    hp_error = read_json(hp_path)["maximum_amplitude_difference"] if hp_ok else None
    separation = []
    for kind in KINDS:
        for e in ENERGIES:
            p = df[(df.kind==kind)&(df.energy==e)]
            r = fine[(fine.kind==kind)&(fine.energy==e)]
            values = dict(kind=kind, energy=e)
            for metric, ref_metric in [("wave_relative_l2", "wave_l2_difference"),
                                       ("r_complex_error", "r_difference"), ("t_complex_error", "t_difference")]:
                reference_difference = max(float(r[ref_metric].max()), 1e-15)
                if hp_ok and metric in ("r_complex_error", "t_complex_error"):
                    reference_difference = max(hp_error, 1e-15)  # conservative complex128 storage floor
                values[metric+"_reference_difference"] = reference_difference
                values[metric+"_minimum_separation"] = float(p[metric].min()/reference_difference)
            separation.append(values)
    pd.DataFrame(separation).to_csv(ROOT/"results"/"reference_separation.csv", index=False)
    separated = all(s[k]>=100 for s in separation for k in s if k.endswith("minimum_separation"))
    gate = dict(passed=bool(df.passed.all() and separated), accepted=int(df.passed.sum()),
               attempted=len(df), reference_separated=separated, group=group,
               high_precision_amplitudes=hp_ok,
               thresholds=cfg["validation"], statement="All seeds retained; conditional formal training requires this gate.")
    save_json(ROOT/"results"/"fixed_validation_gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError("Fixed validation gate failed; conditional formal stage blocked.")


def baseline_campaign(group="mlp_baseline"):
    from .mlp import train_mlp, evaluate_mlp
    cfg = read_json(ROOT/"config.json")
    records = []
    for kind in KINDS:
        for energy in ENERGIES:
            for seed in cfg["seeds"]:
                p = train_mlp(kind, energy, seed, group, adam_steps=cfg["baseline"]["adam_steps"],
                    lbfgs_steps=cfg["baseline"]["lbfgs_steps"], width=cfg["baseline"]["width"],
                    points=cfg["baseline"]["collocation"], boundary_weight=cfg["baseline"]["boundary_weight"])
                metrics = evaluate_mlp(p)
                records.append(metrics)
    aggregate(records, ROOT/"results"/f"{group}_summary.csv")


def conditional_campaign(group="conditional_v1"):
    cfg = read_json(ROOT/"config.json")
    gate = read_json(ROOT/"results"/"fixed_validation_gate.json")
    if not gate["passed"]:
        raise RuntimeError("Fixed energy validation must pass first.")
    records = []
    energies = np.linspace(1.01, 2., cfg["scan"]["points"])
    from .train import training_energies
    from .reference import jost
    train_e = training_energies(cfg["conditional"]["energy_nodes"])
    # Reference labels are loaded after training only. They never enter train().
    for kind in KINDS:
        nets = [train(kind, seed=seed, group=group, **cfg["conditional"]) for seed in cfg["seeds"]]
        for e in energies:
            ref = jost(kind, float(e), length=20.)
            is_training = bool(np.min(abs(train_e-e))<1e-12)
            for p in nets:
                metrics = evaluate_run(p, float(e), ref, points=1601, save=True)
                metrics["training_energy"] = is_training
                metrics["split"] = "training_grid" if is_training else "held_out_energy"
                records.append(metrics)
        print(f"CONDITIONAL {kind}: {len(energies)} energies x {len(nets)} seeds evaluated", flush=True)
    aggregate(records, ROOT/"results"/f"{group}_summary.csv")
    heldout = pd.DataFrame(records).query("training_energy == False")
    save_json(ROOT/"results"/"conditional_validation.json", dict(total=len(records),
         heldout_attempts=len(heldout), heldout_accepted=int(heldout.passed.sum()),
         training_energy_nodes=train_e.tolist(), maximum_heldout_wave_error=float(heldout.wave_relative_l2.max()),
         maximum_heldout_RT_error=float(max(heldout.R_absolute_error.max(), heldout.T_absolute_error.max())),
         statement="Joint physics-only conditional network; no inference-time equation solve or parameter update."))


def retraining_scan(group="retraining_scan_v1"):
    cfg = read_json(ROOT/"config.json")
    energies = np.linspace(1.01, 2., cfg["scan"]["points"])
    from .reference import jost
    records = []
    for kind in KINDS:
        for e in energies:
            p = train(kind, float(e), 0, group, **cfg["fixed"])
            metrics = evaluate_run(p, reference=jost(kind, float(e), length=20.), points=1601)
            records.append(metrics)
        print(f"RETRAIN SCAN {kind}: {len(energies)} independent trainings", flush=True)
    pd.DataFrame(records).to_csv(ROOT/"results"/f"{group}_summary.csv", index=False)


def sensitivity_campaign():
    cfg = read_json(ROOT/"config.json")
    records = []
    for kind in KINDS:
        for energy in (1.02, 2.):
            for tag, modifications in [("grid_refined", dict(collocation=768)),
                                       ("domain_extended", dict(case={**CASE, "half_length":20.}))]:
                args = {**cfg["fixed"], **modifications}
                p = train(kind, energy, 0, tag+"_v1", **args)
                records.append(evaluate_run(p))
    pd.DataFrame(records).to_csv(ROOT/"results"/"sensitivity_summary.csv", index=False)


if __name__ == "__main__":
    fixed_campaign()
